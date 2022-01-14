import argparse
import random
from pathlib import Path
from typing import Optional, Union, Tuple, List, Any, Callable

import torch
import torch.utils.data
import torchaudio
import tqdm
from xumx_slicq import transforms
from xumx_slicq import filtering
from xumx_slicq import model


def load_info(path: str) -> dict:
    """Load audio metadata

    this is a backend_independent wrapper around torchaudio.info

    Args:
        path: Path of filename
    Returns:
        Dict: Metadata with
        `samplerate`, `samples` and `duration` in seconds

    """
    # get length of file in samples
    if torchaudio.get_audio_backend() == "sox":
        raise RuntimeError("Deprecated backend is not supported")

    info = {}
    si = torchaudio.info(str(path))
    info["samplerate"] = si.sample_rate
    info["samples"] = si.num_frames
    info["channels"] = si.num_channels
    info["duration"] = info["samples"] / info["samplerate"]
    return info


def load_audio(
    path: str,
    start: float = 0.0,
    dur: Optional[float] = None,
    info: Optional[dict] = None,
):
    """Load audio file

    Args:
        path: Path of audio file
        start: start position in seconds, defaults on the beginning.
        dur: end position in seconds, defaults to `None` (full file).
        info: metadata object as called from `load_info`.

    Returns:
        Tensor: torch tensor waveform of shape `(num_channels, num_samples)`
    """
    # loads the full track duration
    if dur is None:
        # we ignore the case where start!=0 and dur=None
        # since we have to deal with fixed length audio
        sig, rate = torchaudio.load(path)
        return sig, rate
    else:
        if info is None:
            info = load_info(path)
        num_frames = int(dur * info["samplerate"])
        frame_offset = int(start * info["samplerate"])
        sig, rate = torchaudio.load(path, num_frames=num_frames, frame_offset=frame_offset)
        return sig, rate


def aug_from_str(list_of_function_names: list):
    if list_of_function_names:
        return Compose([globals()["_augment_" + aug] for aug in list_of_function_names])
    else:
        return lambda audio: audio


class Compose(object):
    """Composes several augmentation transforms.
    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            audio = t(audio)
        return audio


def _augment_gain(audio: torch.Tensor, low: float = 0.25, high: float = 1.25) -> torch.Tensor:
    """Applies a random gain between `low` and `high`"""
    g = low + torch.rand(1) * (high - low)
    return audio * g


def _augment_channelswap(audio: torch.Tensor) -> torch.Tensor:
    """Swap channels of stereo signals with a probability of p=0.5"""
    if audio.shape[0] == 2 and torch.tensor(1.0).uniform_() < 0.5:
        return torch.flip(audio, [0])
    else:
        return audio


def _augment_force_stereo(audio: torch.Tensor) -> torch.Tensor:
    # for multichannel > 2, we drop the other channels
    if audio.shape[0] > 2:
        audio = audio[:2, ...]

    if audio.shape[0] == 1:
        # if we have mono, we duplicate it to get stereo
        audio = torch.repeat_interleave(audio, 2, dim=0)

    return audio


class UnmixDataset(torch.utils.data.Dataset):
    _repr_indent = 4

    def __init__(
        self,
        root: Union[Path, str],
        sample_rate: float,
        seq_duration: Optional[float] = None,
        source_augmentations: Optional[Callable] = None,
    ) -> None:
        self.root = Path(args.root).expanduser()
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.source_augmentations = source_augmentations

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def extra_repr(self) -> str:
        return ""


def load_datasets(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> Tuple[UnmixDataset, UnmixDataset, argparse.Namespace]:
    """Loads the specified dataset from commandline arguments

    Returns:
        train_dataset, validation_dataset
    """
    parser.add_argument(
        "--random-track-mix",
        action="store_true",
        default=False,
        help="use random track mix",
    )
    parser.add_argument(
        "--fixed-start",
        type=float,
        default=-1.0,
        help="supply fixed start (in s) of song (<0.0 = random start)",
    )
    parser.add_argument("--samples-per-track", type=int, default=64)
    parser.add_argument("--source-augmentations", type=str, default=["gain", "channelswap"], nargs="+")

    args = parser.parse_args()
    dataset_kwargs = {
        "root": args.root,
        "subsets": "train",
        "download": args.root is None,
        "seed": args.seed,
        "fixed_start": args.fixed_start,
    }

    source_augmentations = aug_from_str(args.source_augmentations)

    train_dataset = MUSDBDataset(
        split="train",
        samples_per_track=args.samples_per_track,
        seq_duration=args.seq_dur,
        source_augmentations=source_augmentations,
        random_track_mix=True,
        **dataset_kwargs,
    )

    valid_dataset = MUSDBDataset(
        split="valid", samples_per_track=1, seq_duration=None, **dataset_kwargs, 
    )

    return train_dataset, valid_dataset, args


class MUSDBDataset(UnmixDataset):
    def __init__(
        self,
        root: str = None,
        download: bool = False,
        subsets: str = "train",
        split: str = "train",
        seq_duration: Optional[float] = 6.0,
        samples_per_track: int = 64,
        source_augmentations: Optional[Callable] = lambda audio: audio,
        random_track_mix: bool = False,
        fixed_start: int = -1,
        seed: int = 42,
        *args,
        **kwargs,
    ) -> None:
        """MUSDB18 torch.data.Dataset that samples from the MUSDB tracks
        using track and excerpts with replacement.

        Parameters
        ----------
        root : str
            root path of MUSDB
        download : boolean
            automatically download 7s preview version of MUSDB
        subsets : list-like [str]
            subset str or list of subset. Defaults to ``train``.
        split : str
            use (stratified) track splits for validation split (``valid``),
            defaults to ``train``.
        seq_duration : float
            training is performed in chunks of ``seq_duration`` (in seconds,
            defaults to ``None`` which loads the full audio track
        samples_per_track : int
            sets the number of samples, yielded from each track per epoch.
            Defaults to 64
        source_augmentations : list[callables]
            provide list of augmentation function that take a multi-channel
            audio file of shape (src, samples) as input and output. Defaults to
            no-augmentations (input = output)
        random_track_mix : boolean
            randomly mixes sources from different tracks to assemble a
            custom mix. This augmenation is only applied for the train subset.
        seed : int
            control randomness of dataset iterations
        args, kwargs : additional keyword arguments
            used to add further control for the musdb dataset
            initialization function.

        """
        import musdb

        self.seed = seed
        random.seed(seed)
        self.seq_duration = seq_duration
        self.subsets = subsets
        self.split = split
        self.samples_per_track = samples_per_track
        self.source_augmentations = source_augmentations
        self.random_track_mix = random_track_mix
        self.fixed_start = fixed_start
        self.mus = musdb.DB(
            root=root,
            is_wav=True,
            split=split,
            subsets=subsets,
            download=download,
            *args,
            **kwargs,
        )
        self.sample_rate = 44100.0  # musdb is fixed sample rate

    def __getitem__(self, index):
        audio_sources = []

        # select track
        track = self.mus.tracks[index // self.samples_per_track]

        # at training time we assemble a custom mix
        if self.seq_duration:
            for k, source in enumerate(self.mus.setup["sources"]):
                # select a random track
                if self.random_track_mix:
                    track = random.choice(self.mus.tracks)

                # set the excerpt duration

                # don't try to make a bigger duration than exists
                # but we still want to limit so that we're not trying
                # to take the NSGT of an entire 5 minute track
                dur = min(track.duration, self.seq_duration)

                track.chunk_duration = dur

                if self.fixed_start < 0:
                    # set random start position
                    track.chunk_start = random.uniform(0, track.duration - dur)
                else:
                    # start at fixed position for debugging purposes
                    track.chunk_start = self.fixed_start
                # load source audio and apply time domain source_augmentations

                audio = torch.as_tensor(track.sources[source].audio.T, dtype=torch.float32)
                audio = self.source_augmentations(audio)
                audio_sources.append(audio)

                if source == 'vocals':
                    y_vocals = audio
                elif source == 'bass':
                    y_bass = audio
                elif source == 'other':
                    y_other = audio
                elif source == 'drums':
                    y_drums = audio

            # create stem tensor of shape (source, channel, samples)
            stems = torch.stack(audio_sources, dim=0)
            # # apply linear mix over source index=0
            x = stems.sum(0)

        # for validation and test, we deterministically yield the full
        # pre-mixed musdb track
        else:
            x = torch.as_tensor(track.audio.T, dtype=torch.float32)
            y_bass = torch.as_tensor(track.targets["bass"].audio.T, dtype=torch.float32)
            y_vocals = torch.as_tensor(track.targets["vocals"].audio.T, dtype=torch.float32)
            y_other = torch.as_tensor(track.targets["other"].audio.T, dtype=torch.float32)
            y_drums = torch.as_tensor(track.targets["drums"].audio.T, dtype=torch.float32)

        return x, y_bass, y_vocals, y_other, y_drums

    def __len__(self):
        return len(self.mus.tracks) * self.samples_per_track
