from pathlib import Path
import torch
import torchaudio
import json
import numpy as np
import os
from xumx_slicq_v2 import data
from .separator import Separator

import argparse


def separate(
    audio,
    separator,
    rate=None,
    device=None,
):
    if rate is None:
        raise Exception("rate` must be provided.")

    if device:
        audio = audio.to(device)
    audio = data.preprocess_audio(audio, rate, separator.sample_rate)

    # getting the separated signals
    estimates = separator(audio)
    estimates = separator.to_dict(estimates)
    return estimates


def main():
    parser = argparse.ArgumentParser(
        description="xumx-sliCQ-V2 Inference",
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--ext",
        type=str,
        default=".wav",
        help="Output extension which sets the audio format",
    )

    parser.add_argument(
        "--start", type=float, default=0.0, help="Audio chunk start in seconds"
    )

    parser.add_argument(
        "--duration",
        type=float,
        help="Audio chunk duration in seconds, negative values load full track",
    )

    parser.add_argument(
        "--audio-backend",
        type=str,
        default="soundfile",
        help="Set torchaudio backend "
        "(`sox_io`, `sox`, `soundfile` or `stempeg`), defaults to `soundfile`",
    )

    args = parser.parse_args()

    if args.audio_backend != "stempeg":
        torchaudio.set_audio_backend(args.audio_backend)

    # explicitly use no GPUs for inference
    use_cuda = False

    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using ", device)

    # create separator only once to reduce model loading
    # when using multiple files
    separator = Separator.load(
        device=device
    )

    separator.freeze()
    separator.to(device)

    if args.audio_backend == "stempeg":
        try:
            import stempeg
        except ImportError:
            raise RuntimeError("Please install pip package `stempeg`")

    # loop over the files
    for wav_file in os.listdir("/input"):
        input_file = os.path.join("/input", wav_file)
        if args.audio_backend == "stempeg":
            audio, rate = stempeg.read_stems(
                input_file,
                start=args.start,
                duration=args.duration,
                sample_rate=separator.sample_rate,
                dtype=np.float32,
            )
            audio = torch.tensor(audio, device=device)
        else:
            audio, rate = data.load_audio(
                input_file, start=args.start, dur=args.duration
            )
        estimates = separate(
            audio=audio,
            rate=rate,
            separator=separator,
            device=device,
        )

        outdir = Path("/output") / Path(input_file).stem
        outdir.mkdir(exist_ok=True, parents=True)

        # write out estimates
        if args.audio_backend == "stempeg":
            target_path = str(outdir / Path("target").with_suffix(args.ext))
            # convert torch dict to numpy dict
            estimates_numpy = {}
            for target, estimate in estimates.items():
                estimates_numpy[target] = (
                    torch.squeeze(estimate).detach().cpu().numpy().T
                )

            stempeg.write_stems(
                target_path,
                estimates_numpy,
                sample_rate=separator.sample_rate,
                writer=stempeg.FilesWriter(multiprocess=True, output_sample_rate=rate),
            )
        else:
            for target, estimate in estimates.items():
                target_path = str(outdir / Path(target).with_suffix(args.ext))
                torchaudio.save(
                    target_path,
                    torch.squeeze(estimate).to("cpu"),
                    sample_rate=separator.sample_rate,
                )


if __name__ == "__main__":
    main()
