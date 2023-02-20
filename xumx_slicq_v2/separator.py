from typing import Optional, Union, Tuple
import sys
from tqdm import trange
from pathlib import Path
import torch
import json
from torch import Tensor
import torch.nn as nn
from .models import Unmix
import norbert
from .transforms import (
    ComplexNorm,
    NSGTBase,
    make_filterbanks,
)


class Separator(nn.Module):
    @classmethod
    def load(
        cls,
        chunk_size: int = 2621440,
        device: Union[str, torch.device] = "cpu",
    ):
        model_path = "/xumx-sliCQ-V2/pretrained_model"
        model_path = Path(model_path)

        # when path exists, we assume its a custom model saved locally
        assert model_path.exists()

        with open(Path(model_path, "separator.json"), "r") as stream:
            enc_conf = json.load(stream)

        xumx_model, encoder = load_target_models(
            model_path,
            sample_rate=enc_conf["sample_rate"],
            device=device,
        )

        separator = Separator(
            xumx_model=xumx_model,
            encoder=encoder,
            sample_rate=enc_conf["sample_rate"],
            chunk_size=chunk_size,
        ).to(device)

        separator.freeze()
        return separator

    def __init__(
        self,
        xumx_model: Unmix = None,
        encoder: Tuple = None,
        sample_rate: float = 44100.0,
        chunk_size: Optional[int] = 2621440,
        device: str = "cpu",
    ):
        super(Separator, self).__init__()
        # saving parameters

        self.device = device
        self.nb_channels = 2
        self.register_buffer("sample_rate", torch.as_tensor(sample_rate))
        self.chunk_size = chunk_size if chunk_size is not None else sys.maxsize

        self.xumx_model = xumx_model
        self.nsgt, self.insgt, self.cnorm = encoder

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            # p.requires_grad = False
            p.grad = None
        self.xumx_model.freeze()
        self.eval()

    def forward(self, audio_big: Tensor) -> Tensor:
        """Performing the separation on audio input

        Args:
            audio (Tensor): [shape=(nb_samples, nb_channels, nb_timesteps)]
                mixture audio waveform

        Returns:
            Tensor: stacked tensor of separated waveforms
                shape `(nb_samples, nb_targets, nb_channels, nb_timesteps)`
        """
        nb_samples = audio_big.shape[0]
        N = audio_big.shape[-1]

        nchunks = N // self.chunk_size
        if (N % self.chunk_size) != 0:
            nchunks += 1

        print(f"n chunks: {nchunks}")

        final_estimates = []

        for chunk_idx in trange(nchunks):
            audio = audio_big[
                ...,
                chunk_idx * self.chunk_size : min((chunk_idx + 1) * self.chunk_size, N),
            ]
            print(f"audio.shape: {audio.shape}")

            n_samples = audio.shape[-1]

            X = self.nsgt(audio)
            Xmag = self.cnorm(X)

            # embedded wiener
            Ycomplex_all = self.xumx_model(X)

            estimates = self.insgt(Ycomplex_all, n_samples)
            final_estimates.append(estimates)

        ests_concat = torch.cat(final_estimates, axis=-1)
        print(f"ests concat: {ests_concat.shape}")
        return ests_concat

    @staticmethod
    def to_dict(estimates: Tensor, aggregate_dict: Optional[dict] = None) -> dict:
        """Convert estimates as stacked tensor to dictionary

        Args:
            estimates (Tensor): separated targets of shape
                (nb_samples, nb_targets, nb_channels, nb_timesteps)
            aggregate_dict (dict or None)

        Returns:
            (dict of str: Tensor):
        """
        estimates_dict = {}

        # follow the ordering in data.py
        for k, target in enumerate(["bass", "vocals", "other", "drums"]):
            estimates_dict[target] = estimates[k]

        if aggregate_dict is not None:
            new_estimates = {}
            for key in aggregate_dict:
                new_estimates[key] = torch.tensor(0.0)
                for target in aggregate_dict[key]:
                    new_estimates[key] = new_estimates[key] + estimates_dict[target]
            estimates_dict = new_estimates
        return estimates_dict


def load_target_models(
        model_path: str, device="cpu", sample_rate=44100,
):
    model_name = "xumx_slicq_v2"
    model_path = Path(model_path).expanduser()

    # load model from disk
    with open(Path(model_path, f"{model_name}.json"), "r") as stream:
        results = json.load(stream)

    # need to configure an NSGT object to peek at its params to set up the neural network
    # e.g. M depends on the sllen which depends on fscale+fmin+fmax
    nsgt_base = NSGTBase(
        results["args"]["fscale"],
        results["args"]["fbins"],
        results["args"]["fmin"],
        fs=sample_rate,
        device=device,
    )

    nb_channels = 2

    seq_dur = results["args"]["seq_dur"]

    target_model_path = Path(model_path, f"{model_name}.pth")
    state = torch.load(target_model_path, map_location=device)

    jagged_slicq, _ = nsgt_base.predict_input_size(1, nb_channels, seq_dur)
    cnorm = ComplexNorm().to(device)

    nsgt, insgt = make_filterbanks(
        nsgt_base, sample_rate
    )
    encoder = (nsgt, insgt, cnorm)

    nsgt = nsgt.to(device)
    insgt = insgt.to(device)

    jagged_slicq_cnorm = cnorm(jagged_slicq)

    xumx_model = Unmix(
        jagged_slicq_cnorm,
    )

    xumx_model.load_state_dict(state, strict=False)
    xumx_model.freeze()
    xumx_model.to(device)

    return xumx_model, encoder
