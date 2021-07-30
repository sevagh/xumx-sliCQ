from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, ReLU, Tanh, BatchNorm2d, ConvTranspose2d, Conv2d, Sequential, Sigmoid
from .filtering import atan2, wiener
from .transforms import make_filterbanks, ComplexNorm, phasemix_sep, NSGTBase, overlap_add_slicq
from collections import defaultdict
import numpy as np
import copy

eps = 1.e-10


# just pass input through directly
class DummyTimeBucket(nn.Module):
    def __init__(
        self,
        slicq_sample_input,
    ):
        super(DummyTimeBucket, self).__init__()

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            #p.requires_grad = False
            p.grad = None
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        return x


class OpenUnmixTimeBucket(nn.Module):
    def __init__(
        self,
        slicq_sample_input,
        input_mean=None,
        input_scale=None,
    ):
        super(OpenUnmixTimeBucket, self).__init__()

        nb_samples, nb_channels, nb_f_bins, nb_slices, nb_t_bins = slicq_sample_input.shape

        channels = [nb_channels, 25, 55]
        layers = len(channels)-1

        if nb_f_bins < 10:
            freq_filter = 1
        elif nb_f_bins < 20:
            freq_filter = 3
        else:
            freq_filter = 5

        if nb_t_bins <= 100:
            time_filter = 7
        else:
            time_filter = 13

        filters = [(freq_filter, time_filter)]*layers

        encoder = []
        decoder = []

        layers = len(filters)

        for i in range(layers):
            encoder.append(
                Conv2d(channels[i], channels[i+1], filters[i], dilation=(1,2), bias=False)
            )
            encoder.append(
                BatchNorm2d(channels[i+1]),
            )
            encoder.append(
                ReLU(),
            )

        for i in range(layers,0,-1):
            decoder.append(
                ConvTranspose2d(channels[i], channels[i-1], filters[i-1], dilation=(1,2), bias=False)
            )
            decoder.append(
                BatchNorm2d(channels[i-1])
            )
            decoder.append(
                ReLU()
            )

        # grow the overlap-added half dimension to its full size
        decoder.append(ConvTranspose2d(nb_channels, nb_channels, (1, 3), stride=(1, 2), bias=True))
        decoder.append(Sigmoid())

        self.cdae = Sequential(*encoder, *decoder)
        self.mask = True

        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean).float()
        else:
            input_mean = torch.zeros(nb_f_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale).float()
        else:
            input_scale = torch.ones(nb_f_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            #p.requires_grad = False
            p.grad = None
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        mix = x.detach().clone()

        x_shape = x.shape
        nb_samples, nb_channels, nb_f_bins, nb_slices, nb_t_bins = x_shape

        x = overlap_add_slicq(x)

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x.permute(0, 1, 3, 2)
        x += self.input_mean
        x *= self.input_scale
        x = x.permute(0, 1, 3, 2)

        for i, layer in enumerate(self.cdae):
            sh1 = x.shape
            x = layer(x)
        
        # crop
        x = x[:, :, :, : nb_t_bins*nb_slices]

        x = x.reshape(x_shape)

        # multiplicative skip connection
        if self.mask:
            x = x * mix

        return x


class OpenUnmix(nn.Module):
    """OpenUnmix Core spectrogram based separation module.

    Args:
        nb_bins, M (int): Number of sliCQ-NSGT tf bins
        nb_channels (int): Number of input audio channels (Default: `2`).
    """
    def __init__(
        self,
        jagged_slicq_sample_input,
        max_bin=None,
        input_means=None,
        input_scales=None,
    ):
        super(OpenUnmix, self).__init__()

        bucketed_unmixes = nn.ModuleList()

        freq_idx = 0
        for i, C_block in enumerate(jagged_slicq_sample_input):
            input_mean = input_means[i] if input_means else None
            input_scale = input_scales[i] if input_scales else None

            freq_start = freq_idx

            if max_bin is not None and freq_start >= max_bin:
                bucketed_unmixes.append(
                    DummyTimeBucket(C_block)
                )
            else:
                bucketed_unmixes.append(
                    OpenUnmixTimeBucket(
                        C_block,
                        input_mean=input_mean,
                        input_scale=input_scale,
                    )
                )

            # advance global frequency pointer
            freq_idx += C_block.shape[2]

        self.bucketed_unmixes_vocals = bucketed_unmixes
        self.bucketed_unmixes_bass = copy.deepcopy(bucketed_unmixes)
        self.bucketed_unmixes_drums = copy.deepcopy(bucketed_unmixes)
        self.bucketed_unmixes_other = copy.deepcopy(bucketed_unmixes)

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            #p.requires_grad = False
            p.grad = None
        self.eval()

    def forward(self, x) -> Tensor:
        futures_vocals = [torch.jit.fork(self.bucketed_unmixes_vocals[i], Xmag_block) for i, Xmag_block in enumerate(x)]
        futures_bass = [torch.jit.fork(self.bucketed_unmixes_bass[i], Xmag_block) for i, Xmag_block in enumerate(x)]
        futures_drums = [torch.jit.fork(self.bucketed_unmixes_drums[i], Xmag_block) for i, Xmag_block in enumerate(x)]
        futures_other = [torch.jit.fork(self.bucketed_unmixes_other[i], Xmag_block) for i, Xmag_block in enumerate(x)]

        y_vocals = [torch.jit.wait(future) for future in futures_vocals]
        y_bass = [torch.jit.wait(future) for future in futures_bass]
        y_drums = [torch.jit.wait(future) for future in futures_drums]
        y_other = [torch.jit.wait(future) for future in futures_other]

        return y_bass, y_vocals, y_other, y_drums


class Separator(nn.Module):
    def __init__(
        self,
        xumx_model,
        xumx_nsgt,
        jagged_slicq_sample_input,
        sample_rate: float = 44100.0,
        nb_channels: int = 2,
        device: str = "cpu",
        niter: int = 1,
        softmask: bool = False,
    ):
        super(Separator, self).__init__()

        # saving parameters
        self.niter = niter
        self.softmask = softmask

        self.device = device

        self.nsgt, self.insgt = make_filterbanks(
            xumx_nsgt, sample_rate=sample_rate
        )

        self.complexnorm = ComplexNorm(mono=nb_channels == 1)
        self.nb_channels = nb_channels

        self.xumx_model = xumx_model
        self.register_buffer("sample_rate", torch.as_tensor(sample_rate))

        # first, get frequency and time limits to build the large zero-padded matrix
        total_f_bins = 0
        max_t_bins = 0
        for i, block in enumerate(jagged_slicq_sample_input):
            nb_samples, nb_channels, nb_f_bins, nb_slices, nb_t_bins = block.shape
            total_f_bins += nb_f_bins
            max_t_bins = max(max_t_bins, nb_t_bins)

        self.total_f_bins = total_f_bins
        self.max_t_bins = max_t_bins

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            #p.requires_grad = False
            p.grad = None
        self.xumx_model.freeze()
        self.eval()

    @torch.no_grad()
    def forward(self, audio: Tensor) -> Tensor:
        """Performing the separation on audio input

        Args:
            audio (Tensor): [shape=(nb_samples, nb_channels, nb_timesteps)]
                mixture audio waveform

        Returns:
            Tensor: stacked tensor of separated waveforms
                shape `(nb_samples, nb_targets, nb_channels, nb_timesteps)`
        """
        nb_sources = 4
        nb_samples = audio.shape[0]

        X = self.nsgt(audio)
        Xmag = self.complexnorm(X)

        # xumx inference - magnitude slicq estimate
        Ymag_bass, Ymag_vocals, Ymag_other, Ymag_drums = self.xumx_model(Xmag)

        print('sliCQ WIENER')

        # block-wise wiener
        # assemble it all into a zero-padded matrix

        nb_slices = X[0].shape[3]
        last_dim = 2

        X_matrix = torch.zeros((nb_samples, self.nb_channels, self.total_f_bins, nb_slices, self.max_t_bins, last_dim), dtype=X[0].dtype, device=X[0].device)
        spectrograms = torch.zeros(X_matrix.shape[:-1] + (nb_sources,), dtype=audio.dtype, device=X_matrix.device)

        freq_start = 0
        for i, X_block in enumerate(X):
            nb_samples, self.nb_channels, nb_f_bins, nb_slices, nb_t_bins, last_dim = X_block.shape

            # assign up to the defined time bins - to the right will be zeros
            X_matrix[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, :] = X_block

            spectrograms[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, 0] = Ymag_vocals[i]
            spectrograms[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, 1] = Ymag_bass[i]
            spectrograms[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, 2] = Ymag_drums[i]
            spectrograms[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, 3] = Ymag_other[i]

            freq_start += nb_f_bins

        spectrograms = wiener(
            torch.squeeze(spectrograms, dim=0),
            torch.squeeze(X_matrix, dim=0),
            self.niter,
            softmask=self.softmask,
        )

        # reverse the wiener/EM permutes etc.
        spectrograms = torch.unsqueeze(spectrograms.permute(2, 1, 0, 3, 4), dim=0)
        spectrograms = spectrograms.reshape(nb_samples, self.nb_channels, self.total_f_bins, nb_slices, self.max_t_bins, *spectrograms.shape[-2:])

        slicq_vocals = [None]*len(X)
        slicq_bass = [None]*len(X)
        slicq_drums = [None]*len(X)
        slicq_other = [None]*len(X)

        estimates = torch.empty(audio.shape + (nb_sources,), dtype=audio.dtype, device=audio.device)

        nb_samples, self.nb_channels, nb_f_bins, nb_slices, nb_t_bins = X_matrix.shape[:-1]

        # matrix back to list form for insgt
        freq_start = 0
        for i, X_block in enumerate(X):
            nb_samples, self.nb_channels, nb_f_bins, nb_slices, nb_t_bins, _ = X_block.shape

            slicq_vocals[i] = spectrograms[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, :, 0].contiguous()
            slicq_drums[i] = spectrograms[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, :, 1].contiguous()
            slicq_bass[i] = spectrograms[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, :, 2].contiguous()
            slicq_other[i] = spectrograms[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, :, 3].contiguous()

            freq_start += nb_f_bins

        estimates[..., 0] = self.insgt(slicq_vocals, audio.shape[-1])
        estimates[..., 1] = self.insgt(slicq_drums, audio.shape[-1])
        estimates[..., 2] = self.insgt(slicq_bass, audio.shape[-1])
        estimates[..., 3] = self.insgt(slicq_other, audio.shape[-1])

        estimates = estimates.permute(0, 3, 1, 2).contiguous()

        return estimates

    def to_dict(self, estimates: Tensor, aggregate_dict: Optional[dict] = None) -> dict:
        """Convert estimates as stacked tensor to dictionary

        Args:
            estimates (Tensor): separated targets of shape
                (nb_samples, nb_targets, nb_channels, nb_timesteps)
            aggregate_dict (dict or None)

        Returns:
            (dict of str: Tensor):
        """
        estimates_dict = {}
        for k, target in enumerate(["vocals", "drums", "bass", "other"]):
            estimates_dict[target] = estimates[:, k, ...]

        if aggregate_dict is not None:
            new_estimates = {}
            for key in aggregate_dict:
                new_estimates[key] = torch.tensor(0.0)
                for target in aggregate_dict[key]:
                    new_estimates[key] = new_estimates[key] + estimates_dict[target]
            estimates_dict = new_estimates
        return estimates_dict
