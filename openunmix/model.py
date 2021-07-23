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
            p.requires_grad = False
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        mix = x.detach().clone()
        return mix


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
            p.requires_grad = False
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
            p.requires_grad = False
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
    """
    Separator class to encapsulate all the stereo filtering
    as a torch Module, to enable end-to-end learning.

    Args:
        targets (dict of str: nn.Module): dictionary of target models
            the spectrogram models to be used by the Separator.
        niter (int): Number of EM steps for refining initial estimates in a
            post-processing stage. Zeroed if only one target is estimated.
            defaults to `1`.
        residual (bool): adds an additional residual target, obtained by
            subtracting the other estimated targets from the mixture,
            before any potential EM post-processing.
            Defaults to `False`.
        wiener_win_len (int or None): The size of the excerpts
            (number of frames) on which to apply filtering
            independently. This means assuming time varying stereo models and
            localization of sources.
            None means not batching but using the whole signal. It comes at the
            price of a much larger memory usage.
    """

    def __init__(
        self,
        xumx_model,
        xumx_nsgt,
        sample_rate: float = 44100.0,
        nb_channels: int = 2,
        device: str = "cpu",
        niter: int = 1,
        softmask: bool = False,
        residual: bool = False,
        wiener_win_len: Optional[int] = 300,
    ):
        super(Separator, self).__init__()

        # saving parameters
        self.niter = niter
        self.residual = residual
        self.softmask = softmask
        self.wiener_win_len = wiener_win_len

        self.n_fft = 4096
        self.n_hop = 1024

        self.device = device

        self.nsgt, self.insgt = make_filterbanks(
            xumx_nsgt, sample_rate=sample_rate
        )

        self.complexnorm = ComplexNorm(mono=nb_channels == 1)
        self.nb_channels = nb_channels

        self.xumx_model = xumx_model
        self.register_buffer("sample_rate", torch.as_tensor(sample_rate))

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

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

        # mix phase + magnitude estimate
        Ycomplex_bass = phasemix_sep(X, Ymag_bass)
        Ycomplex_vocals = phasemix_sep(X, Ymag_vocals)
        Ycomplex_drums = phasemix_sep(X, Ymag_drums)
        Ycomplex_other = phasemix_sep(X, Ymag_other)

        y_bass = self.insgt(Ycomplex_bass, audio.shape[-1])
        y_drums = self.insgt(Ycomplex_drums, audio.shape[-1])
        y_other = self.insgt(Ycomplex_other, audio.shape[-1])
        y_vocals = self.insgt(Ycomplex_vocals, audio.shape[-1])

        # wiener part
        print('STFT WIENER')
        audio = torch.squeeze(audio, dim=0)
        
        mix_stft = torch.view_as_real(torch.stft(audio, self.n_fft, hop_length=self.n_hop, return_complex=True))
        X = torch.abs(torch.view_as_complex(mix_stft))
        
        # initializing spectrograms variable
        spectrograms = torch.zeros(X.shape + (nb_sources,), dtype=audio.dtype, device=X.device)

        for j, target_name in enumerate(["vocals", "drums", "bass", "other"]):
            # apply current model to get the source spectrogram
            if target_name == 'bass':
                target_est = torch.squeeze(y_bass, dim=0)
            elif target_name == 'vocals':
                target_est = torch.squeeze(y_vocals, dim=0)
            elif target_name == 'drums':
                target_est = torch.squeeze(y_drums, dim=0)
            elif target_name == 'other':
                target_est = torch.squeeze(y_other, dim=0)
            spectrograms[..., j] = torch.abs(torch.stft(target_est, self.n_fft, hop_length=self.n_hop, return_complex=True))

        # transposing it as
        # (nb_samples, nb_frames, nb_bins,{1,nb_channels}, nb_sources)

        spectrograms = spectrograms.permute(2, 1, 0, 3)

        # rearranging it into:
        # (nb_samples, nb_frames, nb_bins, nb_channels, 2) to feed
        # into filtering methods
        mix_stft = mix_stft.permute(2, 1, 0, 3)

        # create an additional target if we need to build a residual
        if self.residual:
            # we add an additional target
            nb_sources += 1

        if nb_sources == 1 and self.niter > 0:
            raise Exception(
                "Cannot use EM if only one target is estimated."
                "Provide two targets or create an additional "
                "one with `--residual`"
            )

        nb_frames = spectrograms.shape[0]
        targets_stft = torch.zeros(
            mix_stft.shape + (nb_sources,), dtype=audio.dtype, device=mix_stft.device
        )

        pos = 0
        if self.wiener_win_len:
            wiener_win_len = self.wiener_win_len
        else:
            wiener_win_len = nb_frames
        while pos < nb_frames:
            cur_frame = torch.arange(pos, min(nb_frames, pos + wiener_win_len))
            pos = int(cur_frame[-1]) + 1

            targets_stft[cur_frame] = wiener(
                spectrograms[cur_frame],
                mix_stft[cur_frame],
                self.niter,
                softmask=self.softmask,
                residual=self.residual,
            )

        # getting to (nb_samples, nb_targets, channel, fft_size, n_frames, 2)
        targets_stft = torch.view_as_complex(targets_stft.permute(4, 2, 1, 0, 3).contiguous())

        # inverse STFT
        estimates = torch.empty(audio.shape + (nb_sources,), dtype=audio.dtype, device=audio.device)

        for j, target_name in enumerate(["vocals", "drums", "bass", "other"]):
            estimates[..., j] = torch.istft(targets_stft[j, ...], self.n_fft, hop_length=self.n_hop, length=audio.shape[-1])

        estimates = torch.unsqueeze(estimates, dim=0).permute(0, 3, 1, 2).contiguous()
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
