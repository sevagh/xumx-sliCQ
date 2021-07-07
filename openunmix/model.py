from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, ReLU, Linear, LSTM, Tanh, BatchNorm1d, BatchNorm2d, BatchNorm3d, MaxPool2d, MaxUnpool2d, ConvTranspose2d, Conv2d, Sequential, Sigmoid, Conv3d, LSTM, Conv3d, ConvTranspose3d
from .filtering import atan2, wiener
from .transforms import make_filterbanks, ComplexNorm, phasemix_sep, NSGTBase, overlap_add_slicq
from collections import defaultdict
import numpy as np
import logging

eps = 1.e-10


class OpenUnmixTimeBucket(nn.Module):
    def __init__(
        self,
        slicq_sample_input,
        chans="25,55",
        dropout=-1.,
        min_freq_filter=1,
        max_freq_filter=5,
        min_time_filter=3,
        max_time_filter=23,
        time_stride=5,
        info=False,
        legacy=False,
    ):
        super(OpenUnmixTimeBucket, self).__init__()

        nb_samples, nb_channels, nb_f_bins, nb_slices, nb_t_bins = slicq_sample_input.shape

        self.legacy = legacy

        if legacy:
            channels = [nb_channels, 25, 55]
            layers = len(channels)-1

            frequency_filter = max(nb_f_bins//layers, 1)

            filters = [(frequency_filter, 7), (frequency_filter, 7)]
            strides = [(1, 3), (1, 3)]
            output_paddings = [(0, 0), (0, 0)]

            encoder = []
            decoder = []

            layers = len(filters)

            # batchnorm instead of data preprocessing/whitening
            encoder.append(BatchNorm2d(nb_channels))

            for i in range(layers):
                encoder.append(
                    Conv2d(channels[i], channels[i+1], filters[i], stride=strides[i], bias=False)
                )
                encoder.append(
                    BatchNorm2d(channels[i+1]),
                )
                encoder.append(
                    ReLU(),
                )

            for i in range(layers,0,-1):
                decoder.append(
                    ConvTranspose2d(channels[i], channels[i-1], filters[i-1], stride=strides[i-1], output_padding=output_paddings[i-1], bias=False)
                )
                decoder.append(
                    BatchNorm2d(channels[i-1])
                )
                decoder.append(
                    ReLU()
                )

            self.cdae = Sequential(*encoder, *decoder)

            self.grow = Sequential(
                ConvTranspose2d(nb_channels, nb_channels, (1, nb_t_bins), stride=(1, 2), bias=True),
                Sigmoid()
            )

            self.mask = True
        else:
            channels = [nb_channels] + [int(chan) for chan in chans.split(",")]
            layers = len(channels)-1

            # start off with the biggest possible kernel size
            # the slicq dimension divided by the number of layers
            freq_filter = nb_f_bins//layers
            time_filter = nb_t_bins//layers

            # then, cap it by the max
            if freq_filter >= max_freq_filter:
                freq_filter = min(freq_filter, max_freq_filter)
            else:
                freq_filter = max(min(freq_filter, min_freq_filter), 1)

            if time_filter >= max_time_filter:
                time_filter = min(time_filter, max_time_filter)
            else:
                time_filter = max(min(time_filter, min_time_filter), 1)

            filters = [(freq_filter, time_filter), (freq_filter, time_filter)]
            strides = [(1, time_stride), (1, time_stride)]

            encoder = []
            decoder = []

            layers = len(filters)

            # batchnorm instead of data preprocessing/whitening
            encoder.append(BatchNorm2d(nb_channels))

            for i in range(layers):
                encoder.append(
                    Conv2d(channels[i], channels[i+1], filters[i], stride=strides[i], bias=False)
                )
                encoder.append(
                    BatchNorm2d(channels[i+1]),
                )
                encoder.append(
                    ReLU(),
                )

            if dropout > 0.:
                encoder.append(Dropout(dropout))

            for i in range(layers,0,-1):
                decoder.append(
                    ConvTranspose2d(channels[i], channels[i-1], filters[i-1], stride=strides[i-1], bias=False)
                )
                decoder.append(
                    BatchNorm2d(channels[i-1])
                )
                decoder.append(
                    ReLU()
                )

            decoder.append(ConvTranspose2d(nb_channels, nb_channels, (1, int(1.5*nb_t_bins)), stride=(1, 2), bias=True))
            decoder.append(Sigmoid())

            self.cdae = Sequential(*encoder, *decoder)
            self.mask = True

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        if self.legacy:
            mix = x.detach().clone()
            logging.info(f'0. mix shape: {mix.shape}')
            logging.info(f'0. x shape: {x.shape}')

            x_shape = x.shape
            nb_samples, nb_channels, nb_f_bins, nb_slices, nb_t_bins = x_shape

            #x = torch.flatten(x, start_dim=-2, end_dim=-1)
            x = overlap_add_slicq(x)

            logging.info(f'1. PRE-CDAE: {x.shape}')

            for i, layer in enumerate(self.cdae):
                sh1 = x.shape
                x = layer(x)
                logging.info(f'\t2-{i}. {sh1} -> {x.shape}')

            logging.info(f'3. POST-CDAE: {x.shape}')

            logging.info(f'4. GROW: {x.shape}')

            for i, layer in enumerate(self.grow):
                sh1 = x.shape
                x = layer(x)
                logging.info(f'\t4-{i}. {sh1} -> {x.shape}')

            logging.info(f'5. POST-GROW: {x.shape}')
            
            # crop
            x = x[:, :, :, : nb_t_bins*nb_slices]

            logging.info(f'6. CROPPED: {x.shape}')

            x = x.reshape(x_shape)

            logging.info(f'7. mix shape: {mix.shape}')
            logging.info(f'7. mask shape: {x.shape}')

            if self.mask:
                x = x * mix

            return x
        else:
            mix = x.detach().clone()
            logging.info(f'0. mix shape: {mix.shape}')
            logging.info(f'0. x shape: {x.shape}')

            x_shape = x.shape
            nb_samples, nb_channels, nb_f_bins, nb_slices, nb_t_bins = x_shape

            #x = torch.flatten(x, start_dim=-2, end_dim=-1)
            x = overlap_add_slicq(x)

            logging.info(f'1. PRE-CDAE: {x.shape}')

            for i, layer in enumerate(self.cdae):
                sh1 = x.shape
                x = layer(x)
                logging.info(f'\t2-{i}. {sh1} -> {x.shape}')

            logging.info(f'3. POST-CDAE: {x.shape}')
            
            # crop
            x = x[:, :, :, : nb_t_bins*nb_slices]

            logging.info(f'6. CROPPED: {x.shape}')

            x = x.reshape(x_shape)

            logging.info(f'7. mix shape: {mix.shape}')
            logging.info(f'7. mask shape: {x.shape}')

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
        bin_pass=1,
        dropout=-1.,
        chans="25,55",
        freq_filters=(1,5),
        time_filters=(3,23),
        time_stride=5,
        info=False,
        legacy=False,
    ):
        super(OpenUnmix, self).__init__()

        self.bucketed_unmixes = nn.ModuleList()

        for i, C_block in enumerate(jagged_slicq_sample_input):
            self.bucketed_unmixes.append(
                OpenUnmixTimeBucket(
                    C_block,
                    chans=chans,
                    dropout=dropout,
                    min_freq_filter=freq_filters[0],
                    max_freq_filter=freq_filters[1],
                    min_time_filter=time_filters[0],
                    max_time_filter=time_filters[1],
                    time_stride=time_stride,
                    legacy=legacy,
                )
            )

        self.info = info
        if self.info:
            logging.basicConfig(level=logging.INFO)

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x) -> Tensor:
        futures = [torch.jit.fork(self.bucketed_unmixes[i], Xmag_block) for i, Xmag_block in enumerate(x)]
        y = [torch.jit.wait(future) for future in futures]
        return y


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
        target_models: dict,
        target_models_nsgt: dict,
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

        self.nsgts = defaultdict(dict)
        self.device = device

        # separate nsgt per model
        for name, nsgt_base in target_models_nsgt.items():
            nsgt, insgt = make_filterbanks(
                nsgt_base, sample_rate=sample_rate
            )

            self.nsgts[name]['nsgt'] = nsgt
            self.nsgts[name]['insgt'] = insgt

        self.complexnorm = ComplexNorm(mono=nb_channels == 1)
        self.nb_channels = nb_channels

        # registering the targets models
        self.target_models = nn.ModuleDict(target_models)
        # adding till https://github.com/pytorch/pytorch/issues/38963
        self.nb_targets = len(self.target_models)
        # get the sample_rate as the sample_rate of the first model
        # (tacitly assume it's the same for all targets)
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
        nb_sources = self.nb_targets
        nb_samples = audio.shape[0]

        estimates_1 = torch.zeros(audio.shape + (nb_sources,), dtype=audio.dtype, device=self.device)

        for j, (target_name, target_module) in enumerate(self.target_models.items()):
            nsgt = self.nsgts[target_name]['nsgt']
            insgt = self.nsgts[target_name]['insgt']

            X = nsgt(audio)
            Xmag = self.complexnorm(X)

            Ymag = target_module(Xmag)

            Ycomplex = [None]*len(X)
            for i, X_block in enumerate(X):
                Ycomplex[i] = phasemix_sep(X_block, Ymag[i])

            y = insgt(Ycomplex, audio.shape[-1])
            estimates_1[..., j] = y

        # wiener part
        print('STFT WIENER')
        audio = torch.squeeze(audio, dim=0)

        mix_stft = torch.view_as_real(torch.stft(audio, self.n_fft, hop_length=self.n_hop, return_complex=True))
        X = torch.abs(torch.view_as_complex(mix_stft))

        # initializing spectrograms variable
        spectrograms = torch.zeros(X.shape + (nb_sources,), dtype=audio.dtype, device=X.device)

        for j, target_name in enumerate(self.target_models.keys()):
            # apply current model to get the source spectrogram
            target_est = torch.squeeze(estimates_1[..., j], dim=0)
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

        for j, target_name in enumerate(self.target_models.keys()):
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
        for k, target in enumerate(self.target_models):
            estimates_dict[target] = estimates[:, k, ...]

        if aggregate_dict is not None:
            new_estimates = {}
            for key in aggregate_dict:
                new_estimates[key] = torch.tensor(0.0)
                for target in aggregate_dict[key]:
                    new_estimates[key] = new_estimates[key] + estimates_dict[target]
            estimates_dict = new_estimates
        return estimates_dict
