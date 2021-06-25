from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, ReLU, Linear, LSTM, Tanh, BatchNorm1d, BatchNorm2d, BatchNorm3d, MaxPool2d, MaxUnpool2d, ConvTranspose2d, Conv2d, Sequential, Sigmoid, Conv3d, LSTM, Conv3d, ConvTranspose3d
from .filtering import atan2
from .transforms import make_filterbanks, ComplexNorm, phasemix_sep, NSGTBase, overlap_add_slicq
from collections import defaultdict
import numpy as np
import logging

eps = 1.e-10


class OpenUnmixTimeBucket(nn.Module):
    def __init__(
        self,
        time_bucket,
        slicq_sample_input,
        unidirectional=False,
        info=False,
    ):
        super(OpenUnmixTimeBucket, self).__init__()
        
        self.time_bucket = time_bucket

        nb_samples, nb_channels, nb_f_bins, nb_slices, nb_t_bins = slicq_sample_input.shape

        channels = [nb_channels, 25, 55]
        layers = len(channels)-1

        frequency_filter = max(nb_f_bins//layers, 1)

        filters = [(frequency_filter, 7), (frequency_filter, 7)]
        strides = [(1, 3), (1, 3)]
        dilations = [(1, 1), (1, 1)]
        output_paddings = [(0, 0), (0, 0)]

        encoder = []
        decoder = []

        layers = len(filters)

        for i in range(layers):
            encoder.append(
                Conv2d(channels[i], channels[i+1], filters[i], stride=strides[i], dilation=dilations[i], bias=False)
            )
            encoder.append(
                BatchNorm2d(channels[i+1]),
            )
            encoder.append(
                ReLU(),
            )

        for i in range(layers,0,-1):
            decoder.append(
                ConvTranspose2d(channels[i], channels[i-1], filters[i-1], stride=strides[i-1], dilation=dilations[i-1], output_padding=output_paddings[i-1], bias=False)
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

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
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


class OpenUnmix(nn.Module):
    """OpenUnmix Core spectrogram based separation module.

    Args:
        nb_bins, M (int): Number of sliCQ-NSGT tf bins
        nb_channels (int): Number of input audio channels (Default: `2`).
    """

    def __init__(
        self,
        jagged_slicq_sample_input,
        input_mean=None,
        input_scale=None,
        unidirectional=False,
        info=False,
    ):
        super(OpenUnmix, self).__init__()

        self.bucketed_unmixes = nn.ModuleDict()

        for time_bucket, C_block in jagged_slicq_sample_input.items():
            bucket_name = str(time_bucket)
            self.bucketed_unmixes[bucket_name] = OpenUnmixTimeBucket(bucket_name, C_block)

        #if input_mean is not None:
        #    input_mean = (-input_mean).float()
        #else:
        #    input_mean = torch.zeros(self.nb_bins)

        #if input_scale is not None:
        #    input_scale = (1.0 / input_scale).float()
        #else:
        #    input_scale = torch.ones(self.nb_bins)

        #self.input_mean = Parameter(input_mean)
        #self.input_scale = Parameter(input_scale)

        self.info = info
        if self.info:
            logging.basicConfig(level=logging.INFO)

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, time_bucket: str, x: Tensor) -> Tensor:
        y = self.bucketed_unmixes[time_bucket](x)
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
        seq_dur: float = 6.0,
        device: str = "cpu",
    ):
        super(Separator, self).__init__()

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
        self.seq_dur = seq_dur
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
        N = audio.shape[-1]

        estimates = torch.zeros(audio.shape + (nb_sources,), dtype=audio.dtype, device=self.device)

        for j, (target_name, target_module) in enumerate(self.target_models.items()):
            print(f'separating {target_name}')

            nsgt = self.nsgts[target_name]['nsgt']
            insgt = self.nsgts[target_name]['insgt']

            _, slicq_shape = nsgt.nsgt.predict_input_size(1, 2, self.seq_dur)
            seq_batch = slicq_shape[-2]

            X = nsgt(audio)
            Xmag = self.complexnorm(X)

            Xmagsegs = torch.split(Xmag, seq_batch, dim=3)
            Ymagsegs = []

            for Xmagseg in Xmagsegs:
                # apply current model to get the source magnitude spectrogram
                Ymagseg = target_module(Xmagseg.detach().clone())
                Ymagsegs.append(Ymagseg)

            Ymag = torch.cat(Ymagsegs, dim=3)

            Y = phasemix_sep(X, Ymag)
            y = insgt(Y, audio.shape[-1])

            estimates[..., j] = y

        # getting to (nb_samples, nb_targets, nb_channels, nb_samples)
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
