from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Parameter, ReLU, Sigmoid, BatchNorm2d, Conv2d, ConvTranspose2d, Tanh, LSTM, BatchNorm1d, Dropout, ELU
from .filtering import atan2
from .transforms import make_filterbanks, ComplexNorm, phasemix_sep, NSGTBase
from collections import defaultdict
import numpy as np
import logging

eps = 1.e-10


class OpenUnmix(nn.Module):
    """OpenUnmix Core spectrogram based separation module.

    Args:
        nb_bins, M (int): Number of sliCQ-NSGT tf bins
        nb_channels (int): Number of input audio channels (Default: `2`).
        input_mean (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to zeros(nb_bins)
        input_scale (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to ones(nb_bins)
    """

    def __init__(
        self,
        nb_bins,
        M,
        seq_batch,
        nb_channels=2,
        input_mean=None,
        input_scale=None,
        info=False,
    ):
        super(OpenUnmix, self).__init__()

        self.nb_bins = nb_bins
        self.M = M

        # TODO support causality with causal convolutions
        channels = [nb_channels, 25, 55, 75]
        filters = [(13, 13), (11, 11), (9, 9)]
        strides = [(1, 1), (1, 1), (1, 1)]
        dilations = [(1, 1), (1, 4), (1, 8)]
        output_paddings = [(0, 0), (0, 0), (0, 0)]

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        layers = len(filters)

        # input channel

        for i in range(layers):
            self.encoder.extend([
                Conv2d(channels[i], channels[i+1], filters[i], stride=strides[i], dilation=dilations[i], bias=False),
                BatchNorm2d(channels[i+1]),
                ReLU(),
            ])

        for i in range(layers,1,-1):
            self.decoder.extend([
                ConvTranspose2d(channels[i], channels[i-1], filters[i-1], stride=strides[i-1], dilation=dilations[i-1], output_padding=output_paddings[i-1], bias=False),
                BatchNorm2d(channels[i-1]),
                ReLU(),
            ])

        # last layer has special activation
        self.decoder.append(ConvTranspose2d(channels[1], channels[0], filters[0], stride=strides[0], dilation=dilations[0], output_padding=output_paddings[0], bias=False))
        #self.decoder.append(BatchNorm2d(channels[0]))
        self.decoder.append(Sigmoid())
        self.mask = True

        if input_mean is not None:
            input_mean = (-input_mean).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = (1.0 / input_scale).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

        self.info = info
        if self.info:
            logging.basicConfig(level=logging.INFO)

        # chunked for inputs longer than the seq
        self.seq_batch = seq_batch

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, xlong: Tensor) -> Tensor:
        if self.info:
            print()

        logging.info(f'-3. {xlong.shape}')
        mix = xlong.detach().clone()
        logging.info(f'0. mix shape: {mix.shape}')

        pad = int(np.ceil(xlong.shape[3]/self.seq_batch))*self.seq_batch - xlong.shape[3]
        xlong = F.pad(xlong, (0, 0, 0, pad), mode='constant', value=0)

        x = torch.cat([torch.unsqueeze(x_, dim=0) for x_ in torch.split(xlong, self.seq_batch, dim=3)])
        logging.info(f'-2. {x.shape}')
        nb_splits = x.shape[0]
        nb_samples_orig = x.shape[1]

        # stack splits as samples
        x = x.reshape(x.shape[0]*x.shape[1], *x.shape[2:])
        logging.info(f'-1. {x.shape}')

        nb_samples, nb_channels, nb_f_bins, nb_slices, nb_m_bins = x.shape

        x = x.reshape(nb_samples, nb_channels, nb_f_bins, -1)

        # permute so that batch is last for lstm
        x = x.permute(3, 0, 1, 2)

        logging.info(f'0. {x.shape}')

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x + self.input_mean[: self.nb_bins]
        x = x * self.input_scale[: self.nb_bins]

        logging.info(f'1. POST-SCALE {x.shape}')

        x = x.permute(1, 2, 3, 0)

        x = x.reshape(nb_samples, nb_channels, self.nb_bins, -1)

        logging.info(f'5. PRE-ENCODER {x.shape}')

        for i, layer in enumerate(self.encoder):
            sh1 = x.shape
            x = layer(x)
            sh2 = x.shape
            logging.info(f'\t5-{i} ENCODER {sh1} -> {sh2}')

        post_enc_shape = x.shape
        logging.info(f'6. POST-ENCODER {x.shape}')

        for layer in self.decoder:
            sh1 = x.shape
            x = layer(x)
            sh2 = x.shape
            logging.info(f'\t6-{i} DECODER {sh1} -> {sh2}')

        logging.info(f'7. POST-DECODER {x.shape}')

        x = x.reshape(nb_samples, nb_channels, nb_slices, nb_f_bins, nb_m_bins)
        x = x.permute(0, 1, 3, 2, 4)

        logging.info(f'8. RET {x.shape}')
        x = x.reshape(nb_samples_orig, nb_channels, nb_f_bins, nb_slices*nb_splits, nb_m_bins)
        logging.info(f'9. RET {x.shape}')
        if pad > 0:
            x = x[..., : -pad, :]
        logging.info(f'10. RET PAD CROPPED {x.shape}')

        logging.info(f'10. mix {mix.shape}')

        if self.mask:
            x = x*mix

        return x


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

            slicq_shape = nsgt.nsgt.predict_input_size(1, 2, self.seq_dur)
            seq_batch = slicq_shape[-2]

            X = nsgt(audio)
            Xmag = self.complexnorm(X)

            Xmagsegs = torch.split(Xmag, seq_batch, dim=3)
            Ymagsegs = []

            for Xmagseg in Xmagsegs:
                # apply current model to get the source magnitude spectrogram
                #Xmag_segs = torch.split(Xmag, 
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
