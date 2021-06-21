from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, ReLU, Linear, LSTM, Tanh, BatchNorm1d, BatchNorm2d, BatchNorm3d, MaxPool1d, MaxUnpool1d, ConvTranspose2d, Conv2d, Sequential, Sigmoid
from .filtering import atan2
from .transforms import make_filterbanks, ComplexNorm, phasemix_sep, NSGTBase, coefs_to_db
from collections import defaultdict
import numpy as np
import logging

eps = 1.e-10


class OpenUnmix(nn.Module):
    """OpenUnmix Core spectrogram based separation module.

    Args:
        nb_bins, M (int): Number of sliCQ-NSGT tf bins
        nb_channels (int): Number of input audio channels (Default: `2`).
    """

    def __init__(
        self,
        nb_bins,
        M,
        nb_channels=2,
        nb_layers=3,
        hidden_size=512,
        temporal_pooling=13,
        unidirectional=False,
        info=False,
    ):
        super(OpenUnmix, self).__init__()

        self.nb_bins = nb_bins
        self.M = M

        #hidden_size = nb_bins
        self.hidden_size = hidden_size

        self.temporal_pooling = temporal_pooling

        # dont do input whitening, just use batchnorms
        self.whiten2d = BatchNorm2d(nb_channels)
        self.whiten3d = BatchNorm3d(nb_channels)

        self.mp = MaxPool1d(temporal_pooling, temporal_pooling, return_indices=True)
        self.mup = MaxUnpool1d(temporal_pooling, temporal_pooling)

        if unidirectional:
            rnn_layers = nb_layers
            rnn_hidden_size = hidden_size
        else:
            rnn_layers = 2*nb_layers
            rnn_hidden_size = hidden_size // 2

        self.fc1 = Sequential(
            Linear(in_features=nb_channels*self.nb_bins, out_features=hidden_size, bias=False),
            BatchNorm1d(hidden_size),
            Tanh()
        )

        self.rnn = LSTM(
            input_size=hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            #dropout=0.4 if nb_layers > 1 else 0,
        )

        fc2_hiddensize = hidden_size * 2 - (hidden_size % 2)
        self.fc2 = Sequential(
            Linear(in_features=fc2_hiddensize, out_features=hidden_size, bias=False),
            BatchNorm1d(hidden_size),
            ReLU()
        )

        self.fc3 = Sequential(
            Linear(in_features=hidden_size, out_features=nb_channels*self.nb_bins, bias=True),
            BatchNorm1d(nb_channels*self.nb_bins),
            ReLU()
        )

        self.postgrow = Sequential(
            ConvTranspose2d(nb_channels, 10, (1, 246), stride=(1, 2), bias=False),
            BatchNorm2d(10),
            ReLU(),
            ConvTranspose2d(10, 20, 1, bias=False),
            BatchNorm2d(20),
            ReLU(),
            ConvTranspose2d(20, 30, 1, bias=False),
            BatchNorm2d(30),
            ReLU(),
            Conv2d(30, 20, 1, bias=False),
            BatchNorm2d(20),
            ReLU(),
            Conv2d(20, 10, 1, bias=False),
            BatchNorm2d(10),
            ReLU(),
            ConvTranspose2d(10, nb_channels, 1, bias=True),
            #BatchNorm2d(nb_channels),
            Sigmoid(),
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

    def forward(self, x: Tensor, x3d: Tensor) -> Tensor:
        if self.info:
            print()

        logging.info(f'-1. x {x.shape}')
        logging.info(f'-1. x3d {x3d.shape}')

        mix = x3d.detach().clone()

        nb_samples, nb_channels, nb_f_bins, nb_slices, nb_m_bins = x3d.shape
        x = x.permute(0, 3, 2, 1)

        x3d = self.whiten3d(x3d)

        x = coefs_to_db(x)
        x = self.whiten2d(x)

        logging.info(f'0. x {x.shape}')
        logging.info(f'0. x3d {x3d.shape}')
        logging.info(f'0. mix {mix.shape}')

        logging.info(f'1. PRE-MAXPOOL {x.shape}')

        # stack channels on fbins
        x = x.reshape(nb_samples, nb_channels*nb_f_bins, -1)
        unpool_size = x.size()
        x, pool_inds = self.mp(x)

        logging.info(f'2. POST-MAXPOOL {x.shape}')

        x = x.reshape(-1, nb_channels*nb_f_bins)

        logging.info(f'3. PRE-LINEAR-1 {x.shape}')

        # first dense stage + batch norm
        x = self.fc1(x)

        logging.info(f'4. POST-LINEAR-1 {x.shape}')

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = x.reshape(-1, nb_samples, self.hidden_size)

        logging.info(f'5. PRE-LSTM {x.shape}')

        # apply 3-layers of stacked LSTM
        rnn_out, _ = self.rnn(x)

        logging.info(f'6. LSTM {rnn_out.shape}')

        # lstm skip connection
        x = torch.cat([x, rnn_out], -1)
        x = x.reshape(-1, x.shape[-1])

        logging.info(f'7. SKIP-CONN {x.shape}')

        logging.info(f'8. PRE-LINEAR-2 {x.shape}')

        # second dense stage
        x = self.fc2(x)

        logging.info(f'9. PRE-LINEAR-3 {x.shape}')

        x = self.fc3(x)

        logging.info(f'9. PRE-PRE-MAX-UNPOOL-3 {x.shape}')

        # reshape back to original dim
        x = x.reshape(nb_samples, nb_channels*nb_f_bins, -1)
        logging.info(f'10. PRE-MAX-UNPOOL-3 {x.shape}')

        x = self.mup(x, pool_inds, output_size=unpool_size)

        logging.info(f'11. PRED SLICQ-SPECTROGRAM {x.shape}')
        x = x.reshape(nb_samples, nb_channels, nb_f_bins, -1)

        logging.info(f'12. PRE-GROW {x.shape}')
        for i, layer in enumerate(self.postgrow):
            sh1 = x.shape
            x = layer(x)
            sh2 = x.shape
            logging.info(f'12. GROW: {sh1} -> {sh2}')

        mix = mix.reshape(nb_samples, nb_channels, nb_f_bins, nb_slices*nb_m_bins)

        logging.info(f'13. POST-GROW {x.shape}')
        logging.info(f'13. mix {mix.shape}')

        # mask as much as we predicted, we can't grow exactly
        mix[..., : x.shape[-1]] *= x
        mix = mix.reshape(nb_samples, nb_channels, nb_f_bins, nb_slices, nb_m_bins)
        return mix


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
