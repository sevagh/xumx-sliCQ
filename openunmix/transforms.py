from typing import Optional

import numpy as np
import torch
import torchaudio
from torch import Tensor
import torch.nn as nn
from .filtering import atan2
import warnings

from .nsgt import NSGT_sliced, BarkScale, MelScale, LogScale, VQLogScale, OctScale


def phasemix_sep(X, Ymag):
    Xphase = atan2(X[..., 1], X[..., 0])
    Ycomplex = torch.empty_like(X)

    Ycomplex[..., 0] = Ymag * torch.cos(Xphase)
    Ycomplex[..., 1] = Ymag * torch.sin(Xphase)
    return Ycomplex


def overlap_add_slicq(slicq):
    nb_samples, nb_channels, nb_f_bins, nb_slices, nb_m_bins = slicq.shape

    slicq = torch.flatten(slicq, start_dim=-2, end_dim=-1)

    slicq = slicq.reshape(nb_samples, nb_channels*nb_f_bins, -1)
    sh = nb_m_bins//2
    fr = 0

    slicq[..., fr:fr+sh] = slicq[..., :sh]

    for i in range(nb_slices):
        slicq[..., fr:fr+sh] += slicq[..., i*nb_m_bins:(i*nb_m_bins+sh)]
        fr += sh
        slicq[..., fr:fr+sh] += slicq[..., (i*nb_m_bins+sh):(i*nb_m_bins+2*sh)]

    slicq = slicq[..., : nb_m_bins*nb_slices//2]
    slicq = slicq.reshape(nb_samples, nb_channels, nb_f_bins, -1)

    return slicq


def make_filterbanks(nsgt_base, sample_rate=44100.0):
    if sample_rate != 44100.0:
        raise ValueError('i was lazy and harcoded a lot of 44100.0, forgive me')

    encoder = NSGT_SL(nsgt_base)
    decoder = INSGT_SL(nsgt_base)

    return encoder, decoder


# for plotting
def coefs_to_db(mls, pooling=None):
    # compute magnitude spectrum
    mindb = -100.
    mls = torch.abs(mls)
    mindb = torch.empty_like(mls).fill_(10**(mindb/20.))
    mls = torch.maximum(mls, mindb)
    mls = torch.log10(mls)
    mls *= 20.
    return mls


class NSGTBase(nn.Module):
    def __init__(self, scale, fbins, fmin, sllen, fs=44100, device="cuda", gamma=25.):
        super(NSGTBase, self).__init__()
        self.fbins = fbins
        self.fmin = fmin
        self.fmax = fs/2

        self.scl = None
        if scale == 'bark':
            self.scl = BarkScale(self.fmin, self.fmax, self.fbins)
        elif scale == 'mel':
            self.scl = MelScale(self.fmin, self.fmax, self.fbins)
        elif scale == 'cqlog':
            self.scl = LogScale(self.fmin, self.fmax, self.fbins)
        elif scale == 'vqlog':
            self.scl = VQLogScale(self.fmin, self.fmax, self.fbins, self.gamma)
        elif scale == 'oct':
            self.scl = OctScale(self.fmin, self.fmax, self.fbins)
        else:
            raise ValueError(f'unsupported frequency scale {scale}')

        self.sllen = sllen

        min_sllen = self.scl.suggested_sllen(fs)

        if self.sllen < min_sllen:
            warnings.warn(f"slice length is too short for desired frequency scale, need {min_sllen}")

        trlen = self.sllen//4
        trlen = trlen + -trlen % 2 # make trlen divisible by 2
        self.trlen = trlen

        self.nsgt = NSGT_sliced(self.scl, self.sllen, self.trlen, fs, real=True, matrixform=True, multichannel=True, device=device)
        self.M = self.nsgt.ncoefs
        self.fs = fs
        self.fbins_actual = self.nsgt.fbins_actual

    def max_bins(self, bandwidth): # convert hz bandwidth into bins
        if bandwidth is None:
            return None
        freqs, _ = self.scl()
        max_bin = min(np.argwhere(freqs > bandwidth))[0]
        return max_bin+1

    def predict_input_size(self, batch_size, nb_channels, seq_dur_s):
        x = torch.rand((batch_size, nb_channels, int(seq_dur_s*self.fs)), dtype=torch.float32)
        shape = x.size()
        nb_samples, nb_channels, nb_timesteps = shape

        # pack batch
        x = x.view(-1, shape[-1])

        C = self.nsgt.forward((x,))
        #S, I, F, T = C.shape
        # slice, channels, frequency bins, time bins

        # first, moveaxis S, I, F, T to I, F, S, T
        C = torch.moveaxis(C, 0, -2)

        nsgt_f = torch.view_as_real(C)

        # unpack batch
        nsgt_f = nsgt_f.view(shape[:-1] + nsgt_f.shape[-4:])

        # real shape is magnitude, dropping implicit complex dimension
        return nsgt_f, nsgt_f.shape[:-1]

    def _apply(self, fn):
        self.nsgt._apply(fn)
        return self


class NSGT_SL(nn.Module):
    def __init__(self, nsgt):
        super(NSGT_SL, self).__init__()
        self.nsgt = nsgt

    def _apply(self, fn):
        self.nsgt._apply(fn)
        return self

    def forward(self, x: Tensor) -> Tensor:
        """NSGT forward path
        Args:
            x (Tensor): audio waveform of
                shape (nb_samples, nb_channels, nb_timesteps)
        Returns:
            NSGT (Tensor): complex nsgt of
                shape (nb_samples, nb_channels, nb_bins_1, nb_bins_2, nb_frames, complex=2)
                last axis is stacked real and imaginary
        """
        shape = x.size()
        nb_samples, nb_channels, nb_timesteps = shape

        # pack batch
        x = x.view(-1, shape[-1])

        C = self.nsgt.nsgt.forward((x,))

        #S, I, F, T = C.shape
        # slice, channels, frequency bins, time bins

        # first, moveaxis S, I, F, T to I, F, S, T
        C = torch.moveaxis(C, 0, -2)

        nsgt_f = torch.view_as_real(C)

        # unpack batch
        nsgt_f = nsgt_f.view(shape[:-1] + nsgt_f.shape[-4:])

        return nsgt_f

    def plot_spectrogram(self, mls, ax):
        assert mls.shape[0] == 1
        # remove batch
        mls = torch.squeeze(mls, dim=0)
        # mix down multichannel
        mls = torch.mean(mls, dim=-1)
        fs_coef = self.nsgt.fs*self.nsgt.nsgt.coef_factor
        mls_dur = len(mls)/fs_coef # final duration of MLS
        mls_max = torch.quantile(mls, 0.999)

        mls = mls.detach().cpu().numpy()
        ax.imshow(mls.T, aspect=mls_dur/mls.shape[1]*0.2, interpolation='nearest', origin='lower', vmin=mls_max-60., vmax=mls_max, extent=(0,mls_dur,0,mls.shape[1]))


class INSGT_SL(nn.Module):
    '''
    wrapper for torch.istft to support batches
    Args:
         NSGT (Tensor): complex stft of
             shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
             last axis is stacked real and imaginary
        OR
             shape (nb_samples, nb_targets, nb_channels, nb_bins, nb_frames, complex=2)
             last axis is stacked real and imaginary
     '''
    def __init__(self, nsgt):
        super(INSGT_SL, self).__init__()
        self.nsgt = nsgt

    def _apply(self, fn):
        self.nsgt._apply(fn)
        return self

    def forward(self, X: Tensor, length: int) -> Tensor:
        Xshape = len(X.shape)

        X = torch.view_as_complex(X)

        shape = X.shape

        if Xshape == 6:
            X = X.view(X.shape[0]*X.shape[1], *X.shape[2:])
        else:
            X = X.view(X.shape[0]*X.shape[1]*X.shape[2], *X.shape[3:])

        # moveaxis back into into T x [packed-channels] x F1 x F2
        X = torch.moveaxis(X, -2, 0)

        y = self.nsgt.nsgt.backward(X, length)

        # unpack batch
        y = y.view(*shape[:-3], -1)

        return y


class ComplexNorm(nn.Module):
    r"""Compute the norm of complex tensor input.

    Extension of `torchaudio.functional.complex_norm` with mono

    Args:
        power (float): Power of the norm. (Default: `1.0`).
        mono (bool): Downmix to single channel after applying power norm
            to maximize
    """

    def __init__(self, power: float = 1.0, mono: bool = False):
        super(ComplexNorm, self).__init__()
        self.power = power
        self.mono = mono

    def forward(self, spec: Tensor) -> Tensor:
        """
        Args:
            spec: complex_tensor (Tensor): Tensor shape of
                `(..., complex=2)`

        Returns:
            Tensor: Power/Mag of input
                `(...,)`
        """
        # take the magnitude
        spec = torch.pow(torch.abs(torch.view_as_complex(spec)), self.power)

        # downmix in the mag domain to preserve energy
        if self.mono:
            spec = torch.mean(spec, 1, keepdim=True)

        return spec
