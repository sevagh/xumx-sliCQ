from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import warnings

from .nsgt import NSGT_sliced, BarkScale


def make_filterbanks(nsgt_base, sample_rate=44100.0):
    if sample_rate != 44100.0:
        raise ValueError("i was lazy and harcoded a lot of 44100.0, forgive me")

    encoder = NSGT_SL(nsgt_base)
    decoder = INSGT_SL(nsgt_base)

    return encoder, decoder


class NSGTBase(nn.Module):
    def __init__(self, scale, fbins, fmin, fmax=22050, fs=44100, device="cuda", gamma=25):
        super(NSGTBase, self).__init__()
        self.fbins = fbins
        self.fmin = fmin
        self.gamma = gamma
        self.fmax = fmax

        self.scl = BarkScale(self.fmin, self.fmax, self.fbins, device=device)

        self.sllen, self.trlen = self.scl.suggested_sllen_trlen(fs)
        print(f"sllen, trlen: {self.sllen}, {self.trlen}")

        self.nsgt = NSGT_sliced(
            self.scl,
            self.sllen,
            self.trlen,
            fs,
            real=True,
            multichannel=True,
            device=device,
        )
        self.M = self.nsgt.ncoefs
        self.fs = fs
        self.fbins_actual = self.nsgt.fbins_actual

    def max_bins(self, bandwidth): # convert hz bandwidth into bins
        if bandwidth is None or bandwidth < 0:
            return None
        freqs, _ = self.scl()
        max_bin = min(torch.argwhere(freqs > bandwidth))[0]
        return max_bin+1

    def predict_input_size(self, batch_size, nb_channels, seq_dur_s):
        fwd = NSGT_SL(self)

        x = torch.rand(
            (batch_size, nb_channels, int(seq_dur_s * self.fs)), dtype=torch.float32
        )
        shape = x.size()
        nb_samples, nb_channels, nb_timesteps = shape

        nsgt_f = fwd(x)
        return nsgt_f, x

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
        #nb_samples, nb_channels, nb_timesteps = shape

        # pack batch
        x = x.contiguous().view(-1, shape[-1])

        C = self.nsgt.nsgt.forward((x,))

        for i, nsgt_f in enumerate(C):
            nsgt_f = torch.moveaxis(nsgt_f, 0, -2)
            nsgt_f = torch.view_as_real(nsgt_f)
            # unpack batch
            nsgt_f = nsgt_f.view(shape[:-1] + nsgt_f.shape[-4:])
            C[i] = nsgt_f

        return C


class INSGT_SL(nn.Module):
    """
    wrapper for torch.istft to support batches
    Args:
         NSGT (Tensor): complex stft of
             shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
             last axis is stacked real and imaginary
        OR
             shape (nb_samples, nb_targets, nb_channels, nb_bins, nb_frames, complex=2)
             last axis is stacked real and imaginary
    """

    def __init__(self, nsgt):
        super(INSGT_SL, self).__init__()
        self.nsgt = nsgt

    def _apply(self, fn):
        self.nsgt._apply(fn)
        return self

    def forward(self, X_list, length: int) -> Tensor:
        X_complex = [None] * len(X_list)
        for i, X in enumerate(X_list):
            Xshape = len(X.shape)

            X = torch.view_as_complex(X)

            shape = X.shape

            if Xshape == 6:
                X = X.view(X.shape[0] * X.shape[1], *X.shape[2:])
            else:
                X = X.view(X.shape[0] * X.shape[1] * X.shape[2], *X.shape[3:])

            # moveaxis back into into T x [packed-channels] x F1 x F2
            X = torch.moveaxis(X, -2, 0)

            X_complex[i] = X

        y = self.nsgt.nsgt.backward(X_complex, length)

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

    def __init__(self):
        super(ComplexNorm, self).__init__()

    def forward(self, spec):
        if isinstance(spec, list):
            # take the magnitude of the ragged slicqt list
            ret = [None] * len(spec)

            for i, C_block in enumerate(spec):
                C_block = torch.pow(torch.abs(torch.view_as_complex(C_block)), 1.0)
                ret[i] = C_block

            return ret
        elif isinstance(spec, Tensor):
            return self.forward([spec])[0]
        else:
            raise ValueError(f"unsupported type for 'spec': {type(spec)}")
