import sys
import gc
import os
import musdb
import itertools
import torch
from collections import defaultdict
import museval
from functools import partial
import numpy as np
import argparse
from openunmix.transforms import make_filterbanks, NSGTBase, phasemix_sep, ComplexNorm
from tqdm import tqdm

import scipy
from scipy.signal import stft, istft

eps = 1.e-10


def _fast_sdr(track, estimates_dct, device):
    references = torch.cat([torch.unsqueeze(torch.tensor(source.audio.T, device=device), dim=0) for source in track.sources.values()])
    estimates = torch.cat([torch.unsqueeze(est, dim=0) for est_name, est in estimates_dct.items() if est_name != 'accompaniment'])

    # compute SDR for one song
    num = torch.sum(torch.square(references), dim=(1, 2)) + eps
    den = torch.sum(torch.square(references - estimates), dim=(1, 2)) + eps
    sdr_target = 10.0 * torch.log10(num / den)
    return sdr_target


def stft_fwd(audio):
    return torch.stft(audio, n_fft=4096, hop_length=1024, return_complex=True).type(torch.complex64)


def stft_bwd(X, N):
    return torch.istft(X, n_fft=4096, hop_length=1024, length=N)


def ideal_mixphase_stft(track, device):
    """
    ideal performance of magnitude from estimated source + phase of mix
    which is the default umx strategy for separation
    """
    N = track.audio.shape[0]
    audio = torch.tensor(track.audio.T, device=device)

    # unsqueeze to add (1,) batch dimension
    X = stft_fwd(audio)

    # Compute sources spectrograms
    P = {}
    # compute model as the sum of spectrograms
    model = eps

    # parallelize this
    for name, source in track.sources.items():
        # compute spectrogram of target source:
        # magnitude of STFT
        src_coef = torch.view_as_real(stft_fwd(torch.tensor(source.audio.T, device=device)))
 
        P[name] = torch.abs(torch.view_as_complex(src_coef))

        # store the original, not magnitude, in the mix
        model += src_coef

    # now performs separation
    estimates = {}
    for name, source in track.sources.items():
        source_mag = P[name]

        Yj = phasemix_sep(model, source_mag)

        # invert to time domain
        target_estimate = stft_bwd(torch.view_as_complex(Yj), N)

        # set this as the source estimate
        estimates[name] = target_estimate

    return estimates


if __name__ == '__main__':
    mus = musdb.DB(subsets='train', split='valid', is_wav=True)

    sdr_tot_control = []
    sdr_bass_control = []
    sdr_drums_control = []
    sdr_other_control = []
    sdr_vocals_control = []

    # set degenerate conditions/frequency scales
    # 1. sllen too high (> max sllen)
    # 2. unordered fbins/jagged
    device = "cpu"

    for track in tqdm(mus.tracks):
        N = track.audio.shape[0]
        #print(f'evaluating {track}, {(N/44100):.2f} seconds long')
        ests_stft = ideal_mixphase_stft(track, device=device)
        sdr_stem_control = _fast_sdr(track, ests_stft, device)
        sdr_tot_control.append(torch.mean(sdr_stem_control))

        for i, est_name in enumerate(ests_stft.keys()):
            if est_name == 'bass':
                sdr_bass_control.append(sdr_stem_control[..., i])
            elif est_name == 'drums':
                sdr_drums_control.append(sdr_stem_control[..., i])
            elif est_name == 'other':
                sdr_other_control.append(sdr_stem_control[..., i])
            elif est_name == 'vocals':
                sdr_vocals_control.append(sdr_stem_control[..., i])

        gc.collect()
        del ests_stft
        torch.cuda.empty_cache()

    control_sdr = torch.mean(torch.cat([torch.unsqueeze(control_sdr, dim=0) for control_sdr in sdr_tot_control]))
    control_sdr_bass = torch.mean(torch.cat([torch.unsqueeze(control_sdr, dim=0) for control_sdr in sdr_bass_control]))
    control_sdr_drums = torch.mean(torch.cat([torch.unsqueeze(control_sdr, dim=0) for control_sdr in sdr_drums_control]))
    control_sdr_other = torch.mean(torch.cat([torch.unsqueeze(control_sdr, dim=0) for control_sdr in sdr_other_control]))
    control_sdr_vocals = torch.mean(torch.cat([torch.unsqueeze(control_sdr, dim=0) for control_sdr in sdr_vocals_control]))

    print(f'Control score tot, bass, drums, vocals, other:\n\t{control_sdr:.2f}\t{control_sdr_bass:.2f}\t{control_sdr_drums:.2f}\t{control_sdr_vocals:.2f}\t{control_sdr_other:.2f}')
