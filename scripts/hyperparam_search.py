import sys
import os
import musdb
import itertools
import torch
import museval
from functools import partial
import numpy as np
import random
import argparse
from openunmix.nsgt import NSGT_sliced, MelScale, LogScale, BarkScale, VQLogScale
from tqdm import tqdm

import scipy
from scipy.signal import stft, istft

import json
from types import SimpleNamespace

eps = 1.e-10


'''
from https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021
    nb_sources, nb_samples, nb_channels = 4, 100000, 2
    references = np.random.rand(nb_sources, nb_samples, nb_channels)
    estimates = np.random.rand(nb_sources, nb_samples, nb_channels)
'''
def fast_sdr(track, estimates_dct, target, device):
    references = torch.cat([torch.unsqueeze(torch.tensor(source.audio, device=device), dim=0) for name, source in track.sources.items() if name == target])
    estimates = torch.cat([torch.unsqueeze(est, dim=0) for est_name, est in estimates_dct.items() if est_name == target])

    # compute SDR for one song
    num = torch.sum(torch.square(references), dim=(1, 2)) + eps
    den = torch.sum(torch.square(references - estimates), dim=(1, 2)) + eps
    sdr_instr = 10.0 * torch.log10(num / den)
    sdr_song = torch.mean(sdr_instr)
    return sdr_song


def atan2(y, x):
    r"""Element-wise arctangent function of y/x.
    Returns a new tensor with signed angles in radians.
    It is an alternative implementation of torch.atan2

    Args:
        y (Tensor): First input tensor
        x (Tensor): Second input tensor [shape=y.shape]

    Returns:
        Tensor: [shape=y.shape].
    """
    pi = 2 * torch.asin(torch.tensor(1.0))
    x += ((x == 0) & (y == 0)) * 1.0
    out = torch.atan(y / x)
    out += ((y >= 0) & (x < 0)) * pi
    out -= ((y < 0) & (x < 0)) * pi
    out *= 1 - ((y > 0) & (x == 0)) * 1.0
    out += ((y > 0) & (x == 0)) * (pi / 2)
    out *= 1 - ((y < 0) & (x == 0)) * 1.0
    out += ((y < 0) & (x == 0)) * (-pi / 2)
    return out


def phasemix_sep(X, Ymag):
    Xphase = atan2(X[..., 1], X[..., 0])
    Ycomplex = torch.empty_like(X)

    Ycomplex[..., 0] = Ymag * torch.cos(Xphase)
    Ycomplex[..., 1] = Ymag * torch.sin(Xphase)
    return Ycomplex


def ideal_mixphase(track, tf):
    """
    ideal performance of magnitude from estimated source + phase of mix
    which is the default umx strategy for separation
    """
    N = track.audio.shape[0]

    X = tf.forward(track.audio)
    time_coefs = X.shape[-1]
    #print(f'time coefs: {time_coefs}')

    #(I, F, T) = X.shape

    # Compute sources spectrograms
    P = {}
    # compute model as the sum of spectrograms
    model = eps

    # parallelize this
    for name, source in track.sources.items():
        # compute spectrogram of target source:
        # magnitude of STFT
        src_coef = tf.forward(source.audio)

        P[name] = torch.abs(src_coef)

        # store the original, not magnitude, in the mix
        model += src_coef

    # now performs separation
    estimates = {}
    for name, source in track.sources.items():
        source_mag = P[name]

        #print('inverting phase')
        Yj = torch.view_as_complex(phasemix_sep(torch.view_as_real(model), source_mag))

        # invert to time domain
        target_estimate = tf.backward(Yj, N)

        # set this as the source estimate
        estimates[name] = target_estimate

    return estimates, time_coefs


class TFTransform:
    def __init__(self, fs, fscale="bark", fmin=78.0, fbins=125, fgamma=25.0, sllen=None, device="cuda"):
        self.fbins = fbins
        self.nsgt = None
        self.device = device

        scl = None
        if fscale == 'mel':
            scl = MelScale(fmin, fs/2, fbins)
        elif fscale == 'bark':
            scl = BarkScale(fmin, fs/2, fbins)
        elif fscale == 'cqlog':
            scl = LogScale(fmin, fs/2, fbins)
        elif fscale == 'vqlog':
            scl = VQLogScale(fmin, fs/2, fbins, gamma=fgamma)
        else:
            raise ValueError(f"unsupported scale {fscale}")

        if sllen is None:
            # use slice length required to support desired frequency scale/q factors
            sllen = scl.suggested_sllen(fs)

        self.sllen = sllen
        trlen = sllen//4
        trlen = trlen + -trlen % 2 # make trlen divisible by 2
        self.trlen = trlen

        self.nsgt = NSGT_sliced(scl, sllen, trlen, fs, real=True, matrixform=True, multichannel=True, device=self.device)
        self.name = f'n{fscale}-{fbins}-{fmin:.2f}-{sllen}'

    def forward(self, audio):
        audio = torch.tensor(audio.T, device=self.device)
        audio = torch.unsqueeze(audio, 0)
        C = self.nsgt.forward(audio)
        return C

    def backward(self, X, len_x):
        c = self.nsgt.backward(X, len_x).T
        c = torch.squeeze(c, 0)
        return c

    def printinfo(self):
        print('nsgt params:\n\t{0}\n\t{1} f bins, {2} m bins\n\t{3} total dim'.format(self.name, self.fbins, self.nsgt.ncoefs, self.fbins*self.nsgt.ncoefs))


class TrackEvaluator:
    def __init__(self, tracks, seq_dur_min, seq_dur_max, max_sllen, device="cuda"):
        self.tracks = tracks
        self.seq_dur_min = seq_dur_min
        self.seq_dur_max = seq_dur_max
        self.max_sllen = max_sllen
        self.device = device

    def oracle(self, scale='cqlog', fmin=20.0, bins=12, gamma=25, sllen=None, reps=1, printinfo=False, divide_by_dim=False):
        bins = int(bins)

        med_sdrs_bass = []
        med_sdrs_drums = []
        med_sdrs_vocals = []
        med_sdrs_other = []

        tf = TFTransform(44100, scale, fmin, bins, gamma, sllen=sllen, device=self.device)

        if printinfo:
            tf.printinfo()

        # skip too big transforms
        if tf.sllen > self.max_sllen:
            return (
                float('-inf'),
                float('-inf'),
                float('-inf'),
                float('-inf'),
                tf.sllen
            )

        for _ in range(reps):
            # repeat for reps x duration
            for track in self.tracks:
                seq_dur = np.random.uniform(self.seq_dur_min, self.seq_dur_max)
                track.chunk_duration = seq_dur
                track.chunk_start = random.uniform(0, track.duration - seq_dur)

                #print(f'track:\n\t{track.name}\n\t{track.chunk_duration}\n\t{track.chunk_start}')

                N = track.audio.shape[0]
                ests, dim = ideal_mixphase(track, tf)

                if not divide_by_dim:
                    dim = 1.0

                med_sdrs_bass.append(fast_sdr(track, ests, target='bass', device=self.device)/dim)
                med_sdrs_drums.append(fast_sdr(track, ests, target='drums', device=self.device)/dim)
                med_sdrs_vocals.append(fast_sdr(track, ests, target='vocals', device=self.device)/dim)
                med_sdrs_other.append(fast_sdr(track, ests, target='other', device=self.device)/dim)

        # return 1 sdr per source
        return (
            torch.mean(torch.cat([torch.unsqueeze(med_sdr, dim=0) for med_sdr in med_sdrs_bass])),
            torch.mean(torch.cat([torch.unsqueeze(med_sdr, dim=0) for med_sdr in med_sdrs_drums])),
            torch.mean(torch.cat([torch.unsqueeze(med_sdr, dim=0) for med_sdr in med_sdrs_vocals])),
            torch.mean(torch.cat([torch.unsqueeze(med_sdr, dim=0) for med_sdr in med_sdrs_other])),
            tf.sllen
        )


def evaluate_single(f, params, seq_reps):
    #print(f'{scale} {bins} {fmin} {fmax} {gamma}')

    curr_score_bass, curr_score_drums, curr_score_vocals, curr_score_other, sllen = f(scale=params['scale'], fmin=params['fmin'], bins=params['bins'], gamma=params['gamma'], sllen=params['sllen'], reps=seq_reps, printinfo=True)

    print('bass, drums, vocals, other sdr! {0:.2f} {1:.2f} {2:.2f} {3:.2f}'.format(
        curr_score_bass,
        curr_score_drums,
        curr_score_vocals,
        curr_score_other,
    ))
    print('total sdr: {0:.2f}'.format((curr_score_bass+curr_score_drums+curr_score_vocals+curr_score_other)/4))


def optimize_many(f, params, n_iter, seq_reps, per_target):
    if per_target:
        best_score_bass = float('-inf')
        best_param_bass = None

        best_score_drums = float('-inf')
        best_param_drums = None

        best_score_vocals = float('-inf')
        best_param_vocals = None

        best_score_other = float('-inf')
        best_param_other = None

        fmins = list(np.arange(*params['fmin']))
        gammas = list(np.arange(*params['gamma']))

        #print(f'optimizing target {target_name}')
        for _ in tqdm(range(n_iter)):
            while True: # loop in case we skip for exceeding sllen
                scale = random.choice(params['scales'])
                bins = np.random.randint(*params['bins'])
                fmin = random.choice(fmins)
                gamma = random.choice(gammas)
                
                curr_score_bass, curr_score_drums, curr_score_vocals, curr_score_other, sllen = f(scale=scale, fmin=fmin, bins=bins, gamma=gamma, reps=seq_reps)

                params_tup = (scale, bins, fmin, gamma, sllen)

                if curr_score_bass == curr_score_drums and curr_score_drums == curr_score_vocals and curr_score_vocals == curr_score_other and curr_score_other == float('-inf'):
                    # sllen not supported
                    print('reroll for sllen...')
                    continue

                if curr_score_bass > best_score_bass:
                    best_score_bass = curr_score_bass
                    best_param_bass = params_tup
                    print('good bass sdr! {0}, {1}'.format(best_score_bass, best_param_bass))
                if curr_score_drums > best_score_drums:
                    best_score_drums = curr_score_drums
                    best_param_drums = params_tup
                    print('good drums sdr! {0}, {1}'.format(best_score_drums, best_param_drums))
                if curr_score_vocals > best_score_vocals:
                    best_score_vocals = curr_score_vocals
                    best_param_vocals = params_tup
                    print('good vocals sdr! {0}, {1}'.format(best_score_vocals, best_param_vocals))
                if curr_score_other > best_score_other:
                    best_score_other = curr_score_other
                    best_param_other = params_tup
                    print('good other sdr! {0}, {1}'.format(best_score_other, best_param_other))
                break
        print(f'best scores')
        print(f'bass: \t{best_score_bass}\t{best_param_bass}')
        print(f'drums: \t{best_score_drums}\t{best_param_drums}')
        print(f'other: \t{best_score_other}\t{best_param_other}')
        print(f'vocals: \t{best_score_vocals}\t{best_param_vocals}')
    else:
        best_score_total = float('-inf')
        best_param_total = None

        fmins = list(np.arange(*params['fmin']))
        gammas = list(np.arange(*params['gamma']))

        for _ in tqdm(range(n_iter)):
            while True: # loop in case we skip for exceeding sllen
                scale = random.choice(params['scales'])
                bins = np.random.randint(*params['bins'])
                fmin = random.choice(fmins)
                gamma = random.choice(gammas)
                
                curr_score_bass, curr_score_drums, curr_score_vocals, curr_score_other, sllen = f(scale=scale, fmin=fmin, bins=bins, gamma=gamma, reps=seq_reps)
                tot = (curr_score_bass+curr_score_drums+curr_score_vocals+curr_score_other)/4

                params_tup = (scale, bins, fmin, gamma, sllen)

                if curr_score_bass == curr_score_drums and curr_score_drums == curr_score_vocals and curr_score_vocals == curr_score_other and curr_score_other == float('-inf'):
                    # sllen not supported
                    print('reroll for sllen...')
                    continue

                if tot > best_score_total:
                    best_score_total = tot
                    best_param_total = params_tup
                    print('good total sdr! {0}, {1}'.format(best_score_total, best_param_total))
                break
        print(f'best scores')
        print(f'total: \t{best_score_total}\t{best_param_total}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Search NSGT configs for best ideal mask'
    )
    parser.add_argument(
        '--bins',
        type=str,
        default='10,300',
        help='comma-separated range of bins to evaluate'
    )
    parser.add_argument(
        '--fmins',
        type=str,
        default='10,130,0.1',
        help='comma-separated range of fmin to evaluate'
    )
    parser.add_argument(
        '--gammas',
        type=str,
        default='0,100,0.1',
        help='comma-separated range of gamma to evaluate'
    )
    parser.add_argument(
        '--n-iter',
        type=int,
        default=1000,
        help='number of iterations'
    )
    parser.add_argument(
        '--fscale',
        type=str,
        default='bark,mel,cqlog,vqlog',
        help='nsgt frequency scales, csv (choices: vqlog, cqlog, mel, bark)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='torch device (cpu vs cuda)'
    )
    parser.add_argument(
        '--n-random-tracks',
        type=int,
        default=10,
        help='use N random tracks'
    )
    parser.add_argument(
        '--seq-dur-min',
        type=float,
        default=5.0,
        help='sequence duration per track, min'
    )
    parser.add_argument(
        '--seq-dur-max',
        type=float,
        default=10.0,
        help='sequence duration per track, max'
    )
    parser.add_argument(
        '--seq-reps',
        type=int,
        default=10,
        help='sequence repetitions (adds robustness)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='rng seed to pick the same random 5 songs'
    )
    parser.add_argument(
        '--max-sllen',
        type=int,
        default=32760,
        help='maximum sllen above which to skip iterations'
    )
    parser.add_argument(
        '--sllen',
        type=int,
        default=8192,
        help='sllen to use'
    )
    parser.add_argument(
        '--single',
        action='store_true',
        help='evaluate single nsgt instead of randomized param search'
    )
    parser.add_argument(
        '--per-target',
        action='store_true',
        help='maximize each target separately'
    )

    args = parser.parse_args()

    random.seed(args.random_seed)

    # initiate musdb
    mus = musdb.DB(subsets='train', split='valid', is_wav=True)

    print(f'using {args.n_random_tracks} random tracks from MUSDB18-HQ train set validation split')
    tracks = random.sample(mus.tracks, args.n_random_tracks)

    if not args.single:
        scales = args.fscale.split(',')
        bins = tuple([int(x) for x in args.bins.split(',')])
        fmins = tuple([float(x) for x in args.fmins.split(',')])
        gammas = tuple([float(x) for x in args.gammas.split(',')])
        print(f'Parameter ranges to evaluate:\n\tscales: {scales}\n\tbins: {bins}\n\tfmins: {fmins}\n\tgammas: {gammas}')
        print(f'Ignoring fscales that exceed sllen {args.max_sllen}')

        params = {
            'scales': scales,
            'bins': bins,
            'fmin': fmins,
            'gamma': gammas,
        }

        t = TrackEvaluator(tracks, args.seq_dur_min, args.seq_dur_max, args.max_sllen, device=args.device)
        optimize_many(t.oracle, params, args.n_iter, args.seq_reps, args.per_target)
    else:
        params = {
            'scale': args.fscale,
            'bins': int(args.bins),
            'fmin': float(args.fmins),
            'gamma': float(args.gammas),
            'sllen': int(args.sllen),
        }

        print(f'Parameter to evaluate:\n\t{params}')

        t = TrackEvaluator(tracks, args.seq_dur_min, args.seq_dur_max, args.max_sllen, device=args.device)
        evaluate_single(t.oracle, params, args.seq_reps)
