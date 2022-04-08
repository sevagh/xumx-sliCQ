from typing import Optional

from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, ReLU, Tanh, BatchNorm2d, ConvTranspose2d, Conv2d, Sequential, Sigmoid
from torch.nn import LSTM, BatchNorm1d, Linear, Parameter
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


class OpenUnmixTimeBucketBiLSTM(nn.Module):
    def __init__(
        self,
        slicq_sample_input,
        nb_layers=2,
        input_mean=None,
        input_scale=None,
    ):
        super(OpenUnmixTimeBucketBiLSTM, self).__init__()

        nb_samples, nb_channels, nb_f_bins, nb_slices, nb_t_bins = slicq_sample_input.shape

        self.nb_bins = nb_f_bins
        hidden_size = nb_f_bins*nb_channels
        self.hidden_size = hidden_size

        self.bn1 = BatchNorm1d(self.hidden_size)

        lstm_hidden_size = hidden_size // 2

        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=True,
            batch_first=False,
            dropout=0.4,
        )

        fc2_hiddensize = hidden_size * 2
        self.fc2 = Linear(in_features=fc2_hiddensize, out_features=hidden_size, bias=False)

        self.bn2 = BatchNorm1d(hidden_size)

        self.growth_layer = ConvTranspose2d(nb_channels, nb_channels, (1, 3), stride=(1, 2), bias=True)
        self.growth_act = Sigmoid()

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
        nb_t_frames_deola = nb_slices*nb_t_bins

        x = overlap_add_slicq(x)

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = x.reshape(-1, nb_channels * self.nb_bins)
        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(-1, nb_samples, self.hidden_size)
        # squash range ot [-1, 1]
        x = torch.tanh(x)

        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x)

        # lstm skip connection
        x = torch.cat([x, lstm_out[0]], -1)

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)
        x = F.relu(x)

        # reshape to grow
        x = x.reshape(nb_samples, nb_channels, self.nb_bins, -1)

        # growth layer
        x = self.growth_layer(x)
        x = self.growth_act(x)

        # crop
        x = x[:, :, :, : nb_t_frames_deola]

        # reshape back to original dim
        x = x.reshape(x_shape)

        # since our output is non-negative, we can apply RELU
        x = x * mix
        # permute back to (nb_samples, nb_channels, nb_bins, nb_frames)
        return x


class OpenUnmixTimeBucketCDAE(nn.Module):
    def __init__(
        self,
        slicq_sample_input,
        input_mean=None,
        input_scale=None,
    ):
        super(OpenUnmixTimeBucketCDAE, self).__init__()

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
        umx_bilstm=False,
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
                if not umx_bilstm:
                    bucketed_unmixes.append(
                        OpenUnmixTimeBucketCDAE(
                            C_block,
                            input_mean=input_mean,
                            input_scale=input_scale,
                        )
                    )
                else:
                    bucketed_unmixes.append(
                        OpenUnmixTimeBucketBiLSTM(
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

    def forward(self, x, vocals=True, bass=True, drums=True, other=True) -> Tensor:
        if vocals:
            futures_vocals = [torch.jit.fork(self.bucketed_unmixes_vocals[i], Xmag_block) for i, Xmag_block in enumerate(x)]
        if bass:
            futures_bass = [torch.jit.fork(self.bucketed_unmixes_bass[i], Xmag_block) for i, Xmag_block in enumerate(x)]
        if drums:
            futures_drums = [torch.jit.fork(self.bucketed_unmixes_drums[i], Xmag_block) for i, Xmag_block in enumerate(x)]
        if other:
            futures_other = [torch.jit.fork(self.bucketed_unmixes_other[i], Xmag_block) for i, Xmag_block in enumerate(x)]

        y_vocals = [torch.jit.wait(future) for future in futures_vocals] if vocals else None
        y_bass = [torch.jit.wait(future) for future in futures_bass] if bass else None
        y_drums = [torch.jit.wait(future) for future in futures_drums] if drums else None
        y_other = [torch.jit.wait(future) for future in futures_other] if other else None

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
        stft_wiener: bool = True,
        softmask: bool = False,
        wiener_win_len: Optional[int] = 300,
        chunk_size: Optional[int] = 2621440,
        n_fft: Optional[int] = 4096,
        n_hop: Optional[int] = 1024,
    ):
        super(Separator, self).__init__()
        self.stft_wiener = stft_wiener

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

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.wiener_win_len = wiener_win_len
        self.chunk_size = chunk_size if chunk_size is not None else sys.maxsize

        if not self.stft_wiener:
            # first, get frequency and time limits to build the large zero-padded matrix
            total_f_bins = 0
            max_t_bins = 0
            for i, block in enumerate(jagged_slicq_sample_input):
                nb_samples, nb_channels, nb_f_bins, nb_slices, nb_t_bins = block.shape
                total_f_bins += nb_f_bins
                max_t_bins = max(max_t_bins, nb_t_bins)

            self.total_f_bins = total_f_bins
            self.max_t_bins = max_t_bins

        self.ordered_targets = ["vocals", "drums", "bass", "other"]

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            #p.requires_grad = False
            p.grad = None
        self.xumx_model.freeze()
        self.eval()

    @torch.no_grad()
    def forward(self, audio_big: Tensor) -> Tensor:
        """Performing the separation on audio input

        Args:
            audio (Tensor): [shape=(nb_samples, nb_channels, nb_timesteps)]
                mixture audio waveform

        Returns:
            Tensor: stacked tensor of separated waveforms
                shape `(nb_samples, nb_targets, nb_channels, nb_timesteps)`
        """
        nb_sources = 4
        nb_samples = audio_big.shape[0]
        N = audio_big.shape[-1]

        nchunks = (N // self.chunk_size)
        if (N % self.chunk_size) != 0:
            nchunks += 1

        print(f'n chunks: {nchunks}')

        final_estimates = []

        for chunk_idx in trange(nchunks):
            audio = audio_big[..., chunk_idx * self.chunk_size: min((chunk_idx + 1) * self.chunk_size, N)]
            print(f'audio.shape: {audio.shape}')

            X = self.nsgt(audio)
            Xmag = self.complexnorm(X)

            # xumx inference - magnitude slicq estimate
            Ymag_bass, Ymag_vocals, Ymag_other, Ymag_drums = self.xumx_model(Xmag)

            if self.stft_wiener:
                print('STFT WIENER')

                # initial mix phase + magnitude estimate
                Ycomplex_bass = phasemix_sep(X, Ymag_bass)
                Ycomplex_vocals = phasemix_sep(X, Ymag_vocals)
                Ycomplex_drums = phasemix_sep(X, Ymag_drums)
                Ycomplex_other = phasemix_sep(X, Ymag_other)

                y_bass = self.insgt(Ycomplex_bass, audio.shape[-1])
                y_drums = self.insgt(Ycomplex_drums, audio.shape[-1])
                y_other = self.insgt(Ycomplex_other, audio.shape[-1])
                y_vocals = self.insgt(Ycomplex_vocals, audio.shape[-1])

                # initial estimate was obtained with slicq
                # now we switch to the STFT domain for the wiener step

                audio = torch.squeeze(audio, dim=0)
                
                mix_stft = torch.view_as_real(torch.stft(audio, self.n_fft, hop_length=self.n_hop, return_complex=True))
                X = torch.abs(torch.view_as_complex(mix_stft))
                
                # initializing spectrograms variable
                spectrograms = torch.zeros(X.shape + (nb_sources,), dtype=audio.dtype, device=X.device)

                for j, target_name in enumerate(self.ordered_targets):
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
                        slicq=False, # stft wiener
                    )

                # getting to (nb_samples, nb_targets, channel, fft_size, n_frames, 2)
                targets_stft = torch.view_as_complex(targets_stft.permute(4, 2, 1, 0, 3).contiguous())

                # inverse STFT
                estimates = torch.empty(audio.shape + (nb_sources,), dtype=audio.dtype, device=audio.device)

                for j, target_name in enumerate(self.ordered_targets):
                    estimates[..., j] = torch.istft(targets_stft[j, ...], self.n_fft, hop_length=self.n_hop, length=audio.shape[-1])

                estimates = torch.unsqueeze(estimates, dim=0).permute(0, 3, 1, 2).contiguous()
            else:
                print('sliCQT WIENER')

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
                    spectrograms[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, 1] = Ymag_drums[i]
                    spectrograms[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, 2] = Ymag_bass[i]
                    spectrograms[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, 3] = Ymag_other[i]

                    freq_start += nb_f_bins

                spectrograms = wiener(
                    torch.squeeze(spectrograms, dim=0),
                    torch.squeeze(X_matrix, dim=0),
                    self.niter,
                    softmask=self.softmask,
                    slicq=True,
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
            final_estimates.append(estimates)

        ests_concat = torch.cat(final_estimates, axis=-1)
        print(f'ests concat: {ests_concat.shape}')
        return ests_concat

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
