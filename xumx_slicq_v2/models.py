from typing import Optional
from tqdm import trange
import torch
import torch.nn as nn
import norbert
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    Parameter,
    ReLU,
    BatchNorm2d,
    ConvTranspose2d,
    Conv2d,
    Sequential,
    Sigmoid,
)
from .transforms import (
    make_filterbanks,
    NSGTBase,
)
import copy


# inner class for doing umx for all targets for all slicqt blocks
class Unmix(nn.Module):
    def __init__(
        self,
        jagged_slicq_sample_input,
        input_means=None,
        input_scales=None,
    ):
        super(Unmix, self).__init__()

        self.sliced_umx = nn.ModuleList()

        freq_idx = 0
        for i, C_block in enumerate(jagged_slicq_sample_input):
            input_mean = input_means[i] if input_means else None
            input_scale = input_scales[i] if input_scales else None

            freq_start = freq_idx

            self.sliced_umx.append(
                _SlicedUnmix(
                    C_block,
                    input_mean=input_mean,
                    input_scale=input_scale,
                )
            )

            # advance frequency pointer
            freq_idx += C_block.shape[2]

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            # p.requires_grad = False
            p.grad = None
        self.eval()

    def forward(self, Xcomplex) -> Tensor:
        futures = [
            torch.jit.fork(self.sliced_umx[i], Xblock, torch.abs(torch.view_as_complex(Xblock)))
            for i, Xblock in enumerate(Xcomplex)
        ]
        Ycomplex = [torch.jit.wait(future) for future in futures]
        return Ycomplex


# inner class for doing umx for all targets per slicqt block
class _SlicedUnmix(nn.Module):
    def __init__(
        self,
        slicq_sample_input,
        input_mean=None,
        input_scale=None,
    ):
        hidden_size_1 = 25
        hidden_size_2 = 55

        super(_SlicedUnmix, self).__init__()

        (
            nb_samples,
            nb_channels,
            nb_f_bins,
            nb_slices,
            nb_t_bins,
        ) = slicq_sample_input.shape

        if nb_f_bins < 10:
            freq_filter = 1
        elif nb_f_bins < 20:
            freq_filter = 3
        else:
            freq_filter = 5

        encoder = []
        decoder = []

        window = nb_t_bins
        hop = window // 2

        encoder.extend([
            Conv2d(
                nb_channels,
                hidden_size_1,
                (freq_filter, window),
                stride=(1, hop),
                bias=False,
            ),
            BatchNorm2d(hidden_size_1),
            ReLU(),
        ])

        encoder.extend([
            Conv2d(
                hidden_size_1,
                hidden_size_2,
                (freq_filter, 3),
                bias=False,
            ),
            BatchNorm2d(hidden_size_2),
            ReLU(),
        ])

        decoder.extend([
            ConvTranspose2d(
                hidden_size_2,
                hidden_size_1,
                (freq_filter, 3),
                bias=False,
            ),
            BatchNorm2d(hidden_size_1),
            ReLU(),
        ])

        decoder.extend([
            ConvTranspose2d(
                hidden_size_1,
                nb_channels,
                (freq_filter, window),
                stride=(1, hop),
                bias=True,
            ),
            Sigmoid(),
        ])

        cdae_1 = Sequential(*encoder, *decoder)
        cdae_2 = copy.deepcopy(cdae_1)
        cdae_3 = copy.deepcopy(cdae_1)
        cdae_4 = copy.deepcopy(cdae_1)

        self.cdaes = nn.ModuleList([cdae_1, cdae_2, cdae_3, cdae_4])
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
            # p.requires_grad = False
            p.grad = None
        self.eval()

    def forward(self, xcomplex: Tensor, x: Tensor) -> Tensor:
        mix = x.detach().clone()

        x_shape = x.shape
        nb_samples, nb_channels, nb_f_bins, nb_slices, nb_t_bins = x_shape

        ret = torch.zeros((4, *x_shape,), device=x.device, dtype=x.dtype)

        x = torch.flatten(x, start_dim=-2, end_dim=-1)

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x.permute(0, 1, 3, 2)
        x += self.input_mean
        x *= self.input_scale
        x = x.permute(0, 1, 3, 2)

        for i, cdae in enumerate(self.cdaes):
            x_tmp = x.clone()
            for j, layer in enumerate(cdae):
                #print(f"{x_tmp.shape=}")
                x_tmp = layer(x_tmp)

            # wiener before reshaping to 3d??

            x_tmp = x_tmp.reshape(x_shape)

            # multiplicative skip connection
            if self.mask:
                x_tmp = x_tmp * mix

            ret[i, ...] = x_tmp

        # embedded blockwise wiener-EM (flattened in function then unflattened)
        ret = _wiener_fn(xcomplex, ret)
        return ret


def _wiener_fn(mix_slicqt, slicqtgrams, wiener_win_len_param: int = 5000):
    mix_slicqt = torch.flatten(mix_slicqt, start_dim=-3, end_dim=-2)
    orig_shape = slicqtgrams.shape
    slicqtgrams = torch.flatten(slicqtgrams, start_dim=-2, end_dim=-1)

    # transposing it as
    # (nb_samples, nb_frames, nb_bins,{1,nb_channels}, nb_sources)
    slicqtgrams = slicqtgrams.permute(1, 4, 3, 2, 0)

    # rearranging it into:
    # (nb_samples, nb_frames, nb_bins, nb_channels, 2) to feed
    # into filtering methods
    mix_slicqt = mix_slicqt.permute(0, 3, 2, 1, 4)

    nb_frames = slicqtgrams.shape[1]
    targets_slicqt = torch.zeros(
        *mix_slicqt.shape[:-1] + (4,2,),
        dtype=mix_slicqt.dtype,
        device=mix_slicqt.device,
    )

    pos = 0
    if wiener_win_len_param:
        wiener_win_len = wiener_win_len_param
    else:
        wiener_win_len = nb_frames
    while pos < nb_frames:
        cur_frame = torch.arange(pos, min(nb_frames, pos + wiener_win_len))
        pos = int(cur_frame[-1]) + 1

        targets_slicqt[:, cur_frame, ...] = torch.view_as_real(norbert.wiener(
            slicqtgrams[:, cur_frame, ...],
            torch.view_as_complex(mix_slicqt[:, cur_frame, ...]),
            1,
            False,
        ))

    # getting to (nb_samples, nb_targets, channel, fft_size, n_frames)
    targets_slicqt = targets_slicqt.permute(4, 0, 3, 2, 1, 5).contiguous()
    #print(f"targets_slicqt: {targets_slicqt.shape}")
    #print(f"orig shape: {orig_shape}")
    targets_slicqt = targets_slicqt.reshape((*orig_shape, 2,))
    #print(f"targets_slicqt: {targets_slicqt.shape}")
    return targets_slicqt