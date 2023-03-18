from typing import Optional
from tqdm import trange
import torch
import torch.nn as nn
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
    Tanh,
    LSTM,
    Linear,
    BatchNorm1d,
)
from .transforms import (
    make_filterbanks,
    NSGTBase,
)
from .phase import blockwise_wiener, blockwise_phasemix_sep, abs_of_real_complex
import copy


# inner class for doing umx for all targets for all slicqt blocks
class Unmix(nn.Module):
    def __init__(
        self,
        jagged_slicq_sample_input,
        realtime: bool = False,
        lstm: bool = False,
        input_means=None,
        input_scales=None,
    ):
        super(Unmix, self).__init__()

        self.sliced_umx = nn.ModuleList()

        sliced_umx_module = _SlicedUnmixCDAE if not lstm else _SlicedUnmixLSTM

        freq_idx = 0
        for i, C_block in enumerate(jagged_slicq_sample_input):
            input_mean = input_means[i] if input_means else None
            input_scale = input_scales[i] if input_scales else None

            self.sliced_umx.append(
                sliced_umx_module(
                    C_block,
                    realtime=realtime,
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

    def forward(self, Xcomplex, return_masks=False) -> Tensor:
        Ycomplex = [None] * len(Xcomplex)
        Ymasks = [None] * len(Xcomplex)

        for i, Xblock in enumerate(Xcomplex):
            Ycomplex_block, Ymask_block = self.sliced_umx[i](
                Xblock, abs_of_real_complex(Xblock)
            )
            Ycomplex[i] = Ycomplex_block
            Ymasks[i] = Ymask_block

        if return_masks:
            return Ycomplex, Ymasks
        return Ycomplex


# inner class for doing umx for all targets per slicqt block
class _SlicedUnmixCDAE(nn.Module):
    def __init__(
        self,
        slicq_sample_input,
        hidden_size_1: int = 50,
        hidden_size_2: int = 51,
        freq_filter_small: int = 1,
        freq_filter_medium: int = 3,
        freq_filter_large: int = 5,
        freq_thresh_small: int = 10,
        freq_thresh_medium: int = 20,
        time_filter_2: int = 4,
        realtime: bool = False,
        input_mean=None,
        input_scale=None,
    ):
        super(_SlicedUnmixCDAE, self).__init__()

        (
            nb_samples,
            nb_channels,
            nb_f_bins,
            nb_slices,
            nb_t_bins,
        ) = slicq_sample_input.shape

        if nb_f_bins < freq_thresh_small:
            freq_filter = freq_filter_small
        elif nb_f_bins < freq_thresh_medium:
            freq_filter = freq_filter_medium
        else:
            freq_filter = freq_filter_large

        encoder = []
        decoder = []

        window = nb_t_bins
        hop = window // 2

        if realtime:
            first_conv_module = _CausalConv2d
        else:
            first_conv_module = Conv2d

        encoder.extend(
            [
                first_conv_module(
                    nb_channels,
                    hidden_size_1,
                    (freq_filter, window),
                    stride=(1, hop),
                    bias=False,
                ),
                BatchNorm2d(hidden_size_1),
                ReLU(),
            ]
        )

        encoder.extend(
            [
                Conv2d(
                    hidden_size_1,
                    hidden_size_2,
                    (freq_filter, time_filter_2),
                    bias=False,
                ),
                BatchNorm2d(hidden_size_2),
                ReLU(),
            ]
        )

        decoder.extend(
            [
                ConvTranspose2d(
                    hidden_size_2,
                    hidden_size_1,
                    (freq_filter, time_filter_2),
                    bias=False,
                ),
                BatchNorm2d(hidden_size_1),
                ReLU(),
            ]
        )

        decoder.extend(
            [
                ConvTranspose2d(
                    hidden_size_1,
                    nb_channels,
                    (freq_filter, window),
                    stride=(1, hop),
                    bias=True,
                ),
                Sigmoid(),
            ]
        )

        cdae_1 = Sequential(*encoder, *decoder)
        cdae_2 = copy.deepcopy(cdae_1)
        cdae_3 = copy.deepcopy(cdae_1)
        cdae_4 = copy.deepcopy(cdae_1)

        self.cdaes = nn.ModuleList([cdae_1, cdae_2, cdae_3, cdae_4])
        self.mask = True
        self.realtime = realtime

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

        ret = torch.zeros(
            (
                4,
                *x_shape,
            ),
            device=x.device,
            dtype=x.dtype,
        )
        ret_masks = torch.zeros(
            (
                4,
                *x_shape,
            ),
            device=x.device,
            dtype=x.dtype,
        )

        x = torch.flatten(x, start_dim=-2, end_dim=-1)

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x.permute(0, 1, 3, 2)
        x += self.input_mean
        x *= self.input_scale
        x = x.permute(0, 1, 3, 2)

        for i, cdae in enumerate(self.cdaes):
            x_tmp = x.clone()
            for j, layer in enumerate(cdae):
                # print(f"{x_tmp.shape=}")
                x_tmp = layer(x_tmp)

            # crop if necessary
            x_tmp = x_tmp[..., :nb_f_bins, : nb_slices * nb_t_bins]
            x_tmp = x_tmp.reshape(x_shape)

            # store the sigmoid/soft mask before multiplying with mix
            ret_masks[i] = x_tmp.clone()

            # multiplicative skip connection
            if self.mask:
                x_tmp = x_tmp * mix

            ret[i] = x_tmp

        # use phasemix for realtime model
        if self.realtime:
            ret = blockwise_phasemix_sep(xcomplex, ret)
        else:
            # embedded blockwise wiener-EM (flattened in function then unflattened)
            ret = blockwise_wiener(xcomplex, ret)

        # also return the mask
        return ret, ret_masks


class _CausalConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        self.__padding = kernel_size[1] - 1

        super(_CausalConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        x = F.pad(x, (self.__padding, 0))
        result = super(_CausalConv2d, self).forward(x)
        return result


# inner class for doing umx for all targets per slicqt block

# TODO: support cross-target averaging like x-umx
# https://github.com/sony/ai-research-code/blob/master/x-umx/model.py#L247
class _SlicedUnmixLSTM(nn.Module):
    def __init__(
        self,
        slicq_sample_input,
        nb_layers=3,
        realtime: bool = False,
        input_mean=None,
        input_scale=None,
    ):
        super(_SlicedUnmixLSTM, self).__init__()

        (
            nb_samples,
            nb_channels,
            nb_f_bins,
            nb_slices,
            nb_t_bins,
        ) = slicq_sample_input.shape

        # downsample by half if there are at least 10 frequency bins
        if nb_f_bins > 10:
            self.hidden_size_1 = nb_f_bins*nb_channels//2

            layer1 = Sequential(
                Linear(in_features=nb_f_bins*nb_channels, out_features=self.hidden_size_1, bias=False),
                BatchNorm1d(self.hidden_size_1),
                Tanh(),
            )
        else:
            # else no downsampling
            self.hidden_size_1 = nb_f_bins*nb_channels

            # do nothing
            layer1 = nn.Identity()

        if realtime:
            lstm_hidden_size = self.hidden_size_1
        else:
            lstm_hidden_size = self.hidden_size_1 // 2 + (self.hidden_size_1 % 2)

        self.odd_lstm = (self.hidden_size_1 % 2 != 0)

        lstm = LSTM(
            input_size=self.hidden_size_1,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not realtime,
            batch_first=False,
            dropout=0.4 if nb_layers > 1 else 0,
        )

        fc2_hidden_size = self.hidden_size_1 * 2

        layer2 = Sequential(
            Linear(in_features=fc2_hidden_size, out_features=self.hidden_size_1, bias=False),
            BatchNorm1d(self.hidden_size_1),
            ReLU()
        )

        layer3 = Sequential(
            Linear(in_features=self.hidden_size_1, out_features=nb_f_bins*nb_channels, bias=True),
            Sigmoid()
        )
        self.mask = True
        self.realtime = realtime

        self.layer1s = nn.ModuleList([layer1, copy.deepcopy(layer1), copy.deepcopy(layer1), copy.deepcopy(layer1)])
        self.lstms = nn.ModuleList([lstm, copy.deepcopy(lstm), copy.deepcopy(lstm), copy.deepcopy(lstm)])
        self.layer2s = nn.ModuleList([layer2, copy.deepcopy(layer2), copy.deepcopy(layer2), copy.deepcopy(layer2)])
        self.layer3s = nn.ModuleList([layer3, copy.deepcopy(layer3), copy.deepcopy(layer3), copy.deepcopy(layer3)])

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

        ret = torch.zeros(
            (
                4,
                *x_shape,
            ),
            device=x.device,
            dtype=x.dtype,
        )
        ret_masks = torch.zeros(
            (
                4,
                *x_shape,
            ),
            device=x.device,
            dtype=x.dtype,
        )

        x = torch.flatten(x, start_dim=-2, end_dim=-1)
        nb_frames = x.shape[-1]
        nb_samples = x.shape[0]

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x.permute(0, 1, 3, 2)
        x += self.input_mean
        x *= self.input_scale
        x = x.permute(0, 1, 3, 2)

        xs = [x.clone() for _ in range(4)]

        # reshape before layer 1
        xs = [x.reshape(-1, nb_f_bins*nb_channels) for x in xs]

        lstm_crop = xs[0].shape[-1]

        # first apply layer 1
        xs = [layer1(x) for (x, layer1) in zip(xs, self.layer1s)]

        # reshape before bilstm
        xs = [x.reshape(nb_frames, nb_samples, self.hidden_size_1) for x in xs]

        # apply lstm
        lstm_outs = [lstm(x) for (x, lstm) in zip(xs, self.lstms)]

        # lstm skip conn
        xs = [torch.cat([x, lstm_out[0]], -1) for (x, lstm_out) in zip(xs, lstm_outs)]

        # reshape before layer 2
        xs = [x.reshape(-1, x.shape[-1]) for x in xs]

        if self.odd_lstm:
            xs = [x[..., : lstm_crop] for x in xs]

        # apply layer 2
        xs = [layer2(x) for (x, layer2) in zip(xs, self.layer2s)]

        # apply layer 3
        xs = [layer3(x) for (x, layer3) in zip(xs, self.layer3s)]

        # reshape before multiplying with original mix
        xs = [x.reshape(x_shape) for x in xs]

        # store the sigmoid/soft mask before multiplying with mix
        for i in range(4):
            ret_masks[i] = xs[i].clone()

            # multiplicative skip connection
            if self.mask:
                ret[i] = ret_masks[i] * mix

        # use phasemix for realtime model
        if self.realtime:
            ret = blockwise_phasemix_sep(xcomplex, ret)
        else:
            # embedded blockwise wiener-EM (flattened in function then unflattened)
            ret = blockwise_wiener(xcomplex, ret)

        # also return the mask
        return ret, ret_masks
