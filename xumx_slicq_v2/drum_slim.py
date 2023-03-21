from typing import Optional, Union, Tuple
from tqdm import trange
from pathlib import Path
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
)
from .transforms import (
    make_filterbanks,
    NSGTBase,
)
from .phase import blockwise_wiener, blockwise_phasemix_sep, abs_of_real_complex
import copy


class RealtimeDrumUnmix(nn.Module):
    def __init__(
        self,
        jagged_slicq_sample_input,
        input_means=None,
        input_scales=None,
    ):
        super(RealtimeDrumUnmix, self).__init__()

        self.sliced_umx = nn.ModuleList()

        freq_idx = 0
        for i, C_block in enumerate(jagged_slicq_sample_input):
            input_mean = input_means[i] if input_means else None
            input_scale = input_scales[i] if input_scales else None

            self.sliced_umx.append(
                _SlicedUnmixCDAEDrums(
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
        Ycomplex = [None] * len(Xcomplex)

        for i, Xblock in enumerate(Xcomplex):
            Ycomplex_block = self.sliced_umx[i](
                Xblock, abs_of_real_complex(Xblock)
            )
            Ycomplex[i] = Ycomplex_block

        return Ycomplex


# inner class for doing umx for only the drum target
class _SlicedUnmixCDAEDrums(nn.Module):
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
        input_mean=None,
        input_scale=None,
    ):
        super(_SlicedUnmixCDAEDrums, self).__init__()

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

        encoder.extend(
            [
                _CausalConv2d(
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

        self.cdae = Sequential(*encoder, *decoder)

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

        ret = torch.zeros_like(x)

        x = torch.flatten(x, start_dim=-2, end_dim=-1)

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x.permute(0, 1, 3, 2)
        x += self.input_mean
        x *= self.input_scale
        x = x.permute(0, 1, 3, 2)

        for i, layer in enumerate(self.cdae):
            x = layer(x)

        # crop if necessary
        x = x[..., :nb_f_bins, : nb_slices * nb_t_bins]
        x = x.reshape(x_shape)

        # multiplicative skip connection
        x = x * mix

        # use phasemix for realtime model
        ret = blockwise_phasemix_sep(xcomplex, x)

        # also return the mask
        return ret


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


class RealtimeDrumSeparator(nn.Module):
    def __init__(
        self,
        xumx_model: RealtimeDrumUnmix = None,
        encoder: Tuple = None,
        warmup: int = 0,
        sample_rate: float = 44100.0,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(RealtimeDrumSeparator, self).__init__()
        # saving parameters

        self.device = device
        self.nb_channels = 2
        self.register_buffer("sample_rate", torch.as_tensor(sample_rate))

        self.xumx_model = xumx_model
        self.nsgt, self.insgt = encoder

        if warmup > 0:
            print(f"Running {warmup} inference warmup reps")
            t = trange(warmup, desc="warmup")
            for _ in trange(warmup):
                # random 100-second waveform
                waveform = torch.rand(
                    (1, 2, int(100 * sample_rate)), dtype=torch.float32, device=device
                )
                separator.forward(waveform)

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            # p.requires_grad = False
            p.grad = None
        self.xumx_model.freeze()
        self.eval()

    def forward(self, audio: Tensor) -> Tensor:
        nb_samples = audio.shape[0]
        N = audio.shape[-1]

        final_estimates = []

        n_samples = audio.shape[-1]

        X = self.nsgt(audio)

        Ycomplex_drums = self.xumx_model(X)
        drums_out = self.insgt(Ycomplex_drums, n_samples)

        return drums_out


def load_realtime_drum_separator_onnx(model_path: Optional[Union[str, Path]]):
    model_path = "/xumx-sliCQ-V2/pretrained_model_realtime"
    model_path = Path(model_path).expanduser()

    assert model_path.exists()

    # load model from disk
    with open(Path(model_path, "xumx_slicq_v2.json"), "r") as stream:
        results = json.load(stream)

    onnx_model_path = Path(model_path, "xumx_slicq_v2_realtime_drum_separator.onnx")

    assert onnxruntime_available
    rt_drum_separator = onnx.load(onnx_model_path)
    onnx.checker.check_model(rt_drum_separator)

    available_providers = onnxruntime.get_available_providers()
    sess_options = onnxruntime.SessionOptions()

    provider = ["CPUExecutionProvider"]

    print(f"ONNXRuntime chosen provider: {provider}")

    ort_session = onnxruntime.InferenceSession(
        str(onnx_model_path), providers=provider, sess_options=sess_options
    )
    rt_drum_model = ort_session

    return rt_drum_model
