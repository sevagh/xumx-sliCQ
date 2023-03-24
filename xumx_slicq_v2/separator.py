from typing import Optional, Union, Tuple
import sys
from tqdm import trange, tqdm
from pathlib import Path
import requests
import torch
import numpy as np
import json
from torch import Tensor
import torch.nn as nn
from .model import Unmix
from .transforms import (
    ComplexNorm,
    NSGTBase,
    make_filterbanks,
)
import urllib.request

onnxruntime_available = False
try:
    import onnx
    import onnxruntime

    onnxruntime_available = True
except ModuleNotFoundError:
    pass

_SUPPORTED_RUNTIMES = ["torch-cpu", "torch-cuda", "onnx-cpu", "onnx-cuda"]


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    print(f"Downloading \"{url}\" to {output_path}")
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


class Separator(nn.Module):
    @classmethod
    def load(
        cls,
        chunk_size: int = 2621440,
        model_path: Optional[str] = None,
        runtime_backend: Optional[str] = "torch-cpu",
        warmup: int = 0,
        realtime: bool = False,
        device: Union[str, torch.device] = "cpu",
    ):
        if runtime_backend not in _SUPPORTED_RUNTIMES:
            raise ValueError(
                f"requested runtime backend {runtime_backend} not in {_SUPPORTED_RUNTIMES}"
            )

        xumx_model, encoder, sample_rate = load_target_models(
            model_path,
            runtime_backend=runtime_backend,
            realtime=realtime,
            device=device,
        )

        separator = cls(
            xumx_model=xumx_model,
            encoder=encoder,
            sample_rate=sample_rate,
            runtime_backend=runtime_backend,
            chunk_size=chunk_size,
        ).to(device)

        if runtime_backend.startswith("torch"):
            separator.freeze()

        if warmup > 0:
            print(f"Running {warmup} inference warmup reps")
            t = trange(warmup, desc="warmup")
            for _ in trange(warmup):
                # random 100-second waveform
                waveform = torch.rand(
                    (1, 2, int(100 * sample_rate)), dtype=torch.float32, device=device
                )
                separator.forward(waveform)

        return separator

    def __init__(
        self,
        xumx_model: Unmix = None,
        encoder: Tuple = None,
        runtime_backend: str = "torch",
        sample_rate: float = 44100.0,
        chunk_size: Optional[int] = 2621440,
        device: str = "cpu",
    ):
        super(Separator, self).__init__()
        # saving parameters

        # to be compatible with cadenza code
        # following the same order of data.py
        self.sources = ["bass", "vocals", "other", "drums"]

        self.device = device
        self.nb_channels = 2
        self.register_buffer("sample_rate", torch.as_tensor(sample_rate))
        self.chunk_size = chunk_size if chunk_size is not None else sys.maxsize

        self.xumx_model = xumx_model
        self.runtime_backend = runtime_backend

        if self.runtime_backend.startswith("onnx"):
            assert onnxruntime_available

        self.nsgt, self.insgt, self.cnorm = encoder

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            # p.requires_grad = False
            p.grad = None
        self.xumx_model.freeze()
        self.eval()

    def forward(self, audio_big: Tensor) -> Tensor:
        """Performing the separation on audio input

        Args:
            audio (Tensor): [shape=(nb_samples, nb_channels, nb_timesteps)]
                mixture audio waveform

        Returns:
            Tensor: stacked tensor of separated waveforms
                shape `(nb_samples, nb_targets, nb_channels, nb_timesteps)`
        """
        nb_samples = audio_big.shape[0]
        N = audio_big.shape[-1]

        nchunks = N // self.chunk_size
        if (N % self.chunk_size) != 0:
            nchunks += 1

        final_estimates = []

        t = trange(nchunks, desc="song chunks")
        for chunk_idx in t:
            audio = audio_big[
                ...,
                chunk_idx * self.chunk_size : min((chunk_idx + 1) * self.chunk_size, N),
            ]

            n_samples = audio.shape[-1]

            X = self.nsgt(audio)

            if self.runtime_backend.startswith("torch"):
                Ycomplex_all = self.xumx_model(X)
                estimates = self.insgt(Ycomplex_all, n_samples)
            elif self.runtime_backend.startswith("onnx"):
                if self.runtime_backend.endswith("cuda"):
                    # set up IOBinding for GPU transfers
                    io_binding = self.xumx_model.io_binding()

                    # Ycomplex_all
                    Ycomplex_all = [
                        torch.empty(
                            (
                                4,
                                *X[i].shape,
                            ),
                            device=X[i].device,
                            dtype=X[i].dtype,
                        ).contiguous()
                        for i in range(len(X))
                    ]

                    for i in range(len(X)):
                        io_binding.bind_input(
                            name=f"xcomplex{i}",
                            device_type="cuda",
                            device_id=0,
                            element_type=np.float32,
                            shape=tuple(X[i].shape),
                            buffer_ptr=X[i].data_ptr(),
                        )
                        io_binding.bind_output(
                            name=f"ycomplex{i}",
                            device_type="cuda",
                            device_id=0,
                            element_type=np.float32,
                            shape=tuple(Ycomplex_all[i].shape),
                            buffer_ptr=Ycomplex_all[i].data_ptr(),
                        )

                    self.xumx_model.run_with_iobinding(io_binding)
                    estimates = self.insgt(Ycomplex_all, n_samples)
                else:
                    Ycomplex_all = self.xumx_model.run(
                        [f"ycomplex{i}" for i in range(len(X))],
                        {
                            f"xcomplex{i}": X[i].detach().cpu().numpy()
                            for i in range(len(X))
                        },
                    )
                    estimates = self.insgt(
                        [
                            torch.as_tensor(Ycomplex_all_)
                            for Ycomplex_all_ in Ycomplex_all
                        ],
                        n_samples,
                    )

            final_estimates.append(estimates)

        ests_concat = torch.cat(final_estimates, axis=-1)
        return ests_concat

    @staticmethod
    def to_dict(estimates: Tensor, aggregate_dict: Optional[dict] = None) -> dict:
        """Convert estimates as stacked tensor to dictionary

        Args:
            estimates (Tensor): separated targets of shape
                (nb_samples, nb_targets, nb_channels, nb_timesteps)
            aggregate_dict (dict or None)

        Returns:
            (dict of str: Tensor):
        """
        estimates_dict = {}

        # follow the ordering in data.py
        for k, target in enumerate(self.sources):
            estimates_dict[target] = estimates[k]

        if aggregate_dict is not None:
            new_estimates = {}
            for key in aggregate_dict:
                new_estimates[key] = torch.tensor(0.0)
                for target in aggregate_dict[key]:
                    new_estimates[key] = new_estimates[key] + estimates_dict[target]
            estimates_dict = new_estimates
        return estimates_dict


def load_target_models(
    model_path: str,
    runtime_backend: str = "torch",
    realtime: bool = False,
    device="cpu",
):
    # force realtime for onnx
    if "torch" not in runtime_backend:
        realtime = True

    if model_path is not None:
        # manual model_path is specified, ensure it exists
        model_path = Path(model_path).expanduser()
        json_path = Path(model_path / "xumx_slicq_v2.json")
        assert model_path.exists() and json_path.exists()
    else:
        # load the pretrained model if it's in the expected docker path
        # otherwise, download it
        if not realtime:
            model_path = "/xumx-sliCQ-V2/pretrained_model"
        else:
            model_path = "/xumx-sliCQ-V2/pretrained_model_realtime"
        model_path = Path(model_path).expanduser()

    if model_path.exists():
        # load model from disk
        with open(Path(model_path, "xumx_slicq_v2.json"), "r") as stream:
            results = json.load(stream)

        if runtime_backend.startswith("torch"):
            target_model_path = Path(model_path, "xumx_slicq_v2.pth")
            state = torch.load(target_model_path, map_location=device)
        else:
            onnx_model_path = Path(model_path, "xumx_slicq_v2.onnx")
    else:
        # fetch config json and weights from github
        # use xumx-slicq-v1 urls for now while testing the code
        if not realtime:
            json_url = "https://github.com/sevagh/xumx-sliCQ-V2/raw/main/pretrained_model/xumx_slicq_v2.json"
            pth_url = "https://github.com/sevagh/xumx-sliCQ-V2/raw/main/pretrained_model/xumx_slicq_v2.pth"
        else:
            json_url = "https://github.com/sevagh/xumx-sliCQ-V2/raw/main/pretrained_model_realtime/xumx_slicq_v2.json"
            pth_url = "https://github.com/sevagh/xumx-sliCQ-V2/raw/main/pretrained_model_realtime/xumx_slicq_v2.pth"
            onnx_url = "https://github.com/sevagh/xumx-sliCQ-V2/raw/main/pretrained_model_realtime/xumx_slicq_v2.onnx"

        hub_dir = Path(torch.hub.get_dir())
        hub_dir.mkdir(parents=True, exist_ok=True)

        results = requests.get(json_url).json()
        if runtime_backend.startswith("torch"):
            fname = "xumx_slicq_v2_realtime.pth" if realtime else "xumx_slicq_v2_offline.pth"
            state = torch.hub.load_state_dict_from_url(pth_url, file_name=fname, progress=True)
        elif runtime_backend.startswith("onnx"):
            # let's use torch hub's dir to save onnx file
            onnx_dest = Path(hub_dir / "checkpoints/xumx_slicq_v2_realtime.onnx")
            if not onnx_dest.exists():
                download_url(onnx_url, onnx_dest)
            onnx_model_path = onnx_dest

    sample_rate = results["args"]["sample_rate"]

    # need to configure an NSGT object to peek at its params to set up the neural network
    # e.g. M depends on the sllen which depends on fscale+fmin+fmax
    nsgt_base = NSGTBase(
        results["args"]["fscale"],
        results["args"]["fbins"],
        results["args"]["fmin"],
        fs=sample_rate,
        device=device,
    )

    nb_channels = 2

    seq_dur = results["args"]["seq_dur"]

    jagged_slicq, _ = nsgt_base.predict_input_size(1, nb_channels, seq_dur)
    cnorm = ComplexNorm().to(device)

    nsgt, insgt = make_filterbanks(nsgt_base, sample_rate)
    encoder = (nsgt, insgt, cnorm)

    nsgt = nsgt.to(device)
    insgt = insgt.to(device)

    jagged_slicq_cnorm = cnorm(jagged_slicq)

    if runtime_backend.startswith("torch"):
        xumx_model = Unmix(
            jagged_slicq_cnorm,
            realtime=results["args"]["realtime"],
        )

        xumx_model.load_state_dict(state, strict=False)
        xumx_model.freeze()
        xumx_model.to(device)
    elif runtime_backend.startswith("onnx"):
        assert onnxruntime_available
        xumx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(xumx_model)

        available_providers = onnxruntime.get_available_providers()

        provider = None
        sess_options = onnxruntime.SessionOptions()

        if runtime_backend.endswith("cpu"):
            provider = ["CPUExecutionProvider"]

            # cpu perf tuning: https://fs-eire.github.io/onnxruntime/docs/performance/tune-performance.html#default-cpu-execution-provider-mlas
            # ORT_PARALLEL + num threads
            # sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
            # sess_options.intra_op_num_threads = 8
        elif runtime_backend.endswith("cuda"):
            # provider = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'}), "CPUExecutionProvider"]
            provider = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        print(f"ONNXRuntime chosen provider: {provider}")

        ort_session = onnxruntime.InferenceSession(
            str(onnx_model_path), providers=provider, sess_options=sess_options
        )
        xumx_model = ort_session
    else:
        raise ValueError(f"unsupported runtime backend: {runtime_backend}")

    return xumx_model, encoder, sample_rate
