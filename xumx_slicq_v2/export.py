from typing import Optional, Union, Tuple
import sys
from tqdm import trange
from pathlib import Path
import torch
import json
from torch import Tensor
import torch.nn as nn
from .model import Unmix
from .transforms import (
    ComplexNorm,
    NSGTBase,
    make_filterbanks,
)
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="xumx-sliCQ-V2 export")

    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Use offline model instead of realtime",
    )
    parser.add_argument(
        "--target",
        choices=("onnx", "torchscript"),
        default="onnx",
        help="Export model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="choose a nonstandard pretrained model",
    )

    args = parser.parse_args()
    sample_rate = 44100
    device = torch.device("cpu")

    if args.model is not None:
        model_path = f"/xumx-sliCQ-V2/.github/pretrained_models_other/{args.model}"
    elif args.offline:
        model_path = "/xumx-sliCQ-V2/pretrained_model"
    else:
        model_path = "/xumx-sliCQ-V2/pretrained_model_realtime"

    print(f"Exporting to format: {args.target}")

    model_path = Path(model_path)

    model_name = "xumx_slicq_v2"
    model_path = Path(model_path).expanduser()

    # load model from disk
    with open(Path(model_path, f"{model_name}.json"), "r") as stream:
        results = json.load(stream)

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

    target_model_path = Path(model_path, f"{model_name}.pth")
    state = torch.load(target_model_path, map_location=device)

    jagged_slicq, _ = nsgt_base.predict_input_size(1, nb_channels, seq_dur)
    cnorm = ComplexNorm().to(device)

    nsgt, insgt = make_filterbanks(nsgt_base, sample_rate)
    encoder = (nsgt, insgt, cnorm)

    nsgt = nsgt.to(device)
    insgt = insgt.to(device)

    jagged_slicq_cnorm = cnorm(jagged_slicq)

    xumx_model = Unmix(
        jagged_slicq_cnorm,
        realtime=not args.offline,
    )

    xumx_model.load_state_dict(state, strict=False)
    xumx_model.freeze()
    xumx_model.to(device)

    if args.target == "onnx":
        dest_path = Path(model_path, f"{model_name}.onnx")

        torch.onnx.export(
            xumx_model,
            (tuple([jagged_slicq_.to(device) for jagged_slicq_ in jagged_slicq]),),
            dest_path,
            input_names=[f"xcomplex{i}" for i in range(len(jagged_slicq))],
            output_names=[f"ycomplex{i}" for i in range(len(jagged_slicq))],
            dynamic_axes={
                **{f"xcomplex{i}": {3: "nb_slices"} for i in range(len(jagged_slicq))},
                **{f"ycomplex{i}": {4: "nb_slices"} for i in range(len(jagged_slicq))},
            },
            opset_version=16,
        )
    elif args.target == "torchscript":
        dest_path = Path(model_path, f"{model_name}.pt")

        ts_model = torch.jit.trace(xumx_model, (jagged_slicq,))
        ts_model.save(dest_path)
