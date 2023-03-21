from typing import Optional, Union, Tuple
import sys
from tqdm import trange
from pathlib import Path
import copy
import torch
import json
from torch import Tensor
import torch.nn as nn
from .model import Unmix
from .drum_slim import RealtimeDrumUnmix, RealtimeDrumSeparator
from .transforms import (
    ComplexNorm,
    NSGTBase,
    make_filterbanks,
)
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="xumx-sliCQ-V2 export realtime drum model")

    args = parser.parse_args()
    sample_rate = 44100
    device = torch.device("cuda")

    model_path = "/xumx-sliCQ-V2/pretrained_model_realtime"

    print(f"Exporting to format: ONNX")

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
    encoder = (nsgt, insgt)

    nsgt = nsgt.to(device)
    insgt = insgt.to(device)

    jagged_slicq_cnorm = cnorm(jagged_slicq)

    xumx_model = Unmix(
        jagged_slicq_cnorm,
        realtime=True,
    )

    xumx_model.load_state_dict(state, strict=False)
    xumx_model.freeze()
    xumx_model.to(device)

    rt_drum_model = RealtimeDrumUnmix(
        jagged_slicq_cnorm
    )
    rt_drum_model.freeze()
    rt_drum_model.to(device)

    print(f"Copying drum model from full-size xumx")
    for i in range(len(xumx_model.sliced_umx)):
        # last cdae i.e. index 3 is the 'drum' cdae
        rt_drum_model.sliced_umx[i].cdae = copy.deepcopy(xumx_model.sliced_umx[i].cdaes[3])


    rt_drum_separator = RealtimeDrumSeparator(rt_drum_model, encoder)

    sample_waveform = torch.rand((1, 2, 4096), dtype=torch.float32, device=device)

    dest_path = Path(model_path, f"{model_name}_realtime_drum_separator.onnx")

    torch.onnx.export(
        rt_drum_separator,
        sample_waveform,
        dest_path,
        input_names=["mix_waveform"],
        output_names=["drum_waveform"],
        dynamic_axes={
            "mix_waveform": {2: "nb_frames"},
            "drum_waveform": {2: "nb_frames"},
        },
        opset_version=16,
    )
