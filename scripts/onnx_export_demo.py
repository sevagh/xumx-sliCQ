#!/usr/bin/env python3

import sys
from pathlib import Path
from xumx_slicq.utils import load_target_models
import json
import torch

'''
code from: https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
reproduction of this bug: https://discuss.pytorch.org/t/torch-onnx-export-of-pytorch-model-is-slow-expected-completion-time/127979
'''

if __name__ == '__main__':
    model_str_or_path = sys.argv[1]
    model_path = Path(model_str_or_path).expanduser()

    # when path exists, we assume its a custom model saved locally
    if model_path.exists():
        with open(Path(model_path, "separator.json"), "r") as stream:
            enc_conf = json.load(stream)

        xumx_model, model_nsgt, jagged_slicq_sample = load_target_models(
            model_str_or_path=model_path, pretrained=True, sample_rate=enc_conf["sample_rate"], device="cpu"
        )
    else:
        raise ValueError(f'path {model_path} is not a valid xumx-sliCQ model')

    xumx_model.eval()

    # Input to the model
    torch_out = xumx_model(jagged_slicq_sample)

    print('Starting export to onnx...')
    # Export the model
    torch.onnx.export(xumx_model,                # model being run
                      jagged_slicq_sample,       # model input (or a tuple for multiple inputs)
                      "xumx_slicq.onnx",         # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
