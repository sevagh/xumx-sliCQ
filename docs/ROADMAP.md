# xumx-sliCQ-V2

## Current running command

```
docker run --rm -it \
    --privileged --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /home/sevagh/thirdparty-repos/MUSDB18-HQ:/MUSDB18-HQ \
    -v /home/sevagh/thirdparty-repos/xumx-sliCQ-V2:/xumx-sliCQ-V2 \
    -v /home/sevagh/thirdparty-repos/xumx-sliCQ-V2/trained-model:/model \
    -p 6006:6006 \
    xumx-slicq-v2 \
    python -m xumx_slicq_v2.training --batch-size=256 --batch-size-valid=3
```

* Cadenza challenge registration issues

## Goals

* merge repos
    * Dockerfile.slim
    * training vs. public README

*effort 1: training*
* xumx-sliCQ-V2-training: this repo; 28MB v1-inspired model
* New training
    * Differentiable sliCQT-Wiener w/ complex-MSE, squeeze more juice from network, v1 28MB
    * looking good and we have a complex-valued loss baseline...
    4.2 dB median SDR
    * complex loss, gradients, computational graph in blog post
    * tag as "v1.0.0a"
* TensorRT save script (ala blendmodels)
* README to describe all the cool things (and not so cool things)
    nvcr, training, blending, <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>

*effort 2: inference/public*
    * use warp + potential C++ kernels (C++ for C++ sake is a bad idea, remember) for packed-nsgt
        * must be compatible with regular nsgt!
    * TensorRT script; load + use packed-nsgt with realtime inputs (and offline, same script)
    * provide measurements/SDR docs for it all
1. Inference = '__main__.py'; ensure it works; CPU or GPU inference with outputting files (for demos etc.) is fine
1. Start working on README, paper materials
1. create slim Dockerfile for pytorch runtime inference
