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

xumx-slicq-v2 = rich tensorrt experience
xumx-slicq-v2-slim = dockerhub inference only

*effort 1: training*
* xumx-sliCQ-V2-training: this repo; 28MB v1-inspired model
* Differentiable sliCQT-Wiener w/ complex-MSE, squeeze more juice from network, v1 28MB
* looking good and we have a complex-valued loss baseline: 0.0395, 4.2 dB median SDR
    * Optuna: retrain with best results: 50 MB model
    * Slicqt wiener oracle: new slicqt: ('bark', 288, 43.39999999999988); try it :shrug:
    * TensorRT save script (ala blendmodels)
    * tag as "v1.0.0a"
    * visualization.py: spectrogram plotting code (+ overlap-add, flatten, + per-block vs. unified spectrogram)
        __name__ == __main__ plotting script
* README to describe all the cool things (and not so cool things)
    * nvcr
    * how its a "toolbox" for slicqt-based demixing
    * tuning: <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>
    * optuna + optuna dashboard/screenshots
    * tensorboard + training UI/screenshots

## Optuna optimization rounds

* Optuna round 1: hidden_size_1, hidden_size_2, time_filter_2
    Value:  -1.5838500261306763
      Params:
        hidden_size_1: 50
        hidden_size_2: 51
        time_filter_2: 4

## Wiener-oracle

current config (262, 32.9):
    bass, drums, vocals, other sdr! 9.53 9.90 11.51 9.62
    total sdr: 10.14

new wiener-em oracle: not much better
    total:  10.17373390147092       ('bark', 288, 43.39999999999988)

*effort 2: inference/public*
    * option for TensorRT model
1. Inference = '__main__.py'; ensure it works; CPU or GPU inference with outputting files (for demos etc.) is fine
    * make it inference.py and make __main__ invoke it too
1. Start working on README, paper materials
1. create slim Dockerfile for pytorch runtime inference
