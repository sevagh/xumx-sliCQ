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
* looking good and we have a complex-valued loss baseline: 0.0395, 4.24 dB median SDR
    * Optuna: retrain with best results: 60 MB model: 0.0390, 4.35 dB median SDR
        hidden_size: 50, 51, time_filter_2: 4
    * nn stuff
        * **easy** _Mask Sum Loss_ **doing this now**: 
        * **medium** _Cross-target Skip Connection_ sum all targets between mirror encoder/decoders
            * implement with simple code:  x_encoded1_target1, x_encoded2_target2, etc.
            * skip conn citations:
        * possibly put back bandwidth param? save space 
<https://file.techscience.com/ueditor/files/csse/TSP_CSSE-44-3/TSP_CSSE_29732/TSP_CSSE_29732.pdf>
<https://arxiv.org/pdf/1606.08921.pdf>
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

*effort 3: from wip base, more NN stuff*
* new slicqt: ('bark', 288, 43.39999999999988); try it?
    10.17 wiener oracle vs. 10.14,  see what it does to model size + results
* **hard/future** cross frequency bin mixing (needs new code)?
    "global" bottleneck layer, but it messes with skip conns

*effort 2: inference/public*
    * option for TensorRT model
1. Inference = '__main__.py'; ensure it works; CPU or GPU inference with outputting files (for demos etc.) is fine
    * make it inference.py and make __main__ invoke it too
1. Start working on README, paper materials
1. create slim Dockerfile for pytorch runtime inference
