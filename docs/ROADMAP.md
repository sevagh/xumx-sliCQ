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
    * Mask sum loss: 4.4 dB
    * nn stuff
        * **medium** _Cross-target Skip Connection_ sum all targets between mirror encoder/decoders
            * implement with simple code:  x_encoded1_target1, x_encoded2_target2, etc.
            * skip conn citations:
<https://file.techscience.com/ueditor/files/csse/TSP_CSSE-44-3/TSP_CSSE_29732/TSP_CSSE_29732.pdf>
<https://arxiv.org/pdf/1606.08921.pdf>
        * **optional** new slicqt: ('bark', 288, 43.39999999999988); try it?
            10.17 wiener oracle vs. 10.14,  see what it does to model size + results
        * lightweight variants: `-lite`: no wiener + capped bandwidth
            * try no-wiener in runtime
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
* **hard/future** cross frequency bin mixing (needs new code)?
    "global" bottleneck layer, but it messes with skip conns
    return skip conn from encoder function!

    # list
    encoded, skip_conn = sliced_umx.encoder()

    # global bottleneck
    encoded_concat = concat or whatever
    # try this: https://stats.stackexchange.com/a/552170
    encoded_concat = self.bottleneck(encoded_concat)
    encoded = deconcat
        
    decoded, masks = sliced_umx.decoder(encoded, skip_conn)
    

* complex NN for estimating phase
* spiking neural networks (hah)

*effort 2: inference/public*
    * option for TensorRT model
1. Inference = '__main__.py'; ensure it works; CPU or GPU inference with outputting files (for demos etc.) is fine
    * make it inference.py and make __main__ invoke it too
1. Start working on README, paper materials
1. create slim Dockerfile for pytorch runtime inference
