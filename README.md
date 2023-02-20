# xumx-sliCQ-V2

This is an adaptation of X-UMX (CrossNet-Open-Unmix respectively), a music demixing model that masks the magnitude STFT spectrogram of a mixed song to estimate VDBO (vocals, drums, bass, other) stems.

The STFT is substituted for the sliCQT (sliced Constant-Q Transform), a nonuniform time-frequency transform that addresses Gabor's time-frequency uncertainty principle.

This variant scores 4.1 dB on the MUSDB18-HQ test set with 70 MB of pretrained weights. This has yet to beat the scores of 5.41 dB and 5.79 dB achieved by UMX and X-UMX respectively, which both use 137 MB of weights.

My first attempt, [xumx-sliCQ](https://github.com/sevagh/xumx-sliCQ), scored 3.6 dB using 28 MB of pretrained weights.

## Usage

The included Dockerfile can be used to run xumx-sliCQ-V2 in a pytorch 1.13.1 container. Inference can be done on the CPU or GPU (you need the [nvidia-docker2 runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) for GPU inference in Docker).

1. Pull the xumx-slicq-v2 image from my Dockerhub account ([sevagh2/xumx-slicq-v2]())
```
$ docker pull sevagh2/xumx-slicq-v2
```

2. Put the input wav files you want to demix into a directory
```
$ ls /home/sevagh/test-songs-input
```

3. Create a directory for the output stem wav files
```
$ mkdir -p /home/sevagh/test-songs-output
```

4. Run the docker container with volume mounts to do the demixing:
```
$ docker run \
    -v /home/sevagh/test-songs-input:/input \
    -v /home/sevagh/test-songs-output:/output \
    sevagh2/xumx-slicq-v2
```

5. Enjoy the results
```
```

## Motivation

The sliCQT, like the CQT (Constant-Q Transform), uses long windows in the low frequency regions and short windows in the high frequency regions. This maps closer to characteristics in the human auditory system, music, and speech.

## Design challenges of the sliCQT



## Training

The training code and details for xumx-sliCQ-V2 and the pretrained models are in a separate repository, [xumx-sliCQ-V2-training](https://github.com/sevagh/xumx-sliCQ-V2-training). The pretrained models included in this repo were trained with [MSE](./pretrained_model/mse) loss and mixed [MSE-HAAQI](./pretrained_model/mse-haaqi) loss (Hearing Aid Audio Quality Index), for submission to [the Cadenza Challenge](http://cadenzachallenge.org/docs/cadenza1/cc1_intro).

The model trained on mixed MSE-SDR loss model performs similarly to the MSE-only model, with a higher epoch time, so it wasn't included.
