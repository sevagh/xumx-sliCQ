<!--

# META:TODO

1. Code
    1. Evaluation should also evaluate umx/xumx
    1. TensorRT save/load/inference with time/latency measurements and realtime example with causal inputs
    <https://pytorch.org/TensorRT/getting_started/getting_started_with_python_api.html#getting-started-with-python-api>
    1. Ensure inference.py works
    1. Fix up Dockerfile.inference for light torch inference, build + publish to dockerhub with GHA
    1. Create pypi.org package a la UMX (with an auto-download of the 60MB LFS weights)
        * split into separate package for scripts
        * with publishing via GHA, badges
        * code_quality.sh: implement suggestions, cleanups, etc.
1. README
    1. make collapsible ala: https://raw.githubusercontent.com/noahmr/yolov5-tensorrt/main/README.md
    1. citations/references everywhere!
    1. diagrams (inkscape, block spectrograms, try mermaid?)
    1. dockerized setup/example commands for everything, w/ cpu/gpu
        * optuna + optuna dashboard/screenshots
        * tensorboard + training UI/screenshots
    1. separate performance tuning section: nvcr + perf tuning: <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>
1. tag as "v1.0.0a"
-->

# xumx-sliCQ-V2

xumx-sliCQ-V2 is a neural network for music demixing trained only on MUSDB18-HQ.

It demixes a musical mixture into stems (vocals/drums/bass/other) by masking the magnitude spectrogram. The code is based on [Open-Unmix (UMX)](https://github.com/sigsep/open-unmix-pytorch) with some key differences:
1. Spectral transform: sliced Constant-Q Transform (sliCQT) vs. STFT
1. Neural network architecture: convolutional denoising autoencoder (CDAE) vs. dense + Bi-LSTM
1. All targets are trained together (like [CrossNet-Open-Unmix/X-UMX](insert-link)) with combined loss functions

**xumx-sliCQ-V2 scores a total SDR of 4.4 dB with 60 MB of pretrained weights for all targets** on the MUSDB18-HQ test set. The weights are stored [in this repository](./pretrained_model) with Git LFS.

This beats the 3.6 dB score  of the original [xumx-sliCQ](https://github.com/sevagh/xumx-sliCQ) (28 MB). It also brings the performance closer to the 4.4 and 5.6 dB scored by UMX/X-UMX (137 MB) respectively, which was my original goal when replacing the STFT with the sliCQT.

## Usage

### Basic inference

Pip install:
```
pip install xumx_slicq_v2
```

Docker image:
```
docker run \
    -v /path/to/inputs:/input \    # your input wav files
    -v /path/to/outputs:/output \  # destination for demixed outputs
    sevagh2/xumx-slicq-v2-cpu
```

### Training, development, and TensorRT inference

You need an NVIDIA GPU, the NVIDIA CUDA toolkit, Docker, the [nvidia-docker 2.0 runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker), and the [MUSDB18-HQ dataset](https://zenodo.org/record/3338373).

Note that you can try to use the pip package for these advanced targets, but I only tested the Docker workflow:
```
pip install xumx_slicq_v2[tensorrt] # optional tensorrt inference
pip install xumx_slicq_v2[training] # training
pip install xumx_slicq_v2[devel]    # optuna hyperparam tuning, plotting, etc.
```

<details>
<summary>Why Docker</summary>

* When revisiting my old code, [xumx-sliCQ](https://github.com/sevagh/xumx-sliCQ), I realized the dev environment was not reproducible
* Within the last year, I have grown to dislike Conda, setuptools, pip, requirements.txt files, and everything related to packaging and dependency management for Python
* Docker lets me deliver an image that works without worrying about the user's host environment, and lets me mix multiple paradigms of Python packaging without creating overly-complex install instructions
    * Docker can be run on Windows and OSX, whereas if I don't use Docker, I can only provide instructions for Linux (my OS of choice)

</details>

<details>
<summary>Build the container</summary>

```
$ docker build -t "xumx-slicq-v2" .
```

2. Run training on dataset `/MUSDB18-HQ`, save model in `/model`, visit tensorboard at <http://127.0.0.1:6006/>

```
$ docker run --rm -it \
    --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /path/to/MUSDB18-HQ/dataset:/MUSDB18-HQ \
    -v /path/to/save/trained/model:/model \
    -p 6006:6006 \
    xumx-slicq-v2 \
    python -m xumx_slicq_v2.training --help
```

**N.B.** if your model path already contains a trained model, training will continue from the saved checkpoint

3. Run inference on `/input` tracks with `/model`, save into `/output`

```
$ docker run --rm -it \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /path/to/input/tracks:/input \
    -v /path/to/output/tracks:/output \
    -v /path/to/trained/model:/model \
    xumx-slicq-v2 \
    python -m xumx_slicq_v2.inference --help
```

**N.B.** no GPUs for now; later, use TensorRT for fast GPU inference

4. Run evaluation on `/MUSDB18-HQ` tracks with `/model`, save into `/output` and `/evaluation`

```
$ docker run --rm -it \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /path/to/MUSDB18-HQ:/MUSDB18-HQ \
    -v /path/to/trained/model:/model \
    -v /path/to/output/tracks:/output \
    -v /path/to/store/evaluations:/evaluation \
    xumx-slicq-v2 \
    python -m xumx_slicq_v2.evaluation --help
```

**N.B.** no GPUs for now; later, use TensorRT for fast GPU inference
</details>

## Design

<details>
<summary>Theory</summary>

The sliced Constant-Q Transform (sliCQT) is a realtime implementation of the Nonstationary Gabor Transform (NSGT), which is a generalized nonuniform time-frequency transform with perfect inverse. Nonuniform time-frequency transforms are better suited to representing sounds with time-varying frequencies, such as music and speech. The STFT is limited due to its use of fixed windows and the time-frequency uncertainty principle of Gabor.

The NSGT can be used to implement a Constant-Q Transform (logarithmic scale), but it can use any type of frequency scale. In xumx-sliCQ and xumx-sliCQ-V2, the same Bark scale is used (262 Bark frequency bins from 32.9-22050 Hz).

[insert example here]

</details>

<details>
<summary>Network architecture</summary>

Network architecture for nonuniform time-frequency transforms

</details>

<details>
<summary>Improvements over xumx-sliCQ</summary>

### Using the full frequency bandwidth

In xumx-sliCQ, I didn't use frequency bins above 16,000 Hz in the neural network; the demixing was only done on the frequency bins lower than that limit, copying the `umx` pretrained model of UMX. UMX's other pretrained model, `umxhq`, uses the full spectral bandwidth. In xumx-sliCQ-V2, I removed the bandwidth parameter to pass all the frequency bins of the sliCQT through the neural network.

### Removing the inverse sliCQT and time-domain SDR loss

In xumx-sliCQ, I applied the mixed-domain SDR and MSE loss of X-UMX. However, due to the large computational graph introduced by the inverse sliCQT operation, I was disabling its gradient:
```
X = slicqt(x)
Xmag = torch.abs(X)
Ymag_est = unmix(Xmag)
Ycomplex_est = mix_phase(torch.angle(X), Ymag_est)

with torch.no_grad():
     y_est = islicqt(Ycomplex_est)
```

Without this, the epoch time goes from 1-5 minutes to 30+ minutes. However, by disabling the gradient, the SDR loss can't influence the network performance. In practice, I found that the MSE is an acceptable correlate to SDR performance, and dropped the isliCQT and SDR loss calculation.

### Replacing the overlap-add with pure convolutional layers

A quirk of the sliCQT is that rather than the familiar 2 dimensions of time and frequency, it has 3 dimensions: slice, time-per-slice, and frequency. Adjacent slices have a 50% overlap with one another and must be summed to get the true spectrogram in a destructive operation (50% of the time coefficients are lost, with no inverse):

[insert example here]

In xumx-sliCQ, an extra transpose convolutional layer with stride 2 is used to grow the time coefficients back to the original size after the 4-layer CDAE, to undo the destruction of the overlap-add:

[insert example here]

In xumx-sliCQ-V2, the first convolutional layer takes the overlap into account by setting the kernel and stride to the window and hop size of the destructive overlap-add. The result is that the input is downsampled in a way that is recovered by the final transpose convolution layer in the 4-layer CDAE, eliminating the need for an extra upsampling layer:

[insert example here]

### Differentiable Wiener-EM and complex MSE

Borrowing from [Danna-Sep](https://github.com/yoyololicon/danna-sep), one of the top performers in the MDX 21 challenge, the differentiable Wiener-EM step is used inside the neural network during training, such that the output of xumx-sliCQ-V2 is a complex sliCQT, and the complex MSE loss function is used instead of the magnitude MSE loss. Wiener-EM is applied separately in each frequency block:

[insert example here]

In xumx-sliCQ, Wiener-EM was only applied in the STFT domain as a post-processing step. The network was trained using magnitude MSE loss. The waveform estimate of xumx-sliCQ combined the estimate of the target magnitude with the phase of the mix (noisy phase or mix phase).

### Discovering hyperparameters with Optuna

### Mask sum loss

In spectrogram masking approaches to music demixing, commonly a ReLU or Sigmoid activation function is applied as the final activation layer to produce a non-negative mask for the mix magnitude spectrogram. In xumx-sliCQ, I used a Sigmoid activation in the final layer (UMX uses a ReLU). The final mask is multiplied with the input mixture:
```
mix = x.clone()

# x is a mask
x = cdae(x)

# apply the mask, i.e. multiplicative skip connection
x = x*mix
```

Since the mask for each target is between [0, 1], and the targets must add up to the mix, then the masks must add up to exactly 1:
```
drum_mask*mix + vocals_mask*mix + other_mask*mix + bass_mask*mix = mix
drum_mask + vocals_mask + other_mask + bass_mask = 1.0
```

In xumx-sliCQ-V2, I added a second loss term called the mask sum loss, which is the MSE between the sum of the four target masks and a matrix of 1s. This needs a small code change where both the complex slicqt (after Wiener-EM) and the sigmoid masks are returned in the training loop.

</details>

<details>
<summary>Performance tuning</summary>

The training

</details>
