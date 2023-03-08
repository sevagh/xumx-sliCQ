<!--
# META:TODO

1. Code
    1. diagrams (inkscape, block spectrograms, try mermaid?)
1. git tag as "v1.0.0a" when ready
1. submit to cadenza challenge
-->

# xumx-sliCQ-V2

[![Pypi](https://img.shields.io/pypi/v/openunmix.svg)](https://pypi.python.org/pypi/openunmix)
[![Dockerhub](https://img.shields.io/docker/pulls/MIR-MU/pv211-utils)](https://hub.docker.com/repository/docker/miratmu/pv211-utils)
[![arXiv](https://img.shields.io/badge/arXiv-2112.05509-b31b1b.svg)](https://arxiv.org/abs/2112.05509)

xumx-sliCQ-V2 is a PyTorch neural network for music demixing, trained only on [MUSDB18-HQ](https://zenodo.org/record/3338373).

It demixes a musical mixture into stems (vocals/drums/bass/other) by masking the magnitude spectrogram. The code is based on [Open-Unmix (UMX)](https://github.com/sigsep/open-unmix-pytorch) with some key differences:
1. Spectral transform: sliced Constant-Q Transform (sliCQT) with the Bark scale vs. STFT
1. Neural network architecture: convolutional denoising autoencoder (CDAE) vs. dense + Bi-LSTM
1. All targets are trained together with combined loss functions like [CrossNet-Open-Unmix (X-UMX)](https://github.com/sony/ai-research-code/blob/master/x-umx/x-umx.md)

**xumx-sliCQ-V2 scores a total SDR of 4.4 dB with 60 MB\* of pretrained weights for all targets** on the MUSDB18-HQ test set.

It beats the 3.6 dB score of the original [xumx-sliCQ](https://github.com/sevagh/xumx-sliCQ) (28 MB) with the improvements [described here](#improvements-over-xumx-slicq). It also brings the performance closer to the 4.64 dB and 5.54 dB scored by UMX and X-UMX (137 MB) respectively.\*\*

**xumx-sliCQ-V2 is fast and light!** TODO TensorRT etc. here

(TODO cadenza challenge results) here I worked on xumx-sliCQ-V2 for the Cadenza Challenge, where it placed **14th place** in the first challenge. Cite xumx-sliCQ-V2:
```
(TODO latex citation block here)
```

<sub>

\*: Pretrained weights for xumx-sliCQ-V2 are stored [in this repository](./pretrained_model) with Git LFS

</sub>

<sub>

\*\*: UMX and X-UMX were independently re-evaluated as part of xumx-sliCQ: [1](https://github.com/sevagh/xumx_slicq_extra/blob/main/old-latex/mdx-submissions21/paper.md#results), [2](https://github.com/sevagh/xumx_slicq_extra)

</sub>

## Key concepts

Bark-scale sliCQT to better represent musical signals with a nonuniform time-frequency resolution compared to the fixed resolution STFT:

![slicq-spectral](.github/slicq_spectral.png)

Convolutional network applied to ragged sliCQT (time-frequency blocks with different frame rates):

![slicq-diagram](.github/slicq_diagram.png)

## Usage

### Prerequisites

You can use Python >= 3.8 with the pip package directly, or Docker for convenience. To use your GPU, you need the NVIDIA CUDA Toolkit and an NVIDIA GPU. For Docker + GPU, you also need the [nvidia-docker 2.0 runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). For your own training, tuning, and evaluation, you need the [MUSDB18-HQ dataset](https://zenodo.org/record/3338373).

### Install

xumx-sliCQ-V2 is available on [PyPI.org](todo-link-here):
TODO ensure inference.py works
TODO try-except blocks on scripts w/ optional deps e.g. training tuning
TODO build wheel, publish to pypi.org, package 60MB weights in wheel
```
pip install xumx_slicq_v2           # basic inference

pip install xumx_slicq_v2[tensorrt] # tensorrt inference
pip install xumx_slicq_v2[devel]    # training, tuning, development, etc.
```

TODO write python usage here with inference demo cpu/gpu?

A minimal inference container is also available on [Dockerhub](todo-link-here):
Fix up Dockerfile.inference with inference entrypoint, build + publish to dockerhub
```
docker pull sevagh2/xumx-slicq-v2-inference
```

TODO write docker usage here with inference demo cpu/gpu?

<details>
<summary>List of all scripts of xumx-sliCQ-V2</summary>

| Script | Description | Device |
|:-|:-|:-|
| For end users | |
| xumx_slicq_v2.inference | Demix mixed songs | CPU **or** CUDA GPU |
| xumx_slicq_v2.tensorrt_inference | Demix mixed songs with TensorRT | CUDA GPU |
| For developers | |
| xumx_slicq_v2.evaluation | Evaluate pretrained networks | CPU |
| xumx_slicq_v2.training | Train the network | CUDA GPU |
| xumx_slicq_v2.tensorrt_export | Convert pretrained model to TensorRT | CUDA GPU |
| xumx_slicq_v2.optuna | Optuna hyperparam tuning | CUDA GPU |
| xumx_slicq_v2.slicqfinder | Random sliCQT param search | CPU **or** CUDA GPU |
| xumx_slicq_v2.visualization | Generate spectrograms | CPU |

If you installed the package with pip, run them like `python -m xumx_slicq_v2.$script_name`.

</details>

### Basic inference (CPU, CUDA GPU)

### TensorRT inference

TODO: <https://pytorch.org/TensorRT/getting_started/getting_started_with_python_api.html#getting-started-with-python-api>

### Training, development with Docker

<details>
<summary>Why Docker</summary>

* When revisiting my old code, [xumx-sliCQ](https://github.com/sevagh/xumx-sliCQ), I realized the dev environment was not reproducible
* Within the last year, I have grown to dislike Conda, setuptools, pip, requirements.txt files, and everything related to packaging and dependency management for Python
* Docker lets me deliver an image that works without worrying about the user's host environment, and lets me mix multiple paradigms of Python packaging without creating overly-complex install instructions
    * Docker can be run on Windows and OSX, whereas if I don't use Docker, I can only provide instructions for Linux (my OS)
</details>

Build the development container:

```
$ docker build -t "xumx-slicq-v2" .
```
It is based on the [NVIDIA PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) to include features and optimizations for to NVIDIA GPUs, such as automatic [TF32 for Ampere+](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/), [bfloat16 support for Ampere+](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html), and more.

<details>
<summary>Dynamic source code loading</summary>

To dynamically update the source code in the container to develop new features, you can volume mount your local checkout of xumx-sliCQ-V2 to `:/xumx-sliCQ-V2`. If not, the container will use a frozen copy of the source code when you built the image.

</details>
    
<details>
<summary>Training</summary>

```
$ docker run --rm -it \
    --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /path/to/MUSDB18-HQ/dataset:/MUSDB18-HQ \
    -v /path/to/save/trained/model:/model \
    -p 6006:6006 \
    xumx-slicq-v2 \
    python -m xumx_slicq_v2.training --help
```

Browse the Tensorboard training dashboard at <http://127.0.0.1:6006/>:
TODO tensorboard screenshot
![tensorboard](.github/tensorboard.png)

To persist the model, you can volume mount a host volume to `:/model` (as in the command above). Running multiple times with a persisted model will continue the training process. If not, the trained model will disappear when the container is killed.

TODO training details (epochs, best epoch, lowest loss)

</details>

<details>
<summary>Hyperparameter tuning</summary>

```
$ docker run --rm -it \
    --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /path/to/MUSDB18-HQ/dataset:/MUSDB18-HQ \
    -p 6006:6006 \
    xumx-slicq-v2 \
    python -m xumx_slicq_v2.optuna --help
```
Browse the Optuna tuning dashboard at <http://127.0.0.1:6006/>
TODO optuna dashboard screenshot

</details>

<details>
<summary>Evaluation</summary>

```
$ docker run --rm -it \
    -v /path/to/MUSDB18-HQ/dataset:/MUSDB18-HQ \
    xumx-slicq-v2 \
    python -m xumx_slicq_v2.evaluation --help
```
</details>

## Theory

<details>
<summary>Motivation</summary>

The sliced Constant-Q Transform (sliCQT) is a realtime implementation of the Nonstationary Gabor Transform (NSGT), which is a generalized nonuniform time-frequency transform with perfect inverse. Nonuniform time-frequency transforms are better suited to representing sounds with time-varying frequencies, such as music and speech. The STFT is limited due to its use of fixed windows and the time-frequency uncertainty principle of Gabor.

The NSGT can be used to implement a Constant-Q Transform (logarithmic scale), but it can use any type of frequency scale. In xumx-sliCQ and xumx-sliCQ-V2, the same Bark scale is used (262 Bark frequency bins from 32.9-22050 Hz).

</details>

<details>
<summary>Cadenza Challenge 2023</summary>

In 2021, I worked on xumx-sliCQ to submit to the MDX 21 ([Music Demixing Challenge 2021](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021) on AICrowd), and got my paper published to [the MDX 21 workshop](https://mdx-workshop.github.io/proceedings/hanssian.pdf) at ISMIR 2021 (and [arXiv](https://arxiv.org/abs/2112.05509)). The time-frequency uncertainty principle aligned with my desired thesis topic at the Music Technology Master's program at McGill.

In 2023, I chose to revisit the code of xumx-sliCQ for submission to the [Cadenza Challenge](http://cadenzachallenge.org/), which is a music demixing challenge with the additional context of hearing loss and accessibility. Nonuniform time-frequency transforms, like the sliCQT, are related to the nolinear human auditory system, and I had specific auditory motivations for choosing the Bark scale for the sliCQT in xumx-sliCQ.

</details>

<details>
<summary>Past work</summary>

TODO: fill this in with everything!

</details>

### Improvements over xumx-sliCQ

<details>
<summary>Performance tuning</summary>

First, I improved a lot of sloppy non-neural network code. The embedded [nsgt library](./xumx_slicq_v2/nsgt), which provides the sliCQT (and originates from <https://github.com/sevagh/nsgt>, and before that, the source <https://github.com/grrrr/nsgt>), had a lot of calls to NumPy after my initial conversion to PyTorch, leading to unnecessary host-device communication throughout an epoch trained on my GPU.

Next, I focused on making my epochs faster. The faster I can train it, the more I can work on xumx-sliCQ-V2 within a given time frame. To get the most out of the PyTorch code and my NVIDIA Ampere GPU (3090), I used two resources:
* Using the NVIDIA PyTorch Docker container (`nvcr.io/nvidia/pytorch:22.12-py3`) as the base for my training container to take advantage of implicit speedups provided by NVIDIA (e.g. automatically-enabled [TF32](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/_))
* Modifying my PyTorch code according to the [performance tuning guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

The code changes were the following:
* In the model code:
    1. `bias=False` for every conv layer that was followed by a batch norm:
        ```
        encoder.extend([
            Conv2d(
                hidden_size_1,
                hidden_size_2,
                (freq_filter, time_filter_2),
                bias=False,
            ),
            BatchNorm2d(hidden_size_2),
            ReLU(),
        ])
        ```
* In the training code:
    1. Set the model `.to(memory_format=torch.channels_last)`
    1. Enable cuDNN benchmarking
        ```
        torch.backends.cudnn.benchmark = True
        ```
    1. Forcing some additional more TF32-related settings:
        ```
        torch.backends.cudnn.allow_tf32 = True
        ```
    1. Using AMP (Automatic Mixed Precision) with bfloat16 (on CUDA and CPU) (greatly reduces memory during training, allowing a larger batch size):
        ```
        with torch.autocast("cuda", dtype=torch.bfloat16) \
                torch.autocast("cpu", dtype=torch.bfloat16):
        ```

An epoch takes ~170s (train + validation) on my RTX 3090 with 24GB of GPU memory with `--batch-size=64 --nb-workers=8`. xumx-sliCQ by contrast took 350s per epoch with a batch size of 32 on an RTX 3080 Ti (which had 12GB GPU memory, half of my 3090). However, the old code used PyTorch 1.10, so the upgrade of V2 to 1.13 may also be contributing to improved performance.

</details>

<details>
<summary>Using the full frequency bandwidth</summary>

In xumx-sliCQ, I didn't use frequency bins above 16,000 Hz in the neural network; the demixing was only done on the frequency bins lower than that limit, copying the `umx` pretrained model of UMX. UMX's other pretrained model, `umxhq`, uses the full spectral bandwidth. In xumx-sliCQ-V2, I removed the bandwidth parameter to pass all the frequency bins of the sliCQT through the neural network.

</details>

<details>
<summary>Removing the inverse sliCQT and time-domain SDR loss</summary>

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

</details>

<details>
<summary>Replacing the overlap-add with pure convolutional layers</summary>

TODO: missing overlap-add->pure conv section

A quirk of the sliCQT is that rather than the familiar 2 dimensions of time and frequency, it has 3 dimensions: slice, time-per-slice, and frequency. Adjacent slices have a 50% overlap with one another and must be summed to get the true spectrogram in a destructive operation (50% of the time coefficients are lost, with no inverse):

[insert example here]

In xumx-sliCQ, an extra transpose convolutional layer with stride 2 is used to grow the time coefficients back to the original size after the 4-layer CDAE, to undo the destruction of the overlap-add:

[insert example here]

In xumx-sliCQ-V2, the first convolutional layer takes the overlap into account by setting the kernel and stride to the window and hop size of the destructive overlap-add. The result is that the input is downsampled in a way that is recovered by the final transpose convolution layer in the 4-layer CDAE, eliminating the need for an extra upsampling layer:

[insert example here]

By this point, I had a model that scored **4.1 dB** with 28 MB of weights using magnitude MSE loss.

</details>

<details>
<summary>Differentiable Wiener-EM and complex MSE</summary>

Borrowing from [Danna-Sep](https://github.com/yoyololicon/danna-sep), one of the [top performers in the MDX 21 challenge](https://github.com/yoyololicon/music-demixing-challenge-ismir-2021-entry), the differentiable Wiener-EM step is used inside the neural network during training, such that the output of xumx-sliCQ-V2 is a complex sliCQT, and the complex MSE loss function is used instead of the magnitude MSE loss. Wiener-EM is applied separately in each frequency block as shown in the [architecture diagram at the top of the README](#key-concepts).

In xumx-sliCQ, Wiener-EM was only applied in the STFT domain as a post-processing step. The network was trained using magnitude MSE loss. The waveform estimate of xumx-sliCQ combined the estimate of the target magnitude with the phase of the mix (noisy phase or mix phase).

This got the score to **4.24 dB** with 28 MB of weights trained with complex MSE loss (0.0395).

</details>

<details>
<summary>Discovering hyperparameters with Optuna</summary>

Using the included [Optuna tuning script](./xumx_slicq_v2/tuning.py), new hyperparameters that gave the highest SDR after cut-down training/validation epochs were:
* Changing the hidden sizes (channels) of the 2-layer CDAE from 25,55 to 50,51 (increased the model size from ~28-30MB to 60MB)
* Changing the size of the time filter in the 2nd layer from 3 to 4

Note that:
* The time kernel and stride of the first layer uses the window and hop size related to the overlap-add procedure, so it's not a tunable hyperparameter
* The ragged nature of the sliCQT makes it tricky to modify frequency kernel sizes (since the time-frequency bins can vary in their frequency bins, from 1 single frequency up to 86), so I kept those fixed from xumx-sliCQ
* The sliCQT params could be considered a hyperparameter, but the shape of the sliCQT modifies the network architecture, so for simplicity I kept it the same as xumx-sliCQ (262 bins, Bark scale, 32.9-22050 Hz)

This got the score to **4.35 dB** with 60 MB of weights trained with complex MSE loss of 0.0390.

</details>

<details>
<summary>Mask sum MSE loss</summary>

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

This got the score to **4.4 dB** with 60 MB of weights trained with complex MSE loss + mask sum loss of 0.0405.

</details>

<details>
<summary>Reducing the time downsampling</summary>

</details>
