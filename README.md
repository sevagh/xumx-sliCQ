# xumx-sliCQ

My variant of the [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) (aka UMX) template for music source separation ([Stöter, Uhlich, Liutkus, Mitsufuji 2019](https://hal.inria.fr/hal-02293689/document)). This is a music source separation (or demixing) system which, given a mixed song input, estimates 4 target sources (drums, bass, vocals, other), as defined by [MUSDB18-HQ](https://zenodo.org/record/3338373) dataset. It differs from open-unmix-pytorch in the following ways:
* The spectral transform is the [sliCQ transform](https://github.com/sevagh/nsgt) ([Balazs et al. 2011](http://ltfat.org/notes/ltfatnote018.pdf) and [Dörfler et al. 2014](https://www.univie.ac.at/nonstatgab/cqt/index.php)) instead of an STFT
* Convolutional architecture (based loosely on [Grais, Zhao, and Plumbley 2019](https://arxiv.org/abs/1910.09266)) instead of the UMX linear encoder + LSTM + decoder
* Single network like [CrossNet-Open-Unmix](https://github.com/JeffreyCA/spleeterweb-xumx) ([Sawata, Uhlich, Takahashi, Mitsufuji 2020](https://www.ismir2020.net/assets/img/virtual-booth-sonycsl/cUMX_paper.pdf)), aka X-UMX
    * Combination loss (CL) - loss function on different linear combinations of the 4 targets
    * Multi-domain loss (MDL) - frequency-domain loss (MSE) and time-domain loss ([auraloss](https://github.com/csteinmetz1/auraloss) SI-SDR)

It's a working demonstration of the sliCQ transform in a neural network for music demixing (:heavy_check_mark:), but it failed to beat UMX or XUMX (:x:). The provided pretrained model was trained using the standard [MUSDB18-HQ](https://zenodo.org/record/3338373) dataset.

## Motivation

Time-frequency masking is one strategy for music source separation, where the magnitude spectrogram of the mix is multiplied by an estimated target mask ([more background here](https://source-separation.github.io/tutorial/basics/tf_and_masking.html)). Open-Unmix uses the short-time Fourier transform (STFT) for the spectral representation of music, and learns to estimate the magnitude STFT of a target from the mixture. The STFT is useful in audio and music applications, but it has a uniform and fixed frequency and time resolution controlled by the window size, where one size does not fit all: [paper 1](https://arxiv.org/abs/1504.07372), [paper 2](https://arxiv.org/abs/1905.03330).

Transforms with nonuniform frequency spacing, leading to varying time-frequency resolution, can better represent the tonal and transient characteristics of musical signals. [Frequency-warped transforms](http://elvera.nue.tu-berlin.de/typo3/files/1015Burred2006.pdf) such as the [constant-Q transform](https://arrow.tudublin.ie/cgi/viewcontent.cgi?article=1007&context=argart) have been used in music source separation systems to improve over the STFT.

The sliCQ transform, which is the realtime version of the Nonstationary Gabor Transform (NSGT), is a spectral transform that allows for arbitrary nonlinear frequency scales with perfect inversion. The following visual comparison of sliCQ transform (xumx-sliCQ default) vs. STFT (UMX default) on a 10s excerpt of music ([Mestis - El Mestizo](https://www.youtube.com/watch?v=0kn2doStfp4)) demonstrates the improved visual clarity of musical events:
![slicq_spectral](./docs/slicq_spectral.png)

## Results

**coming soon!**

I will show boxplots of the [BSS scores](https://github.com/sigsep/sigsep-mus-eval) on the full MUSDB18-HQ test set, similar to the [SiSec 2018 evaluation campaign](https://arxiv.org/abs/1804.06267). I will compare xumx-sliCQ to both the pretrained umxhq and pretrained x-umx models.

Here's an early teaser on a small handful of tracks to show how xumx-sliCQ is generally a few points of SDR worse than umx:

![early_boxplot](./docs/boxplot_teaser.png)

**N.B.** I have omitted the pre-trained x-umx model because I get abnormally low scores, so I probably have a bug in running the inference (using the [Sony x-umx code](https://github.com/sony/ai-research-code/tree/master/x-umx)). It will be fixed when I publish the full evaluation.

## Network architecture

The architecture diagram of xumx-sliCQ shows how closely it resembles Open-Unmix:

![xumx_system](./docs/xumx_slicq_system.png)

The ragged sliCQ is stored in a matrix with zero-padding to perform the Wiener EM step directly on the sliCQ transform ([adapting STFT Wiener EM to the sliCQ is discussed here](https://discourse.aicrowd.com/t/umx-iterative-wiener-expectation-maximization-for-non-stft-time-frequency-transforms/6191)).

A look into each of the 4 target networks of xumx-sliCQ shows how the convolutional network architecture is applied to the ragged sliCQ transform, where each block contains the frequency bins that share the same time-frequency resolution:

![xumx_pertarget](./docs/xumx_slicq_pertarget.png)

**N.B.** only two blocks are shown for illustrative purposes in the diagram, but the sliCQ used in the model has 262 frequency bins grouped into 70 time-frequency resolution blocks.

Each "Conv-Net" shown above is loosely based on the 2-layer convolutional denoising autoencoder architecture that can be seen in [Grais, Zhao, and Plumbley 2019](https://arxiv.org/abs/1910.09266). The encoder consists of 2x `Conv2d -> BatchNorm2d -> ReLU`, and the decoder consists of 2x `ConvTranspose2d -> BatchNorm2d -> ReLU`. The LSTM model of Open-Unmix did not produce good results in my experiments, and I had better luck with convolutional models.

The same kernel is used in both layers. The time and filter kernel sizes are chosen based on the number of frequency bins and time coefficients inside each block. Dilations are used in the time axis to increase the receptive field while keeping inference time low.

| Frequency bins per block | Frequency kernel size |
|----------------|------------------|
| nb_f < 10 | 1 |
| 10 <= nb_f < 20 | 3 |
| nb_f >= 20 | 5 |

| Time coefficients per block | Time kernel size |
|-----------------------------|------------------|
| nb_t <= 100 | 7, dilation=2 |
| nb_t > 100 | 13, dilation=2 |

The total number of learnable parameters is ~6.7 million:
```
===============================================================================================
Total params: 6,669,912
Trainable params: 6,669,912
Non-trainable params: 0
Total mult-adds (G): 194.27
===============================================================================================
Input size (MB): 28.63
Forward/backward pass size (MB): 9359.33
Params size (MB): 26.68
Estimated Total Size (MB): 9414.64
```

### sliCQ parameter search

The parameters of the sliCQ were chosen by a 60-iteration random parameter search using the "mix-phase oracle", where the ground truth magnitude sliCQ is combined with the mix phase to get a complex sliCQ. The result is inverted to the time domain to get the SDR of the waveform. 60 iterations are enough to give a statistically good combination of parameters in a large problem space according to [Bergstra and Bengio 2012](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf). The parameter search is described in more detail in [docs/slicq_params.md](./docs/slicq_params.md).

The configuration chosen for the xumx-sliCQ network uses the Bark scale with 262 bins, 32.9 - 22050 Hz, and slice and transition lengths of 18060 and 4514 samples (409 ms and 102 ms respectively). For a detailed look at the shape and nature of the sliCQ transform, as well as its noninvertible 50% overlap, look at [docs/slicq_shape.md](./docs/slicq_shape.md).

## Training and inference

The provided [conda yml file](./scripts/environment-gpu-linux-cuda11.yml) should install all the dependencies needed to train and run inference on xumx-sliCQ (using Python 3.9 with the Fedora 34 amd64 OS). xumx-sliCQ trains for 1000 epochs, like XUMX, with no early stopping patience. The best loss was -0.449, achieved at epoch 583. The [training script](./scripts/train.py) launches a tensorboard process in the background for training monitoring purposes:

![tboard_loss](./docs/tensorboard_loss.png)

The loss applies a mixing coefficient of 0.1 to the time domain SISDR loss, to bring it to a similar order of magnitude of the MSE loss of the sliCQ transform:

```
loss = 0.1*sisdr_loss + mse_loss
```

An epoch takes roughly 5.8 minutes to execute on an RTX 3080 Ti with batch_size=32 and nb_workers=4 (Ryzen 3700x). The same training ideas are used from [open-unmix](https://github.com/sigsep/open-unmix-pytorch/blob/master/docs/training.md):
* chunking with a seq_dur of 1s (the umx default of 6s makes the training prohibitively slow with 15+ minute epochs - on the other hand, >1s durations would have allowed for larger convolution kernels in the time direction)
* random track mixing (enabled explicitly with a flag, --random-track-mix, not hardcoded)
* balanced track sampling (same as UMX)
* gain and channelswap augmentations (same as UMX)

The pretrained model is [included in this repository](./pretrained-model). The weights are 28MB on disk (Linux), considerably smaller than umxhq (137MB) and x-umx (136MB).

## ISMIR 2021 Music Demixing Challenge

I worked on this model for the [ISMIR 2021 Music Demixing Challenge](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021). Out of my 31 submissions, all scored worse than the UMX and XUMX baselines, or timed out and failed to evaluate the hidden data within the specified time limit. The final model stopped being able to run under within competition time limit with the inclusion of Wiener-EM on the sliCQ ([commit dad295f](https://github.com/sevagh/xumx-sliCQ/commit/dad295fe54c3641a49813a29837a2e23445e0918)). In an attempt to speed up CPU inference, I tried a variety of strategies, including faster inference using ONNX, onnxruntime, and Caffe2 (which were slower than PyTorch), and numpy + numba jit in the Wiener EM step (again, slower than PyTorch).

The competition format was a great motivator, and it helped me create and refine xumx-sliCQ. Given the flexibility of the sliCQ transform, this model can be a good starting point for future improvements with different parameter search strategies or neural network architectures.

This published version is actually not the highest scoring xumx/umx-sliCQ variant that I submitted to the competition, but I chose it as my final model because:
* The code for selecting convolution layer parameters is less buggy and better understood than my previous submissions
* The model uses a single configuration of the sliCQ transform for all 4 models (enforced by the combination loss, since we must be able to sum the magnitude sliCQ coefficients for each target)

## History

I have two previous projects where I explored similar ideas:
* [Music-Separation-TF](https://github.com/sevagh/Music-Separation-TF), where I explored the MATLAB Wavelet Toolbox CQT (which is based on [the NSGT](https://www.mathworks.com/help/wavelet/ref/cqt.html)) and other time-frequency resolution ideas in harmonic/percussive/vocal source separation
* [MiXiN](https://github.com/sevagh/MiXiN), an early prototype deep learning model for music source separation based on [Grais and Plumbley 2017](https://arxiv.org/abs/1703.08019)'s Convolutional Denoising Autoencoder architecture, the reference [Python NSGT](https://github.com/grrrr/nsgt) library, Keras, and Tensorflow

Even earlier than that, my interest in source separation and demixing began with harmonic/percussive source separation:
* [Real-Time-HPSS](https://github.com/sevagh/Real-Time-HPSS), a realtime adaptation of [Fitzgerald 2010](https://arrow.tudublin.ie/cgi/viewcontent.cgi?article=1078&context=argcon)'s median filtering harmonic/percussive source separation algorithm
* [Zen](https://github.com/sevagh/Zen), a very fast C++ CUDA implementation of HPSS
