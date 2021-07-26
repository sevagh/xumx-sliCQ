# xumx-sliCQ

My variant of the excellent [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) (aka UMX) template for music source separation ([Stöter, Uhlich, Liutkus, Mitsufuji 2019](https://hal.inria.fr/hal-02293689/document)). This is a music source separation system which, given a mixed song input, estimates 4 target sources (drums, bass, vocals, other), as defined by [MUSDB18-HQ](https://zenodo.org/record/3338373) dataset. It differs from open-unmix-pytorch in the following ways:
* The spectral transform is the [sliCQ transform](https://github.com/sevagh/nsgt) ([Balazs et al. 2011](http://ltfat.org/notes/ltfatnote018.pdf) and [Dörfler et al. 2014](https://www.univie.ac.at/nonstatgab/cqt/index.php)) vs. the STFT
* Convolutional architecture (based loosely on [Grais, Zhao, and Plumbley 2019](https://arxiv.org/abs/1910.09266)) instead of the UMX linear encoder + LSTM + decoder
* Single network like [CrossNet-Open-Unmix](https://github.com/JeffreyCA/spleeterweb-xumx) ([Sawata, Uhlich, Takahashi, Mitsufuji 2020](https://www.ismir2020.net/assets/img/virtual-booth-sonycsl/cUMX_paper.pdf)), aka X-UMX
    * Combination loss (CL) - loss function on different linear combinations of the 4 targets
    * Multi-domain loss (MDL) - frequency-domain loss (MSE) and time-domain loss ([auraloss](https://github.com/csteinmetz1/auraloss) SI-SDR)

I trained this model for the [ISMIR 2021 Music Demixing Challenge](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021). The results showed that:
* (:heavy_check_mark:) the sliCQ transform could successfully be used in a neural network for music source separation
* (:heavy_multiplication_x:) the xumx-sliCQ model could not beat the UMX or XUMX baselines in the hidden test set (placing lower than #30 in the leaderboard)

Given the flexiblity of the sliCQ transform, I still believe the idea can be explored and improved further with a better neural network architecture or different sliCQ parameters.

## Motivation

Time-frequency masking is one strategy for music source separation, where the magnitude spectrogram of the mix is multiplied by an estimated target mask ([read more](https://source-separation.github.io/tutorial/basics/tf_and_masking.html)). Open-Unmix uses the short-time Fourier transform (STFT) for the spectral representation of music, and learns to estimate the magnitude STFT of a target from the mixture. The STFT is useful in audio and music applications, but it has a uniform and fixed frequency and time resolution controlled by the window size, where one size does not fit all: [paper 1](https://arxiv.org/abs/1504.07372), [paper 2](https://arxiv.org/abs/1905.03330).

Transforms with nonuniform frequency spacing, leading to varying time-frequency resolution, can better represent the tonal and transient characteristics of musical signals. [Frequency-warped transforms](http://elvera.nue.tu-berlin.de/typo3/files/1015Burred2006.pdf) such as the [constant-Q transform](https://arrow.tudublin.ie/cgi/viewcontent.cgi?article=1007&context=argart) have been used in music source separation systems to improve over the STFT.

The sliCQ transform, which is the realtime version of the Nonstationary Gabor Transform (NSGT), is a spectral transform that allows for arbitrary nonlinear frequency scales with perfect inversion. The following visual comparison of sliCQ transform (xumx-sliCQ default) vs. STFT (UMX default) on a 10s excerpt of music ([Mestis - El Mestizo](https://www.youtube.com/watch?v=0kn2doStfp4)) demonstrates the improved visual clarity of musical events:
![slicq_spectral](./docs/slicq_spectral.png)

My source separation hypothesis is based on the above spectrograms - given that the sliCQ transform can represent music with more clarity due to its adaptive time-frequency resolution, it is worth exploring in music source separation.

### sliCQ hyperparameter search

The parameters of the sliCQ were chosen by a 60-iteration random parameter search using the "mix-phase oracle", where the ground truth magnitude sliCQ is combined with the mix phase to get a complex sliCQ and invert it to the time domain target waveform with the highest SDR. 60 iterations are enough to give a statistically good combination of parameters in a large problem space according to [Bergstra and Bengio 2012](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf).

The parameter search is described in more detail in [docs/slicq_params.md](./docs/slicq_params.md). The configuration chosen for the xumx-sliCQ network uses the Bark scale with 262 bins, 32.9 - 22050 Hz, and slice and transition lengths of 18060 and 4514 samples (409 ms and 102 ms respectively).

## Network architecture

The architecture diagram of xumx-sliCQ shows how closely it resembles Open-Unmix:

![xumx_system](./docs/xumx_slicq_system.png)

There is an option to perform the Wiener EM step directly on the sliCQ (`stft_wiener=False` in the Separator class), but the execution time is much slower than the STFT-based Wiener EM, for a negligible boost in SDR. In practice, 1 iteration of STFT-based Wiener EM gives a modest performance boost with an acceptable performance penalty ([further discussed here](https://discourse.aicrowd.com/t/umx-iterative-wiener-expectation-maximization-for-non-stft-time-frequency-transforms/6191)).

A look into each of the 4 target networks of xumx-sliCQ shows how the convolutional network architecture is applied per-block of the ragged sliCQ transform, where each block contains the frequency bins that share the same time resolution:

![xumx_pertarget](./docs/xumx_slicq_pertarget.png)

For simplicity, 6 frequency bins (0-5) grouped into 2 time-frequency blocks are drawn above. The real sliCQ used in xumx-sliCQ has 262 frequency bins grouped into 70 time-frequency blocks, but the idea is the same.

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

### Bandwidth

In Open-Unmix, frequency bins above 16000 Hz are not learned by the network. In xumx-sliCQ, the same thing is done. If the starting frequency of a block of frequency bins is above the 16000 Hz bandwidth, the entire block passes through the network unchanged. This cuts down the total number of learnable parameters and allows xumx-sliCQ to evaluate the ISMIR 2021 Music Demixing Challenge hidden test set without timing out.

## Training and inference

## Performance and results

## History

I have two previous projects where I explored similar ideas:
* [Music-Separation-TF](https://github.com/sevagh/Music-Separation-TF), where I explored the MATLAB Wavelet Toolbox CQT (which is based on [the NSGT](https://www.mathworks.com/help/wavelet/ref/cqt.html)) and other time-frequency resolution ideas in harmonic/percussive/vocal source separation
* [MiXiN](https://github.com/sevagh/MiXiN), an early prototype deep learning model for music source separation based on [Grais and Plumbley 2017](https://arxiv.org/abs/1703.08019)'s Convolutional Denoising Autoencoder architecture, the reference [Python NSGT](https://github.com/grrrr/nsgt) library, Keras, and Tensorflow

This published version is actually not the highest scoring xumx/umx-sliCQ variant that I submitted to the competition, but I chose it as my final model because:
* The code for selecting convolution layer parameters is less buggy and better understood than my previous submissions
* The model uses a single configuration of the sliCQ transform for all 4 models (enforced by the combination loss, since we must be able to sum the magnitude sliCQ coefficients for each target)

Historical snapshots of different variants and experiments of umx-sliCQ/xumx-sliCQ are all stored in my [umx-experiments](https://gitlab.com/sevagh/umx-experiments) repository - I tried many ideas over the course of the competition, including Conv-LSTM architectures, 3D convolutions, and more.
