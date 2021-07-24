# xumx-sliCQ

My variant of the excellent [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) (aka UMX) template for music source separation ([Stöter, Uhlich, Liutkus, Mitsufuji 2019](https://hal.inria.fr/hal-02293689/document)). This is a music source separation system which, given a mixed song input, estimates 4 targets (drums, bass, vocals, other), as defined by [MUSDB18-HQ](https://zenodo.org/record/3338373) dataset. It differs from open-unmix-pytorch in the following ways:
* The spectral transform is the [sliCQ transform](https://github.com/sevagh/nsgt) ([Balazs et al. 2011](http://ltfat.org/notes/ltfatnote018.pdf) and [Dörfler et al. 2014](https://www.univie.ac.at/nonstatgab/cqt/index.php)) vs. the STFT
* Convolutional architecture (based loosely on [Grais, Zhao, and Plumbley 2019](https://arxiv.org/abs/1910.09266)) instead of the UMX linear encoder + LSTM + decoder
* Single network like [CrossNet-Open-Unmix](https://github.com/JeffreyCA/spleeterweb-xumx) ([Sawata, Uhlich, Takahashi, Mitsufuji 2020](https://www.ismir2020.net/assets/img/virtual-booth-sonycsl/cUMX_paper.pdf)), aka X-UMX
    * Combination loss (CL) - loss function on different linear combinations of the 4 targets
    * Multi-domain loss (MDL) - frequency-domain loss (MSE) and time-domain loss ([auraloss](https://github.com/csteinmetz1/auraloss) SI-SDR)

I trained this model for the [ISMIR 2021 Music Demixing Challenge](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021), with the following results:
* :heavy_check_mark: It is the first successful use of the sliCQ transform in a neural network for music source separation
* :heavy_multiplication_x: It failed to surpass the scores of UMX or XUMX in the hidden test set (placing lower than #30 in the leaderboard)

Given the flexiblity of the sliCQ transform, I still believe the idea can be explored further and that higher performance can be obtained by designing a better neural network architecture or choosing different sliCQ parameters.

## Motivation

The STFT is very useful in audio and music applications, but has a constant time-frequency resolution that depends on the window size. In practice this means the STFT can only represent a linear frequency scale with constant time resolution. This also means that time and frequency resolution must be traded off, and one size does not fit all: [paper 1](https://arxiv.org/abs/1504.07372), [paper 2](https://arxiv.org/abs/1905.03330).

Nonlinear frequency transforms can allow for an adaptive time-frequency resolution that better represents the tonal and transient characteristics of musical signals. [Frequency-warped transforms](http://elvera.nue.tu-berlin.de/typo3/files/1015Burred2006.pdf) such as the [constant-Q transform](https://arrow.tudublin.ie/cgi/viewcontent.cgi?article=1007&context=argart) have been used in music source separation systems to improve over the STFT.

The sliCQ transform (realtime/sliding window version of the Nonstationary Gabor Transform) is a spectral transform that allows for arbitrary nonlinear frequency scales with perfect inversion. A perfectly invertible constant-Q transform is one of the important applications of the sliCQ transform, but other frequency scales can be designed (including Mel, Bark, ERBlets, etc.). See [this GitHub issue on essentia](https://github.com/MTG/essentia/issues/136#issuecomment-259153939) for a small summary on the history of constant-Q transform implementations. The code for the PyTorch sliCQ transform is stored in this repository (at the path `openunmix/nsgt/`) but the standalone library is published separately [here](https://github.com/sevagh/nsgt).

The following visual comparison of sliCQ transform (xumx-sliCQ default) vs. STFT (UMX default) on a 10s excerpt of music ([Mestis - El Mestizo](https://www.youtube.com/watch?v=0kn2doStfp4)) demonstrates the improved visual clarity of the tonal and transient sounds:
![slicq_spectral](./.github/slicq_spectral.png)

My source separation hypothesis is based on the above spectrograms - given that the sliCQ transform can represent music with more clarity due to its adaptive time-frequency resolution, it is worth exploring to use in a neural network for music source separation.

## Block diagram of the system

## History

[Grais and Plumbley 2017](https://arxiv.org/abs/1703.08019)

Different variants and experiments are all stored in my [umx-experiments](https://gitlab.com/sevagh/umx-experiments/-/tree/master) repo.

This published version is actually not the highest scorer that I submitted to the competition, but I chose it as my final model because:
* The code for selecting convolution layer parameters is less buggy
* The model uses a single configuration of the sliCQ transform for all 4 models (enforced by the combination loss, since we must be able to sum the magnitude sliCQ coefficients for each target)
