# xumx-sliCQ

My variant of the excellent [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) (aka UMX) template for music source separation ([Stöter, Uhlich, Liutkus, Mitsufuji 2019](https://hal.inria.fr/hal-02293689/document)). This is a music source separation system which, given a mixed song input, estimates 4 targets (drums, bass, vocals, other), as defined by [MUSDB18-HQ](https://zenodo.org/record/3338373) dataset. It differs from open-unmix-pytorch in the following ways:
* The spectral transform is the [sliCQ transform](https://github.com/sevagh/nsgt) ([Balazs et al. 2011](http://ltfat.org/notes/ltfatnote018.pdf) and [Dörfler et al. 2014](https://www.univie.ac.at/nonstatgab/cqt/index.php)) vs. the STFT
* Convolutional architecture (based loosely on [Grais, Zhao, and Plumbley 2019](https://arxiv.org/abs/1910.09266)) instead of the UMX linear encoder + LSTM + decoder
* Single network like [CrossNet-Open-Unmix](https://github.com/JeffreyCA/spleeterweb-xumx) ([Sawata, Uhlich, Takahashi, Mitsufuji 2020](https://www.ismir2020.net/assets/img/virtual-booth-sonycsl/cUMX_paper.pdf)), aka X-UMX
    * Combination loss (CL) - loss function on different linear combinations of the 4 targets
    * Multi-domain loss (MDL) - frequency-domain loss (MSE) and time-domain loss ([auraloss](https://github.com/csteinmetz1/auraloss) SI-SDR)

I trained this model for the [ISMIR 2021 Music Demixing Challenge](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021). The results showed that:
* (:heavy_check_mark:) the sliCQ transform could successfully be used in a neural network for music source separation
* (:heavy_multiplication_x:) ultimately it did not beat the UMX or XUMX baselines in the hidden test set (placing lower than #30 in the leaderboard)

Given the flexiblity of the sliCQ transform, I still believe the idea can be explored and improved further with a better neural network architecture or different sliCQ parameters.

## Motivation

Time-frequency masking is one strategy for music source separation, where the magnitude spectrogram of the mix is multiplied by an estimated target mask ([read more](https://source-separation.github.io/tutorial/basics/tf_and_masking.html)). Open-Unmix uses the short-time Fourier transform (STFT) for the spectral representation of music, and learns to estimate the magnitude STFT of a target from the mixture. The STFT is useful in audio and music applications, but it has a uniform and fixed frequency and time resolution controlled by the window size, where one size does not fit all: [paper 1](https://arxiv.org/abs/1504.07372), [paper 2](https://arxiv.org/abs/1905.03330).

Transforms with nonuniform frequency spacing, leading to varied time-frequency resolution, can better represent the tonal and transient characteristics of musical signals. [Frequency-warped transforms](http://elvera.nue.tu-berlin.de/typo3/files/1015Burred2006.pdf) such as the [constant-Q transform](https://arrow.tudublin.ie/cgi/viewcontent.cgi?article=1007&context=argart) have been used in music source separation systems to improve over the STFT.

The sliCQ transform, which is the realtime/sliding window version of the Nonstationary Gabor Transform (NSGT), is a spectral transform that allows for arbitrary nonlinear frequency scales with perfect inversion. The following visual comparison of sliCQ transform (xumx-sliCQ default) vs. STFT (UMX default) on a 10s excerpt of music ([Mestis - El Mestizo](https://www.youtube.com/watch?v=0kn2doStfp4)) demonstrates the improved visual clarity of musical events:
![slicq_spectral](./.github/slicq_spectral.png)

My source separation hypothesis is based on the above spectrograms - given that the sliCQ transform can represent music with more clarity due to its adaptive time-frequency resolution, it is worth exploring to use in a neural network for music source separation.

### sliCQ hyperparameter search

The parameters of the sliCQ were chosen by a 60-iteration random parameter search using the "mix-phase oracle", where given a perfect (aka ground truth) magnitude sliCQ estimate, using the phase of the mix sliCQ to get the time domain target waveform resulted in the highest SDR score. 60 iterations are enough to give a statistically good combination of parameters in a large problem space according to [Bergstra and Bengio 2012](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf).

The parameter search is described in more detail in [docs/slicq_params.md](./docs/slicq_params.md). The configuration chosen for the xumx-sliCQ network uses the Bark scale with 262 bins, 32.9 - 22050 Hz, and slice and transition lengths of 18060 and 4514 samples (409 ms and 102 ms respectively)

This resulted in a theoretical maximum SDR performance of **8.84 dB** for all 4 targets on the validation set of MUSDB18-HQ (from the initial estimates, barring further multi-channel iterative Wiener filtering). Compare this to the STFT (UMX defaults: window = 4096, overlap = 1024), which achieves **8.56 dB**.

## Block diagrams

## Training and inference

## Performance and results

## History

I have two previous projects where I explored similar ideas:
* [Music-Separation-TF](https://github.com/sevagh/Music-Separation-TF), where I explored the MATLAB Wavelet Toolbox CQT (which is based on [the NSGT](https://www.mathworks.com/help/wavelet/ref/cqt.html)) and other time-frequency resolution ideas in harmonic/percussive/vocal source separation
* [MiXiN](https://github.com/sevagh/MiXiN), an early prototype deep learning model for music source separation based on [Grais and Plumbley 2017](https://arxiv.org/abs/1703.08019)'s Convolutional Denoising Autoencoder architecture, the reference [Python NSGT](https://github.com/grrrr/nsgt) library, Keras, and Tensorflow

This published version is actually not the highest scorer that I submitted to the competition, but I chose it as my final model because:
* The code for selecting convolution layer parameters is less buggy
* The model uses a single configuration of the sliCQ transform for all 4 models (enforced by the combination loss, since we must be able to sum the magnitude sliCQ coefficients for each target)

Historical snapshots of different variants and experiments of umx-sliCQ/xumx-sliCQ are all stored in my [umx-experiments](https://gitlab.com/sevagh/umx-experiments) repository.
