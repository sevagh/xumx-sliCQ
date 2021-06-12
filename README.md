# umx-sliCQ

My variant of the excellent [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) template for source separation.

This is a music source separation system, using 4 trained models for the 4 targets of MUSDB18-HQ. The key differences from Open-Unmix are the use of the sliCQ-NSGT transform instead of the NSGT, and a convolutional denoising autoencoder architecture (e.g. plumbley1, 2) instead of the linear encoder + LSTM + decoder.

I submitted this model to the AICrowd ISMIR 2021 [Music Demixing Challenge](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021).
