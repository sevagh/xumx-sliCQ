---
title: 'Better music demixing with the sliCQ transform'
tags:
  - separation
  - cqt
  - time-frequency
authors:
  - name: Sevag Hanssian
    affiliation: "1"
affiliations:
 - name: McGill University
   index: 1
date: 19 September 2021
bibliography: paper.bib
---
<!--
Your report should include:
    abstract
    introduction
    sections on experimental setup/methodology including system information and model/network architecture
    evaluation/results
    discussion
    conclusion
    references.
Please provide an estimation of the computational resources needed. You must describe any external data and pre-existing tools, software and models used.
-->

# Abstract

Music source separation, or music demixing, is the task of decomposing a song into its constituent sources, which are typically isolated instruments (e.g., drums, bass, and vocals). Open-Unmix (UMX) [@umx], and the improved variant CrossNet-Open-Unmix (X-UMX) [@xumx], are high-performing models that use Short-Time Fourier Transform (STFT) as the representation of music signals, and apply masks to the magnitude STFT to separate mixed music into four sources: vocals, drums, bass, and other.

The time-frequency uncertainty principle states that the STFT of a signal cannot be maximally precise in both time and frequency [@gabor1946]. The tradeoff in time-frequency resolution can significantly affect music demixing results [@tftradeoff1]. Our first adaptation of UMX, xumx-sliCQ^[<https://github.com/sevagh/xumx-sliCQ>] [@hanssian21], submitted to the 2021 Music Demixing Challenge [@mdx21], replaced the STFT with the sliCQT [@slicq], a time-frequency transform with varying time-frequency resolution. xumx-sliCQ achieved a score of 3.6 dB. For the Cadenza Challenge in 2023, we submitted an improved xumx-sliCQ-V2 which scored 4.4 dB.

# Introduction

The STFT is computed by applying the Discrete Fourier Transform on fixed-size windows of the input signal. From both auditory and musical motivations, variable-size windows are preferred, with long windows in low-frequency regions to capture detailed harmonic information with a high frequency resolution, and short windows in high-frequency regions to capture transients with a high time resolution [@doerflerphd]. The sliCQ Transform (sliCQT) [@slicq] is a realtime variant of the Nonstationary Gabor Transform (NSGT) [@balazs] for variable-length input signals. These are time-frequency transforms with complex Fourier coefficients and perfect inverses that use varying windows to achieve nonlinear time or frequency resolution. An example application of the NSGT/sliCQT is an invertible Constant-Q Transform (CQT) [@jbrown].

# Methodology

In xumx-sliCQ, the sliCQT parameters of 262 bins on the Bark scale between 32.9--22050 Hz were chosen in a random parameter search [@hanssian21]. STFT and sliCQT spectrograms of a glockenspiel signal^[<https://github.com/ltfat/ltfat/blob/master/signals/gspi.wav>] are shown in \autoref{fig:spectrograms}.

![STFT and sliCQT spectrograms of the musical glockenspiel signal.\label{fig:spectrograms}](https://raw.githubusercontent.com/sevagh/mdx-submissions21/HANSSIAN/static-assets/spectrograms_comparison.png){ width=95% }

The STFT outputs a single time-frequency matrix where all of the frequency bins are spaced uniformly apart and have the same time resolution. The sliCQT groups frequency bins, which may be nonuniformly spaced, in a ragged list of time-frequency matrices, where each matrix contains frequency bins that share the same time resolution. In xumx-sliCQ, convolutional layers adapted from an STFT-based vocal separation model [@plumbley2] were applied separately to each time-frequency matrix, shown in \autoref{fig:ragged}.

![Example of convolutional layers applied to a ragged sliCQT.\label{fig:ragged}](https://raw.githubusercontent.com/sevagh/mdx-submissions21/HANSSIAN/static-assets/xumx_slicq_pertarget.png){ width=100% }

We made three significant changes to xumx-sliCQ which combined account for the improved performance of xumx-sliCQ-V2.

1. Replacing the overlap-add with pure convolution layers

The NSGT processes the entire input signal at once. In cases where the input signal has an arbitrary length, such as realtime streaming, the sliCQT applies slicing windows to the input signal computes the NSGT per slice. The slicing operation is shown in Figure \ref{fig:slicqtukeys}. The slicing windows are ``symmetrically zero-padded to length $2N$'' \parencite[10]{slicq}, such that adjacent slices need to be 50\% overlap-added with each other. There is no inverse operation for the 50\% overlap-add procedure given in the paper. Figure \ref{fig:slicqoverlaps} demonstrates this characteristic of the sliCQT.

In xumx-sliCQ, we applied the overlap-add before the convolutional layers, and added a final transpose convolution layer to grow the coefficients by two to reverse the 50\% overlap-add. In xumx-sliCQ-V2, we didn't use the overlap-add function, but instead incorporated the slice size in the kernel and stride of the first convolution layer and last transpose convolution layer, shown in Figure \todo:

\begin{figure}[ht]
	\centering
	\includegraphics[width=0.5156\textwidth]{./images-misc/slicq_windows.png}
	\caption{Slicing the input signal with 50\% overlapping Tukey windows. N is the slice length and M is the transition area \parencite{slicq}.}
	\label{fig:slicqtukeys}
\end{figure}

2. Applying differentiable Wiener-EM filtering in the network and using complex MSE loss

In xumx-sliCQ, the output of the neural network is the magnitude sliCQT, and the real MSE loss is computed from the real-valued magnitude sliCQT coefficients. In xumx-sliCQ-V2, we apply differentiable Wiener-EM filtering to further separate the four (VDBO) targets in the neural network, such that the output is the complex sliCQT. We replace the magnitude MSE loss with complex MSE loss, inspired by Danna-Sep \todo cite this.

3. Adding a new loss term: mask-sum loss

In music demixing systems that apply masks to magnitude spectral coefficients, the mixture is considered to be a linear sum of the four targets. is that a sum of the four target magnitude spectrograms must equal to the magnitude spectrogram of the input mixture:

\todo{math formula sum here}

For this reason, we use a Sigmoid as a final activation layer in both xumx-sliCQ and xumx-sliCQ-V2, to output a soft mask (in [0, 1]). In xumx-sliCQ-V2, we added a new loss term, the mask sum loss, which sums the intermediate magnitude mask of the four targets, and computes the MSE from a matrix of ones.

# Results

Our model, xumx-sliCQ-V2, was trained on MUSDB18-HQ. On the test set, xumx-sliCQ-V2 achieved a total SDR of 4.4 dB versus the 4.64 dB of UMX and 5.54 dB of X-UMX, performing worse than the original STFT-based models, but better than the first xumx-sliCQ. The overall system architecture of xumx-sliCQ-V2 is similar to X-UMX, shown in \autoref{fig:blockdiagram}.

![xumx-sliCQ overall system diagram.\label{fig:blockdiagram}](https://raw.githubusercontent.com/sevagh/mdx-submissions21/HANSSIAN/static-assets/xumx_overall_arch.png){ width=100% }

# Discussion

The Cadenza Challenge Task 1 applies further processing on top of the VDBO demixing system, and the final evaluation uses the HAAQI metric. We chose to focus on the VDBO music demixing problem, and used the traditional BSS metrics (and SDR) to measure the improvement of xumx-sliCQ-V2 over xumx-sliCQ.

Some of the improvements made in xumx-sliCQ-V2 can be generalized to any magnitude spectrogram masking approach: applying Wiener-EM and complex MSE loss on complex Fourier coefficients, and adding a mask-sum loss term to enforce the linear additive mixture assumption.

# Conclusion

We presented xumx-sliCQ-V2, an improved variant of xumx-sliCQ. These systems propose to replace the STFT of X-UMX with a Bark-scale sliCQT, a nonuniform time-frequency transform whose characteristics more closely map to the human auditory system. The total SDR went from 3.6 dB to 4.4 dB, demonstrating significantly improved music demixing results from the changes in xumx-sliCQ-V2.

We also point out that the weights of xumx-sliCQ-V2 are 60 MB in size, less than half the size of X-UMX. It was designed for fast inference with simple convolutional layers, which might be useful on smaller embedded devices.

\newpage

# References
