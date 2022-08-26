# xumx-sliCQ

A variant of [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) that replaces the STFT with the [sliCQ transform](https://github.com/sevagh/nsgt) ([Balazs et al. 2011](http://ltfat.org/notes/ltfatnote018.pdf) and [Holighaus et al. 2012](https://arxiv.org/abs/1210.0084)) and uses convolutional layers (based loosely on [Grais, Zhao, and Plumbley 2019](https://arxiv.org/abs/1910.09266)).

Transforms with nonuniform frequency spacing can better represent the tonal and transient characteristics of musical signals, such as the CQT. The sliCQ transform is one such transform:

<img src="./docs/slicq_spectral.png" width="70%"/>

xumx-sliCQ represents the first working demonstration of the sliCQ transform in a neural network for music demixing, but it failed to beat UMX or XUMX.

:information_source: The original README with more details is included [here](./docs/README-original.md)

## Inference

The pretrained model is [included in this repository](./pretrained-model). The weights are 28MB on disk (Linux), considerably smaller than umxhq (137MB) and x-umx (136MB). To separate an arbitrary wav file, install xumx_slicq (steps [here](./docs/README-original.md#training-and-inference)) and run:

```
$ python -m xumx_slicq.predict --outdir ./my-ests \
    --no-cuda --model ./pretrained-model/ --sr 44100 /path/to/song.wav
```

## Results

The following boxplot was generated like [SiSec 2018](https://github.com/sigsep/sigsep-mus-2018-analysis) to show the BSSv4 scores of UMXHQ vs. X-UMX vs. xumx-sliCQ (both configurations of Wiener EM) on the 50-track MUSDB18-HQ test set, alongside two oracles (IRM1 and MPI or mixed-phase oracle):

<img src="./docs/boxplot_full.png" width="70%"/>

### Example tracks

I have uploaded two examples to Soundcloud:
* [Action ft. KXNE by OnAir Music](https://soundcloud.com/user-167126026/sets/xumx-slicq-demo-action-ft-kxne-by-onair-music); genre: UK drill/trap
* [Marigold by Periphery](https://soundcloud.com/user-167126026/sets/xumx-slicq-demo-marigold-by-periphery); genre: progressive metal

## Citation

The paper associated with xumx-sliCQ, "Music demixing with the sliCQ transform," was originally published at [MDX21](https://mdx-workshop.github.io) @ ISMIR 2021 ([pdf](https://mdx-workshop.github.io/proceedings/hanssian.pdf)), and later on arXiv with some minor updates:

```
@inproceedings{xumxslicqmdx21,
        title={Music demixing with the sliCQ transform},
        author={Hanssian, Sevag},
        booktitle={MDX21 workshop at ISMIR 2021},
        year={2021}}

@article{xumxslicqarxiv,
        title={Music demixing with the sliCQ transform},
        author={Hanssian, Sevag},
        journal={arXiv preprint arXiv:2112.05509},
        url={https://arxiv.org/abs/2112.05509},
        year={2021}}
```

## Network architecture

xumx-sliCQ, single target:

<img src="./docs/xumx_slicq_pertarget.png" width="75%"/>

**N.B.** only two blocks are shown for illustrative purposes in the diagram, but the sliCQ used in the model has 262 frequency bins grouped into 70 time-frequency resolution blocks.

## ISMIR 2021 Music Demixing Challenge

I worked on this model for the [ISMIR 2021 Music Demixing Challenge](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021).

The competition format was a great motivator, and it helped me create and refine xumx-sliCQ. Given the flexibility of the sliCQ transform, this model can be a good starting point for future improvements with different parameter search strategies or neural network architectures.
