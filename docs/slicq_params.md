# The best sliCQ for music source separation

## Intro

The parameters of the sliCQ are:
* Frequency scale: Bark, Mel, Log (aka Constant-Q or the Western pitch scale), Variable-Q. The octave scale is a wrapper on top of the Log scale, where the bins argument becomes "bins per octave" - I prefer direct control of total bins through the Log scale.
* Fbins: the total number of bins on the frequency scale. Using the Western pitch scale and octave analogy, a lot of CQT literature is focused on multiples of 12 (12 pitches per octave): 12 bins, 24 bins, 84 bins, 96 bins, etc. In this application, I don't make any assumptions and try to find any number between 10-300. 300 is capped along with the slice length to keep a small transform that fits in GPU memory
* Fmin: between 0 Hz (the DC) and 22050 Hz (the Nyquist, assuming a sampling rate of 44100 Hz, that of MUSDB18-HQ), the starting frequency of the frequency scale must be chosen. **n.b.** the sliCQ transform will automatically prepend the DC
* Slice length: the length of the slice (sliCQ equivalent of window size). This is related to the frequency scale and bins chosen - with a large number of bins and a wide frequency scale, one must choose an appropriately large slice length to support the required Q factor per bin. The code to support this is in the embedded [NSGT](https://github.com/sevagh/xumx-sliCQ/blob/develop/openunmix/nsgt/fscale.py#L36) library.

Some parameters were fixed:
* Fmax: between 0 Hz (DC) and 22050 Hz (Nyquist). The sliCQ automatically appends the Nyquist if fmax is lower - for performance reasons, fmax is fixed to the Nyquist
* Transition length: the maximum value of the variable overlap, or hop, between consecutive slices - the sliCQ transform will vary this (up to the max) as needed to meet the time-frequency resolution needs of the input frequency scale. This is [automatically chosen](https://github.com/sevagh/xumx-sliCQ/blob/develop/openunmix/transforms.py#L79-L88) based on the slice length, and shouldn't concern the user

## Mix-phase oracle

The strategy of Open-Unmix is to obtain a first estimate of the target waveform by combining the estimated magnitude STFT and phase of the mix STFT:

```
x = mixed_audio

X = stft(x)
Xmag = abs(X)
Xphase = angle(X)

# model trained for the vocals target
Ymag_vocals_pred = unmix_vocals(Xmag)

Y_vocals_pred = Xphase * Ymag_vocals_pred

y_vocals_pred = istft(Y_vocals)
```
I discussed this idea in two places: [open-unmix-pytorch GitHub issues](https://github.com/sigsep/open-unmix-pytorch/issues/83) and the [AICrowd ISMIR 2021 Music Demixing Challenge discussion](https://source-separation.github.io/tutorial/basics/tf_and_masking.html). I also did a small writeup in a separate repo [here](https://github.com/sevagh/mss-oracle-experiments#oracle-performance-of-mpi-mix-phase-inversion). Open-Unmix performs further separation with multi-channel Wiener filtering after obtaining the initial 4 target estimates, but let's ignore that step for now.

With the sliCQ transform, we can search for the top-performing configuration by testing various parameter combinations and testing the SDR of the mix-phase oracle:

```
# given target_gts aka ground truths
def slicq_optimize_sigsep(mixed_audio, target_gts, fscale, fbins, fmin, fmax, sllen, trlen):
    total_sdr = 0.0

    for target_gt in target_gts:
        x = mixed_audio

        X = sliCQ(x, fscale, fbins, fmin, fmax, sllen, trlen)

        Xmag = abs(X)
        Xphase = angle(X)

        Y_target_gt = sliCQ(target_gt, fscale, fbins, fmin, fmax, sllen, trlen)

        Ymag_target_gt = abs(Y_target_gt)

        Y_target_mixphase = Xphase * Ymag_target_gt

        y_target_mixphase = istft(Y_target_mixphase)

        target_sdr = calculate_sdr(y_target_mixphase, target_gt)
        total_sdr += target_sdr

    # divide by 4 targets for "average" song SDR
    return total_sdr/4.
```

Note that this doesn't try to find which sliCQ is best _inside_ the neural network. It simply checks the highest SDR performance _if_ we had a perfect estimate of the magnitude sliCQ from the neural network.

Various constraints were discovered during the experimentation procedure, such as:
* Maximum slice length, which is the sliCQ equivalent of the STFT window size
* Frequency bins, since more frequency bins result in a larger transform

## Script

The script for the random parameter search is at scripts/slicq_explore.py.

Previously I also tried to use [BayesianOptimization](https://github.com/fmfn/BayesianOptimization), but preferred the random search as it was more intuitive to me (and I wasn't sure whether the sliCQ parameters met the criteria required to be optimized using the Bayesian process).

## Chosen config

```
(umx-gpu) sevagh:umx-sliCQ $ python scripts/slicq_explore.py --max-sllen=44100 --bins=10,300 --n-iter=60 --random-seed=42 --cuda-device=0
total:  8.84532519401798        ('bark', 262, 32.89999999999992, 95.10000000000001, 18060)
```

This resulted in a theoretical maximum SDR performance of **8.84 dB** for all 4 targets on the validation set of MUSDB18-HQ (from the initial estimates, ignoring further refinement through multi-channel iterative Wiener filtering). Compare this to the STFT (UMX defaults: window = 4096, overlap = 1024), which achieves **8.56 dB**.

## Historical log of sliCQ parameter searches

### 2021-07-17

revisit:

```
(umx-gpu) sevagh:umx-sliCQ $ python scripts/slicq_explore.py --max-sllen=44100 --bins=10,300 --n-iter=60 --random-seed=42 --cuda-device=0
total:  8.84532519401798        ('bark', 262, 32.89999999999992, 95.10000000000001, 18060)
```

### 2021-07-01

seed 2, 115:

```
best scores
bass:   8.155295771391716       ('mel', 798, 20.099999999999966, 97.9, 56544)
drums:  8.608401765820433       ('bark', 223, 68.4999999999998, 85.5, 15504)
other:  8.384228831830832       ('bark', 726, 57.99999999999983, 84.0, 50504)
vocals:         10.552040930665541      ('bark', 726, 57.99999999999983, 84.0, 50504)
```

seed 1, 42:

```
$ python scripts/slicq_explore.py --max-sllen=264600 --bins=10,2000 --n-iter=60 --random-seed=42 --per-target
best scores
bass:   8.127697350005228       ('bark', 569, 56.799999999999834, 6.9, 39556)
drums:  8.53666595713493        ('mel', 397, 43.39999999999988, 47.300000000000004, 27468)
other:  8.402055252314938       ('bark', 569, 56.799999999999834, 6.9, 39556)
vocals:         10.653934376348287      ('bark', 569, 56.799999999999834, 6.9, 39556)
```

vs.

```
Control score tot, bass, drums, vocals, other:
        8.56    7.37    8.42    10.37   8.08
```

### single target

```
best scores
total:  8.752033272285418       ('bark', array([850]), 30.09999999999993, 45.300000000000004, 58696)
```

```
best scores
total:  8.30844429398954        ('bark', array([281]), 14.499999999999984, 32.4, 19260)

```

### sllen <= 264600 (giant 6s slices)

seed 1:

```
$ python scripts/slicq_explore.py --seq-dur-min=6 --seq-dur-max=6 --seq-reps=12 --random-seed=35 --n-iter=60 --sllen=264600  --per-target --fscale='mel,bark' --bins="100,2000,1" --max-sllen=264600
best scores
bass:   7.805986587733148       ('bark', array([669]), 49.399999999999864, 82.80000000000001, 46436)
drums:  8.24299856283888        ('mel', array([102]), 107.39999999999965, 96.10000000000001, 6612)
other:  12.288555524165337      ('mel', array([515]), 44.99999999999987, 67.4, 35600)
vocals:         8.480903132587716       ('bark', array([671]), 41.899999999999885, 79.7, 46480)
```

seed 2:
```
$ python scripts/slicq_explore.py --seq-dur-min=6 --seq-dur-max=6 --seq-reps=12 --random-seed=42 --n-iter=60 --sllen=264600  --per-target --fscale='mel,bark' --bins="100,2000,1" --max-sllen=264600
best scores
bass:   8.274528281827637       ('bark', array([810]), 30.09999999999993, 45.300000000000004, 55932)
drums:  9.272527113677704       ('bark', array([1652]), 64.9999999999998, 12.200000000000001, 115192)
other:  11.385197682309528      ('bark', array([1652]), 64.9999999999998, 12.200000000000001, 115192)
vocals:         8.704854496436832       ('bark', array([898]), 33.89999999999991, 19.5, 62084)
```

### sllen <= 8192

2 runs with 2 random seeds:

```
$ python scripts/hyperparam_search.py  --seq-dur-min=5 --seq-dur-max=10 --seq-reps=5 --n-random-tracks=12 --fscale='mel,bark' --max-sllen=8192 --n-iter=1000 --random-seed=35
...
best scores
bass:   8.471699713709183       ('bark', 105, 25.399999999999945, 92.7, 7180)
drums:  10.50357175853227       ('mel', 104, 49.29999999999986, 18.8, 7108)
other:  13.329234187804468      ('bark', 35, 105.19999999999966, 7.2, 2392)
vocals:         9.831030922924944     ('mel', 116, 37.6999999999999, 12.4, 8024)
```

```
$ python scripts/hyperparam_search.py  --seq-dur-min=5 --seq-dur-max=10 --seq-reps=5 --n-random-tracks=12 --fscale='mel,bark' --max-sllen=8192 --n-iter=1000 --random-seed=1337
best scores
bass:   8.382918956174112       ('mel', 119, 128.1999999999996, 18.2, 7588)
drums:  10.345597348013385      ('mel', 113, 91.59999999999971, 21.200000000000003, 7432)
other:  13.946025865654466      ('bark', 64, 89.99999999999972, 26.200000000000003, 4416)
vocals:         9.61874212410149      ('bark', 103, 28.299999999999933, 48.6, 7048)
```

### sllen <= 4096

```
bass:   6.386422226148837       ('bark', 56, 120.6999999999996, 33.9, 3872)
drums:  8.860784311537326       ('bark', 56, 120.6999999999996, 33.9, 3872)
other:  7.620889520246537       ('mel', 59, 80.29999999999976, 23.3, 3888)
vocals:         7.870184351437417       ('mel', 61, 116.79999999999961, 54.0, 3896)
```

### sllen <= 2048

```
best scores
bass:   5.486964611360233       ('mel', 32, 115.49999999999963, 92.0, 2016)
drums:  8.133625758111135       ('mel', 32, 115.49999999999963, 92.0, 2016)
other:  7.405963393680095       ('mel', 32, 115.49999999999963, 92.0, 2016)
vocals:         6.542029765645169       ('bark', 30, 38.4999999999999, 69.5, 2012)
```
