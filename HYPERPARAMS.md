## Choosing best NSGT

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

## Refined full eval, considering coefficients

Best overall:

```
(umx-gpu) sevagh:open-unmix-nsgt $ python scripts/hyperparam_search.py --seq-dur-min=60 --seq-dur-max=90 --seq-reps=2 --n-random-tracks=12 --fscale='mel' --fmins=91.6 --bins=113 --gamma=0 --sllen 7432 --random-seed=7 --single
using 12 random tracks from MUSDB18-HQ train set validation split
Parameter to evaluate:
        {'scale': 'mel', 'bins': 113, 'fmin': 91.6, 'gamma': 0.0, 'sllen': 7432}
nsgt params:
        nmel-113-91.60-7432
        113 f bins, 228 m bins
        25764 total dim
bass, drums, vocals, other sdr! 7.09 8.73 10.57 6.76
total sdr: 8.29
```

### Complexity of using multiple NSGTs

Is it worth it to use 1 different NSGT per target? Hypothetically each optimal NSGT might boost the SDR score.

Supporting it in the code isn't such a big deal either, using phasemix inversion you don't need the same tf transform for 4 targets.

Leave this for "future tweaks"

Concern about overfitting, etc. Single transform it is. Compared to previous "best" oracles via bayesian optimization:

* MPI: 256 bins, 22.8 Hz fmin, 17592 slle
* IRM: 125 bins, 78.0 Hz fmin, 9216 sllen
