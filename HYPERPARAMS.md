## Choosing best NSGT

# vocal lr

Learning rate search finished. See the graph with {finder_name}.plot()
LR suggestion: steepest gradient
Suggested LR: 1.02E+01

# sllen <= 8192

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

# sllen <= 4096

```
bass:   6.386422226148837       ('bark', 56, 120.6999999999996, 33.9, 3872)
drums:  8.860784311537326       ('bark', 56, 120.6999999999996, 33.9, 3872)
other:  7.620889520246537       ('mel', 59, 80.29999999999976, 23.3, 3888)
vocals:         7.870184351437417       ('mel', 61, 116.79999999999961, 54.0, 3896)
```

# sllen <= 2048

```
best scores
bass:   5.486964611360233       ('mel', 32, 115.49999999999963, 92.0, 2016)
drums:  8.133625758111135       ('mel', 32, 115.49999999999963, 92.0, 2016)
other:  7.405963393680095       ('mel', 32, 115.49999999999963, 92.0, 2016)
vocals:         6.542029765645169       ('bark', 30, 38.4999999999999, 69.5, 2012)
```
