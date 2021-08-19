# Wiener-EM on the sliCQT vs. STFT

STFT Wiener-EM (1 iteration), after getting the first estimate with the sliCQT:

```
(umx-gpu) sevagh:xumx-sliCQ $ time python -m xumx_slicq.evaluate --outdir ./ests --evaldir ./results --root ~/TRAINING-MUSIC/MUSDB18-HQ/ --model ./pretrained-model/ --no-cuda --track 'Al James - Schoolboy Facination'
  0%|                                                                                                   | 0/50 [00:00<?, ?it/s]track: AM Contra - Heart Peripheral
not same as specified track, skipping...
track: Al James - Schoolboy Facination
loading separator
getting audio
applying separation
STFT WIENER
performing bss evaluation
Al James - Schoolboy Facination
 vocals          ==> SDR:   2.727  SIR:   7.992  ISR:   4.126  SAR:   2.082
drums           ==> SDR:   2.702  SIR:   1.920  ISR:   5.649  SAR:   1.765
bass            ==> SDR:   4.082  SIR:  10.806  ISR:   1.770  SAR:   2.128
other           ==> SDR:  -0.939  SIR:  -3.560  ISR:  11.802  SAR:   3.516

  2%|█▊                                                                                      | 1/50 [02:21<1:55:11, 141.05s/it]
Aggrated Scores (median over frames, median over tracks)
vocals          ==> SDR:   2.727  SIR:   7.992  ISR:   4.126  SAR:   2.082
drums           ==> SDR:   2.702  SIR:   1.920  ISR:   5.649  SAR:   1.765
bass            ==> SDR:   4.082  SIR:  10.806  ISR:   1.770  SAR:   2.128
other           ==> SDR:  -0.939  SIR:  -3.560  ISR:  11.802  SAR:   3.516


real    2m21.987s
user    13m24.800s
sys     1m23.153s
```

sliCQT Wiener-EM (1 iteration). SDR boost, but slower execution:

```
(umx-gpu) sevagh:xumx-sliCQ $ time python -m xumx_slicq.evaluate --outdir ./ests --evaldir ./results --root ~/TRAINING-MUSIC/MUSDB18-HQ/ --model ./pretrained-model/ --no-cuda --track 'Al James - Schoolboy Facination' --slicq-wiener
  0%|                                                                                                   | 0/50 [00:00<?, ?it/s]track: AM Contra - Heart Peripheral
not same as specified track, skipping...
track: Al James - Schoolboy Facination
loading separator
getting audio
applying separation
sliCQT WIENER
performing bss evaluation
Al James - Schoolboy Facination
 vocals          ==> SDR:   2.753  SIR:   7.953  ISR:   4.158  SAR:   2.024
drums           ==> SDR:   2.864  SIR:   1.677  ISR:   5.835  SAR:   1.657
bass            ==> SDR:   4.140  SIR:  11.599  ISR:   1.350  SAR:   2.016
other           ==> SDR:  -0.893  SIR:  -3.391  ISR:  12.189  SAR:   3.448

  2%|█▊                                                                                      | 1/50 [03:02<2:28:50, 182.26s/it]
Aggrated Scores (median over frames, median over tracks)
vocals          ==> SDR:   2.753  SIR:   7.953  ISR:   4.158  SAR:   2.024
drums           ==> SDR:   2.864  SIR:   1.677  ISR:   5.835  SAR:   1.657
bass            ==> SDR:   4.140  SIR:  11.599  ISR:   1.350  SAR:   2.016
other           ==> SDR:  -0.893  SIR:  -3.391  ISR:  12.189  SAR:   3.448


real    3m3.218s
user    23m39.721s
sys     2m55.630s
```
