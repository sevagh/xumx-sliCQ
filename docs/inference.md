No-chunk inference:
    bss evaluation to store in exp-04-trained-models/slicq-wslicq
    drums           ==> SDR:   3.930  SIR:   8.810  ISR:   6.519  SAR:   5.847
    bass            ==> SDR:  -4.861  SIR:  -5.665  ISR:   6.639  SAR:   6.193
    other           ==> SDR:   0.348  SIR:  -1.541  ISR:   1.419  SAR:   2.100
    vocals          ==> SDR:   5.895  SIR:  13.361  ISR:   9.629  SAR:   7.209
    accompaniment   ==> SDR:  12.948  SIR:  15.880  ISR:  20.559  SAR:  15.627


Chunked inference (2621440, 59.44 seconds @ 44100 Hz)
    bss evaluation to store in exp-04-trained-models/slicq-wslicq
    drums           ==> SDR:   3.990  SIR:   8.558  ISR:   6.609  SAR:   5.896
    bass            ==> SDR:  -4.809  SIR:  -5.768  ISR:   6.494  SAR:   6.074
    other           ==> SDR:   0.350  SIR:  -1.408  ISR:   1.442  SAR:   1.995
    vocals          ==> SDR:   5.887  SIR:  13.241  ISR:   9.703  SAR:   7.153
    accompaniment   ==> SDR:  13.001  SIR:  15.890  ISR:  20.690  SAR:  15.513


tradeoffs here and there, no cause for alarm. keep chunking

Remove float64 in Wiener-EM function. results are still similar:

no float64, chunked inference:
    bss evaluation to store in exp-04-trained-models/slicq-wslicq
    drums           ==> SDR:   3.990  SIR:   8.558  ISR:   6.609  SAR:   5.896
    bass            ==> SDR:  -4.809  SIR:  -5.768  ISR:   6.494  SAR:   6.074
    other           ==> SDR:   0.350  SIR:  -1.408  ISR:   1.442  SAR:   1.995
    vocals          ==> SDR:   5.887  SIR:  13.241  ISR:   9.703  SAR:   7.153
    accompaniment   ==> SDR:  13.001  SIR:  15.890  ISR:  20.690  SAR:  15.513
