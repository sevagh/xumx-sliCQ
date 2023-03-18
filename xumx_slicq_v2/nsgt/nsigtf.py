import torch
from itertools import chain


def nsigtf_sl(cseq, gd, wins, nn, Ls=None, real=False, reducedform=0, device="cpu"):
    dtype = gd[0].dtype

    ifft = torch.fft.irfft if real else torch.fft.ifft

    sl = lambda x: x
    if real and reducedform:
        sl = lambda x: chain(
            x[reducedform : len(gd) // 2 + 1 - reducedform],
            x[len(gd) // 2 + reducedform : len(gd) + 1 - reducedform],
        )

    maxLg = max(len(gdii) for gdii in sl(gd))

    ragged_gdiis = [
        torch.nn.functional.pad(
            torch.unsqueeze(gdii, dim=0), (0, maxLg - gdii.shape[0])
        )
        for gdii in sl(gd)
    ]
    gdiis = torch.conj(torch.cat(ragged_gdiis))

    assert type(cseq) == list
    nfreqs = 0
    for i, cseq_tsor in enumerate(cseq):
        cseq_dtype = cseq_tsor.dtype
        cseq[i] = torch.fft.fft(cseq_tsor)
        nfreqs += cseq_tsor.shape[2]
    cseq_shape = (*cseq_tsor.shape[:2], nfreqs)

    fr = torch.zeros(
        *cseq_shape[:2], nn, dtype=cseq_dtype, device=torch.device(device)
    )  # Allocate output

    fbins = cseq_shape[2]

    loopparams = []
    for gdii, win_range in zip(sl(gd), sl(wins)):
        Lg = len(gdii)
        wr1 = win_range[: (Lg) // 2]
        wr2 = win_range[-((Lg + 1) // 2) :]
        p = (wr1, wr2, Lg)
        loopparams.append(p)

    # frequencies are bucketed by same time resolution
    fbin_ptr = 0
    for i, fc in enumerate(cseq):
        Lg_outer = fc.shape[-1]

        nb_fbins = fc.shape[2]
        for j, (wr1, wr2, Lg) in enumerate(
            loopparams[fbin_ptr : fbin_ptr + nb_fbins][:fbins]
        ):
            freq_idx = fbin_ptr + j
            assert Lg == Lg_outer

            t = fc[:, :, j]

            r = (Lg + 1) // 2
            l = Lg // 2

            t1 = t[:, :, :r]
            t2 = t[:, :, Lg - l : Lg]

            t[:, :, :Lg] *= gdiis[freq_idx, :Lg]
            t[:, :, :Lg] *= Lg

            fr[:, :, wr1] += t2
            fr[:, :, wr2] += t1
        fbin_ptr += nb_fbins

    ftr = fr[:, :, : nn // 2 + 1] if real else fr

    # vvvv the GRADIENT KILLER
    #with torch.no_grad():
    sig = ifft(ftr, n=Ls)
    # ^^^^ find a way to optimize this and win...

    return sig
