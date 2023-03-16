import torch
from math import ceil

from .util import chkM


def nsgtf_sl(f_slices, g, wins, nn, M=None, real=False, reducedform=0, device="cpu"):
    M = chkM(M, g, device)
    dtype = g[0].dtype

    if real:
        assert 0 <= reducedform <= 2
        sl = slice(reducedform, len(g) // 2 + 1 - reducedform)
    else:
        sl = slice(0, None)

    maxLg = max(
        int(ceil(float(len(gii)) / mii)) * mii for mii, gii in zip(M[sl], g[sl])
    )
    temp0 = None

    loopparams = []
    for mii, gii, win_range in zip(M[sl], g[sl], wins[sl]):
        Lg = len(gii)
        col = int(ceil(float(Lg) / mii))
        assert col * mii >= Lg
        assert col == 1

        p = (mii, win_range, Lg, col)
        loopparams.append(p)

    ragged_giis = [
        torch.nn.functional.pad(
            torch.unsqueeze(gii, dim=0), (0, maxLg.int().item() - gii.shape[0])
        )
        for gii in g[sl]
    ]
    giis = torch.conj(torch.cat(ragged_giis))

    ft = torch.fft.fft(f_slices)

    Ls = f_slices.shape[-1]

    assert nn == Ls

    block_ptr = -1
    bucketed_tensors = []
    ret = []

    for j, (mii, win_range, Lg, col) in enumerate(loopparams):
        c = torch.zeros(
            *f_slices.shape[:2], 1, Lg, dtype=ft.dtype, device=torch.device(device)
        )

        t = ft[:, :, win_range] * torch.fft.fftshift(giis[j, :Lg])

        sl1 = slice(None, (Lg + 1) // 2)
        sl2 = slice(-(Lg // 2), None)

        c[:, :, 0, sl1] = t[
            :, :, Lg // 2 :
        ]  # if mii is odd, this is of length mii-mii//2
        c[:, :, 0, sl2] = t[:, :, : Lg // 2]  # if mii is odd, this is of length mii//2

        # start a new block
        if block_ptr == -1 or bucketed_tensors[block_ptr][0].shape[-1] != Lg:
            if block_ptr != -1:
                # run ifft on the completed previous block before moving onto a new one
                bucketed_tensors[block_ptr] = torch.fft.ifft(
                    bucketed_tensors[block_ptr]
                )
            bucketed_tensors.append(c)
            block_ptr += 1
        else:
            # concat block to previous contiguous frequency block with same time resolution
            bucketed_tensors[block_ptr] = torch.cat(
                [bucketed_tensors[block_ptr], c], dim=2
            )

    # bucket-wise ifft
    return bucketed_tensors
