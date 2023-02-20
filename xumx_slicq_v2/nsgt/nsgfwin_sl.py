from .util import hannwin, blackharr
from math import ceil
from warnings import warn
from itertools import chain
import torch


def nsgfwin(
    f,
    q,
    sr,
    Ls,
    sliced=True,
    min_win=4,
    Qvar=1,
    dowarn=True,
    dtype=torch.float32,
    device="cpu",
):
    nf = sr / 2.0

    lim = torch.argmax((f > 0).long())
    if lim != 0:
        f = f[lim:]
        q = q[lim:]

    lim = torch.argmax((f >= nf).long())
    if lim != 0:
        f = f[:lim]
        q = q[:lim]

    assert len(f) == len(q)
    assert torch.all((f[1:] - f[:-1]) > 0)  # frequencies must be increasing
    assert torch.all(q > 0)  # all q must be > 0

    qneeded = f * (Ls / (8.0 * sr))
    if torch.any(q >= qneeded) and dowarn:
        warn(
            "Q-factor too high for frequencies %s"
            % ",".join("%.2f" % fi for fi in f[q >= qneeded])
        )

    fbas = f
    lbas = len(fbas)

    frqs = torch.zeros((fbas.shape[0]+2,), dtype=fbas.dtype, device=fbas.device, requires_grad=False)
    frqs[0] = 0.0
    frqs[1:-1] = fbas
    frqs[-1] = nf

    fbas = torch.cat((frqs, sr - torch.flip(frqs, (0,))[1:-1])).to(torch.device(device))

    fbas *= float(Ls) / sr

    if sliced:
        M = torch.zeros(fbas.shape, dtype=torch.float32, device=torch.device(device), requires_grad=False)
        M[0] = 2 * fbas[1]
        M[1] = fbas[1] / q[0]
        for k in chain(range(2, lbas), (lbas + 1,)):
            M[k] = fbas[k + 1] - fbas[k - 1]
        M[lbas] = fbas[lbas] / q[lbas - 1]
        M[lbas + 2 : 2 * (lbas + 1)] = torch.flip(M[1:lbas+1], (0,))
        M *= Qvar / 4.0
        M = torch.round(M).int()
        M *= 4
    else:
        M = torch.zeros(fbas.shape, dtype=int, device=torch.device(device), requires_grad=False)
        M[0] = torch.round(2 * fbas[1])
        for k in range(1, 2 * lbas + 1):
            M[k] = torch.round(fbas[k + 1] - fbas[k - 1])
        M[-1] = torch.round(Ls - fbas[-2])

    M = torch.clip(M, min_win, torch.inf)

    if sliced:
        g = [blackharr(m, device=device).to(dtype) for m in M]
    else:
        g = [hannwin(m, device=device).to(dtype) for m in M]

    if sliced:
        for kk in (1, lbas + 2):
            if M[kk - 1] > M[kk]:
                Mkk_minus1 = int(M[kk - 1].item())
                Mkk = int(M[kk].item())

                g[kk - 1] = torch.ones(
                    (Mkk_minus1,), dtype=g[kk - 1].dtype, device=torch.device(device), requires_grad=False
                )
                g[kk - 1][
                    Mkk_minus1 // 2
                    - Mkk // 2 : Mkk_minus1 // 2
                    + int(ceil(Mkk / 2.0))
                ] = hannwin(Mkk, device=device)

        rfbas = torch.round(fbas / 2.0).int() * 2
    else:
        fbas[lbas] = (fbas[lbas - 1] + fbas[lbas + 1]) / 2
        fbas[lbas + 2] = Ls - fbas[lbas]
        rfbas = torch.round(fbas).int()

    return g, rfbas, M
