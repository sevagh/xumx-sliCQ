import torch
from math import pi


def hannwin(l, device="cpu"):
    r = torch.arange(l, dtype=float, device=torch.device(device), requires_grad=False)
    r *= pi * 2.0 / l
    r = torch.cos(r)
    r += 1.0
    r *= 0.5
    return r


def blackharr(n, l=None, mod=True, device="cpu"):
    if l is None:
        l = n
    nn = (n // 2) * 2
    k = torch.arange(n, device=torch.device(device), requires_grad=False)
    if not mod:
        bh = (
            0.35875
            - 0.48829 * torch.cos(k * (2 * pi / nn))
            + 0.14128 * torch.cos(k * (4 * pi / nn))
            - 0.01168 * torch.cos(k * (6 * pi / nn))
        )
    else:
        bh = (
            0.35872
            - 0.48832 * torch.cos(k * (2 * pi / nn))
            + 0.14128 * torch.cos(k * (4 * pi / nn))
            - 0.01168 * torch.cos(k * (6 * pi / nn))
        )
    zeros_size = int((l - n).item())
    bh = torch.hstack(
        (
            bh,
            torch.zeros(
                (zeros_size,),
                dtype=bh.dtype,
                device=torch.device(device),
                requires_grad=False,
            ),
        )
    )
    bh = torch.hstack((bh[-n.int() // 2 :], bh[: -n.int() // 2]))
    return bh


def _isseq(x):
    try:
        len(x)
    except TypeError:
        return False
    return True


def chkM(M, g, device="cpu"):
    if M is None:
        M = torch.as_tensor(
            list(map(len, g)), device=torch.device(device), requires_grad=False
        )
    elif not _isseq(M):
        M = (
            torch.ones(
                len(g), dtype=int, device=torch.device(device), requires_grad=False
            )
            * M
        )
    return M


def calcwinrange(g, rfbas, Ls, device="cpu"):
    shift = rfbas[1:] - rfbas[:-1]
    shift2 = torch.zeros(
        shift.shape[0] + 1, dtype=shift.dtype, device=shift.device, requires_grad=False
    )
    to_append = -rfbas[-1] % Ls
    shift2[1:] = shift
    shift2[0] = to_append
    shift = shift2

    timepos = torch.cumsum(shift, 0)
    nn = timepos[-1].clone().item()
    timepos -= shift[0]  # Calculate positions from shift vector

    wins = []
    for gii, tpii in zip(g, timepos):
        Lg = len(gii)
        win_range = torch.arange(
            -(Lg // 2) + tpii,
            Lg - (Lg // 2) + tpii,
            dtype=int,
            device=torch.device(device),
            requires_grad=False,
        )
        win_range %= nn

        wins.append(win_range)

    return wins, nn


def nsdual(g, wins, nn, M=None, device="cpu"):
    M = chkM(M, g, device)

    # Construct the diagonal of the frame operator matrix explicitly
    x = torch.zeros(
        (nn,), dtype=float, device=torch.device(device), requires_grad=False
    )
    for gi, mii, sl in zip(g, M, wins):
        xa = torch.square(torch.fft.fftshift(gi))
        xa *= mii
        x[sl] += xa

    gd = [gi / torch.fft.ifftshift(x[wi]) for gi, wi in zip(g, wins)]
    return gd
