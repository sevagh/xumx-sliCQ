from itertools import cycle, chain
from .util import hannwin
import torch


def slicequads(frec_sliced, hhop):
    ret = torch.zeros(
        frec_sliced.shape[0],
        frec_sliced.shape[1],
        4,
        hhop,
        dtype=frec_sliced.dtype,
        device=frec_sliced.device,
    )

    frec_sliced = frec_sliced.view(frec_sliced.shape[0], frec_sliced.shape[1], -1, hhop)

    # assign even indices
    ret[::2, :, 0] = frec_sliced[::2, :, 3]
    ret[::2, :, 1] = frec_sliced[::2, :, 0]
    ret[::2, :, 2] = frec_sliced[::2, :, 1]
    ret[::2, :, 3] = frec_sliced[::2, :, 2]

    # assign odd indices
    ret[1::2, :, 0] = frec_sliced[1::2, :, 1]
    ret[1::2, :, 1] = frec_sliced[1::2, :, 2]
    ret[1::2, :, 2] = frec_sliced[1::2, :, 3]
    ret[1::2, :, 3] = frec_sliced[1::2, :, 0]

    return ret.permute(0, 2, 1, 3)


def unslicing(frec_sliced, sl_len, tr_area, dtype=float, usewindow=True, device="cpu"):
    hhop = sl_len // 4
    islices = slicequads(frec_sliced, hhop)

    if usewindow:
        tr_area2 = min(2 * hhop - tr_area, 2 * tr_area)
        htr = tr_area // 2
        htr2 = tr_area2 // 2
        hw = hannwin(tr_area2, device=device)
        tw = torch.zeros(sl_len, dtype=dtype, device=torch.device(device))
        tw[max(hhop - htr - htr2, 0) : hhop - htr] = hw[htr2:]
        tw[hhop - htr : 3 * hhop + htr] = 1
        tw[3 * hhop + htr : min(3 * hhop + htr + htr2, sl_len)] = hw[:htr2]
        tw = [tw[o : o + hhop] for o in range(0, sl_len, hhop)]
    else:
        tw = cycle((1,))

    # get first slice to deduce channels
    firstquad = islices[0]
    chns = len(firstquad[0])  # number of channels in first quad

    output = [
        torch.zeros((chns, hhop), dtype=dtype, device=torch.device(device))
        for _ in range(4)
    ]

    for quad in islices:
        for osl, isl, w in zip(output, quad, tw):
            osl[:] += torch.cat([torch.unsqueeze(isl_, dim=0) for isl_ in isl]) * w
        for _ in range(2):
            yield output.pop(0)
            output.append(
                torch.zeros((chns, hhop), dtype=dtype, device=torch.device(device))
            )

    for _ in range(2):
        yield output.pop(0)
