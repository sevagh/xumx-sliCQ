import torch
from itertools import cycle, chain, tee
from math import ceil
from .slicing import slicing
from .unslicing import unslicing
from .nsgfwin_sl import nsgfwin
from .nsgtf import nsgtf_sl
from .nsigtf import nsigtf_sl
from .util import calcwinrange, nsdual
from .reblock import reblock


def arrange(cseq, fwd, device="cpu"):
    for i in range(len(cseq)):
        M = cseq[i].shape[-1]

        if fwd:
            odd_mid = M // 4
            even_mid = 3 * M // 4
        else:
            odd_mid = 3 * M // 4
            even_mid = M // 4

        # odd indices
        cseq[i][1::2, :, :, :] = torch.cat(
            (cseq[i][1::2, :, :, odd_mid:], cseq[i][1::2, :, :, :odd_mid]), dim=-1
        )

        # even indices
        cseq[i][::2, :, :, :] = torch.cat(
            (cseq[i][::2, :, :, even_mid:], cseq[i][::2, :, :, :even_mid]), dim=-1
        )
    return cseq


def starzip(iterables):
    def inner(itr, i):
        for t in itr:
            yield t[i]

    iterables = iter(iterables)
    it = next(iterables)  # we need that to determine the length of one element
    iterables = chain((it,), iterables)
    return [inner(itr, i) for i, itr in enumerate(tee(iterables, len(it)))]


def chnmap_forward(gen, seq, device="cpu"):
    chns = starzip(seq)  # returns a list of generators (one for each channel)

    # fuck generators, use a tensor
    chns = [list(x) for x in chns]

    f_slices = torch.empty(
        len(chns[0]),
        len(chns),
        len(chns[0][0]),
        dtype=torch.float32,
        device=torch.device(device),
    )

    for i, chn in enumerate(chns):
        for j, sig in enumerate(chn):
            f_slices[j, i, :] = sig

    ret = gen(f_slices)

    return ret


class NSGT_sliced(torch.nn.Module):
    def __init__(
        self,
        scale,
        sl_len,
        tr_area,
        fs,
        min_win=16,
        Qvar=1,
        real=False,
        recwnd=False,
        reducedform=0,
        multichannel=False,
        dtype=torch.float32,
        device="cpu",
    ):
        assert fs > 0
        assert sl_len > 0
        assert tr_area >= 0
        assert sl_len > tr_area * 2
        assert min_win > 0
        assert 0 <= reducedform <= 2

        assert sl_len % 4 == 0
        assert tr_area % 2 == 0

        super(NSGT_sliced, self).__init__()

        self.device = torch.device(device)

        self.sl_len = sl_len
        self.tr_area = tr_area
        self.fs = fs
        self.real = real
        self.userecwnd = recwnd
        self.reducedform = reducedform
        self.multichannel = multichannel

        self.scale = scale
        self.frqs, self.q = self.scale()

        self.g, self.rfbas, self.M = nsgfwin(
            self.frqs,
            self.q,
            self.fs,
            self.sl_len,
            sliced=True,
            min_win=min_win,
            Qvar=Qvar,
            dtype=dtype,
            device=self.device,
        )

        if real:
            assert 0 <= reducedform <= 2
            sl = slice(reducedform, len(self.g) // 2 + 1 - reducedform)
        else:
            sl = slice(0, None)

        self.sl = sl

        self.fbins_actual = sl.stop

        # coefficients per slice
        self.ncoefs = max(
            int(ceil(float(len(gii)) / mii)) * mii
            for mii, gii in zip(self.M[sl], self.g[sl])
        )

        if multichannel:
            self.channelize = lambda seq: seq
            self.unchannelize = lambda seq: seq
        else:
            self.channelize = lambda seq: ((it,) for it in seq)
            self.unchannelize = lambda seq: (it[0] for it in seq)

        self.wins, self.nn = calcwinrange(
            self.g, self.rfbas, self.sl_len, device=self.device
        )

        self.gd = nsdual(self.g, self.wins, self.nn, self.M, device=self.device)
        self.setup_lambdas()

    def setup_lambdas(self):
        self.fwd = lambda fc: nsgtf_sl(
            fc,
            self.g,
            self.wins,
            self.nn,
            self.M,
            real=self.real,
            reducedform=self.reducedform,
            device=self.device,
        )
        self.bwd = lambda cc: nsigtf_sl(
            cc,
            self.gd,
            self.wins,
            self.nn,
            self.sl_len,
            real=self.real,
            reducedform=self.reducedform,
            device=self.device,
        )

    def _apply(self, fn):
        super(NSGT_sliced, self)._apply(fn)
        self.wins = [fn(w) for w in self.wins]
        self.g = [fn(g) for g in self.g]
        self.device = self.g[0].device
        self.setup_lambdas()

    def forward(self, sig):
        "transform - s: iterable sequence of sequences"

        # sig = self.channelize(sig)

        # Compute the slices (zero-padded Tukey window version)
        f_sliced = slicing(sig, self.sl_len, self.tr_area, device=self.device)

        cseq = chnmap_forward(self.fwd, f_sliced, device=self.device)

        cseq = arrange(cseq, True, device=self.device)

        # cseq = self.unchannelize(cseq)

        return cseq

    def backward(self, cseq, length):
        "inverse transform - c: iterable sequence of coefficients"
        # cseq = self.channelize(cseq)

        cseq = arrange(cseq, False, device=self.device)

        frec_sliced = self.bwd(cseq)

        # Glue the parts back together
        ftype = float if self.real else complex
        sig = unslicing(
            frec_sliced,
            self.sl_len,
            self.tr_area,
            dtype=ftype,
            usewindow=self.userecwnd,
            device=self.device,
        )

        # sig = list(self.unchannelize(sig))[2:]
        sig = list(sig)[2:]

        # convert to tensor
        ret = next(
            reblock(
                sig,
                length,
                fulllast=False,
                multichannel=self.multichannel,
                device=self.device,
            )
        )
        return ret

    @property
    def coef_factor(self):
        return float(self.ncoefs) / self.sl_len

    def coef_factors(self):
        # coefficients per slice
        all_ncoefs = [
            int(ceil(float(len(gii)) / mii)) * mii
            for mii, gii in zip(self.M[self.sl], self.g[self.sl])
        ]

        return [float(ncoefs) / self.sl_len for ncoefs in all_ncoefs]
