import torch
import math


class Scale:
    dbnd = 1.0e-8

    def __init__(self, bnds, device="cpu"):
        self.bnds = bnds
        self.device = torch.device(device)

    def __len__(self):
        return self.bnds

    def Q(self, bnd=None):
        # numerical differentiation (if self.Q not defined by sub-class)
        if bnd is None:
            bnd = torch.arange(self.bnds, device=self.device, requires_grad=False)
        return (
            self.F(bnd)
            * self.dbnd
            / (self.F(bnd + self.dbnd) - self.F(bnd - self.dbnd))
        )

    def __call__(self):
        f = torch.as_tensor([self.F(b) for b in range(self.bnds)], dtype=torch.float32, device=self.device)
        f.requires_grad = False
        q = torch.as_tensor([self.Q(b) for b in range(self.bnds)], dtype=torch.float32, device=self.device)
        q.requires_grad = False
        return f, q

    def suggested_sllen_trlen(self, sr):
        f, q = self()

        Ls = int(torch.ceil(max((q * 8.0 * sr) / f)))

        # make sure its divisible by 4
        Ls = Ls + -Ls % 4

        sllen = Ls

        trlen = sllen // 4
        trlen = trlen + -trlen % 2  # make trlen divisible by 2

        return sllen, trlen


def hz2bark(f):
    #       HZ2BARK         Converts frequencies Hertz (Hz) to Bark
    #
    b = 6 * math.asinh(f / 600)
    return b


def bark2hz(b):
    #       BARK2HZ         Converts frequencies Bark to Hertz (HZ)
    #
    f = 600 * math.sinh(b / 6)
    return f


class BarkScale(Scale):
    def __init__(self, fmin, fmax, bnds, beyond=0, device="cpu"):
        """
        @param fmin: minimum frequency (Hz)
        @param fmax: maximum frequency (Hz)
        @param bnds: number of frequency bands (int)
        @param beyond: number of frequency bands below fmin and above fmax (int)
        """
        bmin = hz2bark(fmin)
        bmax = hz2bark(fmax)
        Scale.__init__(self, bnds + beyond * 2)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.bbnd = (bmax - bmin) / (bnds - 1)  # mels per band
        self.bmin = bmin - self.bbnd * beyond
        self.bmax = bmax + self.bbnd * beyond
        self.device = torch.device(device)

    def F(self, bnd=None):
        if bnd is None:
            bnd = torch.arange(self.bnds, device=self.device, requires_grad=False)
        return bark2hz(bnd * self.bbnd + self.bmin)


# hacky stuff since i didn't want to delete downstream code
OctScale = BarkScale
MelScale = BarkScale
LogScale = BarkScale
VQLogScale = BarkScale
