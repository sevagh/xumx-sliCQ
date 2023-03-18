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
        f = torch.as_tensor(
            [self.F(b) for b in range(self.bnds)],
            dtype=torch.float32,
            device=self.device,
        )
        f.requires_grad = False
        q = torch.as_tensor(
            [self.Q(b) for b in range(self.bnds)],
            dtype=torch.float32,
            device=self.device,
        )
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


class BarkScale(Scale):
    @staticmethod
    def hz2bark(f):
        #       HZ2BARK         Converts frequencies Hertz (Hz) to Bark
        b = 6 * math.asinh(f / 600)
        return b

    @staticmethod
    def bark2hz(b):
        #       BARK2HZ         Converts frequencies Bark to Hertz (HZ)
        f = 600 * math.sinh(b / 6)
        return f

    def __init__(self, fmin, fmax, bnds, beyond=0, device="cpu"):
        """
        @param fmin: minimum frequency (Hz)
        @param fmax: maximum frequency (Hz)
        @param bnds: number of frequency bands (int)
        @param beyond: number of frequency bands below fmin and above fmax (int)
        """
        bmin = self.hz2bark(fmin)
        bmax = self.hz2bark(fmax)
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
        return self.bark2hz(bnd * self.bbnd + self.bmin)


class LogScale(Scale):
    def __init__(self, fmin, fmax, bnds, beyond=0, gamma=0.0, device="cpu"):
        """
        @param fmin: minimum frequency (Hz)
        @param fmax: maximum frequency (Hz)
        @param bnds: number of frequency bands (int)
        @param beyond: number of frequency bands below fmin and above fmax (int)
        """
        Scale.__init__(self, bnds + beyond * 2)
        lfmin = math.log2(fmin)
        lfmax = math.log2(fmax)
        odiv = (lfmax - lfmin) / (bnds - 1)
        lfmin_ = lfmin - odiv * beyond
        lfmax_ = lfmax + odiv * beyond
        self.fmin = 2**lfmin_
        self.fmax = 2**lfmax_
        self.pow2n = 2**odiv
        self.q = math.sqrt(self.pow2n) / (self.pow2n - 1.0) / 2.0

        # set gamma for the variable-q scale, widening lower bands
        self.gamma = gamma
        self.device = torch.device(device)

    def F(self, bnd=None):
        return (
            self.fmin
            * self.pow2n
            ** (
                bnd
                if bnd is not None
                else torch.arange(self.bnds, device=self.device, requires_grad=False)
            )
            + self.gamma
        )

    def Q(self, bnd=None):
        return self.q


class MelScale(Scale):
    @staticmethod
    def hz2mel(f):
        "\cite{shannon:2003}"
        return math.log10(f / 700.0 + 1.0) * 2595.0

    @staticmethod
    def mel2hz(m):
        "\cite{shannon:2003}"
        return (math.pow(10.0, m / 2595.0) - 1.0) * 700.0

    def __init__(self, fmin, fmax, bnds, beyond=0, device="cpu"):
        """
        @param fmin: minimum frequency (Hz)
        @param fmax: maximum frequency (Hz)
        @param bnds: number of frequency bands (int)
        @param beyond: number of frequency bands below fmin and above fmax (int)
        """
        mmin = self.hz2mel(fmin)
        mmax = self.hz2mel(fmax)
        Scale.__init__(self, bnds + beyond * 2)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.mbnd = (mmax - mmin) / (bnds - 1)  # mels per band
        self.mmin = mmin - self.mbnd * beyond
        self.mmax = mmax + self.mbnd * beyond
        self.device = torch.device(device)

    def F(self, bnd=None):
        if bnd is None:
            bnd = torch.arange(self.bnds, device=self.device, requires_grad=False)
        return self.mel2hz(bnd * self.mbnd + self.mmin)

    def Q1(self, bnd=None):  # obviously not exact
        if bnd is None:
            bnd = torch.arange(self.bnds, device=self.device, requires_grad=False)
        mel = bnd * self.mbnd + self.mmin
        odivs = (torch.exp(mel / -1127.0) - 1.0) * (-781.177 / self.mbnd)
        pow2n = torch.pow(2, 1.0 / odivs)
        return torch.sqrt(pow2n) / (pow2n - 1.0) / 2.0


class LinScale(Scale):
    def __init__(self, fmin, fmax, bnds, beyond=0, device="cpu"):
        """
        @param fmin: minimum frequency (Hz)
        @param fmax: maximum frequency (Hz)
        @param bnds: number of frequency bands (int)
        @param beyond: number of frequency bands below fmin and above fmax (int)
        """
        self.df = float(fmax-fmin)/(bnds-1)
        Scale.__init__(self, bnds+beyond*2)
        self.fmin = float(fmin)-self.df*beyond
        if self.fmin <= 0:
            raise ValueError("Frequencies must be > 0.")
        self.fmax = float(fmax)+self.df*beyond

    def F(self, bnd=None):
        return (bnd if bnd is not None else torch.arange(self.bnds, device=self.device, requires_grad=False))*self.df+self.fmin

    def Q(self, bnd=None):
        return self.F(bnd)/(self.df*2)


class MRSTFTScale(Scale):
    def __init__(self, device="cpu"):
        self.device = device

        self.mr_scales = [
            (1.0, 400.0, 128),
            (401.0, 1200.0, 128),
            (1201.0, 4000.0, 128),
            (4001.0, 12000.0, 128),
            (12001.0, 22050.0, 64),
        ]

        self.freqs = [
            torch.linspace(mr_scale[0], mr_scale[1], mr_scale[2], device=self.device)
            for mr_scale in self.mr_scales
        ]
        self.dfs = [
            torch.as_tensor([(mr_scale[1] - mr_scale[0])/mr_scale[2]]*len(self.freqs[i]))
            for i, mr_scale in enumerate(self.mr_scales)
        ]
        self.Fs = torch.cat(self.freqs)
        self.dfs = torch.cat(self.dfs)
        self.Qs = self.Fs/self.dfs*2

        bnds = len(self.Fs)
        Scale.__init__(self, bnds)

    def F(self, bnd=None):
        return self.Fs[bnd] if bnd is not None else self.Fs

    def Q(self, bnd=None):
        return self.Qs[bnd]
