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
    def __init__(self, fmin, fmax, bnds, beyond=0, gamma=0., device="cpu"):
        """
        @param fmin: minimum frequency (Hz)
        @param fmax: maximum frequency (Hz)
        @param bnds: number of frequency bands (int)
        @param beyond: number of frequency bands below fmin and above fmax (int)
        """
        Scale.__init__(self, bnds+beyond*2)
        lfmin = math.log2(fmin)
        lfmax = math.log2(fmax)
        odiv = (lfmax-lfmin)/(bnds-1)
        lfmin_ = lfmin-odiv*beyond
        lfmax_ = lfmax+odiv*beyond
        self.fmin = 2**lfmin_
        self.fmax = 2**lfmax_
        self.pow2n = 2**odiv
        self.q = math.sqrt(self.pow2n)/(self.pow2n-1.)/2.

        # set gamma for the variable-q scale, widening lower bands
        self.gamma = gamma
        self.device = torch.device(device)
        
    def F(self, bnd=None):
        return self.fmin*self.pow2n**(bnd if bnd is not None else torch.arange(self.bnds, device=self.device, requires_grad=False)) + self.gamma
    
    def Q(self, bnd=None):
        return self.q


class MelScale(Scale):
    @staticmethod
    def hz2mel(f):
        "\cite{shannon:2003}"
        return math.log10(f/700.+1.)*2595.

    @staticmethod
    def mel2hz(m):
        "\cite{shannon:2003}"
        return (math.pow(10.,m/2595.)-1.)*700.

    def __init__(self, fmin, fmax, bnds, beyond=0, device="cpu"):
        """
        @param fmin: minimum frequency (Hz)
        @param fmax: maximum frequency (Hz)
        @param bnds: number of frequency bands (int)
        @param beyond: number of frequency bands below fmin and above fmax (int)
        """
        mmin = self.hz2mel(fmin)
        mmax = self.hz2mel(fmax)
        Scale.__init__(self, bnds+beyond*2)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.mbnd = (mmax-mmin)/(bnds-1)  # mels per band
        self.mmin = mmin-self.mbnd*beyond
        self.mmax = mmax+self.mbnd*beyond
        self.device = torch.device(device)
        
    def F(self, bnd=None):
        if bnd is None:
            bnd = torch.arange(self.bnds, device=self.device, requires_grad=False)
        return self.mel2hz(bnd*self.mbnd+self.mmin)

    def Q1(self, bnd=None): # obviously not exact
        if bnd is None:
            bnd = torch.arange(self.bnds, device=self.device, requires_grad=False)
        mel = bnd*self.mbnd+self.mmin
        odivs = (torch.exp(mel/-1127.)-1.)*(-781.177/self.mbnd)
        pow2n = torch.pow(2, 1./odivs)
        return torch.sqrt(pow2n)/(pow2n-1.)/2.
