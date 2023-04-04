import matplotlib.pyplot as plt
import torch
import numpy
import os
from warnings import warn
import torch
import numpy
from .nsgt import NSGT_sliced, LogScale, MelScale, BarkScale
from argparse import ArgumentParser
import torchaudio


def overlap_add_slicq(slicq, flatten=False):
    nb_channels, nb_f_bins, nb_slices, nb_m_bins = slicq.shape

    if flatten:
        return torch.flatten(slicq, start_dim=-2, end_dim=-1)

    window = nb_m_bins
    hop = window // 2  # 50% overlap window

    ncoefs = nb_slices * nb_m_bins // 2 + hop
    out = torch.zeros(
        (nb_channels, nb_f_bins, ncoefs),
        dtype=slicq.dtype,
        device=slicq.device,
    )

    ptr = 0

    for i in range(nb_slices):
        out[:, :, ptr : ptr + window] += slicq[:, :, i, :]
        ptr += hop

    return out


def blockwise_spectrogram(
    c,
    fs,
    coef_factor,
    freqs,
    frames,
    output_file,
    plot_title,
    flatten=False,
    fontsize=14,
    cmap="inferno",
):
    # dB
    print(f"c.shape: {c.shape}")

    chop = c.shape[-1]

    nb_t_bins = c.shape[-1]
    nb_slices = c.shape[-2]
    ncoefs = int(coef_factor * frames)
    print(f"ncoefs: {ncoefs}")
    print(f"coef factor: {coef_factor}")

    mls = 20.0 * torch.log10(torch.abs(overlap_add_slicq(c, flatten=flatten)))
    mls = mls[:, :, int(chop / 2) :]
    mls = mls[:, :, : -int(chop / 2)]

    plt.rcParams.update({"font.size": fontsize})
    fig, axs = plt.subplots(1)

    print(f"Plotting t*f space")

    # mix down multichannel
    mls = torch.mean(mls, dim=0)

    mls = mls.T
    print(f"mls: {mls.shape}")

    fs_coef = fs * coef_factor  # frame rate of coefficients
    print(f"{fs_coef=}")

    ncoefs = int(coef_factor * frames)
    print(f"{ncoefs=}")
    mls = mls[:ncoefs, :]

    mls_dur = len(mls) / fs_coef  # final duration of MLS

    if flatten:
        mls_dur *= 2.0

    nb_bins = len(freqs)

    mls_max = torch.quantile(mls, 0.999)
    print(f"mls_dur: {mls_dur}")
    print(f"mls.T: {mls.T.shape}")
    print(f"freqs: {freqs.shape}")

    im = axs.pcolormesh(
        numpy.linspace(0.0, mls_dur, num=ncoefs),
        freqs / 1000.0,
        mls.T,
        vmin=mls_max - 120.0,
        vmax=mls_max,
        cmap=cmap,
    )

    axs.set_title(plot_title)

    axs.set_xlabel("Time (s)")
    axs.set_ylabel("Frequency (kHz)")
    axs.yaxis.get_major_locator().set_params(integer=True)

    fig.colorbar(im, ax=axs, shrink=1.0, pad=0.006, label="dB")

    plt.subplots_adjust(wspace=0.001, hspace=0.001)

    DPI = fig.get_dpi()
    fig.set_size_inches(2560.0 / float(DPI), 1440.0 / float(DPI))
    fig.savefig(output_file, dpi=DPI, bbox_inches="tight")


def visualization_main():
    parser = ArgumentParser()

    parser.add_argument(
        "--input-wav",
        type=str,
        default="/xumx-sliCQ-V2/.github/gspi.wav",
        help="Input file",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=44100,
        help="Sample rate used for the NSGT (default=%(default)s)",
    )
    parser.add_argument("--cmap", type=str, default="hot", help="spectrogram color map")
    parser.add_argument(
        "--fscale",
        choices=("mel", "bark", "Bark"),
        default="Bark",
        help="Frequency scale",
    )
    parser.add_argument(
        "--fbins",
        type=int,
        default=262,
        help="Number of frequency bins (total or per octave, default=%(default)s)",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=32.9,
        help="Minimum frequency in Hz (default=%(default)s)",
    )
    parser.add_argument(
        "--fontsize", type=int, default=14, help="Plot font size, default=%(default)s)"
    )
    parser.add_argument(
        "--flatten", action="store_true", help="Flatten instead of overlap"
    )

    args = parser.parse_args()
    if not os.path.exists(args.input_wav):
        parser.error("Input file '%s' not found" % args.input_wav)

    fs = args.sr

    # build transform
    scales = {"mel": MelScale, "bark": BarkScale, "Bark": BarkScale}
    try:
        scale = scales[args.fscale]
    except KeyError:
        parser.error("Scale unknown (--scale option)")

    scl = scale(args.fmin, 22050, args.fbins)

    freqs, qs = scl()

    # Read audio data
    signal, rate = torchaudio.load(args.input_wav)
    nb_channels = signal.shape[0]
    nb_frames = signal.shape[-1]

    # duration of signal in s
    dur = nb_frames / float(fs)

    sllen, trlen = scl.suggested_sllen_trlen(fs)

    slicq = NSGT_sliced(
        scl, sllen, trlen, fs, real=True, multichannel=nb_channels == 2, device="cpu"
    )

    # generator for forward transformation
    c_list = slicq.forward((signal,))

    # list of ragged frequency bins
    transform_name = "sliCQT"

    if args.fmin > 0.0:
        freqs = numpy.r_[[0.0], freqs]

    slicq_params = "{0} scale, {1} bins, {2:.1f}-22050 Hz".format(
        args.fscale, args.fbins, args.fmin
    )

    coef_factors = slicq.coef_factors()

    freq_idx = 0
    for i, c in enumerate(c_list):
        c = c.permute(1, 2, 0, 3)
        print(f"slicqt block shape: {c.shape}")

        output_png_path = os.path.join(
            "/spectrogram-plots",
            f"spectrogram-{os.path.basename(args.input_wav)}-block-{i}.png",
        )
        n_freqs = c.shape[1]

        blockwise_spectrogram(
            c,
            fs,
            coef_factors[i],
            freqs[freq_idx : freq_idx + n_freqs],
            signal.shape[1],
            output_png_path,
            f"Magnitude sliCQT, block {i} ({slicq_params})",
            flatten=args.flatten,
            fontsize=args.fontsize,
            cmap=args.cmap,
        )
        freq_idx += n_freqs


if __name__ == "__main__":
    visualization_main()
