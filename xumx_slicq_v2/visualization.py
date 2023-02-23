import matplotlib.pyplot as plt
import torch
import numpy


def overlap_add_slicq(slicq):
    nb_samples, nb_channels, nb_f_bins, nb_slices, nb_m_bins = slicq.shape

    window = nb_m_bins
    hop = window // 2  # 50% overlap window

    ncoefs = nb_slices * nb_m_bins // 2 + hop
    out = torch.zeros(
        (nb_samples, nb_channels, nb_f_bins, ncoefs),
        dtype=slicq.dtype,
        device=slicq.device,
    )

    ptr = 0

    for i in range(nb_slices):
        out[:, :, :, ptr : ptr + window] += slicq[:, :, :, i, :]
        ptr += hop

    return out


def spectrogram(c, fs, coef_factor, transform_name, freqs, frames, sliced=True, flatten=False, fontsize=14, cmap='inferno', slicq_name='', output_file=None):
    # dB
    if not sliced:
        mls = 20.*torch.log10(torch.abs(c))
        transform_name = 'NSGT'
    else:
        chop = c.shape[-1]
        mls = 20.*torch.log10(torch.abs(overlap_add_slicq(c, flatten=flatten)))
        mls = mls[:, :, :, int(chop/2):]
        mls = mls[:, :, :, :-int(chop/2)]

    plt.rcParams.update({'font.size': fontsize})
    fig, axs = plt.subplots(1)

    print(f"Plotting t*f space")

    # remove batch
    mls = torch.squeeze(mls, dim=0)
    # mix down multichannel
    mls = torch.mean(mls, dim=0)

    mls = mls.T

    fs_coef = fs*coef_factor # frame rate of coefficients

    ncoefs = int(coef_factor*frames)
    mls = mls[:ncoefs, :]

    mls_dur = len(mls)/fs_coef # final duration of MLS

    nb_bins = len(freqs)

    mls_max = torch.quantile(mls, 0.999)
    print(f'mls_dur: {mls_dur}')
    try:
        im = axs.pcolormesh(numpy.linspace(0.0, mls_dur, num=ncoefs), freqs/1000., mls.T, vmin=mls_max-120., vmax=mls_max, cmap=cmap)
    except TypeError:
        freqs = numpy.r_[[0.], freqs]
        im = axs.pcolormesh(numpy.linspace(0.0, mls_dur, num=ncoefs), freqs/1000., mls.T, vmin=mls_max-120., vmax=mls_max, cmap=cmap)

    title = f'Magnitude {transform_name}'

    if slicq_name != '':
        title += f', {slicq_name}'

    axs.set_title(title)

    axs.set_xlabel('Time (s)')
    axs.set_ylabel('Frequency (kHz)')
    axs.yaxis.get_major_locator().set_params(integer=True)

    fig.colorbar(im, ax=axs, shrink=1.0, pad=0.006, label='dB')

    plt.subplots_adjust(wspace=0.001,hspace=0.001)

    if output_file is not None:
        DPI = fig.get_dpi()
        fig.set_size_inches(2560.0/float(DPI),1440.0/float(DPI))
        fig.savefig(output_file, dpi=DPI, bbox_inches='tight')
    else:
        plt.show()
