import torch
from typing import List
import xumx_slicq_v2.norbert as norbert


# Calls blockwise_wiener for each element of the input list
def wiener(mix_slicqt, slicqtgrams, wiener_win_len: int = 5000):
    targets_slicqt = [None] * len(mix_slicqt)
    for i, (mix_slicqt_block, slicqtgrams_block) in enumerate(
        zip(mix_slicqt, slicqtgrams)
    ):
        targets_slicqt[i] = blockwise_wiener(
            mix_slicqt_block, slicqtgrams_block, wiener_win_len
        )
    return targets_slicqt


def blockwise_wiener(mix_slicqt, slicqtgrams, wiener_win_len_param: int = 5000):
    mix_slicqt = torch.flatten(mix_slicqt, start_dim=-3, end_dim=-2)
    orig_shape = slicqtgrams.shape
    slicqtgrams = torch.flatten(slicqtgrams, start_dim=-2, end_dim=-1)

    # transposing it as
    # (nb_samples, nb_frames, nb_bins,{1,nb_channels}, nb_sources)
    slicqtgrams = slicqtgrams.permute(1, 4, 3, 2, 0)

    # rearranging it into:
    # (nb_samples, nb_frames, nb_bins, nb_channels, 2) to feed
    # into filtering methods
    mix_slicqt = mix_slicqt.permute(0, 3, 2, 1, 4)

    nb_frames = slicqtgrams.shape[1]
    targets_slicqt = torch.zeros(
        *mix_slicqt.shape[:-1]
        + (
            4,
            2,
        ),
        dtype=mix_slicqt.dtype,
        device=mix_slicqt.device,
    )

    pos = 0
    if wiener_win_len_param:
        wiener_win_len = wiener_win_len_param
    else:
        wiener_win_len = nb_frames
    while pos < nb_frames:
        cur_frame = torch.arange(pos, min(nb_frames, pos + wiener_win_len))
        pos = int(cur_frame[-1]) + 1

        targets_slicqt[:, cur_frame, ...] = torch.view_as_real(
            norbert.wiener(
                slicqtgrams[:, cur_frame, ...],
                torch.view_as_complex(mix_slicqt[:, cur_frame, ...]),
                1,
                False,
            )
        )

    # getting to (nb_samples, nb_targets, channel, fft_size, n_frames)
    targets_slicqt = targets_slicqt.permute(4, 0, 3, 2, 1, 5).contiguous()
    targets_slicqt = targets_slicqt.reshape(
        (
            *orig_shape,
            2,
        )
    )
    return targets_slicqt


def _atan2(y, x):
    r"""Element-wise arctangent function of y/x.
    Returns a new tensor with signed angles in radians.
    It is an alternative implementation of torch.atan2

    Args:
        y (Tensor): First input tensor
        x (Tensor): Second input tensor [shape=y.shape]

    Returns:
        Tensor: [shape=y.shape].
    """
    pi = 2 * torch.asin(torch.as_tensor(1.0))
    x += ((x == 0) & (y == 0)) * 1.0
    out = torch.atan(y / x)
    out += ((y >= 0) & (x < 0)) * pi
    out -= ((y < 0) & (x < 0)) * pi
    out *= 1 - ((y > 0) & (x == 0)) * 1.0
    out += ((y > 0) & (x == 0)) * (pi / 2)
    out *= 1 - ((y < 0) & (x == 0)) * 1.0
    out += ((y < 0) & (x == 0)) * (-pi / 2)
    return out


def blockwise_phasemix_sep(X_block, Ymag_block):
    Xphase_block = _atan2(X_block[..., 1], X_block[..., 0])

    # phasemix-sep all targets at once
    Ycomplex_block_real = torch.empty_like(Ymag_block)
    Ycomplex_block_imag = torch.empty_like(Ymag_block)

    Ycomplex_block_real = Ymag_block[:, ...] * torch.cos(Xphase_block)
    Ycomplex_block_imag = Ymag_block[:, ...] * torch.sin(Xphase_block)

    Ycomplex = torch.cat(
        [
            torch.unsqueeze(Ycomplex_block_real, dim=-1),
            torch.unsqueeze(Ycomplex_block_imag, dim=-1),
        ],
        dim=-1,
    )
    return Ycomplex


def abs_of_real_complex(Xcomplex_real_view):
    # abs(complex) = sqrt(a^2 + b^2)
    return torch.sqrt(Xcomplex_real_view[..., 0] ** 2 + Xcomplex_real_view[..., 1] ** 2)


# Calls blockwise_phasemix_sep for each element of the input list
def phasemix_sep(X: List[torch.Tensor], Ymag: List[torch.Tensor]):
    Ycomplex = [None] * len(X)
    for i, (X_block, Ymag_block) in enumerate(zip(X, Ymag)):
        Ycomplex[i] = blockwise_phasemix_sep(X_block, Ymag_block)
    return Ycomplex
