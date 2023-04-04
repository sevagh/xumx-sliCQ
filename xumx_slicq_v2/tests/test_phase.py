import torch
from xumx_slicq_v2.phase import blockwise_wiener, wiener


# Pytest function to test blockwise_wiener
def test_blockwise_wiener():
    mix_slicqt = torch.randn((1, 2, 14, 257, 37, 2))
    slicqtgrams = torch.randn((4, 1, 2, 14, 257, 37))
    wiener_win_len_param = 5000
    targets_slicqt = blockwise_wiener(mix_slicqt, slicqtgrams, wiener_win_len_param)
    assert targets_slicqt.shape == (4, 1, 2, 14, 257, 37, 2)
    assert torch.all(torch.isfinite(targets_slicqt))


# Pytest function to test wiener with lists of tensors as inputs
def test_wiener():
    mix_slicqt = [torch.randn((1, 2, 14, 257, 37, 2)) for i in range(4)]
    slicqtgrams = [torch.randn((4, 1, 2, 14, 257, 37)) for i in range(4)]
    wiener_win_len_param = 5000
    targets_slicqt = wiener(mix_slicqt, slicqtgrams, wiener_win_len_param)
    assert all([targets_slicqt[i].shape == (4, 1, 2, 14, 257, 37, 2) for i in range(4)])
    assert all([torch.all(torch.isfinite(targets_slicqt[i])) for i in range(4)])
