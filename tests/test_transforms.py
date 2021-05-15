import pytest
import numpy as np
import torch
from openunmix import transforms

import matplotlib.pyplot as plt


# try some durations
@pytest.fixture(params=[4096, 44100, int(44100*3.5)])
def nb_timesteps(request):
    return int(request.param)


@pytest.fixture(params=[1, 2, 3, 4, 9])
def nb_channels(request):
    return request.param


@pytest.fixture(params=[1, 2, 3, 5, 7, 13, 56])
def nb_samples(request):
    return request.param

@pytest.fixture
def audio(request, nb_samples, nb_channels, nb_timesteps):
    return torch.rand((nb_samples, nb_channels, nb_timesteps))


def test_nsgt_fwd_cuda_inv_cpu(audio):
    nsgt, _ = transforms.make_filterbanks(device="cuda")

    X = nsgt(audio)

    # forward with cuda, backward with cpu
    X = X.detach().cpu()

    shape = X.shape

    _, insgt = transforms.make_filterbanks(device="cpu")

    out = insgt(X, length=audio.shape[-1])

    assert np.sqrt(np.mean((audio.detach().cpu().numpy() - out.detach().numpy()) ** 2)) < 1e-6


def test_nsgt_fwd_cpu_inv_cuda(audio):
    audio = audio.detach().cpu()

    nsgt, _ = transforms.make_filterbanks(device="cpu")

    X = nsgt(audio)

    # forward with cpu, backward with cuda
    X = torch.tensor(X, device=torch.device("cuda"))

    shape = X.shape

    # add fake target of size 1
    X = X.reshape(shape[0], 1, *shape[1:])

    _, insgt = transforms.make_filterbanks(device="cuda")

    out = insgt(X, length=audio.shape[-1])

    # remove fake target of size 1
    out = torch.squeeze(out, dim=1)

    assert np.sqrt(np.mean((audio.detach().numpy() - out.detach().cpu().numpy()) ** 2)) < 1e-6


def test_nsgt_fwd_inv_cpu(audio):
    audio = audio.detach().cpu()

    nsgt, insgt = transforms.make_filterbanks(device="cpu")

    X = nsgt(audio)

    shape = X.shape

    # add fake target of size 1
    X = X.reshape(shape[0], 1, *shape[1:])

    out = insgt(X, length=audio.shape[-1])

    # remove fake target of size 1
    out = torch.squeeze(out, dim=1)

    assert np.sqrt(np.mean((audio.detach().numpy() - out.detach().numpy()) ** 2)) < 1e-6


def test_nsgt_fwd_inv_cuda(audio):
    nsgt, insgt = transforms.make_filterbanks(device="cuda")

    X = nsgt(audio)

    # forward with cuda, backward with cpu
    #X = X.detach().cpu()

    shape = X.shape

    # add fake target of size 1
    X = X.reshape(shape[0], 1, *shape[1:])

    out = insgt(X, length=audio.shape[-1])

    # remove fake target of size 1
    out = torch.squeeze(out, dim=1)

    assert np.sqrt(np.mean((audio.detach().cpu().numpy() - out.detach().cpu().numpy()) ** 2)) < 1e-6


def test_nsgt_fwd_inv_cuda_no_fake_targets(audio):
    nsgt, insgt = transforms.make_filterbanks(device="cuda")

    X = nsgt(audio)

    # forward with cuda, backward with cpu
    X = X.detach().cpu()

    shape = X.shape

    out = insgt(X, length=audio.shape[-1])

    assert np.sqrt(np.mean((audio.detach().cpu().numpy() - out.detach().cpu().numpy()) ** 2)) < 1e-6


import pytest
pytest.main(["-s", "tests/test_transforms.py::test_nsgt_fwd_inv_cuda"])
