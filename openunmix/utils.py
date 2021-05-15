from typing import Optional, Union

import torch
import os
import numpy as np
import torchaudio
import warnings
from pathlib import Path
from contextlib import redirect_stderr
import io
import json

import openunmix
from openunmix import model
from openunmix import transforms


def save_checkpoint(state: dict, is_best: bool, path: str, target: str):
    """Save checkpoint

    Args:
        state (dict): torch model state dict
        is_best (bool): if current model is about to be saved as best model
        path (str): model path
        target (str): target name
    """
    # save full checkpoint including optimizer
    torch.save(state, os.path.join(path, target + ".chkpnt"))
    if is_best:
        # save just the weights
        torch.save(state["state_dict"], os.path.join(path, target + ".pth"))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping(object):
    """Early Stopping Monitor"""

    def __init__(self, mode="min", min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if mode == "min":
            self.is_better = lambda a, best: a < best - min_delta
        if mode == "max":
            self.is_better = lambda a, best: a > best + min_delta


def load_target_models(targets, model_str_or_path="umxhq", device="cpu", pretrained=True, sample_rate=44100):
    """Core model loader

    target model path can be either <target>.pth, or <target>-sha256.pth
    (as used on torchub)

    The loader either loads the models from a known model string
    as registered in the __init__.py or loads from custom configs.
    """
    if isinstance(targets, str):
        targets = [targets]

    model_path = Path(model_str_or_path).expanduser()

    models = {}
    models_nsgt = {}

    for target in targets:
        # load model from disk
        with open(Path(model_path, target + ".json"), "r") as stream:
            results = json.load(stream)

        target_model_path = next(Path(model_path).glob("%s*.pth" % target))
        state = torch.load(target_model_path, map_location=device)

        # need to configure an NSGT object to peek at its params to set up the neural network
        # e.g. M depends on the sllen which depends on fscale+fmin+fmax
        nsgt_base = transforms.NSGTBase(
            results["args"]["fscale"],
            results["args"]["fbins"],
            results["args"]["fmin"],
            results["args"]["sllen"],
            gamma=results["args"]["gamma"],
            fs=sample_rate,
            device=device
        )

        models[target] = model.OpenUnmix(
            nsgt_base.fbins_actual,
            nsgt_base.M,
            nb_channels=results["args"]["nb_channels"],
            nb_layers=results["args"]["layers"],
            unidirectional=results["args"]["unidirectional"],
        )

        models_nsgt[target] = nsgt_base

        if pretrained:
            models[target].load_state_dict(state, strict=False)

        models[target].to(device)
    return models, models_nsgt


def load_separator(
    model_str_or_path: str = "umxhq",
    targets: Optional[list] = None,
    niter: int = 1,
    residual: bool = False,
    wiener_win_len: Optional[int] = 300,
    device: Union[str, torch.device] = "cpu",
    pretrained: bool = True,
):
    """Separator loader

    Args:
        model_str_or_path (str): Model name or path to model _parent_ directory
            E.g. The following files are assumed to present when
            loading `model_str_or_path='mymodel', targets=['vocals']`
            'mymodel/separator.json', mymodel/vocals.pth', 'mymodel/vocals.json'.
            Defaults to `umxhq`.
        targets (list of str or None): list of target names. When loading a
            pre-trained model, all `targets` can be None as all targets
            will be loaded
        device (str): torch device, defaults to `cpu`
        pretrained (bool): determines if loading pre-trained weights
    """
    model_path = Path(model_str_or_path).expanduser()

    # when path exists, we assume its a custom model saved locally
    if model_path.exists():
        if targets is None:
            raise UserWarning("For custom models, please specify the targets")

        with open(Path(model_path, "separator.json"), "r") as stream:
            enc_conf = json.load(stream)

        target_models, target_models_nsgt = load_target_models(
            targets=targets, model_str_or_path=model_path, pretrained=pretrained, sample_rate=enc_conf["sample_rate"], device=device
        )

        separator = model.Separator(
            target_models=target_models,
            target_models_nsgt=target_models_nsgt,
            sample_rate=enc_conf["sample_rate"],
            nb_channels=enc_conf["nb_channels"],
            device=device,
        ).to(device)

    return separator


def preprocess(
    audio: torch.Tensor,
    rate: Optional[float] = None,
    model_rate: Optional[float] = None,
) -> torch.Tensor:
    """
    From an input tensor, convert it to a tensor of shape
    shape=(nb_samples, nb_channels, nb_timesteps). This includes:
    -  if input is 1D, adding the samples and channels dimensions.
    -  if input is 2D
        o and the smallest dimension is 1 or 2, adding the samples one.
        o and all dimensions are > 2, assuming the smallest is the samples
          one, and adding the channel one
    - at the end, if the number of channels is greater than the number
      of time steps, swap those two.
    - resampling to target rate if necessary

    Args:
        audio (Tensor): input waveform
        rate (float): sample rate for the audio
        model_rate (float): sample rate for the model

    Returns:
        Tensor: [shape=(nb_samples, nb_channels=2, nb_timesteps)]
    """
    shape = torch.as_tensor(audio.shape, device=audio.device)

    if len(shape) == 1:
        # assuming only time dimension is provided.
        audio = audio[None, None, ...]
    elif len(shape) == 2:
        if shape.min() <= 2:
            # assuming sample dimension is missing
            audio = audio[None, ...]
        else:
            # assuming channel dimension is missing
            audio = audio[:, None, ...]
    if audio.shape[1] > audio.shape[2]:
        # swapping channel and time
        audio = audio.transpose(1, 2)
    if audio.shape[1] > 2:
        warnings.warn("Channel count > 2!. Only the first two channels " "will be processed!")
        audio = audio[..., :2]

    if audio.shape[1] == 1:
        # if we have mono, we duplicate it to get stereo
        audio = torch.repeat_interleave(audio, 2, dim=1)

    if rate != model_rate:
        warnings.warn("resample to model sample rate")
        # we have to resample to model samplerate if needed
        # this makes sure we resample input only once
        resampler = torchaudio.transforms.Resample(
            orig_freq=rate, new_freq=model_rate, resampling_method="sinc_interpolation"
        ).to(audio.device)
        audio = resampler(audio)
    return audio
