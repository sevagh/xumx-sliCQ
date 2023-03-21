import argparse
import torch
import subprocess
import time
from pathlib import Path
import tqdm
import json
import numpy as np
import random
import os
import signal
import atexit
import gc
import copy
import sys
import torchaudio
import torchinfo
from contextlib import nullcontext
import sklearn.preprocessing
from torch.utils.tensorboard import SummaryWriter

from .data import MUSDBDataset, PeripheryDataset, custom_collate

from xumx_slicq_v2 import model
from xumx_slicq_v2 import transforms
from xumx_slicq_v2.separator import Separator
from xumx_slicq_v2.loss import ComplexMSELossCriterion, MaskSumLossCriterion, SDRLossCriterion

tqdm.monitor_interval = 0


def loop(
    args,
    unmix,
    encoder,
    device,
    sampler,
    mse_criterion,
    sdr_criterion,
    mask_criterion,
    optimizer,
    amp_cm_cuda,
    amp_cm_cpu,
    train=True,
):
    # unpack encoder object
    nsgt, insgt, cnorm = encoder

    losses = _AverageMeter()

    grad_cm = nullcontext
    name = ""
    if train:
        unmix.train()
        name = "Train"
    else:
        unmix.eval()
        name = "Validation"
        grad_cm = torch.no_grad

    pbar = tqdm.tqdm(sampler, disable=args.quiet)

    with grad_cm():
        for track_tensor in pbar:
            pbar.set_description(f"{name} batch")

            # autocast/AMP on forward pass + loss only, _not_ backward pass
            with amp_cm_cuda(), amp_cm_cpu():
                track_tensor_gpu = track_tensor.to(device).swapaxes(0, 1)

                x = track_tensor_gpu[0]

                y_targets = track_tensor_gpu[1:]

                Xcomplex = nsgt(x)

                Ycomplex_ests, Ymasks = unmix(Xcomplex, return_masks=True)

                Ycomplex_targets = nsgt(y_targets)

                if sdr_criterion is not None:
                    y_ests = insgt(Ycomplex_ests, x.shape[-1])

                mse_loss = mse_criterion(
                    Ycomplex_ests,
                    Ycomplex_targets,
                )

                if sdr_criterion is not None:
                    sdr_loss = args.sdr_mcoef*sdr_criterion(
                        y_ests,
                        y_targets,
                    )
                else:
                    sdr_loss = 0.

                mask_mse_loss = mask_criterion(
                    Ymasks
                )

                loss = mse_loss + mask_mse_loss + sdr_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.update(loss.item(), x.size(1))

    return losses.avg


def get_statistics(args, encoder, dataset, time_blocks):
    nsgt, _, cnorm = encoder

    nsgt = copy.deepcopy(nsgt).to("cpu")
    cnorm = copy.deepcopy(cnorm).to("cpu")

    # slicq is a list of tensors so we need a list of scalers
    scalers = [sklearn.preprocessing.StandardScaler() for i in range(time_blocks)]

    dataset_scaler = copy.deepcopy(dataset)
    dataset_scaler.random_chunks = False
    dataset_scaler.seq_duration = None

    dataset_scaler.samples_per_track = 1
    dataset_scaler.augmentations = None
    dataset_scaler.random_track_mix = False
    dataset_scaler.random_interferer_mix = False

    pbar = tqdm.tqdm(range(len(dataset_scaler)), disable=args.quiet)
    for ind in pbar:
        x = dataset_scaler[ind][0]

        pbar.set_description("Compute dataset statistics")
        # downmix to mono channel
        X = cnorm(nsgt(x[None, ...]))

        for i, X_block in enumerate(X):
            X_block_flat = np.squeeze(
                torch.flatten(X_block, start_dim=-2, end_dim=-1)
                .mean(1, keepdim=False)
                .permute(0, 2, 1),
                axis=0,
            )
            scalers[i].partial_fit(X_block_flat)

    # set inital input scaler values
    std = [
        np.maximum(scaler.scale_, 1e-4 * np.max(scaler.scale_)) for scaler in scalers
    ]
    return [scaler.mean_ for scaler in scalers], std


def training_main():
    parser = argparse.ArgumentParser(description="xumx-sliCQ-V2 Trainer")

    # Dataset paramaters
    parser.add_argument("--musdb-root", type=str, default="/MUSDB18-HQ")
    parser.add_argument("--periphery-root", type=str, default="/Periphery")
    parser.add_argument("--samples-per-track", type=int, default=64)
    parser.add_argument(
        "--periphery-dataset",
        action="store_true",
        default=False,
        help="use periphery dataset instead of musdb",
    )

    # Training Parameters
    parser.add_argument("--model-path", type=str, default="/model")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-size-valid", type=int, default=1)
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate, defaults to 1e-3"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=1000,
        help="maximum number of train epochs (default: 1000)",
    )
    parser.add_argument(
        "--lr-decay-patience",
        type=int,
        default=80,
        help="lr decay patience for plateau scheduler",
    )
    parser.add_argument(
        "--lr-decay-gamma",
        type=float,
        default=0.3,
        help="gamma of learning rate scheduler decay",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.00001, help="weight decay"
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )

    # Model Parameters
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="skip dataset statistics calculation",
    )
    parser.add_argument(
        "--sdr-mcoef",
        type=float,
        default=-1.,
        help="enable SDR loss if mcoef > 0.0",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        default=False,
        help="realtime variant (causal convs, no wiener)",
    )
    parser.add_argument(
        "--lstm",
        action="store_true",
        default=False,
        help="use original UMX lstm nn architecture",
    )
    parser.add_argument(
        "--seq-dur",
        type=float,
        default=2.0,
        help="Sequence duration in seconds"
        "value of <=0.0 will use full/variable length",
    )
    parser.add_argument(
        "--fscale",
        choices=("bark", "mel", "cqlog", "vqlog", "linear", "mrstft"),
        default="bark",
        help="frequency scale for sliCQ-NSGT",
    )
    parser.add_argument(
        "--fbins",
        type=int,
        default=262,
        help="number of frequency bins for NSGT scale",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=32.9,
        help="min frequency for NSGT scale",
    )
    parser.add_argument(
        "--fgamma",
        type=float,
        default=0.0,
        help="gamma for variable-Q offset",
    )
    parser.add_argument(
        "--nb-workers", type=int, default=4, help="Number of workers for dataloader."
    )

    # Misc Parameters
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="less verbose during training",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=-1,
        help="choose which gpu to train on (-1 = 'cuda' in pytorch)",
    )

    args = parser.parse_args()

    torchaudio.set_audio_backend("soundfile")
    use_cuda = torch.cuda.is_available()
    print("Using GPU:", use_cuda)

    if use_cuda:
        print("Configuring NSGT to use GPU")

    dataloader_kwargs = (
        {"num_workers": args.nb_workers, "pin_memory": True} if use_cuda else {}
    )

    # use jpg or npy
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda and args.cuda_device >= 0:
        device = torch.device(args.cuda_device)

    if not args.periphery_dataset:
        train_dataset, valid_dataset = MUSDBDataset.load_datasets(
            args.seed,
            args.seq_dur,
            samples_per_track=args.samples_per_track,
            musdb_root=args.musdb_root,
        )
    else:
        train_dataset, valid_dataset = PeripheryDataset.load_datasets(
            args.seed,
            args.seq_dur,
            samples_per_track=args.samples_per_track,
            periphery_root=args.periphery_root,
        )

    # create output dir if not exist
    target_path = Path(args.model_path)
    target_path.mkdir(parents=True, exist_ok=True)

    # check if it already contains a pytorch model
    model_exists = False
    for file in os.listdir(target_path):
        if file.endswith(".pth") or file.endswith(".chkpnt"):
            model_exists = True
            break

    tboard_path = target_path / f"logdir"
    tboard_writer = SummaryWriter(tboard_path)

    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs
    )

    valid_sampler = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size_valid,
        collate_fn=custom_collate,
        **dataloader_kwargs,
    )

    # need to globally configure an NSGT object to peek at its params to set up the neural network
    # e.g. M depends on the sllen which depends on fscale+fmin+fmax
    nsgt_base = transforms.NSGTBase(
        args.fscale,
        args.fbins,
        args.fmin,
        fgamma=args.fgamma,
        fs=train_dataset.sample_rate,
        device=device,
    )

    nsgt, insgt = transforms.make_filterbanks(
        nsgt_base, sample_rate=train_dataset.sample_rate
    )
    cnorm = transforms.ComplexNorm()

    nsgt = nsgt.to(device)
    insgt = insgt.to(device)
    cnorm = cnorm.to(device)

    # pack the 3 pieces of the encoder/decoder
    encoder = (nsgt, insgt, cnorm)

    jagged_slicq, sample_waveform = nsgt_base.predict_input_size(
        args.batch_size, 2, args.seq_dur
    )

    jagged_slicq_cnorm = cnorm(jagged_slicq)
    n_blocks = len(jagged_slicq)

    # data whitening
    if model_exists or args.debug:
        scaler_mean = None
        scaler_std = None
    else:
        scaler_mean, scaler_std = get_statistics(args, encoder, train_dataset, n_blocks)

    unmix = model.Unmix(
        jagged_slicq_cnorm,
        realtime=args.realtime,
        lstm=args.lstm,
        input_means=scaler_mean,
        input_scales=scaler_std,
    ).to(device, memory_format=torch.channels_last)

    if '2.0.0' in torch.__version__:
        print("Running torch.compile (pytorch 2.0)...")
        unmix = torch.compile(unmix)

    if not args.quiet:
        torchinfo.summary(unmix, input_data=(jagged_slicq,))

    optimizer = torch.optim.AdamW(
        unmix.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    mse_criterion = ComplexMSELossCriterion()
    sdr_criterion = None
    if args.sdr_mcoef > 0.0:
        sdr_criterion = SDRLossCriterion()
    mask_criterion = MaskSumLossCriterion()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10,
    )

    es = _EarlyStopping(patience=args.patience)

    # if a model is specified: resume training
    if model_exists:
        print("Model exists, resuming training...")

        with open(Path(target_path, "xumx_slicq_v2.json"), "r") as stream:
            results = json.load(stream)

        target_model_path = Path(target_path, f"xumx_slicq_v2.chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        unmix.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        # train for another epochs_trained
        t = tqdm.trange(
            results["epochs_trained"],
            args.epochs + 1,
            initial=results["epochs_trained"],
            total=args.epochs,
            disable=args.quiet,
        )
        train_losses = results["train_loss_history"]
        valid_losses = results["valid_loss_history"]
        train_times = results["train_time_history"]
        best_epoch = results["best_epoch"]
        es.best = results["best_loss"]
        es.num_bad_epochs = results["num_bad_epochs"]
    # else start from 0
    else:
        t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
        train_losses = []
        valid_losses = []
        train_times = []
        best_epoch = 0

    print("Starting Tensorboard")
    tboard_proc = subprocess.Popen(["tensorboard", "--logdir", tboard_path])
    tboard_pid = tboard_proc.pid

    def kill_tboard():
        if tboard_pid is None:
            pass
        print("Killing backgrounded Tensorboard process...")
        os.kill(tboard_pid, signal.SIGTERM)

    atexit.register(kill_tboard)

    ######################
    # PERFORMANCE TUNING #
    ######################
    print("Performance tuning settings")
    print("Enabling cuDNN benchmark...")
    torch.backends.cudnn.benchmark = True

    print("Enabling FP32 (ampere) optimizations for matmul and cudnn")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision = "medium"

    print(
        "Enabling CUDA+CPU Automatic Mixed Precision with for forward pass + loss"
    )

    # lstm can only support float16 optimizations
    cuda_autocast_dtype = torch.bfloat16 if not args.lstm else torch.float16
    amp_cm_cuda = lambda: torch.autocast("cuda", dtype=cuda_autocast_dtype)

    amp_cm_cpu = lambda: torch.autocast("cpu", dtype=torch.bfloat16)

    for epoch in t:
        t.set_description("Training Epoch")
        end = time.time()
        train_loss = loop(
            args,
            unmix,
            encoder,
            device,
            train_sampler,
            mse_criterion,
            sdr_criterion,
            mask_criterion,
            optimizer,
            amp_cm_cuda,
            amp_cm_cpu,
            train=True,
        )
        valid_loss = loop(
            args,
            unmix,
            encoder,
            device,
            valid_sampler,
            mse_criterion,
            sdr_criterion,
            mask_criterion,
            None,
            amp_cm_cuda,
            amp_cm_cpu,
            train=False,
        )

        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(train_loss=train_loss, val_loss=valid_loss)

        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch

        _save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": unmix.state_dict(),
                "best_loss": es.best,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best=valid_loss == es.best,
            path=target_path,
        )

        # save params
        params = {
            "epochs_trained": epoch,
            "args": {**vars(args), **{"sample_rate": train_dataset.sample_rate, "nb_channels": 2, "seq_dur": args.seq_dur}},
            "best_loss": es.best,
            "best_epoch": best_epoch,
            "train_loss_history": train_losses,
            "valid_loss_history": valid_losses,
            "train_time_history": train_times,
            "num_bad_epochs": es.num_bad_epochs,
        }

        with open(Path(target_path, "xumx_slicq_v2.json"), "w") as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)

        if tboard_writer is not None:
            tboard_writer.add_scalar("Loss/train", train_loss, epoch)
            tboard_writer.add_scalar("Loss/valid", valid_loss, epoch)

        if stop:
            print("Apply Early Stopping")
            break

        # do everything i can to avoid crashing
        gc.collect()
        torch.cuda.empty_cache()


def _save_checkpoint(state: dict, is_best: bool, path: str):
    # save full checkpoint including optimizer
    torch.save(state, os.path.join(path, "xumx_slicq_v2.chkpnt"))
    if is_best:
        # save just the weights
        torch.save(state["state_dict"], os.path.join(path, "xumx_slicq_v2.pth"))


class _AverageMeter(object):
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


class _EarlyStopping(object):
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


if __name__ == "__main__":
    training_main()
