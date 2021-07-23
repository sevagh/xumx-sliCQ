import argparse
import torch
import subprocess
import time
from pathlib import Path
import tqdm
import json
import numpy as np
import random
from git import Repo
import os
import signal
import atexit
import gc
import io
import copy
import sys
import torchaudio
import torchinfo
from contextlib import contextmanager
import sklearn.preprocessing
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from openunmix import data
from openunmix import model
from openunmix import utils
from openunmix import transforms
from openunmix import filtering
from openunmix.loss import LossCriterion

tqdm.monitor_interval = 0


def train(args, unmix, encoder, device, train_sampler, criterion, optimizer):
    # unpack encoder object
    nsgt, _, cnorm = encoder

    losses = utils.AverageMeter()
    unmix.train()
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)

    for x, y_bass, y_vocals, y_other, y_drums in pbar:
        pbar.set_description("Training batch")
        x, y_bass, y_vocals, y_other, y_drums = x.to(device), y_bass.to(device), y_vocals.to(device), y_other.to(device), y_drums.to(device)
        optimizer.zero_grad()
        X = nsgt(x)
        Xmag = cnorm(X)
        Ymag_bass = cnorm(nsgt(y_bass))
        Ymag_vocals = cnorm(nsgt(y_vocals))
        Ymag_drums = cnorm(nsgt(y_drums))
        Ymag_other = cnorm(nsgt(y_other))

        loss = criterion(
            *unmix(Xmag), # forward call to unmix returns bass, vocals, other, drums
            *(Ymag_bass, Ymag_vocals, Ymag_other, Ymag_drums),
            *(y_bass, y_vocals, y_other, y_drums),
            X,
            x.shape[-1]
        )

        loss.backward()
        optimizer.step()
        losses.update(loss.item(), x.size(1))

    return losses.avg


def valid(args, unmix, encoder, device, valid_sampler, criterion):
    # unpack encoder object
    nsgt, _, cnorm = encoder

    losses = utils.AverageMeter()
    unmix.eval()

    with torch.no_grad():
        pbar = tqdm.tqdm(valid_sampler, disable=args.quiet)
        for x, y_bass, y_vocals, y_other, y_drums in pbar:
            pbar.set_description("Validation batch")
            x, y_bass, y_vocals, y_other, y_drums = x.to(device), y_bass.to(device), y_vocals.to(device), y_other.to(device), y_drums.to(device)

            X = nsgt(x)
            Xmag = cnorm(X)

            Ymag_bass = cnorm(nsgt(y_bass))
            Ymag_vocals = cnorm(nsgt(y_vocals))
            Ymag_drums = cnorm(nsgt(y_drums))
            Ymag_other = cnorm(nsgt(y_other))

            loss = criterion(
                *unmix(Xmag), # forward call to unmix returns bass, vocals, other, drums
                *(Ymag_bass, Ymag_vocals, Ymag_other, Ymag_drums),
                *(y_bass, y_vocals, y_other, y_drums),
                X,
                x.shape[-1]
            )

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
            X_block_ola = np.squeeze(transforms.overlap_add_slicq(X_block).mean(1, keepdim=False).permute(0, 2, 1), axis=0)
            scalers[i].partial_fit(X_block_ola)

    # set inital input scaler values
    std = [np.maximum(scaler.scale_, 1e-4 * np.max(scaler.scale_)) for scaler in scalers]
    return [scaler.mean_ for scaler in scalers], std


def main():
    parser = argparse.ArgumentParser(description="Open Unmix Trainer")

    # Dataset paramaters
    parser.add_argument(
        "--dataset",
        type=str,
        default="musdb",
        choices=[
            "musdb",
            "aligned",
            "sourcefolder",
            "trackfolder_var",
            "trackfolder_fix",
        ],
        help="Name of the dataset.",
    )
    parser.add_argument("--root", type=str, help="root path of dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="open-unmix",
        help="provide output path base folder name",
    )
    parser.add_argument("--model", type=str, help="Path to checkpoint folder")
    parser.add_argument(
        "--audio-backend",
        type=str,
        default="soundfile",
        help="Set torchaudio backend (`sox_io` or `soundfile`",
    )

    # Training Parameters
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate, defaults to 1e-3")
    parser.add_argument(
        "--patience",
        type=int,
        default=140,
        help="maximum number of train epochs (default: 140)",
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
    parser.add_argument("--weight-decay", type=float, default=0.00001, help="weight decay")
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument('--mcoef', type=float, default=0.1,
                        help='coefficient for mixing: mcoef*SDR_Loss + MSE_Loss')

    # Model Parameters
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="skip dataset statistics calculation",
    )
    parser.add_argument(
        "--seq-dur",
        type=float,
        default=6.0,
        help="Sequence duration in seconds" "value of <=0.0 will use full/variable length",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=16000.,
        help="network won't consider frequencies above this"
    )
    parser.add_argument(
        "--fscale",
        choices=('bark','mel', 'cqlog', 'vqlog', 'oct'),
        default='bark',
        help="frequency scale for sliCQ-NSGT",
    )
    parser.add_argument(
        "--sllen",
        type=int,
        default=4096,
        help="slicq slice length",
    )
    parser.add_argument(
        "--fbins",
        type=int,
        default=60,
        help="number of frequency bins for NSGT scale",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=20.,
        help="min frequency for NSGT scale",
    )
    parser.add_argument(
        "--nb-channels",
        type=int,
        default=2,
        help="set number of channels for model (1, 2)",
    )
    parser.add_argument(
        "--nb-workers", type=int, default=0, help="Number of workers for dataloader."
    )

    # Misc Parameters
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="less verbose during training",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--enable-sdr-loss", action="store_true", default=False, help="enable SDR loss"
    )
    parser.add_argument(
        "--cuda-device", type=int, default=-1, help="choose which gpu to train on (-1 = 'cuda' in pytorch)"
    )

    args, _ = parser.parse_known_args()

    torchaudio.set_audio_backend(args.audio_backend)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Using GPU:", use_cuda)

    if use_cuda:
        print("Configuring NSGT to use GPU")

    dataloader_kwargs = {"num_workers": args.nb_workers, "pin_memory": True} if use_cuda else {}

    try:
        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        repo = Repo(repo_dir)
        commit = repo.head.commit.hexsha[:7]
    except:
        commit = 'n/a'

    # use jpg or npy
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda and args.cuda_device >= 0:
        device = torch.device(args.cuda_device)

    train_dataset, valid_dataset, args = data.load_datasets(parser, args)

    # create output dir if not exist
    target_path = Path(args.output)
    target_path.mkdir(parents=True, exist_ok=True)

    tboard_path = target_path / f"logdir"
    tboard_writer = SummaryWriter(tboard_path)

    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs
    )

    valid_sampler = torch.utils.data.DataLoader(valid_dataset, batch_size=1, **dataloader_kwargs)

    # need to globally configure an NSGT object to peek at its params to set up the neural network
    # e.g. M depends on the sllen which depends on fscale+fmin+fmax
    nsgt_base = transforms.NSGTBase(
        args.fscale,
        args.fbins,
        args.fmin,
        args.sllen,
        fs=train_dataset.sample_rate,
        device=device
    )

    nsgt, insgt = transforms.make_filterbanks(
        nsgt_base, sample_rate=train_dataset.sample_rate
    )
    cnorm = model.ComplexNorm(mono=args.nb_channels == 1)

    nsgt = nsgt.to(device)
    insgt = insgt.to(device)
    cnorm = cnorm.to(device)
    
    # pack the 3 pieces of the encoder/decoder
    encoder = (nsgt, insgt, cnorm)

    separator_conf = {
        "sample_rate": train_dataset.sample_rate,
        "nb_channels": args.nb_channels,
        "seq_dur": args.seq_dur, # have to do inference in chunks of seq_dur in CNN architecture
    }

    with open(Path(target_path, "separator.json"), "w") as outfile:
        outfile.write(json.dumps(separator_conf, indent=4, sort_keys=True))

    jagged_slicq = nsgt_base.predict_input_size(args.batch_size, args.nb_channels, args.seq_dur)

    jagged_slicq = cnorm(jagged_slicq)
    n_blocks = len(jagged_slicq)

    # data whitening
    if args.model or args.debug:
        scaler_mean = None
        scaler_std = None
    else:
        scaler_mean, scaler_std = get_statistics(args, encoder, train_dataset, n_blocks)

    unmix = model.OpenUnmix(
        jagged_slicq,
        max_bin=nsgt_base.max_bins(args.bandwidth),
        input_means=scaler_mean,
        input_scales=scaler_std,
    ).to(device)

    if not args.quiet:
        torchinfo.summary(unmix, input_data=(jagged_slicq,))

    optimizer = torch.optim.Adam(unmix.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    criterion = LossCriterion(encoder, args.mcoef, enable_sdr=args.enable_sdr_loss)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10,
    )

    es = utils.EarlyStopping(patience=args.patience)

    # if a model is specified: resume training
    if args.model:
        model_path = Path(args.model).expanduser()
        with open(Path(model_path, "xumx_slicq.json"), "r") as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, "xumx_slicq.chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        unmix.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        # train for another epochs_trained
        t = tqdm.trange(
            results["epochs_trained"],
            results["epochs_trained"] + args.epochs + 1,
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

    print('Starting Tensorboard')
    tboard_proc = subprocess.Popen(["tensorboard", "--logdir", tboard_path])
    tboard_pid = tboard_proc.pid

    def kill_tboard():
        if tboard_pid is None:
            pass
        print('Killing backgrounded Tensorboard process...')
        os.kill(tboard_pid, signal.SIGTERM)

    atexit.register(kill_tboard)

    for epoch in t:
        t.set_description("Training Epoch")
        end = time.time()
        train_loss = train(args, unmix, encoder, device, train_sampler, criterion, optimizer)
        valid_loss = valid(args, unmix, encoder, device, valid_sampler, criterion)

        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(train_loss=train_loss, val_loss=valid_loss)

        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch

        utils.save_checkpoint(
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
            "args": vars(args),
            "best_loss": es.best,
            "best_epoch": best_epoch,
            "train_loss_history": train_losses,
            "valid_loss_history": valid_losses,
            "train_time_history": train_times,
            "num_bad_epochs": es.num_bad_epochs,
            "commit": commit,
        }

        with open(Path(target_path, "xumx_slicq.json"), "w") as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)

        if tboard_writer is not None:
            tboard_writer.add_scalar('Loss/train', train_loss, epoch)
            tboard_writer.add_scalar('Loss/valid', valid_loss, epoch)

        if stop:
            print("Apply Early Stopping")
            break

        gc.collect()


if __name__ == "__main__":
    main()
