import argparse
import torch
import time
from pathlib import Path
import tqdm
import json
import sklearn.preprocessing
import numpy as np
import random
from git import Repo
import os
import copy
import sys
import torchaudio
from contextlib import contextmanager
from torch.autograd import Variable

from openunmix import data
from openunmix import model
from openunmix import utils
from openunmix import transforms
from openunmix import filtering

tqdm.monitor_interval = 0

def train(args, unmix, encoder, device, train_sampler, optimizer):
    losses = utils.AverageMeter()
    unmix.train()
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)
    for x, y in pbar:
        pbar.set_description("Training batch")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        X = encoder(x)
        Y_hat = unmix(X)
        Y = encoder(y)
        loss = torch.nn.functional.mse_loss(Y_hat, Y)
        loss.backward()
        #unmix.on_after_backward()

        #####################
        # gradient clipping #
        #####################

        #clipping_value = 1 # arbitrary value of your choosing
        #torch.nn.utils.clip_grad_norm_(unmix.parameters(), clipping_value)
        optimizer.step()
        losses.update(loss.item(), Y.size(1))
    return losses.avg


def valid(args, unmix, encoder, device, valid_sampler):
    losses = utils.AverageMeter()
    unmix.eval()
    with torch.no_grad():
        for x, y in valid_sampler:
            x, y = x.to(device), y.to(device)
            X = encoder(x)
            Y_hat = unmix(X)
            Y = encoder(y)
            loss = torch.nn.functional.mse_loss(Y_hat, Y)
            losses.update(loss.item(), Y.size(1))
        return losses.avg


def get_statistics(args, encoder, dataset):
    scaler = sklearn.preprocessing.StandardScaler()

    dataset_scaler = copy.deepcopy(dataset)
    if isinstance(dataset_scaler, data.SourceFolderDataset):
        dataset_scaler.random_chunks = False
    else:
        dataset_scaler.random_chunks = False
        dataset_scaler.seq_duration = args.stats_seq_dur

    dataset_scaler.samples_per_track = 1
    dataset_scaler.augmentations = None
    dataset_scaler.random_track_mix = False
    dataset_scaler.random_interferer_mix = False

    pbar = tqdm.tqdm(range(len(dataset_scaler)), disable=args.quiet)

    for ind in pbar:
        x, y = dataset_scaler[ind]
        pbar.set_description("Compute dataset statistics")

        # downmix to mono channel
        # norm across frequency bins
        X = encoder(x[None, ...]).mean(1, keepdim=False).permute(0, 2, 1, 3)
        X = X.reshape(-1, X.shape[-2])
        scaler.partial_fit(np.squeeze(X.cpu().detach()))

    # set inital input scaler values
    std = np.maximum(scaler.scale_, 1e-4 * np.max(scaler.scale_))
    return scaler.mean_, std


def main():
    parser = argparse.ArgumentParser(description="Open Unmix Trainer")

    # which target do we want to train?
    parser.add_argument(
        "--target",
        type=str,
        default="vocals",
        help="target source (will be passed to the dataset)",
    )

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
    parser.add_argument("--valid-batch-size", type=int, default=16)
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

    # Model Parameters
    parser.add_argument(
        "--seq-dur",
        type=float,
        default=6.0,
        help="Sequence duration in seconds" "value of <=0.0 will use full/variable length",
    )
    parser.add_argument(
        "--stats-seq-dur",
        type=float,
        default=180.0,
        help="Sequence duration in seconds for limiting full track size for feature scaling",
    )
    parser.add_argument(
        "--valid-seq-dur",
        type=float,
        default=30.0,
        help="Sequence duration in seconds for limiting full track size for validation",
    )
    parser.add_argument(
        "--unidirectional",
        action="store_true",
        default=False,
        help="Use unidirectional LSTM",
    )
    parser.add_argument(
        "--fscale",
        choices=('bark','mel','cqlog','vqlog'),
        default='bark',
        help="Sequence duration in seconds for limiting full track size for validation",
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
        "--gamma",
        type=float,
        default=25.,
        help="variable-q offset in hz (only used in vqlog scale)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=3,
        help="RNN layers",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Speed up training init for dev purposes",
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

    train_dataset, valid_dataset, args = data.load_datasets(parser, args)

    # create output dir if not exist
    target_path = Path(args.output)
    target_path.mkdir(parents=True, exist_ok=True)

    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs
    )

    # use batch/2 validation samples
    valid_sampler = torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size, **dataloader_kwargs)

    # need to globally configure an NSGT object to peek at its params to set up the neural network
    # e.g. M depends on the sllen which depends on fscale+fmin+fmax
    nsgt_base = transforms.NSGTBase(
        args.fscale,
        args.fbins,
        args.fmin,
        args.sllen,
        gamma=args.gamma,
        fs=train_dataset.sample_rate,
        device=device
    )

    nsgt, _ = transforms.make_filterbanks(
        nsgt_base, sample_rate=train_dataset.sample_rate
    )

    encoder = torch.nn.Sequential(nsgt, model.ComplexNorm(mono=args.nb_channels == 1)).to(device)

    separator_conf = {
        "sample_rate": train_dataset.sample_rate,
        "nb_channels": args.nb_channels,
    }

    with open(Path(target_path, "separator.json"), "w") as outfile:
        outfile.write(json.dumps(separator_conf, indent=4, sort_keys=True))

    if args.model or args.debug:
        scaler_mean = None
        scaler_std = None
    else:
        scaler_mean, scaler_std = get_statistics(args, encoder, train_dataset)
        scaler_mean = torch.tensor(scaler_mean, device=device)
        scaler_std = torch.tensor(scaler_std, device=device)

    unmix = model.OpenUnmix(
        nsgt_base.fbins_actual,
        nsgt_base.M,
        input_mean=scaler_mean,
        input_scale=scaler_std,
        nb_channels=args.nb_channels,
        unidirectional=args.unidirectional,
        nb_layers=args.layers,
    ).to(device)

    print('unmix model:\n{0}'.format(unmix))

    optimizer = torch.optim.Adam(unmix.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
        with open(Path(model_path, args.target + ".json"), "r") as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, args.target + ".chkpnt")
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

    for epoch in t:
        t.set_description("Training Epoch")
        end = time.time()
        train_loss = train(args, unmix, encoder, device, train_sampler, optimizer)
        valid_loss = valid(args, unmix, encoder, device, valid_sampler)
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
            target=args.target,
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

        with open(Path(target_path, args.target + ".json"), "w") as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)

        if stop:
            print("Apply Early Stopping")
            break


if __name__ == "__main__":
    main()
