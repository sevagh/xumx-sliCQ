import argparse
import torch
import time
from pathlib import Path
import sklearn.preprocessing
import tqdm
import json
import numpy as np
import random
from git import Repo
import os
import io
import copy
import sys
import torchaudio
import torchinfo
from contextlib import contextmanager
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch_lr_finder import LRFinder, TrainDataLoaderIter
import auraloss
import torchvision
import torchvision.io
from torch.utils.tensorboard import SummaryWriter

from openunmix import data
from openunmix import model
from openunmix import utils
from openunmix import transforms
from openunmix import filtering

tqdm.monitor_interval = 0
torch.autograd.set_detect_anomaly(True)

_BIG_SPLIT = 5_000_000
_PLOT_LIM = 16_000


# apply it on dict entries
def custom_mse_loss(output, target):
    loss = 0
    for time_bucket in target.keys():
        loss += torch.mean((output[time_bucket] - target[time_bucket])**2)
    return loss/len(target)


@profile
def train(args, unmix, encoder, device, train_sampler, mse_criterion, optimizer):
    # unpack encoder object
    nsgt, _, cnorm = encoder

    losses = utils.AverageMeter()
    unmix.train()
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)

    for x, y in pbar:
        pbar.set_description("Training batch")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        X = nsgt(x)
        Xmag = cnorm(X)
        Ymag = cnorm(nsgt(y))

        futures = {time_bucket: torch.jit.fork(unmix, str(time_bucket), Xmag_block) for time_bucket, Xmag_block in Xmag.items()}
        Ymag_hat = {time_bucket: torch.jit.wait(future) for time_bucket, future in futures.items()}

        loss = mse_criterion(
            Ymag_hat,
            Ymag,
        )

        loss.backward()
        optimizer.step()
        losses.update(loss.item(), y.size(1))

    return losses.avg, Xmag


def valid(args, unmix, encoder, device, valid_sampler, mse_criterion, sdr_criterion, slice_batch):
    # unpack encoder object
    nsgt, insgt, cnorm = encoder

    losses = utils.AverageMeter()
    sdrs = utils.AverageMeter()
    unmix.eval()

    ret_tup = None

    with torch.no_grad():
        pbar = tqdm.tqdm(valid_sampler, disable=args.quiet)
        for x, y in pbar:
            pbar.set_description("Validation batch")
            x, y = x.to(device), y.to(device)
            X = nsgt(x)
            Xmag = cnorm(X)
            Ymag = cnorm(nsgt(y))

            futures = {time_bucket: torch.jit.fork(unmix, str(time_bucket), Xmag_block) for time_bucket, Xmag_block in Xmag.items()}
            Ymag_hat = {time_bucket: torch.jit.wait(future) for time_bucket, future in futures.items()}

            loss = mse_criterion(
                Ymag_hat,
                Ymag,
            )

            losses.update(loss.item(), y.size(1))

            # randomly pick a set of spectrograms + audio clip to return
            # and to check sdr
            # set it if ret_tup is unset, and randomly set others
            if ret_tup is None or bool(random.getrandbits(1)):
                Y_hats = {}
                for time_bucket, X_block in X.items():
                    Y_hats[time_bucket] = transforms.phasemix_sep(X_block, Ymag_hat[time_bucket])

                y_hat = insgt(Y_hats, y.shape[-1])
                sdr = sdr_criterion(y_hat, y)

                sdrs.update(sdr.item(), y_hat.size(1))
                ret_tup = (y_hat, None, None, None)

        return losses.avg, sdrs.avg, ret_tup


def get_statistics(args, encoder, dataset):
    # unpack encoder object
    nsgt, _, cnorm = encoder
    encoder = torch.nn.Sequential(nsgt, cnorm)

    encoder = copy.deepcopy(encoder).to("cpu")

    scaler = sklearn.preprocessing.StandardScaler()

    dataset_scaler = copy.deepcopy(dataset)
    if isinstance(dataset_scaler, data.SourceFolderDataset):
        dataset_scaler.random_chunks = False
    else:
        dataset_scaler.random_chunks = False
        dataset_scaler.seq_duration = None

    dataset_scaler.samples_per_track = 1
    dataset_scaler.augmentations = None
    dataset_scaler.random_track_mix = False
    dataset_scaler.random_interferer_mix = False

    pbar = tqdm.tqdm(range(len(dataset_scaler)), disable=args.quiet)

    for ind in pbar:
        x, y = dataset_scaler[ind]
        pbar.set_description("Compute dataset statistics")
        N = x.shape[-1]

        # downmix to mono channel
        # norm across frequency bins
        Xmag = encoder(torch.unsqueeze(x, dim=0))
        Xmag_spectrogram = transforms.overlap_add_slicq(Xmag).mean(1, keepdim=False).permute(0, 2, 1)

        scaler.partial_fit(np.squeeze(Xmag_spectrogram.cpu().detach()))

    # set inital input scaler values
    return scaler.mean_, scaler.scale_


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
        "--print-shapes",
        action="store_true",
        default=False,
        help="Print shapes of data passing through network",
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

    tboard_path = target_path / f"logdir-{args.target}"
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

    if args.model or args.debug:
        scaler_mean = None
        scaler_std = None
    else:
        scaler_mean, scaler_std = get_statistics(args, encoder, train_dataset)
        scaler_mean = torch.tensor(scaler_mean, device=device)
        scaler_std = torch.tensor(scaler_std, device=device)


    jagged_slicq = nsgt_base.predict_input_size(args.batch_size, args.nb_channels, args.seq_dur)

    # small slice for inference to avoid oom
    slice_batch = next(iter(jagged_slicq.values())).shape[3]

    jagged_slicq = cnorm(jagged_slicq)

    unmix = model.OpenUnmix(
        jagged_slicq,
        input_mean=scaler_mean,
        input_scale=scaler_std,
        info=args.print_shapes,
    ).to(device)

    #torchinfo.summary(unmix, input_size=
    #    (
    #        jagged_slicq,
    #    )
    #)

    optimizer = torch.optim.Adam(unmix.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sdr_criterion = auraloss.time.SISDRLoss()
    #mse_criterion = torch.nn.MSELoss(reduction='mean')
    mse_criterion = custom_mse_loss

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

    fig, axs = plt.subplots(3)

    for epoch in t:
        t.set_description("Training Epoch")
        end = time.time()
        train_loss, last_Xmag = train(args, unmix, encoder, device, train_sampler, mse_criterion, optimizer)

        # set the 3 spectrograms to None for now until i write code to plot the non-matrixform spectrogram
        valid_loss, valid_sdr, (audio_sample, _, _, _) = valid(args, unmix, encoder, device, valid_sampler, mse_criterion, sdr_criterion, slice_batch)

        audio_sample = audio_sample[0].mean(dim=0, keepdim=True)

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

        if tboard_writer is not None:
            #nsgt.plot_spectrogram(20.*torch.log10(X_spec), axs[0])
            #nsgt.plot_spectrogram(20.*torch.log10(Y_spec), axs[1])
            #nsgt.plot_spectrogram(20.*torch.log10(Y_spec_hat), axs[2])

            #io_buf = io.BytesIO()
            #fig.savefig(io_buf, format='png', dpi=300)
            #io_buf.seek(0)
            #img = torch.ByteTensor(torch.ByteStorage.from_buffer(io_buf.getvalue()))

            #image = torchvision.io.decode_png(img)
            #tboard_writer.add_image(f'valid-sliCQ-{epoch}', image, global_step=epoch)

            tboard_writer.add_scalar('Loss (MSE)/train', train_loss, epoch)
            tboard_writer.add_scalar('Loss (MSE)/valid', valid_loss, epoch)
            tboard_writer.add_scalar('SI-SDR/valid', valid_sdr, epoch)
            tboard_writer.add_audio(f"valid-sep-{epoch}", audio_sample, global_step=epoch)

        if stop:
            print("Apply Early Stopping")
            break

    if tboard_writer is not None:
        tboard_writer.add_graph(unmix, (last_Xmag))
        for tag, param in unmix.named_parameters():
            if 'input' in tag:
                continue
            tboard_writer.add_histogram(tag, param.grad.data.cpu().numpy(), epoch)


if __name__ == "__main__":
    main()
