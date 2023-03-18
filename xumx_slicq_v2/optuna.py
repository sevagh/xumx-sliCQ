"""
copied from https://raw.githubusercontent.com/optuna/optuna-examples/main/pytorch/pytorch_simple.py
"""

import auraloss
import atexit
import os
import signal
import optuna
import subprocess
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from .data import MUSDBDataset, custom_collate
from xumx_slicq_v2 import model
from xumx_slicq_v2 import transforms
from xumx_slicq_v2.loss import ComplexMSELossCriterion, SDRLossCriterion


DEVICE = torch.device("cuda")
EPOCHS = 10
BATCHSIZE = 32
SEQ_DUR = 2.0
SAMPLE_RATE = 44100.0
NB_WORKERS = 4
SAMPLES_PER_TRACK = 16  # This limits training data for faster epochs.
MAX_VALID_SAMPLES = (
    2  # This limits validation data to avoid OOM, use two smaller tracks
)

# for optuna-dashboard
SQLITE_DB = "sqlite:////db.sqlite3"

# limit slicqt slice size to 1s in length
MAX_SLLEN = 44100


def define_model(trial):
    # try to avoid overly-large slicqts
    # fbins = trial.suggest_int("frequency_bins", 100, 700)
    # fmin = trial.suggest_float("frequency_min", 5., 100.)
    # fscale = trial.suggest_categorical("fscale", ["bark", "mel"]) # as usual, vqlog/cqlog end up being a bust

    fbins = 262
    fscale = "bark"
    fmin = 32.9
    # fmax = trial.suggest_float("fmax", 10000., 22050.)

    # sliCQT config is a hyperparam
    nsgt_base = transforms.NSGTBase(
        fscale,
        fbins,
        fmin,
        # fmax=fmax,
        fs=SAMPLE_RATE,
        device=DEVICE,
    )

    if nsgt_base.sllen > MAX_SLLEN:
        raise ValueError(f"sllen {nsgt_base.sllen} exceeds {MAX_SLLEN}, discarding")

    hidden_size_1 = trial.suggest_int("hidden_size_1", 4, 256)
    hidden_size_2 = trial.suggest_int("hidden_size_2", 4, 256)
    time_filter_2 = trial.suggest_int("time_filter_2", 1, 9)

    # freq_thresh_small = trial.suggest_int("freq_thresh_small", 5, 10)
    # freq_thresh_medium = trial.suggest_int("freq_thresh_medium", 10, 40)
    # freq_filter_medium = trial.suggest_int("freq_filter_medium", 1, 7)
    # freq_filter_large = trial.suggest_int("freq_filter_large", 3, 9)

    nsgt, insgt = transforms.make_filterbanks(nsgt_base, sample_rate=SAMPLE_RATE)
    cnorm = transforms.ComplexNorm()

    nsgt = nsgt.to(DEVICE)
    insgt = insgt.to(DEVICE)
    cnorm = cnorm.to(DEVICE)

    # pack the 3 pieces of the encoder/decoder
    encoder = (nsgt, insgt, cnorm)

    jagged_slicq, _ = nsgt_base.predict_input_size(BATCHSIZE, 2, SEQ_DUR)

    jagged_slicq_cnorm = cnorm(jagged_slicq)
    n_blocks = len(jagged_slicq)

    unmix = model.Unmix(
        jagged_slicq_cnorm,
        hidden_size_1=hidden_size_1,
        hidden_size_2=hidden_size_2,
        # freq_filter_large=freq_filter_large,
        # freq_filter_medium=freq_filter_medium,
        # freq_thresh_small=freq_thresh_small,
        # freq_thresh_medium=freq_thresh_medium,
        time_filter_2=time_filter_2,
    ).to(DEVICE)

    return unmix, encoder


def get_musdb():
    # Load MUSDB dataset.
    dataloader_kwargs = {"num_workers": NB_WORKERS, "pin_memory": True}

    train_dataset, valid_dataset = MUSDBDataset.load_datasets(
        42,  # fixed seed
        SEQ_DUR,  # fixed sequence duration
        samples_per_track=SAMPLES_PER_TRACK,  # cut down samples per track
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCHSIZE, shuffle=True, **dataloader_kwargs
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,  # fixed batch size of 1 for validation
        **dataloader_kwargs,
    )
    return train_loader, valid_loader


def objective(trial):
    # Generate the model.
    model, encoder = define_model(trial)
    nsgt, insgt, cnorm = encoder

    # Generate the optimizers. Use a fixed AdamW with lr=0.001 to reduce total hyperparams
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Get the MUSDB dataset.
    train_loader, valid_loader = get_musdb()

    mse_loss = ComplexMSELossCriterion()
    sdr_loss = SDRLossCriterion()

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for track_tensor in train_loader:
            track_tensor_gpu = track_tensor.to(DEVICE).swapaxes(0, 1)
            x = track_tensor_gpu[0]
            y_targets = track_tensor_gpu[1:]
            Xcomplex = nsgt(x)
            Ycomplex_ests = model(Xcomplex)
            Ycomplex_targets = nsgt(y_targets)
            loss = mse_loss(
                Ycomplex_ests,
                Ycomplex_targets,
            )
            loss.backward()
            optimizer.step()

        # Validation of the model with SDR
        model.eval()
        sdr = 0.0
        with torch.no_grad():
            for i, track_tensor in enumerate(valid_loader):
                if i >= MAX_VALID_SAMPLES:
                    break
                track_tensor_gpu = track_tensor.to(DEVICE).swapaxes(0, 1)
                x = track_tensor_gpu[0]
                y_targets = track_tensor_gpu[1:]
                Xcomplex = nsgt(x)
                Ycomplex_ests = model(Xcomplex)
                Ycomplex_targets = nsgt(y_targets)
                nb_samples = x.shape[-1]
                y_ests = insgt(Ycomplex_ests, nb_samples)
                valid_loss = sdr_loss(
                    y_ests,
                    y_targets,
                )
                sdr += valid_loss

        sdr = sdr / float(MAX_VALID_SAMPLES)

        trial.report(sdr, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return sdr


if __name__ == "__main__":
    # lower SDR loss (already comes out negated by auraloss) is better
    study = optuna.create_study(
        direction="minimize",
        study_name="xumx-sliCQ-V2 tuning",
        storage=SQLITE_DB,
    )

    print("Starting Optuna-Dashboard")
    odash_proc = subprocess.Popen(
        ["optuna-dashboard", SQLITE_DB, "--host", "0.0.0.0", "--port", "6006"]
    )
    odash_pid = odash_proc.pid

    def kill_odash():
        if odash_pid is None:
            pass
        print("Killing backgrounded Optuna Dashboard process...")
        os.kill(odash_pid, signal.SIGTERM)

    atexit.register(kill_odash)

    study.optimize(
        objective,
        n_trials=100,
        timeout=None,
        catch=(
            RuntimeError,  # handle invalid conv kernel sizes etc.
            ValueError,  # handle sllen too long
        ),
    )

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
