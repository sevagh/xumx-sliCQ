import argparse
import functools
import json
import gc
from typing import Optional, Union
import numpy as np
import museval
import musdb
import torch
import tqdm
import random

from .separator import Separator
from .data import preprocess_audio


def separate_and_evaluate(
    separator: Separator,
    track: musdb.MultiTrack,
    device: Union[str, torch.device] = "cpu",
) -> str:

    print("getting audio")
    audio = torch.as_tensor(track.audio, dtype=torch.float32, device=device)
    audio = preprocess_audio(audio, track.rate, separator.sample_rate)

    print("applying separation")
    estimates = separator(audio)

    estimates = separator.to_dict(estimates)

    for key in estimates:
        estimates[key] = estimates[key][0].detach().cpu().numpy().T

    print("evaluating bss...")

    bss_scores = museval.eval_mus_track(
        track,
        estimates,
    )

    gc.collect()
    return bss_scores


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="MUSDB18 Evaluation", add_help=False)

    parser.add_argument(
        "--track", type=str, default=None, help="evaluate only this track name"
    )

    parser.add_argument(
        "--model-path", type=str, default=None, help="custom model path"
    )

    parser.add_argument(
        "--subset", type=str, default="test", help="MUSDB subset (`train`/`test`)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=2621440, help="inference chunk size"
    )

    args = parser.parse_args()

    device = torch.device("cpu")

    mus = musdb.DB(
        root="/MUSDB18-HQ",
        download=False,
        subsets=args.subset,
        is_wav=True,
    )

    print("loading separator")
    separator = Separator.load(
        chunk_size=args.chunk_size,
        device=device,
        model_path=args.model_path,
    )

    tracks = mus.tracks
    if args.track is not None:
        tracks = [t for t in tracks if t.name == args.track]

    if len(tracks) == 0:
        raise ValueError("dataset is empty")

    total_scores = []

    results = museval.EvalStore()

    for track in tqdm.tqdm(tracks):
        print("track: {0}".format(track.name))
        track_scores = separate_and_evaluate(
            separator,
            track,
            device=device,
        )

        print(track, "\n", track_scores)
        results.add_track(track_scores)

    print(results)
