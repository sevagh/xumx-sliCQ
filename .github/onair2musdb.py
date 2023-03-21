import os
import numpy
import itertools
from pathlib import Path
import soundfile
import sys
import multiprocessing
import argparse
import onair
from onair.config import stem_ids

sample_rate = 44100

"""
using:
    - https://github.com/sevagh/OnAir-Music-Dataset
    - https://github.com/sevagh/onair-py
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir", type=str, default="./onair-musdb", help="out dir"
    )
    parser.add_argument(
        "root", type=str, help="path to root onair music dataset"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # initiate musdb
    onair_db = onair.DB(root=args.root)

    for track in onair_db:
        print(f'track: {track}')
        track_stems = track.stems

        for stem, stem_idx in stem_ids.items():
            if stem == 'mix':
                stem = 'mixture'
            curr_track = track_stems[stem_idx, :, :]
            assert track.rate == sample_rate
            target_path = os.path.join(args.outdir, track.name)
            target_path = Path(target_path)
            target_path.mkdir(parents=True, exist_ok=True)
            stem_path = Path(f'{stem}.wav')
            total_path = Path(target_path / stem_path)
            soundfile.write(
                total_path, curr_track, track.rate
            )
            print(f'wrote audio for stem {stem}, stem index {stem_idx}, to {total_path}')


    sys.exit(0)
