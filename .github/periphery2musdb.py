import os
import numpy
import itertools
import soundfile
import sys
import multiprocessing
import argparse

sample_rate = 44100


def write_stems(song_name_instruments_tup, root_dir):
    song_name, instruments = song_name_instruments_tup

    print(f'{song_name}')
    loaded_wavs = [None] * len(instruments)
    drums_idx = -1
    vocals_idx = -1
    bass_idx = -1
    for i, instrument in enumerate(instruments):
        stem = instrument.lower()
        if "drum" in stem:
            drums_idx = i
        elif "vocal" in stem:
            vocals_idx = i
        elif "bass" in stem:
            bass_idx = i

        loaded_wavs[i], rate = soundfile.read(instrument)
        assert rate == sample_rate
    track_len = len(loaded_wavs[0])

    # ensure all stems have the same length
    assert (
        len(loaded_wavs[i]) == track_len
        for i in range(1, len(loaded_wavs))
    )

    other_mix = sum(
        [
            l
            for i, l in enumerate(loaded_wavs)
            if i
            not in [
                drums_idx,
                vocals_idx,
                bass_idx,
            ]
        ]
    )
    full_mix = (
        loaded_wavs[drums_idx]
        + loaded_wavs[vocals_idx]
        + loaded_wavs[bass_idx]
        + other_mix
    )

    song_dir = os.path.join(root_dir, f"./{song_name}")

    if not os.path.isdir(song_dir):
        os.mkdir(song_dir)

    bass_path = os.path.join(song_dir, "bass.wav")
    drums_path = os.path.join(song_dir, "drums.wav")
    mix_path = os.path.join(song_dir, "mixture.wav")
    other_path = os.path.join(song_dir, "other.wav")
    vocals_path = os.path.join(song_dir, "vocals.wav")

    # individual stems
    soundfile.write(
        bass_path, loaded_wavs[bass_idx], sample_rate
    )
    soundfile.write(
        drums_path, loaded_wavs[drums_idx], sample_rate
    )
    soundfile.write(
        vocals_path, loaded_wavs[vocals_idx], sample_rate
    )

    # two mixes
    soundfile.write(
        other_path, other_mix, sample_rate
    )
    soundfile.write(
        mix_path, full_mix, sample_rate
    )



"""
take path to instrument stems
prepare vocal and non-vocal mixes
"""
def prepare_stems(
    stem_dirs, data_dir, n_pool
):

    # create output dirs
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    song_stems = []

    # first get a count of all tracks
    for sd in stem_dirs:
        for song in os.scandir(sd):
            song_name = song.name.replace('-', '_').replace(' ', '_')
            for dir_name, _, file_list in os.walk(song):
                instruments = [
                    os.path.join(dir_name, f) for f in file_list if f.endswith(".wav")
                ]
                if instruments:
                    song_stems.append((song_name, instruments))

    pool = multiprocessing.Pool(n_pool)

    pool.starmap(
        write_stems,
        zip(
            song_stems, # unpack song_name/instruments tuple in the function
            itertools.repeat(data_dir),
        ),
    )

    return 0


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-pool",
        type=int,
        default=multiprocessing.cpu_count(),
        help="size of python multiprocessing pool (default: %(default)s)",
    )
    parser.add_argument(
        "--outdir", type=str, default="./periphery-musdb", help="out dir"
    )
    parser.add_argument(
        "--stem-dirs", nargs="+", help="directories containing instrument stems"
    )

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    prepare_stems(
        args.stem_dirs,
        args.outdir,
        args.n_pool
    )

    sys.exit(0)
