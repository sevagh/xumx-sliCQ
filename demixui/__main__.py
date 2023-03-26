import argparse
import time
import torch
import torchaudio
from torchaudio.io import StreamWriter
import numpy
from pathlib import Path
import sys
from xumx_slicq_v2.separator import Separator
from xumx_slicq_v2.data import load_audio


def demixui_main():
    parser = argparse.ArgumentParser(description="RtDemixUi Demo")

    parser.add_argument(
        "--audio-backend",
        type=str,
        default="soundfile",
        help="Set torchaudio backend (`sox_io`, `sox`, `soundfile` or `stempeg`), defaults to `soundfile`",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=0,
        help="Run some inference warmup iterations",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=".github/gspi.wav",
        help="path to wav file input",
    )

    args = parser.parse_args()

    torchaudio.set_audio_backend(args.audio_backend)

    # Importing a mono .wav file and then splitting the resulting numpy-array in smaller chunks.
    audio, rate = load_audio(args.input_file)

    chunk_size = 18060

    audio_list = torch.split(audio, chunk_size, dim=-1)
    nb_channels = audio.shape[0]
    audio_chunks = iter([torch.unsqueeze(audio, dim=0) for audio in audio_list])

    xumx_separator = Separator.load(
        model_path="./pretrained_model",
        runtime_backend="torch-cpu",
        warmup=args.warmup_iters,
    )
    xumx_separator.quiet = True

    # https://pytorch.org/audio/stable/tutorials/streamwriter_advanced.html#playing-audio
    s = StreamWriter(dst="default", format="alsa")
    s.add_audio_stream(44100, nb_channels)

    with s.open():
        for frame, audio_chunk in enumerate(audio_chunks):
            demixed = xumx_separator(audio_chunk)
            demixed_bass = demixed[0, 0]
            demixed_vocals = demixed[1, 0]
            demixed_other = demixed[2, 0]
            demixed_drums = demixed[3, 0]

            s.write_audio_chunk(0, demixed_other.T)


if __name__ == '__main__':
    demixui_main()
