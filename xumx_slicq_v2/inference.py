from pathlib import Path
import torch
import time
from tqdm import trange, tqdm
import torchaudio
import json
import numpy as np
import os
from xumx_slicq_v2 import data
from .separator import Separator
import argparse


def separate(
    audio,
    separator,
    rate=None,
    device=None,
):
    if rate is None:
        raise Exception("rate` must be provided.")

    if device:
        audio = audio.to(device)
    audio = data.preprocess_audio(audio, rate, separator.sample_rate)

    # getting the separated signals
    start_time = time.time()

    estimates = separator(audio)
    time_delta = time.time() - start_time
    estimates = separator.to_dict(estimates)
    return estimates, time_delta


def inference_main():
    parser = argparse.ArgumentParser(description="xumx-sliCQ-V2 Inference")

    parser.add_argument(
        "--input-dir",
        type=str,
        default="/input",
        help="Input dir (default: /input)",
    )
    parser.add_argument(
        "--audio-backend",
        type=str,
        default="soundfile",
        help="Set torchaudio backend (`sox_io`, `sox`, `soundfile` or `stempeg`), defaults to `soundfile`",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/output",
        help="Output dir (default: /output)",
    )

    parser.add_argument(
        "--ext",
        type=str,
        default=".wav",
        help="Output extension which sets the audio format",
    )
    parser.add_argument(
        "--start", type=float, default=0.0, help="Audio chunk start in seconds"
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Audio chunk duration in seconds, negative values load full track",
    )
    parser.add_argument(
        "--runtime-backend",
        default="torch-cpu",
        choices=("torch-cpu", "torch-cuda", "onnx-cpu", "onnx-cuda"),
        help="Set model backend, defaults to `torch-cpu`",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        default=False,
        help="Use realtime pretrained model",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=0,
        help="Run some inference warmup iterations",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="custom model path",
    )

    args = parser.parse_args()

    torchaudio.set_audio_backend(args.audio_backend)

    device = torch.device("cuda" if args.runtime_backend.endswith("cuda") else "cpu")
    print(f"Using torch device {device} for backend {args.runtime_backend}")

    # create separator only once to reduce model loading
    # when using multiple files
    separator = Separator.load(
        device=device,
        runtime_backend=args.runtime_backend,
        realtime=args.realtime,
        warmup=args.warmup_iters,
        model_path=args.model_path,
    )

    tot_time = 0.0
    n_files = 0

    # loop over the files
    for wav_file_num, wav_file in enumerate(tqdm(os.listdir(args.input_dir))):
        n_files += 1
        input_file = os.path.join(args.input_dir, wav_file)
        audio, rate = data.load_audio(input_file, start=args.start, dur=args.duration)
        estimates, time_taken = separate(
            audio=audio,
            rate=rate,
            separator=separator,
            device=device,
        )

        outdir = Path(args.output_dir) / Path(wav_file).stem
        outdir.mkdir(exist_ok=True, parents=True)

        tot_time += time_taken

        # write out estimates
        for target, estimate in estimates.items():
            target_path = str(outdir / Path(target).with_suffix(args.ext))
            torchaudio.save(
                target_path,
                torch.squeeze(estimate).detach().cpu(),
                encoding="PCM_F",  # pcm float for dtype=float32 wav
                sample_rate=separator.sample_rate,
            )

    if n_files > 0:
        avg_time = tot_time / float(n_files)
        print(f"Inference time in s (averaged across tracks): {avg_time:.2f}")
    else:
        print(
            f"No songs were demixed, are you sure {args.input_dir} contains .wav files?"
        )


if __name__ == "__main__":
    inference_main()
