import argparse
import time
import torch
import torchaudio
import asyncio
from torchaudio.io import StreamWriter
import numpy
from pathlib import Path
import sys
from xumx_slicq_v2.separator import Separator
from xumx_slicq_v2.data import load_audio
from xumx_slicq_v2.kivy import DemixApp
from kivy.app import async_runTouchApp


def next_power_of_two(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()


def demixui_main():
    parser = argparse.ArgumentParser(description="RtDemixUi Demo")

    parser.add_argument(
        "--audio-backend",
        type=str,
        default="soundfile",
        help="Set torchaudio backend (`sox_io`, `sox`, `soundfile` or `stempeg`), defaults to `soundfile`",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Warmup iters for ONNX inference",
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

    xumx_separator = Separator.load(
        model_path=".github/pretrained_models_other/pretrained_model_tiny_rt",
        runtime_backend="onnx-cpu",
        warmup=args.warmup,
    )
    xumx_separator.quiet = True

    chunk_size = next_power_of_two(xumx_separator.nsgt.nsgt.sllen)

    audio_list = torch.split(audio, chunk_size, dim=-1)
    nb_channels = audio.shape[0]
    audio_chunks_list = [torch.unsqueeze(audio, dim=0) for audio in audio_list]
    nb_chunks = len(audio_chunks_list)
    audio_chunks_iter = iter(audio_chunks_list)


    # https://pytorch.org/audio/stable/tutorials/streamwriter_advanced.html#playing-audio
    stream = StreamWriter(dst="default", format="alsa")
    stream.add_audio_stream(44100, nb_channels)

    demix_app = DemixApp(nb_chunks)

    async def run_demix_app(root, demix_loop):
        await root.async_run()
        demix_loop.cancel()

    async def demix_loop(root):
        try:
            with stream.open():
                for frame, audio_chunk in enumerate(audio_chunks_iter):
                    demixed = xumx_separator(audio_chunk)
                    demixed_bass = demixed[0, 0]
                    demixed_vocals = demixed[1, 0]
                    demixed_other = demixed[2, 0]
                    demixed_drums = demixed[3, 0]

                    try:
                        bass_level = root.bass_slider.value_normalized
                        drums_level = root.drums_slider.value_normalized
                        vocals_level = root.vocals_slider.value_normalized
                        other_level = root.other_slider.value_normalized
                    except AttributeError:
                        bass_level = 1.0
                        drums_level = 1.0
                        vocals_level = 1.0
                        other_level = 1.0

                    stream.write_audio_chunk(
                        0,
                        other_level*demixed_other.T + bass_level*demixed_bass.T + vocals_level*demixed_vocals.T + drums_level*demixed_drums.T
                    )

                    root.update_slider(frame)
                    await asyncio.sleep(0) #.01)
        except asyncio.CancelledError as e:
            print('demix loop was cancelled', e)
        finally:
            # when canceled, print that it finished
            print('bye')

    def root_func():
        root = demix_app
        demix_loop_fn = asyncio.ensure_future(demix_loop(root))
        return asyncio.gather(run_demix_app(root, demix_loop_fn), demix_loop_fn)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(root_func())
    loop.close()


if __name__ == '__main__':
    demixui_main()
