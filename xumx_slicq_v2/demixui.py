import argparse
import time
import torch
import torchaudio
import asyncio
from torchaudio.io import StreamWriter
import numpy
from pathlib import Path
import sys
from scipy import signal
from xumx_slicq_v2.separator import Separator
from xumx_slicq_v2.data import load_audio


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
        #model_path=".github/pretrained_models_other/pretrained_model_tiny_rt",
        model_path="pretrained_model_realtime",
        runtime_backend="onnx-cpu",
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

    from xumx_slicq_v2.kivy import DemixApp
    from kivy.app import async_runTouchApp
    import matplotlib
    matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')
    import matplotlib.pyplot as plt
    from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg

    def get_spectrogram(x, fs):
        with plt.style.context(('Solarize_Light2')):
            figure = plt.figure()
            figure.set_facecolor('gray')
            figure.patch.set_alpha(0.3)

            # Compute and plot the spectrogram.
            f, t, Sxx = signal.spectrogram(torch.mean(x.T, axis=0), fs)
            plt.ylabel('Frequency (Hz)')
            plt.xlabel('Time (s)')
            plt.pcolormesh(t, f, Sxx)
            #plt.show()
            wid = FigureCanvasKivyAgg(figure)
            plt.close()
        return wid

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

                    output_mix = other_level*demixed_other.T + bass_level*demixed_bass.T + vocals_level*demixed_vocals.T + drums_level*demixed_drums.T

                    stream.write_audio_chunk(0, output_mix)

                    root.update_slider(frame)
                    root.update_spectrogram(get_spectrogram(output_mix, rate))
                    await asyncio.sleep(0)
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
