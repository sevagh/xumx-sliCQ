from xumx_slicq import utils
import argparse
import torch
import musdb
import stempeg
import os
import numpy


def separate(
    audio,
    rate=None,
    model_str_or_path="umxhq",
    targets=None,
    niter=1,
    residual=False,
    wiener_win_len=300,
    aggregate_dict=None,
    separator=None,
    device=None,
):
    """
    Open Unmix functional interface

    Separates a torch.Tensor or the content of an audio file.

    If a separator is provided, use it for inference. If not, create one
    and use it afterwards.

    Args:
        audio: audio to process
            torch Tensor: shape (channels, length), and
            `rate` must also be provided.
        rate: int or None: only used if audio is a Tensor. Otherwise,
            inferred from the file.
        model_str_or_path: the pretrained model to use
        targets (str): select the targets for the source to be separated.
            a list including: ['vocals', 'drums', 'bass', 'other'].
            If you don't pick them all, you probably want to
            activate the `residual=True` option.
            Defaults to all available targets per model.
        niter (int): the number of post-processingiterations, defaults to 1
        residual (bool): if True, a "garbage" target is created
        wiener_win_len (int): the number of frames to use when batching
            the post-processing step
        aggregate_dict (str): if provided, must be a string containing a '
            'valid expression for a dictionary, with keys as output '
            'target names, and values a list of targets that are used to '
            'build it. For instance: \'{\"vocals\":[\"vocals\"], '
            '\"accompaniment\":[\"drums\",\"bass\",\"other\"]}\'
        separator: if provided, the model.Separator object that will be used
             to perform separation
        device (str): selects device to be used for inference
    """
    if separator is None:
        separator = utils.load_separator(
            model_str_or_path=model_str_or_path,
            niter=niter,
            residual=residual,
            wiener_win_len=wiener_win_len,
            device=device,
            pretrained=True,
        )
        separator.freeze()
        if device:
            separator.to(device)

    if rate is None:
        raise Exception("rate` must be provided.")

    if device:
        audio = audio.to(device)
    audio = utils.preprocess(audio, rate, separator.sample_rate)

    # getting the separated signals
    estimates = separator(audio)
    estimates = separator.to_dict(estimates, aggregate_dict=aggregate_dict)
    return estimates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="xumx-sliCQ inference", add_help=False)

    parser.add_argument(
        "--model",
        default="umxhq",
        type=str,
        help="path to mode base directory of pretrained models",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        help="Results path where audio evaluation results are stored",
    )

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA inference"
    )

    parser.add_argument(
        "--niter",
        type=int,
        default=1,
        help="number of iterations for refining results.",
    )

    parser.add_argument(
        "--sr",
        type=float,
        default=44100.,
        help="Number of frames on which to apply filtering independently",
    )

    parser.add_argument(
        "--slicq-wiener", action="store_true", default=False, help="Apply iterative Wiener-EM on the sliCQT directly, not passing through the STFT domain (slower runtime)"
    )

    parser.add_argument(
        "--residual",
        type=str,
        default=None,
        help="if provided, build a source with given name"
        "for the mix minus all estimated targets",
    )

    parser.add_argument(
        "--aggregate",
        type=str,
        default=None,
        help="if provided, must be a string containing a valid expression for "
        "a dictionary, with keys as output target names, and values "
        "a list of targets that are used to build it. For instance: "
        '\'{"vocals":["vocals"], "accompaniment":["drums",'
        '"bass","other"]}\'',
    )

    parser.add_argument(
        "wav",
        type=str,
        help="path to wav file containing mixed song to separate",
    )

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    aggregate_dict = None if args.aggregate is None else json.loads(args.aggregate)

    track = musdb.audio_classes.Track(path=args.wav, stem_id=0, is_wav=True, subset='test')

    estimates = separate(
        torch.tensor(track.audio),
        rate=args.sr,
        model_str_or_path=args.model,
        niter=args.niter,
        residual=args.residual,
        aggregate_dict=aggregate_dict,
        device=device,
    )

    if args.outdir:
        for target, estimate in list(estimates.items()):
            target_path = os.path.join(args.outdir, target + '.wav')
            stempeg.write_audio(
                path=target_path,
                data=torch.squeeze(torch.permute(estimate, (0, 2, 1)), dim=0).cpu().detach().numpy(),
                sample_rate=args.sr
            )
