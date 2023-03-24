###############
# DEVEL STAGE #
###############
FROM nvcr.io/nvidia/pytorch:22.12-py3 as devel

RUN python -m pip install --upgrade pip

# build a wheel for torchaudio from source to correspond to $PYTORCH_VERSION from nvcr
RUN git clone https://github.com/pytorch/audio /torchaudio
WORKDIR /torchaudio
RUN python -m pip wheel --no-build-isolation --no-deps ./ --wheel-dir /wheelhouse

RUN python -m pip download onnxruntime-gpu -d /wheelhouse

#################
# RUNTIME STAGE #
#################
FROM nvcr.io/nvidia/pytorch:22.12-py3 as runtime

RUN export DEBIAN_FRONTEND="noninteractive" && apt-get update -y && \
	apt-get install -y ffmpeg libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavresample-dev libavutil-dev

RUN python -m pip install --upgrade pip

COPY --from=devel /wheelhouse /wheelhouse

RUN python -m pip install /wheelhouse/torchaudio*.whl
RUN python -m pip install /wheelhouse/onnxruntime*.whl

# install clarity from source
RUN git clone https://github.com/claritychallenge/clarity /clarity
WORKDIR /clarity
# install deps for clarity
RUN python -m pip install hydra pandas omegaconf
RUN python -m pip install -e .

# install xumx-slicq-v2 from source to get its dependencies
COPY . /xumx-sliCQ-V2
WORKDIR /xumx-sliCQ-V2

ARG ONNX_DEPS="onnxruntime-cuda"
RUN python -m pip install --pre -e .[devel,${ONNX_DEPS}] --find-links /wheelhouse
