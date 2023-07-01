###############
# DEVEL STAGE #
###############
FROM nvcr.io/nvidia/pytorch:22.12-py3 as devel

RUN python -m pip install --upgrade pip

# build a wheel for torchaudio from source to correspond to $PYTORCH_VERSION from nvcr
RUN git clone https://github.com/pytorch/audio /torchaudio
WORKDIR /torchaudio
RUN python -m pip wheel --no-build-isolation --no-deps ./ --wheel-dir /wheelhouse

RUN python -m pip download onnxruntime -d /wheelhouse

#################
# RUNTIME STAGE #
#################
FROM nvcr.io/nvidia/pytorch:22.12-py3 as runtime

RUN export DEBIAN_FRONTEND="noninteractive" && apt-get update -y && \
	apt-get install -y ffmpeg libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavresample-dev libavutil-dev git-lfs --fix-missing

RUN python -m pip install --upgrade pip

COPY --from=devel /wheelhouse /wheelhouse

RUN python -m pip install /wheelhouse/torchaudio*.whl
RUN python -m pip install /wheelhouse/onnxruntime*.whl

# install clarity from source
RUN git clone https://github.com/claritychallenge/clarity -b v0.3.4 /clarity
WORKDIR /clarity
# install deps for clarity
RUN python -m pip install hydra pandas omegaconf soxr
RUN python -m pip install -e .

# install xumx-slicq-v2 from source to get its dependencies
COPY . /xumx-sliCQ-V2
WORKDIR /xumx-sliCQ-V2

ARG ONNX_DEPS="onnxruntime-cpu"
RUN python -m pip install --pre -e .[devel,${ONNX_DEPS}] --find-links /wheelhouse

ARG CLONE_V1

RUN if [ "${CLONE_V1}" == "yes" ]; then \
	cd / &&\
	# clone and install xumx-slicq \
	git clone https://github.com/sevagh/xumx-sliCQ.git && \
	cd xumx-sliCQ && \
	# install deps \
	python -m pip install gitpython && \
	python -m pip install -e .; \
fi
