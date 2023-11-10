FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN export DEBIAN_FRONTEND="noninteractive" && apt-get update -y && \
	apt-get install -y ffmpeg libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libswresample-dev libavutil-dev git-lfs --fix-missing

RUN python -m pip install --upgrade pip

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
