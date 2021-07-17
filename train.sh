#!/usr/bin/env sh

musdbdir="/run/media/sevagh/linux-extra/TRAINING-MUSIC/MUSDB18-HQ/"
outdir="xumx-slicq-train"

set -x

batch=48
epochs=1000
seqdur=1

python scripts/train.py \
	--root "${musdbdir}" --is-wav --nb-workers=4 --batch-size=$batch --epochs=$epochs --random-track-mix --patience=$epochs \
	--fscale=bark --fbins=262 --fmin=32.9 --sllen=18060 --conv-chans=25,55 --conv-time-filters=11,23 --conv-time-strides=3,3 \
	--seq-dur=$seqdur \
	$i \
	--output "${outdir}" \
	--source-augmentations gain channelswap
