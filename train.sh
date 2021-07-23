#!/usr/bin/env sh

musdbdir="/run/media/sevagh/linux-extra/TRAINING-MUSIC/MUSDB18-HQ/"
outdir="xumx-slicq-train"

set -x

batch=32
epochs=1000
seqdur=1

python scripts/train.py \
	--root "${musdbdir}" --is-wav --nb-workers=4 --batch-size=$batch --epochs=$epochs --random-track-mix --patience=$epochs \
	--fscale=bark --fbins=262 --fmin=32.9 --sllen=18060 \
	--seq-dur=$seqdur \
	$i --enable-sdr-loss \
	--output "${outdir}" \
	--source-augmentations gain channelswap
