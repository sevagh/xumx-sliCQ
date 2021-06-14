#!/usr/bin/env sh

musdbdir="/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/"
outdir="umx-slicq-1"

batch=16
epochs=1000
workers=4
seqdur=1

python scripts/train.py \
	--root "${musdbdir}" --is-wav --nb-workers=$workers --batch-size=$batch --epochs=$epochs \
	--seq-dur=$seqdur --valid-seq-dur=$seqdur --valid-samples-per-track=64 --valid-batch-size=$batch \
	--target="vocals" \
	--fscale='mel' --fbins=116 --fmin=37.7 --sllen=8024 \
	--output "${outdir}"

#evaldir=$outdir

#python -m openunmix.evaluate --root "${musdbdir}" --is-wav --outdir=./$evaldir-ests --evaldir=./$evaldir-results --model="${evaldir}" --no-cuda

#umx --model="${evaldir}" --no-cuda --outdir="./$evaldir-inference ./glhf.wav --targets bass drums vocals other
