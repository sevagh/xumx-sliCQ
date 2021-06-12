#!/usr/bin/env sh

musdbdir="/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/"
outdir="umx-slicq-1"

batch=16
epochs=1000
workers=4
seqdur=1

# best drum config
python scripts/train.py \
	--root "${musdbdir}" --is-wav --nb-workers=$workers --batch-size=$batch --epochs=$epochs \
	--seq-dur=$seqdur --valid-seq-dur=$seqdur --valid-samples-per-track=32 --valid-batch-size=8 \
	--target="drums" \
	--fscale='mel' --fbins=32 --fmin=115.5 --sllen=2016 \
	--output "${outdir}" --debug

#evaldir=$outdir

#python -m openunmix.evaluate --root "${musdbdir}" --is-wav --outdir=./$evaldir-ests --evaldir=./$evaldir-results --model="${evaldir}" --no-cuda

#umx --model="${evaldir}" --no-cuda --outdir="./$evaldir-inference ./glhf.wav --targets bass drums vocals other
