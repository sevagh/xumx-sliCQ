#!/usr/bin/env sh

musdbdir="/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/"
outdir="umx-nsgt-3"

batch=16
epochs=1000
workers=4
seqdur=1

#for target in drums vocals other bass;
for target in drums;
do
	python scripts/train.py \
		--root "${musdbdir}" --is-wav --nb-workers=$workers --batch-size=$batch --epochs=$epochs \
		--seq-dur=$seqdur --stats-seq-dur=$seqdur --valid-seq-dur=$seqdur --valid-samples-per-track=32 --valid-batch-size=8 \
		--target="$target" \
		--layers=3 \
		--fscale='mel' --fmin=91.6 --fbins=113 --sllen=7432 \
		--output "${outdir}" --debug
done

#python -m openunmix.evaluate --root "${musdbdir}" --is-wav --outdir=./ests-$evaldir --evaldir=./results-$evaldir --model="${evaldir}" --no-cuda
