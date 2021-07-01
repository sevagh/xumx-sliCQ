#!/usr/bin/env sh

musdbdir="/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/"
outdir="umx-slicq-1-3080ti"

set -x

batch=32
epochs=1000
seqdur=6

declare -a targetargs=(
	"--target=vocals --fscale=bark --fbins=569 --fmin=56.8 --sllen=39556"
	"--target=other --fscale=bark --fbins=569 --fmin=56.8 --sllen=39556"
)

for i in "${targetargs[@]}"
do
	python scripts/train.py \
		--root "${musdbdir}" --is-wav --nb-workers=4 --batch-size=$batch --epochs=$epochs --random-track-mix \
		--seq-dur=$seqdur \
		$i --debug \
		--output "${outdir}" \
		--source-augmentations gain channelswap
done
