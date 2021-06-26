#!/usr/bin/env sh

musdbdir="/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/"
outdir="umx-slicq-1-3080ti"

set -x

batch=48
epochs=1000
seqdur=1

declare -a targetargs=(
	"--target=vocals --fscale=mel --fbins=116 --fmin=37.7 --sllen=8024"
	"--target=other --fscale=bark --fbins=64 --fmin=90.0 --sllen=4416"
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
