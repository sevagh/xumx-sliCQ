#!/usr/bin/env sh

musdbdir="/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/"
outdir="umx-slicq-1-2070s"

set -x

batch=32
epochs=1000
seqdur=1

declare -a targetargs=(
	"--target=drums --fscale=mel --fbins=104 --fmin=49.3 --sllen=7108"
	"--target=bass --fscale=bark --fbins=105 --fmin=25.4 --sllen=7180"
)

for i in "${targetargs[@]}"
do
	python scripts/train.py \
		--root "${musdbdir}" --is-wav --nb-workers=4 --batch-size=$batch --epochs=$epochs --random-track-mix \
		--seq-dur=$seqdur \
		$i --debug \
		--output "${outdir}" \
		--source-augmentations gain channelswap \
		--cuda-device=1
done
