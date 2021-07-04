#!/usr/bin/env sh

musdbdir="/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/"
outdir="umx-slicq-4-2070s"

set -x

batch=32
epochs=1000
seqdur=6

declare -a targetargs=(
	#"--target=drums --fscale=bark --fbins=223 --fmin=68.5 --sllen=15504"
	#"--target=bass --fscale=mel --fbins=798 --fmin=20.1 --sllen=56544"
	"--target=vocals --fscale=bark --fbins=569 --fmin=56.8 --sllen=39556 --conv-chans=25,55 --conv-freq-filters=1,5 --conv-time-filters=11,23 --time-stride=3"
)

for i in "${targetargs[@]}"
do
	python scripts/train.py \
		--root "${musdbdir}" --is-wav --nb-workers=4 --batch-size=$batch --epochs=$epochs --random-track-mix \
		--seq-dur=$seqdur \
		$i --debug \
		--output "${outdir}" --patience=1000 \
		--source-augmentations gain channelswap \
		--cuda-device=1
done
