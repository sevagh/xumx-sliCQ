#!/usr/bin/env sh

musdbdir="/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/"
outdir="umx-slicq-4-3080ti"

set -x

batch=32
epochs=1000
seqdur=6

declare -a targetargs=(
	#"--target=vocals --fscale=bark --fbins=569 --fmin=56.8 --sllen=39556 --conv-chans=25,55 --conv-freq-filters=1,5 --conv-time-filters=11,23 --conv-time-stride=3"
	#"--target=other --fscale=bark --fbins=569 --fmin=56.8 --sllen=39556 --conv-chans=25,55 --conv-freq-filters=1,5 --conv-time-filters=11,23 --conv-time-stride=3 --seed=1337"
	"--target=bass --fscale=mel --fbins=798 --fmin=20.1 --sllen=56544 --conv-chans=25,55 --conv-freq-filters=1,5 --conv-time-filters=11,23 --conv-time-stride=3 --dropout=0.2"
)

for i in "${targetargs[@]}"
do
	python scripts/train.py \
		--root "${musdbdir}" --is-wav --nb-workers=4 --batch-size=$batch --epochs=$epochs --random-track-mix \
		--seq-dur=$seqdur \
		$i --debug \
		--output "${outdir}" --patience=1000  \
		--source-augmentations gain channelswap
done
