#!/usr/bin/env sh

musdbdir="/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/"
musdbdebug="/home/sevagh/musdbdebug"
outdir="umx-slicq-1"

set -x

batch=64
epochs=1000
workers=4
seqdur=6

#declare -a targetargs=(
#	"--target=vocals --fscale=mel --fbins=116 --fmin=37.7 --sllen=8024"
#	"--target=drums --fscale=mel --fbins=104 --fmin=49.3 --sllen=7108"
#	"--target=other --fscale=bark --fbins=64 --fmin=90.0 --sllen=4416"
#	"--target=bass --fscale=bark --fbins=105 --fmin=25.4 --sllen=7180"
#)

#vocals:         8.704854496436832       ('bark', array([898]), 33.89999999999991, 19.5, 62084)

declare -a targetargs=(
	"--target=vocals --fscale=mel --fbins=116 --fmin=37.7 --sllen=8024"
)

for i in "${targetargs[@]}"
do
	python -m kernprof -l -v scripts/train.py \
		--root "${musdbdir}" --is-wav --nb-workers=$workers --batch-size=$batch --epochs=$epochs --random-track-mix \
		--seq-dur=$seqdur \
		$i --debug \
		--output "${outdir}" \
		--source-augmentations gain channelswap
done
