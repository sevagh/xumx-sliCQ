#!/usr/bin/env sh

musdbdir="/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/"
musdbdebug="/home/sevagh/musdbdebug"
outdir="umx-slicq-1"

batch=1
epochs=1000
workers=4
seqdur=3
convseq=3
samppertrack=1

python scripts/train.py \
	--root "${musdbdebug}" --is-wav --nb-workers=$workers --batch-size=$batch --epochs=$epochs \
	--target="vocals" --seq-dur=$seqdur --valid-seq-dur=$seqdur --valid-batch-size=$batch --valid-samples-per-track=$samppertrack --samples-per-track=$samppertrack \
	--fscale='mel' --fbins=116 --fmin=37.7 --sllen=8024 --conv-seq=$convseq \
	--output "${outdir}" \
	--skip-statistics --print-shape --debug-plots --print-shape --patience=1000 --fixed-start=13 #--source-augmentations gain channelswap 

#	--target="drums" \
#	--fscale='mel' --fbins=104 --fmin=49.3 --sllen=7108 \
#
#	--target="other" \
#	--fscale='bark' --fbins=64 --fmin=90.0 --sllen=4416 \
#
#	--target="bass" \
#	--fscale='bark' --fbins=105 --fmin=25.4 --sllen=7180 \

#evaldir=$outdir

#python -m openunmix.evaluate --root "${musdbdir}" --is-wav --outdir=./$evaldir-ests --evaldir=./$evaldir-results --model="${evaldir}" --no-cuda

#umx --model="${evaldir}" --no-cuda --outdir="./$evaldir-inference ./glhf.wav --targets bass drums vocals other
