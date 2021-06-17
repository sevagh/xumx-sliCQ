#!/usr/bin/env sh

musdbdir="/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/"
musdbdebug="/home/sevagh/musdbdebug"
outdir="umx-slicq-1"

batch=32
epochs=1000
workers=4
seqdur=1
convseq=1

python scripts/train.py \
	--root "${musdbdir}" --is-wav --nb-workers=$workers --batch-size=$batch --epochs=$epochs \
	--seq-dur=$seqdur --conv-seq=$convseq \
	--target="vocals" --fscale='mel' --fbins=116 --fmin=37.7 --sllen=8024 \
	--output "${outdir}" --skip-statistics
	#--skip-statistics --print-shape --debug-plots --print-shape --patience=1000 --fixed-start=13

#	--target="drums" --fscale='mel' --fbins=104 --fmin=49.3 --sllen=7108 \
#
#	--target="other" --fscale='bark' --fbins=64 --fmin=90.0 --sllen=4416 \
#
#	--target="bass" --fscale='bark' --fbins=105 --fmin=25.4 --sllen=7180 \
