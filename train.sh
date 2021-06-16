#!/usr/bin/env sh

musdbdir="/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/"
musdbdebug="/home/sevagh/musdbdebug"
outdir="umx-slicq-1"

batch=24
epochs=1000
workers=4
seqdur=0.25
spt=64

python scripts/train.py \
	--root "${musdbdir}" --is-wav --nb-workers=$workers --batch-size=$batch --epochs=$epochs \
	--seq-dur=$seqdur --valid-seq-dur=$seqdur --valid-samples-per-track=$spt --valid-batch-size=$batch --samples-per-track=$spt \
	--target="vocals" \
	--fscale='mel' --fbins=116 --fmin=37.7 --sllen=8024 \
	--output "${outdir}" #--skip-statistics #--debug-plots --print-shape --patience=1000 --fixed-start

python scripts/train.py \
	--root "${musdbdir}" --is-wav --nb-workers=$workers --batch-size=$batch --epochs=$epochs \
	--seq-dur=$seqdur --valid-seq-dur=$seqdur --valid-samples-per-track=$spt --valid-batch-size=$batch --samples-per-track=$spt \
	--target="drums" \
	--fscale='mel' --fbins=104 --fmin=49.3 --sllen=7108 \
	--output "${outdir}" #--skip-statistics #--debug-plots --print-shape --patience=1000 --fixed-start

python scripts/train.py \
	--root "${musdbdir}" --is-wav --nb-workers=$workers --batch-size=$batch --epochs=$epochs \
	--seq-dur=$seqdur --valid-seq-dur=$seqdur --valid-samples-per-track=$spt --valid-batch-size=$batch --samples-per-track=$spt \
	--target="other" \
	--fscale='bark' --fbins=64 --fmin=90.0 --sllen=4416 \
	--output "${outdir}" #--skip-statistics #--debug-plots --print-shape --patience=1000 --fixed-start

python scripts/train.py \
	--root "${musdbdir}" --is-wav --nb-workers=$workers --batch-size=$batch --epochs=$epochs \
	--seq-dur=$seqdur --valid-seq-dur=$seqdur --valid-samples-per-track=$spt --valid-batch-size=$batch --samples-per-track=$spt \
	--target="bass" \
	--fscale='bark' --fbins=105 --fmin=25.4 --sllen=7180 \
	--output "${outdir}" #--skip-statistics #--debug-plots --print-shape --patience=1000 --fixed-start


#evaldir=$outdir

#python -m openunmix.evaluate --root "${musdbdir}" --is-wav --outdir=./$evaldir-ests --evaldir=./$evaldir-results --model="${evaldir}" --no-cuda

#umx --model="${evaldir}" --no-cuda --outdir="./$evaldir-inference ./glhf.wav --targets bass drums vocals other
