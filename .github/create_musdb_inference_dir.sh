#!/usr/bin/env bash
set -eou pipefail
set -x

MUSDB_DIR="$1"
DEST_DIR="$2"

mkdir -p "${DEST_DIR}"

wav_files=$(find "${MUSDB_DIR}" -type f -name 'mixture.wav')

echo "${wav_files}" | while read wav_file; do
	song_name=$(basename "$(dirname "${wav_file}")")
	echo "file: $wav_file"
	echo "song: $song_name"

	cp "${wav_file}" "${DEST_DIR}/${song_name}.wav"
done
