#!/usr/bin/env bash

set -eoxu pipefail

xdg-open paper.pdf &

while inotifywait -e close_write paper.md; do
	podman run --rm --volume ${PWD}/:/data:Z --env JOURNAL=joss openjournals/paperdraft;
done
