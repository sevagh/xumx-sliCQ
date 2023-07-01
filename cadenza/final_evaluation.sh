#!/usr/bin/env bash

set -xou pipefail

export BATCH_SIZE=2
export TOTAL_BATCHES=8
export REPO_ARG="/home/sevagh/repos/xumx-sliCQ-V2/:/xumx-sliCQ-V2"
export MUSDB_ARG="/home/sevagh/Music/MDX-datasets/MUSDB18-HQ/:/MUSDB18-HQ"
export CAD1D_ARG="/home/sevagh/Music/MDX-datasets/CAD1/cadenza_data/:/CADENZA"
export CAD1R_ARG="/home/sevagh/Music/MDX-datasets/CAD1/cadenza_results/:/exp"

export PODMAN_CMD="podman run --rm -v ${REPO_ARG} -v ${MUSDB_ARG} -v ${CAD1D_ARG} -v ${CAD1R_ARG} xumx-slicq-v2 python -m cadenza.test"

${PODMAN_CMD}

echo "now exit the script"
