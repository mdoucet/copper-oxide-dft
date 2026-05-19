#!/usr/bin/env bash
# Fine-tune MACE-MP-0 medium on the curated Cu-O dataset.
#
# Hyperparameters match the reference manuscript walkthrough
# (docs/machine-learned-dft.md §3.2). On DGX Spark a 50-epoch run with
# batch=4 over ~5000 structures completes overnight on the Blackwell GPU.
#
# Usage:
#   scripts/finetune_mace.sh <train.extxyz> <test.extxyz> <output-name>
#
# Required env vars:
#   MACE_MP_0_MEDIUM — path to the foundation-model .model file.
#
# See docs/dgx-spark-ml-install.md for the install + smoke-test path.

set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <train.extxyz> <test.extxyz> <output-name>" >&2
    exit 1
fi

train_file="$1"
test_file="$2"
output_name="$3"

if [[ -z "${MACE_MP_0_MEDIUM:-}" ]]; then
    echo "MACE_MP_0_MEDIUM env var is unset. Download the medium foundation" >&2
    echo "model and point this var at the .model file (see docs/dgx-spark-ml-install.md)." >&2
    exit 1
fi

for f in "$train_file" "$test_file" "$MACE_MP_0_MEDIUM"; do
    if [[ ! -f "$f" ]]; then
        echo "Required file not found: $f" >&2
        exit 1
    fi
done

# Manuscript hyperparameters (docs/machine-learned-dft.md §3.2):
#   batch_size=4, max_num_epochs=50, lr=0.01, AMSGrad=True,
#   EMA=True with decay 0.99, E0s=average, float32 precision.
# --start_swa is set past max epochs to effectively disable SWA — the
# manuscript uses EMA only.
mace_run_train \
    --name="$output_name" \
    --foundation_model="$MACE_MP_0_MEDIUM" \
    --train_file="$train_file" \
    --valid_file="$test_file" \
    --device="cuda" \
    --default_dtype="float32" \
    --batch_size=4 \
    --max_num_epochs=50 \
    --start_swa=51 \
    --lr=0.01 \
    --amsgrad=True \
    --ema=True \
    --ema_decay=0.99 \
    --E0s="average" \
    --energy_weight=1.0 \
    --forces_weight=1.0
