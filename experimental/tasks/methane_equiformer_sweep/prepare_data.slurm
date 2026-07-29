#!/bin/bash

#SBATCH --job-name=methane_eqv3_prep
#SBATCH --output=./logs/%j_methane_eqv3_prep.out
#SBATCH --error=./logs/%j_methane_eqv3_prep.err
#SBATCH --time=8:00:00
#SBATCH --qos=long
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -e

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-equiformer_v3}"
DATA_SRC="${DATA_SRC:-$PROJECT_ROOT/experimental/datasets/methane.extxyz}"
SEEDS="${SEEDS:-42 1776 250}"
DATA_SIZES="${DATA_SIZES:-100000 1000000}"
OVERWRITE_ARGS=()
if [ "${OVERWRITE:-0}" = "1" ]; then
  OVERWRITE_ARGS=(--overwrite)
fi

mkdir -p "$SCRIPT_DIR/logs"

module load miniconda3
source ~/.bashrc
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"
for seed in $SEEDS; do
  echo "Preparing methane data for split seed $seed; sizes: $DATA_SIZES"
  python experimental/tasks/methane_equiformer_sweep/prepare_methane_lmdb.py \
    --data-src "$DATA_SRC" \
    --data-sizes $DATA_SIZES \
    --seed "$seed" \
    "${OVERWRITE_ARGS[@]}"
done
