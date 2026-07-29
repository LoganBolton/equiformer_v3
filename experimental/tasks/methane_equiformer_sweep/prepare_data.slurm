#!/bin/bash

#SBATCH --job-name=methane_eqv3_prep
#SBATCH --output=./logs/%j_methane_eqv3_prep.out
#SBATCH --error=./logs/%j_methane_eqv3_prep.err
#SBATCH --time=8:00:00
#SBATCH --qos=long
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-equiformer_v3}"
DATA_SRC="${DATA_SRC:-$PROJECT_ROOT/experimental/datasets/methane.extxyz}"
SPLIT_SEED="${SPLIT_SEED:-42}"
DATA_SIZES="${DATA_SIZES:-1000 1000000}"
EXTERNAL_TEST_SIZE="${EXTERNAL_TEST_SIZE:-1000}"
PURPOSE="${PURPOSE:-technical_smoke_test}"
OVERWRITE_ARGS=()
if [ "${OVERWRITE:-0}" = "1" ]; then
  OVERWRITE_ARGS=(--overwrite)
fi

mkdir -p "$SCRIPT_DIR/logs"

module load miniconda3
set +u
source ~/.bashrc
set -u
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"
echo "Preparing shared methane datasets with split seed $SPLIT_SEED; sizes: $DATA_SIZES; external_test_size=$EXTERNAL_TEST_SIZE"
python experimental/tasks/methane_equiformer_sweep/prepare_methane_lmdb.py \
  --data-src "$DATA_SRC" \
  --data-sizes $DATA_SIZES \
  --external-test-size "$EXTERNAL_TEST_SIZE" \
  --split-seed "$SPLIT_SEED" \
  --purpose "$PURPOSE" \
  "${OVERWRITE_ARGS[@]}"
