#!/bin/bash

#SBATCH --job-name=methane_eqv3_smoke
#SBATCH --output=./logs/%j_methane_eqv3_smoke.out
#SBATCH --error=./logs/%j_methane_eqv3_smoke.err
#SBATCH --time=04:00:00
#SBATCH --partition=shared-gpu-ampere
#SBATCH --constraint=gpu_cc:8.0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# One-run gate before any methane sweep. Submit from this task directory:
#   sbatch run_smoke.slurm

set -euo pipefail

TASK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="$(cd "$TASK_DIR/../../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-equiformer_v3}"
DATA_SIZE="${DATA_SIZE:-100000}"
EXTERNAL_SIZE="${EXTERNAL_SIZE:-80000}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
DATASET_DIR="$TASK_DIR/data/methane_train${DATA_SIZE}_test${EXTERNAL_SIZE}_seed${SEED}"
GENERATED_DIR="$TASK_DIR/generated/smoke"
RUN_ROOT="$TASK_DIR/runs/smoke"
PREDICT_ROOT="$RUN_ROOT/predict"
NAME="methane_eqv3_l3_m2_data${DATA_SIZE}_lr1em04_seed${SEED}"
CONFIG="$GENERATED_DIR/${NAME}.yml"
CHECKPOINT="$RUN_ROOT/checkpoints/$NAME/best_checkpoint.pt"
ALIGNED="$RUN_ROOT/predictions.npz"

mkdir -p "$TASK_DIR/logs" "$GENERATED_DIR" "$RUN_ROOT" "$PREDICT_ROOT"
module load miniconda3
source ~/.bashrc
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export CUDA_VISIBLE_DEVICES="${SMOKE_GPU_ID:-0}"

python "$TASK_DIR/verify_splits.py" \
  --data-size "$DATA_SIZE" --external-size "$EXTERNAL_SIZE" --seed "$SEED" \
  --indices "$DATASET_DIR/split_indices.npz"
python "$TASK_DIR/diagnose_lmdb.py" "$DATASET_DIR/train/data.lmdb" --samples 5
python "$TASK_DIR/diagnose_lmdb.py" "$DATASET_DIR/val/data.lmdb" --samples 3
python "$TASK_DIR/diagnose_lmdb.py" "$DATASET_DIR/test/data.lmdb" --samples 3

python "$TASK_DIR/generate_sweep.py" \
  --output-dir "$GENERATED_DIR" \
  --data-root "$TASK_DIR/data" \
  --run-dir "$RUN_ROOT" \
  --seeds "$SEED" --lmax-values 3 --data-sizes "$DATA_SIZE" \
  --learning-rates 1e-4 --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" --eval-batch-size "$EVAL_BATCH_SIZE" \
  --num-workers "${NUM_WORKERS:-4}"

python -u "$PROJECT_ROOT/scripts/train_equiformer_v3_smoke.py" \
  --mode train --config-yml "$CONFIG" --identifier "$NAME" \
  --timestamp-id "$NAME" --run-dir "$RUN_ROOT" --seed "$SEED"

if [ ! -f "$CHECKPOINT" ]; then
  echo "ERROR: best validation checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

python -u "$PROJECT_ROOT/scripts/train_equiformer_v3_smoke.py" \
  --mode predict --config-yml "$CONFIG" --identifier "$NAME" \
  --timestamp-id "$NAME" --run-dir "$PREDICT_ROOT" --seed "$SEED" \
  --checkpoint "$CHECKPOINT"

mapfile -t RAW_PREDICTIONS < <(find "$PREDICT_ROOT/results/$NAME" -maxdepth 1 -type f -name '*_predictions.npz')
if [ "${#RAW_PREDICTIONS[@]}" -ne 1 ]; then
  echo "ERROR: expected one Fair-Chem prediction archive, found ${#RAW_PREDICTIONS[@]}" >&2
  printf '%s\n' "${RAW_PREDICTIONS[@]}"
  exit 1
fi

python "$TASK_DIR/assemble_predictions.py" \
  "${RAW_PREDICTIONS[0]}" "$DATASET_DIR/test/data.lmdb" --output "$ALIGNED"
python "$TASK_DIR/evaluate_predictions.py" \
  "$ALIGNED" --output "$RUN_ROOT/metrics.json"

python - "$DATASET_DIR/manifest.json" "$CONFIG" "$CHECKPOINT" "$ALIGNED" "$RUN_ROOT/run_manifest.json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

dataset_manifest, config, checkpoint, predictions, output = map(Path, sys.argv[1:])
payload = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "dataset_manifest": str(dataset_manifest.resolve()),
    "config": str(config.resolve()),
    "best_checkpoint": str(checkpoint.resolve()),
    "predictions": str(predictions.resolve()),
    "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "PASS: smoke workflow completed; metrics: $RUN_ROOT/metrics.json"
