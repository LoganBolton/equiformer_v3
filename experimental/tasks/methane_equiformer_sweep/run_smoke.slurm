#!/bin/bash

#SBATCH --job-name=methane_eqv3_smoke
#SBATCH --output=./logs/%j_methane_eqv3_smoke.out
#SBATCH --error=./logs/%j_methane_eqv3_smoke.err
#SBATCH --time=04:00:00
#SBATCH --partition=shared-redstone
#SBATCH --constraint=gpu_cc:8.0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# One-run gate before any methane sweep. Submit from this task directory:
#   sbatch run_smoke.slurm

set -euo pipefail

TASK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="$(cd "$TASK_DIR/../../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-equiformer_v3}"
DATA_SIZE="${DATA_SIZE:-1000}"
EXTERNAL_TEST_SIZE="${EXTERNAL_TEST_SIZE:-1000}"
MODEL_SEED="${MODEL_SEED:-42}"
LMAX="${LMAX:-3}"
EPOCHS="${EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2.5e-3}"
BATCH_SIZE="${BATCH_SIZE:-256}"
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-manual}}"
TAG_SUFFIX="_${RUN_TAG}"
DATASET_ID="methane_train${DATA_SIZE}_test${EXTERNAL_TEST_SIZE}_seed${MODEL_SEED}"
DATASET_DIR="$TASK_DIR/data/$DATASET_ID"
GENERATED_DIR="$TASK_DIR/generated/smoke${TAG_SUFFIX}"
RUN_BASE="$TASK_DIR/runs/smoke${TAG_SUFFIX}"

mkdir -p "$TASK_DIR/logs" "$GENERATED_DIR" "$RUN_BASE"
module load gcc/10.4.0
module load cuda
module load cmake
module load miniconda3
set +u
source ~/.bashrc
set -u
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export CUDA_VISIBLE_DEVICES="${SMOKE_GPU_ID:-0}"

python "$TASK_DIR/verify_splits.py" \
  --data-size "$DATA_SIZE" --external-test-size "$EXTERNAL_TEST_SIZE" --split-seed "$MODEL_SEED" \
  --indices "$DATASET_DIR/split_indices.npz"
python "$TASK_DIR/diagnose_lmdb.py" \
  "$DATASET_DIR/train/data.lmdb" --samples 5 --expected-count 800 --manifest "$DATASET_DIR/manifest.json"
python "$TASK_DIR/diagnose_lmdb.py" \
  "$DATASET_DIR/val/data.lmdb" --samples 3 --expected-count 100 --manifest "$DATASET_DIR/manifest.json"
python "$TASK_DIR/diagnose_lmdb.py" \
  "$DATASET_DIR/heldout/data.lmdb" --samples 3 --expected-count 100 --manifest "$DATASET_DIR/manifest.json"
python "$TASK_DIR/diagnose_lmdb.py" \
  "$DATASET_DIR/test/data.lmdb" --samples 3 --expected-count "$EXTERNAL_TEST_SIZE" --manifest "$DATASET_DIR/manifest.json"

python "$TASK_DIR/generate_sweep.py" \
  --output-dir "$GENERATED_DIR" \
  --data-root "$TASK_DIR/data" \
  --run-dir "$RUN_BASE" \
  --model-seeds "$MODEL_SEED" --lmax-values "$LMAX" --data-sizes "$DATA_SIZE" \
  --external-test-size "$EXTERNAL_TEST_SIZE" --learning-rates "$LEARNING_RATE" --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" --name-suffix "$RUN_TAG" \
  --num-workers "${NUM_WORKERS:-0}"

readarray -t RUN_FIELDS < <(python - "$GENERATED_DIR/sweep_manifest.json" "$TASK_DIR" <<'PY'
import json
import sys
from pathlib import Path

item = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))[0]
task_dir = Path(sys.argv[2])
for key in ("name", "config", "run_dir"):
    value = Path(item[key]) if key != "name" else item[key]
    if key == "name":
        print(value)
    else:
        print(value if value.is_absolute() else task_dir / value)
PY
)
NAME="${RUN_FIELDS[0]}"
CONFIG="${RUN_FIELDS[1]}"
TRAIN_RUN_DIR="${RUN_FIELDS[2]}"
PREDICT_ROOT="$RUN_BASE/predict/$NAME"
CHECKPOINT="$TRAIN_RUN_DIR/checkpoints/$NAME/best_checkpoint.pt"
ALIGNED="$TRAIN_RUN_DIR/predictions.npz"
mkdir -p "$TRAIN_RUN_DIR" "$PREDICT_ROOT"

python -u "$PROJECT_ROOT/scripts/train_equiformer_v3_smoke.py" \
  --mode train --config-yml "$CONFIG" --identifier "$NAME" \
  --timestamp-id "$NAME" --run-dir "$TRAIN_RUN_DIR" --seed "$MODEL_SEED"

if [ ! -f "$CHECKPOINT" ]; then
  echo "ERROR: best validation checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

python -u "$PROJECT_ROOT/scripts/train_equiformer_v3_smoke.py" \
  --mode predict --config-yml "$CONFIG" --identifier "$NAME" \
  --timestamp-id "$NAME" --run-dir "$PREDICT_ROOT" --seed "$MODEL_SEED" \
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
  "$ALIGNED" --output "$TRAIN_RUN_DIR/metrics.json"

python - "$DATASET_DIR/manifest.json" "$CONFIG" "$CHECKPOINT" "$ALIGNED" "$TRAIN_RUN_DIR/run_manifest.json" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

dataset_manifest, config, checkpoint, predictions, output = map(Path, sys.argv[1:])
dataset_payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
payload = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "dataset_manifest": str(dataset_manifest.resolve()),
    "dataset_id": dataset_payload.get("dataset_name"),
    "purpose": dataset_payload.get("purpose"),
    "data_size": dataset_payload.get("data_size"),
    "external_test_size": dataset_payload.get("external_test_size"),
    "split_seed": dataset_payload.get("split_seed"),
    "model_seed": config_payload.get("seed"),
    "config": str(config.resolve()),
    "best_checkpoint": str(checkpoint.resolve()),
    "predictions": str(predictions.resolve()),
    "split_matches_hippynn_model_seed": True,
    "methodology": [
        "The model seed also generates the train/validation/internal-test split, exactly as in HIPPYNN.",
        "Training uses energy RMSE + MAE, componentwise force RMSE + MAE, and 1e-6 weight L2.",
        "Adam starts at 2.5e-3 with plateau LR reduction; the batch remains at HIPPYNN's starting size.",
    ],
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "PASS: smoke workflow completed; metrics: $TRAIN_RUN_DIR/metrics.json"
