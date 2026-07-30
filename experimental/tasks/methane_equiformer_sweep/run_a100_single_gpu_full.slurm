#!/bin/bash

#SBATCH --job-name=methane_eqv3_single_full
#SBATCH --output=./logs/%j_methane_eqv3_single_full.out
#SBATCH --error=./logs/%j_methane_eqv3_single_full.err
#SBATCH --time=2-00:00:00
#SBATCH --qos=long
#SBATCH --partition=shared-gpu-ampere
#SBATCH --constraint=gpu_cc:8.0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append

# One process, one visible A100, no DDP. This job is requeued before its
# walltime and resumes from the latest checkpoint.

set -euo pipefail

TASK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="$(cd "$TASK_DIR/../../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-equiformer_v3}"
MODEL_SEED="${MODEL_SEED:-42}"
LMAX="${LMAX:-3}"
EPOCHS="${EPOCHS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-2.5e-3}"
DATA_SIZE=1000000
EXTERNAL_TEST_SIZE=80000
TRAIN_COUNT=800000
VAL_COUNT=100000
HELDOUT_COUNT=100000
STEPS_PER_EPOCH=$(((TRAIN_COUNT + BATCH_SIZE - 1) / BATCH_SIZE))
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-$STEPS_PER_EPOCH}"
EVAL_EVERY="${EVAL_EVERY:-$STEPS_PER_EPOCH}"
LR_TAG="${LEARNING_RATE//./p}"
LR_TAG="${LR_TAG//-/m}"
RUN_TAG="${RUN_TAG:-full_a100_1gpu_seed${MODEL_SEED}_lmax${LMAX}_b${BATCH_SIZE}_lr${LR_TAG}}"
DATASET_ID="methane_train${DATA_SIZE}_test${EXTERNAL_TEST_SIZE}_seed${MODEL_SEED}"
DATASET_DIR="$TASK_DIR/data/$DATASET_ID"
GENERATED_DIR="$TASK_DIR/generated/$RUN_TAG"
RUN_BASE="$TASK_DIR/runs/$RUN_TAG"
CHILD_PID=""

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
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONFAULTHANDLER=1

requeue_before_timeout() {
    echo "Received pre-timeout signal at $(date); requeueing job $SLURM_JOB_ID."
    if [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" 2>/dev/null || true
    fi
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap requeue_before_timeout USR1

echo "Job ID: $SLURM_JOB_ID"
echo "Restart count: ${SLURM_RESTART_COUNT:-0}"
echo "Node: $(hostname)"
echo "Mode: one process on one A100; DDP disabled"
echo "Model seed / split seed: $MODEL_SEED"
echo "lmax/mmax: $LMAX/$LMAX"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Steps per epoch: $STEPS_PER_EPOCH"
echo "Dataset: $DATASET_DIR"
nvidia-smi

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("visible_gpu_count:", torch.cuda.device_count())
print("visible_gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if torch.cuda.device_count() != 1:
    raise SystemExit("ERROR: expected exactly one visible GPU")
major, minor = torch.cuda.get_device_capability(0)
if (major, minor) != (8, 0):
    raise SystemExit(f"ERROR: expected A100 compute capability 8.0, got {major}.{minor}")
PY

python "$TASK_DIR/verify_splits.py" \
    --data-size "$DATA_SIZE" \
    --external-test-size "$EXTERNAL_TEST_SIZE" \
    --split-seed "$MODEL_SEED" \
    --indices "$DATASET_DIR/split_indices.npz"

for SPLIT_AND_COUNT in \
    "train:$TRAIN_COUNT" \
    "val:$VAL_COUNT" \
    "heldout:$HELDOUT_COUNT" \
    "test:$EXTERNAL_TEST_SIZE"; do
    SPLIT="${SPLIT_AND_COUNT%%:*}"
    COUNT="${SPLIT_AND_COUNT##*:}"
    python "$TASK_DIR/diagnose_lmdb.py" \
        "$DATASET_DIR/$SPLIT/data.lmdb" \
        --samples 3 \
        --expected-count "$COUNT" \
        --manifest "$DATASET_DIR/manifest.json"
done

python "$TASK_DIR/generate_sweep.py" \
    --output-dir "$GENERATED_DIR" \
    --data-root "$TASK_DIR/data" \
    --run-dir "$RUN_BASE" \
    --model-seeds "$MODEL_SEED" \
    --lmax-values "$LMAX" \
    --data-size "$DATA_SIZE" \
    --external-test-size "$EXTERNAL_TEST_SIZE" \
    --learning-rates "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --eval-every "$EVAL_EVERY" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --num-workers 0 \
    --name-suffix "$RUN_TAG"

readarray -t RUN_FIELDS < <(python - "$GENERATED_DIR/sweep_manifest.json" "$TASK_DIR" <<'PY'
import json
import sys
from pathlib import Path

item = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))[0]
task_dir = Path(sys.argv[2])
print(item["name"])
for key in ("config", "run_dir"):
    value = Path(item[key])
    print(value if value.is_absolute() else task_dir / value)
PY
)
NAME="${RUN_FIELDS[0]}"
CONFIG="${RUN_FIELDS[1]}"
RUN_DIR="${RUN_FIELDS[2]}"
CHECKPOINT="$RUN_DIR/checkpoints/$NAME/checkpoint.pt"
BEST_CHECKPOINT="$RUN_DIR/checkpoints/$NAME/best_checkpoint.pt"
CHECKPOINT_ARGS=()
if [ -f "$CHECKPOINT" ]; then
    echo "Resuming from: $CHECKPOINT"
    CHECKPOINT_ARGS=(--checkpoint "$CHECKPOINT")
fi

python -u "$PROJECT_ROOT/scripts/train_equiformer_v3_smoke.py" \
    --mode train \
    --config-yml "$CONFIG" \
    --identifier "$NAME" \
    --timestamp-id "$NAME" \
    --run-dir "$RUN_DIR" \
    --seed "$MODEL_SEED" \
    "${CHECKPOINT_ARGS[@]}" &
CHILD_PID="$!"
wait "$CHILD_PID"
CHILD_PID=""

if [ ! -f "$BEST_CHECKPOINT" ]; then
    echo "ERROR: training finished without a best checkpoint: $BEST_CHECKPOINT" >&2
    exit 1
fi

echo "PASS: full single-GPU run completed"
echo "Best checkpoint: $BEST_CHECKPOINT"
echo "End: $(date)"
