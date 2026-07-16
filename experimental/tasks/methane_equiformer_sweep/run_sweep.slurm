#!/bin/bash

#SBATCH --job-name=methane_eqv3_sweep
#SBATCH --output=./logs/%j_methane_eqv3_sweep.out
#SBATCH --error=./logs/%j_methane_eqv3_sweep.err
#SBATCH --time=2-00:00:00
#SBATCH --qos=long
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --partition=shared-gpu-ampere
#SBATCH --constraint=gpu1_model:NVIDIA_A100-SXM4-40GB&gpu_count:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G

# Usage:
#   cd experimental/tasks/methane_equiformer_sweep
#   sbatch run_sweep.slurm

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TRAINING_SCRIPT="$PROJECT_ROOT/scripts/train_equiformer_v3_smoke.py"
MANIFEST="$SCRIPT_DIR/configs/sweep_manifest.json"
CONDA_ENV="${CONDA_ENV:-equiformer_v3}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
WANDB_MODE="${WANDB_MODE:-offline}"
ATTEMPT_ID="${SLURM_RESTART_COUNT:-0}"
children=()

if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: MAX_PARALLEL must be a positive integer, got '$MAX_PARALLEL'"
  exit 1
fi

if ! [[ "$CHECKPOINT_EVERY" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: CHECKPOINT_EVERY must be a positive integer, got '$CHECKPOINT_EVERY'"
  exit 1
fi

case "$WANDB_MODE" in
  online|offline|disabled) ;;
  *)
    echo "Error: WANDB_MODE must be one of: online, offline, disabled"
    exit 1
    ;;
esac

set -e

requeue_before_timeout() {
  echo ""
  echo "Received Slurm pre-timeout signal at $(date). Requeueing job $SLURM_JOB_ID..."
  for child in "${children[@]:-}"; do
    if kill -0 "$child" 2>/dev/null; then
      kill -TERM "$child" 2>/dev/null || true
    fi
  done
  if scontrol requeue "$SLURM_JOB_ID"; then
    echo "Requeued job $SLURM_JOB_ID. The next allocation will resume from checkpoints."
    exit 0
  fi
  echo "WARNING: scontrol requeue failed; exiting so this failure is visible."
  exit 1
}

trap requeue_before_timeout USR1

echo "Job ID: $SLURM_JOB_ID"
echo "Restart count: ${SLURM_RESTART_COUNT:-0}"
echo "Node: $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Working dir: $(pwd)"
echo "Script dir: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo "Start time: $(date)"
echo "=========================================="
echo ""

echo "Checking GPU availability..."
nvidia-smi

echo ""
echo "Loading conda environment..."
module load miniconda3

if conda info --envs | grep -q "$CONDA_ENV"; then
  source ~/.bashrc
  echo "Activating conda environment '$CONDA_ENV'..."
  conda activate "$CONDA_ENV"
else
  echo "Conda environment '$CONDA_ENV' not found. Set CONDA_ENV or create it first."
  exit 1
fi

if [ ! -f "$MANIFEST" ]; then
  echo "Sweep manifest not found at $MANIFEST"
  echo "Run: python generate_sweep.py"
  exit 1
fi

mkdir -p "$SCRIPT_DIR/logs" "$SCRIPT_DIR/runs" "$SCRIPT_DIR/wandb"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src"
export WANDB_MODE="$WANDB_MODE"
export WANDB_DIR="${WANDB_DIR:-$SCRIPT_DIR/wandb}"

SWEEP_COUNT="$(python - "$MANIFEST" <<'PY'
import json
import sys
print(len(json.loads(open(sys.argv[1], "r", encoding="utf-8").read())))
PY
)"

echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo "Sweep manifest: $MANIFEST"
echo "Sweep tasks: $SWEEP_COUNT"
echo "Parallelism: $MAX_PARALLEL"
echo "Checkpoint every: $CHECKPOINT_EVERY step(s)"
echo "W&B mode: $WANDB_MODE"
echo "=========================================="
echo ""

EXIT_CODE=0
for ((TASK_START = 0; TASK_START < SWEEP_COUNT; TASK_START += MAX_PARALLEL)); do
  pids=()
  for ((GPU_ID = 0; GPU_ID < MAX_PARALLEL; GPU_ID++)); do
    TASK_ID=$((TASK_START + GPU_ID))
    if [ "$TASK_ID" -ge "$SWEEP_COUNT" ]; then
      break
    fi

    TASK_JSON="$(python - "$MANIFEST" "$TASK_ID" <<'PY'
import json
import sys
items = json.loads(open(sys.argv[1], "r", encoding="utf-8").read())
print(json.dumps(items[int(sys.argv[2])], sort_keys=True))
PY
)"
    NAME="$(python - "$TASK_JSON" <<'PY'
import json
import sys
print(json.loads(sys.argv[1])["name"])
PY
)"
    CONFIG_REL="$(python - "$TASK_JSON" <<'PY'
import json
import sys
print(json.loads(sys.argv[1])["config"])
PY
)"
    RUN_DIR_REL="$(python - "$TASK_JSON" <<'PY'
import json
import sys
print(json.loads(sys.argv[1])["run_dir"])
PY
)"
    if [[ "$CONFIG_REL" = /* ]]; then
      CONFIG_PATH="$CONFIG_REL"
    else
      CONFIG_PATH="$SCRIPT_DIR/$CONFIG_REL"
    fi
    if [[ "$RUN_DIR_REL" = /* ]]; then
      RUN_DIR="$RUN_DIR_REL"
    else
      RUN_DIR="$SCRIPT_DIR/$RUN_DIR_REL"
    fi
    CHECKPOINT_PATH="$RUN_DIR/checkpoints/$NAME/checkpoint.pt"
    CHECKPOINT_ARGS=()
    if [ -f "$CHECKPOINT_PATH" ]; then
      CHECKPOINT_ARGS=(--checkpoint "$CHECKPOINT_PATH" --timestamp-id "$NAME")
    fi

    echo "Starting sweep task $TASK_ID / $SWEEP_COUNT on GPU $GPU_ID: $NAME"
    (
      export CUDA_VISIBLE_DEVICES="$GPU_ID"
      python -u "$TRAINING_SCRIPT" \
        --mode train \
        --config-yml "$CONFIG_PATH" \
        --identifier "$NAME" \
        --timestamp-id "$NAME" \
        --run-dir "$RUN_DIR" \
        --seed "$(python - "$TASK_JSON" <<'PY'
import json
import sys
print(json.loads(sys.argv[1])["seed"])
PY
)" \
        --optim.checkpoint_every="$CHECKPOINT_EVERY" \
        "${CHECKPOINT_ARGS[@]}"
    ) > "$SCRIPT_DIR/logs/${SLURM_JOB_ID}_r${ATTEMPT_ID}_${TASK_ID}_${NAME}.out" \
      2> "$SCRIPT_DIR/logs/${SLURM_JOB_ID}_r${ATTEMPT_ID}_${TASK_ID}_${NAME}.err" &

    pids+=("$!")
    children+=("$!")
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      EXIT_CODE=1
    fi
  done
done

echo "Local sweep finished at: $(date)"
echo "Exit code: $EXIT_CODE"
exit "$EXIT_CODE"
