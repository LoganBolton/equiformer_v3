#!/usr/bin/env bash
#
# Run the methane EquiformerV3 experiment on one local consumer GPU.
#
# This is the no-Slurm, no-module-load, no-conda counterpart to
# submit_single_gpu_test.sh. It builds its own virtualenv, fetches the dataset,
# converts it, writes a config, and trains -- all with paths derived from this
# script's own location, so it works from any checkout on any machine.
#
#   bash experimental/tasks/methane_equiformer_sweep/run_local.sh
#
# Every step is idempotent: rerunning skips whatever already exists.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ---------------------------------------------------------------- settings --
# Anything below can be overridden from the environment, e.g.
#   EPOCHS=500 LMAX=4 bash run_local.sh

MODE="${MODE:-test}"                       # test | full
GPU="${GPU:-0}"                            # which physical GPU to use
MODEL_SEED="${MODEL_SEED:-42}"             # seeds both the model and the split
LMAX="${LMAX:-3}"                          # 3 or 4; mmax is set equal to lmax
LEARNING_RATE="${LEARNING_RATE:-2.5e-3}"
NUM_WORKERS="${NUM_WORKERS:-0}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"          # cu128 covers Ampere through Blackwell
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"   # fairchem-core requires <3.13

case "$MODE" in
    test)
        DATA_SIZE="${DATA_SIZE:-1000}"
        EXTERNAL_TEST_SIZE="${EXTERNAL_TEST_SIZE:-1000}"
        EPOCHS="${EPOCHS:-50}"
        BATCH_SIZE="${BATCH_SIZE:-32}"
        EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
        ;;
    full)
        DATA_SIZE="${DATA_SIZE:-1000000}"
        EXTERNAL_TEST_SIZE="${EXTERNAL_TEST_SIZE:-80000}"
        EPOCHS="${EPOCHS:-10000}"
        BATCH_SIZE="${BATCH_SIZE:-256}"
        EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
        ;;
    *)
        echo "MODE must be 'test' or 'full'; got '$MODE'." >&2
        exit 1
        ;;
esac

case "$LMAX" in
    3|4) ;;
    *) echo "LMAX must be 3 or 4; got '$LMAX'." >&2; exit 1 ;;
esac

DATASET_ID="methane_train${DATA_SIZE}_test${EXTERNAL_TEST_SIZE}_seed${MODEL_SEED}"
DATASET_DIR="$SCRIPT_DIR/data/$DATASET_ID"
RUN_TAG="local_${MODE}_l${LMAX}_seed${MODEL_SEED}"
GENERATED_DIR="$SCRIPT_DIR/generated/$RUN_TAG"
RUN_BASE="$SCRIPT_DIR/runs/$RUN_TAG"
STEPS_PER_EPOCH=$(((DATA_SIZE * 8 / 10 + BATCH_SIZE - 1) / BATCH_SIZE))

SOURCE_DATA_DIR="$PROJECT_ROOT/experimental/datasets"
ARCHIVE="$SOURCE_DATA_DIR/methane.extxyz.gz"
DOWNLOAD_URL="https://archive.materialscloud.org/records/kz78r-6nx43/files/methane.extxyz.gz?download=1"
EXPECTED_MD5="11cf7303d8c0fa6ef753103f5439d6e1"
# Point this at an existing copy to skip the 1.1 GiB download. Plain .extxyz
# and gzipped .extxyz.gz are both read directly.
METHANE_SRC="${METHANE_SRC:-}"

echo "=============================================================="
echo " methane EquiformerV3 -- local single-GPU run"
echo "=============================================================="
echo "  repo:        $PROJECT_ROOT"
echo "  mode:        $MODE"
echo "  venv:        $VENV_DIR"
echo "  gpu:         $GPU"
echo "  seed:        $MODEL_SEED"
echo "  lmax/mmax:   $LMAX/$LMAX"
echo "  data:        $DATA_SIZE development + $EXTERNAL_TEST_SIZE external test"
echo "  epochs:      $EPOCHS"
echo "  batch size:  $BATCH_SIZE (about $STEPS_PER_EPOCH steps per epoch)"
echo

# ------------------------------------------------------------------- venv ---

PY="$VENV_DIR/bin/python"
READY_MARKER="$VENV_DIR/.methane_env_ready"

if [ ! -f "$READY_MARKER" ]; then
    echo "--- Building the Python environment (one time, a few minutes) ---"

    if command -v uv >/dev/null 2>&1; then
        uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
        INSTALL=(uv pip install --python "$PY")
    else
        # Fall back to the stdlib tooling when uv is unavailable.
        HOST_PYTHON=""
        for candidate in "python$PYTHON_VERSION" python3.12 python3.11 python3; do
            if command -v "$candidate" >/dev/null 2>&1; then
                if "$candidate" -c 'import sys; sys.exit(0 if (3,9) <= sys.version_info < (3,13) else 1)'; then
                    HOST_PYTHON="$candidate"
                    break
                fi
            fi
        done
        if [ -z "$HOST_PYTHON" ]; then
            echo "No Python between 3.9 and 3.12 found, and 'uv' is not installed." >&2
            echo "Install uv, which fetches a suitable Python itself:" >&2
            echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
            exit 1
        fi
        # Debian and Ubuntu ship python3 without ensurepip, so this is where the
        # no-uv path usually falls over. Say what to do about it.
        if ! "$HOST_PYTHON" -m venv "$VENV_DIR" 2>/dev/null; then
            rm -rf "$VENV_DIR"
            echo >&2
            echo "Could not create a virtualenv with '$HOST_PYTHON -m venv'." >&2
            echo "On Debian/Ubuntu the venv module is packaged separately." >&2
            echo >&2
            echo "Either install uv, which needs no system packages and is what" >&2
            echo "this script prefers:" >&2
            echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
            echo >&2
            echo "or install the matching venv package:" >&2
            echo "  sudo apt install $($HOST_PYTHON -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}-venv")')" >&2
            exit 1
        fi
        "$PY" -m pip install --upgrade pip
        INSTALL=("$PY" -m pip install)
    fi

    echo "--- Installing torch ($TORCH_CUDA) ---"
    "${INSTALL[@]}" torch==2.7.1 torchvision==0.22.1 \
        --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"

    echo "--- Installing fairchem-core from this checkout ---"
    "${INSTALL[@]}" -e "$PROJECT_ROOT/packages/fairchem-core"

    # Pin the same torch-geometric as setup_a100_env.sh. fairchem-core's own
    # dependency is unpinned and resolves to 2.8, which drops the torch-cluster
    # backend for radius_graph and would silently change the graph-building path.
    echo "--- Pinning torch-geometric==2.7.0 to match the cluster environment ---"
    "${INSTALL[@]}" "torch-geometric==2.7.0"

    # Prebuilt CUDA extensions matching this torch build, so the reference
    # code path runs unmodified and nothing has to be compiled locally. If no
    # wheel covers the local GPU architecture, training falls back to
    # pure-torch equivalents and says so loudly on stderr.
    echo "--- Installing prebuilt PyG CUDA extensions ---"
    PYG_INDEX="https://data.pyg.org/whl/torch-2.7.0+$TORCH_CUDA.html"
    if ! "${INSTALL[@]}" --no-build-isolation \
        torch-scatter torch-cluster torch-sparse -f "$PYG_INDEX"; then
        echo "    No prebuilt wheels at $PYG_INDEX; the pure-torch fallbacks will be used."
    fi

    touch "$READY_MARKER"
    echo "--- Environment ready ---"
    echo
else
    echo "--- Reusing the environment at $VENV_DIR ---"
    echo "    (delete $READY_MARKER to force a rebuild)"
    echo
fi

# ------------------------------------------------------------- gpu check ---

CUDA_VISIBLE_DEVICES="$GPU" "$PY" - <<'PY'
import sys

import torch

print("torch:", torch.__version__, "| built for CUDA", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit(
        "ERROR: torch cannot see a CUDA GPU. Check the NVIDIA driver with "
        "nvidia-smi, or set GPU= to a valid index."
    )
major, minor = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
print(f"gpu: {name} (compute capability {major}.{minor})")
if f"sm_{major}{minor}" not in torch.cuda.get_arch_list():
    sys.exit(
        f"ERROR: this torch build has no kernels for sm_{major}{minor}. "
        f"It ships {torch.cuda.get_arch_list()}. Rebuild the venv with a newer "
        "TORCH_CUDA, e.g. TORCH_CUDA=cu128."
    )
PY
echo

# ------------------------------------------------------------------ data ---

if [ -n "$METHANE_SRC" ]; then
    if [ ! -f "$METHANE_SRC" ]; then
        echo "METHANE_SRC does not exist: $METHANE_SRC" >&2
        exit 1
    fi
    DATA_FILE="$METHANE_SRC"
    echo "--- Using the provided methane source: $DATA_FILE ---"
elif [ -f "$SOURCE_DATA_DIR/methane.extxyz" ]; then
    DATA_FILE="$SOURCE_DATA_DIR/methane.extxyz"
    echo "--- Using the existing methane source: $DATA_FILE ---"
else
    mkdir -p "$SOURCE_DATA_DIR"
    if [ ! -f "$ARCHIVE" ]; then
        echo "--- Downloading methane.extxyz.gz (about 1.1 GiB) ---"
        curl --fail --location --continue-at - "$DOWNLOAD_URL" --output "$ARCHIVE"
        echo "$EXPECTED_MD5  $ARCHIVE" | md5sum --check -
    fi
    # The converter streams gzip directly, so there is no need to spend a
    # further 5 GiB of disk decompressing this.
    DATA_FILE="$ARCHIVE"
    echo "--- Using the compressed methane source: $DATA_FILE ---"
fi
echo

# --------------------------------------------------------------- convert ---

if [ -f "$DATASET_DIR/manifest.json" ] && [ -f "$DATASET_DIR/split_indices.npz" ]; then
    echo "--- Dataset already built: $DATASET_DIR ---"
else
    echo "--- Converting frames to LMDB (about 4 KiB per frame on disk) ---"
    "$PY" "$SCRIPT_DIR/prepare_methane_lmdb.py" \
        --data-src "$DATA_FILE" \
        --output-root "$SCRIPT_DIR/data" \
        --data-size "$DATA_SIZE" \
        --external-test-size "$EXTERNAL_TEST_SIZE" \
        --model-seeds "$MODEL_SEED" \
        --purpose "local_${MODE}_single_gpu"
fi

for REQUIRED in \
    "$DATASET_DIR/manifest.json" \
    "$DATASET_DIR/split_indices.npz" \
    "$DATASET_DIR/train/data.lmdb" \
    "$DATASET_DIR/val/data.lmdb" \
    "$DATASET_DIR/heldout/data.lmdb" \
    "$DATASET_DIR/test/data.lmdb"; do
    if [ ! -f "$REQUIRED" ]; then
        echo "Dataset preparation did not produce: $REQUIRED" >&2
        exit 1
    fi
done
echo

# ---------------------------------------------------------------- config ---

echo "--- Verifying the split against the HIPPYNN recipe ---"
"$PY" "$SCRIPT_DIR/verify_splits.py" \
    --data-size "$DATA_SIZE" \
    --external-test-size "$EXTERNAL_TEST_SIZE" \
    --split-seed "$MODEL_SEED" \
    --indices "$DATASET_DIR/split_indices.npz"

echo "--- Writing the run config ---"
mkdir -p "$GENERATED_DIR" "$RUN_BASE"
"$PY" "$SCRIPT_DIR/generate_sweep.py" \
    --output-dir "$GENERATED_DIR" \
    --data-root "$SCRIPT_DIR/data" \
    --run-dir "$RUN_BASE" \
    --model-seeds "$MODEL_SEED" \
    --lmax-values "$LMAX" \
    --data-size "$DATA_SIZE" \
    --external-test-size "$EXTERNAL_TEST_SIZE" \
    --learning-rates "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --eval-every "$STEPS_PER_EPOCH" \
    --checkpoint-every "$((STEPS_PER_EPOCH * 5))" \
    --num-workers "$NUM_WORKERS" \
    --name-suffix "$RUN_TAG"

readarray -t RUN_FIELDS < <("$PY" - "$GENERATED_DIR/sweep_manifest.json" "$SCRIPT_DIR" <<'PY'
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
echo

# ----------------------------------------------------------------- train ---

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONFAULTHANDLER=1
# Keep matplotlib and huggingface caches inside the run tree rather than $HOME.
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUN_BASE/mplconfig}"
mkdir -p "$MPLCONFIGDIR"

# Fair-Chem resumes on its own: BaseTask.setup loads checkpoints/<name>/
# checkpoint.pt whenever that file exists, restoring the step, optimizer, LR
# scheduler, and best-metric state. The run tag carries no timestamp, so that
# path is identical on every invocation with the same MODE, LMAX, and seed, and
# an interrupted run continues where it stopped. Say which case this is, the way
# run_a100_single_gpu_full.slurm does.
LATEST_CHECKPOINT="$RUN_DIR/checkpoints/$NAME/checkpoint.pt"

echo "--- Training ---"
echo "  config: $CONFIG"
echo "  output: $RUN_DIR"
echo "  logs:   tensorboard --logdir $SCRIPT_DIR/runs"
if [ -f "$LATEST_CHECKPOINT" ]; then
    echo "  resume: $LATEST_CHECKPOINT"
else
    echo "  resume: no checkpoint yet; starting from scratch"
fi
echo

cd "$PROJECT_ROOT"
"$PY" -u "$PROJECT_ROOT/scripts/train_equiformer_v3_smoke.py" \
    --mode train \
    --config-yml "$CONFIG" \
    --identifier "$NAME" \
    --timestamp-id "$NAME" \
    --run-dir "$RUN_DIR" \
    --seed "$MODEL_SEED"

CHECKPOINT="$RUN_DIR/checkpoints/$NAME/best_checkpoint.pt"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: training ended without a best checkpoint at $CHECKPOINT" >&2
    exit 1
fi

echo
echo "=============================================================="
echo " Done."
echo "   best checkpoint: $CHECKPOINT"
echo "   predictions:     $RUN_DIR/results/$NAME/"
echo "=============================================================="
