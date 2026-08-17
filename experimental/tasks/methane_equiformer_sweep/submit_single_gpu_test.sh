#!/usr/bin/env bash
#
# Download methane, make the 1k development dataset, and submit one 50-epoch
# single-A100 test. No DDP is used.
#
#   bash experimental/tasks/methane_equiformer_sweep/submit_single_gpu_test.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_NAME="${ENV_NAME:-equiformer_v3}"
SOURCE_DATA_DIR="$PROJECT_ROOT/experimental/datasets"
DATA_FILE="$SOURCE_DATA_DIR/methane.extxyz"
ARCHIVE="$DATA_FILE.gz"
DOWNLOAD_URL="https://archive.materialscloud.org/records/kz78r-6nx43/files/methane.extxyz.gz?download=1"
EXPECTED_MD5="11cf7303d8c0fa6ef753103f5439d6e1"
LOCAL_METHANE_FILE="${LOCAL_METHANE_FILE:-$HOME/Github/hippynn-optimizations-expanded/datasets/methane.extxyz}"

MODEL_SEED="${MODEL_SEED:-42}"
LMAX="${LMAX:-3}"
DATA_SIZE="${DATA_SIZE:-1000}"
EXTERNAL_TEST_SIZE="${EXTERNAL_TEST_SIZE:-1000}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-2.5e-3}"
DATASET_ID="methane_train${DATA_SIZE}_test${EXTERNAL_TEST_SIZE}_seed${MODEL_SEED}"
DATASET_DIR="$SCRIPT_DIR/data/$DATASET_ID"

case "$LMAX" in
    3|4) ;;
    *)
        echo "LMAX must be 3 or 4; got '$LMAX'." >&2
        exit 1
        ;;
esac

module load gcc/10.4.0
module load cuda
module load cmake
module load miniconda3
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Missing conda environment '$ENV_NAME'." >&2
    echo "First run: bash $SCRIPT_DIR/setup_a100_env.sh" >&2
    exit 1
fi
conda activate "$ENV_NAME"

mkdir -p "$SOURCE_DATA_DIR" "$SCRIPT_DIR/logs"

if [ ! -f "$DATA_FILE" ]; then
    if [ -f "$LOCAL_METHANE_FILE" ]; then
        echo "Linking existing methane.extxyz from: $LOCAL_METHANE_FILE"
        ln -sfn "$LOCAL_METHANE_FILE" "$DATA_FILE"
    else
        if [ ! -f "$ARCHIVE" ]; then
            echo "Downloading methane.extxyz.gz (about 1.1 GiB)..."
            curl --fail --location --continue-at - \
                "$DOWNLOAD_URL" \
                --output "$ARCHIVE"
        fi

        echo "$EXPECTED_MD5  $ARCHIVE" | md5sum --check -
        echo "Decompressing methane.extxyz.gz..."
        gzip --decompress --keep "$ARCHIVE"
    fi
fi

cd "$PROJECT_ROOT"
python "$SCRIPT_DIR/prepare_methane_lmdb.py" \
    --data-src "$DATA_FILE" \
    --data-size "$DATA_SIZE" \
    --external-test-size "$EXTERNAL_TEST_SIZE" \
    --model-seeds "$MODEL_SEED" \
    --purpose hippynn_matched_small_a100_stress

for REQUIRED in \
    "$DATASET_DIR/manifest.json" \
    "$DATASET_DIR/split_indices.npz" \
    "$DATASET_DIR/train/data.lmdb" \
    "$DATASET_DIR/val/data.lmdb" \
    "$DATASET_DIR/heldout/data.lmdb" \
    "$DATASET_DIR/test/data.lmdb"; do
    if [ ! -f "$REQUIRED" ]; then
        echo "Dataset preparation did not produce expected file: $REQUIRED" >&2
        exit 1
    fi
done

cd "$SCRIPT_DIR"
JOB_ID="$(
    sbatch --parsable \
        --job-name="methane_eqv3_l${LMAX}_test" \
        --export="ALL,MODEL_SEED=$MODEL_SEED,LMAX=$LMAX,DATA_SIZE=$DATA_SIZE,EXTERNAL_TEST_SIZE=$EXTERNAL_TEST_SIZE,EPOCHS=$EPOCHS,BATCH_SIZE=$BATCH_SIZE,EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE,LEARNING_RATE=$LEARNING_RATE" \
        run_a100_single_gpu_test.slurm
)"

echo
echo "Submitted one single-A100 test (one process; no DDP):"
echo "  job: $JOB_ID"
echo "  seed: $MODEL_SEED"
echo "  lmax/mmax: $LMAX/$LMAX"
echo "  data: $DATA_SIZE development + $EXTERNAL_TEST_SIZE external test"
echo "  dataset: $DATASET_DIR"
echo "  epochs: $EPOCHS"
echo
echo "Check it with: squeue -j $JOB_ID"
echo "Logs will appear in: $SCRIPT_DIR/logs/"
