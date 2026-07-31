#!/usr/bin/env bash
#
# Run a full training baseline with on 1 gpu with 1 model.
#
# Default lmax=3:
#   bash experimental/tasks/methane_equiformer_sweep/submit_single_gpu_full.sh
#
# Example lmax=4 with a different model/split seed:
#   MODEL_SEED=1776 LMAX=4 bash \
#     experimental/tasks/methane_equiformer_sweep/submit_single_gpu_full.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_NAME="${ENV_NAME:-equiformer_v3}"
DATASET_SOURCE_DIR="$PROJECT_ROOT/experimental/datasets"
DATA_FILE="$DATASET_SOURCE_DIR/methane.extxyz"
ARCHIVE="$DATA_FILE.gz"
DOWNLOAD_URL="https://archive.materialscloud.org/records/kz78r-6nx43/files/methane.extxyz.gz?download=1"
EXPECTED_MD5="11cf7303d8c0fa6ef753103f5439d6e"
LOCAL_METHANE_FILE="${LOCAL_METHANE_FILE:-$HOME/Github/hippynn-optimizations-expanded/datasets/methane.extxyz}"

MODEL_SEED="${MODEL_SEED:-42}"
LMAX="${LMAX:-3}"
DATA_SIZE=1000000
EXTERNAL_TEST_SIZE=80000
EPOCHS="${EPOCHS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
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

mkdir -p "$DATASET_SOURCE_DIR" "$SCRIPT_DIR/logs"
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

PREP_JOB=""
DEPENDENCY_ARGS=()
if [ ! -f "$DATASET_DIR/manifest.json" ]; then
    echo "The complete $DATASET_ID LMDB does not exist; submitting CPU preparation first."
    cd "$SCRIPT_DIR"
    PREP_JOB="$(
        sbatch --parsable \
            --job-name="methane_prep_1m_s${MODEL_SEED}" \
            --export="ALL,CONDA_ENV=$ENV_NAME,DATA_SRC=$DATA_FILE,MODEL_SEEDS=$MODEL_SEED,DATA_SIZES=$DATA_SIZE,EXTERNAL_TEST_SIZE=$EXTERNAL_TEST_SIZE" \
            prepare_data.slurm
    )"
    DEPENDENCY_ARGS=("--dependency=afterok:$PREP_JOB")
else
    echo "Reusing complete dataset: $DATASET_DIR"
fi

cd "$SCRIPT_DIR"
RUN_JOB="$(
    sbatch --parsable \
        --job-name="methane_eqv3_l${LMAX}_1m_s${MODEL_SEED}" \
        "${DEPENDENCY_ARGS[@]}" \
        --export="ALL,CONDA_ENV=$ENV_NAME,MODEL_SEED=$MODEL_SEED,LMAX=$LMAX,EPOCHS=$EPOCHS,BATCH_SIZE=$BATCH_SIZE,EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE,LEARNING_RATE=$LEARNING_RATE" \
        run_a100_single_gpu_full.slurm
)"

echo
if [ -n "$PREP_JOB" ]; then
    echo "Dataset preparation job: $PREP_JOB"
fi
echo "Single-A100 training job: $RUN_JOB"
echo "  seed / split seed: $MODEL_SEED"
echo "  lmax/mmax: $LMAX/$LMAX"
echo "  development data: $DATA_SIZE"
echo "  external test data: $EXTERNAL_TEST_SIZE"
echo "  epochs: $EPOCHS"
echo "  batch size: $BATCH_SIZE"
echo "  learning rate: $LEARNING_RATE"
echo
echo "Check with: squeue -j ${PREP_JOB:+$PREP_JOB,}$RUN_JOB"
