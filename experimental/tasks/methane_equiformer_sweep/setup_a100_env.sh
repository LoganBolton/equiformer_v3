#!/usr/bin/env bash
#
# Create the EquiformerV3 environment used by the methane A100 jobs.
# Run this once from anywhere inside the repository:
#
#   bash experimental/tasks/methane_equiformer_sweep/setup_a100_env.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_NAME="${ENV_NAME:-equiformer_v3}"

module load gcc/10.4.0
module load cuda
module load cmake
module load miniconda3

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.11 -c conda-forge
fi
conda activate "$ENV_NAME"

python -m pip install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

python -m pip install -U pip setuptools wheel ninja cmake pybind11
python -m pip install meson-python meson
python -m pip install --only-binary=:all: scipy

export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS="${MAX_JOBS:-8}"

# Rebuild these extensions for the A100 (sm_80). Binary wheels installed for
# another architecture can import successfully but fail inside radius_graph.
python -m pip uninstall -y \
    pyg-lib \
    pyg_lib \
    torch-geometric \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv || true

python -m pip install --no-cache-dir --no-deps --no-build-isolation --no-binary=:all: "torch-scatter==2.1.2"
python -m pip install --no-cache-dir --no-deps --no-build-isolation --no-binary=:all: "torch-sparse==0.6.18"
python -m pip install --no-cache-dir --no-deps --no-build-isolation --no-binary=:all: "torch-cluster==1.6.3"
python -m pip install --no-cache-dir --no-deps --no-build-isolation --no-binary=:all: "torch-spline-conv==1.2.2"
python -m pip install --no-deps "torch-geometric==2.7.0"

# Install the remaining pinned dependencies and this repository's Fair-Chem
# package. The already installed torch and PyG versions satisfy its dependency
# declarations, while pip fills in any Fair-Chem-only dependencies.
python -m pip install -r "$PROJECT_ROOT/experimental/env/conda_requirements.txt"
python -m pip install -e "$PROJECT_ROOT/packages/fairchem-core"

python - <<'PY'
import pathlib

import torch
import torch_cluster
import torch_geometric
import torch_scatter
import torch_sparse
import torch_spline_conv
from fairchem.core.common.registry import registry

print("\nInstalled versions:")
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("torch-geometric:", torch_geometric.__version__)
print("torch-scatter:", torch_scatter.__version__)
print("torch-sparse:", torch_sparse.__version__)
print("torch-cluster:", torch_cluster.__version__)
print("torch-spline-conv:", torch_spline_conv.__version__)

missing = []
for module in (torch_scatter, torch_sparse, torch_cluster, torch_spline_conv):
    path = pathlib.Path(module.__path__[0])
    libraries = sorted(file.name for file in path.glob("*_cuda.so"))
    print(f"{module.__name__} CUDA libraries:", libraries)
    if not libraries:
        missing.append(module.__name__)

radius_library = pathlib.Path(torch_cluster.__path__[0]) / "_radius_cuda.so"
if missing:
    raise RuntimeError(f"Missing CUDA libraries for: {', '.join(missing)}")
if not radius_library.exists():
    raise RuntimeError("torch-cluster is missing _radius_cuda.so")

print("\nSetup successful: the A100 CUDA extensions and Fair-Chem import correctly.")
PY

RADIUS_LIBRARY="$(
    python - <<'PY'
from pathlib import Path
import torch_cluster

print(Path(torch_cluster.__path__[0]) / "_radius_cuda.so")
PY
)"
if command -v cuobjdump >/dev/null 2>&1; then
    if ! cuobjdump --list-elf "$RADIUS_LIBRARY" | grep -q 'sm_80'; then
        echo "ERROR: torch-cluster radius CUDA library does not contain sm_80 code." >&2
        exit 1
    fi
    echo "Verified torch-cluster radius_graph contains A100 sm_80 CUDA code."
fi

echo
echo "Environment ready. Activate it with: conda activate $ENV_NAME"
