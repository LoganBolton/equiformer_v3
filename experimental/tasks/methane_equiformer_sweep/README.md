# Methane EquiformerV3

This workflow reproduces the HIPPYNN methane experiment with EquiformerV3.
Every training job uses exactly one process on one GPU. There is no DDP.

There are two ways to run it:

- **[Locally, on a single consumer GPU](#run-locally-on-one-gpu)** — one script,
  no Slurm and no conda. Start here.
- **[On the Slurm cluster](#1-set-up-the-environment)** — the original A100
  workflow, further down.

## Run locally on one GPU

Clone the repository and run:

```bash
bash experimental/tasks/methane_equiformer_sweep/run_local.sh
```

That single command builds its own virtualenv, downloads and verifies the
methane dataset, converts a 1,000-frame development set to LMDB, writes a run
config, and trains for 50 epochs. It takes a couple of minutes on an RTX 3090
once the dataset is in place, and every step is skipped on reruns if its output
already exists.

Nothing needs to be edited first. All paths are derived from the script's own
location, so any checkout on any machine works.

### Requirements

- An NVIDIA GPU and driver. The default `TORCH_CUDA=cu128` build covers Ampere
  (RTX 3090) through Blackwell (RTX 5090).
- [`uv`](https://docs.astral.sh/uv/), which fetches its own Python:
  `curl -LsSf https://astral.sh/uv/install.sh | sh`. Without it the script falls
  back to `python -m venv`, which needs a system Python between 3.9 and 3.12.
- About 8 GiB of disk for the virtualenv, 1.2 GiB for the dataset archive, and
  roughly 4 KiB per converted frame.

### Options

Everything is set through the environment:

```bash
MODE=test          # test (1k frames, 50 epochs) or full (1M frames, 10k epochs)
GPU=0              # which physical GPU to use
MODEL_SEED=42      # seeds both the model and the HIPPYNN split
LMAX=3             # 3 or 4; mmax is always set equal to lmax
EPOCHS=50
BATCH_SIZE=32
EVAL_BATCH_SIZE=64
LEARNING_RATE=2.5e-3
NUM_WORKERS=0
METHANE_SRC=       # path to an existing methane.extxyz(.gz) to skip the download
VENV_DIR=          # defaults to <repo>/.venv
TORCH_CUDA=cu128   # torch wheel variant
```

For example, a longer lmax-4 run on the second GPU:

```bash
LMAX=4 EPOCHS=500 GPU=1 \
  bash experimental/tasks/methane_equiformer_sweep/run_local.sh
```

The full 1M-frame run needs roughly 4.3 GiB of LMDB on disk:

```bash
MODE=full bash experimental/tasks/methane_equiformer_sweep/run_local.sh
```

### No compiled PyG extensions

`torch-scatter` and `torch-cluster` are deliberately not installed. Their CUDA
kernels have to be compiled for the exact GPU architecture in use, which is the
main reason these environments break when moved between machines. Instead
`scripts/train_equiformer_v3_smoke.py` installs pure-torch replacements for
`scatter` and `radius_graph` at import time. The `radius_graph` fallback is
exact: it reproduces torch-cluster's output edge for edge, including its
neighbour-truncation order.

If either compiled package *is* present and loadable, it is used unchanged.

### Watching training

```bash
tensorboard --logdir experimental/tasks/methane_equiformer_sweep/runs
```

Results and the best checkpoint are written under
`experimental/tasks/methane_equiformer_sweep/runs/<run tag>/`; the script prints
both paths when it finishes.

## Slurm cluster workflow

The rest of this document covers the original single-A100 Slurm workflow.

## 1. Set up the environment

Run once after cloning:

```bash
bash experimental/tasks/methane_equiformer_sweep/setup_a100_env.sh
```

This creates the `equiformer_v3` environment, installs Fair-Chem, and compiles
the PyG CUDA extensions for A100 compute capability 8.0.

Do not use this environment on an RTX 2080 Ti (`sm_75`). Its native extensions
are built specifically for A100 (`sm_80`).

## 2. Single-GPU test

Run:

```bash
bash experimental/tasks/methane_equiformer_sweep/submit_single_gpu_test.sh
```

This one command:

1. downloads and verifies the Materials Cloud methane trajectory if needed;
2. creates 800 training, 100 validation, 100 internal-test, and 1,000 external
   test configurations;
3. submits one 50-epoch lmax-3 job on one A100.

It does not submit both model sizes. Select the model explicitly:

```bash
LMAX=4 MODEL_SEED=1776 \
  bash experimental/tasks/methane_equiformer_sweep/submit_single_gpu_test.sh
```

Test defaults:

```text
MODEL_SEED=42
LMAX=3
EPOCHS=50
BATCH_SIZE=32
EVAL_BATCH_SIZE=64
LEARNING_RATE=2.5e-3
```

## 3. Full 1M single-GPU run

Run:

```bash
bash experimental/tasks/methane_equiformer_sweep/submit_single_gpu_full.sh
```

This submits exactly one model. If its seed-specific 1M dataset does not exist,
the script first submits CPU dataset preparation and makes training depend on
successful preparation. The GPU job is resumable and requeues before its
48-hour Slurm allocation expires.

Select the seed and model configuration without editing files:

```bash
MODEL_SEED=1776 LMAX=4 \
  bash experimental/tasks/methane_equiformer_sweep/submit_single_gpu_full.sh
```

Full-run defaults:

```text
MODEL_SEED=42
LMAX=3
EPOCHS=10000
BATCH_SIZE=256
EVAL_BATCH_SIZE=256
LEARNING_RATE=2.5e-3
```

All of these settings may be overridden:

```bash
MODEL_SEED=250 \
LMAX=4 \
EPOCHS=1000 \
BATCH_SIZE=128 \
EVAL_BATCH_SIZE=128 \
LEARNING_RATE=1e-3 \
  bash experimental/tasks/methane_equiformer_sweep/submit_single_gpu_full.sh
```

`MODEL_SEED` controls both model initialization and the HIPPYNN dataset split.
`LMAX=3` produces `(lmax,mmax)=(3,3)` and `LMAX=4` produces `(4,4)`.

## Dataset and conversion

The scripts automatically download
[`methane.extxyz.gz` from Materials Cloud](https://archive.materialscloud.org/records/kz78r-6nx43)
and verify its published MD5 checksum. The decompressed source is stored at:

```text
experimental/datasets/methane.extxyz
```

Conversion creates seed-specific Fair-Chem LMDBs under:

```text
experimental/tasks/methane_equiformer_sweep/data/
```

For the full run, the first 1,000,000 sequential frames form the development
pool and the following 80,000 frames form the external test set. The
development pool is split into 800,000 training, 100,000 validation, and
100,000 internal-test configurations using `MODEL_SEED`.

## Matched HIPPYNN configuration

- Energy is converted from Hartree to kcal/mol and shifted by
  `-25042.327220945674`.
- Forces are converted from Hartree/Bohr to kcal/mol/Angstrom.
- Forces are calculated as `-dE/dR`; there is no direct force head.
- The objective is energy RMSE + energy MAE + componentwise force RMSE +
  componentwise force MAE + `1e-6` weight L2.
- Adam starts at `2.5e-3`; plateau scheduling and best-checkpoint selection use
  energy MAE.
- Width is 32, radial basis count is 20, cutoff is 10.3 Angstrom, and there is
  one interaction layer.

HIPPYNN dynamically increases batch size on a plateau. This trainer keeps the
configured batch size fixed and uses the matched plateau learning-rate rule.

## Monitoring

```bash
squeue -u "$USER"
ls experimental/tasks/methane_equiformer_sweep/logs/
tail -f experimental/tasks/methane_equiformer_sweep/logs/JOB_ID_methane_eqv3_single_test.out
```

### TensorBoard

From the repository root on `darwin-fe1`, start TensorBoard with the methane
task's `runs` directory:

```bash
cd "$(git rev-parse --show-toplevel)"
tensorboard \
  --logdir experimental/tasks/methane_equiformer_sweep/runs \
  --port 6007 --bind_all \
  --reload_interval 5
```

Then open:

```text
http://darwin-fe1.lanl.gov:6007/?darkMode=true#timeseries
```

Leave the TensorBoard command running while viewing the page; stop it with
`Ctrl+C`. New runs and metrics are scanned every five seconds, although the
browser may occasionally need to be refreshed.

Do not use the repository-level `runs` directory for this workflow. Running
`tensorboard --logdir runs` from the repository root only shows runs stored
there (such as `cpu_smoke` and `gpu_smoke`), not the methane experiments.

If port `6007` is already occupied, choose another unused server port in both
the command and URL. When direct access to `darwin-fe1` is unavailable, forward
the server port using the VS Code **Ports** panel and open the local forwarded
address it provides.

Run the workflow tests with:

```bash
python -m unittest \
  experimental/tasks/methane_equiformer_sweep/test_methane_tools.py -v
```
