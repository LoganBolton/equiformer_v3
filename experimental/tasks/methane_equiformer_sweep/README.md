# Methane EquiformerV3

This workflow reproduces the HIPPYNN methane experiment with EquiformerV3.
Every training job uses exactly one process on one A100. There is no DDP.

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

Run the workflow tests with:

```bash
python -m unittest \
  experimental/tasks/methane_equiformer_sweep/test_methane_tools.py -v
```
