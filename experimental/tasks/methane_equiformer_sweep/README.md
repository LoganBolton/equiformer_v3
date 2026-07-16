# Methane EquiformerV3 Sweep

This directory mirrors the HIP-HOP methane sweep at:

`https://github.com/LoganBolton/hippynn-optimizations-expanded/tree/methane_sweep/examples/methane_sweep`

It runs EquiformerV3 on the same methane frames, with:

- seeds: `42, 1776, 250`
- `lmax`: `3, 4`
- `num_layers`: `1`
- data sizes: configurable grid, default `100000, 1000000`
- test size: `80000`
- train/eval batch size: `65536`
- learning-rate sweep: default `0.00005, 0.0001, 0.0002, 0.0004`

## Data

Download and unpack `methane.extxyz` from Materials Cloud, as in the HIP-HOP example:

```bash
mkdir -p datasets
# place methane.extxyz at datasets/methane.extxyz
```

The prep script reads the first `data_size + test_set_size` frames. It applies the same unit conversion as the HIP-HOP script:

- energy: Hartree to kcal/mol, then subtracts `-25042.327220945674`
- forces: Hartree/Bohr to kcal/mol/Angstrom

## Prepare LMDBs

```bash
python experimental/tasks/methane_equiformer_sweep/prepare_methane_lmdb.py \
  --data-src datasets/methane.extxyz \
  --output-root experimental/tasks/methane_equiformer_sweep/data \
  --data-sizes 100000 1000000 \
  --test-set-size 80000
```

The first `data_size` frames are split deterministically into `train`, `val`, and `heldout` LMDBs with a default `80/10/10` split, matching the HIP-HOP database setup that uses `valid_size=0.1` and `test_size=0.1` inside the training pool. The next `80000` frames are written as the external `test` split.

## Generate Configs

```bash
python experimental/tasks/methane_equiformer_sweep/generate_sweep.py
```

Use `--data-sizes` and `--learning-rates` to change the grid.

## Run on Slurm

```bash
cd experimental/tasks/methane_equiformer_sweep
sbatch run_sweep.slurm
```

The Slurm script follows the same local-worker pattern as the HIP-HOP sweep: one parent Slurm job, one child process per visible GPU, and checkpoint resume on requeue.

Cluster knobs can be overridden at submit time:

```bash
MAX_PARALLEL=4 WANDB_MODE=offline CHECKPOINT_EVERY=5000 sbatch run_sweep.slurm
```

Edit the `#SBATCH` header only if your cluster partition, constraint, memory, or QoS needs to change.
