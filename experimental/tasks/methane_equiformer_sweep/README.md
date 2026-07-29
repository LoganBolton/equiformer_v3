# Methane EquiformerV3 Sweep

This directory mirrors the HIP-HOP methane sweep at:

`https://github.com/LoganBolton/hippynn-optimizations-expanded/tree/methane_sweep/examples/methane_sweep`

It runs EquiformerV3 on the same methane frames, with:

- seeds: `42, 1776, 250`
- `lmax`: `3, 4`
- `num_layers`: `1`
- data sizes: configurable grid, default `100000, 1000000`
- test size: `80000`
- default train/eval batch size: `128/512`
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
  --data-src experimental/datasets/methane.extxyz \
  --output-root experimental/tasks/methane_equiformer_sweep/data \
  --data-sizes 100000 1000000 \
  --test-set-size 80000 \
  --seed 42
```

The first `data_size` frames are split with HIPPYNN's exact two-call
`torch.randperm` algorithm. Internal test is selected first, validation second,
and memberships are sorted. The next `80000` sequential frames are external
test. Because HIPPYNN uses the model seed as its database seed, generate one
directory per seed. Each directory contains `split_indices.npz` and a manifest
with index hashes.

Verify before training:

```bash
python experimental/tasks/methane_equiformer_sweep/verify_splits.py \
  --data-size 100000 --seed 42 \
  --indices experimental/tasks/methane_equiformer_sweep/data/methane_train100000_test80000_seed42/split_indices.npz
```

Inspect rebuilt LMDB samples:

```bash
python experimental/tasks/methane_equiformer_sweep/diagnose_lmdb.py \
  experimental/tasks/methane_equiformer_sweep/data/methane_train100000_test80000_seed42/train/data.lmdb
```

Compute shared metrics after producing the required aligned prediction archive:

```bash
python experimental/tasks/methane_equiformer_sweep/evaluate_predictions.py \
  path/to/predictions.npz --output path/to/metrics.json
```

Convert Fair-Chem's raw prediction archive into that aligned archive:

```bash
python experimental/tasks/methane_equiformer_sweep/assemble_predictions.py \
  runs/<name>/results/<trainer>_predictions.npz \
  data/methane_train100000_test80000_seed42/test/data.lmdb \
  --output runs/<name>/predictions.npz
```

Explicit prediction uses the repository entry point with `--mode predict`,
the same generated config, and `--checkpoint path/to/best_checkpoint.pt`.

Run fast regression checks with:

```bash
python -m unittest experimental/tasks/methane_equiformer_sweep/test_methane_tools.py -v
```

The archive must contain source indices, scalar target/predicted energies, and
target/predicted forces shaped `(frames, atoms, 3)`. Force metrics flatten all
atom Cartesian components and are therefore componentwise, not vector-norm
`l2mae`.

Existing unseeded LMDB directories were produced by the old NumPy/PBC exporter
and must not be used for comparison runs.

## Generate Configs

```bash
python experimental/tasks/methane_equiformer_sweep/generate_sweep.py
```

Use `--data-sizes` and `--learning-rates` to change the grid.

For the required one-run smoke config:

```bash
python experimental/tasks/methane_equiformer_sweep/generate_sweep.py \
  --seeds 42 --lmax-values 3 --data-sizes 100000 \
  --learning-rates 1e-4 --epochs 1 --batch-size 128 --eval-batch-size 512 \
  --output-dir /tmp/methane_smoke_config
```

Important remaining work: add best-checkpoint external prediction and its run
manifest. Fair-Chem's built-in force loss/metric is not yet documented as
identical to HIPPYNN's componentwise definitions.
Do not launch the full sweep until the rebuilt seed-specific dataset and a
single-GPU smoke run pass.

## Run on Slurm

Run the required one-GPU gate first:

```bash
cd experimental/tasks/methane_equiformer_sweep
sbatch run_smoke.slurm
```

This verifies split hashes and LMDB diagnostics, generates one seed-42 l3/m2
configuration, trains, predicts from `best_checkpoint.pt`, aligns predictions
to external targets, and writes `predictions.npz`, `metrics.json`, and
`run_manifest.json`. It exits on the first failed stage.

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
