# HIPPYNN-matched methane EquiformerV3 sweep

This workflow compares one-interaction-layer EquiformerV3 models with the
HIPPYNN methane experiments in
`~/Github/hippynn-optimizations-expanded/examples/methane_sweep`.

## Matched protocol

- The first `data_size` sequential extxyz frames form the development pool.
- The next `external_test_size` frames form the external test set.
- The model seed is also the train/validation/internal-test split seed, exactly
  as in HIPPYNN's `Database(seed=seed, test_size=0.1, valid_size=0.1)`.
- Energy is converted from Hartree to kcal/mol and shifted by
  `-25042.327220945674`.
- Forces are converted from Hartree/Bohr to kcal/mol/Angstrom.
- Forces are `-dE/dR`, not a direct force head.
- The objective is energy RMSE + energy MAE + componentwise force RMSE +
  componentwise force MAE + `1e-6` weight L2.
- Adam starts at `2.5e-3`; best checkpoints and plateau scheduling use energy
  MAE.
- Model width is 32, radial basis count is 20, cutoff is 10.3 Angstrom, and
  there is one interaction layer.
- Equiformer runs use `(lmax,mmax)=(3,3)` and `(4,4)`.
- Methane graph normalization uses `avg_degree=4` and total-energy aggregation
  uses `avg_num_nodes=1`.

HIPPYNN dynamically increases batch size on a plateau before reducing the
learning rate. Fair-Chem cannot safely rebuild this trainer's loader mid-run,
so Equiformer retains HIPPYNN's starting batch size of 256 and uses the same
plateau LR parameters. This is the remaining optimizer-control difference.

## Environment

Place `methane.extxyz` at:

```text
experimental/datasets/methane.extxyz
```

Run commands from this task directory:

```bash
cd experimental/tasks/methane_equiformer_sweep
```

## Fresh 1k smoke data

Create seed-specific smoke datasets. Seeds 42 and 1776 are common to the
existing lmax-3 and lmax-4 HIPPYNN sweeps.

```bash
python prepare_methane_lmdb.py \
  --data-src ../../../experimental/datasets/methane.extxyz \
  --data-size 1000 \
  --external-test-size 1000 \
  --model-seeds 42 1776 \
  --purpose hippynn_matched_smoke
```

Run clean, job-ID-isolated smoke tests:

```bash
EPOCHS=100 MODEL_SEED=42 LMAX=3 sbatch run_smoke.slurm
EPOCHS=100 MODEL_SEED=42 LMAX=4 sbatch run_smoke.slurm
```

Every smoke job uses its Slurm job ID in generated config and run paths, so it
cannot silently resume an older incompatible smoke checkpoint.

## Final 1M data

The prep launcher defaults to the final 1M/80k datasets for seeds 42 and 1776:

```bash
sbatch prepare_data.slurm
```

Equivalent direct command:

```bash
python prepare_methane_lmdb.py \
  --data-src ../../../experimental/datasets/methane.extxyz \
  --data-size 1000000 \
  --external-test-size 80000 \
  --model-seeds 42 1776 \
  --purpose hippynn_matched_equiformer_dataset
```

Expected directories:

```text
data/methane_train1000000_test80000_seed42/
data/methane_train1000000_test80000_seed1776/
```

Each directory is published atomically only after all of these exist and have
the expected counts:

```text
train/data.lmdb
val/data.lmdb
heldout/data.lmdb
test/data.lmdb
split_indices.npz
manifest.json
```

## Generate and launch final configurations

```bash
python generate_sweep.py \
  --output-dir configs \
  --data-root data \
  --run-dir runs/final \
  --model-seeds 42 1776 \
  --lmax-values 3 4 \
  --data-size 1000000 \
  --external-test-size 80000 \
  --learning-rates 2.5e-3 \
  --epochs 10000

python preflight.py configs/sweep_manifest.json
sbatch run_sweep.slurm
```

This produces four final runs: two matched seeds by two lmax values. Add more
seeds to both the data-preparation and config-generation commands to reproduce
larger HIPPYNN seed grids.

`run_sweep.slurm` allocates four A100s and runs one independent model on each
GPU. It refuses to launch stale configs, incomplete LMDBs, seed/split
mismatches, or non-HIPPYNN model/loss settings.

## Tests

```bash
python -m unittest test_methane_tools.py -v
```
