# Methane EquiformerV3 Sweep

This directory mirrors the HIP-HOP methane sweep at:

`https://github.com/LoganBolton/hippynn-optimizations-expanded/tree/methane_sweep/examples/methane_sweep`

It runs EquiformerV3 on the same methane frames, with:

- split seed: `42`
- model seeds: `42, 250, 1776`
- `lmax`: `3, 4`
- `mmax`: `2`
- `num_layers`: `1`
- data sizes: smoke `1000`, final `1000000`
- external test sizes: smoke `1000`, final `80000`
- fixed train/eval batch size: `512/512`
- learning rate: `1e-4`

## Methodology

A single train/validation/test split generated with split seed 42 was used for every run. Model seeds 42, 250, and 1776 changed only model initialization and training randomness.

EquiformerV3 used a fixed batch size of 512 with AdamW and warmup followed by cosine learning-rate decay from 1e-4 toward 1e-6. HIP-HOP used adaptive batch-size increases followed by plateau-based learning-rate reductions.

## Data

Download and unpack `methane.extxyz` from Materials Cloud, as in the HIP-HOP example:

```bash
mkdir -p experimental/datasets
# place methane.extxyz at experimental/datasets/methane.extxyz
```

The prep script reads the first `data_size + external_test_size` frames. It applies the same unit conversion as the HIP-HOP script:

- energy: Hartree to kcal/mol, then subtract `-25042.327220945674`
- forces: Hartree/Bohr to kcal/mol/Angstrom

## 1k smoke dataset

Generate the shared smoke dataset:

```bash
python experimental/tasks/methane_equiformer_sweep/prepare_methane_lmdb.py \
  --data-src experimental/datasets/methane.extxyz \
  --output-root experimental/tasks/methane_equiformer_sweep/data \
  --data-size 1000 \
  --external-test-size 1000 \
  --split-seed 42 \
  --purpose technical_smoke_test
```

Expected counts:

- train: `800`
- validation: `100`
- internal heldout: `100`
- external test: `1000`

Dataset directory:

```text
experimental/tasks/methane_equiformer_sweep/data/methane_train1000_test1000_split42/
```

Required contents:

```text
train/data.lmdb
val/data.lmdb
heldout/data.lmdb
test/data.lmdb
split_indices.npz
manifest.json
```

Verify the saved split indices and LMDBs:

```bash
python experimental/tasks/methane_equiformer_sweep/verify_splits.py \
  --data-size 1000 --external-test-size 1000 --split-seed 42 \
  --indices experimental/tasks/methane_equiformer_sweep/data/methane_train1000_test1000_split42/split_indices.npz

python experimental/tasks/methane_equiformer_sweep/diagnose_lmdb.py \
  experimental/tasks/methane_equiformer_sweep/data/methane_train1000_test1000_split42/train/data.lmdb \
  --expected-count 800 \
  --manifest experimental/tasks/methane_equiformer_sweep/data/methane_train1000_test1000_split42/manifest.json
```

## Smoke config generation

```bash
python experimental/tasks/methane_equiformer_sweep/generate_sweep.py \
  --output-dir experimental/tasks/methane_equiformer_sweep/generated/smoke \
  --data-root experimental/tasks/methane_equiformer_sweep/data \
  --run-dir experimental/tasks/methane_equiformer_sweep/runs/smoke \
  --model-seeds 42 \
  --split-seed 42 \
  --lmax-values 3 \
  --data-size 1000 \
  --external-test-size 1000 \
  --learning-rates 1e-4 \
  --epochs 1 \
  --batch-size 512 \
  --eval-batch-size 512
```

## Smoke train / predict / evaluate

Train:

```bash
python scripts/train_equiformer_v3_smoke.py \
  --mode train \
  --config-yml experimental/tasks/methane_equiformer_sweep/generated/smoke/methane_eqv3_l3_m2_data1000_test1000_split42_lr1em04_seed42.yml \
  --identifier methane_eqv3_l3_m2_data1000_test1000_split42_lr1em04_seed42 \
  --timestamp-id methane_eqv3_l3_m2_data1000_test1000_split42_lr1em04_seed42 \
  --run-dir experimental/tasks/methane_equiformer_sweep/runs/smoke/methane_train1000_test1000_split42/seed42/l3_m2_lr1em04 \
  --seed 42
```

Predict from the best checkpoint:

```bash
python scripts/train_equiformer_v3_smoke.py \
  --mode predict \
  --config-yml experimental/tasks/methane_equiformer_sweep/generated/smoke/methane_eqv3_l3_m2_data1000_test1000_split42_lr1em04_seed42.yml \
  --identifier methane_eqv3_l3_m2_data1000_test1000_split42_lr1em04_seed42 \
  --timestamp-id methane_eqv3_l3_m2_data1000_test1000_split42_lr1em04_seed42 \
  --run-dir experimental/tasks/methane_equiformer_sweep/runs/smoke/methane_train1000_test1000_split42/predict \
  --seed 42 \
  --checkpoint experimental/tasks/methane_equiformer_sweep/runs/smoke/methane_train1000_test1000_split42/seed42/l3_m2_lr1em04/checkpoints/methane_eqv3_l3_m2_data1000_test1000_split42_lr1em04_seed42/best_checkpoint.pt
```

Assemble aligned predictions:

```bash
python experimental/tasks/methane_equiformer_sweep/assemble_predictions.py \
  experimental/tasks/methane_equiformer_sweep/runs/smoke/methane_train1000_test1000_split42/predict/results/methane_eqv3_l3_m2_data1000_test1000_split42_lr1em04_seed42/<trainer>_predictions.npz \
  experimental/tasks/methane_equiformer_sweep/data/methane_train1000_test1000_split42/test/data.lmdb \
  --output experimental/tasks/methane_equiformer_sweep/runs/smoke/methane_train1000_test1000_split42/seed42/l3_m2_lr1em04/predictions.npz
```

Evaluate explicit shared metrics:

```bash
python experimental/tasks/methane_equiformer_sweep/evaluate_predictions.py \
  experimental/tasks/methane_equiformer_sweep/runs/smoke/methane_train1000_test1000_split42/seed42/l3_m2_lr1em04/predictions.npz \
  --output experimental/tasks/methane_equiformer_sweep/runs/smoke/methane_train1000_test1000_split42/seed42/l3_m2_lr1em04/metrics.json
```

Or run the full smoke gate with Slurm:

```bash
cd experimental/tasks/methane_equiformer_sweep
sbatch run_smoke.slurm
```

## Final shared 1M dataset

Generate the one shared final dataset only once:

```bash
python experimental/tasks/methane_equiformer_sweep/prepare_methane_lmdb.py \
  --data-src experimental/datasets/methane.extxyz \
  --output-root experimental/tasks/methane_equiformer_sweep/data \
  --data-size 1000000 \
  --external-test-size 80000 \
  --split-seed 42 \
  --purpose shared_equiformer_dataset
```

This writes:

```text
experimental/tasks/methane_equiformer_sweep/data/methane_train1000000_test80000_split42/
```

Do not generate separate datasets per model seed.

## Final config generation

Generate all final configs against the shared dataset:

```bash
python experimental/tasks/methane_equiformer_sweep/generate_sweep.py \
  --output-dir experimental/tasks/methane_equiformer_sweep/configs \
  --data-root experimental/tasks/methane_equiformer_sweep/data \
  --run-dir experimental/tasks/methane_equiformer_sweep/runs/final \
  --model-seeds 42 250 1776 \
  --split-seed 42 \
  --lmax-values 3 4 \
  --data-size 1000000 \
  --external-test-size 80000 \
  --learning-rates 1e-4 \
  --batch-size 512 \
  --eval-batch-size 512
```

## Final sweep submission

After the 1k smoke pipeline completes successfully end to end:

```bash
cd experimental/tasks/methane_equiformer_sweep
sbatch run_sweep.slurm
```

## Regression checks

```bash
python -m unittest experimental/tasks/methane_equiformer_sweep/test_methane_tools.py -v
```
