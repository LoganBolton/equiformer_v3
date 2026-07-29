# Methane baseline handoff

Status as of 2026-07-29: shared-split support is implemented, but the required GPU smoke run still must be executed on a compute node before launching the 1M sweep.

## Verified from executed HIPPYNN code

- Entry point: `examples/methane_sweep/training/methane_configurations.py`.
- Training pool is the first `data_size` sequential extxyz frames.
- External test is the next sequential `external_test_size` frames.
- `Database(seed=split_seed, test_size=0.1, valid_size=0.1)` selects internal
  test first with `torch.randperm`, removes it, then selects validation with
  fraction `0.1 / 0.9` using the same `torch.Generator`. Selected indices are
  sorted. Train is the sorted remainder.
- Energy: Hartree times `627.5096080305927`, then subtract
  `-25042.327220945674` kcal/mol.
- Force: Hartree/Bohr times `51.42208619083232 * 23.060541945329334`.
- HIPPYNN cutoff is `10.3` Angstrom and forces are energy gradients.
- HIPPYNN training loss is energy RMSE + energy MAE + componentwise force RMSE
  + componentwise force MAE + `1e-6` network L2.
- Reported validation quantities are energy/force RMSE, MAE, and R-squared.

## Implemented here

- `hippynn_splits.py`: reusable exact split construction and canonical hashes.
- `verify_splits.py`: independent construction, ordered and set comparison.
- `prepare_methane_lmdb.py`: now distinguishes `split_seed` from `model_seed`,
  writes one shared dataset per `(data_size, external_test_size, split_seed)`,
  writes `split_indices.npz`, hashes, and manifests, and validates methane atom
  order, nonperiodicity, finite converted labels, and 20 directed edges.
- `generate_sweep.py`: now emits configs that point all model seeds to the same
  shared dataset directory, keeps `split_seed` and `model_seed` separate, uses
  batch/eval batch size `512/512`, AdamW betas `[0.9, 0.98]`, eps `1e-6`, and
  warmup+cosine scheduling toward `lr_min_factor=0.01`.
- `diagnose_lmdb.py`: validates geometry, atom ordering, force shape, PBC, and
  expected directed-edge count.
- `assemble_predictions.py`: joins Fair-Chem predictions to external LMDB
  targets by original source `sid` and rejects missing, duplicate, extra, or
  shape-mismatched records.
- `evaluate_predictions.py`: computes explicit shared energy and flattened
  componentwise force MAE, RMSE, and R-squared.
- `run_smoke.slurm`: now targets the required 1k shared smoke dataset and writes
  `predictions.npz`, `metrics.json`, and `run_manifest.json` under the model run.
- `run_sweep.slurm`: consumes the shared-dataset config manifest and launches
  only model-seed-specific runs.

## Exact smoke split counts

For `data_size=1000`, `external_test_size=1000`, `split_seed=42`:

- train: `800`
- validation: `100`
- internal heldout: `100`
- external test: `1000`

Smoke split hashes:

- train: `aeb892041fdc95976406d3f63077e6e782f0101fb498d2ea7cbdbc991ce5f7d0`
- valid: `1373cdcc6ed6907b7f77cc59a924eee4b729389c83a24848b957a0cc9c6067fb`
- internal test: `59dbe918bcb2f4dedd44412f2a9a1eecaa7cc2417ca0623ab9c019e2dc5c9244`
- external test: `17db61bf83c86a1b36aaa6abfdd2d54e82ddafcc59c08cc850984b3fa6ec82b1`

## Exact final shared dataset target

For `data_size=1000000`, `external_test_size=80000`, `split_seed=42`:

```text
data/methane_train1000000_test80000_split42/
```

All final model seeds `42`, `250`, and `1776` must use that exact dataset.

## Remaining required execution steps

1. Build the 1k shared smoke dataset.
2. Verify `split_indices.npz` and inspect the four LMDBs.
3. Run the one-GPU smoke train/predict/evaluate job with model seed `42`.
4. Confirm finite training loss, validation, checkpoint, best-checkpoint
   discovery, external prediction, explicit metrics, and manifest creation.
5. Only then build the final 1M shared dataset.
6. Generate final configs for `(lmax,mmax)=(3,2)` and `(4,2)` across model
   seeds `42, 250, 1776`.
7. Submit the final training runs.

## Known limitation at handoff time

No GPU is available in the current login-node environment, so batch-size-512 fit
and the end-to-end smoke training/prediction workflow have not yet been executed
here. Run `run_smoke.slurm` on the intended Ampere compute node before any 1M
launch.
