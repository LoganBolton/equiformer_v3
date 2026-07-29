# Methane baseline handoff

Status as of 2026-07-29: split reproduction is implemented and verified, but
the baseline is not yet ready for a full sweep.

## Verified from executed HIPPYNN code

- Entry point: `examples/methane_sweep/training/methane_configurations.py`.
- Training pool is the first `data_size` sequential extxyz frames.
- External test is the next `80_000` sequential frames.
- `Database(seed=model_seed, test_size=0.1, valid_size=0.1)` selects internal
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
- Exporter now writes seed-specific directories, `split_indices.npz`, hashes,
  provenance, source indices (`sid`), and nonperiodic samples. Its strict
  fixed-record CH4 reader avoids ASE's 4.4 GB full-file index scan, and atomic
  publication prevents interrupted builds from appearing complete.
- Config generator now uses seed-specific data, safe 128/512 batches, one
  evaluation/checkpoint interval per epoch, four neighbors, and run names that
  include both `lmax` and `mmax`.
- `diagnose_lmdb.py` validates geometry, atom/force shapes, PBC and source IDs.
- `evaluate_predictions.py` computes explicit energy and flattened-component
  force MAE, RMSE and R-squared with alignment/shape checks.
- `assemble_predictions.py` parses Fair-Chem IDs and force chunk boundaries,
  joins predictions to external LMDB targets by source `sid`, rejects missing,
  duplicate, extra, or shape-mismatched records, and writes canonical
  `predictions.npz`.

Seed-42, 100k verified hashes:

- train: `f3698919c96ff93586067e9cbe41484e4dca69fbb0f9b24fc96f5c79c51decef`
- valid: `8629cfd9ac8354b4f7bf7f274ecbecc46ca7c3c0ce901cda196f4c74cda9cffe`
- internal test: `240ef7c3984b00fc1d11e573bff8a7c35148feb64b239368aa14b2c0b02fd28e`
- external test: `aea4a4998328eca534782c713b83870fded8e7e501ee3dd931dbd3c303cca1ab`

## Next actions, in order

1. Add unit tests around split edge cases and compare against an imported
   HIPPYNN `Database` when its environment is active.
2. Rebuild only the seed-42 100k dataset. Do not use existing unseeded LMDBs.
3. Run `diagnose_lmdb.py` and verify every graph has five atoms and normally 20
   directed non-self edges.
4. Run `scripts/train_equiformer_v3_smoke.py --mode predict` with the generated
   config and `--checkpoint .../best_checkpoint.pt`. Convert its
   `<trainer>_predictions.npz` using `assemble_predictions.py`.
5. Feed the aligned archive to `evaluate_predictions.py`; add
   `run_manifest.json`.
6. Run one GPU, seed 42, 100k, l3/m2, one layer, 1e-4, one epoch. Confirm finite
   gradients/loss, validation, checkpoint, prediction, and index alignment.
7. Only then generate seed-specific data for other seeds and launch the sweep.

The existing generated tracked configs still contain legacy settings until
regenerated. Prefer a small generated smoke-config directory first. The SLURM
launcher correctly assigns independent GPUs, waits for every child, maintains
separate logs, and returns nonzero if any child fails; keep `MAX_PARALLEL=1`
for the smoke test.

The legacy unseeded 100k train LMDB was inspected only as a diagnostic. Its
stored length is 5,653 rather than 80,000 and its samples have
`pbc=[True, True, True]`; it is incomplete/invalid and must not be reused.
Two inspected methane frames did have five atoms, `(5, 3)` forces, 20 directed
non-self edges at 10.3 Angstrom, and finite values. The login-node environment
also reports GLIBC incompatibility warnings for PyG extension libraries, so
actual model smoke testing should run on the intended compute node.

An end-to-end 11-frame streaming conversion succeeded at about 183 frames/s.
`prepare_data.slurm` now generates every configured data size separately for
seeds 42, 1776, and 250; rebuilding requires the explicit `OVERWRITE=1` knob.
`run_smoke.slurm` implements the complete one-GPU gate from split verification
through best-checkpoint prediction, aligned shared metrics, and run manifest.
Both `prepare_data.slurm` (general CPU partition) and `run_smoke.slurm`
(Ampere constraint, explicitly `CUDA_VISIBLE_DEVICES=0`) pass `sbatch
--test-only` on this cluster. No jobs were submitted during validation.
