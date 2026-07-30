# Methane baseline handoff

The active, HIPPYNN-matched workflow is documented in `README.md`.

Important compatibility rules:

- model seed and split seed are the same;
- datasets are seed-specific and named
  `methane_train{data_size}_test{test_size}_seed{seed}`;
- final configs must be regenerated after those datasets complete;
- `preflight.py` is mandatory before `run_sweep.slurm`;
- old directories without a manifest and old configs without `model_seed` are
  intentionally rejected.

The legacy incomplete directory `data/methane_train1000000_test80000/` is not a
valid input. It has only 3,264 of the required 80,000 external-test records.
Do not point new jobs at it.
