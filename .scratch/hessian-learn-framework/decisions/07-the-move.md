# The move: training code from openQHA into openQHA-Hessian

Type: task
Status: resolved
Blocked by: 01, 04, 06
Part of: [hessian-learn-framework](../map.md)

## Question / work

Execute [The package line](01-the-package-line.md)'s boundary as actual code movement. This ticket decides nothing new; it lands what 01, 04 and 06 settled.

- Move the chosen files out of `openqha/training/` into the package; rewrite imports to match the decided direction; keep `openqha` importing the package where 01 says so.
- Keep the callers working: `workflows/hessian_learning/05_train.py`/`06_judge.py`, `hpc/slurm/hl_train.slurm`, openQHA's `tests/run_tests.py` groups and the moved tests.
- Carry [SHA256 retirement](04-sha256-retirement.md)'s record-field changes through the moved files (`run.py`/`judge.py` are in both scopes).
- Old notebooks and old Records: apply 01's compatibility ruling (shim or update).

## Acceptance

- openQHA's unit group + integration group pass; the package's own test command (its twin of `tests/run_tests.py`, defined here) passes; `05_train --dry-run` prints the same mace argv as before except the `--loss_module` string if 01 changed it.
- Report exactly: files moved, checks run, anything not verified (openQHA/AGENTS.md report rule).

## Answer

Resolved by [01d](implementation/01d-the-switchover.md) -- the move landed in two
commits: the package's `ff92141` (`judge`, `run`, `smoke_fit` and their five tests;
04's record fields; 06's package identity) and this repo's commit that follows (the
deletions, the consumer re-addresses, `engine.checkout_commit`, the extended reader
test, the map line). Evidence is in 01d's Answer: openQHA `--all` 73/73, the package
`--all` 9/9, and the dry-run argv unchanged except the `--loss_module` string.
