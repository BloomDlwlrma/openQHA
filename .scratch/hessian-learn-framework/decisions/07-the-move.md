# The move: training code from openQHA into openQHA-Hessian

Type: task
Status: open
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

<!-- resolver: append what was done + evidence; set Status: resolved; add a line to the map's Decisions so far -->
