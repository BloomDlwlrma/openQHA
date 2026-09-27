# 01d: The switchover — the move lands, ticket 07 resolves

Type: task
Status: open
Serves: 01
Blocked by: 01c, 04, 06
Part of: [hessian-learn-framework](../map.md)

> Execution slice for [The package line](../decisions/01-the-package-line.md); the brief is
> [01a](01a-the-package-line.md). This is [The move](../decisions/07-the-move.md) executed:
> `judge`, `run` and `smoke_fit` move once, carrying what 04 and 06 decided for the records
> and constants they write — that is why those two tickets block this slice. When it lands,
> 07 resolves by pointer and [The repo swap](../decisions/11-repo-swap.md) can land the
> package's content. Agent-run, no user action required. Wayfinder conventions apply: the
> `Status` protocol, no triage labels (`docs/agents/issue-tracker.md`, ADR 0009).

## What to build

`judge`, `run` and `smoke_fit` move into `openqha_hessian` under their names, their
outside-facing imports rewritten to `openqha.{data,store,thermochem,potentials}`, with
[SHA256 retirement](../decisions/04-sha256-retirement.md)'s field changes and
[Identity after the split](../decisions/06-identity-after-the-split.md)'s constants
already in the code they carry.

Their tests follow: `t_judge`, `t_smoke_fit`, `t_train_run`, `t_judge_engine`,
`t_train_engine`. The consumers left in openQHA change addresses only — the drivers
(`05_train.py`, `06_judge.py`), the Slurm gate in `hl_train.slurm`, the s0 scripts'
remaining imports, and the code-side comments that still name the old path.
`openqha/training/` disappears, taking with it its `SUBPACKAGES` entry and its line in
the package-layout docstring. The documentation and notebook sweep is
[References sweep](../decisions/12-references-sweep.md)'s; this slice leaves it a
code-clean tree.

## Acceptance

- [ ] `openqha/training/` does not exist, and `import openqha` still pulls nothing from `openqha_hessian`
- [ ] openQHA's `tests/run_tests.py --all` passes; the package runner passes all nine moved tests
- [ ] `05_train --dry-run` prints the argv with `--loss_module openqha_hessian.phl_loss:build` and nothing else changed from before this effort
- [ ] No shipping module, script or test references `openqha.training` in code (documentation and notebooks are 12's)
- [ ] Reported per the operating rule: files moved, checks run, anything not verified
