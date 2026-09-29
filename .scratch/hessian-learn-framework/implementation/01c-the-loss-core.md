# 01c: The loss core — `phl` and `phl_loss` follow

Type: task
Status: resolved
Serves: 01
Blocked by: 01b
Part of: [hessian-learn-framework](../map.md)

> Execution slice for [The package line](../decisions/01-the-package-line.md); the brief is
> [spec-the-package-line.md](../spec-the-package-line.md). Repeats 01b's pattern for the remaining openqha-free
> modules; after this slice, every module of the training side that does not touch openQHA
> has left. Agent-run, no user action required. Wayfinder conventions apply: the `Status`
> protocol, no triage labels (`docs/agents/issue-tracker.md`, ADR 0009).

## What to build

`phl` and `phl_loss` join `hvp` in `openqha_hessian` under their names (`phl_loss`'s
sibling imports become package-internal), with `t_phl` and `t_phl_loss` moving to the
package's groups under the twin runner.

Because `phl_loss` is the module the fork's `--loss_module` address names, the address
flips **in this slice**, the moment the module leaves: the training run's loss-module
constant carries `openqha_hessian.phl_loss:build`, and every test or script that pins the
old address follows — including `t_train_run` and `t_train_engine`, which do not move
until [01d](01d-the-switchover.md) but must keep passing in between. Old Records stay
untouched; only new writes carry the new address.

Everything still living in openQHA gets its outside-facing imports for the two modules
pointed at the package while its remaining `openqha.training` imports stay until 01d:
`judge`, `run` and `smoke_fit` (which read them), the tooling scripts that reach in
(`s0_probe_calibration`, `s0_spice_pt_draw`, and `s0_hl_smoke_fit`'s share),
`t_probe_calibration` (which stays put — its subject is the tooling script), and the few
tests whose references sit inside functions. Mixed addresses across these files are
expected in this slice and must not bend any seam.

## Acceptance

- [x] The package runner passes `t_phl` and `t_phl_loss` beside 01b's tests
- [x] openQHA's `tests/run_tests.py --all` passes, `t_train_run`'s and `t_train_engine`'s loss-address assertions included
- [x] `05_train --dry-run` prints `--loss_module openqha_hessian.phl_loss:build` — the effort's one argv change — and nothing else differs
- [x] The new address resolves end to end: the string the argv carries imports and its `build` runs (no training run needed)
- [x] No module in `openQHA/training/` imports `phl` or `phl_loss` through the old path
- [x] Reported per the operating rule: files moved, checks run, anything not verified

## Answer

Resolved 2026-09-27, agent-run in the WSL `openqha` env. TDD shape: the two tests moved
first and failed on `ImportError: cannot import name 'phl' from 'openqha_hessian'`;
the two modules then made them green.

- **The package.** `openqha_hessian/{phl,phl_loss}.py` are the openQHA modules moved
  wholesale — byte-identical except `phl_loss`'s two self-address docstrings (the
  module's and `build`'s), which now name `--loss_module
  openqha_hessian.phl_loss:build`. `phl` stays numpy-only; `phl_loss`'s sibling
  imports are package-internal (`from . import phl`; `hvp` was already the package's
  from 01b).
- **The tests.** `t_phl` and `t_phl_loss` moved to the package's `tests/unit/` under
  the twin runner: `t_phl` locates the propanal fixture in place through
  `_testlib.openqha_src()` (`OPENQHA_SRC` overrides), `t_phl_loss` is fixture-free
  (direct package import). openQHA's runner no longer lists them.
- **The address.** `run.LOSS_MODULE` carries `openqha_hessian.phl_loss:build` and
  `run.py`'s module docstring flipped with it. `t_train_run` pins the same literal;
  `t_train_engine` follows the constant. Old Records keep the old string — nothing
  resolves it. `05_train.py`'s docstring and `hl_train.slurm`'s comment that still
  name the old path stay for 01d's code-side comment sweep, per its ticket.
- **The consumers.** Outside-facing imports for the two modules now point at the
  package while the remaining `openqha.training` imports wait for 01d: `judge`,
  `run`, `smoke_fit`, the driver `05_train.py`,
  `scripts/tooling/s0_probe_calibration.py`, and the staying tests `t_train_run`,
  `t_judge`, `t_smoke_fit`, `t_dataset_mace_form`, `t_probe_calibration`, plus the
  in-function imports of `t_judge_engine` and `t_train_engine`. Recon found
  `s0_spice_pt_draw.py` and `s0_hl_smoke_fit.py` reference neither module (only
  `judge`/`run`/`smoke_fit`) — nothing to rewire there.
- **Evidence** (WSL `openqha` env, 2026-09-27):
  - Package runner `--all`: **4 pass** (`t_hvp` 1.7 s; `t_phl` 0.3 s; `t_phl_loss`
    4.2 s; `t_hvp_engine` 11.2 s — the engine ran, no SKIP).
  - The address resolves end to end from the exact `run.LOSS_MODULE` string:
    `importlib.import_module("openqha_hessian.phl_loss")` + `build(args)` returns the
    loss instance (a probe mirroring the fork's loader).
  - The dry-run, pristine-HEAD checkout vs a clean checkout carrying this slice only:
    the outputs differ in **one line** — `--loss_module
    openqha.training.phl_loss:build` → `openqha_hessian.phl_loss:build`; every other
    line (the balance `L_H` number included) is byte-identical.
  - openQHA's `tests/run_tests.py --all` on that clean slice checkout: **all 78 tests
    pass**, the rewired seven among them (`t_train_run` 1.1 s, `t_judge_engine`
    43.1 s, `t_train_engine` 79.6 s — it trains to completion against the new
    address). In the day's shared worktree the same suite is 77/78: `t_train_engine`
    dies at `run.py:600 engine.parameter_fingerprint`, a function removed from
    `engine.py` by slice 04a's in-flight state (04a's map line: "`t_train_engine`
    waits on the training-side callers (07)"). 01c changes nothing on that path — the
    clean checkout isolates it, and the boxes above are checked on that evidence.
  - Grep: no `openqha.training` import of either module anywhere in openQHA code;
    `openqha/training/`'s three remaining modules import them from `openqha_hessian`.
- **Coordination with 04a** (in flight in the shared worktree; its ticket already
  records the overlap): "The two tooling scripts this slice edits are also touched by
  01c/01d for their import addresses; whichever side lands second rebases trivially."
  The hunks here are disjoint from 04a's, and 04a's changes stayed unstaged in 01c's
  commit (the shared files' hunks were staged mine-only).
- **Unverified:** nothing 01c governs. The doc/notebook sweep (12) and the code-side
  comments that still name the old path (01d) stay their tickets'.
- **Review (two axes, `3040542..383985d` openQHA / `54d101a..3f286e8` package):** no
  hard findings. Spec: every requirement and all six acceptance boxes borne out by the
  diffs; the two scripts the ticket names but the diff does not touch reconciled by
  recon (`s0_spice_pt_draw`, `s0_hl_smoke_fit` import neither module); no scope creep.
  Standards: conventions held — module names kept, no shim, fixtures reused, the tests'
  shape kept, the vocabulary guards clear; judgement calls left to their owners (the
  `cartesian_loss_full` alias, for the sweep tickets; the interim `openqha.training`
  → package imports, which 01d closes; the pre-existing weighting duplication in
  `phl_loss`, moved wholesale). One carried defect fixed as the package follow-up
  commit (`6d54c01`): the moved `t_phl_loss` docstring still described the retired
  frame-seed / Label-bytes cache.
