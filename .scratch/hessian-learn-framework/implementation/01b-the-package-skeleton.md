# 01b: The package skeleton — `hvp` through the pipe

Type: task
Status: resolved
Serves: 01
Blocked by: None (02 resolved 2026-09-27 — the fork re-homed to the sibling `mace/`; `openQHA-Hessian/` is free)
Part of: [hessian-learn-framework](../map.md)

> Execution slice for [The package line](../decisions/01-the-package-line.md); the brief is
> [01a](01a-the-package-line.md). The tracer bullet: the thinnest module (`hvp`, zero
> openQHA imports) through every layer — package layout, editable install, twin runner,
> fixture locator, consumer rewrite — so the slices after it only repeat a settled
> pattern. Agent-run, no user action required. Wayfinder conventions apply: the `Status`
> protocol, no triage labels (`docs/agents/issue-tracker.md`, ADR 0009).

## What to build

The `openQHA-Hessian` package exists and is alive before more of the training side has
moved. A fresh working tree at `openQHA-Hessian/` (freed by 02) holds the package's new
content under a fresh local repository whose `origin` is `BloomDlwlrma/openQHA-Hessian`
— replacing the remote's content is [The repo swap](11-repo-swap.md)'s act. The content:
`pyproject.toml` (distribution `openqha-hessian`, version 0.1.0, setuptools), the
`openqha_hessian` package with a docstring-only `__init__.py` rewritten for the
extension, and `tests/` with the twin runner — openQHA's `tests/run_tests.py`
conventions (groups `unit`/`integration`, one subprocess per file, `--group`/`--all`,
unit by default) plus a provenance header (python, `openqha.__file__`,
`openqha_hessian.__file__`, mace version).

`hvp` moves in under its name; `t_hvp` and `t_hvp_engine` move to the package's mirrored
groups, with the old `_repo_root()` walk replaced by a locator that finds the openQHA
checkout from the imported `openqha` package (`OPENQHA_SRC` overrides) and keeps reading
the `tests/data/propanal_molecule` fixture in place — reused, not copied. The package
installs editable in the WSL `openqha` env, and that env gains the dev line making the
openQHA checkout importable ([Install and transport](../decisions/05-install-and-transport.md)
owns the productionized story). On the openQHA side the one outside-facing consumer of
`hvp` — `phl_loss`'s sibling import — points at the package; nothing else changes.

## Acceptance

- [x] In the WSL env: `import openqha_hessian.hvp` works from the editable install, and `import openqha` still pulls no torch, mace or openqha_hessian
- [x] The package runner passes `t_hvp` and `t_hvp_engine` (the engine test may SKIP only for a missing model, never for a missing path)
- [x] openQHA's `tests/run_tests.py --all` passes, with the two moved tests gone from its groups and nothing else changed
- [x] `05_train --dry-run` prints the same mace argv as before this effort
- [x] No module in `openQHA/training/` imports `hvp` through the old path
- [x] Reported per the operating rule: files moved, checks run, anything not verified

## Answer

Resolved 2026-09-27, agent-run in the WSL `openqha` env. TDD shape: the moved tests
landed first and failed on the missing package; `hvp.py` then made them green.

- **The repository.** A fresh git repo at `openQHA-Hessian/` (branch `main`; its first
  commit is this slice's; `origin` = `https://github.com/BloomDlwlrma/openQHA-Hessian.git`,
  **not pushed** -- the content swap is [11](../decisions/11-repo-swap.md)). Content:
  `pyproject.toml` (distribution `openqha-hessian` 0.1.0, setuptools, flat one-package
  layout; no declared dependencies -- [05](../decisions/05-install-and-transport.md)
  owns the dependency and install story), the docstring-only `openqha_hessian/__init__.py`
  rewritten for the extension, `hvp.py` moved under its name, and `tests/`.
- **The twin runner and the locator.** `tests/run_tests.py` repeats openQHA's
  conventions (groups `unit`/`integration`, one subprocess per file, `--group`/`--all`,
  unit by default) plus the provenance header: python, `openqha.__file__`,
  `openqha_hessian.__file__`, the `mace-torch` distribution version.
  `tests/_testlib.py` holds `openqha_src()` = `Path(openqha.__file__).resolve().parents[1]`
  (`OPENQHA_SRC` overrides); the engine test keeps reading `tests/data/propanal_molecule`
  in place (reused, not copied) and `t_hvp` is fixture-free.
- **openQHA's side.** `openqha/training/hvp.py` and the two tests are gone from the
  checkout; `phl_loss`'s sibling import became `from openqha_hessian import hvp as
  hvp_mod` -- its only outside-facing consumer (grep-verified). Nothing else changed.
- **The dev environment.** `python -m pip install -e openQHA-Hessian` in the WSL
  `openqha` env (installed `openqha-hessian 0.1.0`), and the env gained the dev line
  `.../envs/openqha/lib/python3.11/site-packages/openqha-dev.pth` holding
  `/mnt/c/Users/10704/Documents/01_Free-Energy-alchemical/openQHA` -- openQHA is
  importable from any cwd. Handed to [05](../decisions/05-install-and-transport.md):
  the productionised form of both, and 01a's wheel-only caveat for the locator's
  `parents[1]`.
- **Evidence** (WSL `openqha` env, 2026-09-27):
  - `import openqha` pulls no torch/mace/openqha_hessian; `import openqha_hessian.hvp`
    and `phl_loss.hvp_mod` both resolve to the package checkout's `hvp.py`.
  - The package runner: `python tests/run_tests.py --all` -- all 2 tests pass
    (`t_hvp` 1.7 s; `t_hvp_engine` 10.0 s -- the engine ran, no SKIP).
  - openQHA's suite: `python tests/run_tests.py --all` -- **all 80 tests pass**, with
    `t_hvp`/`t_hvp_engine` gone from its groups (including the staying `t_train_run`,
    `t_judge_engine`, `t_train_engine` on the old loss address, which 01c flips).
  - `05_train --dry-run` byte-identical before/after -- verified against a tiny
    fabricated Dataset (`S0_RUNS_ROOT=/tmp/01b_runs`, 2 train + 1 valid frame via
    `dataset._write_split`, the recipe `t_train_run` uses); the argv still carries
    `--loss_module openqha.training.phl_loss:build` (unchanged in this slice). The
    run exercises the new chain end to end: the balance measurement goes through
    `phl_loss` -> `openqha_hessian.hvp`.
  - Grep: no `from . import hvp` / `openqha.training.hvp` anywhere in openQHA's code.
- **Note for the next slices.** The `S0_RUNS_ROOT` tiny-dataset recipe is a ready-made
  dry-run harness. `t_frames_engine` rewrites the `SECONDS` field of
  `tests/data/propanal_molecule/frames/frames.{out,toml}` on any `--all` run; the two
  files were reverted before this commit.
- **Unverified:** nothing outstanding for this slice's acceptance. Outside this slice,
  pointed at their owners: README and `install.sh` (01a/05), the push/force-swap (11),
  the doc/notebook sweep (12) -- and two inherited docstrings that still say `hvp`
  lives in `openqha/training/` (`openqha/__init__.py`'s layout map and
  `openqha/training/__init__.py`), which 01d's removal of the directory and 12's sweep
  take with them.
- **Review (two axes, `81a262c..260b0f2`):** the moved bodies are byte-identical to
  their sources (the diffs are the tests' headers and `phl_loss`'s import line only);
  one hard finding -- stale vocabulary in the new package docstring (CONTEXT.md bans
  "projected ... loss") -- plus two precision fixes, all applied as the follow-up
  commit.
