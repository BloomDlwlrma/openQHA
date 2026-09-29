# 01d: The switchover — the move lands, ticket 07 resolves

Type: task
Status: resolved
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
[Identity after the split](../decisions/06-identity-after-the-split.md)'s package-identity
fields (`HL_PACKAGE_VERSION`, `HL_PACKAGE_COMMIT`) and their reader — shared with the fork
reader, per [the spec](../spec-identity-after-the-split.md) — already in the code they
carry (06's constant flip is the separate slice [06a](06a-fork-identity-and-mapping.md)).

Their tests follow: `t_judge`, `t_smoke_fit`, `t_train_run`, `t_judge_engine`,
`t_train_engine`. The consumers left in openQHA change addresses only — the drivers
(`05_train.py`, `06_judge.py`), the Slurm gate in `hl_train.slurm`, the s0 scripts'
remaining imports, and the code-side comments that still name the old path.
`openqha/training/` disappears, taking with it its `SUBPACKAGES` entry and its line in
the package-layout docstring. The documentation and notebook sweep is
[References sweep](../decisions/12-references-sweep.md)'s; this slice leaves it a
code-clean tree.

## Acceptance

- [x] `openqha/training/` does not exist, and `import openqha` still pulls nothing from `openqha_hessian`
      -> in the WSL env: `'training' not in openqha.SUBPACKAGES`; `import openqha` pulls no `openqha_hessian`, no `torch`, no `mace`
- [x] openQHA's `tests/run_tests.py --all` passes; the package runner passes all nine moved tests
      -> openQHA `--all`: 73/73 (unit 60, integration 11, regression 2); the package `--all`: 9/9
- [x] Both Records carry the package identity — `HL_PACKAGE_VERSION` and the best-effort `HL_PACKAGE_COMMIT` (`unknown` tolerated; never a gate) — with the reader shared with the fork reader and the extended tests green
      -> both schemas and both Records on disk; `package_identity()` reads the distribution metadata + `engine.checkout_commit`; the fallback cases run on the package's module file (`t_engine_fork`: 18 checks)
- [x] `05_train --dry-run` prints the argv with `--loss_module openqha_hessian.phl_loss:build` and nothing else changed from before this effort
      -> the four argv builders compare byte-identical to `HEAD:openqha/training/run.py`; the printed argv carries the new loss address (the identity lines print the resolved files, per decision 04)
- [x] No shipping module, script or test references `openqha.training` in code (documentation and notebooks are 12's)
      -> grep over `openqha/`, `scripts/`, `tests/`, `workflows/`, `hpc/` is clean; the remaining mentions are documentation (12's)
- [x] Reported per the operating rule: files moved, checks run, anything not verified
      -> the Answer below

## Answer (2026-09-29, implemented in the two commits)

The move landed. `openQHA-Hessian` @ `ff92141` carries `judge`, `run` and
`smoke_fit` under their names, their outside imports rewritten to
`openqha.{data,store,thermochem,potentials}` (the sibling imports stay relative),
and the five tests (the fixture locator where they read openQHA's `tests/data`).
This repo's commit deletes
`openqha/training/` and its five tests, re-addresses the consumers -- the drivers,
the Slurm gate's import *and* its stale provenance print (it read the retired
`prov["params_sha256"]`/`["params_pin_status"]` keys: a `KeyError` today), the
three s0 scripts, the four staying tests -- drops `training` from `SUBPACKAGES`
and the layout docstring, and generalizes the identity reader:
`engine.checkout_commit` now holds the checkout rules, `mace_fork_info` keeps its
public shape and delegates to it (the dirty question stays the fork guard's).

What the three modules carry, from decisions 04 and 06:

- 04 (training side): `FOUNDATION_FILE` in; `FOUNDATION_PARAMS_SHA256`,
  `MODEL_PARAMS_SHA256`, `MODEL_N_TENSORS`, the `ENGINE_PARAMS_SHA256` alias and
  the judge's `ENGINE_PARAMS_SHA256`/`BASE_PARAMS_SHA256` out; the deleted
  `engine.parameter_fingerprint` call gone from `run_training` (the
  `t_train_engine` failure that [04a's Answer](04a-strip-weight-identity.md)
  declared this slice's); both report identity sections list the resolved files;
  `registry_entry` is a stamped fixed revision
  (`mace_off23_<campaign>/<run>+<YYYYMMDD-HHMMSS>.model`, no pin line;
  `--register-copy` creates the sub-directory).
- 06 (identity after the split): both writers carry `HL_PACKAGE_VERSION` and
  `HL_PACKAGE_COMMIT` through `openqha_hessian.package_identity()` (version from
  the distribution metadata; commit best-effort via `engine.checkout_commit`).

Evidence (WSL `openqha` env, 2026-09-29):

- openQHA `python tests/run_tests.py --all` -> **all 73 passed** (unit 60,
  integration 11, regression 2); `t_engine_fork` -- extended with the package's
  module-file cases -- 18 checks, 0 failed.
- package `python tests/run_tests.py --all` -> **all 9 passed** (the six unit and
  three integration moved tests); `t_train_engine` also asserts the training
  Record on disk carries the package identity.
- `05_train --dry-run`: `mace_argv` / `argv_pairs` / `control_settings` /
  `stage_two_weights` compare byte-identical to `HEAD:openqha/training/run.py`
  (extracted with `ast`); the printed argv carries
  `--loss_module openqha_hessian.phl_loss:build`.
- `import openqha` pulls no `openqha_hessian`, no `torch`, no `mace`; `training`
  is gone from `SUBPACKAGES`.
- grep: no shipping module, script or test references `openqha.training`.

Left to 12, per the spec: the `check_fork` repair message
(`pip install -e <path>/openQHA-Hessian` -- the retired name);
`workflows/hessian_learning/README.md`, the tutorials and the
`.scratch/hessian-learning-set` pointers; the training integration test's pinned
old commit B (`FORK_COMMIT_B`, now in the package) and the fork reader test's
cosmetic fake id.

Not assigned anywhere, flagged: the package `README.md` still lists only the loss
modules as the package's content (not in 12's list); running python from the
workspace root shadows `import mace` with the sibling `mace/` directory (a
namespace package answers `mace_fork_info` "unknown") -- repo roots are used
everywhere it matters.

## Review record (2026-09-29, annotations)

Two-axis review of the two commits (package `ff92141`, this repo `890bade`), per the
`code-review` skill, with the dispositions below.

**Standards.** The move's delta is import rewrites, the 04 field swaps, the identity
additions and the registration stamp; no baseline smell was introduced (the flagged
`HIP_METRICS` dead name, `frame_rows`'s unused `name`, the REF-key clumps and the two
`_num` helpers are pre-existing and moved verbatim). One documented-standard breach
inside a hunk this slice touches: `hl_train.slurm`'s WHAT-RUNS comment still named the
fork's old repository (`BloomDlwlrma/openQHA-Hessian`, where ADR 0011 puts
`BloomDlwlrma/mace`) -- fixed in this round; the earlier "flagged" disposition was too
conservative.

**Spec.** No scope creep; the fields, the reader-sharing and the 04a leftovers
(`run_training`'s `parameter_fingerprint` call, `t_judge_engine`'s final kwargs shape)
all check out. Two findings actioned:

- `06_judge.py`'s train-record auto-discovery still keyed on the retired
  `<base>-<run>` engine naming, so a model registered under the new
  `<campaign>-<run>+<stamp>` key would silently judge without its training curves
  -- fixed to derive the run from the engine name with `--tag` (the caller's own
  tag), help text follows. Not exercised by the suites (no registered engine
  exists yet); reasoned from the two naming rules.
- The Answer's "five tests on the `_testlib` locator" was loose (`t_train_run`
  reads no fixture) -- reworded.

Kept as 12's, per the spec: the `check_fork` repair message, the pinned
`FORK_COMMIT_B` (whose `!=` guard is vacuous against the rebuilt history until 12
updates it), the tutorials/READMEs and the `.scratch/hessian-learning-set` pointers.
Kept as documented in the Answer: the package README content list and the
workspace-root `mace/` shadowing note.
