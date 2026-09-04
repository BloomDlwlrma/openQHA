# `scripts/`

Four categories, and the rule that decides which one a script belongs to. The category
is declared in the script's own module docstring, and
`tests/unit/t_script_taxonomy.py` fails if a declaration and a directory disagree.

| Directory | A script belongs here when… | Count |
|---|---|---:|
| `production/` | its output **enters a deliverable**. Delete it and some edge has no number | 10 |
| `calibration/` | its output is **a number used to make a decision** (cost, error, control). It reaches a checkpoint, not a deliverable | 11 |
| `diagnostics/` | it was written to locate **one numbered defect**. Still runnable, rarely run | 0 — see below |
| `tooling/` | it manages **the repository itself** (identifiers, format conversion, notebook checks, migration) and produces no science | 7 |
| `_superseded/` | retired, kept as evidence, on no live path. See its own README | 26 |

**Why the category is declared as well as implied by the directory.** Classification is a
judgement, not a fact (plan_D section 2.1), and a judgement recorded in only one place
cannot be checked. A script that says `CALIBRATION` while sitting in `production/` is a
disagreement — between two people, or between the same person three weeks apart — and the
taxonomy test is what makes it visible instead of letting the directory quietly win.

## `diagnostics/` is empty, and that is not an oversight

The defect probes it was meant to hold went to `scripts/_superseded/closed-defects/`
instead, on the user's ruling `S0-D-2`. That leaves a tension with the definition above,
which says a probe should stay runnable after its defect closes — and plan_D section 2.3
is how it resolves: each probe becomes a named regression test under `tests/regression/`,
carrying its defect number in the file name. Once the test exists, the probe can be
deleted safely, because the reproduction is no longer the script's job.

## Finding the repository root

Every script here locates the repository by walking up until it finds the directory
containing `openqha/__init__.py`. **No script counts directory levels.** `parents[1]` was
correct only at one specific depth, and when 26 scripts were moved into
`scripts/_superseded/<group>/` it silently started pointing at `scripts/_superseded/`
instead of the root — nothing failed at move time or at import time.
`tests/unit/t_repo_bootstrap.py` enforces the replacement, and checks it from every depth
that actually occurs.

## Writing a product

Do not choose a path. Call `openqha.artifacts.write_artifact(...)`, which requires
`category`, `status` and `produced_by`, refuses a `produced_by` that is not a real file,
and decides the destination itself. See `openqha/artifacts.py` for why.
