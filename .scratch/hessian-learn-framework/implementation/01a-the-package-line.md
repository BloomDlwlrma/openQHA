# The package line: the openQHA-Hessian training package

Type: task
Status: open
Serves: 01
Blocked by: 01d
Part of: [hessian-learn-framework](../map.md)

> The `/to-spec` output for decision [The package line](../decisions/01-the-package-line.md)
> (resolved 2026-09-26). It is the brief [The move](../decisions/07-the-move.md),
> [The repo swap](../decisions/11-repo-swap.md) and
> [References sweep](../decisions/12-references-sweep.md) execute against.
> Per the effort's conventions in `docs/agents/issue-tracker.md` (Wayfinding operations),
> wayfinder tickets carry no `ready-for-*` triage labels; readiness is `Status: open`.

## Problem Statement

openQHA's repository carries two different things: the conformational free-energy framework
(CREST → MACE → OpenMM → msRRHO), and the Hessian-learning research that fine-tunes the
potential. The research side churns with the campaign, and its code — the PHL loss, the
HVP machinery, the judge, the training run — sits inside the framework's package, so every
reader of openQHA meets it and every change to training touches the framework's history.
The framework itself does not use it: nothing in `openqha/` outside that directory imports
it. The research side wants to stand on its own — its own repository, version, README and
tests — while its campaign (drivers, Slurm, tutorials, Records) keeps working exactly as
today.

## Solution

The training side becomes a package of its own: **`openQHA-Hessian`** hosts the distribution
`openqha-hessian` (import `openqha_hessian`, version 0.1.0); `openqha` never imports it. The
openQHA repository keeps the campaign surface — `05_train.py`, `06_judge.py`,
`hl_train.slurm`, the `s0_*` scripts, the campaign doc, T04/T05 — and simply calls the
package. The old `openqha.training` import path ceases to exist; every reference is swept.
The package ships its own test runner, running the tests that moved with their code.

## User Stories

1. As the researcher, I want the training code to live in its own repository, so that its churn no longer appears in openQHA's history
2. As the researcher, I want `openQHA-Hessian` to stand on its own (README, version, tests), so that I can hand someone the training framework without handing them the whole free-energy repository
3. As a reader of openQHA, I want `import openqha` to stay light and bare of training code, so that the framework's boundary is visible in every import
4. As a future reader, I want the ADR, the map and the package README to say where the training side went, so that an old `openqha.training` reference is diagnosed in one look rather than one hunt
5. As the campaign operator, I want `05_train.py` and `06_judge.py` to keep their CLI and behaviour, so that the existing Slurm scripts and the campaign documentation stay valid
6. As the campaign operator, I want the only mace-argv difference to be the `--loss_module` address, so that a `--dry-run` diff is a known, one-string change
7. As the training-run reader, I want each Record to keep naming the loss it was produced with, so that old and new runs stay auditable side by side — old Records untouched, new ones carrying the new address
8. As the implementing agent, I want a fixed list of the files that move and of the addresses that change, so that the move is mechanical and reviewable
9. As the implementing agent, I want the package's identity settled (distribution, import name, version, layout, no console scripts), so that no naming question is reopened mid-move
10. As the package maintainer, I want the moved tests to run under the package's own twin runner, so that the package is verifiable without openQHA's test suite
11. As the package maintainer, I want the tests to reuse openQHA's fixtures rather than carry copies, so that fixture data stays single-sourced
12. As the mace-fork maintainer, I want the loss address in the fork's CLI (`--loss external --loss_module ...`) to resolve to the new package with no fork-side change, so that the fork's diff stays purely the mace-internal work
13. As the operator, I want the campaign's acceptance to be two explicit commands (openQHA's suites; the package's runner), so that "done" is checkable
14. As the Tianhe operator, I want the install story to name the three things a machine needs (openQHA checkout, package, fork), so that a fresh node reaches a training run without archaeology

## Implementation Decisions

- **Boundary**: every module of `openqha/training/` moves wholesale — no line is drawn inside the directory; `judge`, `run` and `smoke_fit` move with the openqha-free core. Module names are kept.
- **Direction**: the package depends on openQHA (`data`, `store`, `thermochem`, `potentials`) and on the mace fork; the `openqha` library never imports the package; openQHA's consumers import the package. The graph is acyclic by construction; interface inversion was rejected (the package's semantics are openQHA's).
- **Compatibility**: no shim, no extension of the `_MOVED` compatibility layer; `openqha.training` dies as an address and every reference is swept.
- **Identity**: distribution `openqha-hessian`, import `openqha_hessian`, version 0.1.0; flat one-package layout with `pyproject.toml` (setuptools), `tests/`, `install.sh`, rewritten README; no console scripts — the entry points stay in openQHA's workflows.
- **Loss address**: `--loss_module` becomes `openqha_hessian.phl_loss:build`; it is the sole mace-argv change; old Records keep the old string (nothing resolves it), new Records carry the new one.
- **Tests**: the nine tests of the moved code move with it (6 unit + 3 integration) into the package's mirrored `tests/{unit,integration}/`, run by a twin of openQHA's runner with the same conventions plus a provenance header; openQHA's runner drops them and gains no forwarding flag.
- **Fixtures**: reused, not copied — the moved tests locate the openQHA checkout from the imported `openqha` package (`OPENQHA_SRC` overrides) and keep reading the existing `tests/data` fixtures.
- **Consequences handed to other tickets**: install/transport owes "how is `openqha` importable" (05); record-field changes ride (04, 06); the move, repo swap and sweep are (07, 11, 12).
- **Durable record**: ADR `0010`.

## Testing Decisions

- A good test here asserts external behaviour — the numbers the loss produces, the argv mace receives, the Record written, the fork gate's refusals — never module layout or private helpers.
- Prior art: the existing plain-script tests under `tests/{unit,integration}/t_*.py`, run by a `run_tests`-style runner (one subprocess per file, `check(label, ok)` lines, a FAIL list, exit code). The moved tests keep that shape; the only new mechanism is the twin runner and the fixture locator.
- Under test: the moved package modules, by the moved tests. No new seams are introduced — the tests run at the existing seams (module functions, CLI argv, Records).
- Acceptance (as [The move](../decisions/07-the-move.md) states it): openQHA's unit + integration groups pass; the package's own runner passes; `05_train --dry-run` prints the same mace argv as before except the loss address.

## Out of Scope

- SHA256 retirement and native model loading changes to the moved files (04).
- Record identity fields, fork constants, the old-to-new commit mapping (06).
- Install and transport mechanics, the fork itself, the upstream PR (05, 02, 10).
- The repo content swap and force-push (11); the documentation/notebook sweep (12).
- New training features, and any change to the campaign's science.

## Further Notes

- Vocabulary: "openQHA-Hessian" means the package; the mace side is "the mace fork" (`BloomDlwlrma/mace`).
- The local folder `openQHA-Hessian/` today is the fork's checkout; it becomes the package's checkout and the fork moves elsewhere (02).
- The `ready-for-agent` label is deliberately not applied: per the tracker doc and ADR 0009, wayfinder tickets use the `Status` protocol, not triage labels.
