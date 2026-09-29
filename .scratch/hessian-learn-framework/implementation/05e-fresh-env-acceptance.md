# 05e: Fresh-env acceptance — the documented lines, then close out

Type: task
Status: resolved
Serves: 05
Blocked by: 05b, 05c, 05d
Part of: [hessian-learn-framework](../map.md)

> Execution slice of [05a](05a-install-and-transport.md): the spec's Testing Decisions and ticket
> 05's acceptance. Wayfinder conventions apply: the `Status` protocol, no triage labels
> (`docs/agents/issue-tracker.md`, ADR 0009).

## What to build

One fresh WSL environment, built from the documented lines (environment files → activate → install
script → probes) — the effort's single testing seam, all through pre-existing interfaces: `import
mace` resolves to the fork checkout; `mace_fork_info()` is clean; `run.check_fork(strict=True)`
passes; `import openqha_hessian` works. Commands and outputs are recorded. Then ticket 05 is
resolved: Answer with the evidence, one line in the map, `Status: resolved` — and the decision's
durable record (ADR, per the 01/02 precedent) unless the resolver explicitly declines it.

## Acceptance

- [x] From the fresh environment and the documented lines: `import mace` → the fork checkout,
      `mace_fork_info()` clean (full commit), `check_fork(strict=True)` passes, package imports
- [x] Commands and outputs recorded on ticket 05's Answer; ticket resolved; map line added
- [x] No suite tests added for the install (per the spec's testing decision); anything not verified
      stated
- [x] Reported per the operating rule

## Answer

Resolved 2026-09-29, agent-run. The effort's single testing seam is green: one fresh WSL
environment, built from the documented lines (environment file → activate → the README's
by-hand exports → `install.sh` → the probes), ends with `import mace` inside the editable
fork clone (`1110ffb`, tracked-clean), `mace_fork_info()` clean, `run.check_fork(strict=True)`
passing and `openqha_hessian` importing. The commands and outputs are recorded on ticket
[05](../decisions/05-install-and-transport.md)'s Answer; the decision's durable record is
[ADR 0012](../../../docs/adr/0012-install-and-transport.md).

Two facts belong to this run specifically:

- **The watch-list item, resolved and enlarged.** [05a](05a-install-and-transport.md) asked
  where an editable VCS clone lands in a conda environment (its guess: the environment's
  `src`). It lands in `<cwd>/src`: pip 26.2.1 classifies a conda environment as a global
  install (`running_under_virtualenv() == False`), so the run from the README's implied cwd —
  the openQHA checkout root — put the fork clone at `openQHA/src/mace-torch` (4.9 MB, blobless,
  `1110ffb`, tracked-clean; `src/` is not git-ignored, so a doc-following user gets a `?? src/`
  entry). Reported, not fixed here — candidates: `install.sh` passing `--src "$CONDA_PREFIX/src"`
  to the fork step, or a docs line.
- **The package the acceptance exercised was the switchover state**: the checkout's HEAD was
  `ff92141` ("The switchover: judge, run and smoke_fit move in — the Records carry the package
  identity", 12:42 the same session), so `run.check_fork` really is the moved module.

The scratch environment and the clone were removed after the run (throwaway acceptance
artifacts, per the 05b throwaway-venv precedent).
