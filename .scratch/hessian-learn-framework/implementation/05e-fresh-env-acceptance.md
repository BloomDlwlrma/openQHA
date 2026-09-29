# 05e: Fresh-env acceptance — the documented lines, then close out

Type: task
Status: resolved
Serves: 05
Blocked by: 05b, 05c, 05d
Part of: [hessian-learn-framework](../map.md)

> Execution slice of [the spec](../spec-install-and-transport.md): its Testing Decisions and ticket
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
passing and `openqha_hessian` importing. The recorded commands, outputs and the two run
findings (the conda clone location; the switchover state the probes exercised) are on
ticket [05](../decisions/05-install-and-transport.md)'s Answer; the decision's durable
record is [ADR 0012](../../../docs/adr/0012-install-and-transport.md). The scratch
environment and the clone were removed after the run (throwaway acceptance artifacts).
