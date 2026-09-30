# 12h: The CI and the close-out

Serves: [12](../decisions/12-references-sweep.md) · spec: [spec-references-sweep.md](../spec-references-sweep.md).

**What to build:** The openQHA repository gains its minimal CI — a workflow shaped like the package's (ubuntu-latest, Python 3.11, on push to main and pull requests; light dependencies only; the eight smoke tests run directly), with a local dry-run of those eight passing before anything is pushed. Then the close-out: a final full-suite run recorded as the retirement baseline plus the package suite; the residual-citation scans over code and documents; the commits (three on openQHA — the sweep, the comment/string passes, the CI and the record — and one on the package) pushed to both remotes; the first CI run green (URL recorded); the ticket record — the Answer, the map line and `Status: resolved` — including the public-face boundary and the two reported facts.

**Blocked by:** 12a–12g, 12i (everything else).

**Status:** resolved

- [x] The workflow is valid and the eight smoke tests pass locally.
- [x] The final full-suite run and the package suite are green and recorded.
- [x] The residual scans are empty (code and documents).
- [x] Both remotes carry the commits; the first CI run is green (URL in the Answer).
- [x] Ticket 12's Answer, the map line and `Status: resolved` are in place; the public-face boundary and the two reported facts are recorded.

## Answer (2026-09-30, the close-out)

**The CI.** `.github/workflows/ci.yml` — ubuntu-latest, Python 3.11, `numpy`, `ase` and
`PyYAML` (the light set the eight actually need — the clean-env dry-run caught
`t_msrrho_presets` reading the repository's YAML config, one light dependency more than the
plan assumed), the eight smoke tests one step each (mirroring the package workflow).
Validated before anything was pushed: a clean clone of `f368bd6` under a scratch env
(Python 3.11 + the install line) ran all eight green; the YAML parses; the first CI run is
green (https://github.com/BloomDlwlrma/openQHA/actions/runs/36665595358).

**The close-out.** Final suites: openQHA `--all` **73/73**; the package `--all` **9/9**;
the `mace` checkout verified frozen (`1110ffb`, clean). Residual scans over both
repositories' public faces — run on a clean clone of the close-out tree with the
citation-file fixes applied, notebook sources only, the archived T03 and the stored outputs
excluded: **0 violations**, 143 kept-class lines each with a recorded reason (the
old-assets sample records; the artifact index's, the baseline's and the store's decision
values; the mem-decide namespace; the decision-format grammar; the functional context
reads; the campaign's own round vocabulary). The scan's wider file-type net also caught
two comment leftovers in `docs/cite/cite_openQHA.bib` (a study code, a ticket range) —
fixed in this close-out. Commits: `f368bd6` (the CI, already on the remote) and the commit
this record rides.

**Reported per the operating rule.** Files changed: `.github/workflows/ci.yml` (new),
`docs/cite/cite_openQHA.bib` (two comments), and the ticket files this close-out edits
(`12h-the-ci-and-the-close-out.md`, `decisions/12-references-sweep.md`, `map.md`,
`spec-references-sweep.md`). Checks run: the workflow's YAML parse; the eight smoke tests
in a clean clone under a scratch env; both suites (`--all` 73/73 and 9/9); the residual
scans (0 violations); the `mace` freeze check; the first remote CI run (green). Not
verified: the `pull_request` trigger path (no PR existed to exercise it); the dry-run's
Python was conda 3.11, not the runner image — the remote run is the authoritative check.

**Recorded, not fixed.** `RETRY_FAILED` (no live counterpart — the retry contract reads
`RETRY_ONLY`); the old-set pointer note (superseded by the boundary ruling).

## Review record (2026-09-30, pre-commit)

Two-axis review of the close-out change set (the `code-review` skill; two read-only
sub-agents on the pre-commit state); findings fixed in-pass:

**Standards.** Hard: the Answers lacked the operating-rule report (files changed / checks
run / not verified) — added to both; the spec file's `(open — …)` twin marker was stale —
dropped with the ticket's. Judgement, fixed: the "nine slices landed under one rule set"
sentence overclaimed (this close-out applies no comment/string pass) — reworded to the
eight passes; the CI job id `smoke` alone reads ambiguously beside the campaign's
smoke/fit vocabulary — `unit-smoke`. Judgement, kept: the eight separate workflow steps
(the 11d review ruled separate named steps are the point); process tokens inside tracker
text (the boundary ruling).

**Spec.** No missing or partial requirements: each of the slice's five checkboxes is
evidenced by the recorded text (the clean-checkout dry-run of the eight and the first
green run for the workflow; both suites; the 0-hit scans; the remote carry and the URL;
the Answer, map lines and resolved status carrying the boundary and the two facts). No
material scope creep — the three extra edits (the spec-file marker, the map Implementation
line, the slice's Review record) are established conventions or needed so a resolved
record does not still read open. Nothing implemented wrong: the test list, the install
line, the triggers and the record wording all match the spec's items; the delivery-shape
deviation is the recorded finer delivery. The clean-env dry-run then added one finding of
its own before anything was pushed — the missing PyYAML — fixed in the install line.
