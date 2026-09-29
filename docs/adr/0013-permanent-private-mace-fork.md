---
status: accepted
date: 2026-09-29
---

# The mace fork is a permanent private fork

## Context

The split's Q4 ruling ("the mace-side changes are organised as upstream-PR-able") led ticket 02 to rebuild `BloomDlwlrma/mace` on upstream history — branch `openqha-hessian` @ `1110ffb` on upstream tag `v0.3.16`, `base-v0.3.16` @ `5c2d761` — and ticket 10 was to turn the six commits into a PR against `ACEsuit/mace`. Ticket 10's grilling (2026-09-29; findings: `research/mace-md-fork-practice.md`) found the route closed for these files: upstream is mid-v1-rewrite, its CONTRIBUTING (same text on `main` and `develop`) lists `mace/data/atomic_data.py`, `mace/tools/arg_parser.py` and `mace/tools/train.py` among the "core and shared — v1 only" surface, admits features on v0.3 only with a v1-impossibility note, a self-contained directory and number-pinned tests, and routes v0.3 fixes through `develop` only (a PR against `main` is retargeted — precedent #1445). The reference private fork of the same trade — `jharrymoore/mace@softcore`, the mace-md project's — has been frozen since 2024-05-01 and was never offered upstream; its author's merged upstream PRs are unrelated small fixes.

## Decision

**No upstream submission.** The fork is a permanent private fork:

- **Release-line base** — the branch sits on the `v0.3.16` tag (already true; a tag anchor, unlike the reference fork's non-tag base).
- **Frozen** — no upstream tracking, no rebases; the refs `openqha-hessian@1110ffb` and `base-v0.3.16@5c2d761` are permanent anchors (Records, installs and the old→new mapping rely on them). A future base move is a new decision, not upkeep.
- **Bare branch-name reference** — consumers keep installing `git+…@openqha-hessian`; the identity layer on top (`0.3.16+openqha`, `mace_fork_info()`, the mapping) is kept — independent of upstreamability and stronger than the reference practice records.
- **Never upstream** — Q4's upstream-PR-able organisation is retired as a goal; the commits stay as written (no scrubbing — SHA stability wins; the AI-attribution trailers stop mattering once no PR will carry them).

## Considered options

- **Submit to `develop` with a scope disclosure** (the guide's "say so in the PR description" path). Rejected: "v1 only" is the likely answer, and the preparation cost (rebase across ~124 commits / 318 files of drift, scrubbing, test-suite adaptation) buys only the option of adoption plus a public record; the project need is already served by the fork.
- **Submit to `main`.** Rejected: `main` receives release cuts from `develop`; the retarget precedent makes it strictly worse.
- **Keep the PR-able posture without submitting.** Rejected: it taxes future work for an unpursued goal; the reference fork shows private freezing is accepted practice.

## Consequences

- The 02 rebuild's PR motive retires; its other gains stand (upstream-comparable history, lean clones from D, the mapping).
- Accepted costs: optionality foregone (no upstream adoption → no maintenance relief), no public review, base-rot risk — a later MACE upgrade means replaying D + the six commits by hand, as a new decision.
- The frozen refs must not be rewritten (SHA churn breaks Records and the mapping); any future change to `openqha-hessian` is itself a decision.
- Supersedes the forward-looking PR note in ADR 0011 (`The PR-side cost … is already anticipated by ticket 10`).
