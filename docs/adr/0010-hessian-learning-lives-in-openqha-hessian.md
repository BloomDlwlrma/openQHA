---
status: accepted
date: 2026-09-26
---

# The Hessian-learning training side lives in the `openQHA-Hessian` package

## Context

`openqha/training/` — the PHL loss, the HVP machinery, the judge and the training run — is
research code that churns with the Hessian-learning campaign, while `openqha/` is the
conformational free-energy framework whose readers meet it as one more subpackage. The
split ruling (2026-09-25/26) moved the training side out of the repository; the boundary,
the names and the direction were settled in ticket 01 of `.scratch/hessian-learn-framework/`
(2026-09-26).

## Decision

The whole of `openqha/training/` (seven modules) moves into the `openqha-hessian`
distribution — import `openqha_hessian`, version 0.1.0 — with its nine tests. The
dependency direction is **package → openqha**: the package imports
`openqha.{data,store,thermochem,potentials}` and the mace fork; openQHA's workflows,
scripts and tutorials import the package; **the `openqha` library never imports it**, and
`openqha.training` ceases to exist — no compatibility shim. The loss address becomes
`openqha_hessian.phl_loss:build`; old Records keep the old string as data.

## Considered options

* **Move only the openqha-free core** (`hvp`, `phl`, `phl_loss`) and leave
  `judge`/`run`/`smoke_fit` in openQHA. Rejected: the training side's substance — the run
  loop, the judge would stay behind, and "the training side moved out" would be half true.
* **Invert the seam**: openQHA supplies data/store/thermochem/potentials through an
  interface the package owns. Rejected: the package's semantics *are* openQHA's (molecule
  trees, Record formats, engine registry); the interface would be a second openQHA bought
  to remove a cycle that does not exist (the library imports nothing).
* **A `_MOVED`-style compatibility alias for `openqha.training`.** Rejected: the existing
  compatibility layer maps old names to in-package homes and cannot express a departure;
  every live reference is swept by the effort's own tickets, and old Records do not resolve
  the string.

## Consequences

- A training machine needs three things: the openQHA checkout, the package, and the mace
  fork; making `openqha` importable is the install ticket's obligation, not a library
  import edge.
- The test surface splits into two commands — openQHA's `tests/run_tests.py` groups and
  the package's twin runner; the moved tests reuse openQHA's `tests/data` fixtures from
  the checkout rather than carrying copies.
- Readers of `openqha/` no longer find the training code; this ADR, the effort map and the
  package's README are where "where did it go" is answered.
- The move, the repo content swap and the reference sweep land as tickets 07/11/12 of the
  effort; this ADR records the shape they execute.
