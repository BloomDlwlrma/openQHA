# Identity after the split: provenance fields, refusal rules, and the old-hash mapping

Type: grilling
Status: resolved
Blocked by: 01, 02
Part of: [hessian-learn-framework](../map.md)

## Question

A Record must still say which mace produced its numbers — now that mace comes from a rebuilt branch on a new fork, and the training code lives in a package. Decide:

1. **Engine constants.** `MACE_FORK = "BloomDlwlrma/openQHA-Hessian@openqha-hessian"` and `MACE_FORK_BASE = "base-v0.3.16"` (`openqha/potentials/engine.py:324–325`) become what? (new repo URL, branch name from [The real fork](02-the-real-fork.md), base on real upstream history).
2. **Package identity in Records.** Should judge/training Records also carry the *package's* version and commit, the way `mace_fork_commit` carries mace's? Field names (`EXTENSION_COMMIT`? `HL_PACKAGE_VERSION`?); or is the package covered by openQHA's own commit and not recorded per-Record?
3. **Refusal rules.** `training/run.py check_fork()` (206–222) refuses `unknown`/dirty mace. Keep unchanged? Extend to the package (unknown/dirty package → refuse)?
4. **Old hashes.** Existing Records cite `904dd3b` etc.; `T05` asserts "commit C or later". The rebuild changes commit ids (Q5 ruled this accepted). Where does the old→new mapping live (the fork's README? a docs page? a note kept in `.scratch/hessian-learning-set`?), and who gets the mechanical notebook updates ([References sweep](12-references-sweep.md)).
5. **`mace_fork_info()` contract.** Checkout root, `.git`, dirty semantics (untracked ignored) — anything to change for the new install shapes from [Install and transport](05-install-and-transport.md)?

## Answer

Resolved 2026-09-29 — the five questions settled by the rulings in
[spec-identity-after-the-split.md](../spec-identity-after-the-split.md) and fully executed:
the fork constant flip and the old→new mapping landed in
[06a](../implementation/06a-fork-identity-and-mapping.md) (`MACE_FORK` reads
`BloomDlwlrma/mace@openqha-hessian`; [ADR 0011](../../../docs/adr/0011-mace-fork-rebuild.md)
now carries the 40-hex table, the old-string note and the bundle path), and the package
identity in the Records (`HL_PACKAGE_VERSION` / `HL_PACKAGE_COMMIT` via
`package_identity()`, best-effort, `unknown` never refuses; one reader
`engine.checkout_commit` shared with the fork guard) rode the move
([01d](../implementation/01d-the-switchover.md), [07](07-the-move.md)). The refusal rules
are unchanged, the package is not gated, and `mace_fork_info()`'s contract is unchanged.
Old Records and fixtures stay untouched.

By the spec, still elsewhere: the sweep items (the training test's pinned old commit B →
rebuilt `61582b0e…`, the fork reader test's cosmetic id, T05's wording, the
`.scratch/hessian-learning-set` pointer notes, the gate's install-path text) are
[References sweep](12-references-sweep.md)'s.

Durable records: [ADR 0011](../../../docs/adr/0011-mace-fork-rebuild.md) (amended),
[ADR 0010](../../../docs/adr/0010-hessian-learning-lives-in-openqha-hessian.md). Findings:
[`research/mace-identity-practice.md`](../research/mace-identity-practice.md).
