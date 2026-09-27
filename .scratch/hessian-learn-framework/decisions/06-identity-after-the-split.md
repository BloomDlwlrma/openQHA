# Identity after the split: provenance fields, refusal rules, and the old-hash mapping

Type: grilling
Status: open
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

<!-- resolver: append the decision; set Status: resolved; add a line to the map's Decisions so far -->
