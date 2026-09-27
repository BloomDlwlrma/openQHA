# 02b: The rebuild — rebase onto upstream history, archive the old fork

Type: task
Status: resolved
Serves: 02
Blocked by: None (can start immediately)
Part of: [hessian-learn-framework](../map.md)

> Execution slice for [The real fork](../decisions/02-the-real-fork.md); the brief is
> [02a](02a-the-mace-fork.md). Agent-run, no user action required. Wayfinder
> conventions apply: the `Status` protocol, no triage labels
> (`docs/agents/issue-tracker.md`, ADR 0009).
> The rebuild runs in the existing `openQHA-Hessian/` checkout — it stays the mace
> workbench until 02d re-homes it.

## What to build

The local mace checkout carries the rebuilt branch `openqha-hessian` on real upstream
history: `4d2da09` (upstream v0.3.16) → D (fork-only: the three bundled foundation-model
binaries dropped) → the six feature commits replayed — with content identical to the old
tip, the tag `base-v0.3.16` at D, and the old fork's history archived as a verified local
bundle. The rebuild performs no network transfer: every object it needs is already local.

## Acceptance

- [x] The archive bundle exists (`_backup/openQHA-Hessian-old-history-2026-09-26.bundle`); `git bundle verify` passes; it contains exactly `refs/heads/openqha-hessian` and `refs/tags/base-v0.3.16` at the old SHAs; it is small (no upstream blobs inside).
- [x] D's parent is `4d2da09`; its only change is dropping the three bundled `.model` paths; its message marks it fork-only and never-upstream; no new objects were fetched to build it.
- [x] The six commits are replayed onto D, with the one expected `.gitignore` hunk resolved by keeping upstream's file and the version bump.
- [x] Content invariance: a diff between the old tip `f14a56f` and the new tip is empty.
- [x] The tag `base-v0.3.16` points at D; base→branch log is exactly six commits; base→branch diff is the same whole-change as the old world (11 files, +869/−17 — see Answer).
- [x] Reported: D and new-tip SHAs (the material ticket 06's mapping consumes), anything unexpected.

## Answer

Resolved 2026-09-27, in `openQHA-Hessian/`, with no network transfer: the clone carries no
partial-clone/promisor filter (so no lazy fetching exists), no fetch was run, and every
object the rebuild touched — v0.3.16's trees and the files the six commits change — was
already local (checked before the rebase).

- **Archive bundle** `_backup/openQHA-Hessian-old-history-2026-09-26.bundle` (workspace
  root; 510 KiB / 522,591 bytes, 279 objects, made before the rebase): exactly
  `refs/heads/openqha-hessian` @ `f14a56f` + `refs/tags/base-v0.3.16` @ `8fac5d1`;
  `git bundle verify` says "records a complete history ... is okay".
- **D = `5c2d7612eed88dc1463b5588a79c2d5f5718d322`** — parent `4d2da09`; built object-level
  from v0.3.16's tree minus
  `mace/calculators/foundations_models/{2023-12-03-mace-mp,ani500k_large_CC,mace-mpa-0-medium}.model`;
  message marks it fork-only / never-upstream. vs upstream: exactly those three deletions;
  vs the old base `8fac5d1`: exactly the old base's stray `.gitignore` line (below).
- **The six commits replayed** — the old→new mapping ticket 06 consumes:

  | # | old | new |
  |---|-----|-----|
  | 1 | `0ae78c4` version 0.3.16+openqha | `a31d0a6e3634896c349190c6e19aa9fb67e77049` |
  | 2 | `1b9c382` hessian label | `1b530325313cb058efe36abed9976be54c9c744c` |
  | 3 | `e68390f` external loss hook | `61582b0efb61cc251457c52367bb8239fe1cb2b5` |
  | 4 | `904dd3b` multihead/eval | `cce52c0963d36b493549a3b78e32c174f98d3c2e` |
  | 5 | `a37f8b6` probes | `66a68f0e4ad2158fcba9e0c64a890bdf5d4ec2d9` |
  | 6 | `f14a56f` probe default (old tip) | `1110ffbafd651a74d1d4678deb4748056d1dff0a` (new tip) |

  `base-v0.3.16` moved `8fac5d1` → `5c2d761` (D).
- **Verification**: old tip vs new tip (`git diff f14a56f 1110ffb`) is empty, and each old
  commit vs its replay is empty too — all six trees are pairwise identical, so the rebuild
  changed parentage only. The expected `.gitignore` conflict needed no manual stop: the
  replayed version-bump commit takes upstream's file as-is and keeps only the version bump
  (its diff is `mace/__version__.py` alone; it keeps its original message, whose subject
  still names the .gitignore drop). Branch `openqha-hessian` = `4d2da09 → D → the six`;
  working tree clean.
- **Reported, unexpected**: the new `base→branch` diff is **11 files, +869/−17**, not the
  old world's 12/+869/−18 — the one difference is the old squashed base's stray
  `.gitignore` line `mace/calculators/foundations_models/*.model` (redundant: upstream's
  `*.model` covers the binaries), which D by definition does not carry. The six-commit
  change itself is byte-identical to the old world; measured against upstream, both worlds
  read 14 files, +869/−17.
- Push is 02c's (user): branch + moved tag from VS Code SCM, expected ≲1 MB. The old fork
  on GitHub is untouched.
