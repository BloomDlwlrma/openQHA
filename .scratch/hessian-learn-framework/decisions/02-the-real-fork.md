# The real fork: `BloomDlwlrma/mace`, rebuilt on upstream history

Type: task
Status: resolved
Part of: [hessian-learn-framework](../map.md)

## Question / work

Create the true fork and land the mace-side changes on it, so the repo shows exactly what is ours (Q1) and the commits can go upstream (Q4a).

Work:

1. **Create the fork** (user action, HITL): a GitHub-side fork of `ACEsuit/mace` as `BloomDlwlrma/mace`.
2. **Rebuild the branch on real upstream history.** Today the branch hangs off the orphan `8fac5d1` (squashed `base-v0.3.16`), so nothing compares against upstream. Rebase the 6 commits (`0ae78c4`, `1b9c382`, `e68390f`, `904dd3b`, `a37f8b6`, `f14a56f`) onto upstream `4d2da09` (tag `v0.3.16`) and push only what is needed — the previous attempt to push full history from this machine failed three times; the 490 KB squashed push worked, so design for small pushes (partial clone / blob filter if the 153 MB of bundled binaries get in the way).
3. **Decide and record, in this ticket:** the branch name on the fork (recommend keeping `openqha-hessian`); whether the branch carries a "drop the three bundled foundation-model binaries" commit at all now (the old base dropped them; on a real fork they exist upstream and never need a push — but check what the local build needs); what `base-v0.3.16` should mean now (a tag at upstream `4d2da09`? at the rebuilt base?); where the local mace checkout lives after the split, given `openQHA-Hessian/` becomes the package (sibling `mace/`? the checkout pip makes from a git URL?); where the old history bundle (Q5) is saved.
4. **Verify:** a fresh clone of the fork installs with `pip install -e`, `mace.__version__ == 0.3.16+openqha`, and `engine.mace_fork_info()` reads a clean commit (40-hex, not dirty) from it.

## Facts to stand on

- Old fork: 7 commits; whole change = 12 files, +869/−18; the base dropped `mace/calculators/foundations_models/*.model` (153 MB) and upstream's `*.model` gitignore line already covers them.
- Remotes on the local checkout: `origin BloomDlwlrma/openQHA-Hessian`, `upstream ACEsuit/mace` (upstream kept locally only).
- This ticket owns the fork. Installing it everywhere is [Install and transport](05-install-and-transport.md); recording its identity is [Identity after the split](06-identity-after-the-split.md); the upstream PR is [Upstream: submit the mace-side changes](10-upstream-pr.md).

## Answer

**Decisions settled** (grilling 2026-09-26/27). **Amended 2026-09-27: execution + hashes
landed — appended below.** Slice:
[`implementation/02a-the-mace-fork.md`](../implementation/02a-the-mace-fork.md); durable
record: ADR `0011` (`openQHA/docs/adr/0011-mace-fork-rebuild.md`).

1. Branch name stays **`openqha-hessian`**.
2. **Carry a fork-only binaries-drop commit D** as the branch root (parent `4d2da09`);
   the local build needs none of the three files (mace downloads/caches; openQHA weights
   live in `data/potentials/`; mace's tests passed without them on the old fork). The old
   drop was for push feasibility; on a true fork the reason is lean checkouts / blobless
   workflows, and 10 already anticipates excluding it from the PR.
3. **Default branch: no settings change** — the fork keeps whatever GitHub creates (= the
   parent's default, `develop`; `main` is a *differing* line — the "2 commits ahead of /
   1 behind develop" banner is a non-default-branch banner, not evidence main is the
   default). **Revised 2026-09-27: switched to `openqha-hessian` so the landing page shows
   our branch — see the [02c](../implementation/02c-fork-and-push.md) Answer postscript.**
4. Local mace checkout re-homes to the sibling **`mace/`** (the fresh-clone acceptance
   lands there); the old folder retires to `_to_delete/` after verification. Tianhe
   landing guidance is owed by **05** (mace-md pattern).
5. Old history archive: **`_backup/openQHA-Hessian-old-history-2026-09-26.bundle`**,
   made before the rebase, containing exactly the branch + `base-v0.3.16` (never `--all`),
   `git bundle verify`-checked.
6. Push channel: HTTPS `origin` (new fork), user pushes both refs from VS Code SCM
   (≲1 MB); agent read-only-verifies SHAs via the GitHub API.
7. **`base-v0.3.16` points at D** and is pushed with the branch — the name keeps meaning
   "upstream v0.3.16 minus the three bundled binaries" and
   `git diff base-v0.3.16..openqha-hessian` stays the six commits.
8. Expected single rebase conflict: the version-bump commit's `.gitignore` hunk
   (resolution: upstream's file + the version bump).

Facts verified for the above: the new fork does not exist yet (404); the six commits and
the orphan base are still the local tip (`f14a56f`); the required objects (v0.3.16 tree
at depth 1) are already local, so the rebuild downloads nothing large; the fork being a
true fork means the push is an object diff against the shared upstream pool.

**Execution (2026-09-27).** Done across the three slices — [02b](../implementation/02b-the-rebuild.md)
(rebuild), [02c](../implementation/02c-fork-and-push.md) (push), [02d](../implementation/02d-clone-verify-rehome.md)
(clone, verify, re-home). The fork `BloomDlwlrma/mace`
(https://github.com/BloomDlwlrma/mace) carries `openqha-hessian` @
`1110ffbafd651a74d1d4678deb4748056d1dff0a` and `base-v0.3.16` @
`5c2d7612eed88dc1463b5588a79c2d5f5718d322` (D); the default branch was switched to
`openqha-hessian` (decision 3, revised). A fresh blobless clone at the sibling `mace/`
installs editable in the WSL `openqha` env — `mace.__version__ == 0.3.16+openqha` and a
clean `mace_fork_info()` (commit `1110ffb`, not dirty) read from that clone. The old
checkout is archived at `_to_delete/openQHA-Hessian-old-fork-2026-09-26/`; the old
history stays bundled at `_backup/openQHA-Hessian-old-history-2026-09-26.bundle`; the
old → new SHA mapping feeds ticket 06 (table in the
[02d](../implementation/02d-clone-verify-rehome.md) Answer).
