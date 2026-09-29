# The mace fork: rebuild on upstream history, push, verify, re-home

Type: task
Status: resolved
Serves: 02
Part of: [hessian-learn-framework](../map.md)

> The `/to-spec` output for decision [The real fork](../decisions/02-the-real-fork.md)
> (decisions settled 2026-09-26/27; execution pending). It is the brief
> [Install and transport](../decisions/05-install-and-transport.md) consumes ("install
> the fork from its URL at this branch") and the base [Identity after the
> split](../decisions/06-identity-after-the-split.md) maps against.
> Per the effort's conventions in `docs/agents/issue-tracker.md` (Wayfinding
> operations), wayfinder tickets carry no `ready-for-*` triage labels; readiness is
> `Status: open`.

## Problem Statement

The six mace-internal commits of the Hessian work hang off an orphan base on
`BloomDlwlrma/openQHA-Hessian`: nothing compares against upstream, nothing can go
upstream, and the local checkout that holds them sits in the folder the package is about
to claim. The side needs its own true fork — real upstream history, a real diff, a
PR-able branch — without re-introducing the 153 MB the old base deliberately left out,
and without losing the old history when the old fork's content is later replaced.

> 2026-09-29 — the upstream PR was decided against ([ticket 10](../decisions/10-upstream-pr.md)): the fork is permanent-private ([ADR 0013](../../../docs/adr/0013-permanent-private-mace-fork.md)).

## Solution

`BloomDlwlrma/mace` becomes a true GitHub-side fork of `ACEsuit/mace`. On it, branch
**`openqha-hessian`** = upstream v0.3.16 (`4d2da09`) → **D** (fork-only: the three
bundled `.model` binaries dropped) → the six commits rebased. Tag **`base-v0.3.16`**
points at D; `git diff base-v0.3.16..openqha-hessian` is the six commits — the old
diff convention, kept. The fork's default branch is left exactly as GitHub creates it.
The old orphan history is archived as a local bundle before anything moves; the local
checkout re-homes as the sibling `mace/` and the old folder is retired to `_to_delete/`.

## User Stories

1. As the researcher, I want the mace changes on a true fork of upstream, so that the fork's diff against upstream is exactly what is ours
2. As the upstream reviewer (ticket 10), I want the feature commits rebased onto real upstream history, so that a PR is a rebase away rather than a re-creation. 2026-09-29 — the upstream PR was decided against ([ticket 10](../decisions/10-upstream-pr.md)): the fork is permanent-private ([ADR 0013](../../../docs/adr/0013-permanent-private-mace-fork.md)).
3. As the fork's consumer (WSL env, Tianhe), I want no bundled binaries in the branch's tree, so that checkouts, tarballs and blob-filtered clones stay source-sized
4. As the old Records' auditor (ticket 06), I want the old commit ids and base tag archived in a bundle, so that cited hashes stay resolvable after the old fork's content is replaced
5. As the diff reader, I want `base-v0.3.16` to mean "upstream v0.3.16 minus the three bundled binaries" on the new fork too, so that "base..branch = our whole change" survives the rebuild
6. As the operator, I want the rebuild to change ancestry and not content, so that a single tree-identity check proves the six commits survived the move
7. As the pusher, I want the push to be a small object diff (the fork already holds upstream's objects), so that the old failed-full-history-push episode cannot recur
8. As the owner, I want the default branch left untouched, so that the fork reads as "upstream mirror + one branch of ours"
9. As the local operator, I want the mace checkout at a sibling `mace/`, so that the `openQHA-Hessian/` path is free for the package (ticket 07/11)
10. As the installer (ticket 05), I want a git URL + branch to consume (`git+https://github.com/BloomDlwlrma/mace.git@openqha-hessian`), so that no artifact type needs inventing
11. As the training run, I want a fork checkout whose `mace_fork_info()` reads a clean 40-hex commit, so that the fork gate's refusals stay silent
12. As the verifying agent, I want acceptance to be clone-install-import checks, so that "the fork works" is demonstrated, not asserted

## Implementation Decisions

- **Fork identity**: true GitHub-side fork `BloomDlwlrma/mace` (site fork of
  `ACEsuit/mace`; upstream copy retains no old-fork content). Branch
  `openqha-hessian`; tag `base-v0.3.16` → D; upstream's own tags and branches are
  inherited by the fork unchanged.
- **Rebuild**: `git rebase --onto D 8fac5d1 openqha-hessian` on the existing checkout.
  D is built object-level from `v0.3.16`'s tree (the three
  `mace/calculators/foundations_models/*.model` paths removed) with parent `4d2da09`
  and a message marking it fork-only / never-upstream.
- **Expected conflict**: the version-bump commit carries a `.gitignore` hunk whose lines
  do not exist upstream (the tip's `.gitignore` already equals upstream's); resolution =
  keep upstream's file, keep the version bump.
- **Content invariance check**: `git diff f14a56f <new tip>` must be empty — the rebuild
  changes parentage only.
- **Archive**: `_backup/openQHA-Hessian-old-history-2026-09-26.bundle` (workspace root
  `_backup/`), made **before** the rebase, containing exactly
  `refs/heads/openqha-hessian` + `refs/tags/base-v0.3.16` (never `--all` — that would
  pull upstream's ~160 MB into the bundle), verified with `git bundle verify`.
- **Push channel**: a HTTPS `origin` on the new fork; the user pushes `openqha-hessian`
  and `base-v0.3.16` from VS Code's SCM (stored credentials, per the Sept precedent);
  expected volume ≲1 MB.
- **Verification split**: remote refs are read back via the GitHub API; the acceptance
  checks (`git clone --filter=blob:none` into `mace/`, `pip install -e`, import checks)
  run beforehand in a scratch clone so the acceptance is demonstrated, not assumed.
- **Re-home**: the new `mace/` checkout is a fresh clone; its `upstream` remote keeps the
  old checkout's pinned tag refspec (never fetch upstream's whole history); the old
  `openQHA-Hessian/` checkout retires to `_to_delete/openQHA-Hessian-old-fork-2026-09-26/`
  after verification.
- **Consequences handed to other tickets**: install/transport (05) consumes the URL;
  identity constants + old→new mapping (06) must be updated against the new ids; the
  upstream PR (10) excludes D and the version bump, or marks them clearly; the repo
  swap (11) points its README at the bundle. 2026-09-29 — the upstream PR was decided against ([ticket 10](../decisions/10-upstream-pr.md)): the fork is permanent-private ([ADR 0013](../../../docs/adr/0013-permanent-private-mace-fork.md)).
- **Durable record**: ADR `0011`.

## Testing Decisions

- A good check here asserts external state — a bundle that verifies, a tree that is
  identical, refs that resolve upstream-side, a clone that imports and reports a clean
  commit — never internal layout or command transcripts.
- Seams (three, all existing):
  1. the git object/ref level (`git bundle verify`; `git diff <old-tip> <new-tip>`;
     branch/tag SHAs read from the API);
  2. the fresh clone + import level — `pip install -e` a blob-filtered clone,
     `mace.__version__ == 0.3.16+openqha`, then `openqha.potentials.engine.mace_fork_info()`
     returning 40-hex / not dirty from the clone;
  3. the env level — the WSL `openqha` env's `mace` resolves to the `mace/` checkout
     afterwards.
- Prior art: the fork-gate runbook checks in the hessian-learning set (engine
  `mace_fork_info` against a real checkout, version before commit gate), and the
  classifier/PR hygiene checks already in `hpc/tools/` and the repo provisioning
  scripts.
- Acceptance (the ticket's own): bundle verifies; tree-identity diff empty; fork shows
  branch + tag at expected SHAs; scratch clone installs and imports as above; old-fork
  checkout archived.

## Out of Scope

- The install recipes, pins and transport story — the fork is only made consumable here
  (05; the Tianhe landing guide is theirs).
- Identity/sweeping changes that cite the fork — constants, refusal rules, old-hash
  mapping note, notebooks (06, 12).
- The upstream PR's preparation and its commit-set pruning (10). 2026-09-29 — the upstream PR was decided against ([ticket 10](../decisions/10-upstream-pr.md)): the fork is permanent-private ([ADR 0013](../../../docs/adr/0013-permanent-private-mace-fork.md)).
- The `openQHA-Hessian` content replacement and force-push (11).
- Any change of the mace base to a newer upstream release (map, out of scope).

## Further Notes

- The old fork on GitHub stays as-is until ticket 11 replaces its content; the bundle is
  its archive, made while the old refs still point at the old commits.
- Vocabulary: "the fork" is `BloomDlwlrma/mace`; `BloomDlwlrma/openQHA-Hessian` is always
  "the old fork". D never goes upstream; the six do (ticket 10 decides their final set). 2026-09-29 — the upstream PR was decided against ([ticket 10](../decisions/10-upstream-pr.md)): the fork is permanent-private ([ADR 0013](../../../docs/adr/0013-permanent-private-mace-fork.md)).
- Local workbench today: the refresh and bundle run in `openQHA-Hessian/` (it is still
  the mace checkout until the re-home), so nothing needs a second clone to rebuild.
- The `ready-for-agent` label is deliberately not applied: per the tracker doc and
  ADR 0009, wayfinder tickets use the `Status` protocol, not triage labels.

## Answer (2026-09-28, bookkeeping -- the slices carry the evidence)

The fork line landed complete: [02b](02b-the-rebuild.md) rebuilt the branch on upstream
history (tree-identity diff empty; old history bundled), [02c](02c-fork-and-push.md)
pushed both refs to the true fork `BloomDlwlrma/mace` (branch `openqha-hessian` @
`1110ffb`, tag `base-v0.3.16` @ `5c2d761`; default branch switched to the branch, per
02c's Postscript), and [02d](02d-clone-verify-rehome.md) demonstrated the acceptance --
a fresh blobless clone at `mace/` installs editable, reports `0.3.16+openqha` and a
clean `mace_fork_info()`; the old checkout is archived under `_to_delete/`. Every
acceptance item of this brief is evidenced in those three Answers; nothing remains open
under it (install/transport, identity and the PR are their own tickets). 2026-09-29 — the upstream PR was decided against ([ticket 10](../decisions/10-upstream-pr.md)): the fork is permanent-private ([ADR 0013](../../../docs/adr/0013-permanent-private-mace-fork.md)).
