# 02c: Fork and push — create the true fork and publish the refs

Type: task
Status: resolved
Serves: 02
Blocked by: 02b
Part of: [hessian-learn-framework](../map.md)

> Execution slice for [The real fork](../decisions/02-the-real-fork.md); the brief is
> [02a](02a-the-mace-fork.md). HITL: the fork is created and the refs pushed by the
> user (GitHub UI with dialog defaults; push from VS Code SCM); this slice prepares the
> remote and verifies read-only. Wayfinder conventions apply: the `Status` protocol, no
> triage labels.

## What to build

`BloomDlwlrma/mace` exists as a GitHub-side fork of `ACEsuit/mace` and carries, at the
rebuilt SHAs, the branch `openqha-hessian` and the tag `base-v0.3.16`. The fork's default
branch is exactly as GitHub created it — no settings changed at creation (switched later;
see Postscript). The old fork is untouched.
The push is an object diff against the shared upstream pool, so it stays small. These
are the refs ticket 05's install URL consumes and ticket 06 maps against.

## Acceptance

- [x] The fork exists; its parent is `ACEsuit/mace`; upstream branches and tags are inherited; no repository setting was changed (default branch as created; switched later — see Postscript).
- [x] The local checkout's `origin` points at the new fork over HTTPS; the old-fork URL no longer appears in the remotes.
- [x] Branch `openqha-hessian` and tag `base-v0.3.16` are live on the fork at 02b's tip / D; a read-back via the GitHub API shows exactly those two refs; nothing was force-pushed.
- [x] The push volume is small (≲1 MB) — the shared-object expectation held, unlike the old squashed-base episode.
- [x] The old fork `BloomDlwlrma/openQHA-Hessian` is untouched; ticket 11 owns its later content replacement.

## Answer

Resolved 2026-09-27. HITL: the user created the fork and pushed from VS Code SCM; the
agent prepared the remote and verified read-only via the GitHub API.

- **The fork**: `BloomDlwlrma/mace` is a true fork of `ACEsuit/mace` (id `1390537995`,
  created 2026-09-27T09:14:36Z; shared object pool — repo size 180,888 KiB equals
  upstream's). All upstream refs were inherited unchanged: 70 branches + 21 tags.
- **The push**: branch `openqha-hessian` = `1110ffbafd651a74d1d4678deb4748056d1dff0a` and
  lightweight tag `base-v0.3.16` = `5c2d7612eed88dc1463b5588a79c2d5f5718d322` (D) — pushed
  via "Publish Branch" (remote `origin`) then "Git: Push Tags". Volume: 63 objects /
  64,236 B (≈63 KiB) — the shared-object expectation held (≲1 MB).
- **Read-only verification**: both refs resolve at exactly those SHAs; the full ref set is
  upstream's 70 branches + 21 tags unchanged plus exactly these two new refs — nothing was
  force-pushed; events show the fork creation (09:14:36Z) and the branch creation
  (09:17:25Z).
- **Local checkout**: `openQHA-Hessian/`'s `origin` is the new fork over HTTPS; no
  old-fork URL remains among its remotes; the stale old-fork tracking
  (`origin/openqha-hessian` @ `f14a56f`) was cleared before the push.
- **Old fork**: `BloomDlwlrma/openQHA-Hessian` untouched (`openqha-hessian` @ `f14a56f`,
  `base-v0.3.16` @ `8fac5d1`); ticket 11 owns its later content replacement.

**Postscript (2026-09-27):** by explicit decision, the fork's default branch was switched
`develop` → `openqha-hessian` (ticket 02 decision 3 revised) so the landing page shows our
branch. No other setting, ref, or pinned URL changed. Read-only re-check: `default_branch`
= `openqha-hessian`; both refs still at the SHAs above.
