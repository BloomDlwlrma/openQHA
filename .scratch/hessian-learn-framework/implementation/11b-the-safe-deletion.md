# 11b: The safe deletion — the bundle verified, the old checkout gone

Type: task
Status: resolved
Serves: 11
Blocked by: None
Part of: [hessian-learn-framework](../map.md)

> Execution slice of [The repo swap](../decisions/11-repo-swap.md) per
> [spec-repo-swap.md](../spec-repo-swap.md): decisions 5 and 8; stories 10, 16, 17.
> Independent of 11a — either can start now. Wayfinder conventions apply: the `Status`
> protocol, no triage labels (`docs/agents/issue-tracker.md`, ADR 0009).

## What to build

The old checkout deleted whole, on a proven bundle — the only old-history home:

1. The check (decision 5): `git bundle verify` the workspace-root
   `_backup/openQHA-Hessian-old-history-2026-09-26.bundle`; compare
   `git bundle list-heads` with `git ls-remote` on `BloomDlwlrma/openQHA-Hessian` —
   both must agree (`openqha-hessian` = `f14a56f`, `base-v0.3.16` = `8fac5d1`; checked
   equal 2026-09-29). If the remote holds anything the bundle lacks, re-bundle first
   and report — nothing is deleted on a mismatch.
2. The deletion (decision 8): remove `_to_delete/openQHA-Hessian-old-fork-2026-09-26/`
   whole — old tree, `logs/`, `results/`, `mace_torch.egg-info/`, `.pytest_cache/`,
   the four `mp_finetuning*.xyz`. Windows-side (`Remove-Item -Recurse -Force`): WSL
   `rm`/`mv` on `/mnt/c` hits EACCES while VS Code holds handles in the subtree; if
   Windows holds handles, close the editors touching it and retry. The tracked history
   stays recoverable from the bundle; the untracked artifacts are accepted as
   disposable.

The bundle stays at the workspace root — no copy of it lands in the package repo.

## Acceptance

- [x] `git bundle verify` passes; `list-heads` equals `ls-remote` on both old refs — or the mismatch stopped the deletion and a re-bundle was reported
- [x] `_to_delete/openQHA-Hessian-old-fork-2026-09-26/` is gone (`Test-Path` false); the workspace-root bundle is untouched
- [x] The old-history pointers stand: the bundle path and the ADR 0011 mapping report these as the old-history home
- [x] Reported per the operating rule: files changed, checks run, anything not verified (see the Answer)

## Answer (2026-09-29, implemented in this commit)

**Bundle check (before any deletion).** `git bundle verify` on
`_backup/openQHA-Hessian-old-history-2026-09-26.bundle`: okay — 2 refs, a complete history:

- `f14a56fe40330766219cb83f9ebbf2d813b99e05` `refs/heads/openqha-hessian`
- `8fac5d11bb34954e17ed7a41e7a4bb6f908017be` `refs/tags/base-v0.3.16`

`git bundle list-heads` lists exactly those two; `git ls-remote
https://github.com/BloomDlwlrma/openQHA-Hessian.git` lists the same two SHAs plus
`HEAD` = `f14a56f` (`--symref`: `HEAD` → `refs/heads/openqha-hessian`). Equal on both old
refs — the remote holds nothing the bundle lacks, so the deletion proceeded. Bundle
sha256 `729ad1bb8b0ddf3b94435d0838f92ac0966a5dbbf79a9abfd457e6fb5c73a3e7` (522,591 B),
re-checked after the deletion: unchanged, still at the workspace root.

**Deletion (Windows side, per decision 8).** `Remove-Item -Recurse -Force` on
`_to_delete/openQHA-Hessian-old-fork-2026-09-26` — gone (`Test-Path` false); 395 files
removed, including `logs/`, `results/`, `mace_torch.egg-info/`, `.pytest_cache/` and the
four `mp_finetuning*.xyz`; the rest of `_to_delete/` untouched. A repo-wide grep finds the
path only in tracker records (history, by design). The WSL `openqha` env still imports
`mace` from the editable `mace/` clone (`0.3.16+openqha`) and `openqha_hessian` from
`openQHA-Hessian/`.

**Pointers stand.** ADR 0011 still names the bundle
(`../../../_backup/openQHA-Hessian-old-history-2026-09-26.bundle`) as the old-history
home; no copy of the bundle was placed in the package repo.

Reported per the operating rule — files changed: this ticket and the map's Implementation
line; checks run: `git bundle verify`, `bundle list-heads` vs `ls-remote`, `Test-Path`,
bundle sha256, old-path grep, env imports; not verified: the Python test suites were not
re-run — no code was touched and no runtime consumer of the deleted path exists.

Transcripts: `C:\Users\10704\AppData\Local\Temp\11b-check.log` (bundle verify, list-heads,
ls-remote), `11b-post.log` (post-deletion state, sha256 re-check), `11b-probe2.log` (env
imports).

## Review record (2026-09-29, two-axis review of `7a59882`)

Two-axis review of the resolution commit (parent `24fc576`) per the `code-review` skill;
both axes clean.

**Standards (repo standards + the smell baseline).**

- Conformant; no hard violations. The resolve protocol is followed (evidence `## Answer`,
  `Status: resolved`, one `Implementation` line, no triage labels), and every evidence
  claim in the Answer re-checks against the session logs (verify output, refs equality,
  sha256 unchanged, 395 files, the rest of `_to_delete/` untouched).
- The `Status: claimed` step is a working state, recorded in the session log only -> kept
  as is.
- Baseline smells (Data Clumps on the ref/SHA pair; docs-variant duplication between
  "What to build" and the Answer) -> judgement calls, kept: tracker gisting is the
  convention here.

**Spec (this ticket).**

- Faithful; all four acceptance items are backed by evidence. Exactly the two tracker
  files changed; nothing outside the contract.
- The Answer quotes the check outputs rather than appending raw transcripts -> the
  transcript paths added above (this annotation).
- Beyond-contract checks (env import probe, old-path grep) are reported-only and stay.
