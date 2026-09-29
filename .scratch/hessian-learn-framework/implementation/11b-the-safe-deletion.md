# 11b: The safe deletion — the bundle verified, the old checkout gone

Type: task
Status: open
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

- [ ] `git bundle verify` passes; `list-heads` equals `ls-remote` on both old refs — or the mismatch stopped the deletion and a re-bundle was reported
- [ ] `_to_delete/openQHA-Hessian-old-fork-2026-09-26/` is gone (`Test-Path` false); the workspace-root bundle is untouched
- [ ] The old-history pointers stand: the bundle path and the ADR 0011 mapping report these as the old-history home
- [ ] Reported per the operating rule: files changed, checks run, anything not verified

## Answer

<!-- resolver: append the bundle check output + the deletion evidence; set Status: resolved; add a line to the map's Implementation -->
