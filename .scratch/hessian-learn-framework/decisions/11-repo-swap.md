# The repo swap: replace openQHA-Hessian's content and clean the local tree

Type: task
Status: resolved
Blocked by: 05, 07
Part of: [hessian-learn-framework](../map.md)

## Question / work

Q5 execution: `BloomDlwlrma/openQHA-Hessian` keeps its name and URL and gets new content — the package, the installer, the README that says what this extension is and where mace lives — with no MACE-native files in it.

Work:

1. Land the new tree (from [The move](07-the-move.md) / [Install and transport](05-install-and-transport.md)); rewrite the README (extension ≠ mace; install; link to the fork; the old-history pointers); keep the GitHub description honest (today: "openQHA Extension: Active Hessian Learning Framework Instance for MLIPs - MACE").
2. Replace the GitHub default branch content (force-push or a fresh branch — state which); save the old history as a **local bundle** only; the old remote branch goes away.
3. Clean the working tree: untracked `logs/`, `results/`, `mace_torch.egg-info/`, `.pytest_cache/`, `mp_finetuning*.xyz` — removed or explicitly kept; the point is that the local tree stops looking like a second mace checkout.
4. Decide: a minimal CI workflow for the repo (install the fork + run the package's tests), since mace-md has one — if yes, it lands here.

## Answer (2026-09-29: swap executed via 11c; CI + resolution via 11d)

Decision 2's sequence ran as approved; swap-side evidence (transcripts under
`C:\Users\10704\AppData\Local\Temp\11c\`):

- **Before** (2026-09-29): `ls-remote --symref` = `ref: refs/heads/openqha-hessian HEAD`,
  `f14a56fe40330766219cb83f9ebbf2d813b99e05` HEAD + `refs/heads/openqha-hessian`,
  `8fac5d11bb34954e17ed7a41e7a4bb6f908017be refs/tags/base-v0.3.16`; the workspace-root
  bundle `_backup/openQHA-Hessian-old-history-2026-09-26.bundle` re-verified —
  `bundle verify` okay (2 refs, complete history), `list-heads` = `ls-remote`, sha256
  `729ad1bb8b0ddf3b94435d0838f92ac0966a5dbbf79a9abfd457e6fb5c73a3e7` (522,591 B).
  Transcripts: `before.txt`, `gh-before.json`.
- **Publish**: fresh branch `main` → `a5f8103c7d644f42c892a495ecb13298f38bfe64` (the six
  package commits `909b91f` … `ff92141` + 11a's README; no force-push, no squash — ids as
  ruled). Transcript: `11c-push2.log` (`* [new branch] main -> main`).
- **Default branch**: switched by the human; verified `"default_branch": "main"` via the
  repo API **before** the deletion (the deletion script's own precondition gate).
- **Delete** (one push): `git push origin --delete refs/heads/openqha-hessian
  refs/tags/base-v0.3.16` → `- [deleted] openqha-hessian`, `- [deleted] base-v0.3.16`.
  Transcript: `11c-delete2.log`.
- **After**: `ls-remote --symref` = `ref: refs/heads/main HEAD` and only
  `a5f8103 refs/heads/main` (no tags); `git status` clean; the retired path gone; local
  bookkeeping done (stale tracking ref pruned, `origin/HEAD` → `main`, local old tag
  deleted, `main` tracks `origin/main`). Transcript: `final-check.txt`.
- **Description — user ruling 2026-09-29: KEPT OLD.** The ruled replacement sentence was
  decided against; web action 2 was dropped; the served description remains "openQHA
  Extension: Active Hessian Learning Framework Instance for Machine-Learned Interatomic
  Potentials (MLIPs) - MACE".
- **Push-mechanics finding** (recorded on [11c](../implementation/11c-the-swap.md) and in
  the spec's execution notes): GitHub's SSH endpoint closed the connection with
  `Received disconnect … Bye Bye` twice when the passphrase was not typed instantly
  (push attempt 1, delete attempt 1; a fast push retry passed). Both pushes ran over the
  ruled WSL-SSH route with the same key; the deletion used a session `ssh-agent` (one
  local `ssh-add` prompt — no server-side window). Socket may still be alive at
  `/tmp/11c-agent.sock` for 11d's push.

- **CI** ([11d](../implementation/11d-the-minimal-ci.md)): `.github/workflows/ci.yml` committed
  on the package `main` (`ccc5a38`) and pushed; its first run is green — https://github.com/BloomDlwlrma/openQHA-Hessian/actions/runs/36564158729
  (conclusion `success`; ubuntu-latest, Python 3.11; `bash install.sh` URL mode, then
  `t_hvp` + `t_phl_loss`). With it the evidence list of decision 11 is complete;
  closing re-checks 2026-09-29: the package checkout's `git status` clean, the
  retired path gone, the package suite 9/9 on the pushed tip.

Reported per the operating rule — **files changed**: the tracker (`spec-repo-swap.md`,
`implementation/11a-…`–`11d-…`, this ticket, `map.md`'s ticket-11 lines) and the package
repository (`README.md`, `.github/workflows/ci.yml`); the retired
`_to_delete/openQHA-Hessian-old-fork-2026-09-26/` deleted. **Checks run**: `bundle
verify` + `list-heads` = `ls-remote` (11b); `ls-remote --symref` before/after and the
repo API for the default branch and description (11c); the package suite 9/9 on the
pushed tip; the retired-path and `git status` re-checks; the CI workflow's first run
green (11d). **Not verified**: nothing outstanding.

## Review record (2026-09-29, ticket-level two-axis review)

Two-axis review of the closed ticket per the `code-review` skill — openQHA
`dd18a84...3334955` (spec, slices, this ticket, map lines) and package `ff92141...ccc5a38`
(README, CI); both axes ran as parallel sub-agents, with a lead-verification pass.

- **Standards** — one hard finding: this Answer's missing "files changed / checks run /
  not verified" report (discharged above). Judgement calls, all fixed or noted: the
  README's "projected Hessian loss" against `CONTEXT.md`'s `_Avoid_`, and a duplicated
  Citation lead-in — both fixed in the package repo (`7d85d27`, pushed);
  [11c](../implementation/11c-the-swap.md)'s step 4 and the spec's decision 6 lacking dated
  amendment notes (the annotations commit).
- **Spec** — faithful: decision 11's evidence list complete; the swap sequence (1/2/4/10),
  the bundle check (5), the CI acceptance (9) and decision 13 verified against the
  records; slice blockers match the approved breakdown; no scope creep beyond the
  recorded user rulings.
- Per-slice review records ([11a](../implementation/11a-the-readme-rewrite.md) in-file;
  [11b](../implementation/11b-the-safe-deletion.md), [11c](../implementation/11c-the-swap.md),
  [11d](../implementation/11d-the-minimal-ci.md) annotation commits) stand; nothing
  re-litigated here.
