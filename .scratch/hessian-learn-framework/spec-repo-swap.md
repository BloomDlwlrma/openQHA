# Spec: The repo swap — the package on `main`, the old refs gone, a clean tree

Label: `ready-for-agent`. Tracker: `.scratch/hessian-learn-framework/`. Spec for
[11-repo-swap](decisions/11-repo-swap.md), rulings of 2026-09-29; durable records: the local
bundle `_backup/openQHA-Hessian-old-history-2026-09-26.bundle` (workspace root) and the
old→new mapping in [ADR 0011](../../docs/adr/0011-mace-fork-rebuild.md).

**Rulings taken with this spec.** Q1: the content replacement is a **fresh branch** — `main`
pushed as a new ref; no force-push, no squash; the old branch and tag are deleted only after
the default branch has moved. Q2: the six package commits publish as they stand (`909b91f` …
`ff92141`), so the ids the trackers and the mapping cite stay valid. Q3: the git steps run
from the WSL checkout over SSH (the human types the key's passphrase when the terminal
prompts); the human's part is exactly two web actions — switch the default branch, set the
description — interleaved as push → switch → delete → prune. Q4: the retired old-fork checkout
is deleted whole, its junk included; the bundle remains the only old-history home. Q5: the
README is rewritten clean in the register of hip and the PHL source repository — index links
to openQHA and to the mace fork, a citation section (PHL paper + openQHA software), no History
section — and the GitHub description is set to the ruled sentence verbatim. Q6: a **minimal
CI** lands as the ticket's last, separable step — install through `install.sh` and run the two
tests that need nothing from the private openQHA checkout; a green run, not a committed file,
is the acceptance. Q7: execution follows the sequence in decision 2 and records the evidence
of decision 11 on the ticket.

Settled elsewhere, not this spec's work: the fork rebuild ([02](decisions/02-the-real-fork.md));
the package line and the move ([01](decisions/01-the-package-line.md),
[07](decisions/07-the-move.md)); the identity after the split
([06](decisions/06-identity-after-the-split.md)); the weight-identity strip
([04](decisions/04-sha256-retirement.md)); the install and transport story
([05](decisions/05-install-and-transport.md)); publication ([13](decisions/13-publication.md));
the references sweep ([12](decisions/12-references-sweep.md)).

## Problem Statement

`BloomDlwlrma/openQHA-Hessian` does not yet say what it is. Its default branch
(`openqha-hessian`, tip `f14a56f`) and its tag (`base-v0.3.16`, `8fac5d1`) still serve the old
fork's content, and its description still advertises a pre-split "framework instance … MACE".
The package the split produced — six commits, tip `ff92141`, the tree whose `install.sh` and
tests the 05 family verified — has never been pushed; a visitor to the repository finds the
old experiment instead. The retired old-fork checkout still sits in the local tree under
`_to_delete/` with its logs, results, egg-infos, pytest cache and finetuning xyz — the "second
mace checkout" the ticket wants gone. And nothing yet exercises the public install path on a
clean machine.

## Solution

Publish the package's history on a fresh `main`, switch the default branch to it, then delete
the old branch and tag — the old history survives only in the local bundle. Rewrite the README
clean, in the register of hip and the PHL source repository: what this package is, index links
to openQHA and the mace fork, install and verify instructions, and a citation section (the PHL
paper and the openQHA software). Set the GitHub description to the ruled sentence. Delete the
retired checkout whole. Land a minimal CI that installs through `install.sh` and runs the two
openQHA-free tests. Record the swap's evidence on the ticket.

## User Stories

1. As a visitor to `BloomDlwlrma/openQHA-Hessian`, I want the default branch to be the package's `main`, so that the code I land on is what the description describes.
2. As a visitor, I want the README to say in one paragraph what this package is, so that I understand it is the training side of openQHA — not a MACE fork.
3. As a visitor, I want index links to openQHA and to the mace fork, so that I can find the two things this package needs.
4. As a visitor, I want the repository tree to contain only the package (no MACE-native files), so that the content matches the claim "the package, not a fork".
5. As a user setting up a machine, I want the install instructions (both `install.sh` modes) preserved in the rewrite, so that the verified path from the 05 family stays the documented one.
6. As a user, I want the verify instructions (the runner, `--all`, the provenance header) preserved, so that a wrong stack is visible in seconds.
7. As a citing author, I want a citation section with the PHL paper and the openQHA software entries, so that I can credit what the numbers used.
8. As the maintainer, I want no History section and no old-hash talk in the README, so that the repository reads as a clean standalone package.
9. As the maintainer, I want the old branch and the old tag deleted from the remote, so that no pre-split content is served under this repository's name.
10. As the maintainer, I want the old history to survive as the local bundle only, so that nothing of the old experiment is lost and nothing of it stays public.
11. As a tracker reader, I want the bundle path and the old→new mapping (ADR 0011) to stay the pointers, so that old Records remain traceable.
12. As the operator, I want the swap done as a fresh branch (no force-push, no squash), so that the published ids are the ones the trackers and the mapping cite.
13. As the operator, I want the push to already carry the final README, so that the first visitor sees the finished state.
14. As the human, I want my part to be exactly two web actions (default branch, description), so that the swap does not depend on me driving git.
15. As the agent, I want the sequence fixed (push → switch → delete → prune), so that I never attempt to delete the default branch or race the default switch.
16. As the workspace owner, I want the retired old-fork checkout deleted whole, so that the local tree stops looking like a second mace checkout.
17. As the workspace owner, I want the bundle verified against the remote before anything is deleted, so that the deletion is provably safe for the tracked history.
18. As the maintainer, I want a minimal CI that installs through `install.sh` and runs the openQHA-free tests, so that the public install path is exercised on a clean machine.
19. As a future session, I want the acceptance evidence on the ticket (ls-remote before/after, the served description, the bundle check, the CI run), so that the swap is checkable without trust.
20. As a reviewer, I want old Records left untouched with their old hashes mapped, so that nothing "repairs" immutable history.
21. As the effort, I want one line in the map's Decisions so far when the ticket resolves, so that the trail shows the swap.

## Implementation Decisions

1. **Publication shape.** A fresh branch, not a force-push: push the local `main` as a new
   remote branch; move the GitHub default branch to it; delete the old branch
   (`openqha-hessian`) and the old tag (`base-v0.3.16`). No history rewrite, no squash — the
   six commit ids are load-bearing. Preconditions for the deletion: `main` exists on the
   remote and the default branch already points at it.

2. **Order of operations** (the approved sequence): commit the README rewrite → verify the
   bundle → push `main` → the human switches the default branch → delete the old refs, prune,
   set-head → the human sets the description → CI commit + push + green run → record the
   evidence and resolve.

3. **The two web actions (human).** (a) GitHub → Settings → Branches → default branch: `main`
   — after the push, before any deletion (GitHub refuses to delete a repository's default
   branch). (b) Edit description →
   `openQHA Extension: openqha_hessian — the training side of active Hessian learning (PHL fine-tuning of MACE).`
   verbatim (the machine has no `gh`; the web UI is the tool).

4. **The deletions (agent).** One push removes both old refs (the `openqha-hessian` branch and
   the `base-v0.3.16` tag); then `git fetch --prune` clears the stale remote-tracking ref
   (`origin/openqha-hessian`), `git remote set-head origin -a` points `origin/HEAD` at `main`,
   and the local copy of the old tag is deleted (its value survives in the bundle).

5. **Old-history check before any deletion.** `git bundle verify` the bundle and compare
   `git bundle list-heads` with `git ls-remote`: both must agree (`openqha-hessian` =
   `f14a56f`, `base-v0.3.16` = `8fac5d1` — checked equal on 2026-09-29). If the remote holds
   anything the bundle lacks, re-bundle first. The bundle stays at the workspace-root
   `_backup/`; no copy of it lands in the package repo.

6. **README rewrite.** Clean register (hip / PHL source): title; one short paragraph of what
   the package is; the index links — `openQHA` = `https://github.com/BloomDlwlrma/openQHA`,
   `mace` = `https://github.com/BloomDlwlrma/mace` (branch `openqha-hessian`); an Install
   section (both `install.sh` modes; the run-it-last and eval-wheel notes survive, tightened);
   a Verify section (the runner, `--all`, the provenance header); a Citation section — the PHL
   paper (`rodriguez2026projectedhessianlearningfast`, arXiv 2603.04523) and the openQHA
   software entry (`@misc{openqha}`, per openQHA's `docs/cite/`). No History section, no old
   hashes or mapping talk — that story lives in the tracker and ADR 0011. No badge; the
   register stays clean.

7. **Wording rule.** Externally, the predecessor is "an early MACE fork experiment"; "the old
   fork" stays internal tracker vocabulary. The README itself does not discuss the
   predecessor at all.

8. **The retired checkout.** Delete `_to_delete/openQHA-Hessian-old-fork-2026-09-26/` whole —
   the old tree and its junk (`logs/`, `results/`, `mace_torch.egg-info/`, `.pytest_cache/`,
   `mp_finetuning*.xyz`). Run the deletion Windows-side (e.g. `Remove-Item -Recurse -Force`);
   WSL-side `rm`/`mv` on `/mnt/c` hits EACCES while VS Code holds handles in the subtree. The
   tracked history is recoverable from the bundle; the untracked artifacts are not, and are
   accepted as disposable.

9. **CI (the separable last step).** A workflow at `.github/workflows/ci.yml`, on push/PR to
   `main`: ubuntu-latest, Python 3.11; `bash install.sh` (URL mode — the fork from its branch
   URL, the package editable; the script's own verification block ends green without openQHA
   by design, its openQHA-dependent line skipped); then `python tests/unit/t_hvp.py` and
   `python tests/unit/t_phl_loss.py` — the only two tests that need nothing from the private
   openQHA checkout. The full suite stays local/Tianhe. Committed and pushed on its own,
   after the swap; its first run must be green before resolution.

10. **Push mechanics and local bookkeeping.** Pushes run from WSL over SSH
    (`git@github.com:BloomDlwlrma/openQHA-Hessian.git` as the push URL; the key's passphrase
    is typed when the terminal prompts — no ssh-agent on the machine; two SSH pushes total:
    the new branch, then the deletions). The HTTPS `origin` URL stays for fetch and later
    VS Code use; after the swap, `git fetch --prune origin` + `git branch -u origin/main main`
    restore tracking.

11. **Evidence (what the ticket records).** Before: `ls-remote --symref` (the two old refs +
    HEAD) and the `bundle verify` / `list-heads` output. After: `ls-remote --symref` (only
    `refs/heads/main`, `HEAD` → `main`, the new tip); the served description and default
    branch (repo API/page); the CI run URL and its green conclusion; `git status` clean; the
    retired path gone. Plus the command transcripts of push / delete / prune.

12. **Records.** The ticket's Answer (branch shape, bundle path, CI) + `Status: resolved` +
    one line in the map's Decisions so far; ADR 0011 and the bundle stay the old-history and
    mapping home; old Records keep their old hashes, untouched.

13. **What must not change.** The package tree other than the README and the CI workflow
    (`install.sh`, `pyproject.toml`, `openqha_hessian/`, `tests/`, `.gitignore`); openQHA's
    code (its trackers aside); the mace fork; the openQHA-Hessian name and URL.

## Testing Decisions

- **What makes a good check here:** externally observable state, through pre-existing seams —
  the remote as `ls-remote` sees it, the repository metadata as the web serves it, the
  bundle's own `verify`, and the CI run. Nothing asserts anything about the local machine.
- **The one new seam:** CI — the public install path exercised on a clean machine, the same
  shape as [05e](implementation/05e-fresh-env-acceptance.md)'s fresh-environment run, at the
  only place a stranger would run it. Its assertions: `install.sh` exits 0 (its verification
  block includes the fork identity) and the two openQHA-free tests pass.
- **Seams reused:** `git ls-remote` for "before" (`f14a56f` + `8fac5d1`) and "after" (only
  `refs/heads/main`); `git bundle verify` for the old history; the repo metadata for the
  description and default branch.
- **Prior art:** 05e's acceptance (hand-run there, runner-run in CI); mace-md's workflow (the
  precedent the ticket cites — install the stack, smoke the tool).
- **Deliberately not done:** no full-suite CI (openQHA is private; a public runner cannot have
  the checkout or the weights); no new package tests (no package code changes); no re-run of
  the Tianhe path.
- **Acceptance:** decision 11's evidence, recorded on the ticket; a committed CI workflow is
  not an accepted one — the green run is.

## Out of Scope

- Model publication and releases (ticket [13](decisions/13-publication.md)); the model
  repository is a separate, dedicated repo.
- The references sweep ([12](decisions/12-references-sweep.md)): stale mentions of the old
  repository elsewhere. The fact pass (2026-09-29) found one in
  `openQHA/workflows/hessian_learning/README.md`, which names `BloomDlwlrma/openQHA-Hessian`
  as the mace fork — sweep territory, not fixed here.
- Squashing, force-pushing, or rewriting any history; re-doing install/transport (05); the
  upstream PR cycle and the mace fork's content; license selection for the package; the map's
  fog item about old recorded paths (it sharpens here but stays its own future ticket). 2026-09-29 — the upstream PR was decided against ([ticket 10](decisions/10-upstream-pr.md)): the fork is permanent-private ([ADR 0013](../../docs/adr/0013-permanent-private-mace-fork.md)).
- Any change to old Records, fixtures, or the Tianhe path.

## Further Notes

- **Facts the implementation leans on** (checked 2026-09-29): the remote holds exactly one
  branch (`openqha-hessian` @ `f14a56f`) and one tag (`base-v0.3.16` @ `8fac5d1`) — no PRs,
  no forks, no issues; the bundle's heads equal both; the local tip is `ff92141` (six
  commits; `origin` = HTTPS, pushes over SSH); the two tests runnable without openQHA are
  `t_hvp` and `t_phl_loss` (the runner has no per-test selector, so CI invokes the files
  directly); `install.sh`'s verification skips its openQHA-dependent line when openQHA is not
  importable (green by design on a bare runner); the retired checkout's junk inventory is
  `logs/` (10 files), `results/` (5), `mace_torch.egg-info/`, `.pytest_cache/`, and four
  `mp_finetuning*.xyz`.
- **The human's two actions, precisely:** (1) Settings → Branches → default `main`;
  (2) Edit description → the ruled sentence. Everything else in the sequence is the agent's.
- **Register references:** hip's README and the PHL source repository's README (extracted at
  `source-code/final-workflow-design/hessian-train/PHL-main/`) — title, one-liner, index
  links, install, citation with bibtex; neither carries a History section or badges.
- **openQHA is not currently public** — the index link is included as ruled; nothing else in
  the README depends on it being reachable.
- **CI is separable:** dropping it means deleting decision 9 and the workflow file; the swap
  itself does not depend on it.
- **Workflow note:** the README draft is shown to the human in the execution session before it
  is committed (it is public text); the description sentence is fixed by this spec.
- **Unverified:** the WSL SSH push end-to-end (02c's precedent was VS Code SCM); if SSH is not
  provisioned for this machine, the fallback is VS Code SCM's push with the human at the
  keyboard.
- **Done means:** the remote serves only `main` (rewritten README, new description), the
  retired checkout is gone, the bundle verifies, and a green CI run exists — all recorded on
  the ticket.

## Execution notes (2026-09-29, after 11c — append-only; the sections above stand as the spec's letter)

- **Decision 10, corrected in practice.** GitHub's SSH endpoint closes the connection with
  `Bye Bye` when the signature doesn't arrive promptly after the key offer is accepted; the
  as-written "passphrase typed at the terminal prompt" flow failed twice for anything but
  instant typing (push attempt 1, delete attempt 1) and passed on a fast retry (push
  attempt 2). Both pushes still ran over the ruled WSL-SSH route with the same key; the
  deletion used a session `ssh-agent` — one **local** `ssh-add` prompt, no server-side
  window. If still alive: socket `/tmp/11c-agent.sock` (reuse: `export
  SSH_AUTH_SOCK=/tmp/11c-agent.sock`; kill: `pkill -f 11c-agent.sock`).
- **Decision 3(b), superseded by the user (2026-09-29).** The description keeps its
  pre-existing text — the ruled replacement sentence was decided against; the description
  web action was dropped. The default-branch switch (action 1) stands.
- **UI pointer corrected:** the default-branch switch lives in Settings → **General**
  ("Default branch"), not Settings → Branches, in the current GitHub UI.
- **Push end-to-end verified:** the WSL key `id_ed25519` authenticates to GitHub (over
  `ssh.github.com:443`, per the existing WSL `~/.ssh/config`).
- **Status:** the swap is executed and verified (11c resolved, 2026-09-29); the CI and the
  ticket-11 record + resolution ride [11d](../implementation/11d-the-minimal-ci.md).

- **11d green (2026-09-29):** the minimal CI landed (`.github/workflows/ci.yml` on `main`, `ccc5a38`) and its first run is green — https://github.com/BloomDlwlrma/openQHA-Hessian/actions/runs/36564158729; ticket 11's record completed and resolved with it. Nothing in this spec remains open.
