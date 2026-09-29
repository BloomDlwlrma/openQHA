# 11c: The swap — the package on `main`, the old refs gone

Type: task
Status: open
Serves: 11
Blocked by: 11a, 11b
Part of: [hessian-learn-framework](../map.md)

> Execution slice of [The repo swap](../decisions/11-repo-swap.md) per
> [spec-repo-swap.md](../spec-repo-swap.md): decisions 1-4 and 10-11; stories 1, 9,
> 12-15, 19. HITL: the human's part is exactly the two web actions (default branch,
> description); the git steps run from the WSL checkout over SSH with the key's
> passphrase typed when the terminal prompts (no ssh-agent; two pushes). Wayfinder
> conventions apply: the `Status` protocol, no triage labels
> (`docs/agents/issue-tracker.md`, ADR 0009).

## What to build

The swap itself, in the approved sequence (decision 2):

0. Capture the before state: `git ls-remote --symref` (both old refs + `HEAD`).
1. Push local `main` — the six package commits plus 11a's README — as a **new** remote
   branch over SSH; fresh branch, no force-push, no squash (decision 1; the six ids are
   load-bearing).
2. **The human**: GitHub → Settings → Branches → default branch := `main`. After the
   push, before any deletion — GitHub refuses to delete a repository's default branch.
3. Delete the old refs in one push (`openqha-hessian` branch + `base-v0.3.16` tag);
   then `git fetch --prune` (clears `origin/openqha-hessian`), `git remote set-head
   origin -a` (`origin/HEAD` → `main`), and delete the local copy of the old tag (its
   value survives in the bundle) (decision 4).
4. **The human**: edit the description to the ruled sentence, verbatim —
   `openQHA Extension: openqha_hessian — the training side of active Hessian learning (PHL fine-tuning of MACE).`
5. Local bookkeeping (decision 10): `git fetch --prune origin` + `git branch -u
   origin/main main`; the HTTPS `origin` stays for fetch.

Evidence for the ticket (decision 11): the before/after `ls-remote --symref`, the
served description and default branch, and the push/delete/prune transcripts. If SSH
is not provisioned for this machine, the fallback is VS Code SCM's push with the human
at the keyboard.

## Acceptance

- [ ] The remote serves exactly one branch, `main`, at the pushed tip; `HEAD` → `main`; the old branch and the old tag are gone
- [ ] The pushed tip's README is 11a's rewrite (the first visitor sees the finished state)
- [ ] The web description is the ruled sentence verbatim; the default branch is `main`
- [ ] Local: the stale `origin/openqha-hessian` pruned, `origin/HEAD` → `main`, the local old tag deleted, `main` tracks `origin/main`
- [ ] Every item of decision 11's evidence list recorded on ticket 11
- [ ] Reported per the operating rule: files changed, checks run, anything not verified

## Answer

<!-- resolver: append the branch shape, the push/delete/prune summary, the after-state evidence; set Status: resolved; add a line to the map's Implementation -->
