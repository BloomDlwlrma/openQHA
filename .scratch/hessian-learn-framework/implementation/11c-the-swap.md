# 11c: The swap — the package on `main`, the old refs gone

Type: task
Status: resolved
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
   *— amended 2026-09-29 per user ruling: **kept old** (the ruled sentence decided
   against; web action 2 dropped); see acceptance item 3 and the Answer.*
5. Local bookkeeping (decision 10): `git fetch --prune origin` + `git branch -u
   origin/main main`; the HTTPS `origin` stays for fetch.

Evidence for the ticket (decision 11): the before/after `ls-remote --symref`, the
served description and default branch, and the push/delete/prune transcripts. If SSH
is not provisioned for this machine, the fallback is VS Code SCM's push with the human
at the keyboard.

## Acceptance

- [x] The remote serves exactly one branch, `main`, at the pushed tip; `HEAD` → `main`; the old branch and the old tag are gone
- [x] The pushed tip's README is 11a's rewrite (the first visitor sees the finished state)
- [x] The web description is the ruled sentence verbatim — amended 2026-09-29 per user ruling: **kept old** (the ruled sentence decided against; web action 2 dropped); default branch = `main`
- [x] Local: the stale `origin/openqha-hessian` pruned, `origin/HEAD` → `main`, the local old tag deleted, `main` tracks `origin/main`
- [x] Every item of decision 11's evidence list recorded on ticket 11 — swap-side items recorded; the CI item rides 11d (as sliced)
- [x] Reported per the operating rule: files changed, checks run, anything not verified

## Answer (2026-09-29, agent-run; SSH pushes WSL-side, human at the terminal)

The swap is executed. The approved sequence ran as ruled: push → default-branch switch
(human) → one-push deletion → prune/set-head/bookkeeping. Branch shape: the package's
`main` (`a5f8103` = the six package commits `909b91f` … `ff92141` plus 11a's README)
published as a **new remote branch** — no force-push, no squash; the old refs were then
deleted in one push.

- **Push** (WSL SSH; `git@github.com:BloomDlwlrma/openQHA-Hessian.git` set as the push
  URL, HTTPS `origin` kept for fetch): `git push origin main` → `* [new branch] main ->
  main`. Served tip `a5f8103c7d644f42c892a495ecb13298f38bfe64`.
- **Default branch** (human web action): switched to `main`; verified via the repo API
  (`"default_branch": "main"`) before any deletion — the deletion ran only on that check.
- **Deletion** (one push): `git push origin --delete refs/heads/openqha-hessian
  refs/tags/base-v0.3.16` → `- [deleted] openqha-hessian`, `- [deleted] base-v0.3.16`.
- **After state**: `ls-remote --symref` = `ref: refs/heads/main HEAD` and only
  `a5f8103 refs/heads/main` (no tags — exactly one branch).
- **Local**: stale `origin/openqha-hessian` pruned, `origin/HEAD` → `main`, local tag
  `base-v0.3.16` deleted, `main` tracks `origin/main` (`## main...origin/main`, clean);
  end refs = `refs/heads/main`, `refs/remotes/origin/{HEAD,main}` all `a5f8103`.
- **Description — user ruling 2026-09-29: kept old.** The ruled replacement sentence was
  decided against; the served text stays as it was; web action 2 was dropped (spec
  execution note appended).
- **Push-mechanics finding, recorded honestly (replaces decision 10's as-written flow).**
  The "passphrase typed at the terminal prompt" flow hit GitHub's post-accept auth window
  twice: `Received disconnect … Bye Bye` whenever the passphrase entry was not instant
  (push attempt 1 and delete attempt 1; a fast push retry passed). The deletion ran
  through a session `ssh-agent` instead — one **local** `ssh-add` prompt (no server-side
  window), then agent-signed pushes; same key, same WSL-SSH route, same passphrase. The
  socket may still be alive at `/tmp/11c-agent.sock` (reuse for 11d's push:
  `export SSH_AUTH_SOCK=/tmp/11c-agent.sock`; kill when done: `pkill -f 11c-agent.sock`).
- **Checks**: package `tests/run_tests.py --all` on the pushed tip → all 9 passed (log
  `C:\Users\10704\AppData\Local\Temp\11c\11c-suite.log`); retired path gone (`Test-Path`
  false); before-state bundle re-verified (`bundle verify` okay; 2 refs; `list-heads` =
  `ls-remote`).
- **Transcripts** (under `C:\Users\10704\AppData\Local\Temp\11c\`): `before.txt`,
  `gh-before.json`, `11c-push2.log`, `11c-delete2-pre.txt`, `11c-delete2.log`,
  `final-check.txt`, `gh-final.json`.
- **Rides 11d**: the CI run item of decision 11's evidence list and ticket 11's record +
  resolution (decision 12), as sliced.

Reported per the operating rule — files changed: this ticket, [ticket 11](../decisions/11-repo-swap.md)'s
swap evidence, the map's Implementation line, the spec's execution notes, 11d's
ready-state note (`map.md` additionally carried other sessions' riding lines — see the
review record). Checks run: `ls-remote --symref` before/after, the repo API before the
deletion and after, `bundle verify` + `list-heads`, `git status` / `branch -vv` /
`for-each-ref`, the retired-path check, the package suite (9/9). Not verified: the CI
install path on a clean machine (11d's job); nothing else in this slice's list.

## Review record (2026-09-29, two-axis review of `088adfe`)

Two-axis review of the resolution commit (fixed point `af23691`) per the `code-review`
skill; the axes ran as parallel sub-agents. No hard violations; two findings were fixed
in this annotation commit, the rest kept or reported (gist of both reports):

**Standards.** Resolve protocol conformant — `## Answer`, `Status: resolved`, the map's
Implementation line, no triage labels; "Reported per the operating rule" complete on
this ticket and on ticket 11.

- Ticket 11's Answer is written ahead of its resolution (the resolve sequence completes
  in 11d) — kept: the Answer's title and its `<!-- 11d: ... -->` footer make the split
  explicit.
- This ticket's acceptance items 3 and 5 were found rewritten in place — fixed in this
  annotation: originals restored with dated amendment notes (the spec's append-only
  execution notes are the model; `AGENTS.md`'s "done only when its acceptance criteria
  are paid" anchors the letter of the list).
- Smell baseline: the push-mechanics/agent-socket instructions repeat across ticket 11,
  this ticket, the spec and 11d, and the SHA/ref set travels with them — kept as
  judgement calls (same precedent as 11b's review record: docs-variant duplication is
  the house style, and each site is a doc variant, not code).
- Reported, not fixed: `map.md`'s "Out of scope" still reads "The upstream review cycle
  after the PR is opened" while the riding ticket-10 line reads "not submitted ...
  never upstream" — ticket 10's session owns those lines.

**Spec.** Faithful to the swap's decisions; two micro-gaps, both closed in this
annotation: the bundle path is now named on ticket 11's Answer (decision 12 asks for
branch shape + bundle path + CI), and the README acceptance item is covered by tip
identity (`a5f8103` = the README commit — same SHA, same content).

- Scope note accepted as the tracked convention: `map.md` carried other sessions'
  riding lines (ticket 10's record, the 08a/11a lines, the fog replacement) into this
  commit — per 11a's own Answer, "the 11a line rides the next tracker commit". The
  commit message covers this slice; the riding lines' records live in their own tickets.
- `final-check.txt`'s `description_exact: False` compares against the superseded
  sentence — per the user ruling the old text is the intended state (not a failed
  check).
- Deliberately out (sliced): the CI item and ticket 11's resolution ride 11d.
