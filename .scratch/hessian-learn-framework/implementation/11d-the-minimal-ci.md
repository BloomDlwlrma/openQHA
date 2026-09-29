# 11d: The minimal CI — the install path on a clean machine, then the record

Type: task
Status: resolved
Serves: 11
Blocked by: 11c
Part of: [hessian-learn-framework](../map.md)

> Execution slice of [The repo swap](../decisions/11-repo-swap.md) per
> [spec-repo-swap.md](../spec-repo-swap.md): decision 9 (the CI) and decisions 11-12
> (the evidence and the records); stories 18-19, 21. The CI is the ruled separable
> last step — dropping it means deleting decision 9 and the workflow; the record step
> stays. Wayfinder conventions apply: the `Status` protocol, no triage labels
> (`docs/agents/issue-tracker.md`, ADR 0009).
>
> **Ready state (2026-09-29, after [11c](11c-the-swap.md)):** the swap is executed and
> verified — the remote serves only `main` @ `a5f8103`; the description stays the old
> text (user ruling; no description action); local tracking is clean. The WSL session
> ssh-agent from 11c may still be alive with the key loaded — reuse it for this slice's
> push (`export SSH_AUTH_SOCK=/tmp/11c-agent.sock`; fallback `ssh-add
> ~/.ssh/id_ed25519`); kill when done: `pkill -f 11c-agent.sock`. Swap transcripts under
> `C:\Users\10704\AppData\Local\Temp\11c\`.

## What to build

1. `.github/workflows/ci.yml`: on push/PR to `main` — ubuntu-latest, Python 3.11;
   `bash install.sh` (URL mode; its verification block is green without openQHA by
   design, the openQHA-dependent line skipped), then `python tests/unit/t_hvp.py` and
   `python tests/unit/t_phl_loss.py` — the only two tests that need nothing from the
   private openQHA checkout. Committed and pushed on its own, after the swap. **A
   committed workflow is not the acceptance: the first run must be green.**
2. The record (decisions 11-12): ticket 11's `## Answer` with the whole evidence list
   — branch shape, bundle path, the served description, the CI run URL, `git status`
   clean, the retired path gone — plus `Status: resolved` and one line in the map's
   Decisions so far. ADR 0011 and the bundle stay the old-history home; old Records
   keep their old hashes, untouched.

## Acceptance

- [x] The workflow is committed and pushed; its first run on GitHub is green; the run URL is recorded
- [x] The workflow installs through `install.sh` and runs exactly the two openQHA-free tests — no full-suite or openQHA-dependent step
- [x] Ticket 11's Answer carries every item of decision 11's evidence list; `Status: resolved`; one line in the map's Decisions so far
- [x] The package tree changed only by the README (11a) and this workflow (decision 13); old Records and the ADR 0011 mapping untouched
- [x] Reported per the operating rule: files changed, checks run, anything not verified

## Answer (2026-09-29, agent-run)

The minimal CI landed on the package repo's `main` and its first run is green.

- **The workflow**: `.github/workflows/ci.yml` (`ccc5a38`) — on push/PR to `main`:
  ubuntu-latest, Python 3.11; `bash install.sh` (URL mode — the fork from its branch
  URL, the package editable), then `python tests/unit/t_hvp.py` and
  `python tests/unit/t_phl_loss.py` — exactly the two tests that need nothing from the
  private openQHA checkout; no full-suite or openQHA-dependent step (decision 9; the
  install step ends with `install.sh`'s own verification block green, its openQHA line
  skipped by design).
- **First run**: https://github.com/BloomDlwlrma/openQHA-Hessian/actions/runs/36564158729 — conclusion `success` (workflow `ci`, head `ccc5a38`). This is
  the acceptance; the same URL is recorded on ticket 11.
- **Local pre-flight** (before the push): a fresh Python 3.11 venv against a clean
  clone ran the same three steps green — URL-mode install resolving the fork at
  `1110ffb`, then both tests; log `C:\Users\10704\AppData\Local\Temp\11d\dryrun.log`.
- **Ticket 11's record** (decision 12): its Answer now carries the CI item — with that,
  every item of decision 11's evidence list is on the ticket (before/after `ls-remote
  --symref`, the bundle verify/list-heads, the served description and default branch,
  the CI run, `git status` clean, the retired path gone, the push/delete/prune
  transcripts) — `Status: resolved`, and one line in the map's Decisions so far.
- **Tree check** (decision 13): `git diff --stat ff92141..ccc5a38` shows exactly
  `README.md` + `.github/workflows/ci.yml`; old Records and the ADR 0011 mapping
  untouched. The push reused the 11c session `ssh-agent` socket, then killed it
  (`pkill -f 11c-agent.sock`).

Reported per the operating rule — files changed: `.github/workflows/ci.yml` (package
repo, pushed); [ticket 11](../decisions/11-repo-swap.md), this ticket, `map.md`, the
spec's closing execution note (openQHA tracker). Checks run: the dry-run (install +
both tests), the CI first run (green), the package suite `--all` 9/9 on the pushed
tip, the tree diff, `git status` + retired-path re-checks. Not verified: nothing
remaining in this slice's list.

## Review record (2026-09-29, two-axis review of `3145591` + `ccc5a38`)

Two-axis review per the `code-review` skill (fixed point `e83b310`; the package side from
`a5f8103`), the axes as parallel sub-agents. No hard violations on either axis. One
cosmetic item fixed in this annotation; the rest kept or reported (gist of both reports):

**Standards.** Resolve protocol conformant on both tickets — Answer + `Status: resolved` +
one map line each, no triage labels, the resolver placeholders consumed, the
operating-rule report present. Kept: ticket 11's Answer heading was updated in place (an
in-progress status falsified by resolution; the stricter 11c alternative — a dated note —
flagged). Fixed here: the map's 11d Implementation line was blank-flanked in a
consecutive list. Kept as house style: the SHA/URL clump repeats across the doc sites
(11b/11c precedent), and the workflow's two test steps repeat one shape (separate named
steps are the point; a matrix decided against).

**Spec.** Conforms; nothing missing or wrong. Checked: the recorded run's steps match the
workflow 1:1 (`ccc5a38`, attempt 1, success); decision 11's evidence list is complete on
ticket 11; `ff92141..ccc5a38` = README + workflow only; both map lines conformant.
Reported, kept: the spec closing note is outside "What to build"'s letter (append-only
Execution-notes convention, self-reported); the workflow's header comment cites
`spec-repo-swap.md` — a tracker path dangling in the public repo — kept (tracker-reference
precedent; a rewrite would cascade a second run and stale the recorded tip; optional
future cleanup).
