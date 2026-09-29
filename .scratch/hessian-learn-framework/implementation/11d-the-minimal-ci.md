# 11d: The minimal CI — the install path on a clean machine, then the record

Type: task
Status: open
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

- [ ] The workflow is committed and pushed; its first run on GitHub is green; the run URL is recorded
- [ ] The workflow installs through `install.sh` and runs exactly the two openQHA-free tests — no full-suite or openQHA-dependent step
- [ ] Ticket 11's Answer carries every item of decision 11's evidence list; `Status: resolved`; one line in the map's Decisions so far
- [ ] The package tree changed only by the README (11a) and this workflow (decision 13); old Records and the ADR 0011 mapping untouched
- [ ] Reported per the operating rule: files changed, checks run, anything not verified

## Answer

<!-- resolver: append the CI run URL + the green conclusion + the ticket-11 record; set Status: resolved; add a line to the map's Implementation -->
