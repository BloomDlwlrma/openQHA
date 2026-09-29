# 10a: Retire the PR expectation in the fork records

Type: task
Status: resolved
Serves: 10
Blocked by: None
Part of: [hessian-learn-framework](../map.md)

> Execution slice of [Upstream: submit the mace-side changes](../decisions/10-upstream-pr.md):
> the pointers its "not submitted — permanent private fork" decision owes the records that
> still expect a PR. Wayfinder conventions apply: the `Status` protocol, no triage labels
> (`docs/agents/issue-tracker.md`, ADR 0009).

## What to build

Ticket 10 retired the upstream-PR expectation (never upstream; the fork is a permanent
private fork — ADR 0013). The earlier records still speak of a coming PR; each such spot
gets a **dated pointer, not a rewrite** — history stays history, but a reader lands on the
decision. Suggested line:

> 2026-09-29 — the upstream PR was decided against (ticket 10): the fork is permanent-private (ADR 0013).

The 2026-09-29 scan (`\bPR\b | upstream reviewer | go upstream | goes upstream | a rebase away`, over `.scratch/hessian-learn-framework/`) found:

- `decisions/02-the-real-fork.md` — "the commits can go upstream (Q4a)" (the question); "the upstream PR is (10)" (facts list); "10 already anticipates excluding it from the PR".
- `implementation/02a-the-mace-fork.md` — six spots, including the "As the upstream reviewer (ticket 10) …" user story and "the six do (ticket 10 decides their final set)".
- `spec-the-package-line.md` — "the upstream PR (05, 02, 10)" in its out-of-scope list.
- `spec-install-and-transport.md` — "the upstream PR cycle" in its out-of-scope list.
- `spec-repo-swap.md` — the future-items line naming the upstream PR cycle.
- `map.md` — the Out-of-scope line "The upstream review cycle after the PR is opened, and any switch of the mace base to a newer release" rewritten to the no-submission / frozen meaning.
- `research/mace-md-fork-practice.md` — one line at the top, `Outcome (2026-09-29): not submitted — see ticket 10 / ADR 0013`; its §5 reads like a live plan.
- `openQHA/AGENTS.md` — one line beside the environment note: the fork's refs (`openqha-hessian` @ `1110ffb`, `base-v0.3.16` @ `5c2d761`) are frozen — no rebase, no force-push, no rewrite (ADR 0013).

Not in scope (owned elsewhere): ticket 10's own text (the decision itself);
[12](../decisions/12-references-sweep.md)'s learning-set pointer item; the package repo's
README (11a's user-approved register, no PR claims); the mace fork and its refs.

## Acceptance

- [x] Re-running the scan shows no un-pointed upstream-PR expectation (benign hits: ticket 10's text, dated research notes with the outcome line, this slice, other repos' PR mentions)
- [x] `map.md`'s Out-of-scope line states the no-submission / frozen meaning
- [x] `openQHA/AGENTS.md` carries the freeze line with both refs verbatim
- [x] `research/mace-md-fork-practice.md` carries the outcome line
- [x] No edits beyond the effort's records (`decisions/`, `implementation/`, `spec-*.md`, `research/`, `map.md`) and `openQHA/AGENTS.md`
- [x] Reported per the operating rule: files changed, checks run, anything not verified

## Answer (2026-09-29, implemented in this commit)

**Pointed — the dated line** `2026-09-29 — the upstream PR was decided against (ticket 10):
the fork is permanent-private (ADR 0013)`, linkified per file; history kept, pointers added:

- `decisions/02-the-real-fork.md` — 3 spots: the Q4a question line (a blockquote under it);
  the "upstream PR is (10)" facts bullet; decision 2's "10 already anticipates excluding it
  from the PR".
- `implementation/02a-the-mace-fork.md` — 6 spots: the Problem Statement's "PR-able branch"
  (a blockquote under the paragraph); user story 2 (the upstream reviewer); the consequences
  bullet ("(10) excludes D"); the Out-of-scope "(10)" bullet; Further Notes' "the six do";
  the Answer's "(… the PR are their own tickets)".
- `spec-the-package-line.md`, `spec-install-and-transport.md`, `spec-repo-swap.md` — each
  out-of-scope line naming the upstream PR / its cycle.
- `research/mace-md-fork-practice.md` — the outcome line at the top (`**Outcome (2026-09-29):
  not submitted — see ticket 10 / ADR 0013.**`); §5 stands as written, under it.
- `map.md` — the Out-of-scope line rewritten: "The upstream review cycle — the PR was decided
  against; never submitted ([10](decisions/10-upstream-pr.md), [ADR 0013](…)) — and any switch
  of the mace base to a newer release (the fork is frozen)."
- `openQHA/AGENTS.md` — the freeze bullet with both refs verbatim: `openqha-hessian` @
  `1110ffb`, `base-v0.3.16` @ `5c2d761` (no rebase, no force-push, no rewrite; ADR 0013).

**Re-scan (the same regex, over `.scratch/hessian-learn-framework/` + `openQHA/AGENTS.md`).**
Transcript: `C:\Users\10704\AppData\Local\Temp\10a-verify-out.txt`. Every remaining hit is a
pointer or benign:

- `decisions/10-upstream-pr.md` — 9 hits, all its own text (the decision; benign by acceptance).
- The pointer lines themselves — one per spot (02's blockquote included).
- `02a`'s "classifier/PR hygiene checks already in `hpc/tools/`" — a prior-art tooling mention
  (past tense), not an expectation.
- The CI triggers — `spec-repo-swap`'s and `11d`'s "on push/PR to `main`" (the GitHub
  mechanism, not an upstream PR).
- `11c`'s record quoting the stale map line while deferring it to ticket 10's session —
  history, and this slice is the fix it names.
- The research note's mace-md / upstream PR references (other repos' PRs) and §5's
  pre-decision inferences — the outcome line governs the note.
- This slice's own text.

**Reported per the operating rule.** Files changed: the eight records above plus this ticket
and `map.md`'s Implementation line. Checks run: the full diff read back; the re-scan (grep,
the ticket's regex) — no un-pointed expectation; `file` — UTF-8/CRLF unchanged. Not verified:
nothing runtime — no code touched, so no typecheck and no test suite was run (docs-only). The
ticket-10 records this slice points at (`10-upstream-pr.md`'s resolution, `docs/adr/0013`, the
research note) were landed in the same commit — uncommitted from ticket 10's session.
