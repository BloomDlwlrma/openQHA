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
Transcript: `C:\Users\10704\AppData\Local\Temp\10a-verify-out.txt` (mid-work); the
post-commit re-run on `4c77a81` is `C:\Users\10704\AppData\Local\Temp\10a-postscan.txt`.
Every remaining hit is a pointer or benign:

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

**Reported per the operating rule.** Files changed: commit `4c77a81`'s eleven — the eight
records pointed above (the research note among them; `map.md` also carries the Implementation
line), this ticket, and the ticket-10 records landed with it (`decisions/10-upstream-pr.md`'s
resolution and `docs/adr/0013`, uncommitted from ticket 10's session). Checks run: the full
diff read back; the re-scan (grep, the ticket's regex) — no un-pointed expectation; the
post-commit re-run on `4c77a81`; `file` — UTF-8/CRLF unchanged. Not verified: nothing runtime —
no code touched, so no typecheck and no test suite was run (docs-only).

## Review record (2026-09-29, two-axis review of `4c77a81`)

Two-axis review of the resolution commit (fixed point `e7f079b`) per the `code-review`
skill; the axes ran as parallel sub-agents. No hard violations; two findings were fixed in
this annotation commit, the rest kept or reported (gist of both reports):

**Standards.** Resolve protocol conformant — both tickets carry `Status: resolved` and
`## Answer`, no triage labels; ADR 0013 follows the house ADR shape; the dated-pointer
appends respect "Records are immutable" (the shape 11c's review record blessed). Baseline
smells (all judgement, kept): the pointer sentence's repetition (worst:
`decisions/02-the-real-fork.md` carries it twice ~8 lines apart) and the ref/SHA clump
travelling across the records — the 11b record already ruled that clump a kept call.

- This ticket's Answer "Files changed" line named 9 of the commit's 11 files — fixed in
  this annotation (the ticket-10 records are now named in it).
- Ticket 10's Answer carries no "Reported per the operating rule" block — reported, not
  fixed (ticket 10's text is outside this slice).
- Noted (outside the diff): the map's ticket-10 Decisions line and Q4 revision landed a
  commit before ticket 10's resolution (`e7f079b`, where the ticket still read
  `Status: open`) — timing only; the endpoint is consistent.

**Spec.** Faithful — all spec'd spots pointed, every relative link resolves at its file's
depth, both refs verbatim, no scope creep (the ticket-10 records riding in the commit are
the sanctioned set), and the re-scan on the committed tree stays clean (the post-commit
transcript added above).

- Reported, not fixed: ticket 10's Answer and ADR 0013 cite a PR precedent (`#1445`
  retargeted main→develop before merge) that the findings note does not carry; spot-checked
  during this review on the PR page — base changed main→develop, merged into `develop` on
  2026-06-05 — verified, but its provenance lives only in ticket 10's text.
- `11c`'s record still quotes the stale map line while deferring it to ticket 10's session;
  the Answer discloses it as history — the deferral it names is this slice's fix, now
  landed.
- The mid-work transcript gave way to the post-commit re-run (fixed above); the earlier
  file stays as the work log.
