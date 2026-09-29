# 12d: The library pass

Serves: [12](../decisions/12-references-sweep.md) · spec: [spec-references-sweep.md](../spec-references-sweep.md).

**What to build:** The one comment/string rule set applied to the whole library subtree: process tokens stripped (ticket numbers, study codes, ruling dates, grilling/decision/round references, scratch paths); event sentences become current-state sentences; measured-fact dates kept; user-visible strings changed together with the tests that pin them; no behaviour change; identifiers untouched.

**Blocked by:** 12a, 12b, 12c — the content modifications land first; this pass runs on a stable tree.

**Status:** resolved

- [x] Every touched file compiles (`py_compile`) — clean over every `openqha/**/*.py`.
- [x] The residual-citation scan for the subtree is empty — apart from the two live-contract carve-outs recorded in the Answer.
- [x] The unit suite is green — 60/60 (`C:\Users\10704\AppData\Local\Temp\t12d_suite.log`).

## Answer (2026-09-30, implemented in this commit)

**The pass.** The one rule set over the whole library subtree (`openqha/`, 46 files
touched; comment/docstring/schema-description/user-visible-string text only — code,
identifiers, regexes and behaviour untouched). Counted first, then edited: the
pre-pass scan read 108 ticket, 35 ruling, 27 ADR, 15 `CONTEXT.md`, 42 `S0-*`,
42 `D0-*`, 19 Q, 21 plan, 15 round-N and 2 checkpoint hits (with the kept classes:
211 dates and the live fingerprint machinery — the qm9 source checksum, the patch
source fingerprint, the probe seeds). Event sentences became current-state
sentences (“It has now been run, on two molecules (branch A checkpoint 5)” → “…two
molecules”); ruling citations became `(since <date>)` or vanished; measured-fact
dates kept; `defect NN` and stage vocabulary kept (public, per the 12c precedent).
Beyond the strip, three stale-claim spots in the subtree were taken to current
behaviour: `store/report.py`'s provenance row now reads “the engine name and its
resolved weight file” (the SHA-256 identity is retired), `potentials/engine.py`'s
header states the current identity model instead of narrating the retirement
decision, and `store/artifacts.py`'s retired-mechanics citations became plain
descriptions.

**The canonical example** (`md_record.py`): “Records redesign (user ruling
2026-09-15, tickets 16-18): no setting LEVEL under …” → “Records layout: no setting
LEVEL under …” — the description kept, the provenance gone.

**Live-contract carve-outs (reported, not changed).** `store/artifacts.py` keeps
the `decision` field's accepted-value grammar: the `_DECISION_RE` regex, docstring
item 4 (`S0-*` / `D0-*`) and the `_check_decision` refusal message — the field's
value IS a study code by design (the unit test writes `decision="S0-D-10"` and
round-trips it), so a literal-free message would lose its only concrete guidance;
this is an API contract, not a citation. The scan's second hit is the `scratch=`
keyword argument in `frame_labels.py` (a false positive). No other residual tokens
remain in `openqha/`.

**Evidence.** Residual scan over `openqha/` (pattern:
ticket|ruling|grilling|ADR|CONTEXT.md|.scratch|.mem|S0-*|D0-*|Q|R4|round-N|plan
-refs|checkpoint|research-note|set names): only the two carve-outs above). Compile:
`py_compile` over every `openqha/**/*.py` — all green. Unit suite: 60/60 (run on
the shared tree, which also carried the in-flight 12e/12f edits). Not run here: the
full `--all` suite — the 12d acceptance is the unit suite, the pass changes no
behaviour, and the retirement baseline is [12h](12h-the-ci-and-the-close-out.md)'s.
