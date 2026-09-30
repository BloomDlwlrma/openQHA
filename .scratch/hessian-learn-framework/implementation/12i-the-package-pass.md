# 12i: The package pass

Serves: [12](../decisions/12-references-sweep.md) · spec: [spec-references-sweep.md](../spec-references-sweep.md).

**What to build:** The one comment/string rule set applied to the package repository (its modules and its tests): process tokens stripped from docstrings and comments ("Ticket N of the Hessian-learning set…", study codes, scratch paths); event sentences become current-state sentences; measured-fact dates kept; user-visible strings (error messages, printed guidance) changed together with the tests that pin them; no behaviour change; identifiers untouched.

**Blocked by:** 12a, 12b, 12c — the content modifications land first; this pass runs on a stable tree.

**Status:** resolved

- [x] Every touched file compiles (`py_compile`).
- [x] The residual-citation scan for the repository is empty.
- [x] The package suite is green (`--all`).

## Answer (2026-09-30, `openQHA-Hessian` @ `cc183eb` + review annotations `669f823`)

**What landed.** The one rule set over the package repository's public face — 7 modules
and 9 test programs plus `ci.yml` and `pyproject.toml` (18 files): module docstrings,
comments and user-visible strings (the run/judge report sections, the validation
refusal, the `num_samples_pt` `TypeError`, the Record schema help) rewritten to current
state; ticket numbers, study codes (`S0-C-*`), ruling dates, round/decision references
and scratch paths (`.scratch/...`, `spec-phl-verbatim.md`) out; event sentences
current-state; identifiers untouched; no behaviour change beyond the message text.

Content fixes riding the rule set: the T04 filename corrected
(`T04_openQHA_Theory_Hessian_Surface_Learning.ipynb`); the derivation/design pointers
mapped onto the public T03/T04/T05 names; the report's balance pointer now T05 §4; the
`VALID_PROBES` schema help dropped its stale "rademacher k=4"; the fixed-probe phrase
and the split vocabulary (whole-molecule / per-frame) unified; `curve_moved`'s
docstring no longer overclaims.

**The pinned pair.** The validation refusal (`openqha_hessian/phl_loss.py`) keeps
"no valid_probes" + `04_dataset` as its actionable core, and
`tests/unit/t_phl_loss.py` now pins exactly those substrings — changed together.

**Evidence.** Strict residual scan (v2 families: `ticket|ruling|grilling|ADR|CONTEXT.md|
.scratch|S0-*|D0-*|plan_[A-D]|checkpoint N|round-N Q*|decision N|hessian-learning-set|
spec-*.md`) over the repo: **0** at the tip (the two-axis review's independent re-scan
counted 153 hits at `cc183eb^`). Review families (43 intended keeps): the README
citation urldate, the fixture log timestamps, "the campaign's", the T03/T04/T05
pointers (T03 is public — archived, referenced by T04/T05), the fork's commit A–D
letters (resolvable in the fork's public history). Compile: `py_compile` over every
module and test — clean. Suites: `--all` **9/9** before and after the annotations pass
(fresh runs; `SUITE_EXIT=0`).

**Review record (two-axis, per the `code-review` skill).** Range `16d3e7a..cc183eb`,
two read-only sub-agents; findings fixed in the annotations commit:
- Spec: one missed citation — `tests/unit/t_phl.py`'s "(spec step 2)" banner (the only
  one left repo-wide).
- Standards: `curve_moved`'s reworded docstring overclaimed ("not in the loss" is false
  for `w_H 0` / a one-epoch run); the split vocabulary was mixed (by-molecule/by-frame
  vs whole-molecule) — unified; judgement-only notes accepted (banner re-dashing; the
  two verified content corrections: T04 filename, T05 §4).

**State.** `cc183eb` is on the package remote; the annotations commit `669f823` is
local-only (this session has no SSH key — the push needs the human's agent) and rides
the next hand-off push.
