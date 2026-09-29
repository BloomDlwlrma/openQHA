# 11a: The README rewrite — clean register, no History

Type: task
Status: resolved
Serves: 11
Blocked by: None
Part of: [hessian-learn-framework](../map.md)

> Execution slice of [The repo swap](../decisions/11-repo-swap.md) per
> [spec-repo-swap.md](../spec-repo-swap.md): decision 6 (the rewrite) and decision 7
> (the wording rule); stories 2-8, 13. The draft is public text: it is shown to the
> human before it is committed. Wayfinder conventions apply: the `Status` protocol, no
> triage labels (`docs/agents/issue-tracker.md`, ADR 0009).

## What to build

The package repository's `README.md`, rewritten clean in the register of hip and the
PHL source repository (references extracted at
`source-code/final-workflow-design/hessian-train/PHL-main/`), committed on the local
`main` on top of `ff92141` so that the swap's first push (11c) already carries the
finished text:

- title; one short paragraph: the training side of openQHA — not a MACE fork;
- the index links — openQHA `https://github.com/BloomDlwlrma/openQHA`, mace
  `https://github.com/BloomDlwlrma/mace` (branch `openqha-hessian`);
- Install — both `install.sh` modes, the run-it-last and eval-wheel notes, tightened;
- Verify — the runner, `--all`, the provenance header;
- Citation — the PHL paper (bibtex key `rodriguez2026projectedhessianlearningfast`,
  arXiv 2603.04523) and the openQHA software entry (`@misc{openqha}`, per openQHA's
  `docs/cite/`).

No History section, no old hashes or mapping talk, no badge. The README does not
discuss the predecessor at all; externally it is only "an early MACE fork experiment".

## Acceptance

- [x] The draft is shown to the human before the commit (public text)
- [x] The README carries every element above and none of the forbidden ones; both index links and both citation entries are exact
- [x] One commit on the package repo's `main`; the diff touches `README.md` only — `install.sh`, `pyproject.toml`, `openqha_hessian/`, `tests/`, `.gitignore` unchanged (decision 13)
- [x] Reported per the operating rule: files changed, checks run, anything not verified (see the Answer)

## Answer (2026-09-29, agent-run)

Resolved 2026-09-29, agent-run. The rewrite landed on the package repo's `main` as
`a5f8103c7d644f42c892a495ecb13298f38bfe64` -- `README.md` only (97 insertions, 9
deletions; tree clean afterwards). The commit was amended in place pre-push with the
user's final revision waves; nothing has ever been public under an earlier id. The
draft was shown to the human before the commit and approved after four revision
rounds; the user's rulings stand as amendments to the spec's letter and are the record
of what the README now carries beyond the ticket's element list:

- the intro's `Important` warning, in the PHL register but adapted to this stack:
  custom Hessian training "does not work with the current unmodified upstream mace
  package; it needs the fork `BloomDlwlrma/mace` (branch `openqha-hessian`), installed
  editable";
- **How PHL enters MACE** (new): the Cartesian target `||H_theta - H_r||_F^2 / (9
  N^2)`, the probe estimator and its cost (`2 + 2k` vs `2 + 6N`), the fork's three
  pieces, the package's six modules, the 00-06 workflow, and a PHL-paper citation
  hint;
- a brief Install opening; the "machines that only evaluate" note removed;
- the intro trimmed: "openQHA never imports it" and the not-a-fork sentences out;
- the Citation section carries only the single openQHA entry, under "If you use
  openQHA, openQHA-Hessian in your work, please cite it"; the PHL paper's own entry
  is out of it (its citation hint above stands);
- a License section, CC BY-NC 4.0 with badge -- the badge overriding decision 6's
  "no badge".

Evidence: `git show --stat a5f8103` (README.md only); fences balanced, no trailing
whitespace; `python tests/run_tests.py --all` in the WSL `openqha` env -> all 9
test(s) passed, EXIT=0 (unit 6/6, integration 3/3; log
`C:\Users\10704\AppData\Local\Temp\11a-suite.log`); `git status` clean on `main` at
`a5f8103`, unpushed (11c pushes it). The map's Implementation line is on disk; this
commit carries the ticket only -- `map.md` held a parallel session's uncommitted
10/08a lines, so the 11a line rides the next tracker commit. Not verified: the
README's URLs were not fetch-tested (openQHA is not public); no code was touched (no
type or lint pass applies beyond the suite).

## Review record (2026-09-29, two-axis review of the README commit, pre-amend)

Two-axis review (reviewed at `b012e4b`, fixed point `ff92141`) per the `code-review`
skill. Spec axis: clean -- no missing requirements, no creep beyond the session
amendments, and the new section's facts check against `hvp.py`/`phl.py`/`phl_loss.py`/
`run.py` and the workflow scripts. Standards axis: one hard finding -- the slice's
resolve protocol was still pending at review time (discharged with this ticket) --
plus three judgement calls left as user-approved text (the citation hint and the full
entry then coexisted; the fork facts repeat across intro/Install by design; "E-F-H"
is not expanded). No annotations commit; nothing actionable beyond the resolve. The
commit was afterwards amended in place pre-push with the user's final revisions
(intro trim; Citation restructure and wording -- user-specified, README-only); the
reviewed content is otherwise unchanged.
