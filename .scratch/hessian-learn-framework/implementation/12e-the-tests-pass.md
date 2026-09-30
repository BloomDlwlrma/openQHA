# 12e: The tests pass

Serves: [12](../decisions/12-references-sweep.md) · spec: [spec-references-sweep.md](../spec-references-sweep.md).

**What to build:** The one comment/string rule set applied to the test tree: process tokens stripped from comments and docstrings; event sentences become current-state sentences; measured-fact dates kept; strings that pin changed messages updated together with them; no behaviour change; assertion logic untouched.

**Blocked by:** 12a, 12b, 12c — the content modifications land first; this pass runs on a stable tree.

**Status:** resolved

- [x] Every touched file compiles (`py_compile`).
- [x] The residual-citation scan for the subtree is empty.
- [x] The unit suite is green.

## Answer (2026-09-30, implemented in this commit)

**The pass.** The one rule set over the whole test tree — 73 files changed (+308/−316:
67 `.py` including the runner and `_testlib`, `tests/README.md`, and five comment- or
prose-bearing data files under `tests/data/`). Stripped: every internal-process citation
(ticket numbers, `S0-*`/`D0-*` study codes, rulings and their dates, grilling `Q`/round
references, `decision N`, `plan_*`/checkpoint/skills-section references, "records
redesign", "Hessian-learning set/effort", `.scratch`/ADR provenance mentions, the round-2
seam codes). Rewritten: event sentences → current-state sentences ("were voided",
"used to be", "Replaces …", "Until …", "The first run did exactly that", "was missing …
until"). Kept: measurement dates and observed-incident notes (`an113`, the Tianhe runs,
measured wall times, `defect 57`), and every identifier, path and assertion.

**The kept classes** (deliberately untouched, enumerated for the scan): the
`decision="S0-D-10"` value in `t_artifacts_identity` — the decision-identifier format
`store/artifacts.py` validates (the same live-API carve-out 12d recorded); functional
environment and directory names (`S0_REQUIRE_CAPS`, `S0_RUNS_ROOT`, the `.mem` entry in
the module-walk skip tuple); the retirement contracts that assert the old keys do NOT
creep back (`params_sha256`, `ENGINE_PIN_STATUS`); functional test data (`COMMIT`,
`crest_version`, the old-attribute frame rewrite in `t_dataset`); "Branch B criterion N"
acceptance-criterion labels and the design-term "seam" vocabulary.

**Evidence.** Residual scan over the subtree: `.py` — 78 files, process-pattern hits 4,
all in the kept classes or false positives (the two `S0-D-10` API strings; the `.mem`
skip name; `around 1e15` matching `round \d`); non-`.py` — 94 files, 0 hits. Compile:
`py_compile` 67/67 and `tomllib` 4/4 on every touched file. Suites: unit 60/60; `--all`
73/73 (`C:\Users\10704\AppData\Local\Temp\oqt12e_unit.log`, `oqt12e_all.log`; scan logs
`oqt12e_scan.log`, `oqt12e_scan2.log`). No pinned message string needed changing: this
pass alters no library message, and 12a–12c kept their pinned substrings (`t_dataset`'s
refusal, the fork guard in `t_train_run`), which stay green here.

**Reported, not acted on:** a stray untracked file in the repository root from a botched
PowerShell redirect in a parallel slice (`ers10704AppDataLocalTemp12f_annot_commit2…`);
left for the close-out.
