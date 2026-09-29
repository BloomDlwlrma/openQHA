# 12g: The hpc and workflows pass

Serves: [12](../decisions/12-references-sweep.md) · spec: [spec-references-sweep.md](../spec-references-sweep.md).

**What to build:** The one comment/string rule set applied to the hpc tree and the hessian-learning workflow code: process tokens stripped; event sentences become current-state sentences; measured-fact dates kept; user-visible strings changed together with the tests that pin them; no behaviour change. The Slurm scripts that 12a touched are cleared in the same files, after it.

**Blocked by:** 12a, 12b, 12c — the content modifications land first; this pass runs on a stable tree.

**Status:** resolved

- [x] Every touched file compiles (`py_compile`) or parses (`bash -n` for scripts).
- [x] The residual-citation scan for the subtree is empty.
- [x] The unit suite is green.

## Answer (2026-09-30, implemented in this commit)

**What the pass changed.** The one comment/string rule set over the 43 tracked
files of `hpc/**` and `workflows/hessian_learning/**` (~190 rewrites): every
process token out (ticket numbers, `S0-X-N`/`D0-*` study codes, ruling phrases
and their dates, `ADR NNNN`, `CONTEXT.md`, `.scratch`/`.mem`/`_superseded`
paths, `plan_*`, `round NN`, `open item NN`); event sentences turned into
current-state sentences (`` `set -eo pipefail` removed 2026-09-13 … `` → "No
`set -eo pipefail`: a failing step must not end the job", "Until `<date>` X" →
"X used to", "was reopened by ruling" → "by decision", "carries a ticket" → a
plain noun); measurement dates and `(user, <date>)` attributions kept;
`defect NN` and acceptance-criterion references kept (the 12c precedent);
identifiers untouched (the `S0_*` env-var names, `FIX_COMMIT`'s hash, every
flag and value). Retired-mechanics claims fixed where they stood: the two
"weights hash / hash correctly" lines (`branchA_debug.slurm`,
`tianhe_cpu.py`) now say the weights resolve, and the workflow scripts'
docstrings, help texts and echoes lost the internal study shorthand while
keeping every behaviour-relevant sentence.

**The Slurm scripts 12a touched** (`branchA_debug.slurm`,
`branchA_deimos.slurm`, `branchB_traj_tianhe_a.slurm`, `hl_branchA.slurm`,
`hl_labels.slurm`, `hl_pipeline_debug.slurm`, `q5-rerun-parked.sh`) carry
their content fixes plus this pass's strips in the same files. The rerun
script keeps its fixed commit hash and now carries the why ("the commit that
carries the frequency-floor census screen", the spec's Further-Notes bullet);
its `say` strings lost their ticket references.

**User-visible strings touched:** the `04_dataset.py` R4 print, argparse help
texts (steps 02–06, `hl_list.py`'s `--force`), the `hl_labels.slurm` /
`hl_train.slurm` echoes, the q5 `say` lines; no test pins those strings
(verified by grep), and every pin-adjacent suite was re-run anyway —
`t_frame_labels` PASS, `t_hl_campaign` PASS, `t_hl_list` PASS,
`t_dataset_cli` PASS, `t_walltime_forms` PASS, `t_tianhe_root_by_partition`
PASS, `t_repo_bootstrap` / `t_module_entry_points` rc=0.

**Evidence.** Compile/parse: `py_compile` on every touched `.py` (the seven
workflow steps, `hpc/labels.py`, `hpc/providers.py`, all
`hpc/resource_configs/*.py`, `hl_list.py`), `bash -n` on every touched shell
script, `json.load` on the five `hpc/configs/*.json` — all green. Residual
scan over `hpc` + `workflows` (`ticket|ruling|grilling|ADR N|CONTEXT.md|
.scratch|.mem/|S0-X-N|D0-*|plan_*|hessian-learning-set|round N|open item|
memory.md|superseded`) — empty; the only `_superseded` left is a runtime
path guard in `03_labels.py`, and the only surviving `user` forms are the
deliberately kept `(user, <date>)` attributions. Unit suite: 60/60 green on
the swept tree (`12g-unit.log`). Two-axis review ran on the pre-commit diff;
its findings (a "the the" splice in `hl_labels.slurm`, the array-script
referent, a dropped observation date) are fixed in the pass and the
pin-reading tests re-ran green after them.

**Not verified:** nothing was submitted to Tianhe — the Slurm scripts are
compile-checked and content-pinned by the existing tests, not executed as
jobs. The working tree also carries the other sweep slices' edits
(`openqha/`, `tests/`, `scripts/`, root files — 12d/12e/12f in flight); they
are not part of this commit.
