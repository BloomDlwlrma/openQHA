# 12a: The Slurm drift

Serves: [12](../decisions/12-references-sweep.md) · spec: [spec-references-sweep.md](../spec-references-sweep.md).

**What to build:** The trajectory and branch Slurm preflights print the engine name and the resolved weights path — the retired digest fields are gone, so submitting no longer dies with a KeyError. The three weights-check pipelines (branch-A job, pipeline debug, parked rerun) match the check tool's current stdout, so their fallback warning fires only when a weight file is actually missing. The labels job captures the Dataset build's exit status — a failed build fails the job — and names the split mode explicitly.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Every touched script passes `bash -n`.
- [x] The three preflights print `engine` + the resolved weights path; no reference to the retired digest keys remains.
- [x] Each grep matches the check tool's current weights-path line; the fallback/warning texts name what is checked.
- [x] The labels job exits non-zero when the Dataset build fails; its build call names the split mode explicitly.

## Answer (2026-09-30, implemented in this commit)

**The three preflights** now read the reduced provenance contract and print the engine
name plus the resolved weights path — a missing weight file still raises inside
`engine.provenance()` before any work:

- `branchA_debug.slurm:58–59` (the inline comment now says what the call resolves),
  `branchA_deimos.slurm:73–74` and `branchB_traj_tianhe_a.slurm:108–109` — the retired
  `sha256` / `sha256_pinned` prints are replaced by
  `p["engine"]` / `p["weights_path"]`.

No digest-key reference remains in any of the seven touched files (scanned).

**The three weights-check pipelines** take the reduced tool's weights-path line as the
sentinel — `grep -E "^  path"` — and their fallbacks name what is checked:

- `hl_branchA.slurm:63`, `hl_pipeline_debug.slurm:67` — `"openQHA: the weights check did
  not list a weight file"`.
- `q5-rerun-parked.sh:420–421` — the same sentinel; the warning names the weight file.

Verified against the tool's live stdout in the `openqha` env (`S0_MACE_ROOT` on temp
roots): one weight file present → the sentinel matches its `  path` line; empty root →
the tool prints "no registered weight file is present", the sentinel does not match, so
the fallback fires only when no weight file resolves.

**The labels job** (`hl_labels.slurm:138–147`): the Dataset build's exit status is
captured (`rc_ds=${PIPESTATUS[0]}`; the script runs without `set -e`, so the capture is
the idiom), a failed build makes the job exit non-zero (`exit "$rc_ds"`), an assemble
failure still exits first, and the build call passes `--split-by molecule` explicitly —
the production default the campaign dataset was built with
(`dataset.SPLIT_MODES = ("frame", "molecule")`, `DEFAULT_SPLIT_MODE = "molecule"`).

Reported per the operating rule — files changed: the seven scripts
(`branchA_debug.slurm`, `branchA_deimos.slurm`, `branchB_traj_tianhe_a.slurm`,
`hl_branchA.slurm`, `hl_labels.slurm`, `hl_pipeline_debug.slurm`, `q5-rerun-parked.sh`).
Checks run: `bash -n` on all seven; the retired-string scan (empty); the live sentinel
check above; `t_engine_identity` PASS; unit group 60/60; full suite `--all` 73/73
(`C:\Users\10704\AppData\Local\Temp\suite12a.log`). Not verified: nothing was submitted
to Tianhe, so the preflights' python blocks and the labels job's failure path are not
exercised as live Slurm jobs (the exit-code wiring is verified by the no-`set -e`
pipeline semantics). The working tree also carries another slice's edits
(`openqha/data/dataset.py`, `tests/unit/t_engine_fork.py` — 12b's); they are not part of
this commit.
