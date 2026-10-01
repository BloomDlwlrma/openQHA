# 15f: The yhbatch NAME defect -- the timing job's first submissions and the DSET fix

Type: task
Status: resolved
Part of: [hessian-learn-framework](../map.md)
Serves: [15](../decisions/15-the-balance-on-the-probe-estimator.md) · spec: [spec-the-balance-on-the-probe-estimator.md](../spec-the-balance-on-the-probe-estimator.md).

**What happened.** The first timing submissions of the post-15e deployment died in seconds:
the site's `/usr/bin/yhbatch` wrapper delivers `NAME=yhbatch` to every job it submits, so
`05_train.py` looked for `_datasets/yhbatch/mace_yhbatch.<level>.extxyz` and refused with
`FileNotFoundError` (`run 04_dataset.py first`). The fix -- the Dataset name rides `DSET` --
landed as openQHA `f2aa2ee`; the resubmitted timing job (260224) is green at its echo line
(`dataset draw300_r1`). This record captures the incident, the probes and the fix.

## The incident (2026-10-01, tianhexy-ai)

| job | what it was | outcome |
|---|---|---|
| 260202 | the first timing submit (multi-line block; `NAME=draw300_r1`) | died in seconds -- `dataset yhbatch`; looked for `_datasets/yhbatch/...` |
| 260205 | a standalone `yhbatch` re-run (the assignment prefix lost) | died in seconds -- `TAG not set` |
| 260206 | the block re-run with the preflight appended (`env \| grep`) | submitted; cancelled (a clean single-line submit superseded it) |
| 260207 | clean single-line submit -- `NAME=draw300_r1` plainly on the command line | died in seconds -- **still `dataset yhbatch`**, so the clobber was systematic, not a typo |
| 260214/260215 | probe jobs (bare script printing `$NAME`/`$TAG`/`$1`) | `NAME=yhbatch` already at job entry; `TAG`/`RUN` intact; **positional args survive** (`ARG1=probe99`) |
| 260220 | probe with `--export=ALL,NAME=probe49` | `NAME=probe49` -- `--export` reclaims it (production does not use it; below) |
| 260221 | probe via `/usr/bin/sbatch` directly (bypassing the wrapper) | `NAME=probe51` intact -- the wrapper is the polluter |
| **260224** | the timing job on the fixed checkout (`DSET=draw300_r1`) | **running**; echo line reads `tag draw300 dataset draw300_r1 run timing1` |

## Root cause (evidence)

`/usr/bin/yhbatch` is a five-line site wrapper:

```bash
#!/bin/bash
NAME=yhbatch
CMD=/usr/bin/sbatch

exec -a $NAME $CMD "$@"
```

Every job submitted through it arrives with `NAME=yhbatch` in its environment, overwriting
even an explicitly exported `NAME`. Only `NAME` is affected -- `TAG`, `RUN`, `EXTRA`, the
`PT_*` paths and positional script arguments all arrive intact. Our own stack was ruled out
(no `NAME` assignment in `hl_train.slurm`, `hpc/env/common.sh`, `hpc/env/tianhe.sh`,
`hpc/env/root.sh`). The gap was latent: this was the first GPU-side job family that relied
on `NAME != TAG` (the CN-side labeling ran with stock `sbatch`, which is clean).

## The fix (`f2aa2ee`, openQHA)

`hl_train.slurm` reads `NAME="${DSET:-$TAG}"`; the Dataset name rides **`DSET`**
(probe-verified to survive). Flipped in the same commit: `hpc/slurm/hl_train.slurm` (the
line + the header example), `workflows/hessian_learning/README.md`, the T05 production
cell, `docs/hessian_learning_campaign.md`, the three 09g blocks + its second revision
note, `spec-round-1-run.md` story 14 (dated amendment), and `t_frame_labels` (the pin).
`--export=ALL,NAME=...` reclaims `NAME` but is ruled out for production (user, 2026-10-01);
direct `/usr/bin/sbatch` is clean but using the site wrapper stays the practice. Deploy:
repack `openQHA-main-dotgit.tar` (sha256 `43eab445...`), per side `tar -xf` +
`git reset --hard f2aa2ee` (AI side done 2026-10-01 ~20:06; CN side per the same steps).

## Notes

- **Submit-line discipline** (kept from the incident): preflight and submit as separate
  commands (`env | grep -E '^(TAG|DSET|RUN)='` vs the block ending in `yhbatch ...`); after
  submit, check the wrapper log's echo line `tag <TAG> dataset <NAME> run <RUN>` and
  `scancel` on mismatch.
- **Timing note for the fixed head** (09g step 2's wall check): the pre-training phases are
  read off the run dir's mtimes -- `config.yaml` marks split + balance done, `logs/<tag>.log`
  marks mace entry (mace-start - job-start = split + balance + the before-anchor); the full
  wall check uses `wall - MAX_EPOCHS x S/E` at the end.
- **Sibling scripts still read `NAME`** (`hl_labels`, `hl_frames`, `hl_branchA`,
  `hl_pipeline_debug`): unaffected until a GPU round passes `NAME != TAG` through
  `yhbatch`; sweep them when their rounds come.

## Answer

Incident diagnosed to the site wrapper, fix landed and verified live (260224's echo line);
the jobs, probes and surfaces are tabulated above. Not verified: the CN-side refresh and
260224's receipts (they ride [09g](09g-the-timing-and-production-runbook.md)).
