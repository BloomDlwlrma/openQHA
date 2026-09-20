# The Hessian-learning campaign on Tianhe — the page it is run from

The sequence, the cost, what each log must say, how a killed job is resubmitted, and the
one table that says where the campaign stands. Site facts (partitions, filesystems,
install) are in [`tianhe_runbook.md`](tianhe_runbook.md); what each step computes is in
[`../workflows/hessian_learning/README.md`](../workflows/hessian_learning/README.md).
Everything below is TianheXY-C (`debug` for the gates, `deimos` for the campaign; one
node = 64 cores, 512 GB). Numbers are marked **[measured]** or **[estimate]**.

The rules this page follows (rulings 2026-09-18 … 20): nothing runs on the login node
except the second-long steps; every test is a `debug` job; a submitted job is plain bash +
`xargs`, no parsl (parsl only in the ALF mode, §6); one campaign is one tag
(`TAG=draw300`, the Dataset has the same name); the molecule tree is flat
(`<root>/draw300/<qid>/`), a frame's ORCA files are `frames/orca.<level>.<frame>.*`.

---

## 1. The sequence

```bash
cd ~/openQHA-main; export OPENQHA_PARTITION=deimos; source hpc/env/common.sh && source hpc/env/tianhe.sh

# 00  the draw: 300 per structure class outside SPICE, the union (login node, seconds; the same seed = the same 6,458 molecules)
python workflows/hessian_learning/00_draw.py --tag draw300 --per-class 300 --seed 0

# the gate: every stage once on debug, 16 molecules / 16 frames -- read each log (§4) before the next line
TAG=draw300 LIMIT=16 sbatch --partition=debug --time=00:30:00 hpc/slurm/hl_branchA.slurm
TAG=draw300 LIMIT=16 sbatch --partition=debug --time=00:30:00 hpc/slurm/hl_frames.slurm
python workflows/hessian_learning/01_select.py --tag draw300
TAG=draw300 LIMIT_FRAMES=16 sbatch --partition=debug --time=00:30:00 hpc/slurm/hl_labels.slurm

# the campaign: arrays of 12 one-node tasks, the list split round-robin, finished work skipped
TAG=draw300 sbatch --array=0-11 --time=1-00:00:00 hpc/slurm/hl_branchA.slurm
TAG=draw300 sbatch --array=0-11 --time=04:00:00 hpc/slurm/hl_frames.slurm
python workflows/hessian_learning/01_select.py --tag draw300                  # the whole draw, with basins / frames present
TAG=draw300 sbatch --array=0-11 --time=3-00:00:00 hpc/slurm/hl_labels.slurm     # round 1 of 3 (§3)
python scripts/tooling/s0_hl_progress.py --tag draw300                        # any time, seconds
TAG=draw300 sbatch --array=0-11 --time=3-00:00:00 hpc/slurm/hl_labels.slurm     # round 2, when round 1 has ended
TAG=draw300 sbatch --array=0-11 --time=3-00:00:00 hpc/slurm/hl_labels.slurm     # round 3 -- until task 0's assemble exits 0
python workflows/hessian_learning/04_dataset.py --tag draw300 --export openreact   # the Dataset (by frame, 90/5/5), the merged xyz, the HDF5
```

The order that buys nothing blind: branch A and 02 for the WHOLE draw first (a day),
then `01_select` and `s0_hl_progress` give the exact frame count, and the labels array
goes in with a cost you have read off, not guessed.

The gate's labels job on 19-atom molecules is the first real measurement of the
Hessian and gradient job times (§2): if its table shows `FAILED` with an ORCA timeout
tail, the Hessian did not fit in 30 minutes — rerun that gate as
`TAG=draw300 LIMIT_FRAMES=16 sbatch --time=03:00:00 hpc/slurm/hl_labels.slurm` on `deimos`.

## 2. The cost

Layout everywhere: 16 workers × 4 cores per node (branch A: 4 CREST/MACE threads; labels:
4 ORCA ranks, `%maxcore 6000`). 12 nodes = 768 cores. The draw: 6,458 molecules, 83 %
with 9 heavy atoms (≈ 19 atoms); a molecule's Frame set is ~3 basins × (1 + 4 displaced)
plus merged / saddle frames ≈ 15–17 frames, of which 3–5 are Hessian jobs (basin /
merged / saddle: `EnGrad Freq`) and ~12 gradient jobs (displaced: `EnGrad`; round 5 Q7 b).

| stage | per unit | basis | draw300 (12 nodes) | measured on the campaign |
|---|---|---|---|---|
| A branch A | 460–590 s / molecule at 16 per node **[measured 2026-09-19, debug]** | 6,458 × 525 s × 4 cores = 3,800 core-h | **~5 h** (array `--time=1-00:00:00` is slack) | — |
| 02 frames | ~1 min / molecule **[estimate from the smoke set]** | 6,458 × 60 s × 4 = 430 core-h | **< 1 h** | — |
| 03 Hessian jobs | 10 atoms: 245–305 s **[measured]**; 19 atoms: 40–80 min **[estimate, N³–N⁴ scaling]** | 6,458 × 4 × 60 min × 4 cores = 103,000 core-h | 5.6 days | — |
| 03 gradient jobs | 19 atoms: 3–5 min **[estimate]** | 6,458 × 12 × 4 min × 4 = 20,700 core-h | 1.1 days | — |
| 03 total | | **≈ 125,000–155,000 core-h** | **≈ 7–8.5 days** → 3 rounds of 3 days | — |
| 04 Dataset | seconds per molecule, login node or task 0 | | minutes | — |

The formula behind the 03 rows, for N molecules with F frames each at T per frame on
P cores: `N × F × T × P / 768` hours of 12 nodes. The gate's log (§4) gives T for both
job kinds on the draw's own molecules; write them into the last column, and if the
Hessian is over 80 min the rounds become four.

Storage, kept (no `.gbw`, `.loc`, `property.txt`; round 5 Q8): ~0.35 MB per frame
(`inp` + full `out` + `hess` or `engrad`; a 19-atom `.hess` is ~0.15 MB) → **~35 GB** for
~100,000 frames; ~1 MB per molecule of branch A / Frame sets → 6.5 GB; the Dataset:
~7 GB of extxyz per level per 100,000 frames (three split files + the merged one =
twice that) and 2.6 GB of HDF5. Node-local scratch holds ORCA's integrals during a job
(GB per analytic Hessian) and is removed with the run directory.

## 3. The 3 × 3-day scheme for the labels

`deimos` allows 7 days, but a 3-day walltime queues faster and loses at most 3 days of
one node to a crash. Each round is the same command:

```bash
TAG=draw300 sbatch --array=0-11 --time=3-00:00:00 hpc/slurm/hl_labels.slurm
```

What one round does: each of the 12 tasks lists the PENDING frames of the selection at
its start (`03_labels.py --list`: no finished job, no fresh `.running` claim), takes every
12th of them (`SLURM_ARRAY_TASK_ID`), and runs them 16 at a time under `taskset`; a frame
is claimed with `frames/<stem>.running` while ORCA runs and the claim is removed after.
At the end task 0 assembles every molecule's label files and builds the Dataset; its exit
code is **0 when nothing is left unlabelled, 1 otherwise** — that is the signal for the
next round. A round that hits its walltime dies mid-frame: the frames in flight have a
`.out` without the terminal line and a stale claim; the next round lists them as pending
(a claim older than 2 h is ignored), reruns them, and skips everything finished. So the
three rounds are three identical submissions, and a fourth is the same again. Nothing is
lost, nothing is repeated, no state but the files.

If round 1's per-frame times (§4) say the campaign is shorter or longer than 8 days,
change nothing but the number of rounds.

## 4. What each log's last lines must say

Logs land in the submission directory as `openqha_hl_<stage>_<jobid>_<task>.out`.

| stage | the lines | a bad sign |
|---|---|---|
| A | `branchA: N drawn, M pending, K for task i/n -> list` at the top; one `qid rc=0 S s [PASS] …` per molecule; last: `== A  K done, 0 not done, of K in this task; wall W s` | `not done` > 0 → open `$S0_SCRATCH/branchA_<qid>.log` of that molecule; a CREST that ran out of `TIMEOUT_S` (3600) is normal for a few and is simply rerun by the next submission |
| 02 | one `qid rc=0 S s` per molecule; last: `== 02  K Frame sets written, 0 not, of K in this task; wall W s` | `not` > 0 → the molecule's `02_frames.py --species` on debug, read its Record |
| 03 | `frames  T pending, K for this task`; one `<qid> <frame> <status> <s> <MB>` per frame (`labelled` / `reused` / `refused` / `running` / `FAILED …`); `== 03 wall W s for K frames`; task 0 then `== 03 assemble`, the per-molecule table, `== 04 Dataset`, `== assemble exit 0|1` | `refused` (geometry mismatch: a rerun of A / 02 after 03 — the frame is stale, rerun the label); `FAILED` with a timeout tail (raise `--time`, or the Hessian is bigger than estimated); `MB` near 6000 (`%maxcore` exhausted: lower `CONCURRENCY`) |
| 04 | `dataset 'draw300' at <level> (split by frame): N molecules (7 test), F frames: train … valid … test … pool …; H with a Hessian`; `per class:` table; `merged …/mace_draw300.<level>.extxyz` | `pool` > 0 after the last round → frames still unlabelled: another round |

The per-frame `<s>` of the gate's 03 log, averaged over `basin_*` (Hessian) and
`displaced_*` (gradient) frames, is the measured column of §2.

## 5. Progress and resubmission

```bash
python scripts/tooling/s0_hl_progress.py --tag draw300          # seconds on the login node
```
One row per structure class and the total: molecules drawn / branch A done / Frame sets;
frames total / labelled / unlabelled / running — read from disk (`basins.done`, the Frame
set Record, the finished file groups, fresh `.running` claims), never from `squeue`. When
`unlabelled` is 0 the campaign is done; when `running` is 0 and `unlabelled` is not, no
job is working on it and a round is due. `01_select.py --tag draw300` prints the same
facts per molecule (`select.out`) with the classes column.

A killed array of any stage is resubmitted **as it is**: every stage script lists only
what is not on disk (`basins.done`, the Frame set Record, the finished ORCA job), respects
a fresh claim, takes over a stale one. There is no state to reset and no file to delete.
The one thing not to do: run two rounds of the same stage at once — they would share
the pending list and, apart from the claims, race for the same frames.

## 6. The ALF mode (parsl, for a campaign nobody wants to resubmit by hand)

The same on-disk state, driven from the login node in `tmux`: a `SlurmProvider` submits
up to `--max-blocks` one-node blocks with `sbatch` as the queue demands and releases them
as it drains (`hpc/resource_configs/tianhe_cpu.py`, role `labels`, per
`alframework/parsl_resource_configs`). The two modes can be mixed: what one labels, the
other skips.

```bash
tmux new -s hl-labels
python -u workflows/hessian_learning/03_labels.py --tag draw300 --resource tianhe_cpu \
    --max-blocks 12 --walltime 3-00:00:00 2>&1 | tee $S0_RUNS_ROOT/logs/labels_draw300_$(date +%F_%H%M).log
# Ctrl+b d detaches; tmux attach -t hl-labels returns; squeue -u $USER shows the blocks
```
