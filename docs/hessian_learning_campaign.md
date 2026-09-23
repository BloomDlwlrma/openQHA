# The Hessian-learning campaign on Tianhe — the page it is run from

The sequence, the cost, what each log must say, how a killed job is resubmitted, and the
one table that says where the campaign stands. Site facts (partitions, filesystems,
install) are in [`tianhe_runbook.md`](tianhe_runbook.md); what each step computes is in
[`../workflows/hessian_learning/README.md`](../workflows/hessian_learning/README.md).
Everything below is TianheXY-C (`debug` for the gates, `deimos` for the campaign; one
node = 64 cores, 512 GB). Numbers are marked **[measured]** or **[estimate]**.

**Before anything: the geometries.** A molecule of the draw starts from its curated QM9
file, and `data/qm9/` is not in git — the first gate on 2026-09-21 failed on all 16
molecules with `reference geometry not found … curatedQM9 at None`. Do not copy 133 661
files onto Lustre; copy the ONE archive (ticket 17):

```bash
# on the workstation, once (~20 min): one group per molecule into one HDF5, verified by read-back
python scripts/tooling/s0_pack_curated_qm9.py
scp data/qm9/curated_qm9.h5 tianhe:~/openQHA-main/data/qm9/
# on tianhe: `curated_qm9.find()` extracts a molecule into data/qm9/_h5cache/ the first time
# it is asked for -- 6,458 files for the campaign, not 133,661
```

The rules this page follows (rulings 2026-09-18 … 22): nothing runs on the login node
except the second-long steps and, for step 03, the parsl **driver** in `tmux` — a process
that polls `squeue` and submits blocks, no chemistry; every test is a `debug` job; a job
submitted by hand is plain bash + `xargs`, no parsl inside it (the driver's blocks are
parsl's own one-node jobs); one campaign is one tag (`TAG=draw300`, the Dataset has the
same name); the molecule tree is flat (`<root>/draw300/<qid>/`), a frame's ORCA files are
`frames/orca.<level>.<frame>.*`; a frame is attempted once (§3).

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

# the campaign, six commands, each stage sized to its cost (§2) and started by hand when the
# one before is done (s0_hl_progress says when; every stage lists its own pending work from
# disk, so a wrong order only runs empty)
TIMEOUT_S=14400 TAG=draw300 sbatch --array=0-11 --time=1-00:00:00 hpc/slurm/hl_branchA.slurm   # 1  A: 12 submissions, ~10 h
TAG=draw300 sbatch --array=0-1 --time=04:00:00 hpc/slurm/hl_frames.slurm                        # 2  02: 2 submissions, ~1 h, after A
python workflows/hessian_learning/01_select.py --tag draw300                                     # 3  01: select.dat, seconds, after 02
tmux new -s hl-labels                                                                            # 4  a session the driver survives logout in
python -u workflows/hessian_learning/03_labels.py --tag draw300 --resource tianhe_cpu \
    --max-blocks 12 --walltime 3-00:00:00 2>&1 | tee $S0_RUNS_ROOT/logs/labels_draw300_$(date +%F_%H%M).log
                                                                                                 # 5  03: the parsl driver, <= 12 blocks of 3 days, ~8 days
python workflows/hessian_learning/04_dataset.py --tag draw300 --split-by molecule --export openreact   # 6  04: after the driver ends
python scripts/tooling/s0_hl_progress.py --tag draw300                                           # any time, seconds
```

(inside the tmux session, before command 5: `export OPENQHA_PARTITION=deimos; source hpc/env/common.sh && source hpc/env/tianhe.sh` — a new tmux shell is bare.)

**Rebuilding the Frame sets (ticket 28, 2026-09-23).** The displaced draw is the scale of
the judge's diagnostics (msRRHO reads basins, the training set is basin frames), so one
campaign carries one draw. When it changes — it did on 2026-09-23, from the classical
equipartition Gaussian to ANI-1's bounded coefficient randomisation — every Frame set of the
draw is rebuilt rather than mixed:

```bash
FORCE=1 TAG=draw300 sbatch --array=0-11 --time=1-00:00:00 hpc/slurm/hl_frames.slurm
```

`FORCE=1` makes the list stop skipping molecules that already have a `frames.toml` and passes
`--force` to `02_frames.py`, so the full ~25 min/molecule is paid again (1,250 rebuilt +
~4,000 still to build ≈ 1 day on 12 nodes). Reference labels already computed on displaced
frames of the old draw would be orphaned — check `s0_hl_progress` for labelled displaced
frames before starting one.

**Why this shape and not a chain of arrays** (ruling 2026-09-22, round 11). The tenant's
quota on TianheXY-CN, read off the portal on 2026-09-21:

| | submitted jobs | nodes | running jobs |
|---|---:|---:|---:|
| this user | 32 | unlimited | unlimited |
| tenant `hku2021_fos4` (shared by the group) | **32** | **32** | **32** |

Slurm counts **every array task as a submission**: `--array=0-11` pending is 12 of the 32.
Three pre-queued rounds of labels (36) are refused outright (`AssocMaxSubmitJobLimit`), and a
`--dependency` chain of 12-node arrays holds 26 of the group's 32 for days, most of them
idle in `PD (Dependency)`. So: branch A is one array (12; ~10 h, no rounds needed), 02 is two
nodes (~108 core-h in all), and step 03 — the only stage that needs rounds — is driven by
parsl from the login node: `init_blocks=0, min_blocks=0, max_blocks=12` in
`hpc/resource_configs/tianhe_cpu.py` (role `labels`, 16 × 4 per node), a block is submitted
only while frames are waiting and released as the queue drains, a block that dies at its
3-day time limit is replaced while frames remain, and the frames it had in flight are rerun
whole (§3) on another block within a minute. The queue never holds more than 12 of this
campaign's jobs. The chain-job alternative — `--signal=B:USR1@900` and a `trap` that
resubmits the array from inside, plus `01_select` inside `hl_labels.slurm` — was designed
and set aside: parsl already does both.

**The tmux gate, five minutes, before command 5.** Three things about this route have never
run on TianheXY-CN; one debug frame measures all three:

```bash
command -v tmux || echo no-tmux          # no tmux: `screen -S hl-gate` (Ctrl+a d / screen -r) does the same job
tmux new -s hl-gate
export OPENQHA_PARTITION=deimos; source hpc/env/common.sh && source hpc/env/tianhe.sh
python -u workflows/hessian_learning/03_labels.py --tag draw300 --resource tianhe_cpu --debug --limit-frames 1
```

| what the gate shows | what it means |
|---|---|
| `squeue -u $USER` has one `parsl.*` block on `debug`; within minutes one `<qid> <frame> labelled <s> <MB>` line; the driver exits 0 | the route works: the block was submitted, the compute node's HTEX worker reached the driver's interchange on the login node, a label came back — run command 5 |
| the block runs but the driver sits at the frame list for > 5 min, then the block ends with nothing labelled | the worker could not connect back: `tianhe_cpu.config` sets no `address=` and HTEX guessed the login node's hostname, which on a multi-homed login node with the proxy set-up may not be the interface the compute nodes reach. Fix, one line after the measurement: `HighThroughputExecutor(address=address_by_interface("<nic>"), …)` in `tianhe_cpu.py`, the interface from `ip -4 addr` on the login node that the compute network routes to — then the gate again |
| `sbatch` refused (`AssocMaxSubmitJobLimit`, partition) | the group is at its 32; wait or ask, nothing to fix here |

**Living with the driver.** `Ctrl+b d` leaves the session (the driver keeps running);
`tmux attach -t hl-labels` returns; `Ctrl+b [` scrolls the log (`q` back). To stop it:
attach, `Ctrl+C` — parsl cancels its blocks on the way out — then `squeue -u $USER` and
`scancel` anything named `parsl.*` that is left. `tmux kill-session -t hl-labels` from
outside kills the driver without that clean-up. A driver that died (a login-node reboot,
a kill) is simply started again with command 5: it lists the frames without an ORCA job,
skips finished and failed ones, and takes over the frames whose blocks Slurm calls dead
(§3). The progress table (§5) is the same in either route. The tenant has several login
nodes; a `tmux ls` that shows nothing may mean the session lives on another one
(`pgrep -u $USER tmux` there).

The gate's labels job on 19-atom molecules is the first real measurement of the
Hessian and gradient job times (§2): if its table shows `FAILED` with an ORCA timeout
tail, the Hessian did not fit in 30 minutes — rerun that gate as
`TAG=draw300 LIMIT_FRAMES=16 sbatch --time=03:00:00 hpc/slurm/hl_labels.slurm` on `deimos`.

## 2. The cost

**The layout, and what actually occupies the 64 cores** (ruling 2026-09-21: the jobs
declare what they need and derive the rest). Every stage script asks for the whole node —
`--nodes=1 --exclusive --ntasks=1 --cpus-per-task=64 --mem=0` — reads back what Slurm
granted (`SLURM_CPUS_PER_TASK`, captured **before** the loop that unsets `SLURM_*` for
ORCA's mpirun) and sets `CONCURRENCY = CORES / per-worker cores`, so the same script fills
a 32-core node with 8 × 4 instead of failing:

| stage | per worker | on 64 cores | what is busy |
|---|---|---|---|
| A branch A | 4 (CREST `-T 4`) | 16 × 4 | 64 during CREST — the metadynamics runs are the one thread axis `common.sh` leaves open; 16 during the MACE relax that follows it (single-threaded) |
| 02 frames | 1 (MACE on the CPU) | **64 × 1** | 64. Before 2026-09-21 this job pinned 4-core ranges and ran 16 workers — 48 cores idle, the stage four times slower than it needed to be |
| 03 labels | 4 (ORCA MPI ranks, `%pal nprocs 4`) | 16 × 4 | 64, and `%maxcore 6000` per rank keeps 64 × 6 GB = 384 GB under the node's 512 |

`nproc` is not a way to ask how many cores a job has: it honours `OMP_NUM_THREADS`, which
`common.sh` sets to 1, and it answered **1** on a 64-core node on 2026-09-20 — the guard
in the branch A and labels scripts then refused the job. `SLURM_CPUS_PER_TASK` (or
`nproc --all` off a cluster) is the answer. 12 nodes = 768 cores. The draw: 6,458 molecules, 83 %
with 9 heavy atoms (≈ 19 atoms); a molecule's Frame set is ~3 basins × (1 + 4 displaced)
plus merged / saddle frames ≈ 15–17 frames, of which 3–5 are Hessian jobs (basin /
merged / saddle: `EnGrad Freq`) and ~12 gradient jobs (displaced: `EnGrad`; round 5 Q7 b).

| stage | per unit | basis | draw300 (12 nodes) | measured on the campaign |
|---|---|---|---|---|
| A branch A | 283–1,493 s / molecule at 16 per node, 3 of 16 still running at the 30-min debug cap **[measured 2026-09-21, debug gate on draw300, 9 heavy atoms; the smoke set gave 460–590 s on 2026-09-19]** | 6,458 × ~1,000 s × 4 cores ≈ 7,200 core-h | **~10 h** (array `--time=1-00:00:00` is slack; a molecule past `TIMEOUT_S=3600` is skipped with rc≠0 and stays pending) | — |
| 02 frames | ~1 min / molecule **[estimate from the smoke set]** | 6,458 × 60 s × 1 core = 108 core-h | **< 15 min** (64 × 1 since 2026-09-21) | — |
| 03 Hessian jobs | 10 atoms: 245–305 s **[measured]**; 19 atoms: 40–80 min **[estimate, N³–N⁴ scaling]** | 6,458 × 4 × 60 min × 4 cores = 103,000 core-h | 5.6 days | — |
| 03 gradient jobs | 19 atoms: 3–5 min **[estimate]** | 6,458 × 12 × 4 min × 4 = 20,700 core-h | 1.1 days | — |
| 03 total | | **≈ 125,000–155,000 core-h** | **≈ 7–8.5 days** → 3 rounds of 3 days | — |
| 04 Dataset | seconds per molecule, login node or task 0 | | minutes | — |

The formula behind the 03 rows, for N molecules with F frames each at T per frame on
P cores: `N × F × T × P / 768` hours of 12 nodes. The gate's log (§4) gives T for both
job kinds on the draw's own molecules; write them into the last column; the driver's
first blocks (§1, command 5) fill the campaign column, and if the Hessian is over 80 min
the driver simply runs longer — there is no round count to change.

Storage, kept (no `.gbw`, `.loc`, `property.txt`; round 5 Q8): ~0.35 MB per frame
(`inp` + full `out` + `hess` or `engrad`; a 19-atom `.hess` is ~0.15 MB) → **~35 GB** for
~100,000 frames; ~1 MB per molecule of branch A / Frame sets → 6.5 GB; the Dataset:
~7 GB of extxyz per level per 100,000 frames (three split files + the merged one =
twice that) and 2.6 GB of HDF5. Node-local scratch holds ORCA's integrals during a job
(GB per analytic Hessian) and is removed with the run directory.

## 3. The frames' states; the sbatch rounds as the fallback route

**The fallback: labels as sbatch arrays.** If the tmux gate (§1) fails and the fix is not at
hand, step 03 runs as rounds of the array script — the same on-disk state, the same
per-frame worker, no parsl:

```bash
TAG=draw300 sbatch --array=0-11 --time=3-00:00:00 hpc/slurm/hl_labels.slurm     # a round; the next when this one has ENDED
```

`deimos` allows 7 days, but a 3-day walltime queues faster and loses at most 3 days of one
node to a crash; ~8 days of labels are three rounds. Each of the 12 tasks lists the PENDING
frames of the selection at its start (`03_labels.py --list`: no ORCA job on disk, not held
by another process), takes every 12th of them (`SLURM_ARRAY_TASK_ID`), and runs them 16 at a
time under `taskset`. At the end task 0 assembles every molecule's label files and builds
the Dataset; its exit code is **0 when no frame is without an ORCA job, 1 otherwise** — the
signal for the next round, which is the same command again, until task 0's assemble exits 0. The rounds are submitted one
at a time, by hand, each when the previous has ended: pre-queued they would take 36 of the
tenant's 32 submissions (§1), and a `--dependency` chain would hold them idle for days. The
two routes can even overlap — the lock below partitions the frames between them.

**A frame's three states, and the lock** (ticket 24, round 11, S0-G-96). Every frame is
attempted **once**, bounded by `TIMEOUT_S` (8 h per ORCA job, `hl_labels.slurm`; 3× the
19-atom analytic Hessian's estimated upper end):

| on disk | state | the next round |
|---|---|---|
| `<stem>.out` with `****ORCA TERMINATED NORMALLY****` and the `.hess` / `.engrad` | **finished** | skipped |
| `<stem>.out` without that line (a crash; or the `TIMEOUT_S` kill, whose `.out` ends with `openQHA: ORCA killed after TIMEOUT_S=28800 s`) | **failed** | **not rerun** — the reader judges from the worker's `FAILED` line in the Slurm `.out`, the `.err`, and the ORCA `.out`; `python -m openqha.data.frame_labels <molecule> <generator> <basin> <k> --retry` is the human's rerun |
| no `<stem>.out` | never run, or **cut** (walltime, SIGKILL, a dead node: the node-local run directory is gone, nothing came back) | rerun whole — nothing resumes, ORCA's `.gbw` never leaves the node |

While ORCA runs the frame is held by `frames/<stem>.running`, which names the Slurm job
and is touched every minute by the worker. Another process treats the frame as held only
while **Slurm does not call that job dead** (`squeue -j`: COMPLETING or any terminal state
frees it at once; no answer = no opinion) **and the lock was touched within 30 min** (a
worker that died alone inside a living job stops touching). A walltime kill's SIGTERM is
turned into an exception inside the worker, so the lock is released on the spot and the
frame in flight leaves no `.out`. Two rounds or a round and the tmux driver (§6) can
therefore overlap without labelling the same frame twice, and a block that dies at its
time limit hands its frames to the next one within a minute.

If round 1's per-frame times (§4) say the campaign is shorter or longer than 8 days,
change nothing but the number of rounds.

## 4. What each log's last lines must say

Logs land in `logs/slurm/` of the checkout (the directory the job was submitted from) as `openqha_hl_<stage>_<jobid>_<task>.out`; a job submitted without `--array` gets Slurm's "no task" number, `_4294967294`. The directory ships with the repository (`.gitkeep`), its contents are git-ignored.

| stage | the lines | a bad sign |
|---|---|---|
| A | `branchA: N drawn, M pending, K for task i/n -> list[; F failed earlier]` at the top; one `qid rc=0 S s CREST reports C conformers … -> B basins` per molecule; last: `== A  K done, 0 not done, of K in this task; wall W s` | two kinds of `rc=1` (measured 2026-09-22: ~4 % + ~5 % of draw300): **(i)** `terminated EARLY N; shake used 1; fell back True … all criteria passed: False … written branchA.toml` — the published SHAKE=2 run lost some metadynamics, the SHAKE=1 retry (or, since ticket 26, the published run itself when the retry crashed) supplied the ensemble; criterion 10 records it; the molecule is **done**; **(ii)** `FileNotFoundError: CREST produced no ensemble` — neither attempt left an ensemble; `_records/branchA.failed` is written (reason, both attempts' last lines, the rerun command), the molecule is **not rerun** by any round and shows in the progress table's `A failed` column; a human reads the marker and reruns `s0_A_pipeline.py --species <qid> --tag draw300` by hand (a success clears the marker). `not done` > 0 also for a molecule cut by the walltime — the next submission redoes it |
| 02 | one `qid rc=0 S s` per molecule; last: `== 02  K Frame sets written, 0 not, of K in this task; wall W s` | `not` > 0 → the molecule's `02_frames.py --species` on debug, read its Record |
| 03 | `frames  T pending, K for this task`; `timeout    ORCA per frame TIMEOUT_S=28800 s`; one `<qid> <frame> <status> <s> <MB>` per frame (`labelled` / `reused` / `refused` / `failed` (earlier, not rerun) / `running` / `FAILED …` (now)); `== 03 wall W s for K frames`; task 0 then `== 03 assemble`, the per-molecule table with its `failed` count, `== 04 Dataset`, `== assemble exit 0|1` | `refused` (geometry mismatch: a rerun of A / 02 after 03 — the frame is stale, rerun the label); `FAILED` whose ORCA `.out` ends with the `TIMEOUT_S` trailer (the Hessian is bigger than estimated: read it, `--retry` with a larger `TIMEOUT_S` if it deserves one); `MB` near 6000 (`%maxcore` exhausted: lower `CONCURRENCY`) |
| 04 | `dataset 'draw300' at <level> (split by frame): N molecules (7 test), F frames: train … valid … test … pool …; H with a Hessian`; `per class:` table; `merged …/mace_draw300.<level>.extxyz` | `pool` > 0 after the driver ended → frames still without a job (its blocks were cut, or the driver died): run command 5 again; frames the Batch table calls `failed` stay out until a human `--retry`s them |

The per-frame `<s>` of the gate's 03 log, averaged over `basin_*` (Hessian) and
`displaced_*` (gradient) frames, is the measured column of §2.

## 5. Progress and resubmission

```bash
python scripts/tooling/s0_hl_progress.py --tag draw300          # seconds on the login node
```
One row per structure class and the total: molecules drawn / branch A done / Frame sets;
frames total / labelled / **failed** / unlabelled / running — read from disk (`basins.done`,
the Frame set Record, the finished file groups, the `.out`s without the terminal line, the
`.running` locks — `squeue` is asked only about the job a lock names). When `unlabelled`
is 0 the campaign's computing is done and `failed` is the list a human reads (§3's table);
when `running` is 0 and `unlabelled` is not, no job is working on it: the driver has ended
or died — start it again (§1, command 5), or submit a round of the fallback (§3).
`01_select.py --tag draw300` prints the same facts per molecule (`select.out`) with the
classes column.

A killed array of any stage is resubmitted **as it is**, and a dead driver is started
again as it was: every stage lists only what is not on disk (`basins.done`, the Frame set
Record, the ORCA job), leaves a held frame to its holder and takes over a dead one (§3).
There is no state to reset and no file to delete. Two rounds of the same stage at once, or
a round beside the driver, are safe but pointless: the locks partition the frames between
them.

## 6. The ALF mode

Is the campaign's own route for step 03 since 2026-09-22 and lives in §1 (commands 4–5,
the tmux gate, living with the driver): a `SlurmProvider` submitting up to `--max-blocks`
one-node blocks as the queue demands and releasing them as it drains
(`hpc/resource_configs/tianhe_cpu.py`, role `labels`, per `alframework/parsl_resource_configs`).

## 7. After the labels: the fine-tune (one A800, not tianhe's CPU nodes)

The labels feed ONE training row (S0-C-60): **R4** -- Replay = 4 × the train frames that carry a
Hessian at `config_weight = 10`, `w_H` = the epoch-0 balance the driver measures by default,
the gate closed (every judge row reported, `VERDICT = REPORTED`). The five commands, with what
each needs from this page, are the block "The production row R4, end to end" of
[`../workflows/hessian_learning/README.md`](../workflows/hessian_learning/README.md):
`04_dataset` (prints `N_TRAIN_HESSIAN` and `REPLAY_R4_FRAMES`) → the two SPICE draws
(`s0_spice_test_draw.py --n 5000`, `s0_spice_pt_draw.py --n <REPLAY_R4_FRAMES> --weight 10`;
the SPICE release is on tianhe, the draws are minutes on the login node) → `hl_train.slurm`
with `TAG=draw300 RUN=R4` on the A800 partition → `05_train.py --register-copy`, then branch A
and the msRRHO `mace` / `compare` steps for the pinned seven with `S0_ENGINE=MACE-OFF23_medium-R4
--tag r4` → `06_judge.py --engine MACE-OFF23_medium-R4 --thermo-tag r4`. The smoke set of §1's
gate (one day CREST + one day ORCA) runs the same five steps first with `--tag smoke` and
`MAX_EPOCHS=20`; its Record's `SECONDS_PER_EPOCH` sets R4's walltime. The Dataset for R4 is
built with the by-frame split and basin frames only (`--train-generators basin`, the default):
command 6 of §1 as written (`--split-by molecule`) is the smoke set's mode and holds out whole
molecules, which is not R4's split -- use `--split-by frame` for the campaign Dataset.
