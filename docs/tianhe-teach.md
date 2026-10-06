# Tianhe teach sheet — the hessian-learning production arms

The operator-facing reference for the fine-tuning campaign on TianheXY-AI: the site
constants, the parameter tables, the Dataset line, and every production arm's launch
command in one numbered ledger (`J1`…). Written to be taught from and to be copied from:
read §1–§4 once, then use §5 as the checklist and §6 for the receipts.

**The ledger is the campaign's canonical job numbering.** Each ticket's `J1`/`J4`
([17a](../.scratch/hessian-learn-framework/implementation/17a-the-four-arms-submitted.md)),
`J1`–`J4` ([17b2](../.scratch/hessian-learn-framework/implementation/17b2-the-four-fixed-arms-submitted.md))
and `J1`/`J2` ([17b3](../.scratch/hessian-learn-framework/implementation/17b3-the-two-1colfix-arms.md))
are local to its own paste block; §5's table records each beside its ledger number so a
pasted receipt can be matched to the arm that produced it.

**What this file is not.** The label pipeline's internals and the first-login sequence
([`tianhe_runbook.md`](tianhe_runbook.md)), the campaign's command-by-command sequence
([`hessian_learning_campaign.md`](hessian_learning_campaign.md)), the round-1 history
([`branchA_production.md`](branchA_production.md)), and the per-arm acceptance battery
(ticket [17b](../.scratch/hessian-learn-framework/implementation/17b-the-six-run-battery.md)).
This sheet is the fine-tune half: what an arm is made of, and how to submit it.

**Master relationship.** The dissertation's appendices (campaign ticket 18's writing
slice) transcribe their command blocks and option tables from this file, so a change to a
knob or a command happens here first.

---

## 1. Site constants

`v100x`, one card per arm. `--gpus=1` is mandatory on the AI clusters and
`--exclusive` is banned there; `--mem` is forbidden and `-G`/`-c` belong to the
fine-grained `ai` environment, not to this partition.

| name | value | note |
|---|---|---|
| `S2` / `R` | `/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin` | the site root |
| repository | `$R/openQHA-main` | **submit from here** — `hl_train.slurm` `cd`s to the submit directory and then sources `hpc/env/common.sh` on a relative path |
| package copy | `$R/openQHA-Hessian-fd6d34f` | the capability (`fd6d34f`), fronted as `PYTHONPATH="$COPY:$PYTHONPATH"`; a code-only copy with no `.git`, so `HL_PACKAGE_COMMIT` reads `unknown` by design (ADR 0017) |
| runs root | `$R/runs` | export `S0_RUNS_ROOT` **before** sourcing `hpc/env/*.sh`, or `root.sh` picks the `/XYAIFS00` prefix and creates a second, wrong root |
| partition | `v100x` | export `OPENQHA_PARTITION=v100x` for the same reason |
| wall | `-t 72:00:00` | a K=1 arm's 60 epochs; a continuation is a submission like any other |
| job ceiling | 6 submitted / 6 running / 6 nodes | the tenant's (`hku2021_fos4`), shared with the group — so every sitting is a sequence of waves |
| mace fork | `1110ffb` | the job refuses a mace that is not the fork or whose checkout is dirty |

Two warnings appear in **every** `hl_train` job and are not defects: `pthreads OpenBLAS
under an OpenMP CREST`, and `role cpu, but no crest on PATH in 'openqha-gpu'`.

## 2. The shell header

One header per sitting; a new shell (a second wave, after a logout) needs it again.

```bash
cd $S2/openQHA-main
export R=/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin
export S0_RUNS_ROOT=$R/runs
export OPENQHA_PARTITION=v100x
export COPY=$R/openQHA-Hessian-fd6d34f
export PYTHONPATH="$COPY:$PYTHONPATH"
squeue -u $USER                            # the ceiling: what is already running
```

## 3. Parameter tables

### 3.1 The wrapper's variables — `hpc/slurm/hl_train.slurm`

Every variable is `${VAR:-default}`, so a leftover export wins silently: **pin the ones
that matter on the line** (`PROBE=`, `N_PROBES=`, `MULTIHEADS=`, `HESSIAN_WEIGHT=`).

| variable | default | reaches mace as | note |
|---|---|---|---|
| `TAG` | *required* | `--tag` | the campaign tag; the Dataset lives under it |
| `DSET` | `$TAG` | `--name` | the Dataset name. **Never `NAME`** — the site's `yhbatch` wrapper injects `NAME=yhbatch` into every job (15f) |
| `RUN` | *required* | `--run` | the run directory: `<dataset>/train/<run>/` |
| `LEVEL` | `wb97m-d3bj_def2-tzvppd` | `--level` | the label level |
| `PROBE` | `gaussian` | `--probe` | `gaussian` / `rademacher` / `cartesian`; `onehot` joins when the copy is on `PYTHONPATH` |
| `N_PROBES` | `4` | `--n-probes` | K, the probes per configuration per step; **ignored** by `cartesian` |
| `HESSIAN_WEIGHT` | `balance` | `--hessian-weight` | a number, or `balance` = `w_F·L_F/L_H` measured on the base model over the run's own train file **under the run's own protocol** |
| `ENERGY_WEIGHT` | `1.0` | `--energy-weight` | the fleet's recipe |
| `FORCES_WEIGHT` | `100.0` | `--forces-weight` | the fleet's recipe |
| `MAX_EPOCHS` | `100` | `--max-epochs` | the production arms use `60` |
| `BATCH_SIZE` | `4` | `--batch-size` | |
| `SEED` | `123` | `--seed` | mace's seed and the training probe generator's |
| `MULTIHEADS` | `0` | `--multiheads` | `1` concatenates the Replay beside the fine-tuning head |
| `PT_TRAIN_FILE` | — | `--pt-train-file` | the Replay file (drawn by `s0_spice_pt_draw.py`); **its frame count is the size knob** |
| `PT_VALID_FILE` | — | `--pt-valid-file` | the draw tool's companion `<stem>.valid.extxyz` |
| `EXTRA` | — | appended to the driver's argv | the channel for everything else — §3.3 |
| `OPENQHA_ENV` | `openqha-gpu` | — | the conda env the env files activate |

### 3.2 The driver's own flags — `workflows/hessian_learning/05_train.py`

The wrapper names the ones above; these reach the job through `EXTRA`.

| flag | default | meaning |
|---|---|---|
| `--foundation` | `S0_ENGINE` | the base potential |
| `--valid-batch-size` | `= --batch-size` | |
| `--lr` | mace's | the fleet pins `0.0001` |
| `--scheduler-patience` / `--patience` | `20` / `50` | ReduceLROnPlateau on the total validation loss / early stop |
| `--eval-interval` | `1` | |
| `--no-ema` / `--ema-decay` | EMA on / mace's | multihead mode trains at the fork's decay unless `--force-mh-ft-lr` |
| `--no-swa`, `--start-swa`, `--swa-lr`, `--swa-energy-weight`, `--swa-forces-weight`, `--swa-hessian-weight` | Stage Two on, 3/4 of the epochs, `lr/40` | the production arms are single-stage (`--no-swa`) |
| `--no-exact-anchors` | anchors on | the two full-matrix readings of the Hessian term on the validation file, ≈⅓ epoch each; the production arms keep them |
| `--mace-arg ARG` | — | repeatable, passed to mace verbatim — the two-token form (`--mace-arg=--key --mace-arg=value`) is the house form |
| `--register` / `--register-copy` | off | print the `ENGINES` entry / copy the model into `data/potentials/`. **Must ride the run** — registering afterwards retrains |
| `--no-strict-fork` | strict | do not use in production |
| `--dry-run` | off | print the `mace_run_train` argv and the Record header, then stop — the local gate's switch |

### 3.3 The `EXTRA` channel

Three variants carry the whole production campaign. The driver resolves the protocol
tokens out of `EXTRA` before it builds mace's argv, so the extras' verdict wins both at
mace's parser and in the Record (`run.protocol_overrides`).

```bash
# the round-1 recipe: registration, the re-selected knobs, persistence across a resume
EXTRA_FRESH='--register --register-copy --lr 0.0001 --no-swa --mace-arg=--clip_grad --mace-arg=1.0 --mace-arg=--weight_decay --mace-arg=0.0 --mace-arg=--ema_decay --mace-arg=0.99999 --mace-arg=--restart_latest --mace-arg=--save_all_checkpoints --mace-arg=--keep_checkpoints --mace-arg=--plot --mace-arg=False'

# + the fixed protocol's source: the frame's stored probe rows, retained throughout training
EXTRA_STORED="$EXTRA_FRESH --mace-arg=--hessian_probe_source --mace-arg=stored"

# + the one-column family (the one-hot probe). PROBE=onehot is also pinned on the line.
EXTRA_ONEHOT="$EXTRA_STORED --mace-arg=--hessian_probe --mace-arg=onehot"
```

| running arm | `EXTRA` | `PROBE` / `N_PROBES` | protocol it trains under |
|---|---|---|---|
| round-1 pair, naive pair | `$EXTRA_FRESH` | `gaussian` / `4` | stochastic gaussian, k=4 (fresh draws per step); the naive pair's extras are as in 17a's block |
| `phlfix` pair, `noH` pair | `$EXTRA_STORED` | `gaussian` / `1` | frozen stored gaussian rows, K=1 |
| `1colfix` pair | `$EXTRA_ONEHOT` | `onehot` / `1` | frozen one-hot vector (one axis per frame), K=1 |

Two protocol refusals, both fired before a run directory is created: `stored` with
`cartesian` (nothing stores the exact set), and `onehot` with K > 1 (the one-column
protocol probes one column).

### 3.4 What comes back — `train.toml`

The slim Record per run: the identity lines (`MACE_FORK_COMMIT`, `HL_PACKAGE_VERSION`,
`HL_PACKAGE_COMMIT`), one valid row per epoch with the three validation curves, and the
protocol surface below. Read it; do not assume it from the launch line.

| field | reads |
|---|---|
| `PROBE` / `PROBE_SOURCE` / `N_PROBES` | the protocol the run actually used |
| `VALID_PROBES` | the validation label (the one-hot arms carry the fresh-k1 label) |
| `ONEHOT_AXIS_N_DISTINCT` / `ONEHOT_AXIS_MAX_LOAD` | the axis spread; **written only on one-hot arms**, `-1` elsewhere |
| `HESSIAN_WEIGHT` / `HESSIAN_WEIGHT_RULE` | the resolved weight and whether it was `balance` or `given` |
| `BALANCE_L_E` / `L_F` / `L_H`, `BALANCE_PROBE`, `BALANCE_N_PROBES` | the balance's own reading, under the run's protocol |
| `MULTIHEADS`, `PT_*` | the Replay head and the file's frames / `config_weight` |

The axis rule (one-hot arms): `c = argmax_j |v0[j]|` of the frame's **first stored row**,
`v = sqrt(3N)·δ_c` — derived, never stored, frozen across steps and across a resume.
Validation is uniform for every arm: a fresh single gaussian probe per frame per pass, so
the ruler never flatters the training objective.

## 4. The Dataset line (before any arm)

The Dataset is built once per draw; [`hessian_learning_campaign.md`](hessian_learning_campaign.md)
is the authority on the sequence and the gates. The commands an arm depends on:

```bash
cd $S2/openQHA-main
export OPENQHA_PARTITION=deimos; source hpc/env/common.sh && source hpc/env/tianhe.sh

python workflows/hessian_learning/00_draw.py --tag draw300 --per-class 300 --seed 0
TAG=draw300 sbatch --array=0-1 --time=04:00:00 hpc/slurm/hl_frames.slurm            # the Frame sets
python workflows/hessian_learning/01_select.py --tag draw300                         # the task list
python -u workflows/hessian_learning/03_labels.py --tag draw300 --resource tianhe_cpu --generators basin \
    --max-blocks 12 --walltime 3-00:00:00                                           # the labels (run in tmux)
python workflows/hessian_learning/04_dataset.py --tag draw300 --name draw300_r1 \
    --split-by molecule --export openreact                                          # the Dataset the arms read
python scripts/tooling/s0_hl_progress.py --tag draw300                              # any time: what is pending

# the Replay: 30,000 SPICE train frames at each production weight (same seed, so the two
# frame sets are identical by construction and only config_weight differs)
python scripts/tooling/s0_spice_pt_draw.py --n 30000 --seed 0 --weight 1  --out $S0_RUNS_ROOT/spice/spice_pt_replay30k_w1.extxyz
python scripts/tooling/s0_spice_pt_draw.py --n 30000 --seed 0 --weight 10 --out $S0_RUNS_ROOT/spice/spice_pt_replay30k_w10.extxyz
```

## 5. The arm ledger

Fifteen production arms: six ablation cells (the round-1 pair is the paper's own row),
four fixed-protocol arms in this cut, four deferred family arms, and the full-H
comparison run. `timing1` (09g's per-epoch measurement) is deliberately not in the
ledger: it is an instrument, not a production arm, and its receipts stay outside the
grid's reconciliation. Numbers are assigned in submission order, so `J7`–`J15` are
provisional until they are submitted.

### 5.1 The table

| J | `RUN` | family / cell | `PROBE` | `SOURCE` | K | `MULTIHEADS` | `w_H` | stage shape | status | ticket label |
|---|---|---|---|---|---|---|---|---|---|---|
| J1 | `replay30k_w1` | round 1: replay+H, w1 | gaussian | fresh | 4 | 1 | balance | two-stage 30→60, `-t 72:00:00` each | running — `260950`/`260951` (which is w1 unconfirmed) | 09g's pair |
| J2 | `replay30k_w10` | round 1: replay+H, w10 | gaussian | fresh | 4 | 1 | balance | two-stage | running — as above | 09g's pair |
| J3 | `naive` | naive+H (no Replay) | gaussian | fresh | 4 | 0 | balance | single-stage 60, `-t 144:00:00` | running — `264247` | 17a's J1 |
| J4 | `naive_noH` | naive H0 | gaussian | fresh | 4 | 0 | `0` | single-stage | running — `264257` | 17a's J4 |
| J5 | `replay30k_w1_phlfix` | fixed PHL, w1 | gaussian | stored | 1 | 1 | balance | single-stage 60, `-t 72:00:00` | running — `264527` | 17b2's J1 |
| J6 | `replay30k_w10_phlfix` | fixed PHL, w10 | gaussian | stored | 1 | 1 | balance | single-stage | running — `264528` | 17b2's J2 |
| J7 | `replay30k_w1_noH` | replay+H0, w1 | gaussian | stored | 1 | 1 | `0` | single-stage | **not submitted** (two free slots) | 17b2's J3 |
| J8 | `replay30k_w10_noH` | replay+H0, w10 | gaussian | stored | 1 | 1 | `0` | single-stage | **not submitted** | 17b2's J4 |
| J9 | `replay30k_w1_1colfix` | fixed one-column, w1 | onehot | stored | 1 | 1 | balance | single-stage | **not submitted** (two free slots) | 17b3's J1 |
| J10 | `replay30k_w10_1colfix` | fixed one-column, w10 | onehot | stored | 1 | 1 | balance | single-stage | **not submitted** | 17b3's J2 |
| J11 | `replay30k_w30_1colfix` | ladder, w30 | onehot | stored | 1 | 1 | balance | single-stage | deferred (draw pending) | — |
| J12 | `replay30k_w100_1colfix` | ladder, w100 | onehot | stored | 1 | 1 | balance | single-stage | deferred (draw pending) | — |
| J13 | `replay30k_w1_phlfix4` | K=4 analogue, w1 | gaussian | stored | 4 | 1 | balance | two-stage 30→60 | deferred | — |
| J14 | `replay30k_w10_phlfix4` | K=4 analogue, w10 | gaussian | stored | 4 | 1 | balance | two-stage | deferred | — |
| J15 | *(`replay30k_w1_fullH`, name to be ruled)* | full-H comparison, outside the grid | cartesian | fresh | (3N, K ignored) | 1 | balance | two-stage, ≈6 days | planned — ≈8–9 epochs at ≈8.6× the per-epoch cost of k=4 | — |

### 5.2 The commands, group by group

**The submitted three pairs (J1–J6)** — as launched; keep for the record, do not resubmit
into an existing run directory.

```bash
# J1/J2 — round 1, two-stage: stage 1 (30 epochs) then stage 2 (60) chained on its id
JOB1=$(TAG=draw300 DSET=draw300_r1 RUN=replay30k_w1 MAX_EPOCHS=30 MULTIHEADS=1 PROBE=gaussian N_PROBES=4 HESSIAN_WEIGHT=balance EXTRA="$EXTRA_FRESH" \
    PT_TRAIN_FILE=$S0_RUNS_ROOT/spice/spice_pt_replay30k_w1.extxyz \
    PT_VALID_FILE=$S0_RUNS_ROOT/spice/spice_pt_replay30k_w1.valid.extxyz \
    yhbatch -p v100x --gpus=1 -t 72:00:00 hpc/slurm/hl_train.slurm | awk '{print $NF}')
TAG=draw300 DSET=draw300_r1 RUN=replay30k_w1 MAX_EPOCHS=60 MULTIHEADS=1 PROBE=gaussian N_PROBES=4 HESSIAN_WEIGHT=balance EXTRA="$EXTRA_FRESH" \
    PT_TRAIN_FILE=$S0_RUNS_ROOT/spice/spice_pt_replay30k_w1.extxyz \
    PT_VALID_FILE=$S0_RUNS_ROOT/spice/spice_pt_replay30k_w1.valid.extxyz \
    yhbatch -p v100x --gpus=1 -t 72:00:00 --dependency=afterany:$JOB1 hpc/slurm/hl_train.slurm
# J2: the same two lines with RUN=replay30k_w10 and the _w10 PT paths.
# 09g's block wrote -p a800x for these; the jobs actually running are on v100x (17a's read),
# and every arm since is on v100x.

# J3/J4 — the naive pair: single-stage 60, -t 144:00:00 (17a's operator ruling, which
# superseded the chain for this pair). J4 is the same line with HESSIAN_WEIGHT=0.
J=$(TAG=draw300 DSET=draw300_r1 RUN=naive MAX_EPOCHS=60 MULTIHEADS=0 HESSIAN_WEIGHT=balance EXTRA="$EXTRA_FRESH" \
    yhbatch -p v100x --gpus=1 -t 144:00:00 hpc/slurm/hl_train.slurm | awk '{print $NF}')
J=$(TAG=draw300 DSET=draw300_r1 RUN=naive_noH MAX_EPOCHS=60 MULTIHEADS=0 HESSIAN_WEIGHT=0 EXTRA="$EXTRA_FRESH" \
    yhbatch -p v100x --gpus=1 -t 144:00:00 hpc/slurm/hl_train.slurm | awk '{print $NF}')

# J5/J6 — the fixed PHL pair (frozen stored gaussian, K=1). J6: the same line with
# RUN=replay30k_w10 and the _w10 PT paths.
J=$(TAG=draw300 DSET=draw300_r1 RUN=replay30k_w1_phlfix MAX_EPOCHS=60 MULTIHEADS=1 PROBE=gaussian N_PROBES=1 HESSIAN_WEIGHT=balance EXTRA="$EXTRA_STORED" \
    PT_TRAIN_FILE=$S0_RUNS_ROOT/spice/spice_pt_replay30k_w1.extxyz \
    PT_VALID_FILE=$S0_RUNS_ROOT/spice/spice_pt_replay30k_w1.valid.extxyz \
    yhbatch -p v100x --gpus=1 -t 72:00:00 hpc/slurm/hl_train.slurm | awk '{print $NF}')
```

**J7/J8 — the `noH` pair.** Same lines as J5/J6 with `HESSIAN_WEIGHT=0` (a zero weight
keeps the term's cost, so each pair differs in one knob only) and the `_noH` run names.

**J9/J10 — the one-column pair.** The first `onehot` jobs on site: the family token rides
`EXTRA` and `PROBE=onehot` is pinned so the wrapper's own echo (`probe onehot k=1 …`,
printed in the job's first seconds) is truthful. `N_PROBES=1` is mandatory.

```bash
J=$(TAG=draw300 DSET=draw300_r1 RUN=replay30k_w1_1colfix MAX_EPOCHS=60 MULTIHEADS=1 PROBE=onehot N_PROBES=1 HESSIAN_WEIGHT=balance EXTRA="$EXTRA_ONEHOT" \
    PT_TRAIN_FILE=$S0_RUNS_ROOT/spice/spice_pt_replay30k_w1.extxyz \
    PT_VALID_FILE=$S0_RUNS_ROOT/spice/spice_pt_replay30k_w1.valid.extxyz \
    yhbatch -p v100x --gpus=1 -t 72:00:00 hpc/slurm/hl_train.slurm | awk '{print $NF}')
echo "J=$J   # replay30k_w1_1colfix"
```

**J11/J12 — the ladder.** One draw per weight first (≈4 min each; 09d measured 242.4 s /
222.5 s for the w1/w10 pair), then the same line with `_w30` / `_w100`:

```bash
python scripts/tooling/s0_spice_pt_draw.py --n 30000 --seed 0 --weight 30  --out $S0_RUNS_ROOT/spice/spice_pt_replay30k_w30.extxyz
python scripts/tooling/s0_spice_pt_draw.py --n 30000 --seed 0 --weight 100 --out $S0_RUNS_ROOT/spice/spice_pt_replay30k_w100.extxyz
```

**J13/J14 — the K=4 analogue.** `EXTRA_STORED` with `N_PROBES=4` (four frozen gaussian
rows) on the fleet's two-stage 30→60 chain, since the K=4 per-epoch cost is the k=4
fleet's.

**J15 — the full-H comparison.** The exact eq. 1′ loss via `--probe cartesian`: the
training term becomes exact at 3N HVPs, no new code. Budget-bound, stage-chained, and
**outside the grid** — its receipts do not enter the grid's reconciliation:

```bash
TAG=draw300 DSET=draw300_r1 RUN=replay30k_w1_fullH MAX_EPOCHS=30 MULTIHEADS=1 PROBE=cartesian HESSIAN_WEIGHT=balance EXTRA="$EXTRA_FRESH" \
    PT_TRAIN_FILE=$S0_RUNS_ROOT/spice/spice_pt_replay30k_w1.extxyz \
    PT_VALID_FILE=$S0_RUNS_ROOT/spice/spice_pt_replay30k_w1.valid.extxyz \
    yhbatch -p v100x --gpus=1 -t 72:00:00 hpc/slurm/hl_train.slurm
```

## 6. Receipts

**At submission** — the `J=` echo (the `awk` captures the id; the literal `Submitted batch
job <id>` line is consumed by the pipe) and `squeue -u $USER`. Under the six-job ceiling,
submit in waves and never assume four free slots.

**Ten minutes in** — the only read that can see a job which died in seconds while still
showing in `squeue`, and the only one that reads the wrapper's own echo:

```bash
squeue -u $USER
grep -H '^tag' logs/slurm/openqha_hl_train_*.out       # `tag draw300 dataset draw300_r1 run <RUN> level …`, or an error tail
grep -H 'probe onehot' logs/slurm/openqha_hl_train_*.out   # one-hot arms: the balance finished and the protocol took
```

**Three hours in** — each arm's first-epoch wall, which decides the continuation lines
(`afterany` may be appended to a job any time before it ends; the resume reads the
per-epoch checkpoints):

```bash
grep -H 'epochs .* in .* s' logs/slurm/openqha_hl_train_*.out
```

**Per-arm acceptance** — the battery of ticket 17b, one invocation per arm over that
arm's `train.toml` and logs, with the raw output pasted back. **Per-arm judgement** — the
shipped-Hessian ruler, once per arm once its model is registered:

```bash
python workflows/hessian_learning/06_judge.py --tag draw300 --engine <registered model> \
    --spice-file data/training_sets/spice_test_5000.extxyz
```

## 7. What a good arm reads like

- The Record's `HESSIAN_WEIGHT_RULE` is what the arm was launched with — `balance` for
  J1, J2, J3, J5, J6, J9–J15; `given` with `HESSIAN_WEIGHT` exactly `0` and an all-zero
  balance trio for J4, J7, J8.
- Both exact anchors are present; the ordering `0 < AFTER < BEFORE` is relaxed on the H0
  arms, which are allowed to get worse on the anchor.
- `MULTIHEADS` matches the grid, and the Replay fields (`PT_*`) are present only on the
  replay arms.
- The three fresh-k4 balance arms (J1/J2/J3) resolve the **same** `w_H`; the two
  `phlfix` arms form a cohort of their own equal to each other; the two `one-column` arms
  form a third. The dbg gates read `8.170329761632427` (fresh k=4), `7.777607775366844`
  (`phlfix`) and `9.049569935920484` (`1colfix`) on `draw300_r1dbg` — **the production
  values are whatever the production Records say**, and a last-digit difference between
  arms on different cards is a device artifact, not a failure.
- The one-hot arms write both axis readings, and no other arm does. The dbg gate read
  **37 distinct axes, max column load 5.4 % over 92 training frames**; the production
  reading is a spread statement, not a threshold.
- `HESSIAN_CURVE_MOVED` true, and non-degenerate validation curves on all three terms.

## 8. Traps

- **Under the ceiling**: 6 jobs total, shared with the group. A continuation competes for
  the same slots; a freed slot stays free.
- **A stored arm whose Dataset lacks the stored rows dies in minutes** inside
  `stored_probes` (the balance, the arm's first act) — leaving an empty run directory.
  Check the `ls` before any resubmit: nothing may already exist at the run name.
- **A job that dies in the balance does not show in a bare `squeue` read as dead** — the
  ten-minute re-read is what catches it.
- **Exports do not survive a logout**: the header, `EXTRA`, and the replay paths must be
  re-set in a new shell.
- **Submit from `$S2/openQHA-main`** and pass the Dataset as `DSET`, never `NAME`.
- **Local dry-runs**: never run Python from the workspace root (the sibling `mace/`
  shadows the editable install as an empty namespace package — the fork identity then
  reads a false `unknown`). `cd` into a repository root first.
