# Workflow `hessian_learning`: the E‑F‑H Dataset for fine‑tuning MACE

Numbered Batches that turn a molecule list into a Dataset of
frames, each carrying the engine's and the reference's energy, forces and Cartesian
Hessian at the same geometry. Each step writes Records into the molecule directories it
touches; the Slurm log is the Batch's report.

| step | driver | Calculation | writes |
|---|---|---|---|
| 00 | `00_draw.py` | `openqha.data.structure_classes.draw` | `<root>/<tag>/_datasets/<name>/draw.{out,toml,dat}` — the campaign's molecule list: 500 per structure class (`configs/structure_classes.yaml`) from the gated QM9 targets outside MACE-OFF23's SPICE training file, the union over classes, the pinned seven always in |
| 01 | `01_select.py` | `openqha.data.dataset.select` | `<root>/<tag>/_datasets/<name>/select.{out,toml,dat}` — the molecule list (the whole draw, when there is one) with stratum, classes, SPICE membership, pin status, basins / frames present |
| 02 | `02_frames.py` | `openqha.data.frames.generate` | `<molecule>/frames/<generator>.<mace level>.extxyz`, `frames/frames.{out,toml}` |
| 03 | `03_labels.py` | `openqha.data.frame_labels.label_one` per frame, `assemble` per molecule | `<molecule>/frames/orca.<level>.<generator>_bBB_kK.{inp,out,hess,engrad}`, `frames/<generator>.<level>.extxyz`, `frames/labels.<level>.{out,toml}` |
| 04 | `04_dataset.py` | `openqha.data.dataset.build` (+ `export_openreact`) | `<root>/<tag>/_datasets/<name>/{train,valid,test,pool}.<level>.extxyz` (REF_* keys), `mace_<name>.<level>.extxyz`, `index.dat` (classes), `dataset.{out,toml}` ([[Class]]), `molecules-<name>.h5` |
| 05 | `05_train.py` | `openqha_hessian.run.run_training` → `mace.cli.run_train.run` (the fork) | `<root>/<tag>/_datasets/<name>/train/<run>/` — mace's own files (`<run>.model`, `checkpoints/`, `logs/`, `results/`) plus `config.yaml` and the Record `train.{out,toml,dat}` (settings, epoch table, the foundation and fine-tuned weight files, the config SHA, the mace fork's commit) |
| 06 | `06_judge.py` | `openqha_hessian.judge.run` | `<root>/<tag>/_datasets/<name>/judge/<run>/judge.{out,toml,dat}` — per frame, per structure class and per distribution; the entropy tier read from the msRRHO Records; the forgetting line on a fixed SPICE draw; one PASS / FAIL line per threshold |

`run.sh --tag T [--tag T2] [--name NAME] [--limit N] [--stratify] [--with-labels]` runs
01 → 02 → (03 with `--with-labels`, else skipped) → 04 here; the Dataset lives under the
first tag's `_datasets/<name>/`.

**One campaign = one tag = one Dataset**: every step's
`--name` and every stage script's `NAME` default to the tag, so `TAG=draw300` alone is the
whole address — `<root>/draw300/<qid>/` for the molecules (the tag directory is flat; the
earlier shard layers are gone),
`<root>/draw300/_datasets/draw300/` for the Dataset. `--name` is only for a second subset
on the same tree. A frame's ORCA files are a FILE GROUP of the molecule directory,
`orca.<level>.<generator>_bBB_kK.{inp,out,hess,engrad}` (`layout.orca_frame_stem`), not a
directory two levels down; the fields are joined by `.` because the level and the frame tag
contain `_`; they live in `frames/` beside the Frame sets they label. The msRRHO study has
its own sub‑folder `msrrho/` (`orca.<level>.basinNN.*` file groups, `thermo/` Records,
`crest_entropy/`, `xtb/`); a molecule directory is `crest/ mace/ md_*/ frames/ msrrho/ _records/`.

## The frame recipe

Seeds are branch A's MACE basins (CREST on GFN2‑xTB with `refine = "sp"`, MACE
tightening to fmax 1e‑4, MACE analytic Hessian). Four generators: `basin` (the minimum),
`displaced` (4 draws per basin from the **classical** harmonic distribution at 298.15 K
along the basin's modes, ⟨q²⟩ = k_BT/ω², RMS ≤ 0.15 Å), `merged` (a
CREST conformer branch A merged into a basin), `saddle`. Seeds are
`SHA‑256(qm9_index|basin|generator|k)`. The displaced frames are drawn by **normal‑mode
sampling at 450 K** (a random partition of at most (3/2) N_a k_B T over the basin's modes,
random signs; bounded in energy, no RMS ceiling). The only filter is the energy window
(275 kcal/mol above the basin); a changed bond graph is reported, never a reason to drop.

Beside ours, for the record: **SPICE** draws 10 RDKit conformers → 500 K OpenFF MD
100 ps → 25 max‑min‑RMSD hot + 25 cooled frames, cut at 1e4 kJ/mol; **ANI‑1** displaces
along normal modes at 450 K for 8‑heavy‑atom molecules (2000 K for one heavy atom), mean
¾ N_a k_BT per frame, 275 kcal/mol window; **OpenREACT** trains on stationary points and
tests on IRC and normal‑mode frames. None checks bond graphs.

## Step 03: reference labels

Per frame ORCA runs `! <level single point> EnGrad Freq` (analytic Hessian; `NumFreq`
for a level without one) **at the frame's fixed geometry** — a single point, never an
optimisation: the label of a displaced frame is the raw Cartesian Hessian there, gradient
term included, which is what the loss compares to the engine's.

**The D1 guard (2026-09-25): never optimise a frame before labelling it.** The
same-method rule that governs the census and the msRRHO thermochemistry —
interpret frequencies only at a stationary point of the method that produced them
— is a rule about interpreting a frequency, and it does NOT reach a curvature
label at a fixed geometry. Labels are computed at the frame's own geometry and are never
optimised; training rows are stationary basin frames, the off-stationary generators
(`displaced`, `merged`, `saddle`) are held out (Rodriguez 2025; PHL's NMS
rationale). "Fixing" a displaced frame by relaxing it first would destroy the raw
off-stationary curvature the loss trains on and invalidate the held-out rows.

Units are converted once (Eh → eV, Eh/bohr → eV/Å, Eh/bohr² → eV/Å²); positions in the labelled file are
the MACE file's verbatim, and a `.hess` geometry whose shape is more than 1e‑7 Å away is refused (ORCA writes the `.hess` in the centre‑of‑mass frame; the translation is removed and reported; the residual is ORCA's print precision, ~1e‑8 Å). The
full `.out` of every job is kept.

**One attempt per frame**: an ORCA job that did not terminate normally
— a crash, or the `TIMEOUT_S` kill (8 h; its `.out` ends with an `openQHA: ORCA killed
after TIMEOUT_S=…` line) — leaves a **failed** frame; the reader judges from the `.out`
and the worker's `FAILED` line. A failed frame is re‑attempted at most **once**, and
**every round carries the failures without an archive by default**: the array round's task list includes
them (its 5th column `retry`/`-` tells the worker to pass the CLI's `--retry`), while
`RETRY_ONLY=1` makes a round that sweeps **only** those failures; the failed `.out` is
renamed to `<stem>.failed.out` before ORCA overwrites it — the one archive slot, replaced
each time it is written, and the durable marker that makes a retried‑and‑failed frame
final. The per‑frame `python -m openqha.data.frame_labels <molecule> <generator> <basin>
<k> --retry` is the human's lever (refused once the archive exists); `--force` is the
deliberate full relabel, never a round. A frame **cut** before
anything came back (walltime, a dead node) has no `.out` and is rerun whole; nothing
resumes. While ORCA runs the frame is held by `frames/<stem>.running` — the lock names its
Slurm job and is touched every minute; it counts as held only while Slurm does not call
that job dead **and** it was touched within 30 min, and a walltime SIGTERM releases it at
once (`frame_labels.running_elsewhere`, `_Heartbeat`).

```bash
python workflows/hessian_learning/03_labels.py --tag rings --all --dry-run          # the round's summary only
python workflows/hessian_learning/03_labels.py --tag rings --species dsgdb9nsd_000048 --local   # here, no Parsl
```

### On tianhe (TianheXY‑C)

**The campaign is run from [`docs/hessian_learning_campaign.md`](../../docs/hessian_learning_campaign.md)**
: the six‑command sequence — branch A and 02 as sbatch arrays sized to
their cost, `01_select`, then step 03 as a **parsl driver in `tmux`** on the login node
(`--resource tianhe_cpu --max-blocks 12`, the only stage that needs rounds; the tenant's
32‑submission quota counts every array task, so rounds are not pre‑queued), then 04 — with
the five‑minute tmux gate, the cost table with its measured column, what each log's last
lines must say, the progress table (`scripts/tooling/s0_hl_progress.py --tag draw300`).
What follows is the mechanism.

**A submitted job is plain bash + `xargs`, no parsl** — the hkuhpc shape
(`qm9_reaction_eng/docs/bash_orca_remain_workflow_lecture.md`): a task list, `awk` gives
each slot a core range, `xargs -P 16` runs an idempotent worker under `taskset` on the
node‑local scratch, the terminal line decides; many nodes = a Slurm **array**, the list
split round‑robin. Nothing runs on the login node. Every stage script skips what is on
disk — a finished label stays done, and a failed frame without an archive is carried once
by the round that comes next — so a killed or time‑limited job is resubmitted
as it is; `TIMEOUT_S` (default 28800) bounds one ORCA job.

| stage | script | worker per line | layout |
|---|---|---|---|
| A branch A | `hl_branchA.slurm` | `hl_branchA_worker.sh` → `s0_A_pipeline.py --species` | 16 × 4 (CREST `-T 4`); one attempt per molecule: no ensemble → `_records/branchA.failed`, not rerun |
| 02 frames | `hl_frames.slurm` | `02_frames.py --species` (MACE, CPU) | **64 × 1** (MACE is single-threaded) |
| 03 labels | `hl_labels.slurm` | `hl_label_worker.sh` → `python -m openqha.data.frame_labels` | 16 × 4 ORCA ranks, `%maxcore 6000` |

Every stage script asks for the whole node (`--exclusive --ntasks=1 --cpus-per-task=64
--mem=0`) and derives `CONCURRENCY` from what Slurm granted (`SLURM_CPUS_PER_TASK`, read
before the `SLURM_*` unset loop); `nproc` is not usable for this — it honours
`common.sh`'s `OMP_NUM_THREADS=1` and reported 1 core on a 64-core node (2026-09-20).
| all of A → 04 | `hl_pipeline_debug.slurm` | the above in one `debug` job (one round of 16 labels) | |

The label of a **basin / merged / saddle** frame is `EnGrad Freq` (E, F, H); of a
**displaced** frame `EnGrad` only (E, F). Kept per frame: `<molecule>/frames/orca.<level>.
<frame>.{inp,out,hess,engrad}`; no `.gbw`, `.loc`, `property.txt`. ORCA 6.1.1 (OpenMPI 4.1.8 in the conda
env `orca611`) is located by `hpc/env/orca.sh` from `~/env_orca611.sh` without activating
that env in the job.

```bash
cd ~/openQHA-main; export OPENQHA_PARTITION=deimos; source hpc/env/common.sh && source hpc/env/tianhe.sh
# 00 the draw (login node, seconds; the same seed gives the same 6,458 molecules)
python workflows/hessian_learning/00_draw.py --tag draw300 --per-class 300 --seed 0
# the gate: each stage once on debug, LIMIT=16
TAG=draw300 LIMIT=16 sbatch --partition=debug --time=00:30:00 hpc/slurm/hl_branchA.slurm
TAG=draw300 LIMIT=16 sbatch --partition=debug --time=00:30:00 hpc/slurm/hl_frames.slurm
python workflows/hessian_learning/01_select.py --tag draw300
TAG=draw300 LIMIT_FRAMES=16 sbatch --partition=debug --time=00:30:00 hpc/slurm/hl_labels.slurm
# the campaign: arrays of 12 one-node tasks; the labels three times at 3 days (~8.4 days of 12 nodes at 300 per class)
TAG=draw300 sbatch --array=0-11 --time=1-00:00:00 hpc/slurm/hl_branchA.slurm
TAG=draw300 sbatch --array=0-11 --time=04:00:00 hpc/slurm/hl_frames.slurm
python workflows/hessian_learning/01_select.py --tag draw300
TAG=draw300 sbatch --array=0-11 --time=3-00:00:00 hpc/slurm/hl_labels.slurm     # x3, until task 0's assemble exits 0 (docs/hessian_learning_campaign.md)
python workflows/hessian_learning/04_dataset.py --tag draw300 --export openreact
```
Read each task's log (`logs/slurm/openqha_hl_<stage>_<jobid>_<task>.out`): the task list size, one
line per molecule/frame, the summary line (`N done, N not, wall`). Measured 2026‑09‑19 on
a debug node: branch A 460–590 s per molecule (16 at once), a 10‑atom wB97M‑D3BJ/
def2‑TZVPPD Hessian label 245–305 s and 75–92 MB per rank under 16‑way contention.

**The campaign's route for 03 is the parsl driver** (the ALF mode): on the login node in
`tmux`, `SlurmProvider` submitting up to `--max-blocks` one‑node blocks with `sbatch` as the
queue demands and releasing them as it drains; a block cut at its time limit is replaced
and its frames in flight rerun whole. The sbatch array above is the fallback if the tmux
gate fails, and the same on‑disk state lets the two overlap:
```bash
tmux new -s hl-labels        # then source hpc/env/common.sh + tianhe.sh inside it
python -u workflows/hessian_learning/03_labels.py --tag draw300 --resource tianhe_cpu \
    --max-blocks 12 --walltime 3-00:00:00 2>&1 | tee $S0_RUNS_ROOT/logs/labels_draw300_$(date +%F_%H%M).log
# Ctrl+b d detaches; tmux attach -t hl-labels returns; squeue -u $USER shows the blocks
```

## Step 00: the structure-class draw

`configs/structure_classes.yaml` defines 23 classes by SMARTS or ring rule (three‑ and
four‑membered rings, bicyclic, polycyclic, aromatic, eight‑membered, heterocyclic;
carbonitrile, primary/secondary alcohol, trialkylamine, tertiary/aliphatic/aromatic amine,
alkyne, dialkyl ether, epoxide, aldehyde, ketone, amide, carboxylic acid, ester,
cyclopropane). `scripts/tooling/s0_structure_census.py` counts every class over the
curated QM9 files against the quoted counts and flags a pattern >10 % off. `00_draw.py`
draws 500 per class (seeded per class over the sorted candidates) from the 119,275 gated
targets outside SPICE at any match level; the union is the list (a molecule counts for
every class it is in), a short class takes all it has and the shortfall is in the Record.
First real draw (`--tag draw500 --seed 0`): **10,471 molecules**; eight‑membered rings
378 (all), carboxylic acids 0 (QM9's gated targets have no free COOH).

## Step 01: the selection

Every molecule of the draw (`draw.dat`, when the Dataset has one) is a row of `select.dat`
— with `has_basins` / `has_frames` false until branch A / 02 have run for it, so the
campaign's progress is one table and `hl_labels.slurm` (`03_labels --name`) always sees the
whole draw — plus every branch‑A‑finished molecule under the tags. Each row carries the
stratification keys, the structure `classes` (from `draw.dat`, else classified from the
SMILES with `configs/structure_classes.yaml`), SPICE membership, pin status; the Record has
a `[[Class]]` table (molecules / with basins / with frames per class).

## The production row R4, end to end

One row is trained: Replay = **4 × the train frames that carry a Hessian**, every Replay
frame at `config_weight = 10`, `w_H` = the epoch-0 balance measured on the base model over
the run's own train file (the default of `05_train.py`), everything else `05_train.py`'s defaults.
The judge reports every row and decides nothing (the gate is closed).

```bash
# 1. labels -> the Dataset: basin frames train and validate, the other generators are held out
python workflows/hessian_learning/04_dataset.py --tag draw300 --name draw300 --train-generators basin
#    ... prints N_TRAIN_HESSIAN and REPLAY_R4_FRAMES = 4 x N_TRAIN_HESSIAN (also in dataset.toml)

# 2. the two SPICE draws (the release is on tianhe): the forgetting set, then the Replay of exactly that size
python scripts/tooling/s0_spice_test_draw.py --n 5000
python scripts/tooling/s0_spice_pt_draw.py --n <REPLAY_R4_FRAMES> --seed 0 --weight 10 --out $S0_RUNS_ROOT/spice/spice_pt_R4.extxyz
#    ... writes spice_pt_R4.extxyz (+ .ids.dat, .toml) and spice_pt_R4.valid.extxyz

# 3. the fine-tune (one A800): w_H = balance is the default; the Record prints REPLAY_PER_HESSIAN_FRAME = 4.0
TAG=draw300 RUN=R4 MAX_EPOCHS=100 MULTIHEADS=1 PT_TRAIN_FILE=$S0_RUNS_ROOT/spice/spice_pt_R4.extxyz \
    PT_VALID_FILE=$S0_RUNS_ROOT/spice/spice_pt_R4.valid.extxyz yhbatch -p ai -G 1 -c 12 -t 24:00:00 hpc/slurm/hl_train.slurm

# 4. register R4 as an engine (needed before anything can run WITH it), then its own minima and msRRHO for the seven
python workflows/hessian_learning/05_train.py --tag draw300 --run R4 --register-copy   # prints the ENGINES entry and the name to select
export S0_ENGINE=<the name --register-copy printed>
python scripts/production/s0_A_pipeline.py --tag r4 --species dsgdb9nsd_000035   # x 7
python scripts/production/s0_thermo_msrrho.py --tag r4 --species dsgdb9nsd_000035 --step mace
#    (--step reference reuses the ORCA basins of the smoke tags; --step compare writes MODEL_ERROR_S_REF under tag r4)

# 5. the judge, gate closed: every row against its number, VERDICT = REPORTED
python workflows/hessian_learning/06_judge.py --tag draw300 --engine <that name> \
    --spice-file data/training_sets/spice_test_5000.extxyz --thermo-tag r4
#    the MD ramp only afterwards, if wanted:  ... --run R4_ramp --ramp --ramp-max-K 600
```

The smoke run on the tianhe test set (one day of CREST + one day of ORCA) is the same
five steps with `--tag smoke` and `MAX_EPOCHS=20`; its Record's `SECONDS_PER_EPOCH` sets
the walltime of step 3.

## Step 05: the fine-tune

`05_train.py` is openQHA's own entry point into the mace **fork**
(`BloomDlwlrma/mace`, branch `openqha-hessian`): it builds mace's arguments and
calls `mace.cli.run_train.run`, the Hessian labels come from the Dataset's
`mace_<name>.<level>.extxyz` (`REF_hessian`, fork commit A), the loss from
`openqha_hessian.phl_loss` through the fork's generic hook (commit B), kept in multihead
mode and given the force graph at evaluation (commit C):

    --loss external --loss_module openqha_hessian.phl_loss:build

Nothing here reimplements a training loop. **The target is the Cartesian matrix**
(and there is no switch to say otherwise): `L_H = ‖H_θ − H_r‖²_F / (9N²)`,
PHL's eq. 2.1', no mass weighting, no projection — derived in
`docs/tutorials/T05_openQHA_Theory_PHL_Finetune_EFH_to_wB97M.ipynb` and T04. Fork commit D
took `--hessian_mode_weighting` and `--hessian_probe modes` out of mace's parser, so a run
that asks for the projected target of T03 fails in argument parsing.

```bash
# what would run, and the Record header -- no training
python workflows/hessian_learning/05_train.py --tag smoke --run w1 --dry-run

# the smoke Dataset, two epochs (the exact Cartesian loss: --probe cartesian, 3N HVPs)
python workflows/hessian_learning/05_train.py --tag smoke --run w1 --hessian-weight 0.01 --max-epochs 2

# the Replay file, once per campaign (one seed, one file; a smaller --n is a prefix of a larger one)
python scripts/tooling/s0_spice_test_draw.py --n 5000                         # the forgetting draw first
python scripts/tooling/s0_spice_pt_draw.py --n 5000 --seed 0 --out $S0_RUNS_ROOT/spice/spice_pt_5000.extxyz

# a scan row, on one A800
TAG=draw300 RUN=R1 HESSIAN_WEIGHT=0.01 N_PROBES=4 MAX_EPOCHS=100 MULTIHEADS=1 \
    PT_TRAIN_FILE=$S0_RUNS_ROOT/spice/spice_pt_5000.extxyz PT_VALID_FILE=$S0_RUNS_ROOT/spice/spice_pt_5000.valid.extxyz \
    yhbatch -p ai -G 1 -c 12 -t 24:00:00 hpc/slurm/hl_train.slurm
```

**Training estimates; validation estimates on fixed probes.** During a step
the Hessian term is the Hutchinson estimator (`--probe rademacher --n-probes 4`: `k`
Hessian‑vector products per structure, eq. 6'); at validation the SAME estimator runs on
four Rademacher probes fixed per frame (seeded from the frame's Label), so the
validation Hessian curve is a cheap, unbiased, reproducible reading that enters the
total validation loss mace's scheduler, best checkpoint and Stage Two read. The full
matrix is the judge's tool, not the trainer's. `--probe cartesian` makes the training
term exact (3N HVPs).

**The control is explicit (the base's recipe).** `--lr 0.01`,
`--scheduler-patience 20`, `--patience 50`, `--eval-interval 1`, `--ema`, Stage Two at
`--start-swa` 3/4 of the epochs with `--swa-lr lr/40` and the Stage Two weights
(`--swa-energy-weight 1000 --swa-forces-weight 100`, `swa_hessian_weight = w_H ×
swa_forces_weight / forces_weight`); every value is in `config.yaml` and the Record,
with the three validation curves (E, F, Hessian terms per epoch, epoch −1 = the base).

**The Replay.** `--multiheads --pt-train-file FILE`
concatenates the file `s0_spice_pt_draw.py` writes — the only source of a Replay file —
as mace's pretraining head. **The file is the size knob**: mace's `--num_samples_pt`
acts only on its Materials‑Project download path and is never emitted; mace's
`--real_pt_data_ratio_threshold` (default 0.1, which silently duplicates the fine‑tune
frames when they are fewer than a tenth of the Replay) is passed as 0. The Record counts
the file's frames (`PT_N_FRAMES`), reads their `config_weight` (`PT_CONFIG_WEIGHT`),
parses both heads' counts from mace's log and prints `REPLAY_PER_HESSIAN_FRAME`
(`PT_N_FRAMES / N_TRAIN_HESSIAN`), the number the scan rows R0–R4 are defined by.
`--pt-valid-file` names the draw tool's companion `<stem>.valid.extxyz`; without it mace
takes `--valid_fraction` (10 %) of the Replay for the pretraining head's own validation.

**The fork is required, and its commit is recorded.** `05_train.py` refuses a mace that
is not an editable checkout of the fork, or one with uncommitted changes to tracked
files: a potential whose loss cannot be reproduced from a commit is not a product
(`--no-strict-fork` overrides it for experiments, and the Record says so).
`--register` prints the `ENGINES` entry for the fine-tuned model (the Dataset's
`index.dat` plus the config SHA as its `source`); `--register-copy` also puts the file
into `data/potentials/`.

## Step 04: the Dataset

Two splits (`--split-by`). **`molecule`, the default (production):** `test` = whole
molecules: the pinned seven (acetone 000018, acetamide 000019, propanal 000035,
N‑methylformamide 000036, 2‑methyloxirane 000044, cyclopropanol 000046, oxetane 000048)
plus `TEST_FRACTION` (5 %) of the others drawn per stratum (ring count × heteroatom
pattern); `valid` = `VALID_FRACTION` (5 %) of the training molecules' labelled frames,
drawn per molecule; `train` = the rest — conformers of one molecule
never sit on both sides. **`frame` (the smoke / fit mode):** the pinned seven are whole
test molecules; every other labelled frame goes to train / valid / test at **90 / 5 / 5**
by its own draw (a generator seeded from `--seed` and the frame's name, so a frame's split
never depends on what else is labelled; the fractions are expectations, ±0.1 % over 100,000
frames). In both,
`pool` = frames without a label at the level yet, whatever their molecule's side
(`molecule_split` in the index). The split is set at
write time and never recomputed: a rebuild keeps every decision of the previous
`index.dat`. `index.dat` is the record (one row per frame: molecule, tag, basin,
generator, k, split, levels, seed, engine, ORCA version, SPICE membership,
stratum, **classes**, file + row, engine file); the Record has a `[[Class]]` table
(molecules, frames, labelled, per split).

`train/valid/test` files carry the **reference** E‑F‑H under `energy` / `forces` /
`hessian` **and** under MACE‑torch's default training keys `REF_energy` (info),
`REF_forces` (arrays) and `REF_hessian` (info, flattened; only at basin / merged / saddle
frames — `has_hessian` says which), plus `split`; `pool` carries the
engine's. **`mace_<name>.<level>.extxyz`** is the three labelled splits in one file — the
single xyz for MACE Hessian learning (05_train reads `REF_hessian` where present and
masks the Hessian term otherwise). `--export openreact` writes `molecules-<name>.h5` in OpenREACT's layout —
**Å, Eh, Eh/Å, Eh/Å²** (read off `molecules-RTP.h5`: with Eh/Å² its C–H stretches
project to 3156–3183 cm⁻¹; Eh/bohr² would give ~6000) — with our `split`, `generator`,
`basin`, `k` datasets beside the standard ones.

## Step 06: the judge

The ruler of Algorithm 3. It takes the SHIPPED full Hessian (`MACECalculator.get_hessian`,
mace's `compute_hessians_vmap`) against the Label through `hessian_compare` — the same
four metric families every earlier comparison in this repository used — and **never calls
the estimator or the training loss**: the optimiser reads eq. 6, the judge reads eq. 1
exactly. A loss that flatters itself cannot flatter the ruler.

```bash
# the base model on a Dataset, and the two calibrations
python workflows/hessian_learning/06_judge.py --tag rings --name smoke --engine base
python workflows/hessian_learning/06_judge.py --tag rings --name smoke --engine base --scale 0.9

# a fine-tuned potential, with the forgetting line
python scripts/tooling/s0_spice_test_draw.py --n 5000          # once; writes the ids beside the frames
python workflows/hessian_learning/06_judge.py --tag draw300 --engine <engine name>     --spice-file data/training_sets/spice_test_5000.extxyz
```

**Three rows, not one**: `interpolation` (frames of the fine-tuned
molecules, held out by frame — interpolation within them), `out_of_molecule` (whole
molecules, the only generalisation number) and `in_distribution` (the molecules the BASE
model was trained on, where the question is damage, not accuracy).

**Both calibrations, every time.** `--engine base` must not fail a no-degradation line and
gives ~0 against its own Hessian as the Label; `--scale 0.9` (forces ×0.9, Hessian ×0.81,
so every frequency ×0.9) must FAIL the low-mode line. Measured on the 2-methyloxirane
basin frame: the base model's low-mode MAE is 12.7 cm⁻¹ and the scaled one's 32.1, against
a threshold of 8.5 — so **the base model fails that line on an out-of-distribution ring,
which is the error the fine-tune exists to fix**. Every line prints the Label's
own grid noise beside it (24.4 cm⁻¹ on that frame): no threshold means anything
below the floor of the number it is judging.

**What the judge will not do**: recompute thermochemistry. The entropy tier is read from
the msRRHO Records on disk (`s0_thermo_msrrho.py` with `S0_ENGINE=<engine>` writes them),
and a molecule without one is reported as absent rather than filled in — a judge that
produced the numbers it judges would be marking its own work. A run with no labelled frame
refuses instead of reporting PASS.

## Registering the fine-tuned weights

See README.md "Registering a fine-tuned potential": `05_train.py --register-copy` copies
the model into `data/potentials/mace_off23_<campaign>/` as a stamped, fixed revision and
prints the `ENGINES` entry to paste; `scripts/tooling/s0_check_weights.py` lists, for every
registered name whose file is present, the path it resolves to.
