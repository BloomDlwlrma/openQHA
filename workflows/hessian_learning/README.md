# Workflow `hessian_learning`: the E‑F‑H Dataset for fine‑tuning MACE

Numbered Batches (CONTEXT.md *Workflow*) that turn a molecule list into a Dataset of
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
| 05 | `05_train.py` | `openqha.training.run.run_training` → `mace.cli.run_train.run` (the fork) | `<root>/<tag>/_datasets/<name>/train/<run>/` — mace's own files (`<run>.model`, `checkpoints/`, `logs/`, `results/`) plus `config.yaml` and the Record `train.{out,toml,dat}` (settings, epoch table, the base and fine-tuned fingerprints, the config SHA, the mace fork's commit) |
| 06 | `06_judge.py` | `openqha.training.judge.run` | `<root>/<tag>/_datasets/<name>/judge/<run>/judge.{out,toml,dat}` — per frame, per structure class and per distribution; the entropy tier read from the msRRHO Records; the forgetting line on a fixed SPICE draw; one PASS / FAIL line per threshold |

`run.sh --tag T [--tag T2] [--name NAME] [--limit N] [--stratify] [--with-labels]` runs
01 → 02 → (03 with `--with-labels`, else skipped) → 04 here; the Dataset lives under the
first tag's `_datasets/<name>/`.

**One campaign = one tag = one Dataset** (ticket 09, ruling 2026‑09‑20): every step's
`--name` and every stage script's `NAME` default to the tag, so `TAG=draw300` alone is the
whole address — `<root>/draw300/<qid>/` for the molecules (the tag directory is flat: the
two shard layers of 2026‑09‑14 are gone),
`<root>/draw300/_datasets/draw300/` for the Dataset. `--name` is only for a second subset
on the same tree. A frame's ORCA files are a FILE GROUP of the molecule directory,
`orca.<level>.<generator>_bBB_kK.{inp,out,hess,engrad}` (`layout.orca_frame_stem`), not a
directory two levels down; the fields are joined by `.` because the level and the frame tag
contain `_`; they live in `frames/` beside the Frame sets they label. The msRRHO study has
its own sub‑folder `msrrho/` (`orca.<level>.basinNN.*` file groups, `thermo/` Records,
`crest_entropy/`, `xtb/`); a molecule directory is `crest/ mace/ md_*/ frames/ msrrho/ _records/`.

## The frame recipe (rounds 3–4, rulings 2026‑09‑18)

Seeds are branch A's MACE basins (CREST on GFN2‑xTB with `refine = "sp"`, MACE
tightening to fmax 1e‑4, MACE analytic Hessian). Four generators: `basin` (the minimum),
`displaced` (4 draws per basin from the **classical** harmonic distribution at 298.15 K
along the basin's modes, ⟨q²⟩ = k_BT/ω², RMS ≤ 0.15 Å; round‑2 Q14 (b)), `merged` (a
CREST conformer branch A merged into a basin), `saddle`. Seeds are
`SHA‑256(qm9_index|basin|generator|k)`. The only filter is ANI‑1's energy window
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
term included, which is what the loss compares to the engine's. Units are converted
once (Eh → eV, Eh/bohr → eV/Å, Eh/bohr² → eV/Å²); positions in the labelled file are
the MACE file's verbatim, and a `.hess` geometry whose shape is more than 1e‑7 Å away is refused (ORCA writes the `.hess` in the centre‑of‑mass frame; the translation is removed and reported; the residual is ORCA's print precision, ~1e‑8 Å). The
full `.out` of every job is kept.

```bash
python workflows/hessian_learning/03_labels.py --tag rings --all --dry-run          # the frame list
python workflows/hessian_learning/03_labels.py --tag rings --species dsgdb9nsd_000048 --local   # here, no Parsl
```

### On tianhe (TianheXY‑C; rulings 2026‑09‑18 / 2026‑09‑19)

**The campaign is run from [`docs/hessian_learning_campaign.md`](../../docs/hessian_learning_campaign.md)**
(ticket 08): the sequence, the cost table with its measured column, the 3 × 3‑day
resubmission scheme, what each log's last lines must say, the progress table
(`scripts/tooling/s0_hl_progress.py --tag draw300`). What follows is the mechanism.

**A submitted job is plain bash + `xargs`, no parsl** — the hkuhpc shape
(`qm9_reaction_eng/docs/bash_orca_remain_workflow_lecture.md`): a task list, `awk` gives
each slot a core range, `xargs -P 16` runs an idempotent worker under `taskset` on the
node‑local scratch, the terminal line decides; many nodes = a Slurm **array**, the list
split round‑robin. Nothing runs on the login node. Every stage script skips what is on
disk, so a killed or time‑limited job is resubmitted as it is.

| stage | script | worker per line | layout |
|---|---|---|---|
| A branch A | `hl_branchA.slurm` | `hl_branchA_worker.sh` → `s0_A_pipeline.py --species` | 16 × 4 (CREST `-T 4`) |
| 02 frames | `hl_frames.slurm` | `02_frames.py --species` (MACE, CPU) | **64 × 1** (MACE is single-threaded) |
| 03 labels | `hl_labels.slurm` | `hl_label_worker.sh` → `python -m openqha.data.frame_labels` | 16 × 4 ORCA ranks, `%maxcore 6000` |

Every stage script asks for the whole node (`--exclusive --ntasks=1 --cpus-per-task=64
--mem=0`) and derives `CONCURRENCY` from what Slurm granted (`SLURM_CPUS_PER_TASK`, read
before the `SLURM_*` unset loop); `nproc` is not usable for this — it honours
`common.sh`'s `OMP_NUM_THREADS=1` and reported 1 core on a 64-core node (2026-09-20).
| all of A → 04 | `hl_pipeline_debug.slurm` | the above in one `debug` job (one round of 16 labels) | |

The label of a **basin / merged / saddle** frame is `EnGrad Freq` (E, F, H); of a
**displaced** frame `EnGrad` only (E, F) — round 5, Q7 (b). Kept per frame: `<molecule>/frames/orca.<level>.
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

**The ALF mode** (parsl, for a campaign nobody wants to resubmit by hand): the driver on
the login node in `tmux`, `SlurmProvider` submitting up to `--max-blocks` one‑node
blocks with `sbatch` as the queue demands and releasing them as it drains — the same
on‑disk state, so the two modes can be mixed:
```bash
tmux new -s hl-labels
python -u workflows/hessian_learning/03_labels.py --tag draw300 --resource tianhe_cpu \
    --max-blocks 12 --walltime 3-00:00:00 2>&1 | tee $S0_RUNS_ROOT/logs/labels_draw_$(date +%F_%H%M).log
# Ctrl+b d detaches; tmux attach -t hl-labels returns; squeue -u $USER shows the blocks
```

## Step 00: the structure-class draw (round 5, 2026‑09‑19)

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

## Step 01: the selection (tickets 04 and 07)

Every molecule of the draw (`draw.dat`, when the Dataset has one) is a row of `select.dat`
— with `has_basins` / `has_frames` false until branch A / 02 have run for it, so the
campaign's progress is one table and `hl_labels.slurm` (`03_labels --name`) always sees the
whole draw — plus every branch‑A‑finished molecule under the tags. Each row carries the
stratification keys, the structure `classes` (from `draw.dat`, else classified from the
SMILES with `configs/structure_classes.yaml`), SPICE membership, pin status; the Record has
a `[[Class]]` table (molecules / with basins / with frames per class).

## Step 05: the fine-tune (ticket 13)

`05_train.py` is openQHA's own entry point into the mace **fork**
(`BloomDlwlrma/openQHA-Hessian`, branch `openqha-hessian`): it builds mace's arguments and
calls `mace.cli.run_train.run`, the Hessian labels come from the Dataset's
`mace_<name>.<level>.extxyz` (`REF_hessian`, fork commit A) and the loss from
`openqha.training.phl_loss` through the fork's generic hook (commit B):

    --loss external --loss_module openqha.training.phl_loss:build

Nothing here reimplements a training loop; the loss is the projected Hessian loss of
`design-phl-loss.md` (eq. 11), derived and verified in
`docs/tutorials/T03_openQHA_Theory_Projected_Hessian_Loss.ipynb`.

```bash
# what would run, and the Record header -- no training
python workflows/hessian_learning/05_train.py --tag smoke --run w1 --dry-run

# the exact loss (probe = modes, eq. 10) on the smoke Dataset, two epochs
python workflows/hessian_learning/05_train.py --tag smoke --run w1     --probe modes --hessian-weight 0.01 --max-epochs 2

# the campaign, on one A800
TAG=draw300 RUN=prod1 HESSIAN_WEIGHT=0.01 N_PROBES=4 MAX_EPOCHS=100 MULTIHEADS=1     PT_TRAIN_FILE=$S0_RUNS_ROOT/spice/spice_pt_5000.extxyz     yhbatch -p ai -G 1 -c 12 -t 24:00:00 hpc/slurm/hl_train.slurm
```

**Training estimates, evaluation is exact.** During a step the Hessian term is the
Hutchinson estimator (`--probe rademacher --n-probes 4`: `k` Hessian-vector products per
structure, eq. 6); at validation mace is asked for the full 3N x 3N Hessian and the term
is eq. 1 exactly, so the logged validation number is the ruler's quantity and not a
sample. `--probe modes` makes the training term exact too, at `n_vib` HVPs -- the setting
the smoke fit (ticket 15) measures the ceiling and the cost with.

**The fork is required, and its commit is recorded.** `05_train.py` refuses a mace that
is not an editable checkout of the fork, or one with uncommitted changes to tracked
files: a potential whose loss cannot be reproduced from a commit is not a product
(`--no-strict-fork` overrides it for experiments, and the Record says so).
`--register` prints the `ENGINES` entry for the fine-tuned model (ticket 01's recipe:
the Dataset's `index.dat` plus the config SHA as its `source`), `--register-copy` also
puts the file into `data/potentials/`.

## Step 04: the Dataset (rounds 3–4 Q3/Q6 (b); round 5 Q4, ticket 07)

Two splits (`--split-by`). **`frame`, the default (production, round 5 Q4):** the pinned
seven (acetone 000018, acetamide 000019, propanal 000035, N‑methylformamide 000036,
2‑methyloxirane 000044, cyclopropanol 000046, oxetane 000048) are whole test molecules;
every other labelled frame goes to train / valid / test at **90 / 5 / 5** by its own draw
(a generator seeded from `--seed` and the frame's name, so a frame's split never depends
on what else is labelled; the fractions are expectations, ±0.1 % over 100,000 frames).
**`molecule` (the smoke set):** `test` = whole molecules: the pinned seven plus
`TEST_FRACTION` (0.1) of the others drawn per stratum (ring count × heteroatom pattern);
`valid` = `VALID_FRACTION` (0.1) of the training molecules' labelled frames, drawn by
frame; `train` = the rest. In both, `pool` = frames without a label at the level yet,
whatever their molecule's side (`molecule_split` in the index). The split is set at
write time and never recomputed: a rebuild keeps every decision of the previous
`index.dat`. `index.dat` is the record (one row per frame: molecule, tag, basin,
generator, k, split, levels, seed, engine fingerprint, ORCA version, SPICE membership,
stratum, **classes**, file + row, engine file); the Record has a `[[Class]]` table
(molecules, frames, labelled, per split).

`train/valid/test` files carry the **reference** E‑F‑H under `energy` / `forces` /
`hessian` **and** under MACE‑torch's default training keys `REF_energy` (info),
`REF_forces` (arrays) and `REF_hessian` (info, flattened; only at basin / merged / saddle
frames — `has_hessian` says which; round 5 Q7 (b)), plus `split`; `pool` carries the
engine's. **`mace_<name>.<level>.extxyz`** is the three labelled splits in one file — the
single xyz for MACE Hessian learning (05_train reads `REF_hessian` where present and
masks the Hessian term otherwise). `--export openreact` writes `molecules-<name>.h5` in OpenREACT's layout —
**Å, Eh, Eh/Å, Eh/Å²** (read off `molecules-RTP.h5`: with Eh/Å² its C–H stretches
project to 3156–3183 cm⁻¹; Eh/bohr² would give ~6000) — with our `split`, `generator`,
`basin`, `k` datasets beside the standard ones.

## Step 06: the judge (ticket 14)

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
python workflows/hessian_learning/06_judge.py --tag draw300 --engine MACE-OFF23_medium-prod1     --spice-file data/training_sets/spice_test_5000.extxyz
```

**Three rows, not one** (CONTEXT *Held-out*): `interpolation` (frames of the fine-tuned
molecules, held out by frame — interpolation within them), `out_of_molecule` (whole
molecules, the only generalisation number) and `in_distribution` (the molecules the BASE
model was trained on, where the question is damage, not accuracy).

**Both calibrations, every time.** `--engine base` must not fail a no-degradation line and
gives ~0 against its own Hessian as the Label; `--scale 0.9` (forces ×0.9, Hessian ×0.81,
so every frequency ×0.9) must FAIL the low-mode line. Measured on the 2-methyloxirane
basin frame: the base model's low-mode MAE is 12.7 cm⁻¹ and the scaled one's 32.1, against
a threshold of 8.5 — so **the base model fails that line on an out-of-distribution ring,
which is the error the fine-tune exists to fix** (S0-C-41). Every line prints the Label's
own grid noise beside it (24.4 cm⁻¹ on that frame, S0-C-44): no threshold means anything
below the floor of the number it is judging.

**What the judge will not do**: recompute thermochemistry. The entropy tier is read from
the msRRHO Records on disk (`s0_thermo_msrrho.py` with `S0_ENGINE=<engine>` writes them),
and a molecule without one is reported as absent rather than filled in — a judge that
produced the numbers it judges would be marking its own work. A run with no labelled frame
refuses instead of reporting PASS.

## Registering the fine‑tuned weights

See README.md "Registering a fine‑tuned potential": `scripts/tooling/s0_check_weights.py
--pin <file>` prints the `ENGINES` entry with the parameter fingerprint.
