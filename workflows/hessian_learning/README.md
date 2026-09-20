# Workflow `hessian_learning`: the E‑F‑H Dataset for fine‑tuning MACE

Numbered Batches (CONTEXT.md *Workflow*) that turn a molecule list into a Dataset of
frames, each carrying the engine's and the reference's energy, forces and Cartesian
Hessian at the same geometry. Each step writes Records into the molecule directories it
touches; the Slurm log is the Batch's report.

| step | driver | Calculation | writes |
|---|---|---|---|
| 00 | `00_draw.py` | `openqha.data.structure_classes.draw` | `<root>/<tag>/_datasets/<name>/draw.{out,toml,dat}` — the campaign's molecule list: 500 per structure class (`configs/structure_classes.yaml`) from the gated QM9 targets outside MACE-OFF23's SPICE training file, the union over classes, the pinned seven always in |
| 01 | `01_select.py` | `openqha.data.dataset.select` | `<root>/<tag>/_datasets/<name>/select.{out,toml,dat}` — the molecule list with stratum, SPICE membership, pin status |
| 02 | `02_frames.py` | `openqha.data.frames.generate` | `<molecule>/frames/<generator>.<mace level>.extxyz`, `frames/frames.{out,toml}` |
| 03 | `03_labels.py` | `openqha.data.frame_labels.label_one` per frame, `assemble` per molecule | `<molecule>/frames/orca.<level>.<generator>_bBB_kK.{inp,out,hess,engrad}`, `frames/<generator>.<level>.extxyz`, `frames/labels.<level>.{out,toml}` |
| 04 | `04_dataset.py` | `openqha.data.dataset.build` (+ `export_openreact`) | `<root>/<tag>/_datasets/<name>/{train,valid,test,pool}.<level>.extxyz`, `index.dat`, `dataset.{out,toml}`, `molecules-<name>.h5` |
| 05, 06 | `05_train.py`, `06_judge.py` | stubs: refuse until round 2 is ruled | |

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

**A submitted job is plain bash + `xargs`, no parsl** — the hkuhpc shape
(`qm9_reaction_eng/docs/bash_orca_remain_workflow_lecture.md`): a task list, `awk` gives
each slot a core range, `xargs -P 16` runs an idempotent worker under `taskset` on the
node‑local scratch, the terminal line decides; many nodes = a Slurm **array**, the list
split round‑robin. Nothing runs on the login node. Every stage script skips what is on
disk, so a killed or time‑limited job is resubmitted as it is.

| stage | script | worker per line | layout |
|---|---|---|---|
| A branch A | `hl_branchA.slurm` | `hl_branchA_worker.sh` → `s0_A_pipeline.py --species` | 16 × 4 threads |
| 02 frames | `hl_frames.slurm` | `02_frames.py --species` (MACE, CPU) | 16 × 4 |
| 03 labels | `hl_labels.slurm` | `hl_label_worker.sh` → `python -m openqha.data.frame_labels` | 16 × 4 ORCA ranks, `%maxcore 6000` |
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
TAG=draw300 sbatch --array=0-11 --time=3-00:00:00 hpc/slurm/hl_labels.slurm     # x3, until task 0's assemble exits 0
python workflows/hessian_learning/04_dataset.py --tag draw300 --export openreact
```
Read each task's log (`openqha_hl_<stage>_<jobid>_<task>.out`): the task list size, one
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

## Step 04: the Dataset (rounds 3–4, Q3/Q6 (b))

`test` = whole molecules the model never sees: the pinned seven (acetone 000018,
acetamide 000019, propanal 000035, N‑methylformamide 000036, 2‑methyloxirane 000044,
cyclopropanol 000046, oxetane 000048) plus `TEST_FRACTION` (0.1) of the others drawn per
stratum (ring count × heteroatom pattern) with `--seed`; `valid` = `VALID_FRACTION` (0.1)
of the training molecules' labelled frames, drawn **by frame** (early stopping sees
interpolation, as SPICE); `train` = the rest; `pool` = frames without a label at the
level yet, whatever their molecule's split (`molecule_split` in the index). The split is
set at write time and never recomputed; `index.dat` is the record (one row per frame:
molecule, tag, basin, generator, k, split, levels, seed, engine fingerprint, ORCA
version, SPICE membership, stratum, file + row, engine file). `train/valid/test` files
carry the **reference** E‑F‑H under `energy` / `forces` / `hessian`; `pool` carries the
engine's. `--export openreact` writes `molecules-<name>.h5` in OpenREACT's layout —
**Å, Eh, Eh/Å, Eh/Å²** (read off `molecules-RTP.h5`: with Eh/Å² its C–H stretches
project to 3156–3183 cm⁻¹; Eh/bohr² would give ~6000) — with our `split`, `generator`,
`basin`, `k` datasets beside the standard ones.

## Registering the fine‑tuned weights

See README.md "Registering a fine‑tuned potential": `scripts/tooling/s0_check_weights.py
--pin <file>` prints the `ENGINES` entry with the parameter fingerprint.
