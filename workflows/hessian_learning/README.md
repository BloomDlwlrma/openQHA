# Workflow `hessian_learning`: the E‑F‑H Dataset for fine‑tuning MACE

Numbered Batches (CONTEXT.md *Workflow*) that turn a molecule list into a Dataset of
frames, each carrying the engine's and the reference's energy, forces and Cartesian
Hessian at the same geometry. Each step writes Records into the molecule directories it
touches; the Slurm log is the Batch's report.

| step | driver | Calculation | writes |
|---|---|---|---|
| 01 | `01_select.py` | `openqha.data.dataset.select` | `<root>/<tag>/_datasets/<name>/select.{out,toml,dat}` — the molecule list with stratum, SPICE membership, pin status |
| 02 | `02_frames.py` | `openqha.data.frames.generate` | `<molecule>/frames/<generator>.<mace level>.extxyz`, `frames/frames.{out,toml}` |
| 03 | `03_labels.py` | `openqha.data.frame_labels.label_one` per frame, `assemble` per molecule | `<molecule>/orca/<level>/frames/<generator>_bBB_kK/job.*`, `frames/<generator>.<level>.extxyz`, `frames/labels.<level>.{out,toml}` |
| 04 | `04_dataset.py` | `openqha.data.dataset.build` (+ `export_openreact`) | `<root>/<tag>/_datasets/<name>/{train,valid,test,pool}.<level>.extxyz`, `index.dat`, `dataset.{out,toml}`, `molecules-<name>.h5` |
| 05, 06 | `05_train.py`, `06_judge.py` | stubs: refuse until round 2 is ruled | |

`run.sh --tag T [--tag T2] --name NAME [--limit N] [--stratify] [--with-labels]` runs
01 → 02 → (03 with `--with-labels`, else skipped) → 04 here; the Dataset lives under the
first tag's `_datasets/<name>/`.

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

**Nothing runs on the login node and no Batch holds the command line: every step is a
submitted job.** `hpc/resource_configs/tianhe_cpu.py` runs *in‑allocation* whenever the
driver is inside a Slurm job (LocalProvider on the job's node(s), `srun` launcher across
nodes, nothing submitted); the login‑node "nested" mode (parsl submits blocks with
`sbatch`, S0‑G‑74) still exists but is not the way to test.

One frame per Parsl task, `%pal nprocs 4`, `%maxcore 6000` (512 GB × 0.75 / 64), 16
frames per 64‑core node (role `labels`). ORCA 6.1.1 (shared build, OpenMPI 4.1.8 in the
conda env `orca611`, verified 2026‑09‑19: 4 MPI processes, water in 41 s) is located by
`hpc/env/orca.sh` from `~/env_orca611.sh` (`ORCA_PATH` + alias) without activating that
env in the worker; ORCA's scratch goes to node‑local `S0_SCRATCH`.

```bash
cd ~/openQHA-main
# 1. the whole pipeline on debug, one node, 30 min: branch A -> 02 -> 01 -> 03 (one round of
#    16 frames, in the allocation) -> 04. Skips what is already on disk.
sbatch hpc/slurm/hl_pipeline_debug.slurm                    # TAG=smoke, the seven pinned molecules
# 2. the remaining labels + the Dataset, as one job (deimos 1 node 4 h by default;
#    finished frames are skipped, so resubmit after a time limit)
sbatch hpc/slurm/hl_labels.slurm                            # TAG=smoke
TAG=smoke PARTITION=debug TIME=00:30:00 LIMIT_FRAMES=32 sbatch hpc/slurm/hl_labels.slurm   # or two rounds on debug
# 3. the 200-molecule draw on 12 nodes
TAG=draw NAME=draw200 sbatch --nodes=12 --time=7-00:00:00 hpc/slurm/hl_labels.slurm
```
Read the Slurm log (`openqha_hl_*_<job>.out` in the submit directory): the frame list,
`workers IN THIS ALLOCATION`, the Batch table (frame, route, MB, rigid block), then step
04's per‑molecule split. Measured 2026‑09‑19 on a debug node: one wB97M‑D3BJ/def2‑TZVPPD
label of a 10‑atom frame = **278 s, 91 MB per rank**.

Environment check, once, on the login node (seconds, no compute):
`source ~/env_orca611.sh && mpirun --version | head -1` (Open MPI 4.1.x) and
`python -c "import procrustes"` in `openqha` (branch A needs `qc-procrustes`).

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
