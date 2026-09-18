# Workflow `hessian_learning`: the E‑F‑H Dataset for fine‑tuning MACE

Numbered Batches (CONTEXT.md *Workflow*) that turn a molecule list into a Dataset of
frames, each carrying the engine's and the reference's energy, forces and Cartesian
Hessian at the same geometry. Each step writes Records into the molecule directories it
touches; the Slurm log is the Batch's report.

| step | driver | Calculation | writes |
|---|---|---|---|
| 02 | `02_frames.py` | `openqha.data.frames.generate` | `<molecule>/frames/<generator>.<mace level>.extxyz`, `frames/frames.{out,toml}` |
| 03 | `03_labels.py` | `openqha.data.frame_labels.label_one` per frame, `assemble` per molecule | `<molecule>/orca/<level>/frames/<generator>_bBB_kK/job.*`, `frames/<generator>.<level>.extxyz`, `frames/labels.<level>.{out,toml}` |
| 01, 04 | selection; the Dataset split and index | ticket 04 | |

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

### On tianhe (TianheXY‑C, user ruling 2026‑09‑18)

One frame per Parsl task, `%pal nprocs 4`, `%maxcore 6000` (512 GB × 0.75 / 64), 16
frames per 64‑core node — role `labels` in `hpc/resource_configs/tianhe_cpu.py`; the
CPU cluster submits with `sbatch` (S0‑G‑74). ORCA 6.1.1 lives in the conda env
`orca611`; `hpc/env/orca.sh` enters `~/env_orca611.sh` once in the worker init, records
`S0_ORCA_BIN` / `S0_ORCA_PATH` / `S0_ORCA_LIB`, and leaves again, so the worker stays in
`openqha` and only the ORCA subprocess sees those paths. ORCA's scratch goes to the
node‑local `S0_SCRATCH`; `job.{inp,out,hess,engrad,xyz,property.txt}` come back.

```bash
# 1. one frame on the debug queue (30 min): modules, conda, ORCA, sbatch, status query
python -u workflows/hessian_learning/03_labels.py --tag smoke --all --resource tianhe_cpu --debug --limit-frames 1
# 2. the smoke set (7 molecules, ~80 frames) on one node
python -u workflows/hessian_learning/03_labels.py --tag smoke --all --resource tianhe_cpu --max-blocks 1
# 3. the 200-molecule draw on 12 nodes (7-day walltime allowed on deimos)
python -u workflows/hessian_learning/03_labels.py --tag draw --all --resource tianhe_cpu --limit 200 --stratify --walltime 7-00:00:00
```

A resubmission skips every frame whose `job.out` carries the terminal line and whose
`job.hess` exists, and reruns the rest; `assemble` then rewrites the file and the Record
from what is on disk. Environment check on the login node before step 1:
`source ~/env_orca611.sh && which orca && orca --version | head -3` (expect 6.1.1), and
`conda activate openqha && python -c "import procrustes"` (branch A needs
`qc-procrustes`; `pip install qc-procrustes` if it fails).

## Registering the fine‑tuned weights

See README.md "Registering a fine‑tuned potential": `scripts/tooling/s0_check_weights.py
--pin <file>` prints the `ENGINES` entry with the parameter fingerprint.
