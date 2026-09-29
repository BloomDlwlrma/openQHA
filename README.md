# openQHA: Active Learning Framework Instance for Conformational Free Energy

Conformational free energy for small organic molecules, from a SMILES string to an
entropy — built on a machine-learned interatomic potential, with every criterion written
so that it can fail.

---

## What it does

Give it a molecule. It finds the conformers, converges each one on the potential,
screens out the saddle points, gets a symmetry number from each geometry, and returns a
basin list with Boltzmann weights and thermodynamic terms.

```
   SMILES  or  QM9 index
        │
        ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  BRANCH A — conformer search                                             │
 │                                                                          │
 │   gates F0–F7 ──▶ CREST iMTD-GC ──▶ pool reference ──▶ tighten           │
 │                   GFN2-xTB workhorse    geometry         fmax 1e-4 eV/Å   │
 │                   MACE refines                              │             │
 │                                                             ▼             │
 │                       basin list ◀── σ per basin ◀── analytic Hessian     │
 │                       + Boltzmann       geometric      floor ithr -50     │
 │                         weights         superposition                     │
 └───────────────────────────────┬──────────────────────────────────────────┘
                                 │  basin GEOMETRIES  (never the trajectory)
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
 ┌───────────────────────────────┐ ┌───────────────────────────────────────┐
 │  BRANCH B — quasi-harmonic    │ │  BRANCH C — Hessian training          │
 │                               │ │                                       │
 │  unbiased MD, dt = 1 fs       │ │  GFN2 labels ──▶ pre-train            │
 │  no constraints, real H mass  │ │  RI-MP2 labels ──▶ fine-tune          │
 │        │                      │ │        with PHL loss                  │
 │        ▼                      │ │        │                              │
 │  mass-weighted covariance     │ │        ▼                              │
 │        │                      │ │  corrected potential ─────────────────┼──┐
 │        ▼                      │ └───────────────────────────────────────┘  │
 │  ν = √(k_B T / λ)  ──▶  S_QH  │                                            │
 └───────────────────────────────┘ ◀──────────────────────────────────────────┘
                 │                          the trajectory runs on it
                 ▼
        free energy, with the
        anharmonicity separated

 ┌──────────────────────────────────────────────────────────────────────────┐
 │  BRANCH E — execution layer (Parsl).  Computes nothing.                  │
 │  Decides which machine runs what. Replacing it with a for-loop must not  │
 │  change a single number.                                                 │
 └──────────────────────────────────────────────────────────────────────────┘
```

**The interface between A and B is the basin geometries, never a trajectory.** Branch A's
dynamics is metadynamics with SHAKE and deuterium masses — excellent for crossing
barriers, ruinous for the fluctuations branch B measures. The two protocols sit
side by side in [`configs/mdp/`](configs/mdp/) precisely so that the incompatibility is
mechanical rather than something you have to remember.

---

## Status

| branch | what it is | state |
|---|---|---|
| **A** conformer search | CREST iMTD-GC + MACE refinement → basin list | **runs**; 9/9 acceptance criteria |
| **B** quasi-harmonic | unbiased MD → covariance → entropy | driver and analysis exist |
| **C** Hessian training | PHL fine-tuning on RI-MP2 labels | designed; calibration gate not passed |
| **E** execution | Parsl, local and cluster | local proven; nothing submitted to a cluster yet |

Open items are listed, not hidden: see [`docs/branchA_workflow.md`](docs/branchA_workflow.md) §8.

---

## Features

* **Conformer search** with CREST's published iMTD-GC protocol — the settings from the
  papers, not the program's defaults.
* **Composite calculator**: a GFN2-xTB workhorse drives ~10⁵ energy+gradient calls; the
  foundation model refines the ~2% that survive.
* **Basin criteria of our own**: a search program's conformer count is not a basin count.
  Measured on acetone — CREST reported 2 conformers 0.8118 kcal/mol apart, and after
  tightening their energies were identical to the last digit.
* **Deduplication by CREGEN's own three-fold criterion** — RMSD 0.125 Å *and*
  ΔE 0.05 kcal/mol *and* rotational constants within 1%. Upstream's values, not ours.
* **Analytic Hessians** by double backward through the potential: more accurate *and*
  cheaper than finite differences, with no trade-off to weigh.
* **Symmetry numbers from geometry**, per basin — not from the molecular graph, whose
  automorphism count is 72 for ethane against a true σ of 6.
* **Quasi-harmonic entropy** from unbiased trajectories, produced by **two independent
  routes** — ASE and OpenMM — that write the same product and are read by the same
  analysis, and cross-checked against `gmx covar -mwa` (agreement: 1.27e-04 kcal/mol on a
  real 25 ps trajectory).
* **One environment for everything**, including branch B's OpenMM route. Merging it
  required pytorch 2.13.0 → 2.12.1 and numpy 2.4.6 → 1.26.4, and the cost of that was
  *measured*: energy, forces, every Hessian frequency and `T*S` on fixed frames all
  **bit-identical** (`scripts/calibration/s0_B_stack_fingerprint.py`).
* **A thermostat chosen by measurement, with its cost written down.** Branch B runs a
  Nosé–Hoover chain at a 20 fs coupling — `openmmtools`' own default, and the value in the
  QHA reference paper. It is calibrated against a *closed form*, not against other
  thermostats: on a harmonic surface built from the production potential's own Hessian the
  conventional 100–1000 fs coupling returns the softest mode near 229 cm⁻¹ against a true
  79.7, while reporting a perfectly correct temperature.
* **Curvature training** with Hessian-vector products instead of Hessians.

### To-do

* **`refine=opt` vs `sp` is measured and awaiting a decision.** On `OCCC(=O)CO` under
  GFN2, `opt` finds 23 basins to `sp`'s 37 (union 45), costs 7.7× the wall clock, and
  errs by 0.40 kcal/mol against a ~0.1 target. Propanal, with 2 basins, showed no
  difference at all. The production value stays `opt` pending a ruling.
* nothing has been submitted to Tianhe; three scheduler command names unverified

---

## Dependencies

### Python
* **numpy**, **scipy**
* **ase** — structures, optimisers, xyz I/O
* **rdkit** — SMILES, automorphism-minimised RMSD
* **qc-procrustes** — the chirality test behind the enantiomer degeneracy g′ (rotational vs
  orthogonal Procrustes, `meng2022procrustes`); our Kabsch is asserted equal to it on every pair
* **torch**, **mace-torch** — the potential
* **pymsym** — point-group labels (σ itself comes from geometry)
* **PyYAML**, **pandas**, **pyarrow**
* **parsl** — branch E only; one molecule runs without it
* **matplotlib**, **jupyter** — tutorials
* **h5py** — branch C datasets
* **openmm**, **openmm-torch**, **openmmtools** — **core** for branch B: its production
  route and its Nosé–Hoover chain. Conda-forge only; `pip install openmm` does not give a
  working build
* **MDAnalysis** — **core**: the independent implementation behind acceptance criterion 2
* **mdtraj** — an extension. Read `openqha/mdtraj_io.py` before trusting it: it cannot
  mass-weight, and its float32 solver silently returned the identity rotation for 300 of
  3125 frames

### What this installation can actually do

Probed, never assumed — a broken build must read as unavailable, so the probes are real
imports and real executions:

```bash
python -c "from openqha import capabilities; print(capabilities.summary())"
```

The contract follows ACEsuit/mace's: locally a missing **extension** skips, and a run that
*declares* it fails instead.

```bash
S0_REQUIRE_CAPS=gromacs python tests/run_tests.py    # a skip is now an ERROR
```

That second line exists because acceptance criterion 2 twice reported "no comparison
produced" for reasons it does not measure. A criterion that goes quiet instead of red
teaches people to ignore it.

### External programs
* **crest** ≥ 3.0.2 — conformer search
* **xtb** — the GFN2 workhorse, and branch C's labels
* **gromacs** — an **extension**, not a requirement. `openqha/extensions/gromacs.py`;
  no number depends on it. MDAnalysis is the routine independent check and agreed to
  **8.0e-09 kcal/mol** where GROMACS gave 1.27e-04, without needing an external binary
* **orca** ≥ 6.0 — branch C RI-MP2 reference labels (registration required)

### The potential's weights

MACE-OFF weights are **not in this repository** — they are large binaries that belong
beside the run. `bash install_dependency.sh` downloads them for you into
`data/potentials/`, flat, keeping their own filenames.

Loading one is three steps and there is no fourth: `openqha/potentials/engine.py` maps an
engine **name** to a **filename**, looks for that filename in that one directory, and uses
the file it finds. Nothing can refuse a file for what it contains (beyond the physical
ceiling on its numbers). What every product records is the engine name, the path, and
the **parameter fingerprint**: a SHA-256 over the model's `state_dict` sorted by tensor
name (name, dtype, shape, raw bytes). A file hash would answer "the same bytes?"; the
fingerprint answers "the same numbers?" -- a `torch.save`/`load` round trip changes the
byte count and the file hash and leaves the fingerprint alone, while one changed
parameter changes it (`tests/unit/t_engine_fingerprint.py`). The registry may pin a
fingerprint (`params_sha256`; the production default is pinned); `provenance()` then
writes `params_pin_status` = matches / differs / unpinned into the record and warns on
stderr -- reported, never enforced.

**Registering a fine-tuned potential** (the Hessian-learning workflow produces one):

```bash
cp <run>/<name>.model data/potentials/                      # flat, keep the name
python scripts/tooling/s0_check_weights.py --pin data/potentials/<name>.model
```

paste the printed entry into `engine.ENGINES` with `source` = the Dataset index it was
trained on plus the training config's SHA, and select it with `S0_ENGINE=<name>`. Every
Record made with it then carries its fingerprint, so a frame's MACE labels are tied to
the exact numbers that produced them. `s0_check_weights.py` with no arguments prints the
three lines (bytes, file hash, fingerprint) for every registered file present;
`--json` + `--compare` settle in one command whether two machines hold the same numbers.

**The mace fork.** Fine-tuning needs two small changes inside mace-torch (a Hessian field on
the batch, an external-loss hook), so the mace this repository imports is the fork
`BloomDlwlrma/mace`, branch `openqha-hessian`, whose base tag `base-v0.3.16` is upstream
v0.3.16 minus three bundled foundation-model binaries -- mace-md's pattern
(`jharrymoore/mace@softcore`): what must touch mace's internals is a fork branch, everything
else (the Hessian-vector product, the probes, the loss, the judge -- the `openqha-hessian`
package) uses the public API. One script installs both the fork and the `openqha-hessian`
package, editable, in the active environment:

```bash
bash ../openQHA-Hessian/install.sh          # the fork + the package, editable (offline: pass a checkout path)
python scripts/tooling/s0_check_weights.py  # which mace answers: version 0.3.16+openqha, fork commit
```

Run it last -- or again after any re-run of `install_dependency.sh`: that installer
reinstalls the fork from its requirement line (non-editable).

`engine.provenance()` records `mace_fork_commit` (and `mace_fork_dirty`); `05_train` refuses
to train on a fork it cannot name. Machines that only evaluate tolerate a mace wheel, with
that commit recorded as `unknown`. Effort: `.scratch/hessian-learn-framework/`.

Verify what you have:

```bash
python check_dependency.py
```

It reports each dependency together with **what stops working without it**, and ends
with a single verdict: whether branch A can run.

---

## Installation

```bash
bash install_dependency.sh              # local workstation
bash install_dependency.sh --tianhe     # Tianhe or a site like it
bash install_dependency.sh --minimal    # smallest set that runs branch A
bash install_dependency.sh --check      # report only, install nothing
bash install_dependency.sh --no-weights # skip the MACE-OFF download
```

Or by hand — **one environment, everything in it**:

```bash
conda env create -f environment.yml        # CPU: branches A and B
conda activate openqha
```

For **branch C training only**, where a GPU is genuinely faster:

```bash
conda env create -f environment-cuda.yml   # CUDA 12, for Tianhe's AI partition
conda activate openqha-cuda
```

| | file | for |
|---|---|---|
| `openqha` | `environment.yml` | branches A and B. CPU, and that is not a compromise |
| `openqha-cuda` | `environment-cuda.yml` | branch C training. CUDA 12, pinned — check `nvidia-smi` reports 12.x |

Do **not** run the conformer search on a GPU partition. Branch A's cost is ~10⁵ GFN2-xTB
gradient calls and xtb never touches a GPU; branches A and B are both measured *slower*
on this project's GPU than on its CPU. On Tianhe that choice also bills a whole card for
the 14 CPUs that come with it.

Built and run, not merely written: 3m20s with the libmamba solver, **3.2 GB**, and
`dsgdb9nsd_000018` through the full analysis chain with 9/9 acceptance criteria. Against
the two-environment layout it replaces — same code, same reused CREST directory, minutes
apart — the electronic energy is identical to 4.6e-10 eV and `G − Eel` differs by
**2.2×10⁻⁴ kcal/mol**, 230× below the CREGEN energy condition. That residual is not the
BLAS: eigenvalues of a fixed matrix agree to 1e-14 and the potential is bit-identical on
a fixed geometry. It is the optimiser converging to a different point *inside* its own
`fmax = 1e-4 eV/Å` tolerance, which the softest vibrational mode reports first.

### One environment, and the OpenBLAS warning

conda-forge's CREST pulls the **pthreads** build of OpenBLAS while CREST itself is
OpenMP-parallel, and OpenBLAS then prints *"Detect OpenMP Loop and this application may
hang"* on every step. `environment.yml` pins the **OpenMP** build instead, which removes
the mismatch at its source.

Measured on `dsgdb9nsd_000018`, `--gfn2 --noreftopo -T 4`, same input throughout — both
binaries report the same 2 conformers, so this is cost and noise, not a different answer:

| environment | `OPENBLAS_NUM_THREADS` | wall | output lines | OpenBLAS warnings |
|---|---|---|---|---|
| **one env, `openmp_*` OpenBLAS** | unset | **18.5 s** | 745 | **0** |
| one env, `openmp_*` OpenBLAS | 1 | 18.4 s | 749 | 0 |
| old split env, `pthreads_*` | unset | 38.0 s | 4910 | 4164 |
| old split env, `pthreads_*` | 1 | 25.1 s | 747 | 0 |

The pin is the fix: with it, **no environment variable is needed**, and it is still 1.35×
faster than the pthreads build with the variable set. `env_openqha.sh` exports
`OPENBLAS_NUM_THREADS=1` anyway, because it costs nothing here and it is what protects a
CREST that came from somewhere other than this file.

> This repository used to ship **two** conda environments to avoid that warning. The
> measurement behind that decision changed two things at once — it separated the
> environments *and* set the variable — and credited the whole difference to the
> separation. Isolating them showed the separation was never doing anything: the bottom
> two rows above are one binary, one input, one variable. The two environments are gone;
> `environment-crest.yml` is in [`_superseded/`](_superseded/) with the numbers that
> retired it.

### Environment

```bash
source env_openqha.sh          # written by install_dependency.sh
```

or by hand:

```bash
export OPENQHA_ROOT=/path/to/openQHA
export PYTHONPATH=$OPENQHA_ROOT:$PYTHONPATH
export S0_MACE_ROOT=$OPENQHA_ROOT/data/potentials
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

> **Do not `set -u` before sourcing.** The conda GROMACS activation hook fails under it
> and leaves the environment half-built *without stopping* — measured, and it once made
> a working GROMACS look absent.

---

## Data

| what | size | shipped? |
|---|---|---|
| 7 reference geometries + a 7-row index excerpt | 11 KB | **yes** — the core deliverable reproduces with no external data |
| QM9 (133 885 molecules) | ~340 MB | no — `scripts/tooling/s0_prepare_data.py` |
| curatedQM9 (repaired geometries) | ~200 MB, 133 661 files | no — unpack under `data/qm9/`, **or one file**: `scripts/tooling/s0_pack_curated_qm9.py` writes it into `data/qm9/curated_qm9.h5` — one group per molecule (`dsgdb9nsd_%06d`: species, positions, Mulliken charges, frequencies, both SMILES and InChI, and the file's text); a cluster gets that one file and `curated_qm9.find()` extracts a molecule on demand |
| MACE-OFF weights | ~120 MB | no — downloaded by the installer, hash-checked on every load |

**curatedQM9** repairs the molecules whose deposited QM9 geometry is not the molecule its
SMILES describes. Whether you use it is your choice, and the `f7_mode` switch is where
you make it:

```yaml
f7_mode: "drop_all"   # remove all 3054 — conservative and lossy (default)
f7_mode: "curated"    # use the repaired geometry; drop only the 224
f7_mode: "off"        # debugging only
```

`"curated"` **refuses to run** if the archive is not present, rather than silently
falling back — a mode that quietly does something else is not a mode.

> If you set `f7_mode: "curated"`, cite curatedQM9: Senthil, Chakraborty & Ramakrishnan, *Chem. Sci.* **2021**, 12, 5566. https://doi.org/10.1039/D0SC05591C


---

## Running

```bash
python scripts/production/s0_A_pipeline.py --species dsgdb9nsd_000018
python scripts/production/s0_A_pipeline.py --smiles "OCCC(=O)CO" --label dhb
```

Fan out over many molecules:

```bash
python scripts/production/s0_E_branchA_parsl.py --edges --resource local
python scripts/production/s0_E_branchA_parsl.py --edges --resource deimos --account XXX
```

### Where the results are

One directory per molecule under a tag -- the tag directory is flat since 2026-09-20
(the two shard layers of 2026-09-14 are gone) -- with one folder per engine inside it and
the engines' own files in those folders (`docs/output_inventory.md` sections 6 and 8); a
frame's reference label is a file group of the molecule directory (ADR 0001, amendment 3):

```
<root>/<tag>/dsgdb9nsd_000018/
  crest/               CREST's working directory, verbatim
  mace/basinNN/        basin.extxyz  hessian.npy        (mace/confNN/: every relaxation)
  md_openmm/basinNN/   start.pdb system.xml integrator.xml traj.dcd state.csv state.xml state.chk
  frames/              <generator>.<level>.extxyz  frames.{out,toml}  labels.<level>.{out,toml}
                       orca.<level>.<generator>_bBB_kK.{inp,out,hess,engrad}   one frame's label (a file group)
  msrrho/              orca.<level>.basinNN.{inp,out,hess,xyz}  thermo/<level>.<step>.*  crest_entropy/  xtb/
  _records/            this repository's Records, one per Calculation, in the form ORCA and
                       CREST use: a .out Report for a person (last line: terminated normally)
                       and a .toml Property file for a program ([Calculation_Status] first)
    branchA.out  branchA.toml
    md_openmm/basinNN/md.out  md.toml  driver.log
    md_openmm/collect.out  collect.toml  collect.dat      ensemble.out  ensemble.toml
                       (collect.dat is the Table: sections [trajectories] [blank] [assembly],
                        one comment per column above each header)
```

Another setting of the same basin keeps the same folders and puts the setting in the file
stem (`traj_s2.dcd`, `md_s2.toml`); a Batch (a driver over many molecules) leaves no record,
its Slurm log is the report. `<root>` is derived from the cluster's partition on Tianhe
(`hpc/env/root.sh`) and is `~/runs/openQHA` elsewhere; `S0_RUNS_ROOT` overrides it. Sample
Records: `examples/02b_qha_openmm_propanal/sample_records/`.

### Free energy from the basins, and the reference level

The msRRHO free energy (Pracht & Grimme, Chem. Sci. 2021, 12, 6551, assembled inside
openQHA with CREST's conventions: tau = 25 cm^-1, rotor moment capped by the mean principal
moment, entropy and Cp interpolated, imaginary modes inverted inside the -50 cm^-1 floor
while a mode below the floor excludes the basin) is one Calculation per level,
written to the molecule's level folder (ADR 0004):

```bash
python scripts/production/s0_thermo_msrrho.py --species dsgdb9nsd_000035 --tag propanal --step mace
python scripts/production/s0_thermo_msrrho.py --species dsgdb9nsd_000035 --tag propanal --step reference --nprocs 8
python scripts/production/s0_thermo_msrrho.py --species dsgdb9nsd_000035 --tag propanal --step hessian_compare
python scripts/production/s0_thermo_msrrho.py --species dsgdb9nsd_000035 --tag propanal --step compare
```

```
<molecule>/msrrho/thermo/                                  (flat since 2026-09-20, ticket 09b)
  mace-off23_medium.degeneracy.{out,toml}  mace-off23_medium.thermo_msrrho.{out,toml}
  wb97m-d3bj_def2-tzvppd.thermo_msrrho.{out,toml}  wb97m-d3bj_def2-tzvppd.merge_map.dat
  gfn2.thermo_msrrho.{out,toml}                            (the CREST --entropy seam, ticket 25)
  hessian_compare.{out,toml}  the MACE Hessian at the reference geometry vs the reference Hessian
  level_compare.{out,toml}  every level beside the reference and the experiment, every term
<molecule>/msrrho/orca.wb97m-d3bj_def2-tzvppd.basinNN.{inp,out,hess,xyz}   (engine files, a file group)
<molecule>/msrrho/crest_entropy/runNN/  msrrho/xtb/entropy_runNN/confKK/  (ticket 25's engines)
<molecule>/mace/basinNN/{hessian,forces}_at_wb97m-d3bj_def2-tzvppd.npy      (engine files)
```

**Hessian against Hessian, at one geometry.** `hessian_compare` evaluates the MACE
Hessian at the reference geometry of every kept basin and compares it with the ORCA
`.hess` as the MLIP-Hessian literature does (HIP `eval_horm.py`, PFT, Rodriguez 2025,
PHL, Deng 2025): element-wise MAE in eV/A^2, relative Frobenius error of the projected
matrix, sorted-index eigenvalue / frequency MAE (all modes and below 300 cm^-1), a
softening slope, HIP's eigenvector cosines and overlap error, and -- with no mode
assignment at all -- the MACE curvature along each DFT normal mode, `D = L_r^T K_e L_r`
(PHL's Hessian-vector product with the DFT modes as probes), turned into per-mode T*S.
Propanal, three basins, 10 s: Hessian MAE 0.026-0.033 eV/A^2, frequency MAE 2.9-4.7
cm^-1, softening slope 1.0003-1.0016 (none), cos v1 >= 0.992. The model error in S_abs
(+0.162 cal/mol/K) is not curvature: the curvature-only S_vib deltas are +0.003 / -0.20
/ -0.05 and `MODEL_ERROR_S_REF` is -0.02, while `MODEL_ERROR_S_CONF_PRIME` is +0.158 --
MACE puts the gauche pair 0.13 kcal/mol too low relative to cis. Thermochemistry is
compared at each level's own geometry (never at the reference geometry), and
`$dipole_derivatives` are not compared at the MACE level (no charges). `mode_match.match`
is not used here: its argmax pairing was built for covariance-vs-Hessian bases.

**Is the molecule in MACE-OFF23's training set?** Before a model-error tier is read as
generalisation, `level_compare` checks the molecule against the released MACE-OFF23
training data (Moore et al., Apollo doi:10.17863/CAM.107498: 951,005 SPICE-derived
training frames over 17,132 molecules, ~55 conformers each, and a 50,195-frame test file
split by frame, not by molecule; energies and forces only). Identity is canonical, not a
string match (the files' SMILES are atom-mapped with explicit H), at three declared
strictnesses -- isomeric, stereo removed, InChIKey connectivity -- and monomer frames are
counted apart from dimer frames. The index built from the two files is in the repository
(`data/training_sets/mace-off23_spice_index.dat`, 4.8 MB, 78 s to build), so the question
is answered without the 5.2 GB sources; the `[Training_Set]` block of `level_compare.toml`
and one sentence in the Report carry the answer, and both are absent (and say so) when
no index is available -- never a silent false. **All four shipped molecules with basins
are in it** (acetone, acetamide, propanal, N-methylformamide: 48-49 monomer conformers
each under `DES370K Monomers`, plus 2,000-5,300 dimer frames), so their model-error tiers
are in-distribution numbers; the three ring species (2-methyloxirane, cyclopropanol,
oxetane) are not, at any strictness (`data/training_sets/shipped_species_membership.dat`;
`scripts/tooling/s0_training_set_membership.py`).

**And the whole target set is not.** The same check over every curated QM9 molecule
that passes the branch A gate (`--set qm9-targets`, 133,661 read, 119,451 targets, 24 min;
Dataset `data/training_sets/qm9_targets_membership.{dat,toml}` with the SPICE provenance
and the gate configuration) finds **176 targets (0.15 %) in MACE-OFF23's training data**
-- 156 at the strictest identity, 176 at connectivity -- and none in the test file only.
The overlap is the DES370K monomer set showing through, and it lives at the small end:
by heavy-atom count 1..9 the targets are [3, 5, 9, 31, 124, 588, 2971, 16622, 99098] and
those in training [3, 5, 7, 14, 29, 32, 28, 35, 23]. So the four shipped molecules
(4 heavy atoms: 14 of 31 in training) are the exception; for the 8-9 heavy-atom bulk of
QM9 the model-error tiers are out-of-distribution numbers by construction. The SI
sentence is generated from `[Summary]`, never typed.

**What out-of-distribution looks like.** The three ring species run through the whole
chain (branch A, MACE msRRHO, the wB97M reference with the analytic Hessian, 156-223 s
per basin, `hessian_compare`, `level_compare`; tag `rings`): Hessian MAE 0.053-0.074
eV/A^2 against propanal's 0.026-0.033, frequency MAE 4.8-10.2 cm^-1 against 2.9-4.7,
and the low modes (< 300 cm^-1) off by 13 / 57 / 66 cm^-1 against 4-8.5 -- the mode
shapes are right (cos v1 >= 0.997, softening slope ~1), the curvatures are not. Oxetane's
ring-puckering mode is the largest single number in the project: 16.7 cm^-1 at wB97M
(the near-barrierless double well, planar geometry) against 82.6 at MACE, a 1.39
cal/mol/K error in S_abs from one mode, entirely in S_ref; for the rings the model error
is curvature, for propanal it was relative energy. That mode is also where the harmonic
model itself fails, so it is both the first label site for Hessian learning and the place
where the judge must separate model error from approximation error.

**Reference level.** `wb97m-d3bj_def2-tzvppd` is the level MACE-OFF23 was trained to
(SPICE), so "model error" is the model and not a level difference. ORCA input line:
`! wB97M-D3BJ def2-TZVPPD TightOpt Freq TightSCF`. **The analytic Hessian works** for this
meta-GGA range-separated hybrid in ORCA 6.0.1 (the `SCF Response` module runs; propanal
basin 0, 10 atoms, 8 cores: 226 s wall, 5 optimisation cycles, 0 imaginary modes,
measured 2026-09-16). `NumFreq` is the declared fallback for a build that refuses, and the
route actually taken is written into `merge_map.dat` (`hessian_route`) for every basin.
Production reference calculations run on deimos with ORCA 6.1.1; a local run is a dry run
and says so through the ORCA version in the Report. On propanal (three MACE basins) the
MACE level gives S_abs = 72.355 cal/mol/K against the experimental 72.75 (LBH set, see
`docs/cite/cite_openQHA.bib`, keys `li2016lbh`, `nist_webbook`, `frenkel1994`).

**Imaginary modes: one production policy, and the seam's own.** CREST 3.0.2 does not
have one rule for an imaginary mode but three (`src/entropy/thermocalc.f90:207-216` and
`thermo.f90:135-138`, measured with `crest --numhess` on propanal, 2026-09-16): a mode in
(ithr, 0) with ithr = -50 cm^-1 is inverted; a mode below ithr is **kept negative**,
carries zero entropy, and still enters the zero-point energy, H(T)-H(0) and Cp with its
negative frequency; nothing is ever refused. Propanal's third `--entropy` conformer has
-68.4 cm^-1 in CREST's own numerical Hessian and went into CREST's dS_bar with
S_vib = 5.035 cal/mol/K (its neighbours: 8.97, 8.47). Since 2026-09-25 (ticket 35)
openQHA runs **one production policy**: `invert_below` with the frequency floor
ithr = -50 cm^-1 -- a mode in [ithr, 0) is inverted, a mode below the floor excludes the
basin, which is listed with its reason -- and a mode with |omega| < 1 cm^-1 is dropped
from every sum ORCA-style (`CutOffFreq`), counted in `N_BELOW_FLOOR` with its value
recorded; three or more in one spectrum are the signature of an unprojected spectrum and
raise. The third policy `refuse` was removed with the ruling (`[Imaginary_Spread]`, the
spread over the policies, went with it); the GFN2 seam keeps `crest_native` (CREST line
for line), which is what closed its Hessian tier (dS_bar within 0.03 of CREST instead of
0.1). `[Calculation_Info].ITHR_POLICY` names the policy every record used. On propanal
the MACE and reference levels have no imaginary mode; at GFN2 the seam's kept-negative
conformer changes S_abs by 0.09 cal/mol/K, all of it that one conformer.

**A numerical reference level, and what the dry run found (ticket 32, 2026-09-17).**
`dlpno-ccsdt_cc-pvtz` is declared in `orca.LEVELS` with the hkuhpc keywords (`! DLPNO-CCSD(T)
cc-pVTZ cc-pVTZ/JK RIJK cc-pVTZ/C TightPNO TightSCF Opt NumGrad NumFreq`, `%mdci TCutPairs
1e-6`, `%loc AHFB`); ORCA 6 has no analytic gradient for any coupled-cluster method, so the
route is numerical end to end and `s0_thermo_msrrho.py --step write_jobs` writes the hkuhpc
bundle (`orca_jobs.py`: inputs, `worker.sh`, `run.sbatch`, README with the (6N)^2 point
count and the wall time at the measured 93 s per point). Before such a Batch is submitted,
`--step mode_curvature` measures the higher level's curvature along chosen reference modes
from a line of single points (`mode_curvature.py`; rise 1e-5 Eh at k = 1, half-step points
for the delta_q error bar, a 7-point wide profile below 30 cm^-1), with the reference and
the engine evaluated on the same points as self-checks. The dry run on propanal and the
three rings refuted the Batch as designed and taught three things that are now in the
records: (1) DLPNO energies carry ~5e-6 Eh of PNO noise between distinct geometries
(mirror-image points differ by that much), so a NumFreq at that level is noise on every
low mode -- canonical CCSD(T)/cc-pVTZ is smooth, refuses RIJK, and costs ~6 min a point
for 10 atoms; (2) the curvature of a level at a *foreign* geometry contains a gradient
term: RI-MP2/cc-pVTZ gives 136.2 cm^-1 for propanal's aldehyde torsion at its own minimum
(experiment 135.1) but 93 at the wB97M geometry 0.009 A away, canonical CCSD(T) -65 there;
a straight displacement along a torsion lengthens C=O at second order and a level whose
bond wants to be longer gains g dr, the same order as the harmonic rise -- so the Cartesian
Hessian at a fixed geometry (the Hessian-learning target, what `hessian_compare` measures at
x_r) and the harmonic frequency at each level's own minimum (what thermochemistry uses) are
two quantities, and both are reported; (3) the analytic wB97M Hessian's softest mode moves
131.3 -> 137.2 cm^-1 between ORCA's DefGrid2 and DefGrid3 at the same geometry (the
energies' finite-difference curvature says 140.8), i.e. ~10 cm^-1 of label noise with the
default grid. Every Hessian record carries `NOISE_FLOOR_CM`, the largest |eigenvalue| of
the rigid-body block of the *unprojected* mass-weighted Hessian: 5-30 cm^-1 for an analytic
Hessian at a tight minimum (the rotational block feels the residual gradient), the
finite-difference noise on top for a numerical one; a low-mode difference below the larger
floor of the two Hessians compared is unresolved, not model error.

```python
from openqha.store import basins, branch_a_property
basins.molecule_for("dsgdb9nsd_000018", tag="prod")     # the molecule directory
basins.read_basins("dsgdb9nsd_000018", tag="prod")      # the basins as ase.Atoms
rec = basins.read_record("dsgdb9nsd_000018", tag="prod")  # branchA.toml as blocks, or None
branch_a_property.relative_kcal(rec)                    # RELATIVE of every [[Basin]]
basins.census(tag="prod")                               # how many per chunk
```

---

## Documentation

|Document|Content|
|---|---|
| [`docs/tutorials/T01_…ipynb`](docs/tutorials/) | **practice** — a real conformer search, end to end |
| [`docs/tutorials/T01b_…ipynb`](docs/tutorials/) | **practice** — branch B: MD, the thermostat, both routes, end to end |
| [`docs/branchB_workflow.md`](docs/branchB_workflow.md) | branch B operationally: the prohibitions, the two routes, the acceptance criteria |
| [`docs/tutorials/T02_…ipynb`](docs/tutorials/) | **theory** — AD Hessians and Projected Hessian Learning |
| [`configs/`](configs/) | every parameter, classified and sourced |
| [`hpc/README.md`](hpc/README.md) | which machine runs what, and why |
| [`docs/tianhe_install.md`](docs/tianhe_install.md) | **installing on Tianhe** — the five site facts, exact pinned versions, and every failure seen so far |
| [`docs/tianhe_runbook.md`](docs/tianhe_runbook.md) | the Tianhe machines themselves: partitions, quotas, scheduler commands |

---

## Citation

If **openQHA** contributes to your research, please cite it, together with the methods it
is built on. Ready-made entries are in [`docs/cite/`](docs/cite/)
([BibTeX](docs/cite/cite_openQHA.bib), [RIS](docs/cite/cite_openQHA.ris)).

* **openQHA** — see [`docs/cite/cite_openQHA.bib`](docs/cite/cite_openQHA.bib).
  A paper for openQHA itself is not yet published; cite the software.

Methods it stands on:

* **CREST / iMTD-GC** — Pracht, Bohle & Grimme, *Phys. Chem. Chem. Phys.* **2020**, 22,
  7169. https://doi.org/10.1039/C9CP06869D
* **RMSD metadynamics** — Grimme, *J. Chem. Theory Comput.* **2019**, 15, 2847.
  https://doi.org/10.1021/acs.jctc.9b00143
* **GFN2-xTB** — Bannwarth, Ehlert & Grimme, *J. Chem. Theory Comput.* **2019**, 15, 1652.
  https://doi.org/10.1021/acs.jctc.8b01176
* **MACE** — Batatia, Kovács, Simm, Ortner & Csányi, *NeurIPS* **2022**.
  https://doi.org/10.48550/arXiv.2206.07697
* **MACE-OFF23** — Kovács et al. https://doi.org/10.48550/arXiv.2312.15211
* **Quasi-harmonic analysis** — Andricioaei & Karplus, *J. Chem. Phys.* **2001**, 115,
  6289; Rinaldo & Field, *Biophys. J.* **2003**, 85, 3485.

Data sets are cited **if you use them** — see [Data](#data) above.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0
International License.
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
