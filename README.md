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
 │                       + Boltzmann       geometric      reject imaginary   │
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
* **Quasi-harmonic entropy** from unbiased trajectories, cross-checked against GROMACS.
* **Curvature training** with Hessian-vector products instead of Hessians.

### To-do

* `refine=opt` vs `sp` under the GFN2 workhorse — measured on propanal (same basins,
  4.2× the cost); a second molecule is running
* nothing has been submitted to Tianhe; three scheduler command names unverified

---

## Dependencies

### Python
* **numpy**, **scipy**
* **ase** — structures, optimisers, xyz I/O
* **rdkit** — SMILES, automorphism-minimised RMSD
* **torch**, **mace-torch** — the potential
* **pymsym** — point-group labels (σ itself comes from geometry)
* **PyYAML**, **pandas**, **pyarrow**
* **parsl** — branch E only; one molecule runs without it
* **matplotlib**, **jupyter** — tutorials
* **h5py** — branch C datasets

### External programs
* **crest** ≥ 3.0.2 — conformer search
* **xtb** — the GFN2 workhorse, and branch C's labels
* **gromacs** — branch B's independent cross-check only
* **orca** ≥ 6.0 — branch C RI-MP2 reference labels (registration required)

### The potential's weights

MACE-OFF weights are **not in this repository** — they are fetched, not shipped.
`bash install_dependency.sh` downloads them for you; `openqha/engine.py` holds the
expected **SHA-256 for every selectable model and recomputes it on every load**, so a
truncated download, a proxy that served an HTML error page, or a silently swapped
potential all fail loudly instead of quietly changing the level of theory that every
downstream number claims.

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
| curatedQM9 (repaired geometries) | ~200 MB | no — unpack under `data/qm9/` |
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

A campaign is 133 885 molecules, so results are **sharded**, not one directory each:

```
data/basins/<tag>/1_16000/1_4000/dsgdb9nsd_000018.basins.json
data/basins/<tag>/1_16000/1_4000/dsgdb9nsd_000018.basins.xyz
```

```python
from openqha import basin_store
basin_store.paths_for("dsgdb9nsd_000018", tag="prod")   # where it would be
basin_store.read("dsgdb9nsd_000018", tag="prod")        # the record, or None
basin_store.census(tag="prod")                          # how many per chunk
```

---

## Documentation

|Document|Content|
|---|---|
| [`docs/tutorials/T01_…ipynb`](docs/tutorials/) | **practice** — a real conformer search, end to end |
| [`docs/tutorials/T02_…ipynb`](docs/tutorials/) | **theory** — AD Hessians and Projected Hessian Learning |
| [`configs/`](configs/) | every parameter, classified and sourced |
| [`hpc/README.md`](hpc/README.md) | which machine runs what, and why |

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
