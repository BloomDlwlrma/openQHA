# `examples/`

What openQHA publishes is **the method**, not this project's eleven edges (plan_D
section 6.2). Someone who installs openQHA wants "give me a molecule, compute its
conformational entropy", not "rerun some group's eleven edges". But the worked example
still has to be here and has to run, because a method library with no runnable example is
the repository-scale version of a criterion nobody has built a failing case for.

## What exists now

| Directory | What it does | State |
|---|---|---|
| `01_crest_composite_acetone/` | acetone through the CREST composite calculator: GFN sampling with MACE refinement over a socket | **runs** |
| `02a_qha_openmm_acetone/` | **the real-molecule debug check.** Acetone end to end, a grid of branch B settings, and the closed-form estimator control. **One basin**, so it tests basin residence, not the ensemble | **runs** |
| `02b_qha_openmm_propanal/` | **the whole chain on a multi-basin molecule.** Propanal: 3 basins (one Cs, a degenerate gauche pair), branch A → branch B → collect → **F_conf** | **runs** |
| `02c_hessian_benchmark_levels/` | **the scale.** MACE-OFF vs GFN2-xTB vs RI-MP2/cc-pVTZ on the same molecules, energies and forces through to `G − E_el`. Answers "how far is the potential's own thermochemistry from the reference", which every other deviation is measured against | **runs** (MACE + GFN2 in seconds; the reference level is hours) |
| `02d_qha_frequency_identity/` | **can `ν(QHA)` stand in for `ω(Hessian)`?** Harmonic-limit control, real trajectory, and the hybrid spectrum priced against both. **Measured: yes for entropy, no for ZPE and enthalpy** — the obstacle is the frame count | **runs** |

Both new examples use the same two molecules as 02a and 02b, so their numbers can be
carried straight across.

### Two steps per example: one CPU job, then one GPU job

Every example is submitted twice. **Step 1 establishes the basins on the CPU cluster;
step 2 consumes them.** Step 2 never re-runs branch A — it finds the record in the basin
store and skips it.

```bash
# step 1 -- CPU. Branch A only. Minutes.
bash examples/run_chain.sh examples/02b_qha_openmm_propanal/branchA.conf deimos

# step 2 -- GPU. Everything else, starting from those basins.
bash examples/run_chain.sh examples/02b_qha_openmm_propanal/chain.conf ai
```

| example | step 1 (`branchA.conf`) | step 2 (`chain.conf`) | shared TAG |
|---|---|---|---|
| `02a` acetone | `deimos` | `ai` / `h100x` | `acetone` |
| `02b` propanal | `deimos` | `ai` / `h100x` | `propanal` |
| `02c` levels | `deimos` | **`deimos`** — ORCA has no GPU path | `02c_prod` |
| `02d` identity | `deimos` | `ai` / `h100x` | `02d_prod` |

**The two confs of an example share a TAG, and they must.** The basin store is keyed by
`(species, tag)`: a mismatch would send step 2 looking somewhere empty, and on a GPU
partition it then refuses — correctly, but for a reason that takes a while to see.

**Branch A cannot run on a GPU partition at all.** It is CREST + GFN2-xTB; `xtb` has no
GPU path, and the `openqha-gpu` environment contains neither `crest` nor `xtb`. A step-2
submission with no basins is therefore refused **on the login node**, before the
allocation is spent, and prints the step-1 command for that example. Running step 1 on
`PARTITION=debug` (30 minutes) first is free.

Locally there is only one step: `bash examples/run_chain.sh <chain.conf>` makes the basins
if they are missing and reuses them if they are not.

### How a run is put together

```bash
bash examples/run_chain.sh <conf>            # here, now — no scheduler
bash examples/run_chain.sh <conf> deimos     # TianheXY-CN, CPU,     3 days
bash examples/run_chain.sh <conf> debug      # TianheXY-CN, CPU,     30 min
bash examples/run_chain.sh <conf> ai         # TianheXY-A,  8 cards, 7 days
bash examples/run_chain.sh <conf> temp       # TianheXY-A,  8 cards, 30 min
bash examples/run_chain.sh <conf> h100x      # TianheXY-AI, 1 card,  3 days
```

Three files, and the split is what makes the output land where you expect:

| file | what it is |
|---|---|
| `run_chain.sh` | **submits.** Reads the conf, picks the `.slurm`, hands it to `sbatch`/`yhbatch`, prints the job id and exits. Does nothing heavy — it runs on a login node |
| `slurm/<partition>.slurm` | **one queue's directives.** `--partition`, `--nodes`, `--ntasks`, `--cpus-per-task`, `--time`, `--gpus`/`--exclusive`, and `--output`/`--error` into `logs/` |
| `chain_body.sh` | **the work.** Sourced by every `.slurm` and used directly for a local run, so the five queues can differ in allocation and cannot differ in what they compute |

It used to be one self-submitting file. That failed on the machine: the job's output
arrived on the **login node's terminal** instead of its `--output` file, so the prompt
never came back to submit step 2; `#SBATCH` lines cannot be parameterised, so
`--cpus-per-task` and `--exclusive` — which differ per cluster — were simply absent; and
the login node did real work (importing torch, loading the potential) before submitting.

`sbatch` on the CPU cluster, `yhbatch` on the GPU ones. Job output goes to
`logs/openqha_<name>_<jobid>.{out,err}` under the repository.

### `02a_qha_openmm_acetone/` is the debug check for the whole chain

```bash
python examples/02a_qha_openmm_acetone/s0_debug_realmole.py \
    --conf examples/02a_qha_openmm_acetone/debug_realmole.conf
```

It runs branch A → **every basin of acetone** → branch B per basin → the ensemble
free energy, over a grid configured in a `.conf` (the convention of
`00_QM9_reaction_eng/hkuhpc/REPT-dNN/search/*.conf`), and writes one row per cell:
length, interval, thermostat, atom set, frames, frames/DOF, mean T·S, spread, **F_conf**,
and the basin-crossing counts.

The point is the last two. A single-basin scan cannot show whether a setting that moves
T·S moves every basin *together* — in which case it changes the deliverable by nothing —
and it cannot show whether the trajectory stayed in its basin at all. Both need a real
molecule with all of its basins.

```
python examples/02a_qha_openmm_acetone/s0_qha_openmm_demo.py
```

It runs the ASE route in whatever interpreter you start it with, and looks for a second
one that has OpenMM, openmm-torch and openmmtools — `S0_OPENMM_PYTHON`, else the `qm9fe`
environment. If it cannot find one, the OpenMM half is **skipped with a message** rather
than quietly omitted.

The trajectories are 3 ps, which is a smoke length and is labelled as one: acceptance
criteria 1 and 5 are refused below 20 ps, because a 0.4 ps run once passed the saturation
criterion and the reason it passed was that it had not begun to rise.

```
python -m openqha.potentials.mace_server --socket /tmp/s0_mace_engrad.sock &
S0_MACE_SOCKET=/tmp/s0_mace_engrad.sock \
    python examples/01_crest_composite_acetone/s0_crest_acetone_demo.py
```

It needs CREST on the path and a MACE-OFF model in place — see the engine registry in
`openqha/potentials/engine.py`, which searches for the model root and refuses to run on a checksum
it does not recognise.

## Tutorials live in `docs/tutorials/`

Two notebooks, moved there 2026-09-04 because they are documentation rather than
worked examples of this project's own campaign:

| | What it is |
|---|---|
| [`../docs/tutorials/T02_openQHA_Practice_CREST_conformers.ipynb`](../docs/tutorials/T02_openQHA_Practice_CREST_conformers.ipynb) | **Practice.** A conformer search end to end — the published iMTD-GC protocol and its boundary, the composite calculator, and the tighten → deduplicate → Hessian chain that turns an ensemble into a basin list. Worked on `OCCC(=O)CO` (24 conformers → 23 basins) and `OCCO` (10 basins, σ varying between them). |
| [`../docs/tutorials/T03_openQHA_Theory_AD_Hessian_and_PHL.ipynb`](../docs/tutorials/T03_openQHA_Theory_AD_Hessian_and_PHL.ipynb) | **Theory.** Where a Hessian comes from in a MACE model, why the analytic one beats finite differences, and how to supervise curvature with Hessian-vector products instead of Hessians — Projected Hessian Learning, with the two corrections it needs and a variance criterion that can fail. |

```bash
export S0_CREST_BIN=/path/to/crest            # T02 only
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
jupyter lab docs/tutorials/
```

Both find the repository root by walking up for `openqha/__init__.py`, so neither
depends on how deep it sits — which is why moving them cost nothing. Both execute
against the repository's own modules, so every number they print is the number the
production pipeline would get, and re-running them after a change is worth doing:
executing T03 contradicted four statements in its own prose, and executing T02
contradicted a fifth.

## What does not exist yet, stated plainly

plan_D section 6.3 names two examples. Neither is complete:

- **`01_single_molecule/`** — one molecule from SMILES all the way to `S_QH`. The acetone
  example above covers the conformer-search end of that chain and stops before the
  molecular dynamics and the quasi-harmonic analysis. Branch B's driver
  (`scripts/production/s0_B_qha_trajectory.py`) and analysis
  (`scripts/production/s0_B_qha_analyse.py`) exist; the example that ties them together
  does not.
- **`02_qm9_isomerisation/`** — the eleven-edge worked example of this project.

They are listed here rather than left out so that the gap is visible. `git init` is
gated on both of these running end to end (plan_D section 7.3), so this list is the
remaining distance to that gate.

## Data

What ships with the repository is the reference geometries of 7 species (11 KB) and a
7-row excerpt of the index, which is what lets package 2 reproduce with no external data
(`D0-41`). The full QM9 set is placed by `scripts/tooling/s0_prepare_data.py` and does not
go into version control.
