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
| `02d-2_qha_settings_array/` | **02d at many settings at once**: one row of `settings.tsv` per run, one card per row, packed into a Slurm job array of whole-node tasks. Length x sampling-interval grid by default | **submits** |

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
bash examples/run_chain.sh <conf> deimos     # TianheXY-CN, CPU, whole node (64 cores), 3 days
bash examples/run_chain.sh <conf> debug      # TianheXY-CN, CPU, whole node,            30 min
bash examples/run_chain.sh <conf> ai         # TianheXY-A,  GPU, per card (12 CPUs),    24 h
bash examples/run_chain.sh <conf> temp       # TianheXY-A,  same queue,                 30 min
bash examples/run_chain.sh <conf> h100x      # TianheXY-AI, GPU, per card (14 CPUs),    3 days
```

**Before `ai` or `temp`, once per shell:** `source /APP/u22/ai_x86/toolshs/set-XY-I.sh`.
TianheXY-A's login node has two Slurm controllers behind one prompt; the fine-grained
one (cards are the unit, `-G` mandatory, `--mem` forbidden, 1 GPU = 12 CPUs = 120 GB) is
the one this project uses, and `run_chain.sh` refuses to submit until `sinfo` shows it
is the one in effect. The whole story, with the measurements, is
[`docs/tianhe_runbook.md`](../docs/tianhe_runbook.md) §0 and §0b; the scheduler itself
is stock Slurm — [slurm.schedmd.com/overview.html](https://slurm.schedmd.com/overview.html)
— and the options this project relies on are quoted from its manual in §0b.

Three files, and the split is what makes the output land where you expect:

| file | what it is |
|---|---|
| `run_chain.sh` | **submits.** Reads the conf, picks the `.slurm`, sizes the allocation from the conf (`NODES`, `CPUS`, `GPUS`, `WALLTIME`, passed as flags that override the file's defaults), prints the exact command, hands it to `sbatch`/`yhbatch` and exits. Does nothing heavy — it runs on a login node |
| `slurm/<partition>.slurm` | **one queue's defaults.** `--partition`, `--nodes`, `--ntasks`, `--cpus-per-task`, `--time`, `--gpus` or `--exclusive` as the queue requires, and `--output`/`--error` into `logs/` |
| `chain_body.sh` | **the work.** Sourced by every `.slurm` and used directly for a local run, so the five queues can differ in allocation and cannot differ in what they compute |

**The GPU allocation is one job.** Driver and workers share it: `tianhe_a.py` sees
`SLURM_JOB_ID` and runs parsl inside the allocation, so a one-molecule example is
`--gpus=1 --cpus-per-task=12` and nothing else. Acetone's branch B is 1 basin × 3 seeds =
3 trajectories on one card. `S0_PARSL_NESTED=1` restores block submission for a campaign.

**Tags must match.** Every branch A product lands under `data/basins/<TAG>/`, where `TAG`
is the one in the `branchA.conf` that made it. A command that reads it must name the same
tag — `--tag 02c_prod` for 02c, `--tag 02d_prod` for 02d, `acetone`/`propanal` for 02a/02b.
`no branch A product for <species> under tag '<x>'` means exactly that: look in
`data/basins/` for the tag that exists.

**Storage.** TianheXY-CN and TianheXY-A share one filesystem (`/XYFS02/...`), so branch A
on `deimos` and branch B on `ai` use the same checkout and nothing is copied. TianheXY-AI
(`h100x`) is on `/XYAIFS00/`; `hpc/tools/xfer_tianhe_ai.sh` moves files there and back.

It used to be one self-submitting file. That failed on the machine: the job's output
arrived on the **login node's terminal** instead of its `--output` file, so the prompt
never came back to submit step 2; `#SBATCH` lines cannot be parameterised, so
`--cpus-per-task` and `--exclusive` — which differ per cluster — were simply absent; and
the login node did real work (importing torch, loading the potential) before submitting.

`sbatch` on the CPU cluster, `yhbatch` on the GPU ones. **The job is named from the
conf** -- `openqha_<species>_<chain>_<tag>`, and for `identity`
`openqha_<species>_identity_e<equil>_p<prod>_s<sample>_x<seeds>_nu<cuts>` -- and the
log is `logs/<that name>_<jobid>.{out,err}`, so a directory of logs reads as a table of
settings. `squeue -u $USER -o "%.10i %.70j %.2t %.10M"` shows the full name. The
submitted line is printed in full before submission.

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
| [`../docs/tutorials/archive/T03_openQHA_Theory_Projected_Hessian_Loss.ipynb`](../docs/tutorials/archive/T03_openQHA_Theory_Projected_Hessian_Loss.ipynb) | **Archived (superseded 2026-09-23, S0-C-53 / S0-C-64).** The projected Hessian loss for the msRRHO entropy — the Eckart projector, the reference modes, the entropy weights, the HVP estimator and its variance, verified on propanal. Kept as the record of the superseded design; the training loss is now PHL verbatim (see T04 / T05). |
| [`../docs/tutorials/T04_openQHA_Theory_Hessian_Surface_Learning.ipynb`](../docs/tutorials/T04_openQHA_Theory_Hessian_Surface_Learning.ipynb) | **Theory (2026-09-22, PHL verbatim).** The derivation under T05: what E/F training leaves free (Proposition 1), PHL's full-Hessian loss and its random-probe estimator (unbiased, its variance, exact with 3N unit probes, what it bounds by Weyl), what the held-out generator reads off the minimum, the HVP's gradient and cost, the masked objective, and the algorithm listings as the code stands. Nothing projected but the probe. |
| [`../docs/tutorials/T05_openQHA_Theory_PHL_Finetune_EFH_to_wB97M.ipynb`](../docs/tutorials/T05_openQHA_Theory_PHL_Finetune_EFH_to_wB97M.ipynb) | **Theory (2026-09-22).** The goal of the Hessian-learning set as PHL gave it: fine-tune MACE-OFF23_medium with E, F and Hessian-vector products to its own level at the molecules' minima — PHL's loss verbatim as the target (S0-C-53), basin frames only with the held-out generator as the extrapolation readout (S0-C-54), fixed-probe validation (S0-C-55), the Replay as a drawn file and the rows R0–R4 with R4 the one that runs (S0-C-56/57/60), `w_H` measured by the driver, the judge's gate on the matrix itself and its reference rows (S0-C-58/59), the five-step recipe. |
| [`../docs/tutorials/T02_openQHA_Theory_AD_Hessian_and_PHL.ipynb`](../docs/tutorials/T02_openQHA_Theory_AD_Hessian_and_PHL.ipynb) | **Theory.** Where a Hessian comes from in a MACE model, why the analytic one beats finite differences, and how to supervise curvature with Hessian-vector products instead of Hessians — Projected Hessian Learning, with the two corrections it needs and a variance criterion that can fail. |

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
