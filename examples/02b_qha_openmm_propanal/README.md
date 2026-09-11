# 02b — propanal: the whole chain on a multi-basin molecule

Branch A and branch B end to end, on a molecule where **the ensemble is real**.

```bash
bash examples/run_chain.sh examples/02b_qha_openmm_propanal/chain.conf                    # local
bash examples/run_chain.sh examples/02b_qha_openmm_propanal/chain.conf ai
bash examples/run_chain.sh examples/02b_qha_openmm_propanal/chain.conf h100x
```

Two modes, one file, both at **production settings**. There is no smoke mode: a short run
is not a smaller version of the answer, it is a different quantity that looks like one.

---

## Why propanal

Branch A, measured 2026-09-08 — 240.1 s, 9/9 acceptance criteria:

```
CREST reports 3 conformers; + 1 reference geometry = 4 frames in -> 3 basins

basin        E / eV      rel/kcal  sigma   pg   nu_min/cm-1   G-Eel/kcal
    0  -5259.188956        0.0000      1   Cs        128.44      36.0252
    1  -5259.152703        0.8360      1   C1         73.99      35.7700
    2  -5259.152703        0.8360      1   C1         73.96      35.7699
```

That is the textbook propanal surface: one **Cs** (syn) minimum and a **degenerate pair of
gauche** minima that are each other's mirror image. The identical energies are a free
check on the search — a run that found only one of the pair would be visibly wrong.

**Acetone has one basin**, so `F_conf ≡ 0` there and no setting can move it. Propanal is
the smallest molecule in this project's own edge set where the ensemble is real, and it is
acetone's partner in edge `C3H6O1N0_18_35` — not a toy picked for convenience.

From the electronic energies alone, before any entropy:

```
 basin     dE_el/kcal      dG/kcal   population
     0         0.0000       0.0000       0.6721
     1         0.8360       0.8360       0.1639
     2         0.8360       0.8360       0.1639

F_conf = -0.2354 kcal/mol over 3 basins, 2.36 effective
```

**The point of the chain is what T·S does to that number.**

---

## What the chain runs

| step | driver | what it produces |
|---|---|---|
| 1 | `scripts/production/s0_A_pipeline.py` | every basin, with σ, point group, ν_min, ΔE_el |
| 2 | `scripts/production/s0_E_branchB_parsl.py` | one trajectory per (basin, seed) — `--basins auto` |
| 3 | `scripts/production/s0_E_branchB_collect_parsl.py` | the quasi-harmonic analysis per molecule |
| 4 | `scripts/production/s0_B_report_ensemble.py` | **F_conf over the ensemble** |

`run_chain.sh` adds **no science**. Every step is the production driver, so the example
cannot drift from what a campaign does.

### `--basins auto`, and the defect behind it

Until 2026-09-08 the Parsl branch B driver **never passed branch A's basin file**. A run
with `--basins 3` launched three tasks that all started from the same QM9 reference
geometry — three copies of one basin under three indices, and an `F_conf` that looked
like a multi-basin answer and was not. Nothing failed; the trajectory driver recorded
`geometry_source` in a per-trajectory `meta.json` that nothing read.

Now the count and the geometries come from the same place — branch A's own record — so
they cannot disagree, and a species with no branch A product is reported rather than
quietly run from the reference geometry.

---

## Modes

| | `MODE=local` | `MODE=hpc PARTITION=ai` | `MODE=hpc PARTITION=h100x` |
|---|---|---|---|
| cluster | this machine | TianheXY-A | TianheXY-AI |
| resource config | `local` | `tianhe_a` | `tianhe_ai` |
| route | ASE | OpenMM, CUDA | OpenMM, CUDA |
| allocation | — | **one card**, 12 CPUs, 120 GB (fine-grained; `set-XY-I.sh` first) | **one card**, 14 CPUs |
| walltime | — | 7 days | 3 days |

### The job gets its settings by `source`, not `--export`

`--export=ALL,VAR=x` carries the submitting shell's whole environment into the job and
patches names on top. What the job saw then depends on which login shell submitted it, and
the settings survive only in a scheduler record nobody reads.

Here the conf path is the script's **argument** — Slurm passes a batch script its arguments
unchanged — and the job sources it. The settings are a file that can be diffed, committed
and re-run.

`run_chain.sh` submits `examples/slurm/<partition>.slurm`, whose `#SBATCH` lines are the
queue's defaults; the conf's `GPUS`/`CPUS`/`NODES`/`WALLTIME`, when set, are passed as
flags and override them, and the full command is echoed before submission. Propanal's
branch B is 3 basins × 3 seeds = 9 trajectories: still one card (`GPUS=1`, up to 12
workers), so the defaults need nothing from this conf.

---

## Where the answer lands

```
data/basins/<tag>/…/dsgdb9nsd_000035.basins.{json,xyz}   branch A: the basins
$S0_RUNS_ROOT/qha/<tag>/dsgdb9nsd_000035/basinNN/seedNN/  branch B: frames + meta
analysis/qha/<tag>/                                       per-molecule analysis
analysis/qha/<tag>/dsgdb9nsd_000035_ensemble.json         F_conf, populations, ΔG
```

Read in this order:

1. **`distinct_crossings`** — non-zero means a trajectory left its basin and its T·S is
   inflated, however converged the saturation curve looks.
2. **`effective_basins`** — 3 basins of which one holds 99% is a one-basin molecule
   wearing a costume. Propanal's electronic-only value is 2.36.
3. **`F_conf_kcal`** — and only if the two above are sound.

A basin with **no** entropy is reported as `MISSING`, never as zero, and the script exits
non-zero. "We did not run it" and "its entropy is zero" are different statements and only
one of them is ever true.

---

## Cost

Production is 50 ps equilibration + 500 ps at one frame per 1.0 ps
(`configs/branchB_protocol.yaml`). Per trajectory, at the only figure this repository has
— 96.1 s/ps, one CPU thread, uncontended — that is ~15 hours, and the chain runs
`3 basins × 3 seeds = 9` of them.

**That is a CPU number quoted for a GPU job**, because it is the only measurement that
exists; this repository's one GPU figure for branch B is 3.5× *slower* than CPU on a T400,
which does not transfer to an 80 GB card and has not been replaced. Read
`seconds_per_ps_this_run` out of the first `meta.json` and plan from that.
