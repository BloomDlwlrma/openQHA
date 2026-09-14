# Branch A — conformer search: design, workflow, and how to run it on Tianhe

**One molecule → one basin list.** This document is the operational description: what
runs, in what order, with which parameters, and how to put it on a cluster.

- Parameters: [`configs/conformers.yaml`](../configs/conformers.yaml) — the authority.
- Argumentation: `.mem/plan/plan_A_conformational-search.md`
- Measurements: `.mem/checkpoints/checkpoints_A_conformational-search.md`
- Dynamics protocol, side by side with branch B:
  [`configs/mdp/s0_branchA_crest_metadynamics.mdp`](../configs/mdp/s0_branchA_crest_metadynamics.mdp)

---

## 1. What comes out, and who consumes it

A **basin list**. Each basin carries:

| field | criterion |
|---|---|
| geometry | converged on MACE-OFF23_medium to `fmax = 1e-4 eV/Å` |
| electronic energy | on that geometry, `float64` |
| analytic Hessian | zero imaginary frequencies, Eckart-projected |
| symmetry number σ | from the **geometry**, per basin |
| RRHO terms | `G − E_el`, four terms, each recoverable |
| Boltzmann weight | at 298.15 K |

It is the **only** structure source for the rest of the project: branch B starts its
unbiased trajectories from these geometries; branch C computes RI-MP2 labels on them.

**The interface between branches is the geometries — never the trajectory.** Section 7
is why.

---

## 2. The six steps

```
  SMILES or QM9 index
        │
  ┌─────▼──────────────────────────────────────────────────────────────┐
  │ 1. GATES F0–F7            openqha/conformer_search/filters.py                       │
  │    short-circuit, so the failing gate is unique                    │
  └─────┬──────────────────────────────────────────────────────────────┘
        │  passes
  ┌─────▼──────────────────────────────────────────────────────────────┐
  │ 2. CREST iMTD-GC          workhorse GFN2-xTB, quality layer MACE   │
  │    published protocol: SHAKE all bonds + 5 fs + H mass 2 amu       │
  │    ~1e5 energy+gradient calls; MACE sees ~2% of them               │
  └─────┬──────────────────────────────────────────────────────────────┘
        │  crest_conformers.xyz          ← CREST's count, NOT a basin count
  ┌─────▼──────────────────────────────────────────────────────────────┐
  │ 3. POOL the reference geometry   the only non-CREST starting point │
  └─────┬──────────────────────────────────────────────────────────────┘
        │
  ┌─────▼──────────────────────────────────────────────────────────────┐
  │ 4. TIGHTEN to fmax = 1e-4 eV/Å   tighter than CREST's optlev=tight │
  │    the step COUNT is a result: how far the handover sat from our   │
  │    minimum. Measured 34–120 steps on structures already "converged"│
  └─────┬──────────────────────────────────────────────────────────────┘
        │
  ┌─────▼──────────────────────────────────────────────────────────────┐
  │ 5. DEDUPLICATE — CREGEN three-fold criterion, all three required   │
  │      all-atom best RMSD  < 0.125 Å      (CREST RTHR)               │
  │      |ΔE|                < 0.05 kcal/mol (CREST ETHR)              │
  │      rotational constants within 1%      (CREST BTHR)              │
  └─────┬──────────────────────────────────────────────────────────────┘
        │
  ┌─────▼──────────────────────────────────────────────────────────────┐
  │ 6. ANALYTIC HESSIAN → reject imaginary → σ per basin → RRHO → w_i  │
  └─────┬──────────────────────────────────────────────────────────────┘
        │
   basins.json + basins.xyz
```

### Why steps 4–6 exist at all

**A search program's conformer count is not a basin count on your surface.** Two
criteria differ — convergence and duplicate detection — and one physical fact does too:
a converged gradient is not a minimum.

Measured, defect 54: on acetone CREST reported **2 conformers 0.8118 kcal/mol apart**;
tightened to `fmax = 1e-4` their energies were **identical to the last digit**. That
0.8118 was CREST's convergence residual, not an energy difference.

And package 2 once found a "basin" of cyclopropanol with a **−195.79 cm⁻¹** imaginary
frequency. Saddle points have zero gradient too.

---

## 3. The complete CREST parameter set

Everything below is in `configs/conformers.yaml`. Classification: `published_protocol`
= verbatim from a paper; `modifiable_convention` = a choice whose consequence must be
reported; `resource_budget` = affects speed only.

### 3.1 What runs

| parameter | value | class | source |
|---|---|---|---|
| `runtype` | `imtd-gc` | modifiable_convention | PCCP 2020, 22, 7169 |
| `optlev` | `tight` | modifiable_convention | upstream ladder crude…extreme |
| `threads` | 4 | resource_budget | also the condition the 285 s/species cost was measured under |

### 3.2 Workhorse and quality layer

| parameter | value | class | why |
|---|---|---|---|
| `workhorse` | **`gfn2`** | published_protocol | upstream example 1 is `crest struc.xyz --gfn2` |
| `refine` | **`sp`** | user ruling 2026-09-04 | quality layer scores survivors; it does **not** re-optimise and merge them |
| `backend` | `generic` | — | this repo's socket client; `mlip` needs CREST 3.1 |
| `engine_client` | `scripts/production/s0_mace_engrad.py` | — | one resident server per parallel slot |

**Why `gfn2` and not the cheaper `gfnff`** — measured end to end, same species, same
`refine` setting, both zero aborts:

| workhorse | wall | energy+gradient calls |
|---|---|---|
| **gfn2** | **285.5 s** | **86 834** |
| gfnff | 352.1 s | 116 535 |

A better workhorse converges in fewer calls and hands fewer structures to the expensive
refinement. One species does not overturn `D0-50` in general; it removes cost as an
objection here.

> **Settled 2026-09-04.** `refine=opt` vs `sp` was previously measured only under
> **gfnff**, and the 2026-09-03 ruling chose `opt` on the stated grounds that those
> numbers no longer applied under gfn2. Acceptance criterion 3 tested that premise and it
> did not hold. Under gfn2, on `OCCC(=O)CO`:
>
> | refine | CREST wall | conformers | basins | error vs union |
> |---|---:|---:|---:|---:|
> | `opt` | 5980.0 s | 24 | **23** | **0.4023 kcal/mol** |
> | `sp` | 776.6 s | 82 | **37** | 0.0025 kcal/mol |
>
> Union of both, pooled and re-deduplicated: 45 basins. `opt` loses 14 of the 37 basins
> `sp` finds, costs 7.7×, and errs by 0.40 kcal/mol — above this branch's ~0.1 target and
> 8× the CREGEN energy condition. **Propanal showed none of it** (2 basins each, error
> 0.0000): a two-basin molecule has no room to lose fourteen, which is the same trap as
> using single-basin molecules to probe the deduplication plateau.
>
> **Why `opt` loses them** — measured, not assumed. Sampling is identical (28 MTD blocks,
> 2 iterations in both runs); the divergence is entirely in CREST's screening, and it
> starts at the first screen (44 survivors against 91). Two components, not yet separated
> from each other: relaxing on MACE before CREGEN merges structures, and the 6 kcal/mol
> energy window bites on MACE energies under `opt` and gfn2 energies under `sp`
> (retention 11.2% against 19.2%). Both happen **inside CREST, before the ensemble
> reaches us**, so nothing downstream can recover what was discarded. See
> `scripts/calibration/s0_A_refine_mechanism.py` and checkpoint 9.

### 3.3 The dynamics package — three settings, one protocol

> Grimme, JCTC 2019, 15, 2847, verbatim: *"the SHAKE algorithm for constraining **all
> covalent bonds** … with an MD time step dτ of **5 fs** … The atomic mass of hydrogen is
> set to **2 amu (deuterium)**."*

| parameter | value | class |
|---|---|---|
| `shake` | **2** (all bonds) | published_protocol |
| `tstep_fs` | **5.0** | published_protocol |
| `hydrogen_mass_amu` | **2.0** | published_protocol — *recorded, not set*: already CREST's default (`src/confparse.f90:215`) |

**These three are one package.** The 5 fs step is stable only because bonds are
constrained *and* hydrogens are heavy. Four-arm attribution, one variable at a time,
`dsgdb9nsd_000019`:

| arm | workhorse | shake | dt | `terminated EARLY` | wall | engrad |
|---|---|---|---|---|---|---|
| A | gfn2 | 1 | 5 fs | **29** | 48.6 s | 5 837 |
| B | gfn2 | 1 | 2 fs | 0 | 285.5 s | 86 834 |
| **C** | **gfn2** | **2** | **5 fs** | **0** | **283.4 s** | **45 463** |
| D | gfnff | 1 | 5 fs | 0 | 352.1 s | 116 535 |

B and C cost the same, so this is **not a cost decision**. C is the published,
benchmarked protocol; B is a compromise nobody has validated.

**Its cost, recorded not hidden** — `shake=2` diverges on strained rings:

| molecule | SMILES | shake=2 | shake=1 |
|---|---|---|---|
| `dsgdb9nsd_000607` | `O[C@H]1C[C@H]1C#N` | 31 aborts | 0 |
| `dsgdb9nsd_003163` | `O[C@H]1CC[C@]21CO2` | 24 aborts | 0 |
| `dsgdb9nsd_002334` | `CCC[C@H]1C[C@H]1C` | 4 aborts | 0 |

Hence `shake_fallback`: **pointwise**, applied only to a molecule that actually aborted,
retried once at `shake=1` **in a sibling directory**, and marked with `shake_used` /
`used_shake_fallback` in its own record.

> **Open:** the fallback has never fired. Five molecules, none aborted. Written and
> unproven — the three rings above are the test it needs.

### 3.4 Basin criteria (after CREST)

| parameter | value | class | source |
|---|---|---|---|
| `tighten_fmax_eV_A` | `1e-4` | numerical_tolerance | same as the Hessian criterion |
| `dedup_criterion` | `cregen_three_fold` | published_protocol | CREST CREGEN |
| `dedup_rmsd_A` | **0.125** | published_protocol + measured plateau | CREST `RTHR` |
| `dedup_ethr_kcal` | **0.05** | published_protocol | CREST `ETHR` |
| `dedup_bthr_relative` | **0.01** | published_protocol | CREST `BTHR` lower bound |
| `dedup_metric` | `all_atom` | measured departure | heavy-atom misses H-only rotors |
| `hessian_mode` | `analytic` | correction, not optimisation | defect 64 |
| `symmetry_tolerance_A` | 0.10 | modifiable_convention | swept in every record |

**The threshold sweep that settled 0.125** (two molecules):

| threshold / Å | `OCCC(=O)CO` | `OCCO` |
|---|---|---|
| 0.05 – 0.25 | **23** | **10** |
| **0.125** | **23** | **10** |
| 0.30 (old) | 22 | 9 |
| 0.40 / 0.50 | 17 | 7 / 4 |

The plateau is a factor of five wide; 0.30 was the first value past its edge, and it
merged structures **2.2 kcal/mol apart**. CREGEN's own criterion returns the plateau
count independently. Three routes, one answer.

### 3.5 Environment — not optional, and not tuning

```bash
OPENBLAS_NUM_THREADS=1
OPENBLAS_MAIN_FREE=1
```

conda-forge's CREST links the **pthreads** OpenBLAS while CREST is OpenMP-parallel.
Measured on acetone, this variable the only change: **4.4 s → 2.7 s**, **3626 warning
lines → 0**, `crest.out` **476 KB → 34 KB**. This fixes a defect.

---

## 4. Running one molecule

```bash
export S0_CREST_BIN=/path/to/crest
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

# a species this repository ships
python scripts/production/s0_A_pipeline.py --species dsgdb9nsd_000018 --tag prod

# any molecule, by SMILES
python scripts/production/s0_A_pipeline.py \
    --smiles "OCCC(=O)CO" --label dihydroxybutanone --tag multibasin

# reuse a finished CREST directory, re-run only the analysis
python scripts/production/s0_A_pipeline.py --smiles "OCCO" --label ethyleneglycol \
    --tag multibasin --skip-crest --reuse ~/runs/openQHA/branchA/multibasin/ethyleneglycol
```

Products land in the molecule directory (`<root>/<tag>/_label/<label>/` for a SMILES
molecule; since 2026-09-14, `docs/output_inventory.md` section 6): `mace/confNN/`,
`mace/basinNN/`, and under `_records/`: `basins.json`, `basins.xyz`,
`driver.log`. Scratch stays under `$S0_RUNS_ROOT/branchA/<tag>/<label>/`.

### The SMILES path, and what it costs

A molecule with no deposited geometry needs *some* starting structure, so one RDKit
embedding (`n_embed = 1`) is relaxed on the potential. **This is not ETKDG returning as a
conformer search** — every conformer in the result comes from CREST's metadynamics, and
upstream's own example (`crest struc.xyz --gfn2`) needs a `struc.xyz` from somewhere too.

What *is* lost: there is no independent reference geometry to pool at step 3, so the
systematic error of a CREST-only ensemble (mean **+0.1209**, max **+0.5824** kcal/mol) is
**unbounded** for that molecule. Acceptance criterion 6 then reports **not applicable**
rather than passing.

### Acceptance criteria, checked against the product

| # | criterion |
|---|---|
| 1 | workhorse identity — read back from the work directory's `input.toml`, not from what we passed in |
| 2 | the cost claim is unambiguous (`wall_is_valid_cost` present either way) |
| 4 | zero imaginary, exactly 6 rigid modes removed, separation > 1e8 |
| 6 | reference geometry pooled and located — or explicitly **not applicable** |
| 7 | CREST / pooled / basin counts are three separate numbers **that add up** |
| 9 | σ is a group order, stable up to the working tolerance, positive margin |
| 10 | zero `terminated EARLY`, or a declared fallback |
| 11 | the dynamics package went in **whole** (two of three ≠ the published protocol) |
| 12 | no merge crossed the energy threshold; the criterion in force is recorded |

---

## 5. Branch E for branch A — the execution layer

**Branch E computes nothing.** It decides which machine runs branch A and how many
molecules run at once. Replacing it with a `for` loop must not change a single number.

### Resource profile

Conformer search is a **CPU workload parallel over molecules**:

- GFN2-xTB never touches a GPU;
- MACE calls are single 10–20 atom structures, which fill no card (`D0-56`).

**So it must not go on a GPU partition.** Tianhe's `h100x` bills by the whole card and
gives 14 CPUs with it — running branch A there burns an H100 to do CPU work *and*
starves the CPU work.

### Two nested parallelisms, not interchangeable

| level | knob | why |
|---|---|---|
| **between** molecules | Parsl workers | embarrassingly parallel |
| **inside** one CREST job | `threads = 4` | parallelises independent metadynamics runs |
| **inside** linear algebra | pinned to **1** | MACE thread scaling on 10 atoms: 111/90/72/**101** ms at 1/2/4/8 — eight is *slower* than four |

`workers × threads` must stay under the physical core count, or the measured cost of a
molecule stops meaning anything.

### One MACE server per worker

A server holds one model behind one lock. **N workers sharing one socket are not
parallel — they queue**, and nothing in the logs says so. Each worker starts its own
server on its own socket under `$S0_RUNS_ROOT/sockets/`, named by molecule and PID.

### Running it

```bash
source hpc/env/common.sh          # do NOT `set -u` first — see the file
python scripts/production/s0_E_branchA_parsl.py --species dsgdb9nsd_000018 --dry-run
python scripts/production/s0_E_branchA_parsl.py --edges --resource local
python scripts/production/s0_E_branchA_parsl.py --edges --resource deimos --account XXX
```

`--dry-run` prints the plan and submits nothing. The `settings` line it prints is read
back from `configs/conformers.yaml` — proof the execution layer has no science settings
of its own.

Cost is reported as **three separate numbers** and never divided: batch wall clock,
single-task seconds, and *slot extrapolation is explicitly not computed* (`D0-P1-12`;
defects 34 and 56).

---

## 6. Generating conformers on Tianhe

> **Status: the provider is written and nothing has been submitted.** Everything marked
> **UNVERIFIED** below must be settled on the login node before a production run. This
> section is the procedure, not a report.

### 6.1 Site facts (confirmed)

| fact | source |
|---|---|
| directives are `#SBATCH` — a Slurm derivative | site job scripts |
| submission **`yhbatch`**, launcher **`yhrun`** | site job scripts |
| `h100x` = 1 GPU / 14 CPU / 240 GB, billed **per card** | `D0-56` |
| a submission **without explicit `-G` fails** | site announcement |
| `--array` is available | user, 2026-09-03 |
| outbound network via proxy; no container; CREST from conda-forge | `D0-C-22` |
| `ulimit -l unlimited` and `GLEX_USE_ZC_RNDV=0` required | `D0-C-24` |

### 6.2 Step 0 — before anything is submitted

`D0-C-20`: **the first thing done on a cluster is a benchmark, not a production run.**
Prove the chain locally first; going to the cluster then changes one argument.

```bash
python scripts/production/s0_E_branchA_parsl.py --species dsgdb9nsd_000018 --resource local
```

### 6.3 Step 1 — settle the scheduler command names

Parsl hard-codes **four** command names, verified with `inspect.getsource` on the
installed 2026.08.10: `sbatch`, `sacct`, `squeue`, `scancel`. Only two of Tianhe's
replacements are attested.

```bash
python -c "import sys; sys.path.insert(0,'hpc'); import providers;
           import json; print(json.dumps(providers.preflight('tianhe'), indent=1))"
```

| role | configured | status |
|---|---|---|
| submit | `yhbatch` | confirmed |
| launcher | `yhrun` | confirmed |
| status | `sacct` | **UNVERIFIED** — candidates `yhacct`, `sacct` |
| status fallback | `squeue` | **UNVERIFIED** — candidates `yhqueue`, `squeue`, `yhinfo` |
| cancel | `scancel` | **UNVERIFIED** — candidates `yhcancel`, `scancel` |

**Why this matters more than it looks.** A wrong *submit* command fails immediately. A
wrong *status* command does not fail at all — Parsl believes every job is still pending
and **the queue quietly stops**. Edit `TIANHE_COMMANDS` in `hpc/providers.py` with what
`preflight()` finds; do not guess.

**No alias, and no shim named `sbatch` on `PATH`.** Both would work, and both would make
*how was this job submitted* something you cannot read off the record.

### 6.4 Step 2 — inspect the rendered script, then submit one job

```bash
python -c "
import sys; sys.path.insert(0,'hpc')
from providers import TianheSlurmProvider
p = TianheSlurmProvider('cpu', account='YOUR_ACCOUNT', nodes_per_block=1,
                        init_blocks=0, min_blocks=0, max_blocks=1, walltime='00:10:00')
print(p.render_only('/tmp/openqha_render_check.sh'))"
cat /tmp/openqha_render_check.sh
```

Acceptance criterion 3 is written against **the rendered script**, not the config: a
config that reads correctly and renders wrongly is exactly the failure it exists for.

If the partition needs a GPU count, pass **`gpus_per_node`**, not a `scheduler_options`
string — `SlurmProvider.__init__` accepts it and Parsl renders it into the template,
whereas a string is never checked.

### 6.5 Step 3 — a CPU partition for conformer search

Branch A wants CPU nodes, not `h100x`. Write `hpc/resource_configs/tianhe_cpu.py` by
copying `deimos.py` and changing three things:

```python
from providers import TianheSlurmProvider          # not SlurmProvider

provider=TianheSlurmProvider(
    partition,                    # the site's CPU partition
    account=account,
    nodes_per_block=1,
    init_blocks=0, min_blocks=0, max_blocks=max_blocks,
    scheduler_options="#SBATCH --ntasks-per-node=1",
    launcher=SimpleLauncher(),    # one worker pool per node, which forks its own workers
    worker_init=("source {0}/env/common.sh; source {0}/env/tianhe.sh; "
                 "openqha_report_env").format(here),
    walltime=walltime,            # 95% of the queue limit, so results are written out
    cmd_timeout=60,
)
```

`WORKERS_PER_NODE = CORES_PER_NODE // 4` — four CREST threads per molecule, no
oversubscription. **16 concurrent molecules on a 64-core node is a node capacity, not a
speed-up**: the 285 s/species figure is uncontended, and the per-molecule cost under
contention has not been measured.

### 6.6 Step 4 — submit

```bash
python scripts/production/s0_E_branchA_parsl.py \
    --edges --resource tianhe_cpu --account YOUR_ACCOUNT --partition CPU_PARTITION \
    --tag tianhe_prod --dry-run          # look at it first
python scripts/production/s0_E_branchA_parsl.py \
    --edges --resource tianhe_cpu --account YOUR_ACCOUNT --partition CPU_PARTITION \
    --tag tianhe_prod
```

### 6.7 Things that will bite

| | |
|---|---|
| **Walltime** | set to **95%** of the queue limit. The job must finish and *write out*, not be killed holding results (`D0-C-25`). Every molecule writes on completion, so a kill costs at most one molecule — provided nothing buffers. |
| **Scratch** | CREST writes many small files into parallel `_N` subdirectories. Use node-local `$TMPDIR`, copy **results** back — not the process. |
| **`set -u`** | `hpc/env/common.sh` must not use it. The conda GROMACS activation hook fails under `set -u` (`GMXRC: line 10: shell: unbound variable`) and the job then runs with a half-built environment **without stopping**. |
| **`OPENBLAS_NUM_THREADS`** | Must be `1`. conda-forge's CREST can link the pthreads OpenBLAS while CREST is OpenMP-parallel; unset, that costs 3992 warning lines and 11 s on a 10-atom molecule. One environment now, not two — `_superseded/environment-crest.README.md` has the numbers that retired the split. `S0_CREST_BIN` still overrides, for a CREST from elsewhere. |
| **`--array`** | available, and **not used**. Parsl queues through `max_blocks`. Recorded because it keeps `_superseded/cluster_2/s0_submit_array.slurm` open as a fallback shape if Parsl ever proves unworkable there. |

---

## 7. The boundary — the part to remember

Every setting in section 3.3 exists to **cross barriers**, and each buys that by
destroying the physical thermal fluctuations:

| setting | what it does to a quasi-harmonic analysis |
|---|---|
| bias potential | flattens the fluctuations the method measures → **entropy systematically high** |
| SHAKE all bonds | *removes* degrees of freedom → the `3N−6` non-zero eigenvalues **cannot all be filled** |
| H mass 2 amu | C–H stretches ~3000 → ~2200 cm⁻¹ → **every vibrational term wrong** |

Any one is fatal on its own. Branch A does not care, because **its product is a geometry,
not a trajectory**.

> **A CREST trajectory is a metadynamical trajectory full of non-physical bias. It
> cannot be used to compute real vibrational frequencies or entropies.**

The handover is **the basin geometries**. Branch B restarts from them with `dt = 1 fs`,
`constraints = none`, real hydrogen mass and no bias, and uses **not one frame** of the
CREST run.

The two stages share a molecule and a directory. They do not share a trajectory, and
they must not share an `.mdp`.

---

## 8. Open items

| | item | status 2026-09-04 |
|---|---|---|
| 1 | `refine=opt` vs `sp` under gfn2 | **paid, and the answer reversed on the second molecule.** Propanal: same 2 basins, error 0.0000, `opt` 4.2× the cost. `OCCC(=O)CO`: `opt` 5980 s → **23** basins, error **0.4023 kcal/mol**; `sp` 777 s → **37** basins, error 0.0025; union 45. `opt` loses 14 of 37 basins, costs 7.7×, and errs by 0.40 kcal/mol — above the ~0.1 target. Propanal has only 2 basins and could not show it. **Closed: `refine` changed to `"sp"` on 2026-09-04** by user ruling, after this measurement showed the premise of the 2026-09-03 ruling did not hold. The mechanism is in checkpoint 9 and `scripts/calibration/s0_A_refine_mechanism.py`; one component of it (structural merging vs the energy window) is still unseparated. |
| 2 | `shake_fallback` never fired | **closed.** All three strained rings run. `000607` 9 aborts → fallback → 0, 3 basins, 9/9. `003163` 15 → fallback → 0, 2 basins, 9/9. `002334` **0 aborts, no fallback needed**, 6 basins, 9/9. Against `D0-P1-34`'s 31/24/4 the counts did not reproduce, and on `002334` the aborts did not happen at all — so **zero-vs-nonzero is not a stable criterion either** at low counts. That is what makes the pointwise, reactive design right: it responds to the abort in the run at hand instead of assuming a prior list still holds. |
| 3 | dedup plateau on two molecules | **closed 2026-09-04 by ruling, not by measurement.** All three deduplication conditions are CREST's own published CREGEN values — `RTHR` 0.125 Å, `ETHR` 0.05 kcal/mol, `BTHR` 1%. What this repository owes them is fidelity to the source; a plateau study is the evidence you need to pick a number *of your own*, and we are not picking one. The sweep that retired the unsourced 0.30 Å stays in §3.4 as the reason the old value went. |
| 4 | **The energy and rotational conditions have never blocked a merge** — both counters still zero. |
| 5 | **Nothing has been submitted to Tianhe.** Three scheduler command names unverified. |
| 6 | **Per-molecule cost under contention unmeasured** — only the uncontended 285 s exists. |
| 7 | **`f7_mode` implemented and curatedQM9 in place** (`openqha/conformer_search/filters.py`, `openqha/curated_qm9.py`). Default is still `drop_all`; switching the default to `curated` is a separate ruling and has not been made. `curated` refuses to run when the archive is absent rather than falling back silently. |

---

## 8b. Is branch A ready to publish?

**It runs, and it is not ready.** Both halves of that matter.

### What is done

| | evidence |
|---|---|
| the chain runs end to end | 11 products on disk, every one 9/9 on the acceptance criteria |
| the fallback works and is marked | 3 strained rings; 2 fell back, both recovered, `used_shake_fallback` in the record |
| deduplication is CREST's own published criterion | RTHR 0.125 Å + ETHR 0.05 kcal/mol + BTHR 1%, all three |
| σ comes from geometry, per basin | 13/13 on the validation set, including the four cases the first version got wrong |
| the environment builds and is measured | one CPU environment (3.2 GB, 3m20s), one CUDA environment for branch C |
| numbers are not lost when files move | `configs/baseline.yaml` + `t_translation_preserved_numbers`, 109 files, 0 lost (that gate was retired 2026-09-09 once the migration finished) |

### What blocks publication

1. ~~Nothing has been run under the shipped default.~~ **Closed 2026-09-04.**
   `OCCC(=O)CO` through the full chain with `refine="sp"`: **62 conformers → 32 basins,
   0 `terminated EARLY`, 9/9 acceptance criteria, wall 2526 s** (CREST 885 s), written to
   `analysis/branchA/sp_default/`. The shipped configuration now has a product.
2. **The tutorial is being re-executed against that product.** `docs/tutorials/T01` shows
   a run made with `refine="opt"`; its loading cell now prefers the shipped-default
   product and prints which one it used, so the state is visible either way. Its outputs
   are measurements and are never hand-edited — they change by re-running the notebook.
3. ~~**`t_code_is_english` fails**: 25 files still hold non-English text.~~
   **Closed 2026-09-09.** The live tree is English: the final run reported `ok -- 0 lines`
   after clearing 26 files / 1278 lines, with `t_translation_preserved_numbers` confirming
   `no number lost`. The 28 files under `_superseded/` were counted but never failed and
   remain an open ruling (`S0-D-2`) -- they are marked ready to delete, not to translate.
   Both gates have been retired now that the migration they policed is complete.
4. **`t_filters_f7` fails on one case** — `dsgdb9nsd_000080` passes F7 under
   `f7_scope="identity_from_geometry_only"` because its index uses the relaxed SMILES,
   which its geometry matches. Pre-existing and not yet ruled on.
5. **Nothing has been submitted to Tianhe**, and three of five scheduler command names
   are still unverified. A wrong SUBMIT fails loudly; a wrong STATUS does not.
6. ~~The basin count is not reproducible and the consequence is unmeasured.~~
   **Closed 2026-09-05, five runs.** The count varies by 14%; the correction varies by
   0.0034 kcal/mol, 3.4% of the target. Report the correction; report a basin count as
   one draw, never as the answer. Untested on a second molecule.
7. **`f7_mode` default is still `drop_all`** — conservative and lossy (drops ~2.2% of QM9
   to avoid 0.05%). Switching it to `curated` is a separate ruling that has not been made.

### How repeatable is the answer itself?

Worth knowing before comparing your run to ours. Two runs of the **identical** protocol on
`OCCC(=O)CO`, differing only in the random seed of the metadynamics:

| run | CREST conformers | basins | CREST wall |
|---|---:|---:|---:|
| calibration (`s0_A_refine_compare`) | 82 | **37** | 777 s |
| production (`s0_A_pipeline --tag sp_default`) | 62 | **32** | 885 s |

**A spread of 5 basins in 37, about 14%.** iMTD-GC is a stochastic search and two runs do
not find the same set; this is the same lesson the `terminated EARLY` counts taught
(31/24/4 in one campaign, 9/15/0 in another). All runs are far above `refine="opt"`'s 23,
so the ruling is unaffected — if anything they confirm it independently.

### And what that costs the answer — measured 2026-09-05, five runs

| run | conformers | basins | correction / kcal·mol⁻¹ | w(lowest) | n < 5 kT |
|---|---:|---:|---:|---:|---:|
| calibration (`sp` arm) | 82 | 37 | −0.4486 | — | — |
| `sp_default` | 62 | 32 | −0.4467 | 0.4705 | 7 |
| `repro_2` | 63 | 35 | −0.4458 | 0.4712 | 7 |
| `repro_3` | 80 | 37 | −0.4481 | 0.4694 | 7 |
| `repro_4` | 73 | 35 | −0.4492 | 0.4685 | 7 |

|  | min | max | spread |
|---|---:|---:|---:|
| basins | 32 | 37 | **5 (14% of the mean)** |
| correction | −0.4492 | −0.4458 | **0.0034 kcal/mol** (sd 0.0014) |

**The count is not reproducible; the deliverable is.** 0.0034 kcal/mol is 3.4% of this
branch's ~0.1 target and 15× below the CREGEN energy condition (0.05). The weight of the
lowest basin lands in 0.4685–0.4712 every time, and every run puts 7 basins within 5 kT.

That was the argument when the spread first appeared — the count varies in the sparse
tail while the correction is carried by low-lying basins every run finds — and it is now
a measurement instead of an argument. It cuts both ways: **reporting a basin count as a
result is reporting noise, and concluding from the varying count that the method is
unreliable would be equally wrong.**

One molecule, five runs. Whether the ratio holds for a molecule with a flatter landscape
is untested.

None of these is a defect in the science; they are the difference between "the author can
run it" and "a stranger can run it and get what the documentation promises".

---

## 9. Where the results are, and how to find one again

> **Since 2026-09-14 this is history.** Results live in one molecule directory per (tag, molecule) with one folder per engine -- `docs/output_inventory.md` section 6 is the description; `docs/adr/0001` and `0002` the decisions. The block below describes the layout before that date and is kept as its record.


A full QM9 campaign is 133 885 molecules. **One directory per molecule is not an option** —
`ls` becomes unusable, ext4 without `dir_index` degrades past ~10k entries, a Lustre MDT
(Tianhe) serialises metadata on the parent directory's stripe so 16 workers creating
siblings all contend on one server, and `rsync`/`tar` spend their time in `readdir`.

So results are **sharded by index**, with files rather than directories at the leaf:

```
data/basins/<tag>/<range>/<chunk>/<qm9_index>.basins.json
data/basins/<tag>/<range>/<chunk>/<qm9_index>.basins.xyz

data/basins/prod/1_16000/1_4000/dsgdb9nsd_000018.basins.json
data/basins/prod/1_16000/4001_8000/dsgdb9nsd_005271.basins.json
data/basins/prod/128001_144000/132001_136000/dsgdb9nsd_133885.basins.json
```

Two files per molecule and at most 4000 molecules per leaf gives at most 8000 entries
per directory — comfortable everywhere, and still readable by a person. The chunk is
**arithmetic on the index** (`package1.chunk_size` = 4000), so nothing has to search and
no index file is needed to locate a result.

### Locally

```python
from openqha import basin_store

basin_store.paths_for("dsgdb9nsd_000018", tag="prod")   # where it would be
basin_store.read("dsgdb9nsd_000018", tag="prod")        # the record, or None
basin_store.exists("dsgdb9nsd_000018", tag="prod")
basin_store.census(tag="prod")                          # how many per chunk
basin_store.completed(tag="prod", chunk_dir="1_4000")   # for a resume
```

From the shell:

```bash
ls data/basins/prod/1_16000/1_4000/ | head
python -c "from openqha import basin_store as b; print(b.census(tag='prod'))"
```

### On Tianhe

The store lives under `$S0_RUNS_ROOT` while a job runs and is copied back afterwards.
**Copy the results, not the process**: CREST's scratch is many small files in parallel
`_N` subdirectories and has no value once the ensemble exists.

```bash
# in the job script, after the batch
rsync -a "$S0_RUNS_ROOT/basins/" "$HOME/openQHA/data/basins/"

# on the login node — how far did it get?
python -c "
from openqha import basin_store as b
c = b.census(tag='tianhe_prod')
print(c['total'], 'molecules')
for k, v in sorted(c['chunks'].items()): print('%-24s %d' % (k, v))"
```

**Resuming a chunk** is a set difference, which is why the layout is per-molecule files:

```bash
python -c "
from openqha import basin_store as b
done = b.completed(tag='tianhe_prod', chunk_dir='1_4000')
print(len(done), 'already computed in 1_4000')"
```

`completed(chunk_dir=...)` scans **one leaf**, not the whole tree — scanning 133k files
to resume 4000 of them is the kind of thing that makes a big campaign slow at the wrong
end.

### Why not one file, or a database

Because runs are interrupted. Per-molecule files mean an interrupted campaign holds
exactly the molecules it finished, each complete, and resuming is a set difference. An
append-only file would need locking across a node's 16 workers, and a partial write at a
walltime kill would corrupt its tail — which is the case that actually happens
(`D0-C-25`). Writes go to `.part` and are renamed, so **a missing result is resumable
and a corrupt one would have to be found first**.

The parquet summaries under `analysis/` are built *from* these files, not instead of them.
