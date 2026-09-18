# `hpc/` — where openQHA runs, and why

Branch E. **This directory produces no scientific number.** It decides which
machine each workload runs on and how many copies run at once. Every number comes
from `configs/openqha.yaml` and the branch drivers under `scripts/`.

That separation is the point, and it is checkable: branch E acceptance criterion 1
is *the same input gives the same result on this laptop and on a cluster*. If
changing a file in `hpc/` can change a result, this directory is wrong.

Shape adapted from [ALF](https://github.com/lanl/ALF) (LANL, BSD-3-Clause) —
`alframework/parsl_resource_configs`. openQHA itself is CC BY-NC 4.0; the ALF
copyright and disclaimer travel with the code adapted from it.

## What is here

```
hpc/
├── labels.py                    executor labels — defined ONCE, checked before submit
├── providers.py                 schedulers Parsl does not already know (yhbatch/yhrun)
├── resource_configs/
│   ├── local.py                 this machine — branch E step 0
│   ├── deimos.py                the group's own 64-core CPU nodes
│   ├── tianhe_cpu.py            TianheXY-C   debug + deimos   branch A, collection
│   ├── tianhe_a.py              TianheXY-A   temp + ai        branch B traj, branch C
│   └── tianhe_ai.py             TianheXY-AI  per GPU card     branch C
├── env/
│   ├── common.sh                three measured settings, every machine
│   ├── deimos.sh                conda root, CREST path, scratch
│   └── tianhe.sh                the same for Tianhe, plus the proxy
├── slurm/                       the plain yhbatch route — same work, no Parsl
└── configs/
    ├── master.json              ties the stages together
    ├── crest.json               branch A placement (not its science)
    └── qha_md.json              branch B placement (not its science)
```

**Two routes, one set of numbers.** `resource_configs/` + the `s0_E_*_parsl.py` drivers
are the Parsl route; `slurm/` is the same work through `yhbatch` and `xargs`. Both call
the same production drivers with the same arguments, so a product must not differ between
them — comparing the two on one molecule is the cheapest test of criterion 1 there is.

**Branches A and B are live; C is not.** `qm_label.json` and
`train.json` are named in `master.json` as disabled, because the workloads they would
describe have no measured unit cost yet — and writing a resource description for a
cost you have not measured is how you get a plan that reads well and does not run.
`qha_md.json` was in that list until 2026-09-03 and left it the moment branch B had
a measured cost, which is the only thing that was ever holding it there.

## Start here

```bash
source hpc/env/common.sh                     # do NOT set -u first; see the file
python scripts/production/s0_E_branchA_parsl.py --species dsgdb9nsd_000018 --dry-run
python scripts/production/s0_E_branchA_parsl.py --species dsgdb9nsd_000018

# branch B: trajectories, then the analysis that tests every plan_B criterion
python scripts/production/s0_E_branchB_parsl.py --species dsgdb9nsd_000018 --dry-run
python scripts/production/s0_E_branchB_parsl.py --species dsgdb9nsd_000018
python scripts/production/s0_B_qha_analyse.py --species dsgdb9nsd_000018 --tag prod
```

Local first is not a formality. `D0-C-20` rules that the first thing done on a
cluster is a benchmark, not a production run, and debugging a Parsl workflow on a
cluster costs far more than debugging it here. Once the chain runs locally, going
to a cluster changes one argument: `--resource tianhe_cpu`.

**On a cluster the gate is `--debug`, not `--dry-run`** (user ruling 2026-09-07): the same
command, on the site's short partition, at a 30-minute walltime, capped at one
allocation. A real short job tests what a rendered plan cannot — that the module loads,
that conda activates on a *compute* node, that the weights hash correctly there, and that
Parsl can read its own status query. `--debug` is **refused** on a resource config that
has no short partition, rather than quietly running production settings.

See [`../docs/branchA_production.md`](../docs/branchA_production.md) for the whole
production shape and [`slurm/README.md`](slurm/README.md) for the non-Parsl route.

## The four workloads have four different resource profiles

This is the table the directory exists for. Putting these into one job
specification guarantees that one of them is wasted.

| workload | branch | parallel over | device | per task | source |
|---|---|---|---|---|---|
| conformer search — CREST(GFN2-xTB) + MACE `refine=sp` | A | molecules | **CPU** (TianheXY-C) | 4 threads, 16/node | 285 s/species measured, `S0-A-8` |
| quasi-harmonic trajectories — unbiased MACE MD | B | basin × seed | **GPU** (TianheXY-A) | **1 card**, 8/node | user ruling 2026-09-07. **Cost UNMEASURED there** — see below |
| collection — quasi-harmonic analysis of finished trajectories | B | molecules | **CPU** (TianheXY-C) | **1 core**, 64 on **one** node | small, serial, float64; its real cost is Lustre metadata |
| QM labels — `xtb --hess`, ORCA RI-MP2 | C | structures | CPU | xtb 1 core; ORCA 4 processes | `NumFreq` parallelises over displacements; ORCA 1.85 GB/process measured |
| reference E-F-H labels per frame — ORCA wB97M-D3BJ single point + EnGrad + analytic Hessian (`workflows/hessian_learning/03_labels.py`, role `labels`) | Hessian learning | frames | **CPU** (TianheXY-C, `sbatch`) | **4 ranks**, 16/node, `%maxcore 6000` | user ruling 2026-09-18; 1 node smoke / 12 nodes draw; ORCA 6.1.1 from conda env `orca611` via `hpc/env/orca.sh` |
| training — MACE + PHL loss | C | data-parallel | **GPU** | 1 card | the first workload here that batches naturally |

Three things follow directly, and only one of them is a preference:

1. **Conformer search must not go on a GPU partition.** Tianhe's `h100x` bills by
   the whole card and gives 14 CPUs with it; GFN2-xTB never touches a GPU. That is
   arithmetic, not taste (`D0-56`).
2. **Training is the only workload that certainly wants a GPU.** The other three
   need a benchmark before anyone decides.
3. **The quasi-harmonic device was reopened, by ruling rather than by measurement.**
   Until 2026-09-07 branch B was CPU, on two measurements: a 10-atom structure fed one
   at a time fills no card (`D0-56`), and on this repo's T400 the same trajectory ran
   3.5× *slower* than on the CPU (`D0-C-5`). The user has ruled that the trajectories
   run on TianheXY-A, one per card, through OpenMM.

   That ruling is followed and the state of the evidence is stated rather than
   dressed up: **a T400 is a 2 GB entry-level card and 80 GB HBM2e is not, so D0-C-5
   does not transfer — but nothing has replaced it either, and no batched force
   interface exists yet (`D0-54` criterion (ii)).** Run the 30-minute `temp` smoke test
   and read `SECONDS_PER_PS` out of `md.toml` before sizing a campaign. The
   CPU route is kept as `--route ase`, which is also the independent implementation pair
   that makes the OpenMM numbers checkable.

   Branch B also takes **one** thread per task where branch A takes four, from the
   same thread-scaling measurement in the next section: four threads buy 1.54×, four
   trajectories buy 4×, and branch B always has (basins × 3 seeds) of independent
   work because those three seeds *are* its blank control. The two executors are
   therefore not interchangeable, and `s0_E_branchB_parsl.py` submits only to
   `openqha_qha`.

## Parallelism lives between processes, never inside threads

MACE thread scaling on a 10-atom molecule, measured: **111 / 90 / 72 / 101 ms** at
1 / 2 / 4 / 8 threads. Eight threads is *slower* than four, and 1 → 8 buys 1.1×.
So `hpc/env/common.sh` pins `OMP`, `MKL`, `NUMEXPR` and `OPENBLAS` to one thread and
the executors run many single-threaded workers.

CREST's own `threads` is a different axis — it parallelises independent
metadynamics runs — and is set per task in `configs/crest.json`. The two must not
be confused: `workers × threads` has to stay under the physical core count, or the
measured cost of a molecule stops meaning anything.

`OPENBLAS_NUM_THREADS=1` is in that file for a third reason again: it **fixes a
defect**, not a slowdown. conda-forge's CREST links the pthreads OpenBLAS while
CREST is OpenMP-parallel, and the mismatch prints *"Detect OpenMP Loop and this
application may hang"* on every step. Measured on acetone: 4.4 s → 2.7 s, 3626
warning lines → 0, `crest.out` 476 KB → 34 KB.

## One MACE server per worker

CREST's quality layer reaches MACE over a Unix socket held by a resident process.
That process holds one model behind one lock. **N workers sharing one socket are
not parallel — they queue**, and nothing in the logs says so. Each worker starts
its own server on its own socket under `$S0_RUNS_ROOT/sockets/`.

## Tianhe

Confirmed on site (`D0-C-24`, `D0-C-25`, `D0-56`):

- directives are `#SBATCH` — it is a Slurm derivative;
- submission is **`yhbatch`**, launcher **`yhrun`**;
- `h100x` = 1 GPU / 14 CPU / 240 GB, H100 80 GB, **billed per card**;
- a submission **without an explicit `-G` fails**;
- `--array` is available (user, 2026-09-03);
- outbound network through a proxy, no container needed, CREST from conda-forge.

Parsl hard-codes **four** command names, verified with `inspect.getsource` on the
installed 2026.08.10 rather than from documentation: `sbatch`, `sacct`, `squeue`,
`scancel`. `providers.py` overrides them in a named subclass.

**Two cautions that are load-bearing:**

- Only `yhbatch` and `yhrun` are actually confirmed. The status and cancel commands
  have never been seen on that machine, and this repo does not invent them: a wrong
  *status* command does not fail loudly — Parsl believes every job is still pending
  and the queue quietly stops. Run `providers.preflight()` on the login node and
  fill in what it finds.
- Pass the GPU count as `gpus_per_node`, **not** as a `scheduler_options` string.
  `SlurmProvider.__init__` accepts it, Parsl renders it into the template, and a
  string is never checked. Branch E acceptance criterion 3 is written against the
  **rendered script** for exactly this reason — a config that reads correctly and
  renders wrongly is the failure the criterion is for.

`--array` being available does not change the design: Parsl queues through
`max_blocks`. It is recorded because it keeps
`_superseded/cluster_2/s0_submit_array.slurm` open as a fallback shape if Parsl
ever proves unworkable there.

**No alias, and no shim named `sbatch` on `PATH`.** Both would work, and both would
make *how was this job submitted* something you cannot read off the record. This
repo has already lost a dataset to a condition invisible in its own products
(defect 57).

## Adding a machine

Copy `local.py`, change the provider, and add an `env/<machine>.sh` with only what
is true of that machine. Nothing chemical belongs in either file. If a value would
have to appear in both `hpc/` and `configs/openqha.yaml`, the YAML wins
and the `hpc/` copy is a bug.
