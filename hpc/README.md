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
├── providers.py                 schedulers Parsl does not already know
├── resource_configs/
│   ├── local.py                 this machine — branch E step 0
│   └── deimos.py                64-core CPU nodes — branches A and B production
├── env/
│   ├── common.sh                three measured settings, every machine
│   └── deimos.sh                conda root, CREST path, scratch
└── configs/
    ├── master.json              ties the stages together
    ├── crest.json               branch A placement (not its science)
    └── qha_md.json              branch B placement (not its science)
```

**Branches A and B are live; C is not.** `tianhe_h100x.py`, `qm_label.json` and
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
to a cluster changes one argument: `--resource deimos`.

## The four workloads have four different resource profiles

This is the table the directory exists for. Putting these into one job
specification guarantees that one of them is wasted.

| workload | branch | parallel over | device | per task | source |
|---|---|---|---|---|---|
| conformer search — CREST(GFN2-xTB) + MACE `refine=opt` | A | molecules | **CPU** | 4 threads | 285 s/species measured, `S0-A-8` |
| quasi-harmonic trajectories — unbiased MACE MD | B | basin × seed | **CPU** | **1 thread** | 96.1 s/ps measured (10 atoms, 1 thread, uncontended); GPU rejected on `D0-48`, `D0-C-5` |
| QM labels — `xtb --hess`, ORCA RI-MP2 | C | structures | CPU | xtb 1 core; ORCA 4 processes | `NumFreq` parallelises over displacements; ORCA 1.85 GB/process measured |
| training — MACE + PHL loss | C | data-parallel | **GPU** | 1 card | the first workload here that batches naturally |

Three things follow directly, and only one of them is a preference:

1. **Conformer search must not go on a GPU partition.** Tianhe's `h100x` bills by
   the whole card and gives 14 CPUs with it; GFN2-xTB never touches a GPU. That is
   arithmetic, not taste (`D0-56`).
2. **Training is the only workload that certainly wants a GPU.** The other three
   need a benchmark before anyone decides.
3. **The quasi-harmonic device is settled, and its condition is written down.** A
   10-atom structure fed one at a time fills no card (`D0-56`), and on this repo's
   T400 the same trajectory ran 3.5× *slower* than on the CPU (`D0-C-5`). Branch B
   is CPU until a batched force interface exists — `D0-54` criterion (ii) — and the
   question reopens then, not before.

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
