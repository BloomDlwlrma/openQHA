# Branch A in production, and branch E under it

What actually runs, where each setting lives, how to submit it both ways, and where the
answer lands. Written 2026-09-07, after branch B's trajectories moved to
the GPU and the CPU cluster was cut down to two partitions.

**Nothing in this repository has ever been submitted to Tianhe.** Everything below is
either verified locally (the configs build, the job scripts render and parse, the
environment solves) or marked as unverified. Read section 6 before spending an allocation.

---

## 1. The two halves, and why they are two

Branch A **searches** for conformers. Branch E **places** that search on a machine.
The dividing line is a rule, not a style:

> Replacing the execution layer with a `for` loop must not change a single number.

That is branch E acceptance criterion 1. It is what lets the same code run on a laptop, on
`deimos` through `yhbatch`, and on `deimos` through Parsl, and be compared afterwards. Every
file below is on one side of that line.

```
BRANCH A -- the science. Changing anything here changes the answer.
  configs/openqha.yaml          global settings + the include index
  configs/conformers.yaml       crest:, package1:      <- the CREST protocol
  configs/filters.yaml          species_filter         <- the F0-F7 gates
  configs/edges_testset.yaml    edges:, species:       <- which molecules
  configs/hessian.yaml          package2               <- frequencies
  openqha/conformer_search/crest.py         writes the CREST input, parses its output
  openqha/conformer_search/conformers.py    dedup, ordering, basin construction
  openqha/conformer_search/crest_census.py  the census a product is built from
  openqha/conformer_search/filters.py       F0-F7
  openqha/conformer_search/symmetry.py      sigma per basin
  openqha/potentials/engine.py              THE POTENTIAL: registry, path, provenance
  openqha/potentials/mace_server.py         the resident MACE server CREST talks to
  openqha/potentials/mace_patch.py          translation-invariant neighbour list
  openqha/thermochem/hessian.py             analytic / finite-difference Hessians
  openqha/thermochem/thermo.py              partition functions
  openqha/store/basin_store.py              WHERE results go (sharded)
  openqha/store/record.py                   the product record
  scripts/production/s0_A_pipeline.py    the driver: ONE molecule, start to finish

BRANCH E -- the execution layer. Changing anything here must change NO number.
  hpc/labels.py                 executor labels, defined once
  hpc/providers.py              TianheSlurmProvider (yhbatch/yhrun)
  hpc/resource_configs/         one module per machine
      local.py                  this workstation -- step 0
      tianhe_cpu.py             TianheXY-C: debug + deimos
      tianhe_a.py               TianheXY-A: temp + ai (8 cards/node)
      tianhe_ai.py              TianheXY-AI: one card per allocation
  hpc/env/common.sh             what every job sources, on every machine
  hpc/env/tianhe.sh             what only Tianhe needs
  hpc/slurm/                    the plain yhbatch route (no Parsl)
  openqha/store/worklist.py     what is left to compute, and how it decided
  scripts/production/s0_E_worklist.py       that, as a command
  scripts/production/s0_E_branchA_parsl.py  branch A over Parsl
```

If you are ever unsure which side a change belongs on, ask whether a person reproducing
the result needs to know about it. The queue name: no. The dedup threshold: yes.

### The package is grouped by role (changed 2026-09-07)

`openqha/` was 29 modules in one flat directory. It is now grouped the way ALF
(`alframework/{builders,samplers,qm_interfaces,ml_interfaces,tools}`) and MACE
(`mace/{calculators,cli,data,modules,tools}`) are grouped:

```
openqha/
    config.py           every module reads it, so it stays at the top
    capabilities.py     introspects the WHOLE package, so it cannot live inside one part
    conformer_search/   crest conformers crest_census refine_analysis filters symmetry
    quasi_harmonic/     qha openmm_mace vdos torsion_cv perturb mdtraj_io gmx_io
    potentials/         engine mace_patch mace_server          (ALF: ml_interfaces)
    thermochem/         hessian thermo critical
    store/              basin_store artifacts record report worklist
    data/               curated_qm9 qm9_uncharacterized
    qm_interfaces/      orca
    extensions/         optional cross-checks; no production number depends on one
```

Named for what they **do**, not for which branch of this project's plan they serve — a
reader of the published package should not have to learn the branch letters to find the
conformer search.

**Every existing import still works.** `from openqha import qha`, `import openqha.qha`
and `from openqha.thermo import KB_KCAL` all resolve, through a lazy PEP 562
`__getattr__` in `openqha/__init__.py`. Lazy is the load-bearing half: `import openqha`
still pulls in no torch, ase, rdkit or openmm, which is what lets
`openqha.capabilities` report on packages that are not installed. New code should prefer
the explicit path — `from openqha.quasi_harmonic import qha` — because it says which part
of the pipeline is being reached into.

---

## 2. One molecule, end to end

This is what `s0_A_pipeline.py --species dsgdb9nsd_000018` does. Every arrow is a function
you can call yourself.

```
QM9 geometry                config.qm9_xyz()          data/qm9/xyz_files/ or the
                                                      7 vendored ones or curatedQM9
      |
      v  filters.screen()   F0-F7. A molecule that fails is dropped HERE, with the
      |                     gate that dropped it recorded -- never silently.
      v
CREST iMTD-GC               crest.py builds the input
   workhorse GFN2-xTB       configs/conformers.yaml: crest.workhorse
   SHAKE all bonds,         crest.shake = 2, crest.tstep_fs = 5.0
   5 fs, H mass 2 amu       ONE PACKAGE from Grimme JCTC 2019, 15, 2847 --
                            they were measured together and are not separable
   refine = "sp"            single points on MACE, NOT optimisation.
      |                     `opt` costs 14 of 37 basins and 0.4023 kcal/mol.
      v
MACE-OFF23_medium           mace_server.py holds ONE model behind ONE lock and
   over a unix socket       serves it over $S0_MACE_SOCKET. One server per
                            molecule, never one per machine -- see section 5.
      |
      v  CREGEN dedup       RTHR 0.125 A AND ETHR 0.05 kcal/mol AND BTHR 1%
      |                     (CREST's own published defaults; all three must hold)
      v
basins                      conformers.py
      |
      +--> symmetry.py      sigma per basin
      +--> hessian.py       frequencies (--hessian-mode analytic)
      +--> thermo.py        q_rot, q_vib, and the conformational correction
      |
      v
PRODUCT  (since 2026-09-14; docs/output_inventory.md sections 6 and 8)
   <molecule>/crest/                          CREST's working directory, verbatim
   <molecule>/mace/confNN/                    opt.traj opt.log conf.extxyz  every relaxation
   <molecule>/mace/basinNN/                   basin.extxyz hessian.npy      every basin
   <molecule>/_records/branchA.out, branchA.toml   the Record (driver.log beside it via parsl)
   <molecule> = <root>/<tag>/<range>/<chunk>/<qid>
```

**Report the correction, not the basin count.** Five repeats on one molecule: the basin
count varies by 14% run to run (iMTD-GC is a stochastic search), while the conformational
correction varies by 0.0034 kcal/mol -- 3.4% of the 1 kcal/mol target. The deliverable is
reproducible; the count is one draw.

---

## 3. Environments: which file, which machine

| file | env name | machine | BLAS | what it is for |
|---|---|---|---|---|
| `environment.yml` | `openqha` | workstation | OpenBLAS/OpenMP | everything, including branch B's OpenMM |
| `environment-tianhe.yml` | `openqha` | **TianheXY-C** | OpenBLAS/OpenMP | branch A + collection. CPU torch, CREST, xtb |
| `environment-tianhe-gpu.yml` | `openqha-gpu` | **TianheXY-A and TianheXY-AI** | MKL | branch B trajectories + branch C training. CUDA 12.3 |
| `environment-cuda.yml` | `openqha-cuda` | any GPU box | OpenBLAS/OpenMP | CUDA without the Tianhe specifics |
The BLAS column is there because it is the axis the two Tianhe environments are split
along and the one you cannot read off the name. (The branch-B-only lean variant,
`environment-openmm.yml` / `openqha-openmm`, retired 2026-09-29 -- a strict subset of
`environment.yml`; the file is in `_superseded/`.) `openqha-gpu` takes MKL because a CUDA
build of pytorch depends on it, which makes `nomkl` unsatisfiable there; everything that
could be *moved* by that choice — the quasi-harmonic diagonalisation above all — runs in
the collection pass, in `openqha`, against OpenBLAS. [`tianhe_install.md`](tianhe_install.md) §1.5 has the
three measurements behind the split.

```bash
bash install_dependency.sh --tianhe        # TianheXY-C  -> openqha
bash install_dependency.sh --tianhe-a      # TianheXY-A  -> openqha-gpu, CUDA/12.3
bash install_dependency.sh --tianhe-cuda   # TianheXY-AI -> openqha-gpu, CUDA/12.3

# Only one cluster on your account? Build both on it -- two solves, not one big env:
bash install_dependency.sh --tianhe-cuda --both
```

Then pick one per job rather than activating both — conda *stacks* activations, and a
stacked shell has an OpenBLAS `lib/` and an MKL `lib/` on one loader path:

```bash
OPENQHA_ROLE=cpu source hpc/env/tianhe.sh    # branch A, QHA collection
OPENQHA_ROLE=gpu source hpc/env/tianhe.sh    # branch B, branch C
```

### One GPU environment, not two (changed 2026-09-07)

`environment-tianhe-cuda.yml` (12.4) and `environment-tianhe-a-cuda.yml` (12.3) are now in
`_superseded/`. They differed in two things and both went away:

* **CUDA version.** TianheXY-A's module tree tops out at 12.3 and TianheXY-AI has 12.3 too,
  so 12.3 fits both. Going the other way does not: `module load CUDA/12.4` fails on A, and
  a conda CUDA pinned above the runtime imports cleanly and dies at the first kernel launch.
* **The openmpi built against that CUDA.** Removed entirely — **nothing openQHA runs on a
  card needs collectives.** Training is one model per allocation; branch B is one trajectory
  per card. Eight independent single-card workers talk through the filesystem, not MPI. Not
  needing MPI is what made one file possible.

**One thing to confirm before the first GPU job on TianheXY-AI:** that CUDA/12.3 is really
there. The recorded listing was elided (`CUDA/11.8 ... CUDA/13.2`), so 12.3 is inside the
recorded range but was never read off the screen.

```bash
module avail CUDA 2>&1 | grep -o "CUDA/12[.][0-9]*"
```

If it is absent, change `CUDA_VERSION` in `hpc/resource_configs/tianhe_ai.py` and the pin in
the yml. Not a second file.

### The BLAS pins are absent from the GPU file on purpose

Every other environment file pins `libopenblas=*=openmp*`, because conda-forge's `crest`
links the pthreads build while CREST is OpenMP-parallel, and the mismatch cost 4164 warning
lines and 20 s per molecule. **There is no crest in the GPU environment**, so the defect
cannot occur — and the pins were not free. Measured 2026-09-07, one variable at a time:

```
openmmtools + BLAS pins, no CUDA                       SOLVES
openmmtools + CUDA torch, no BLAS pins                 SOLVES
openmmtools + CUDA torch + cuda-version=12.3 + pins    FAILS
the file as it now stands                              SOLVES
```

Dropping `openmmtools` instead was not an option: its `NoseHooverChainVelocityVerlet`
defaults **are** branch B's thermostat (50/ps, chain 5, MTS 5, YS 5), adopted rather than
chosen. Plain OpenMM's integrator defaults to a 7-term Yoshida–Suzuki decomposition, so
substituting it would change the thermostat silently.

---

## 4. MACE-OFF weights — what that warning actually asks you to do

The line that is confusing:

> **MACE-OFF weights** — NOT downloaded on a login node — outbound traffic goes through a
> proxy and a 100 MB pull from a login node is antisocial. Fetch them where you have
> bandwidth and rsync them in.

Unpacked, it is three separate facts:

**1. The weights are not in this repository.** They are large binaries that belong beside
the run rather than in version control. The registry in `openqha/potentials/engine.py`
holds their *filenames*, and nothing else about them.

**2. `install_dependency.sh` downloads them for you — except in `--tianhe` mode.**
On a workstation it fetches `MACE-OFF23_medium.model` (~100 MB) from
`github.com/ACEsuit/mace-off` and puts it in `data/potentials/`.
Under `--tianhe` it deliberately does not, and prints the rsync line instead.

**3. Why not on a login node.** The login node reaches the internet only through the site
proxy (`source /APP/u22/ai_x86/toolshs/setproxy.sh 172.16.31.200 3138`). Three reasons,
in increasing order of how much they will cost you:
   - it is slow and shared — a login node is for everyone;
   - `conda` and `pip` *hang* rather than fail without the proxy set, and a hung job is
     charged for its whole walltime;
   - **a proxy can return an HTML error page under a 200 status.** That is a perfectly
     valid file and a completely wrong potential, and it looks like a successful download.
     Nothing downstream inspects the file, so check the size where you copy it from —
     `ls -l data/potentials/` — before you rsync.

### The actual procedure

```bash
# --- on your workstation, where you have bandwidth -----------------------------------
cd openQHA
bash install_dependency.sh                    # downloads the default
# or:  bash install_dependency.sh --all-weights   # the whole committee, for branch C

ls -l data/potentials/
#   MACE-OFF23_medium.model

# --- copy it in ----------------------------------------------------------------------
rsync -a --progress data/potentials/ <you>@tianhe:~/openQHA/data/potentials/

# --- on Tianhe: prove it arrived intact ----------------------------------------------
cd ~/openQHA && source env_openqha.sh
python -c "
from openqha import engine, json
print(json.dumps(engine.provenance(), indent=1, default=str))"
```

Read back `engine` and `weights_path` — those two are the identity of the potential this
run will use, and they are what every product records.

### Where the model setting lives

| what | where | note |
|---|---|---|
| the registry | `openqha/potentials/engine.py` → `ENGINES` | name → **filename**, source, note — no paths, nothing verified |
| the directory | `openqha/potentials/engine.py` → `model_root()` | `<repo>/data/potentials`, or `S0_MACE_ROOT` |
| add a potential | an `ENGINES` entry: `filename=`, `source=`, `note=` | any MLIP. Drop the file in that directory, flat |
| production default | `openqha/potentials/engine.py` → `DEFAULT_ENGINE` | `MACE-OFF23_medium` |
| select another | `S0_ENGINE=MACE-OFF23_large` | must be a registered name; unknown names raise |
| move the directory | `S0_MACE_ROOT=/path/to/potentials` | the one flat directory of weights |
| one file | `S0_MACE_MODEL=/path/to/x.model` | overrides that engine only |
| what the run used | `configs/openqha.yaml` → `engine:` | records *which* weights, **not authoritative for the path** |

**ONE directory, FLAT**. `model_root()/<filename>` and nothing
else — no subdirectories, no search:

```
<repo>/data/potentials/                     <- model_root(); S0_MACE_ROOT moves it
    MACE-OFF23_medium.model                 <- the production default
    MACE-OFF23_small.model                  <- committee members, --all-weights
    MACE-OFF23_large.model
    MACE-OFF23b_medium.model
    MACE-OFF24_medium.model
    MACE-OFF23-SC_swa.model                 <- kept selectable; package 1 lives on it
```

This replaced a search over four candidate roots × seven relative layouts, with a
per-family `mace_off23/` subdirectory on top. That search was written because a literal
path had already broken twice here — and it failed worse, in a way worth remembering:

> A search with many candidates does not have one answer you can check. It has a list of
> places it *might* have looked, and when it fails nobody can tell whether the file is
> missing or merely somewhere the list does not cover.

Both of its failures were real and both were on Tianhe: weights copied flat into
`data/potentials/` — what everyone does when moving them to a cluster — were invisible
because a directory only counted if it held a literal `mace_off23` subdirectory; and
`install_dependency.sh` told people to use `data/potentials/`, which **was not in the
search list at all**. That appeared to work only because the installer also writes
`env_openqha.sh`, which exports `S0_MACE_ROOT` — and a cluster job never sources it. The
documented location failed on exactly the path the documentation was written for.

The original fragility is answered by `S0_ROOT` being *computed* rather than written
down, not by searching harder.

**Never falls back.** A missing or altered weight file raises. Silently swapping the
potential strips every downstream number of the level it claims to be at, and level
consistency is the only condition under which the composite decomposition holds.

---

## 5. One MACE server per molecule, never one per machine

CREST's quality layer reaches MACE over a unix socket held by a resident process. That
process holds **one model behind one lock**. N workers sharing one socket are therefore not
parallel — they queue on the lock, and the fan-out is imaginary while looking real in every
log.

Both routes handle this, differently, and neither needs configuring:

* **Parsl route** — `s0_E_branchA_parsl.py` starts a server per task, on a socket named
  after `(molecule, pid)` so two workers can never collide, and stops it when the molecule
  is done.
* **yhbatch route** — `s0_A_pipeline.py` starts its own when `S0_MACE_SOCKET` is not
  already exported. Under `xargs -P 16` that is 16 servers, one per molecule.

The cost is 16 copies of a ~100 MB model in RAM on a 512 GB node. That is the correct
trade.

---

## 6. Submitting: `debug` first, then `deimos`

**There is no `--dry-run` gate any more**. A 30-minute real job on
the short partition tests five things a rendered plan cannot: that the module loads, that
conda activates on a *compute* node, that the weights are there and hash correctly, that the
scheduler accepts the directives, and that Parsl can read its own status query.

`--dry-run` still exists and still prints the plan. It is for reading, not for gating.

| cluster | smoke | production |
|---|---|---|
| TianheXY-C | `debug`, **00:30:00**, 1 node | `deimos`, **3-00:00:00**, 12 nodes |
| TianheXY-A | `temp`, **00:30:00**, 1 node | `ai`, **24:00:00**, 8 nodes |

`xyfree`, `mars` and `e9` exist on the CPU cluster and are deliberately unused: one
production queue means one set of costs to compare against. They are recorded in
`tianhe_cpu.PARTITIONS_NOT_USED` so "why not mars" has an answer.

### 6.0 Before anything: first login

```bash
ssh <you>@tianhe
source /APP/u22/ai_x86/toolshs/setproxy.sh 172.16.31.200 3138

# Probe the mirror this account actually uses -- TUNA, via custom_channels in ~/.condarc.
curl -sI https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/noarch/repodata.json | head -1
```

Expect `HTTP/… 200`. If it hangs, stop — everything after it will hang too, and a hung job
is charged. A login node has **no direct DNS**, so before `setproxy.sh` nothing resolves
and conda reports that as an outage of whichever mirror it was reaching for; that is
exactly the 2026-09-08 failure, and [`tianhe_install.md`](tianhe_install.md) is the handbook and the post-mortem.

```bash
type -a yhbatch yhrun sacct squeue scancel        # all five exist (measured 2026-09-05)
sinfo -o "%P %c %m %D"                            # confirm `debug` and `deimos`
```

`sbatch` is **not** present; `yhbatch` is. `sacct`/`squeue`/`scancel` are used rather than
their `yh*` variants because Parsl *parses* their output and nobody has read `yhacct`'s
format.

### 6.1 Install

```bash
cd ~/openQHA
bash install_dependency.sh --tianhe        # builds `openqha` from environment-tianhe.yml
rsync -a data/potentials/ <tianhe>:~/openQHA/data/potentials/    # section 4
source env_openqha.sh
python check_dependency.py                 # ends: branch A READY / NOT READY
```

### 6.2 Smoke test — one edge, 30 minutes

**Parsl:**

```bash
mkdir -p $HOME/HDD_POOL/runs/openQHA/logs
python -u scripts/production/s0_E_branchA_parsl.py \
    --species dsgdb9nsd_000018 dsgdb9nsd_000035 \
    --resource tianhe_cpu --tag tianhe_debug --debug
```

`--debug` swaps in `debug` / `00:30:00` and caps the run at one allocation. It is **refused**
on a resource config that has no short partition, rather than quietly running production
settings under a flag that says debug.

**yhbatch:**

```bash
mkdir -p $HOME/HDD_POOL/runs/openQHA/logs
EDGE=C3H6O1N0_18_35 TAG=tianhe_debug yhbatch hpc/slurm/branchA_debug.slurm
```

Both should end with 9/9 acceptance criteria per molecule. **Then compare the two
products** — if a molecule run through Parsl differs from the same molecule run through
`xargs`, the execution layer is doing something it must not.

### 6.3 Production — 4 threads × 16 concurrent × 12 nodes

**192 molecules in flight.** The quota allows 32 nodes; 12 is what is configured, and
`MAX_BLOCKS` in `hpc/resource_configs/tianhe_cpu.py` is the one number to change.

**Parsl** — settings live in the config, not the command line:

```bash
python -u scripts/production/s0_E_branchA_parsl.py \
    --range 1 16000 --resource tianhe_cpu --tag prod \
    2>&1 | tee $HOME/HDD_POOL/runs/openQHA/logs/prod_$(date +%Y%m%d_%H%M).log
```

`--range` resumes: molecules already complete under that tag are subtracted, and molecules
with no geometry or that fail F0–F7 are dropped, each with a count and a reason in the plan
record. Re-running the same command does the remainder.

**yhbatch** — twelve shards, one per node:

```bash
bash hpc/slurm/submit_branchA_deimos.sh 1 16000 prod
```

The submitter makes the log directory first (Slurm opens `--output` *before* the job script
runs, so a directory the script creates is created too late), scans once and refuses to
submit twelve jobs for nothing, and records shards + tag + job ids next to the logs.

Each shard **re-scans when it starts**, not when it was submitted. A job that waited six
hours in the queue therefore does not redo what its neighbours finished. `--shard K/N` takes
every N-th entry of the *already filtered* list, not a slice of the index range: QM9 indices
are not contiguous and the gates drop molecules unevenly, so splitting the range gives one
node an hour of work and another three days.

### 6.4 Watching it

```bash
yhq -a                 # queue (works even with no allocation left)
yhi                    # node states: idle / mix / alloc
yhcancel <jobid>

python -c "
from openqha import basin_store
c = basin_store.census(tag='prod')
print(c['total'], 'molecules done across', len(c['chunks']), 'chunks')"

python scripts/production/s0_E_worklist.py --range 1 16000 --tag prod --explain \
    > /dev/null          # the decision record on stderr: done / no geometry / gated
```

Scale with `MAX_BLOCKS`. `init_blocks=0` and `min_blocks=0` mean nothing is requested until
there is work and an idle allocation is given back — **a held node is charged whether or not
it computes.**

### 6.5 What the two routes do differently — and what they must not

| | Parsl | yhbatch |
|---|---|---|
| unit submitted | a *block* (one node), grown on demand | one job per shard |
| worklist | `openqha/store/worklist.py`, in the driver | `s0_E_worklist.py`, in the job |
| resume | subtract completed under the tag | same code, same criterion |
| failure of one molecule | task recorded, batch continues | `FAILED <id>` appended, `xargs` continues |
| card/thread pinning | `cores_per_worker` | `xargs -P` |
| **any number in the product** | **identical** | **identical** |

The last row is the only one that matters. Both call `s0_A_pipeline.py` with the same
arguments; neither touches chemistry.

---

## 7. Where the answer lands, and how to read it

> **The layout changed on 2026-09-14.** Results live in one molecule directory per (tag, molecule) with one folder per engine -- `docs/output_inventory.md` section 6 is the description. The block below describes the layout before that date and is kept as its record.

### The basin store — sharded, because 133 885 molecules is not one directory (before 2026-09-14)

```
data/basins/<tag>/<range>/<chunk>/<qid>.basins.json
                                 /<qid>.basins.xyz

data/basins/prod/1_16000/1_4000/dsgdb9nsd_000018.basins.json
data/basins/prod/1_16000/1_4000/dsgdb9nsd_000018.basins.xyz
```

Two files per molecule, at most 4000 molecules per leaf, so at most 8000 entries per
directory — comfortable on ext4, on Lustre and for `ls`. The chunk that owns a molecule is
arithmetic on its index (`basin_store.shard()`), so nothing has to search and no index file
is needed to find one result.

Each file is written to `.part` and **renamed**, so a job killed at its walltime leaves a
missing result rather than a corrupt one. A missing result is resumable; a corrupt one has
to be found first. `worklist.COMPLETION` therefore requires **both** files: the JSON is the
record and the xyz is the deliverable, and a run killed between the two renames must not
count as done.

### The full record

```
<molecule>/_records/branchA.toml             the Property file: status, inputs, CREST run, census,
                                             one [[Basin]] block per basin, the criteria count
<molecule>/_records/branchA.out              the Report: provenance, settings, gate, CREST, census,
                                             per-basin symmetry and thermochemistry, criteria, the
                                             full record expanded; last line = terminal line
<molecule>/_records/driver.log               the driver's stdout, when a Batch ran it
```
(records redesign 2026-09-15, `docs/output_inventory.md` section 8; before 2026-09-14:
`analysis/branchA/<tag>/<qid>/basins.json`. A Batch leaves no record: the Slurm log is its
report, one aligned line per molecule with rc, STATUS and the path of `branchA.toml`.)

The Slurm log carries the plan the driver printed before the batch, including
`settings_source` (which value came from the command line, which from the resource
config, which from a built-in default) and `submission` (which partition and walltime
this run actually used). Without the latter a `--debug` run and a production run print
the same `resource` block and become indistinguishable afterwards.

### Reading it

```python
from openqha.store import basins, branch_a_property

# how far has the campaign got?
c = basins.census(tag="prod")
print(c["total"], "molecules;", len(c["chunks"]), "chunks")

# one molecule: branchA.toml as blocks
rec = basins.read_record("dsgdb9nsd_000018", tag="prod")
print(rec["Census"]["N_BASINS"], "basins")
print(rec["Criteria"]["ALL_PASSED"], basins.status("dsgdb9nsd_000018", tag="prod"))
for b in branch_a_property.basin_rows(rec):
    print(b["SIGMA"], b["ENERGY"], b["RELATIVE"], b["G_MINUS_EEL"])

# where is it on disk?
print(basins.molecule_for("dsgdb9nsd_000018", tag="prod"))
```

```bash
# the geometries, as an ordinary trajectory
python -c "
from ase.io import read
from openqha import basin_store
_, x = basin_store.paths_for('dsgdb9nsd_000018', tag='prod')[:2]
print(len(read(str(x), index=':')), 'basins')"
```

### Analysis

```bash
# per-molecule summary tables (parquet) from the whole store
python scripts/production/s0_package1_collect.py --tag prod
python scripts/production/s0_package1_crest_summarise.py --tag prod

# why refine=opt is worse than refine=sp, with the mechanism
python scripts/calibration/s0_A_refine_mechanism.py --species dsgdb9nsd_000018
```

**What to report, and what not to.** Wall clock, single-job time and slot extrapolation are
three separate numbers and must stay separate (branch E acceptance criterion 5). The batch
record computes the first two and deliberately refuses to compute the third: dividing a
batch wall clock by a worker count describes neither one molecule nor the batch. The only
cost figure this repository has is **285 s per species, 4 threads, uncontended, on a
workstation** — not on Tianhe, and not under contention with 15 neighbours sharing memory
bandwidth. Replace it with a real one from the first production shard.

---

## 8. Branch B, since it changed at the same time

Branch B's two halves now run on different machines:

```
trajectories   TianheXY-A, GPU, ONE TRAJECTORY PER CARD
               8 cards/allocation x 5 allocations = 40 concurrent
               python scripts/production/s0_E_branchB_parsl.py --edges \
                   --resource tianhe_a --route openmm
               bash hpc/slurm/submit_branchB_tianhe_a.sh prod ai

collection     TianheXY-C, CPU, ONE MOLECULE PER CORE, 64 at a time, ONE NODE
               python scripts/production/s0_E_branchB_collect_parsl.py --edges \
                   --resource tianhe_cpu --tag prod
               TAG=prod yhbatch hpc/slurm/branchB_collect.slurm
```

**The protocol is the published one** (`configs/branchB_protocol.yaml`): 1 fs timestep,
**520 ps equilibration + 1500 ps production, one frame every 0.5 ps → 3000 frames**, from
Rinaldo & Field, *Biophys. J.* 2003 — the paper this branch already cited for its
thermostat. Both trajectory routes and the Parsl driver read that one file; until
2026-09-07 they ran 50 + 200 ps and 2 + 25 ps respectively, which made the "independent
implementation pair" a pair of different protocols.

The constraint that stopped the source paper converging — a covariance from T frames has
at most T non-zero eigenvalues, and they had 7758 degrees of freedom against 3000 frames —
is 53–100× away from a 10–19 atom molecule. That is why this protocol is cheap here.

**TianheXY-A uses the whole node**: 56 workers, one per core, **7 sharing each card**. One
trajectory per card left 48 of 56 cores idle, and a 10-atom molecule cannot fill an 80 GB
H100 — the cost there is kernel-launch latency, not arithmetic. At the 5-node quota that
is 280 concurrent trajectories rather than 40.

**One node for collection, not twelve.** The pass reads frames and diagonalises a 3N×3N
covariance per molecule: small, serial, float64 — a card buys nothing, and its real cost is
metadata traffic on Lustre, so more nodes would buy contention rather than throughput. If it
ever becomes the bottleneck, batch more molecules per task.

**The GPU cost of branch B is now measured, and the answer depends entirely on one
platform property.** an45, 2026-09-12, MACE-OFF23_medium on 10 atoms, one trajectory at a
time on an A800 80 GB:

| platform | MACE dtype | CUDA `Precision` | ms/step |
|---|---|---|---|
| CPU (112 threads) | float64 | n/a | 61.7 |
| CPU (112 threads) | float32 | n/a | 34.9 |
| CUDA A800 80 GB | float64 | `single` (the default — **mismatched**) | **74.2** |
| CUDA A800 80 GB | float64 | `double` (matched) | **39.2** |
| CUDA A800 80 GB | float32 | `single` (matched) | **25.9** |

Match the CUDA `Precision` to the MACE dtype and the card wins by 1.6x (float64) or 1.3x
(float32). Leave it at OpenMM's default `single` under a float64 module and it loses to
the CPU -- that mismatched row is where the earlier claim in this file, "the card loses",
came from, and it has been withdrawn. `openmm_mace.platform_properties_for()` now sets
the matching property so the mismatch cannot be reached by leaving an argument out.

The 3.5x-slower T400 is not contradicted by this: it is a different card and
was never re-run. Still read `SECONDS_PER_PS` out of `md.toml` (`[Production]`) before sizing a
campaign.

The CPU route is kept, not deprecated: `--route ase --resource tianhe_cpu` is the
independent implementation pair that makes the OpenMM numbers checkable. Both write the
same Record (`md.out` + `md.toml`, `docs/output_inventory.md` section 8) beside their own
engine files, so the analysis reads either without knowing which produced it.

---

## 9. Executor labels

One place, `hpc/labels.py`, because a label repeated in two files is a defect waiting for
someone to edit one of them. **A Parsl app that names an executor the config does not
provide builds cleanly, renders cleanly, and then schedules nothing.**

| role | label | runs | where |
|---|---|---|---|
| `crest` | `openqha_crest_executor` | branch A conformer search | TianheXY-C, 16 × 4 threads |
| `qha` | `openqha_qha_executor` | branch B trajectories | TianheXY-A, 8 × 1 card |
| `collect` | `openqha_collect_executor` | branch B analysis | TianheXY-C, 64 × 1 core |
| `qm` | `openqha_qm_executor` | branch C ORCA reference | TianheXY-C |
| `train` | `openqha_train_executor` | branch C MACE fine-tuning | TianheXY-A / -AI |

Each has a `_standby_executor` variant (ALF's convention) for a cheaper or shorter queue;
omitting one is valid, and nothing schedules to a label that does not exist.

**A label never names a machine.** `openqha_crest_tianhe` was wrong twice over: it named the
site, and it did not match what the driver bound. When branch B's trajectories moved from
the CPU cluster to the GPU one, `openqha_qha_executor` did not have to change.

```bash
python hpc/labels.py            # the table
```

---

## 10. Verified here, and not

| checked without Tianhe | result |
|---|---|
| every config builds, every role, debug and production | **8/8 pass** |
| every job script renders (`#SBATCH` directives read) | **pass** |
| `--exclusive` absent on the GPU clusters | **pass** |
| `--gpus=N` present on TianheXY-A, submitted from the fine-grained environment | **revised 2026-09-11** — mandatory there (1 card = 12 CPUs = 120 GB); refused in the default environment, which has `Gres=(null)`. N is sized to the work, not to the node. See `docs/tianhe_runbook.md` footnote 2. |
| `--exclusive` present on the CPU cluster | **pass** |
| per-card pinning gives `['0'…'7']` | **pass** |
| all four `.slurm` and two `.sh` parse (`bash -n`) | **pass** |
| every changed Python file compiles and renders `--help` | **pass** |
| `--debug` refused where there is no short partition | **pass** |
| the GPU environment solves at CUDA 12.3 | **pass** (dry run) |
| the wall budget follows `--debug` (1620 s, not 544 320) | **fixed here** |

Defects found this way in code that had been written and never executed: the provider's
`_command_map` built after `super().__init__()`; `render_only` reading a module attribute
as an instance one; a duplicated `--ntasks-per-node`; an executor label that matched
nothing; `--exclusive` defaulted on by Parsl and banned by the site; a per-block
`CUDA_VISIBLE_DEVICES` that would have put all 8 workers on card 0; the branch B Parsl
driver still asking for the retired label `openqha_qha`; the OpenMM driver having no way to
address one `(basin, seed)` and writing to `$HOME` instead of `S0_RUNS_ROOT`.

**Still unverified, and only the machine can settle it:**

* the `debug` partition name and both walltimes are the user's, not read off `sinfo` here;
* CUDA/12.3 on TianheXY-AI (section 3);
* whether `--exclusive` is actually required on TianheXY-A — its manual says its nodes are
  exclusive and its `yhbatch --help` lists the flag, while the current guidance says not to
  pass it. If a submission is refused for want of exclusivity, `EXCLUSIVE` in
  `hpc/resource_configs/tianhe_a.py` is the line to change;
* every cost, on every cluster;
* `tianhexy-i` appears in the quota table and this repository knows nothing about it.
