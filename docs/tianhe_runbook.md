# Tianhe runbook — the site facts, and the first hour

Site facts and the first-login sequence. **For what to run, where each setting lives and
where the answer lands, read [`branchA_production.md`](branchA_production.md)** — this
file is the machine, that one is the work.

**Nothing in this repository has ever been submitted to Tianhe.** Steps are marked
**[measured]** or **[unverified]** so you know which is which before you spend an
allocation.

---

## 0. Three clusters, not one

An earlier version of this document described one machine and got branch A onto the wrong
one. There are three, with different partitions, different allocation policies and
different filesystems:

| | config | kind | partitions | allocation | `--exclusive` | `--gpus` |
|---|---|---|---|---|---|---|
| **TianheXY-C** | `tianhe_cpu.py` | CPU | `debug`, `deimos` | whole node, 64 cores | **required** | n/a |
| **TianheXY-AI** | `tianhe_ai.py` | GPU | `hx`, `h100x`, `a100x`, `a800x`, `v100x` | **per card** | **BANNED** | **mandatory** |
| **TianheXY-A** | `tianhe_a.py` | GPU | `temp`, `ai` | node, 8 cards, 56 cores | not passed¹ | passed |

¹ By the 2026-09-05 ruling TianheXY-A follows TianheXY-AI's rules. Recorded tension: A's
own manual says its nodes are exclusive and its `yhbatch --help` *does* list the flag. If
a submission is ever refused for want of exclusivity, `EXCLUSIVE` in `tianhe_a.py` is the
line to change.

**Note how opposite the first two are.** `--exclusive` is required on one and rejected by
the other. Parsl's `SlurmProvider` defaults it to `True`, so a stock provider is refused
by both GPU clusters — which is why these are three files and not one with an `if`.

### What runs where

```
branch A  (CREST conformer search)      TianheXY-C   16 x 4 threads x 12 nodes = 192
branch B  trajectories                  TianheXY-A   8 cards/node x 5 = 40
branch B  collection                    TianheXY-C   64 x 1 core x 1 node
branch C  training                      TianheXY-A (preferred) or TianheXY-AI
```

Branch A cannot move: its cost is ~1e5 GFN2-xTB gradient calls and xtb has no GPU path.

### Quotas (Starlight, **[measured 2026-09-05]**)

| cluster | running jobs | **nodes** |
|---|---:|---:|
| tianhexy-cn (C) | 32 | **32** |
| tianhexy-a (A) | 10 | **5** |
| tianhexy-ai (AI) | 6 | **6** |
| tianhexy-i | 10 | 5 |

The node quota binds first. **Parsl does not know quotas exist** — past one it keeps
asking, the scheduler keeps refusing, and the run looks *stalled* rather than capped. Each
resource config enforces the cap with `min()` and `describe()` reports it beside the
setting. `tianhexy-i` appears in the table and this repository knows nothing else about it.

---

## 1. First login — five minutes, no allocation spent  **[unverified]**

```bash
ssh <you>@tianhe
cd $HOME                               # /HOME/hku2021_fos4/hku2021_fos4xy_2

# The proxy. Without it a login node has NO DNS -- conda does not fail cleanly, it
# retries for minutes and then blames whichever mirror it was reaching for. Do it first.
source /APP/u22/ai_x86/toolshs/setproxy.sh 172.16.31.200 3138

# Probe the mirror you actually use, which on this account is TUNA (see §3a).
curl -sI https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/noarch/repodata.json | head -1
```

Expect `HTTP/… 200`. If that `curl` hangs, stop — everything after it will hang too, and a
hung job is charged for its whole walltime. `install_dependency.sh --tianhe*` runs this
same probe for you and refuses to start when it fails.

---

## 2. Confirm the machine has not moved  **[measured 2026-09-05]**

```bash
type -a yhbatch yhrun sacct squeue scancel yhacct yhqueue yhcancel
sinfo -o "%P %c %m %D"
module avail CUDA 2>&1 | grep -o "CUDA/12[.][0-9]*"      # GPU clusters only
```

**All five scheduler commands exist**, and so do the `yh*` variants:

| role | used | why |
|---|---|---|
| submit | `yhbatch` | `sbatch` is **not** present |
| launcher | `yhrun` | |
| status | `sacct` | `yhacct` exists too, but Parsl **parses** this output and nobody has read `yhacct`'s format |
| status fallback | `squeue` | same reason |
| cancel | `scancel` | same |

```bash
python -c "
import sys; sys.path.insert(0,'hpc')
import providers, json; print(json.dumps(providers.preflight('tianhe'), indent=1))"
```

Every `configured_exists` must be `true`.

### Still to confirm

* the `debug` partition name and the two walltimes (30 min / 3 days) are the user's, not
  read off `sinfo` here;
* ~~**CUDA/12.3 on TianheXY-AI.**~~ **Closed 2026-09-08.** A full `module avail` on
  `ln302%TianheXY-AI` lists `CUDA/12.3` outright, beside 12.0, 12.1, 12.2 and 12.4. The
  earlier listing had been recorded elided (`CUDA/11.8 … CUDA/13.2`), so 12.3 was inside
  the range but had never been read off the screen. **12.4 is present on -AI and must
  still not be used** — TianheXY-A has no 12.4, and matching both is what makes one
  `environment-tianhe-gpu.yml` serve both clusters;
* on TianheXY-A, **9 of 25 `ai` nodes were in state O** at that reading. Effective capacity
  is below the node count; check `yhi` before planning around 25.

---

## 3. Install  **[install path measured 2026-09-08; the run itself still unverified]**

```bash
git clone <this repo> openQHA && cd openQHA     # or rsync it in

bash install_dependency.sh --tianhe        # TianheXY-C  -> `openqha`      (CPU)
bash install_dependency.sh --tianhe-a      # TianheXY-A  -> `openqha-gpu`  (CUDA 12.3)
bash install_dependency.sh --tianhe-cuda   # TianheXY-AI -> `openqha-gpu`  (the same file)

# ONLY ONE CLUSTER ON YOUR ACCOUNT? Build both environments on it. Two separate solves,
# not one bigger environment -- see "Two environments" below for why that distinction is
# the whole point.
bash install_dependency.sh --tianhe-cuda --both     # -> openqha AND openqha-gpu
```

Neither GPU flag loads MPI: nothing openQHA runs on a card needs collectives, and dropping
it is what let one environment file serve both GPU clusters.

### 3a. The install that failed, and what it was really telling you

Run on `ln302%TianheXY-AI`, 2026-09-08. Forty urllib3 retry lines, then:

```
CondaHTTPError: HTTP 000 CONNECTION FAILED for url
<https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/linux-64/repodata.json>
Failed to resolve 'mirrors.tuna.tsinghua.edu.cn' ([Errno -3] Temporary failure in name resolution)
```

**Read the second line, not the first. "Failed to resolve" is DNS — not a refused
connection, not a 404, not a stale mirror.** A Tianhe login node has no direct DNS, so
until `setproxy.sh` has run *nothing* resolves: TUNA, `conda.anaconda.org` or anything
else. One cause, one fix. The forty retry lines and the TUNA URL in the message are what
made it look like a mirror problem, and it was never one.

TUNA is the intended mirror on all three clusters and `~/.condarc` is correct as it
stands — same file on TianheXY-CN, -A and -AI:

```yaml
channels: [defaults]
default_channels: [<TUNA>/anaconda/pkgs/{main,r,msys2}]
custom_channels:  {conda-forge: <TUNA>/anaconda/cloud, pytorch: <TUNA>/anaconda/cloud}
```

The environment files add `conda-forge` and `nodefaults`, so `conda-forge` resolves
through `custom_channels` to `<TUNA>/anaconda/cloud/conda-forge` and that is the only
channel a solve here touches. Checked 2026-09-08: that path returns **HTTP 200**, and its
`linux-64/repodata.json` (443 MB) was last modified **the same day** — the mirror is live
and current. `pypi.tuna.tsinghua.edu.cn` serves `mace-torch` too.

**The installer therefore changes nothing about your channels.** It does three things,
none of which reads or writes `~/.condarc`:

1. **Sources the site proxy itself** (host/port from `configs/cluster_tianhe.yaml`) and
   exports all four spellings plus `no_proxy`, so pip, curl and the weight transfer see
   the same proxy conda does. This used to be a *warning* telling you to export it
   yourself, which is what let the failing run start at all.
2. **Probes the mirror your condarc actually names** — read out of
   `conda config --show custom_channels`, not hard-coded — once, with a 15-second
   timeout, and stops there. So a missing proxy is one line naming the proxy, not forty
   naming a mirror.
3. **Silences the notice/retry storm** with `CONDA_NUMBER_CHANNEL_NOTICES=0` and the
   `CONDA_REMOTE_*` timeouts. These are *scalar* conda parameters, which environment
   variables replace cleanly — so no file is touched. (That is specifically not true of
   `channels`; see the table below.)

pip gets `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple` for the run, to match —
`mace-torch`, `pymsym` and `parsl` have no conda-forge package and are otherwise the
slowest part of the install. Exported per-run; `~/.pip/pip.conf` is not written.
`OPENQHA_PIP_INDEX=<url>` overrides it, `OPENQHA_PIP_INDEX=` opts out to `pypi.org`.

```bash
TIANHE_PROXY_HOST=<host> TIANHE_PROXY_PORT=<port> bash install_dependency.sh --tianhe-cuda
OPENQHA_SKIP_NET_CHECK=1 bash install_dependency.sh --tianhe-cuda   # skip the probe
```

#### If TUNA is missing a package — a different failure, and a different lever

TUNA mirrors conda-forge's *content*. If it is stale or incomplete for a pin this project
needs (`openmm-torch=*cuda*`, `cuda-version=12.3`, `crest>=3.0.2`), the solve fails with
**`PackagesNotFoundError`** — a content error. No proxy setting helps and retrying will
not either. Only then:

```bash
OPENQHA_FORCE_ANACONDA=1 bash install_dependency.sh --tianhe-cuda   # one run, direct
```

It is off by default, still needs the proxy, and still does not edit `~/.condarc`. If
what you see instead is `Failed to resolve`, `HTTP 000` or a hang, this is the wrong
lever — that is the proxy.

**Why that flag needs a scratch `HOME` rather than one obvious env var:** both obvious
mechanisms were measured against conda 24.1.2 on 2026-09-08 and neither works.

| attempt | result |
|---|---|
| `CONDARC=<file> conda …` | **ignored outright.** `conda config --show-sources` with it set lists only `~/.condarc`; the named file never appears |
| `CONDA_CHANNELS=<url> conda …` | **merges, does not replace.** `channels` is a sequence and conda concatenates across sources, so the result kept the TUNA-mapped names in the list |

Map parameters cannot be emptied from the environment at all (`CONDA_CUSTOM_CHANNELS=''`
is a type error). A `.condarc` inside a `HOME` we control has nothing to merge with,
which is the whole reason for the indirection — and it is why the *scalar* settings in
point 3 above need no such trick. One hole remains and the installer reports it every
run: `$CONDA_PREFIX/.condarc` sits *above* `~/.condarc` in the search path, so a config
written into the miniforge3 base is not shadowed; the effective channel list is printed.
### 3b. Two environments, and why they are not one

`openqha` (CPU) and `openqha-gpu` (GPU) are split along their **linear algebra**, which is
the one difference their names do not show:

| | `openqha` | `openqha-gpu` |
|---|---|---|
| BLAS | OpenBLAS, **OpenMP build**, pinned; `nomkl` | MKL (whatever the solver picks) |
| torch | CPU | CUDA 12.3 |
| CREST + xtb | yes | **no** |
| runs | branch A, QHA collection | branch B trajectories, branch C training |

Three requirements, and no single solve satisfies all three:

* conda-forge's `crest` links the **pthreads** OpenBLAS while CREST is OpenMP-parallel.
  Measured 2026-09-04 on `dsgdb9nsd_000018`, `-T 4`, same 2 conformers: pthreads 38.0 s and
  4164 warning lines, OpenMP 18.5 s and none. Hence the pins.
* A CUDA build of pytorch **depends on MKL**, so `nomkl` and a CUDA torch cannot both hold.
* Dry runs 2026-09-07: `openmmtools` + pins solves; `openmmtools` + CUDA torch solves;
  `openmmtools` + CUDA torch + `cuda-version=12.3` + pins **fails**. Each pair, never the
  triple.

So the pins live where CREST and the quasi-harmonic diagonalisation are, and the GPU half
takes MKL — where neither of those runs. **The diagonalisation, the only step whose numbers
a BLAS could move, always executes in the collection pass, in `openqha`, against OpenBLAS.**

Pick one per job and let `hpc/env/tianhe.sh` activate it:

```bash
OPENQHA_ROLE=cpu source hpc/env/tianhe.sh    # branch A, QHA collection
OPENQHA_ROLE=gpu source hpc/env/tianhe.sh    # branch B trajectories, branch C training
```

> **Never `conda activate openqha` and then `conda activate openqha-gpu`.** Conda *stacks*
> rather than replaces, so both prefixes stay on the loader's search path — one carrying
> OpenBLAS and one MKL — and which you get is decided by link order, not by the name you
> typed. It imports cleanly and returns numbers from a library you did not choose.
> `hpc/env/tianhe.sh` deactivates down to base first, and reports the BLAS it ended up with
> (`OPENQHA_BLAS`, in every job log via `openqha_report_env`).
>
> That report reads conda-meta build strings, and not the two things you would try first:
> `numpy.show_config()` reports `"name": "blas"` for both providers under conda-forge's
> `libblas` metapackage, and the `libblas.so.3` symlink resolves to `libopenblasp-*.so`
> even in an OpenMP-pinned environment — the `p` is conda-forge's file naming, not
> "pthreads". Both were checked on 2026-09-08; both would have given the wrong answer.

**None of them downloads the MACE-OFF weights.** Fetch them where you have bandwidth and
copy them in — the reasoning, the procedure and the verification are section 4 of
[`branchA_production.md`](branchA_production.md):

```bash
# on your workstation
bash install_dependency.sh
rsync -a data/potentials/ <tianhe>:$HOME/openQHA/data/potentials/
```

Then on Tianhe — the hash is recomputed on every load, so a truncated transfer fails here
rather than three hours into a campaign:

```bash
source env_openqha.sh
python -c "from openqha import engine, json; print(json.dumps(engine.provenance(), indent=1, default=str))"
python check_dependency.py              # ends with: branch A READY / NOT READY
```

---

## 4. Read the job script before submitting  **[verified locally]**

Optional now — section 5 is the real gate — but cheap, and it is how five defects were
found in code that had been written and never executed:

```bash
python - <<'PY'
import sys, pathlib
sys.path.insert(0, "hpc")
from resource_configs import load
import json
m = load("tianhe_cpu")
print(json.dumps(m.describe(), indent=1))              # measured vs assumed
p = m.config(role="crest").executors[0].provider
print(p.render_only(pathlib.Path("/tmp/openqha_render.sh")))
PY
```

Check the partition, the walltime and the `#SBATCH` directives. **A config that reads
correctly and renders wrongly is exactly the failure this step exists to catch** — a
duplicated `--ntasks-per-node` was found this way, and so was `--exclusive` appearing on a
cluster that bans it.

---

## 5. The gate: a real 30-minute job  **[unverified]**

**Not a dry run** (user ruling 2026-09-07). A short real job tests five things a plan
cannot: that the module loads, that conda activates on a *compute* node, that the weights
are there and hash correctly, that the scheduler accepts the directives, and that Parsl can
read its own status query.

```bash
mkdir -p $HOME/HDD_POOL/runs/openQHA/logs

# branch A, one edge, `debug`, 00:30:00 -- either route
python -u scripts/production/s0_E_branchA_parsl.py \
    --species dsgdb9nsd_000018 dsgdb9nsd_000035 \
    --resource tianhe_cpu --tag tianhe_debug --debug
EDGE=C3H6O1N0_18_35 TAG=tianhe_debug yhbatch hpc/slurm/branchA_debug.slurm

# branch B, one species, `temp`, 00:30:00
bash hpc/slurm/submit_branchB_tianhe_a.sh a_debug temp
```

Expect 9/9 acceptance criteria per molecule. **Then compare the two routes' products for
the same molecule** — if they differ, the execution layer is doing something it must not.

---

## 5b. Branch B: trajectories on the GPU, collection on the CPU

Branch B is split across two clusters, and the split is not arbitrary — the two halves
want opposite machines.

```
trajectories   TianheXY-A `ai`      GPU, one trajectory per card share, 7 days
               (or TianheXY-AI `h100x` for a single molecule end to end)
collection     TianheXY-C `deimos`  CPU, one molecule per core, 64 on ONE node
```

### Smoke test first — always

```bash
mkdir -p $HOME/HDD_POOL/runs/openQHA/logs

# Parsl route: one species, one seed, 20 ps, 30 minutes on the short queue
python -u scripts/production/s0_E_branchB_parsl.py --species dsgdb9nsd_000018 \
    --resource tianhe_a --route openmm --seeds 1 --prod-ps 20 --debug

# yhbatch route
bash hpc/slurm/submit_branchB_tianhe_a.sh a_debug temp
```

**Read `seconds_per_ps_this_run` out of `meta.json` before sizing anything.** Branch B on
a card is UNMEASURED here: the only GPU figure this repository has is 3.5x *slower* than
CPU, on a T400 (`D0-C-5`). That does not transfer to an 80 GB card, and nothing has
replaced it.

Also check the `platform` field in every `meta.json`. **If it is not `CUDA`, the job ran
on the CPU** and the timings mean something else entirely.

### Production

```bash
python -u scripts/production/s0_E_branchB_parsl.py --edges \
    --resource tianhe_a --route openmm \
    2>&1 | tee $HOME/HDD_POOL/runs/openQHA/logs/branchB_$(date +%Y%m%d_%H%M).log

# when the trajectories are done, collect on the CPU cluster
python -u scripts/production/s0_E_branchB_collect_parsl.py --edges \
    --resource tianhe_cpu --tag prod
```

Both resume: trajectories flush every 250 frames and stop themselves at 90% of the
walltime, and the collector subtracts molecules that already have a result. Re-running
the same command does the remainder.

### The protocol is in a config file, not on the command line

[`configs/branchB_protocol.yaml`](../configs/branchB_protocol.yaml) — 1 fs, 50 ps
equilibration, 500 ps production, one frame per 1.0 ps. Both trajectory routes and the
Parsl driver read it, so `--prod-ps` is for a smoke test, not for production.

**These are NOT the published protein numbers**, and the reason is in
[`branchB_production.md`](branchB_production.md) §2: a 10-19 atom molecule crosses its
torsional barriers on a timescale where a protein does not move, and a trajectory that
changes basin is not measuring an intra-basin entropy. Longer is not better here.

### The one thing to check before believing a number

```bash
python - <<'PY'
import json, numpy as np, sys
from openqha.quasi_harmonic import qha, basin_residence as br
meta = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "meta.json"))
frames = np.load("frames.npy")
res = br.basin_residence(frames, meta["symbols"])
sat = qha.saturation_curve(frames, meta["masses_amu"])
print(json.dumps(br.interpret_saturation(sat, res), indent=1))
PY
```

`stayed_in_one_basin: false` means the trajectory left its basin and T*S is inflated —
however converged the saturation curve looks. A failed criterion 1 has two causes that
demand **opposite** actions (run longer / run shorter), and this is what tells them apart.

---

## 6. Production

Settings live in the config, not the command line. See
[`branchA_production.md`](branchA_production.md) §6.3 for both routes, §7 for where the
results land and how to read them.

```bash
yhq -a                 # queue; works even with no allocation left
yhi                    # node states: idle / mix / alloc
yhcancel <jobid>
```

Scratch is node-local (`$TMPDIR`); products are written back into the repository. **Never
let CREST write its working directories on Lustre** — one molecule is 2.6–15 MB across
dozens of small files, and that is slow for you and for everyone else on the machine.
`hpc/env/tianhe.sh` sets this up; a worker that does not source it is a different machine.

Home is quota'd at 100 GB and is for configuration. Code, data and job output go in
`HDD_POOL`. **XYFS01 has no backup** — a deleted file is gone.

---

## 7. What to report, and what not to

From five repeat runs on one molecule (**[measured 2026-09-05]**, see
[`branchA_workflow.md`](branchA_workflow.md) §8b):

* the **basin count varies by 14%** run to run — iMTD-GC is a stochastic search;
* the **conformational correction varies by 0.0034 kcal/mol**, 3.4% of the target.

**Report the correction. Report a basin count as one draw, never as the answer.**

The only cost figure this repository has is **285 s per species, 4 threads, uncontended,
on a workstation** — not on Tianhe, and not under contention. Branch B on a GPU is
**unmeasured** here; the only figure is D0-C-5, 3.5× *slower* than CPU on a T400, which
does not transfer to an 80 GB card and has not been replaced. When you have real numbers,
report wall clock, single-job time and slot extrapolation as three separate numbers, so
nobody later divides one by another.
