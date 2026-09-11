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
| **TianheXY-A** | `tianhe_a.py` | GPU | `ai` (fine-grained²) | **per card**: 12 CPUs, 120 GB | never (= 8 cards) | **mandatory²** |

¹ By the 2026-09-05 ruling TianheXY-A follows TianheXY-AI's rules. Recorded tension: A's
own manual says its nodes are exclusive and its `yhbatch --help` *does* list the flag. If
a submission is ever refused for want of exclusivity, `EXCLUSIVE` in `tianhe_a.py` is the
line to change.

² **TianheXY-A has TWO Slurm environments behind one login prompt.** Measured on ln207,
2026-09-11, and explained by the site document 星逸 AI 集群 GPU 细粒度调用使用说明:

| | how you get there | `ai` nodes | gres | allocation | `-G` | `--mem` |
|---|---|---|---|---|---|---|
| default | fresh login | an[9..43] | `(null)` | by CPU | **refused** | — |
| fine-grained | `source /APP/u22/ai_x86/toolshs/set-XY-I.sh` (a PATH to a Slurm 24.05 client) | an[45-47,49-51,53,65] — 8 × (96 CPU, 960 GB, 8 × A800) | `gpu:a800:8` | **by card**: 1 GPU = 12 CPUs = 120 GB (= the node ÷ 8) | **mandatory** | **forbidden** |

Billing in the fine-grained environment is `max(gpus, ceil(cpus/12)) × hours × weight`
on the invoice (site PDF), over Slurm's own `TRESBillingWeights=CPU=1.0,Mem=0.0,GRES/gpu:a800=4.0`
(one card + 12 CPUs = 16, a node = 128). So a card's 12 CPUs cost nothing extra and a
13th costs a whole card. `nvidia-smi` inside a job shows only the allocated cards. The 12
CPUs are **6 physical cores × 2 threads** (`ThreadsPerCore=2`), which is why a worker
here is single-threaded. Limits: `MaxSubmit=10`, no node quota, `OverSubscribe=NO`,
`MaxTime=UNLIMITED`. **It is the controller's only partition** — `sinfo` after the script
lists `ai` and nothing else; the default controller's `temp` (2 h) is not reachable from
it, so the 30-minute gate is a short job on `ai`. **At the 2026-09-11 reading all 64
cards were allocated** while 248 CPUs sat idle — a new job waits on cards, not on CPUs.

**This project uses the fine-grained environment.** `examples/run_chain.sh` refuses to
submit to `ai` until `sinfo -h -p ai -o %G` mentions gpu — that is, until the script has
been sourced in the submitting shell. `D0-C-25` ("Tianhe requires an explicit `-G`") and
the 2026-09-11 "any `-G` is refused" are both true, of the two environments respectively.

One job carries the driver and the workers: `tianhe_a.py` sees `SLURM_JOB_ID` and runs
parsl in-allocation, so there is no second allocation and no idle driver card.
`S0_PARSL_NESTED=1` restores block submission for campaigns.

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

## 0b. Storage: two filesystems, and which clusters share one  **[storage tags read 2026-09-11]**

| filesystem | mounted by | path for this project |
|---|---|---|
| **XYFS02** | `tianhexy-cn`, `tianhexy-a`, `k8s_xingyi`, `k8s_xingyiAI` | `/XYFS02/HDD_POOL/<acct>/<user>/…/openQHA-main` |
| **XYAIFS00** | `tianhexy-ai`, `k8s_xingyiAI_2` | `/XYAIFS00/HDD_POOL/<acct>/<user>/…/openQHA-main` (and `/XYAIFS00/HOME/<acct>/<user>`) |

**TianheXY-CN and TianheXY-A see the same files.** So the two-step flow — branch A on
`deimos`, branch B on `ai` — needs no data movement at all: the basin list branch A writes
under `data/basins/<tag>/` is the file branch B reads, in the same repository checkout.
Submit both from the XYFS02 checkout and forget about copying.

**TianheXY-AI (`h100x`) is on a different filesystem.** Its checkout is a separate copy,
and the manual's §3.2.2 recipe is the only way across:

```bash
# once: upload the TianheXY-AI account's key to the XYFS02 side and lock it down
chmod 400 ~/accountB.id

# then, from the XYFS02 checkout: what h100x needs ...
bash hpc/tools/xfer_tianhe_ai.sh push data/basins/acetone data/potentials
# ... and what it produced
bash hpc/tools/xfer_tianhe_ai.sh pull data/trajectories/acetone
```

The script is `scp -i accountB.id -r …  accountB@XYAIFS00:…` with both roots filled in
(`XYAI_KEY`, `XYAI_ACCOUNT`, `XYAI_ROOT` override the defaults). Paths are relative to
the repository root on both sides so the trees stay congruent.

Practical consequence: **prefer `ai` over `h100x` for branch B** — not for the cards, but
because nothing has to be copied.

### Slurm semantics this project relies on  **[slurm.schedmd.com, read 2026-09-11]**

TianheXY-A's manual points at the stock Slurm documentation, so these are quoted from it
rather than from memory:

| option / variable | meaning (quoted) | how this project uses it |
|---|---|---|
| `--gpus`, `-G N` | "total number of GPUs required for the job" | the card count, per job |
| `--gpus-per-node` | "equivalent to the `--gres` option for GPUs" | not used; one node |
| `--gpus-per-task` | per task; "requires the job to specify a task count" | not used — that is the MPI shape |
| `--cpus-per-task`, `-c` | "job steps will require ncpus processors per task"; since 22.05 propagates as `SRUN_CPUS_PER_TASK` | `12 × G` on `ai` (what a card brings); `THREADS` on deimos |
| `--cpus-per-gpu` | CPUs per allocated GPU; inheriting steps imply `--exact` | not used; the site fixes 12 |
| `--mem`, `--mem-per-gpu` | memory | **forbidden** on `ai` (site PDF) |
| `CUDA_VISIBLE_DEVICES` | "set for each job step … restricted to allocated GPUs"; with cgroup fencing **renumbered from 0** inside the job | why `nvidia-smi -L` inside the job equals the allocation, and why parsl's card indices are `0..G-1` |
| `SLURM_GPUS` / `SLURM_GPUS_ON_NODE` / `SLURM_JOB_GPUS` | total / per node / "global GPU IDs allocated to the job" | `tianhe_a.py` reads them to size the in-allocation pool |
| `SLURM_CPUS_PER_TASK`, `SLURM_CPUS_ON_NODE` | what `-c` asked for; CPUs on this node | same |
| `#SBATCH` parsing | stops at "the first non-comment, non-whitespace line" | every directive precedes the first command in each `.slurm` |
| MPS | "GRES types of GPU and MPS can not be requested within a single job"; one MPS server per node | not something a job can enable for itself here |

**The shape: one task, many processes.** `--nodes=1 --ntasks=1 --gpus=G --cpus-per-task=12G`.
Parsl's worker pool is the fan-out (12 workers per card for branch B), and it pins each
worker to a card from `available_accelerators`. `--gpus-per-task` and `srun -n` are for
one process per card and are not what a bag of independent trajectories wants.

**Multi-threading on the CPU cluster** is the same option read the other way:
`--cpus-per-task=N` with `OMP_NUM_THREADS=N` inside — `hpc/env/common.sh` does that from
`THREADS`, and `deimos` still charges the whole 64-core node (`--exclusive` is required
there), so the way to use it is more molecules per node, not more threads per molecule.

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
| submit | `yhbatch` **on the GPU clusters**, `sbatch` **on TianheXY-CN** | Corrected 2026-09-09 on the machine. This row used to read "`sbatch` is not present", which was true of the cluster it was measured on and false of the CPU one. `run_chain.sh` takes the submitter from the partition |
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

> **A fuller treatment is now [`tianhe_install.md`](tianhe_install.md)** — the five site
> facts that decide whether an install works, exact pinned versions for both environments
> if you would rather drive it by hand, verification, and a table of every failure seen
> here so far. What follows is the summary; that file is where to go when it breaks.

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

## 5c. Submitting a chain: one file, two modes, three chains  **[dispatch verified locally 2026-09-09]**

Everything above submits through the same file. `examples/run_chain.sh` carries the
invariant `#SBATCH` directives and **submits itself**, taking partition, walltime and
`--gpus` as `yhbatch` flags, because those depend on the cluster and a `#SBATCH` line
cannot be parameterised.

```bash
# run it here, now, at production settings
bash examples/run_chain.sh examples/02d_qha_frequency_identity/chain.conf

# submit it
bash examples/run_chain.sh <conf> ai   # TianheXY-A,  8 cards, 7 d
bash examples/run_chain.sh <conf> h100x   # TianheXY-AI, 1 card,  3 d
bash examples/run_chain.sh <conf> deimos   # TianheXY-C,  CPU,     3 d
bash examples/run_chain.sh <conf> temp   # GPU short queue, 30 min
bash examples/run_chain.sh <conf> debug   # CPU short queue, 30 min
```

### Three site facts, all three measured on the machine 2026-09-09

**1. The submitter is not the same on both sides.** TianheXY-CN (the CPU cluster) is
stock Slurm and takes `sbatch`; the GPU clusters take the site's `yhbatch` wrapper. It
comes out of the partition table, so nobody has to remember:

| partition | cluster | submitter |
|---|---|---|
| `ai`, `temp` | TianheXY-A | `yhbatch` |
| `h100x` | TianheXY-AI | `yhbatch` |
| `deimos`, `debug` | TianheXY-CN | **`sbatch`** |

`OPENQHA_SUBMIT=<command>` overrides it. A submitter that is not on PATH is refused on
the login node with the partition named, rather than failing as an unrecognised command.

**2. The conda module names carry no dot.** `anaconda3/202309`, **not**
`anaconda3/2023.09`. Also present: `anaconda3/20250601`, `miniconda3/202409`,
`miniforge/24.7.1`. The script now tries these in turn, prints which one it loaded, and
accepts "conda is already on PATH" as a perfectly good answer — the previous version
asked for a name that does not exist and printed a bare `Unable to locate a modulefile`
that looks like a failure and was not. `OPENQHA_CONDA_MODULE=<name>` is tried first.

**3. The environment follows the partition.** `OPENQHA_ROLE` is set from the partition:
CPU → `openqha` (crest, xtb, pinned OpenBLAS), GPU → `openqha-gpu` (CUDA torch, OpenMM,
MKL). The old script hard-coded `openqha-gpu`, which would have put a CPU chain into an
environment with **no crest and no xtb** — branch A would then have failed on the compute
node for a reason that had nothing to do with branch A.

### Getting the weights onto the cluster

Loading a potential is three steps: the registry maps an engine **name** to a
**filename**, that filename is looked for in one flat directory, and that file is used.
Nothing reads the file, hashes it, or can refuse it. So the whole job is *put the right
file in the right place under the right name*.

```bash
# on a machine with bandwidth
bash install_dependency.sh                     # -> data/potentials/MACE-OFF23_medium.model
ls -l data/potentials/                         # check the size here, where you can see it

rsync -a --partial --progress data/potentials/ <you>@tianhe:~/openQHA/data/potentials/

# on Tianhe
ls -l ~/openQHA/data/potentials/               # same size? then it arrived
```

Two things that decide which file a run actually opens, and both are printed:

* `S0_MACE_ROOT` moves the whole directory; `S0_MACE_MODEL` overrides one file.
* the MACE server prints the engine name and the path it loaded on its first line, and
  `engine.provenance()` puts both in every product. **If a run's numbers are wrong,
  that line is where you look first** — it says which file produced them.

A truncated or wrong file will not be caught for you. It shows up as bad numbers, so
compare the size at both ends before you start a campaign.

### The two-step flow: one CPU job, then one GPU job

Every example is submitted twice, and the split is not administrative — branch A
**cannot** run on a GPU partition. It is CREST + GFN2-xTB: `xtb` has no GPU path, and
`openqha-gpu` contains neither `crest` nor `xtb`.

```bash
# step 1 -- CPU, minutes. Branch A only; the basins are the product.
bash examples/run_chain.sh examples/02b_qha_openmm_propanal/branchA.conf deimos

# step 2 -- GPU, hours to days. Finds those basins and skips branch A.
bash examples/run_chain.sh examples/02b_qha_openmm_propanal/chain.conf ai
```

| example | step 1 | step 2 | shared TAG |
|---|---|---|---|
| `02a` acetone | `deimos` | `ai` / `h100x` | `acetone` |
| `02b` propanal | `deimos` | `ai` / `h100x` | `propanal` |
| `02c` levels | `deimos` | **`deimos`** | `02c_prod` |
| `02d` identity | `deimos` | `ai` / `h100x` | `02d_prod` |

The two confs of an example **share a TAG** because the basin store is keyed by
`(species, tag)`. A step-2 submission with no basins under that tag is refused **on the
login node** — not in the queue — and prints the step-1 command for that example.

**Placement travels as script arguments, not in the environment.** `MODE` and `PARTITION`
are appended after the conf path (`... "$0" "$CONF" "$MODE" "$PARTITION"`) and take
precedence over the conf. Without that, a conf whose default is `PARTITION=ai` — which
02a and 02b both have — would be re-sourced inside a `deimos` job, derive `KIND=gpu`, and
try to `module load CUDA/12.3` on a CPU node. Found by inspection 2026-09-09, before it
cost an allocation.

### Node-local scratch: run there, carry the end state back  **[measured 2026-09-09]**

```
/tmp/<owner>/<SLURM_JOB_ID>/sockets/s0_mace_pool_0.sock
/tmp/<owner>/<SLURM_JOB_ID>/runs/branchA/...
```

**The node-local scratch had never taken effect.** `hpc/env/tianhe.sh` sets `S0_SCRATCH`
and `S0_RUNS_ROOT` under `TMPDIR`, and this runbook has always said so — but `common.sh`
is sourced first and its own `S0_RUNS_ROOT="${S0_RUNS_ROOT:-$HOME/runs/openQHA}"` had
already set the variable, so `tianhe.sh`'s `:-` kept it. The job banner read
`runs root  /HOME/…/runs/openQHA`, and CREST had been writing its thousands of small
files onto Lustre the whole time. `common.sh` now marks its value as a default and
`tianhe.sh` overrides a default — never an explicit `S0_RUNS_ROOT` you exported yourself.

> `${VAR:-default}` cannot be used for layered configuration: it cannot tell "nobody set
> this" from "the previous layer just set it". Mark the source, or override outright.

**What comes back.** `logs/openqha_<name>_<jobid>.{out,err}` as before, and the whole
node-local tree — sockets, CREST working directories, everything — copied to

```
logs/node_local/<jobid>/
```

before it is removed. `MANIFEST.txt` is written **first**, from `ls -laR`, because it is
the part that always works: `cp -a` recreates socket nodes on a filesystem that supports
them and drops them **silently** on one that does not. After copying, the entry counts
are compared and any shortfall is reported rather than left as a quietly shorter
directory.

`<owner>` is `S0_SOCKET_OWNER`, defaulting to your login name; `<SLURM_JOB_ID>` is
**Slurm's**, read from the environment and never invented — outside a job it falls back
to `pid<N>`. Base directory: `S0_SOCKET_DIR`, else `TMPDIR`, else `/tmp`.

They used to go under `runs_root/sockets/`, which is the **shared** filesystem. Two
branch A jobs submitted back to back (7346431 on `cnode1948`, 7346432 on `cnode2001`)
both opened `runs_root/sockets/s0_mace_pool_0.sock`, and whichever bound second unlinked
the first one's socket. A Unix socket is a rendezvous between processes on **one**
machine; it has no business on Lustre. `sun_path` also truncates past 108 bytes rather
than refusing, so `config.socket_path` raises above 100 characters instead.

**What the job leaves behind for you.** The MACE servers write `<socket>.log` beside
their socket, on the compute node's own disk, which is gone when the job ends. Before
removing that directory the job copies every **regular** file in it back to

```
logs/node_local/<jobid>/
```

The sockets themselves are not copied — a socket is not a file you can read afterwards.
The `rm -rf` is guarded: it only ever removes a path under a tmp base, and says so
loudly if the path resolved somewhere else.

### The CREST client's executable bit  **[measured 2026-09-09]**

`scripts/production/s0_mace_engrad.py` is **executed directly by CREST**, not as
`python <script>`, so it needs the bit. Git tracked it as mode `100644` while a Windows
working copy reports `rwxr-xr-x` for everything — so it looked fine on the workstation
and every fresh clone on a cluster arrived non-executable. It killed two branch A jobs
before anything else ran.

Fixed in the index (`git update-index --chmod=+x`), and `crest.write_input` now **sets
the bit** rather than refusing: it is our requirement, on our own script, and it is one
syscall. It still raises if the chmod fails — a read-only or `noexec` mount.

If you transferred the repository some way that drops modes, this is the check:

```bash
ls -l scripts/production/s0_mace_engrad.py     # want -rwx
```

### Which chain, and where each one belongs

| `CHAIN` | what it runs | partition |
|---|---|---|
| `conformers` | **branch A only** — the basins, nothing after | **`deimos`** |
| `qha` (default) | branch A → branch B → collect → `F_conf` | `ai` / `h100x` |
| `identity` | 02d: may `ν_k` replace `ω_i` in ZPE, enthalpy, entropy | `ai` / `h100x` |
| `levels` | 02c: MACE vs GFN2-xTB vs RI-MP2 | **`deimos` only** |

`CHAIN` is set **in the conf**, not in the environment, so a submission is reproducible
from the file alone — the job re-sources the same conf on the compute node.

**`CHAIN=levels` on a GPU partition is refused on the login node.** ORCA has no GPU path
in this repository and `D0-75` puts production quantum chemistry on deimos; catching it
before `yhbatch` turns a wasted allocation into a one-line error.

### What Tianhe actually buys you here, and what it does not

**It does not make one trajectory faster.** A ten-atom molecule on MACE-OFF23_medium is
latency-bound, not throughput-bound: the only GPU figure this repository has is 3.5×
*slower* than CPU on a T400 (`D0-C-5`), which does not transfer to an 80 GB card and has
**not been replaced by a measurement**. Do not plan as though it will.

**It buys concurrency.** A production 02d run is 2 molecules × up to 4 basins × 3 seeds =
**up to 24 independent trajectories**, and they have no communication between them at all.
That is what the resource profiles are shaped for:

| profile | layout | good for |
|---|---|---|
| `tianhe_a` (`ai`) | 56 workers on 8 cards, 7 per card | many short-ish trajectories |
| `tianhe_ai` (`h100x`) | the allocation **is** one card + 14 CPUs | one species, one long run |
| `tianhe_cpu` (`deimos`) | 64 workers × 1 core × 1 node | `levels`, and the collection step |

`available_accelerators` is passed to Parsl as an **int**, never a list. Replay of the
arithmetic (2026-09-07): an int gives `{card 0: 7 workers, …, card 7: 7}`; a hand-built
list gave `{0: 49, 1: 7}` — 49 of 56 workers on one card, which reads as "the GPU is
slow" and is not.

### The two settings a production 02d run must get right

```bash
PROD_PS=500        # the protocol's production length, not shortened
SAMPLE_EVERY=2     # steps, i.e. one frame per 2 fs -- NOT the protocol's 1.0 ps
```

The second one is the whole point and it is worth understanding before you spend a week
of allocation on it. The protocol samples every 1.0 ps to filter high-frequency noise out
of the **entropy**. Measured in 02d stage 1, on a surface that is *exactly* harmonic so
that sampling is the only thing that can be wrong:

```
frames      dZPE       dT*S
   500    +1.4246    +0.0226      <- what 500 ps at 1.0 ps actually gives you
 12500    -0.0035    -0.0037
```

**At the protocol's own interval a 500 ps run carries a +1.42 kcal/mol sampling bias in
the zero-point energy before any physics enters**, while the entropy is already fine. The
two products of one trajectory want opposite intervals. 12 500 frames at 1.0 ps would be
12.5 ns; at 2 fs it is 25 ps, and 500 ps at 2 fs gives 250 000 frames — 20× more than the
harmonic limit needs, which leaves no room to argue that a residual is undersampling.

Cost of the density: **60 MB per trajectory** (250 000 × 10 × 3 × 8 bytes). The entropy
analysis subsamples back to 1.0 ps from the same file, so one run answers both questions
and the two answers cannot come from different trajectories.

### Before the long one: the 30-minute gate

```bash
bash examples/run_chain.sh <conf> temp   # GPU chains
bash examples/run_chain.sh <conf> debug   # CHAIN=levels
```

Same file, same conf, same settings — only the queue and the walltime change. **It is not
a smoke test and its numbers are not results**: it is a check that the environment loads,
the weights are present, the partition accepts the flags and the drivers start. Read the
gate for "did it start", never for "what is the answer".

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
`HDD_POOL` — on **XYFS02** for the CN and A clusters (one checkout serves both), on
**XYAIFS00** for the AI cluster (a separate copy; see §0b). **XYFS01 has no backup** — a
deleted file is gone.

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
