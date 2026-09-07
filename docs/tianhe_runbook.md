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

# The proxy. conda HANGS rather than fails without it, so do this first.
source /APP/u22/ai_x86/toolshs/setproxy.sh 172.16.31.200 3138
curl -sI https://conda.anaconda.org | head -1     # expect: HTTP/... 200 or 301
```

If that `curl` hangs, stop. Everything after it will hang too, and a hung job is charged.

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
* **CUDA/12.3 on TianheXY-AI.** The recorded listing was elided (`CUDA/11.8 ...
  CUDA/13.2`), so 12.3 is inside the range but was never read off the screen. The conda
  side is pinned to it. One line to change if it is absent —
  `CUDA_VERSION` in `hpc/resource_configs/tianhe_ai.py` and the pin in
  `environment-tianhe-gpu.yml`;
* on TianheXY-A, **9 of 25 `ai` nodes were in state O** at that reading. Effective capacity
  is below the node count; check `yhi` before planning around 25.

---

## 3. Install  **[unverified]**

```bash
git clone <this repo> openQHA && cd openQHA     # or rsync it in

bash install_dependency.sh --tianhe        # TianheXY-C  -> `openqha`      (CPU)
bash install_dependency.sh --tianhe-a      # TianheXY-A  -> `openqha-gpu`  (CUDA 12.3)
bash install_dependency.sh --tianhe-cuda   # TianheXY-AI -> `openqha-gpu`  (the same file)
```

Neither GPU flag loads MPI: nothing openQHA runs on a card needs collectives, and dropping
it is what let one environment file serve both GPU clusters.

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
