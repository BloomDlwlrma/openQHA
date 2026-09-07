# Running branch A on Tianhe — a runbook

**Nothing in this repository has ever been submitted to Tianhe.** This document is written
from the site facts in [`configs/cluster_tianhe.yaml`](../configs/cluster_tianhe.yaml)
(supplied 2026-08-31) plus everything that could be checked without the machine. Steps are
marked **[verified locally]** or **[unverified]** so you know which is which before you
spend an allocation.

---

## 0. What was already checked here, and what could not be

Building the Tianhe configuration on a workstation found **four real defects** in code
that had been written and never executed, and the site manual then exposed a fifth in an
upstream default. That is the argument for doing steps 1–4 before step 5 rather than
after.

| checked without Tianhe | result |
|---|---|
| the Parsl config constructs | **fixed** — `_command_map` was built *after* `super().__init__()`, and parsl probes the scheduler *inside* it. Every construction raised `AttributeError`. |
| the job script renders | **fixed** — `render_only()` read `self.template_string`; in parsl 2026.08.10 that is a *module* attribute. The check meant to precede every first submission had never once run. |
| the rendered directives | **fixed** — `#SBATCH --ntasks-per-node=1` appeared twice; parsl emits it already. |
| the executor label | **fixed** — the config labelled it `openqha_crest_tianhe` while the driver binds `executors=["openqha_crest"]`. It would have built cleanly and matched no executor at run time. |
| `--exclusive` in the rendered script | **fixed** — parsl sets `exclusive=True` by default and the site **bans** the flag. Every job would have been rejected. |
| a failing status query is loud | **works** — raises with the command and what to edit. |
| env scripts parse | **works** |

**What still cannot be checked from here:** whether the proxy line is current, the exact
CPU package behind one GPU, and any real cost.

---

## 1. First login — five minutes, no allocation spent  **[unverified]**

```bash
ssh <you>@tianhe                       # your usual route
cd $HOME                               # /HOME/hku2021_fos4/hku2021_fos4xy_2

# The proxy. conda HANGS rather than fails without it, so do this first.
source /APP/u22/ai_x86/toolshs/setproxy.sh 172.16.31.200 3138
curl -sI https://conda.anaconda.org | head -1     # expect: HTTP/... 200 or 301
```

If that `curl` hangs, stop. Everything after it will hang too, and a hung job is charged.

---

## 2. What the site actually is  **[measured 2026-09-05]**

These were open questions until the first login. They are answered now, and the answers
changed the configuration substantially.

```bash
type -a yhbatch yhrun sacct squeue scancel yhacct yhqueue yhcancel
sinfo -o "%P %c %m %D"
```

**All five scheduler commands exist**, and so do the `yh*` variants:

| role | used | why |
|---|---|---|
| submit | `yhbatch` | `sbatch` is **not** present |
| launcher | `yhrun` | |
| status | `sacct` | `yhacct` exists too, but parsl **parses** this output and nobody has read `yhacct`'s format |
| status fallback | `squeue` | same reason as `sacct` |
| cancel | `scancel` | same |

**There is no CPU partition.** All five are GPU partitions:

```
hx     128 CPUs   2 nodes      h100x  128 CPUs  11 nodes
a100x  112 CPUs  10 nodes      a800x  112 CPUs   4 nodes
v100x   56 CPUs  15 nodes   <- the default
```

Branch A is CPU work but must hold a card anyway, so the question is which card to waste.
`v100x`: oldest, most nodes, least contended. Burning an H100 to run xTB is worse in
every direction.

### Three site rules, from the manual (§6.2.2) — two of them are showstoppers

| rule | text | what would have happened |
|---|---|---|
| `-G`/`--gpus` **mandatory** | 提交作业时必须使用 --gpus 或 -G 参数 | job never starts |
| `--exclusive` **banned** | 该参数已被集群调度系统禁用 | **parsl defaults `exclusive=True`** — every job rejected |
| memory **must not be set** | 禁止指定内存大小 | we never set it; compliant by luck |

The middle one is the one to remember: the defect was in an upstream default, and only
rendering the script showed it.

### Still to confirm before a large campaign

`CPUS_PER_GPU = 14` comes from the manual's own "1GPU/14CPUs" example and v100x's 56
CPUs/node = 4×14. **Check it against your own allocation** (`yhi`, or the Starlight
resource page) — if the package differs, the worker count is wrong.

---

## 3. Install  **[unverified]**

```bash
git clone <this repo> openQHA && cd openQHA     # or rsync it in
bash install_dependency.sh --tianhe
```

`--tianhe` loads `anaconda3/2023.09` (conda is not on PATH until you do) and builds
`openqha` from `environment-tianhe.yml` — the **CPU** torch build, deliberately, on a
machine where every job holds a GPU it does not use. It **does not download the MACE-OFF
weights** — a 100 MB pull through the proxy from a login node is
antisocial. Fetch them where you have bandwidth and copy them in:

```bash
# on your workstation
bash install_dependency.sh              # downloads and hash-checks
rsync -a data/potentials/ <tianhe>:$HOME/openQHA/data/potentials/
```

Then verify on Tianhe — the hash is recomputed on every load, so a truncated transfer
fails here rather than three hours into a campaign:

```bash
source env_openqha.sh
python -c "from openqha import engine; import json; print(json.dumps(engine.provenance(), indent=1, default=str))"
python check_dependency.py              # ends with: branch A READY / NOT READY
```

---

## 4. Look at the job script before submitting anything  **[verified locally]**

This is branch E acceptance criterion 3, and it is cheap:

```bash
cd $HOME/openQHA && source env_openqha.sh

python - <<'PY'
import sys, pathlib
sys.path.insert(0, "hpc")
from resource_configs import load
m = load("tianhe")

import json
print(json.dumps(m.describe(), indent=1))          # what is measured, what is assumed

p = m.config().executors[0].provider    # SETTINGS supply partition and account
print(p.render_only(pathlib.Path("/tmp/openqha_render_check.sh")))
PY
```

Read the script. Check the partition, the account, the walltime, and that the `#SBATCH`
directives are what the site expects. **A config that reads correctly and renders wrongly
is the failure this step exists to catch** — the duplicate `--ntasks-per-node` above was
found exactly this way.

Then confirm the command map is what you decided in step 2:

```bash
python -c "
import sys; sys.path.insert(0,'hpc')
import providers, json; print(json.dumps(providers.preflight('tianhe'), indent=1))"
```

Every `configured_exists` must be `true` and every `evidence` should now say measured.

---

## 5. One molecule, one job  **[unverified]**

Do not start a campaign. Run the smallest thing that exercises the whole chain:

```bash
cd $HOME/openQHA && source env_openqha.sh

python -u scripts/production/s0_A_pipeline.py \
    --species dsgdb9nsd_000018 \
    --tag tianhe_smoke --threads 4 --timeout-s 14400
```

Expect: `9/9 acceptance criteria`, and a product under
`analysis/branchA/tianhe_smoke/dsgdb9nsd_000018/`. If a criterion fails, read its
`detail` line — each one names what it compared.

Then the same molecule *through Parsl*, still one molecule, to test the execution layer
rather than the science:

```bash
python -u scripts/production/s0_E_branchA_parsl.py \
    --species dsgdb9nsd_000018 \
    --resource tianhe --tag tianhe_smoke_parsl
```

**Replacing Parsl with a for-loop must not change a single number.** Compare the two
products; if they differ, the execution layer is doing something it should not.

---

## 6. Production  **[unverified]**

**Settings live in the config, not the command line** (user ruling 2026-09-05). Edit the
SETTINGS block at the top of `hpc/resource_configs/tianhe.py` — `ACCOUNT`, `PARTITION`,
`GPUS_PER_JOB`, `MAX_BLOCKS`, `TAG`, `TIMEOUT_S` — then:

```bash
cd $HOME/openQHA && source env_openqha.sh

python -u scripts/production/s0_E_branchA_parsl.py --edges --resource tianhe \
    2>&1 | tee $HOME/openQHA_prod_$(date +%Y%m%d_%H%M).log
```

A flag still wins where you give one, for a one-off. The plan record says **where each
value came from** (`settings_source`), so "why 14400" has an answer later. A campaign
whose parameters live in a shell history is a campaign nobody can reproduce.

Scale with `MAX_BLOCKS` (allocations held at once). `init_blocks=0` and `min_blocks=0`
mean nothing is requested until there is work and an idle allocation is given back —
**a held node is charged whether or not it computes.**

### Watching it

```bash
yhq -a                 # recommended by the manual; works even with no allocation left
yhi                    # node states: idle / mix / alloc
yhcancel <jobid>
```

### Where the results land

Sharded, because a campaign is 133 885 molecules and one directory each is not an option:

```
data/basins/prod/1_16000/1_4000/dsgdb9nsd_000018.basins.json
data/basins/prod/1_16000/1_4000/dsgdb9nsd_000018.basins.xyz
```

```bash
# how far has it got?
python -c "
from openqha import basin_store
c = basin_store.census(tag='prod')
print(sum(c.values()), 'molecules done across', len(c), 'chunks')"

# one molecule
python -c "
from openqha import basin_store, json
print(basin_store.paths_for('dsgdb9nsd_000018', tag='prod'))"
```

Scratch is node-local (`$TMPDIR`), products are written back into the repository. **Never
let CREST write its working directories on Lustre** — one molecule is 2.6–15 MB across
dozens of small files, and that is slow for you and for everyone else on the machine.
`hpc/env/tianhe.sh` sets this up; a worker that does not source it is a different machine.

---

## 7. What to report, and what not to

From five repeat runs on one molecule (measured 2026-09-05, see
[`branchA_workflow.md`](branchA_workflow.md) §8b):

* the **basin count varies by 14%** run to run — iMTD-GC is a stochastic search;
* the **conformational correction varies by 0.0034 kcal/mol**, 3.4% of the target.

**Report the correction. Report a basin count as one draw, never as the answer.**

Cost: the only figure this repository has is **285 s per species, 4 threads, uncontended,
on a workstation** — not on Tianhe, and not under contention. When you have a real number,
report wall clock, single-job time and slot extrapolation as three separate numbers
(branch E acceptance criterion 5), so nobody later divides one by another.
