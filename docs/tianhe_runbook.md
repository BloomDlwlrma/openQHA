# Running branch A on Tianhe — a runbook

**Nothing in this repository has ever been submitted to Tianhe.** This document is written
from the site facts in [`configs/cluster_tianhe.yaml`](../configs/cluster_tianhe.yaml)
(supplied 2026-08-31) plus everything that could be checked without the machine. Steps are
marked **[verified locally]** or **[unverified]** so you know which is which before you
spend an allocation.

---

## 0. What was already checked here, and what could not be

Building the Tianhe configuration on a workstation found **three real defects** in code
that had been written and never executed. That is the argument for doing steps 1–4 before
step 5 rather than after.

| checked without Tianhe | result |
|---|---|
| the Parsl config constructs | **fixed** — `_command_map` was built *after* `super().__init__()`, and parsl probes the scheduler *inside* it. Every construction raised `AttributeError`. |
| the job script renders | **fixed** — `render_only()` read `self.template_string`; in parsl 2026.08.10 that is a *module* attribute. The check meant to precede every first submission had never once run. |
| the rendered directives | **fixed** — `#SBATCH --ntasks-per-node=1` appeared twice; parsl emits it already. |
| a failing status query is loud | **works** — raises with the command and what to edit. |
| env scripts parse | **works** |

**What cannot be checked from here:** the scheduler command names, the CPU partition's
name, cores per node, whether the proxy line is current, and any real cost. Those are
steps 1–3.

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

## 2. Answer the three open questions  **[unverified — this is the point of the step]**

```bash
# a) the scheduler command names. THREE OF FIVE ARE ASSUMPTIONS.
type -a yhbatch yhrun sacct squeue scancel yhacct yhqueue yhcancel yhinfo 2>&1

# b) the CPU partition's name -- the site information records only GPU partitions
sinfo -s 2>/dev/null || yhinfo -s 2>/dev/null || scontrol show partition 2>/dev/null

# c) cores per node on that partition
sinfo -o "%P %c %m %D" 2>/dev/null
```

Write the answers into two places:

* `hpc/providers.py` → `TIANHE_COMMANDS`, and add each newly confirmed key to
  `TIANHE_CONFIRMED` so the record stops calling it an assumption;
* `hpc/resource_configs/tianhe_cpu.py` → `PARTITION`, `CORES_PER_NODE`, and set
  `PARTITION_IS_ASSUMED` / `CORES_PER_NODE_IS_ASSUMED` to `False`.

> **Why this matters more than it looks.** A wrong SUBMIT name fails loudly. A wrong
> STATUS name did not: Parsl read the empty output, matched no job, and left everything
> `PENDING` for ever while the queue quietly stopped. Since 2026-09-05 that raises
> instead — but only *after* the provider is built, so it cannot tell you the name is
> wrong before you submit. This step can.

---

## 3. Install  **[unverified]**

```bash
git clone <this repo> openQHA && cd openQHA     # or rsync it in
bash install_dependency.sh --tianhe
```

`--tianhe` builds `openqha-cuda` from `environment-cuda.yml` and **deliberately does not
download the MACE-OFF weights** — a 100 MB pull through the proxy from a login node is
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
m = load("tianhe_cpu")

import json
print(json.dumps(m.describe(), indent=1))          # what is measured, what is assumed

p = m.config(partition="<YOUR PARTITION>", account="<YOUR ACCOUNT>").executors[0].provider
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
    --resource tianhe_cpu --partition <YOUR PARTITION> --account <YOUR ACCOUNT> \
    --tag tianhe_smoke_parsl --threads 4
```

**Replacing Parsl with a for-loop must not change a single number.** Compare the two
products; if they differ, the execution layer is doing something it should not.

---

## 6. Production  **[unverified]**

```bash
cd $HOME/openQHA && source env_openqha.sh

python -u scripts/production/s0_E_branchA_parsl.py \
    --edges \
    --resource tianhe_cpu \
    --partition <YOUR PARTITION> --account <YOUR ACCOUNT> \
    --tag prod --threads 4 --timeout-s 14400 \
    2>&1 | tee $HOME/openQHA_prod_$(date +%Y%m%d_%H%M).log
```

Scale up by editing `max_blocks` (nodes requested at once) in `tianhe_cpu.py`, or pass
`--max-workers` to override the assumed workers-per-node. `init_blocks=0` and
`min_blocks=0` mean nothing is requested until there is work and an idle allocation is
given back — **a held node is charged whether or not it computes.**

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
