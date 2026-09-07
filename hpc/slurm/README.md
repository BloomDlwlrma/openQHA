# The plain `yhbatch` route

Everything here does what `scripts/production/s0_E_*_parsl.py` does, without Parsl.
Both routes call the same production drivers with the same arguments, so **a product
must not differ between them** — that is branch E acceptance criterion 1, and comparing
the two on one molecule is the cheapest test of it there is.

| file | cluster | partition | what |
|---|---|---|---|
| `branchA_debug.slurm` | TianheXY-C | `debug`, 00:30:00 | one edge. **The gate before production.** |
| `branchA_deimos.slurm` | TianheXY-C | `deimos`, 3-00:00:00 | one shard: 16 molecules × 4 threads on one node |
| `submit_branchA_deimos.sh` | | | submits the 12 shards, and records what it submitted |
| `branchB_traj_tianhe_a.slurm` | TianheXY-A | `ai` / `temp` | 8 trajectories, one per card |
| `submit_branchB_tianhe_a.sh` | | | one job per species, `ai` or `temp` |
| `branchB_collect.slurm` | TianheXY-C | `deimos` | 64 molecules × 1 core, **one node** |

## Use it like this

```bash
# --- branch A ---------------------------------------------------------------------
mkdir -p $HOME/HDD_POOL/runs/openQHA/logs           # BEFORE the first submit. See below.
EDGE=C3H6O1N0_18_35 TAG=tianhe_debug yhbatch hpc/slurm/branchA_debug.slurm
bash hpc/slurm/submit_branchA_deimos.sh 1 16000 prod

# --- branch B ---------------------------------------------------------------------
bash hpc/slurm/submit_branchB_tianhe_a.sh a_debug temp    # 30-minute smoke test
bash hpc/slurm/submit_branchB_tianhe_a.sh prod ai         # production
TAG=prod yhbatch hpc/slurm/branchB_collect.slurm          # back on the CPU cluster
```

## Five things every script here does, and why

1. **`mkdir -p` the log directory before submitting, not inside the job.** Slurm opens
   `--output` *before* the job script runs. A directory the script creates is created too
   late and the job dies with no log saying why. The debug script writes `%x_%j` into the
   submit directory for the same reason; the submitters make the directory themselves.

2. **Unset inherited `SLURM_*` / `PMI_*` variables.** They make a child process misread
   the task layout and try to relaunch itself through the scheduler. CREST forks its own
   parallel workers and is exposed to exactly that.

3. **`xargs -P`, not `yhrun`.** Every task is an independent single-node process, so there
   is nothing for the launcher to lay out — and going through it would re-import the
   variables just unset.

4. **Refuse early.** Each script checks the executables and recomputes the MACE-OFF
   SHA-256 before doing any work. A missing weight file discovered one task at a time
   wastes an allocation; discovered at the top it wastes ten seconds.

5. **Scan when the job STARTS, not when it was submitted.** `branchA_deimos.slurm` builds
   its worklist from what is on disk at that moment (`s0_E_worklist.py`), so a shard that
   waited six hours in the queue does not redo what its neighbours finished.

## `--shard K/N` is a stride, not a slice

`--shard 3/12` takes every 12th entry of the *already filtered* worklist, offset 2. It is
not indices `X..Y`. QM9 indices are not contiguous and the F0–F7 gates drop molecules
unevenly, so splitting the index range into twelve blocks gives one node an hour of work
and another three days. A stride gives twelve balanced ones.

## Why a shard rather than `--array`

Tianhe supports `--array`. It is not used because an array index has to be mapped to
molecules **in advance**, and that map goes stale the moment anything finishes out of
order or a job is resubmitted. A shard re-derives its own list from disk every time.

## The card assignment in `branchB_traj_tianhe_a.slurm`

One trajectory per card, `CUDA_VISIBLE_DEVICES` exported inside each background subshell,
never once for the whole script. Setting it once is the version that was written first
for the Parsl config, and it puts all eight workers on card 0: the job runs, the numbers
are correct, and it is **8× slower** — which reads as "the GPU is slow" rather than as a
placement bug. The table the script prints at the end, and the `platform` field in each
`meta.json`, are what make it visible.
