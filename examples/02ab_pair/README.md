# 02ab — branch A for acetone and propanal in one CPU job

```bash
bash examples/run_chain.sh examples/02ab_pair/branchA-pair.conf deimos
```

One node, two CREST pipelines side by side, 4 threads each: `CHAIN=conformers_pair`.
The products land under the canonical tags — `data/basins/acetone/…` and
`data/basins/propanal/…` — so they are the inputs 02a's and 02b's GPU chains consume.

## Why one job

TianheXY-CN (`deimos`) allocates by node and requires `--exclusive`: a job holds all 64
cores whether it uses them or not, and a single-molecule branch A used about 1.8% of the
node it held (job 7347197). Two molecules in one allocation is 8 cores of real work on
one node instead of 8 cores across two. There is no smaller request on that cluster; the
only way to waste less is to put more work in the node.

4 threads per molecule is measured, not chosen: MACE on a ten-atom molecule runs
111/90/72/101 ms at 1/2/4/8 threads — eight is slower than four — while CREST's own
`threads` parallelises independent metadynamics runs. See the conf's header for the
socket-collision fix this pair also exercises.

## What the conf carries

| line | meaning |
|---|---|
| `PAIR_1="dsgdb9nsd_000018 acetone 4 --max-conformers 100"` | species, tag, threads, extra args for **this molecule only** |
| `PAIR_2="dsgdb9nsd_000035 propanal 4"` | no ceiling for propanal — its conformer count is not verified |
| `KEEP_CALCSPACE`, `MACE_TRACE` | diagnostics, applied to both |

The `SPECIES`/`TAG`/`THREADS` lines at the bottom only name the run for `run_chain.sh`'s
banner; the pair lines are what runs.

## After it finishes

Both products are on `/XYFS02/…`, which TianheXY-A shares, so the GPU step needs no copy:

```bash
source /APP/u22/ai_x86/toolshs/set-XY-I.sh
bash examples/run_chain.sh examples/02a_qha_openmm_acetone/chain.conf ai
bash examples/run_chain.sh examples/02b_qha_openmm_propanal/chain.conf ai
```

Each is one card (`--gpus=1 --cpus-per-task=12`), driver and trajectories in the same job.
The submission method as a whole: [`../README.md`](../README.md), and
[`../../docs/tianhe_runbook.md`](../../docs/tianhe_runbook.md) §0b for the measurements.
