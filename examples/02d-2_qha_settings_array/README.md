# 02d-2 — 02d at many settings, one card each, one job array

```bash
# once, on the CPU cluster: the basins every row reads (tag 02d_prod)
bash examples/run_chain.sh examples/02d_qha_frequency_identity/branchA.conf deimos

# then, on TianheXY-A, in the fine-grained environment
source /APP/u22/ai_x86/toolshs/set-XY-I.sh
bash examples/02d-2_qha_settings_array/submit_array.sh --plan    # look first
bash examples/02d-2_qha_settings_array/submit_array.sh           # submit
```

[`settings.tsv`](settings.tsv) is the experiment: one row per production run of 02d,
each with its own `EQUIL_PS PROD_PS SAMPLE_EVERY SEEDS NU_CUT` — the settings listed in
[`../02d_qha_frequency_identity/chain.conf`](../02d_qha_frequency_identity/chain.conf).
The default grid is length × sampling interval, 3 × 3 = 9 rows, with the protocol's own
row (`p500_s2`) first.

## A 30-minute test: two tiny rows

`array_t30.conf` + `settings_t30.tsv` are the array with two 1 + 5 ps rows, tags
`02d2t30_<NAME>`, basins from `02d_prod`. Two ways to run it, and they test different
things:

```bash
source /APP/u22/ai_x86/toolshs/set-XY-I.sh
# the array path itself (arithmetic, per-card launch), from the login node:
bash examples/02d-2_qha_settings_array/submit_array.sh --plan examples/02d-2_qha_settings_array/array_t30.conf
bash examples/02d-2_qha_settings_array/submit_array.sh        examples/02d-2_qha_settings_array/array_t30.conf
# one row by hand where you can watch it (--plan has already written generated/<NAME>.conf):
bash hpc/tools/gpu_shell.sh 1 02:00:00
conda activate openqha-gpu
bash hpc/tools/test30.sh examples/02d-2_qha_settings_array/generated/t5_s2.conf
```

Delete `analysis/qha/02d2t30_*/` afterwards.

## What one row is

**One card.** A row's branch B is 1 basin × 3 seeds = 3 trajectories; on this partition a
card comes with 12 CPUs and 120 GB and is the smallest thing you can ask for, so a row
is a card whatever its length. Its tag is `02d2_<NAME>`, its trajectories go under that
tag, and its report is `analysis/qha/02d2_<NAME>/<species>_02d_frequency_identity.json`.
All rows read the **same** branch A product (`BASIN_TAG=02d_prod`): one branch A, many
branch Bs, which is what makes the rows comparable.

## How the rows become a job array

`submit_array.sh` does the arithmetic and prints it before submitting:

```
N  rows                              9
T  = ceil(N / GPUS_PER_JOB)          2 array tasks     --array=0-1
G  = ceil(N / T)                     5 cards per task  --gpus=5 --cpus-per-task=60
row = task × G + k                   task 0: rows 0-4 on cards 0-4;  task 1: rows 5-8 on cards 0-3
```

Each array task is one Slurm job holding G cards. Inside it, [`array.slurm`](array.slurm)
starts G copies of `examples/chain_body.sh`, one per card, and waits for all of them.
G is balanced (5 + 4, not 8 + 1) so the last task idles one card, not seven.

**Why rows are packed into cards rather than one array element per row**: this account
has `MaxSubmit=10` (measured 2026-09-11) and every array element counts against it. Nine
one-card elements would leave room for a single other job; a 27-row grid could not be
submitted at all. Two tasks of five cards is the same nine cards for the same time,
billed the same, and it leaves the queue usable.

## The three things that make G rows in one job safe

Each is enforced in `array.slurm`, and each was a real failure mode, not a precaution:

| mechanism | what it prevents |
|---|---|
| `CUDA_VISIBLE_DEVICES=k` | OpenMM and torch in row k see card k only |
| `S0_CARD=k` | tells `tianhe_a.py` to run parsl on one card and **not pin workers itself** — parsl counts cards with `nvidia-smi -L`, which ignores `CUDA_VISIBLE_DEVICES`, and its fallback would put every row's three workers on cards 0, 1, 2 |
| (until 2026-09-14) `S0_SCRATCH_TAG=card<k>`, `S0_KEEP_DIR=…_card<k>` | node-local scratch is keyed on the job id and `chain_body.sh`'s exit trap **`rm -rf`s it** — shared, the first row to finish would delete the other rows' running trajectories |

Each row also gets its own parsl `run_dir` (`parsl_card<k>`), so G drivers starting at
once do not race for `runinfo/NNN`.

## Where things land

```
examples/02d-2_qha_settings_array/generated/<NAME>.conf    the file each row sourced (written at submit time)
examples/02d-2_qha_settings_array/generated/manifest       row order; the only thing the job reads to find its work
logs/openqha_<species>_02d2_array_<arrayjob>_<task>.{out,err}   per array task
logs/02d2_<NAME>_<arrayjob>_<task>.log                     per row
<molecule>/openmm/basinNN/traj_<NAME>.dcd ...              each row's trajectory: same basin folder, the row in the file name
<molecule>/_records/openmm/<NAME>/02d_frequency_identity.json   the answer, per row
   (<molecule> = <root>/<BASIN_TAG>/<range>/<chunk>/<species>; since 2026-09-14, docs/output_inventory.md section 6;
    before: analysis/qha/02d2_<NAME>/... and logs/node_local/<jobid>_card<k>/)
```

To change the experiment, edit `settings.tsv` and resubmit; `generated/` is rewritten. To
run propanal, change `SPECIES` in `array.conf` after making its basins under `02d_prod`
(`bash examples/run_chain.sh examples/02d_qha_frequency_identity/branchA-propanal.conf deimos`
-- a separate file, because a conf's own `SPECIES=` line overrides the shell).

Slurm references: [job arrays](https://slurm.schedmd.com/job_array.html),
[overview](https://slurm.schedmd.com/overview.html); the fine-grained environment's rules
and measurements are in [`../../docs/tianhe_runbook.md`](../../docs/tianhe_runbook.md) §0b.
