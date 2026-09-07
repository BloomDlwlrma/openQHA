# 02 — acetone through branch B

Two things live here. They answer different questions and should not be confused.

| | what it answers | where it runs | how long |
|---|---|---|---|
| `s0_qha_openmm_demo.py` | *does the chain work?* | your workstation | minutes |
| `tianhe_ai_h100x.slurm` | *what is the conformational free energy of this molecule?* | TianheXY-AI, `h100x` | up to 3 days |

---

## The local demo — a smoke test that says so

```bash
python examples/02_qha_openmm_acetone/s0_qha_openmm_demo.py
```

Acetone through both production routes: the closed-form check, the OpenMM force against
the ASE one, two short trajectories, an independent superposition. The trajectories are
**3 ps, which is a smoke length and is labelled as one** — acceptance criteria 1 and 5 are
refused below 20 ps, because a 0.4 ps run once passed the saturation criterion and the
reason it passed was that it had not begun to rise.

It runs the ASE route in whatever interpreter you start it with, and looks for a second
one with OpenMM, openmm-torch and openmmtools (`S0_OPENMM_PYTHON`, else `qm9fe`). If it
cannot find one the OpenMM half is **skipped with a message**, not quietly omitted.

---

## The full calculation — one molecule, one h100x allocation

```bash
# 1. always this first: 20 ps, 30 minutes, one seed
bash examples/02_qha_openmm_acetone/run_on_tianhe_ai.sh --smoke

# 2. then the real thing
bash examples/02_qha_openmm_acetone/run_on_tianhe_ai.sh
```

One allocation, three steps, in order:

```
1. branch A   CREST iMTD-GC + MACE refine=sp        ->  a basin list
2. branch B   one trajectory per (basin, seed)      ->  frames.npy + meta.json
3. collect    quasi-harmonic analysis               ->  T*S and the correction
```

### The allocation, and why the layout is what it is

An `h100x` allocation is **1 GPU + 14 CPUs + 240 GB** (site manual 1.1). Asking for more
of either is a rejected submission, not a slow job.

| step | uses | why |
|---|---|---|
| branch A | 14 CPUs, **no card** | CREST's workhorse is GFN2-xTB and xtb has no GPU path at all |
| branch B | **14 workers sharing the one card** | a 10-atom molecule cannot fill an 80 GB H100 — the cost is kernel-launch latency and Python, not arithmetic |
| collect | 1 CPU | reading frames and diagonalising a 3N×3N covariance |

One trajectory per allocation would leave 13 CPUs and most of the card idle. That is the
whole point of the layout, and it is the same reasoning that took TianheXY-A from 8
workers per node to 56.

**For a campaign, branch A belongs on TianheXY-C.** It is here because this example is one
molecule end to end, and splitting one molecule across two clusters to save a card-hour
would make the example about scheduling instead of about the method.

### `CUDA_VISIBLE_DEVICES` is *not* set here — and that is the interesting part

On this cluster the allocation **is** one card and Slurm has already set
`CUDA_VISIBLE_DEVICES` to it. Every worker inherits that unchanged.

On TianheXY-A the allocation is the whole node with 8 cards, nothing has narrowed the
world, and Parsl must spread the workers over the cards itself. Doing the TianheXY-A thing
here would compute an absolute device index from `nvidia-smi -L` — which reports the
node's 8 *physical* cards, not the one this job owns — and hand most workers a card the
job does not have.

Two clusters, two answers. `hpc/resource_configs/tianhe_ai.py` and `tianhe_a.py` say the
same thing in code.

### The protocol, and what it costs

From [`configs/branchB_protocol.yaml`](../../configs/branchB_protocol.yaml) — the published
protocol of Rinaldo & Field, *Biophysical Journal* 2003, which is the paper this branch
already cites for its thermostat:

```
520 ps equilibration  +  1500 ps production,  one frame every 0.5 ps  ->  3000 frames
```

**Cost, honestly:** the only number this repository has is 96.1 s/ps for a 10-atom
molecule on **one CPU thread**, uncontended, on a workstation. That makes one trajectory
`(520 + 1500) × 96.1 s ≈ 54 hours`. It is a CPU number quoted for a GPU job because it is
the only measurement that exists — and this repository's one GPU measurement of branch B
(`D0-C-5`) is **3.5× slower than CPU, on a T400**. An H100 is not a T400 and that number
does not transfer, but nothing has replaced it.

So: run `--smoke` first and read `seconds_per_ps_this_run` out of `meta.json`. That is the
number to plan with.

The job is resumable either way. Trajectories flush every 250 frames and stop themselves
at 90% of the walltime, so if three days is not enough, resubmit the same command and it
continues from the frames already on disk.

### Where the answer lands

```
analysis/branchA/<tag>/<species>/basins.json     the basin list and its criteria
$S0_RUNS_ROOT/qha/<tag>/<species>/               basinNN/seedNN/{frames.npy,meta.json}
analysis/qha/<tag>/                              T*S, the spectrum, the criteria
```

The job prints a table at the end — basin, seed, frames, `complete`, `s/ps this run`,
platform. **If `platform` is not `CUDA` on every row, the job ran on the CPU** and the
timings mean something else entirely.

### Before the first run

```bash
bash install_dependency.sh --tianhe-cuda      # builds openqha-gpu, CUDA 12.3
rsync -a data/potentials/ <tianhe>:$HOME/openQHA/data/potentials/
```

The MACE-OFF weights are **not** downloaded on a login node — see
[`docs/branchA_production.md`](../../docs/branchA_production.md) §4 for why and for the
verification step. `run_on_tianhe_ai.sh` checks them on the login node before submitting,
because a missing weight file is the commonest way to waste an allocation and it costs a
second to rule out.
