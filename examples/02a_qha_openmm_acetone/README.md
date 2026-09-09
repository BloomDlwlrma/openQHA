# 02a — acetone: the real-molecule debug check

This directory is where openQHA is checked **on a real molecule, end to end, with all its
basins** — not on a synthetic control, not on one basin, not at one setting.

| | what it answers | cost |
|---|---|---|
| `chain.conf` + `examples/run_chain.sh` | **the whole chain on acetone** | hours → days |
| `s0_debug_realmole.py` | do the settings change the answer? | hours |
| `s0_qha_parameter_scan.py --stage estimator` | is the *estimator* accurate? | seconds |
| `s0_qha_openmm_demo.py` | does the chain run at all? | minutes |

---

## The whole chain

```bash
bash examples/run_chain.sh examples/02a_qha_openmm_acetone/chain.conf                    # local
bash examples/run_chain.sh examples/02a_qha_openmm_acetone/chain.conf ai
bash examples/run_chain.sh examples/02a_qha_openmm_acetone/chain.conf h100x
```

**One file, two modes, both production.** There is no smoke mode: a short run is not a
smaller version of the answer, it is a different quantity that looks like one. If you want
to know whether the machinery works, run the estimator stage of the parameter scan — it is
seconds, it is exact, and it cannot be mistaken for a result.

For the ensemble, use [`../02b_qha_openmm_propanal/`](../02b_qha_openmm_propanal/README.md):
acetone has one basin, so its `F_conf` is identically zero.

---

## The settings grid: `debug_realmole`

```bash
python examples/02a_qha_openmm_acetone/s0_debug_realmole.py \
    --conf examples/02a_qha_openmm_acetone/debug_realmole.conf --print-conf
python examples/02a_qha_openmm_acetone/s0_debug_realmole.py \
    --conf examples/02a_qha_openmm_acetone/debug_realmole.conf
```

It runs **branch A → every basin of acetone → branch B per basin → the ensemble free
energy**, over a grid of the four branch B settings that were in dispute, and writes one
row per cell:

```
length_ps  interval_ps  thermostat  atoms  n_frames  frames_per_dof  mean_TS_kcal
spread_TS_kcal  n_basins  F_conf_kcal  distinct_crossings  symmetry_crossings
criterion_1  verdict
```

### Why it exists alongside the estimator scan

The estimator scan draws from the closed-form canonical distribution, so its bias is known
exactly and it settles the sampling question in seconds. It is **silent on the two things
that actually decide these settings**, because neither has a closed form:

1. **Does the real trajectory stay in its basin at that length?** This is the reason the
   1.5 ns protein protocol was withdrawn. Acetone's methyl barrier is ≈0.8 kcal/mol
   against kT = 0.59; it turns freely, and a covariance that contains two wells is not an
   intra-basin entropy.

2. **Do the basins reweight when T·S moves?** The deliverable is not one basin's entropy —
   it is

   ```
   F_conf = −kT ln Σ_i exp(−ΔG_i / kT)          over every basin branch A found
   ```

   **A setting that shifts every basin by the same amount changes no answer at all.** A
   0.3 kcal/mol wobble is alarming in a single basin and irrelevant in a correction if it
   is common to all of them. A single-basin scan cannot see this, and it is the thing
   worth knowing.

### What acetone actually gives you — measured, 2026-09-07

Branch A on `dsgdb9nsd_000018`, run through this check:

```
CREST reports 3 conformers; + 1 reference geometry = 4 frames in -> 1 basins
basin        E / eV      rel/kcal  sigma   pg    nu_min/cm-1   G-Eel/kcal
    0  -5259.499980        0.0000      2   C2v         79.70      35.5090
9/9 acceptance criteria PASS,  wall 251.8 s
```

**Acetone has ONE basin.** That is a property of the molecule (C2v, and its two rotors
are symmetric), and it splits the two questions this check exists for:

| question | acetone | needs |
|---|---|---|
| does the trajectory stay in its basin? | **yes, this is the right molecule** — two freely turning methyl rotors, barrier ≈0.8 kcal/mol against kT = 0.59 | — |
| do the basins reweight when T·S moves? | **cannot be shown**: with one basin, `F_conf ≡ 0` for every row | a multi-conformer species |

The driver says so rather than printing a column of zeros as if it meant something. For
the reweighting question set `SPECIES` in the `.conf` to a species whose branch A record
has `n_basins > 1` — the tutorial's `OCCC(=O)CO` gives 23, `OCCO` gives 10.

### The grid is 150 cells and costs far fewer trajectories

Length is a **truncation**, interval is a **stride**, the atom set is a **mask** — all
three are read off the same trajectory. Only the thermostat needs its own run:

```
5 lengths × 3 intervals × 5 thermostats × 2 atom sets = 150 cells
n_basins × SEEDS × 5 thermostats                      = the trajectories actually run
```

### Configured in a `.conf`, not on the command line

Same convention as `00_QM9_reaction_eng/hkuhpc/REPT-dNN/search/*.conf`:

```bash
export PROD_PS_LIST=(200 500 750 1000 1500)      # a swept axis
export INTERVAL_PS_LIST=(0.5 1.0 2.0)
export THERMOSTAT_LIST=(nhc_20 nhc_100 langevin_1 langevin_5 langevin_50)
export ANALYSIS_ATOMS_LIST=(all heavy)

export SEEDS="${SEEDS:-3}"                       # fixed
export RESULT_LOG="${RESULT_LOG:-analysis/qha/debug_realmole/results.csv}"
```

`debug_realmole.conf` is the grid. There is no smoke variant, deliberately — see above.

`--print-conf` shows what the parser understood **before** anything is spent. That matters:
the first version of the parser required end-of-line right after a value, so every setting
written with a trailing comment — `SEEDS`, `ROUTE`, `PLATFORM`, `ALL_BASINS` — was silently
dropped and took the driver's default instead. A config parser that ignores half the config
is worse than none, because the run still starts.

### Three things in the grid are deliberately there to fail

* **200 ps** is below this branch's own floor for criteria 1 and 5.
* **`nhc_100`** is the coupling that measured 1.15 kcal/mol of error on a harmonic surface
  where the answer is known.
* **`langevin_50`** is the source paper's own friction, which is a *Langevin* friction and
  is not the same quantity as our production `collision_frequency = 50/ps` (a Nosé–Hoover
  **coupling time** of 20 fs).

A grid whose every row passes has not tested anything.

### Read `distinct_crossings` before `mean_TS_kcal`

A non-zero count means the trajectory left its basin and T·S is inflated **however
converged the saturation curve looks**. `symmetry_crossings` counts methyl turns into
indistinguishable copies: the conformer count and σ are unaffected, but the hydrogens
physically moved, so the Cartesian covariance is still contaminated — which is the main
argument for the `heavy` atom set.

---

## The other three

**`s0_qha_openmm_demo.py`** — acetone through both production routes, 3 ps. A smoke length,
labelled as one: criteria 1 and 5 are refused below 20 ps because a 0.4 ps run once passed
the saturation criterion, and the reason it passed was that it had not begun to rise.

**`s0_qha_parameter_scan.py --stage estimator`** — the closed-form control. Measured on
acetone's own MACE Hessian, 8 draws per row: the estimator is unbiased to <0.01 kcal/mol
from 250 to 3000 frames, and **all 24 modes are recovered at 2 ps spacing**. That last is
the direct disproof of the idea that downsampling filters stiff modes out of the covariance
— if it did, T·S would fall below the closed form.

**`examples/run_chain.sh`** — the whole chain, and the only file that submits anything.
It carries the invariant `#SBATCH` directives, takes partition / walltime / `--gpus` as
`yhbatch` flags (a `#SBATCH` line cannot be parameterised), and passes the conf as the
script's **argument** for the job to `source` — not `--export=ALL`. See
[`../../docs/branchB_production.md`](../../docs/branchB_production.md) §6.

---

## Before the first run

```bash
bash install_dependency.sh                      # weights, hash-checked
python -m openqha.potentials.mace_server --socket /tmp/s0_mace.sock &   # if running by hand
```

Note the dotted path. `python -m openqha.mace_server` **no longer works**: the package was
split into subpackages on 2026-09-07.

Every *import* form still resolves: a meta path finder in `openqha/__init__.py` maps each
old name to its new home and returns the **same module object**, so an alias and the real
module never become two copies with separate caches. `python -m` cannot be covered —
`runpy` asks the finder for a submodule file before any of the package's own code runs.

Two tests guard this, and both exist because the first version of each check shared a bug
with the code it was checking:

* `tests/unit/t_module_entry_points.py` — every `python -m openqha.<x>` named anywhere in
  the repository resolves.
* `tests/unit/t_legacy_import_forms.py` (retired 2026-09-09) — each import form in a **fresh interpreter**,
  with that form first. An in-process check cannot catch this class of bug: touching
  `openqha.thermo` as an attribute registers it in `sys.modules`, after which
  `from openqha.thermo import X` cannot fail. That is exactly how the first version of
  the check passed while branch B was broken.
