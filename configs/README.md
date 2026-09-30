# `configs/` — every setting, and who reads it

Ten YAML files, one shell-template directory, one `.mdp` directory. **English only.**

`configs/openqha.yaml` is the entry point and the only file anything loads by name.
Everything else is merged into it through its `include:` list, one level deep. It is a
**MOVE, not a copy**: `openqha/config.py` raises if two files define the same top-level
key, so every key has exactly one home.

```
openqha.yaml ── include: ──┬── conformers.yaml         crest:, package1:
                           ├── edges_testset.yaml      edges:, species:
                           ├── filters.yaml            species_filter:
                           ├── hessian.yaml            package2:
                           ├── branchB_protocol.yaml   branch_b:
                           ├── state_points.yaml       package3_state_points:
                           ├── electronic.yaml         package5:
                           └── cluster_tianhe.yaml     cluster:
```

`baseline.yaml` is **not** in that list — it is read only by tooling, never by the
pipeline.

---

## The one distinction that matters: read, or recorded?

Three files are marked **RECORD ONLY** in their own headers. Nothing in `openqha/` or
`scripts/` reads them. They exist so a decision is written down where the person changing
it will look — not so code can consume them. Editing one changes **nothing** at run time.

| file | lines | top-level key | read by code? |
|---|---:|---|---|
| `openqha.yaml` | 127 | `include`, `meta`, `runtime`, `thermodynamics`, `engine`, `data` | **yes** — the entry point |
| `conformers.yaml` | 565 | `conformers_meta`, `crest`, `package1` | **yes** — `cfg["crest"]`, `cfg["package1"]` |
| `hessian.yaml` | 137 | `package2` | **yes** — `cfg["package2"]` |
| `filters.yaml` | 147 | `species_filter` | **yes** — the F0–F7 gates |
| `edges_testset.yaml` | 120 | `edges_provenance`, `edges`, `species` | **yes** — `config.edges()`, `config.species()` |
| `branchB_protocol.yaml` | 182 | `branch_b` | **yes** — the published trajectory protocol |
| `electronic.yaml` | 39 | `package5` | **RECORD ONLY** |
| `state_points.yaml` | 135 | `package3_state_points` | **RECORD ONLY** |
| `cluster_tianhe.yaml` | 93 | `cluster` | **RECORD ONLY** — the live settings are in `hpc/` |
| `baseline.yaml` | 104 | `baseline`, `superseded`, `acknowledged_losses` | tooling only |

> **`cluster_tianhe.yaml` is the one that catches people.** It records Tianhe site facts —
> partitions, the proxy line, scratch conventions, quirks. **The settings a job actually
> runs under are in `hpc/`** (`hpc/env/tianhe.sh`, `hpc/resource_configs/*.py`). Changing
> the YAML changes a record; changing `hpc/` changes a run.

---

## What each file decides

### `openqha.yaml` — global, and the index
`runtime.runs_root` (everything this repo writes goes under one root, overridable by
`S0_RUNS_ROOT`), `thermodynamics` (298.15 K, 1 bar, and the 1.0 kcal/mol target every
approximation is judged against), `engine` (which potential), `data` (where QM9 lives).

**`engine.name` is a record, not a switch.** No code reads it. The potential is
`DEFAULT_ENGINE` in `openqha/potentials/engine.py`, overridden per run by `S0_ENGINE`.
Those two disagreed from 2026-09-03 to 2026-09-09 — the config said `MACE-OFF23-SC` while
runs used `MACE-OFF23_medium`, so anyone reproducing from the config would have reported
the wrong level. They now agree, and the header says why keeping them in step matters.

### `conformers.yaml` — branch A, the largest file
The CREST protocol: workhorse GFN2-xTB, `shake=2`, `tstep=5.0 fs`, H mass 2 amu — **one
package from Grimme JCTC 2019, 15, 2847, measured together and not separable**. Also
`refine="sp"` (single points on MACE, not optimisation: `opt` costs 14 of 37 basins and
0.4023 kcal/mol), and CREGEN deduplication at RTHR 0.125 Å **and** ETHR 0.05 kcal/mol
**and** BTHR 1%, all three of which must hold.

### `edges_testset.yaml` — this project's campaign, not the method
The 11 QM9 isomerisation edges and the 7 species they name. **σ and g0 are declared, never
derived**: `openqha/symmetry.py` can derive σ, and these declarations are the
independent answers it is checked against — 7/7 agreement, including acetone, which ORCA
gets wrong (C1, σ=1; correct is C2v, σ=2, worth RT ln 2 = 0.411 kcal/mol). Oxetane's
declaration is deliberately **conditional** on ring planarity.

> The `zh:` field (a Chinese common name per species) was **removed 2026-09-09**. It was
> data, not a comment: two calibration scripts read `spec["zh"]` and one wrote it into
> product records. `name` and `smiles` already identify the species. **Records written
> before that date carry a `zh` key and later ones do not**; nothing downstream keys on it.

### `filters.yaml` — the F0–F7 gates
Which molecules are admitted, and the reason each rejection is recorded rather than
silently dropped. F7 is QM9's official 3054-entry "uncharacterized" list, pinned by SHA-256.

### `hessian.yaml` — package 2
Finite-difference displacement, the Eckart projection, the reference level for frequencies.

### `branchB_protocol.yaml` — how long, how often
The published quasi-harmonic trajectory protocol: equilibration and production lengths,
sampling interval, the thermostat.

### `electronic.yaml` / `state_points.yaml` / `cluster_tianhe.yaml` — records
Package 5's ORCA convention (`DLPNO-CCSD(T)/cc-pVTZ // MACE-OFF23_medium`, ORCA 6.1.1 on
deimos); the near-critical gas state points with their **explicitly flagged** weaknesses
(cyclopropanol's Tc/Pc are a JOBACK estimate, not experimental; the extrapolation span is
344–376 K against 130–185 K for the C3H6O species); and the Tianhe site facts.

### `baseline.yaml` — for tooling only
Declares the snapshot (`_backup/baseline_2026-09-04`) that the number-preservation check
compared against, plus the accounting for every difference the previous baseline reported.
**That check was retired on 2026-09-09** once the translation it guarded finished; the
declaration stays because re-anchoring it is the first step in bringing the check back.

---

## Not YAML

| path | what it is |
|---|---|
| `configs/init_template/` | shell templates to copy and edit: `init_conda_tianhe.sh`, `init_orca.sh`, `env_orca611.sh`. **Templates — not sourced by anything** |
| `configs/mdp/` | GROMACS `.mdp` files for the branch A metadynamics and the branch B production run. Read by GROMACS, not by Python; their comments carry measurements, so they are in scope for the English rule |

---

## Rules that hold across all of them

1. **One key, one file.** `openqha/config.py` raises on a duplicate rather than letting
   the last loader win.
2. **Every parameter carries a class**: `physical_constant |
   derived_criterion | literature_value | modifiable_convention | numerical_tolerance |
   resource_budget`. The class says what changing it costs.
3. **Comments hold measurements.** The SHAKE deviation table, the `refine=opt/sp`
   comparison, the deduplication sweep — **those numbers exist nowhere else**. Losing one
   while rewriting a comment deletes an experiment.
4. **A path in a config is a record, not an authority.** `engine.weights` says *which*
   weights, not where; `openqha/potentials/engine.py` resolves `model_root()/<filename>`.
