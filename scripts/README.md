# `scripts/`

Four categories, and the rule that decides which one a script belongs to. The category
is declared in the script's own module docstring, and
`tests/unit/t_script_taxonomy.py` fails if a declaration and a directory disagree.

| Directory | A script belongs here when… | Count |
|---|---|---:|
| `production/` | its output **enters a deliverable**. Delete it and some edge has no number | 16 |
| `calibration/` | its output is **a number used to make a decision** (cost, error, control). It reaches a checkpoint, not a deliverable | 20 |
| `diagnostics/` | it was written to locate **one numbered defect**. Still runnable, rarely run | 0 — see below |
| `tooling/` | it manages **the repository itself** (identifiers, format conversion, notebook checks, migration) and produces no science | 6 |
| `_superseded/` | retired, kept as evidence, on no live path. See its own README | 26 |

Counts refreshed 2026-09-09 from the directories themselves; the earlier ones had gone
stale as branch B and branch E scripts landed.

**Why the category is declared as well as implied by the directory.** Classification is a
judgement, not a fact (plan_D section 2.1), and a judgement recorded in only one place
cannot be checked. A script that says `CALIBRATION` while sitting in `production/` is a
disagreement — between two people, or between the same person three weeks apart — and the
taxonomy test is what makes it visible instead of letting the directory quietly win.

## `diagnostics/` is empty, and that is not an oversight

The defect probes it was meant to hold went to `scripts/_superseded/closed-defects/`
instead, on the user's ruling `S0-D-2`. That leaves a tension with the definition above,
which says a probe should stay runnable after its defect closes — and plan_D section 2.3
is how it resolves: each probe becomes a named regression test under `tests/regression/`,
carrying its defect number in the file name. Once the test exists, the probe can be
deleted safely, because the reproduction is no longer the script's job.

## Finding the repository root

Every script here locates the repository by walking up until it finds the directory
containing `openqha/__init__.py`. **No script counts directory levels.** `parents[1]` was
correct only at one specific depth, and when 26 scripts were moved into
`scripts/_superseded/<group>/` it silently started pointing at `scripts/_superseded/`
instead of the root — nothing failed at move time or at import time.
`tests/unit/t_repo_bootstrap.py` enforces the replacement, and checks it from every depth
that actually occurs.

## Writing a product

Do not choose a path. Call `openqha.artifacts.write_artifact(...)`, which requires
`category`, `status` and `produced_by`, refuses a `produced_by` that is not a real file,
and decides the destination itself. See `openqha/artifacts.py` for why.

---

## What each script is for

Prefixes are the branch: `s0_A_` conformer search, `s0_B_` quasi-harmonic analysis,
`s0_C_` Hessian training, `s0_E_` the execution layer, `s0_branch2_`/`s0_package*_` the
older package numbering that predates the branch letters.

### `production/` — 16

| Script | What it delivers |
|---|---|
| `s0_A_pipeline.py` | **The branch A driver**: one molecule, QM9 geometry to basin record, end to end |
| `s0_package1_crest_census.py` | The batch version: **one worker process per molecule**, model loaded serially under a lock |
| `s0_package1_collect.py` | Concatenates the per-molecule parquet into the global tables |
| `s0_package1_crest_summarise.py` | The summary — runnable **while the job is still running**, because the processing order is a fixed-seed permutation |
| `s0_mace_engrad.py` | The CREST `method="generic"` client. **Started once per gradient call**, so it imports only `os/socket/sys` |
| `s0_B_qha_trajectory.py` / `_openmm.py` | The two branch B trajectory routes (ASE+Langevin, OpenMM+Nosé–Hoover) |
| `s0_B_qha_analyse.py` | Trajectory → mass-weighted covariance → frequencies → entropy |
| `s0_B_report_ensemble.py`, `s0_B_saturation_benchmark.py` | Ensemble reporting; how long a trajectory must be |
| `s0_branch2_opt_freq.py` | `OPT FREQ` per conformer at RI-MP2/cc-pVTZ; free energy from the partition function |
| `s0_write_orca_inputs.py` | Writes the DLPNO-CCSD(T) inputs to the deimos convention. **Writes only; never calls ORCA** |
| `s0_isomerisation_electronic_energy.py` | ΔE_el across an edge — one of stage 0's delivered quantities |
| `s0_E_worklist.py` | What is left to compute, and how it decided |
| `s0_E_branchA_parsl.py`, `s0_E_branchB_parsl.py`, `s0_E_branchB_collect_parsl.py` | The same work over Parsl. **Must change no number** |

### `calibration/` — 20

Costs, error bars and controls. None of these reaches a deliverable.

| Group | Scripts |
|---|---|
| branch A protocol | `s0_A_dedup_threshold_scan`, `s0_A_refine_compare`, `s0_A_refine_mechanism`, `s0_A_workhorse_edge_measure` |
| symmetry | `s0_A_symmetry_number`, `s0_A_symmetry_backtest` |
| branch B protocol | `s0_B_thermostat_choice`, `s0_B_thermostat_fixcm`, `s0_B_thermostat_openmm`, `s0_B_md_stability`, `s0_B_short_range_wall`, `s0_B_stack_fingerprint` |
| reference levels | `s0_package2_hessian_benchmark`, `s0_package2_highlevel_freq`, `s0_composite_energy_force`, `s0_qm9_native_reference`, `s0_B_engine_thermo_delta` |
| cost models | `s0_branch2_cost_model`, `s0_engine_benchmark` |
| low frequencies | `s0_lowfreq_and_separable_terms` |
| committee | `s0_C_committee_calibration` |

### `tooling/` — 6

`s0_prepare_data` (put QM9 in place) · `s0_analysis_to_readable` (JSON → `.log` + parquet)
· `s0_mem_decide` (allocate a decision identifier under a lock) · `openqha_index_artifacts`
· `openqha_migrate_analysis` · `check_notebook_forward_refs` · `dump_notebook`.

---

## Checkpoint — 2026-09-09, the branch D pass

Everything under `scripts/` is **English** as of this date, and the gate that enforced it
has been retired along with the migration it policed (`tests/README.md` records why).

| What | Result |
|---|---|
| CJK in live code | **0 lines** (was 26 files / 1278 lines) |
| Measured numbers lost | **0** — verified against `_backup/baseline_2026-09-04` before the check was retired |
| Python compiles | all of `scripts/`, `openqha/`, `tests/`, `hpc/` |
| Tests | **11 unit + 2 regression pass** |

**19 scripts changed. Two changed behaviour; the rest changed only text.** Verified by
parsing both revisions and comparing the ASTs with every string constant normalised, so
comments, docstrings and message text are invisible and anything left is real:

| Script | Behavioural change |
|---|---|
| `calibration/s0_composite_energy_force.py` | dropped `zh=spec["zh"]` / `zh=it["zh"]` |
| `calibration/s0_package2_hessian_benchmark.py` | dropped `zh` from `SPECIES[...]`, from a print, from the `("name","zh","smiles","g0")` tuple and from the product record |

The `zh:` field (a Chinese common name per species) was removed from
`configs/edges_testset.yaml` and `configs/state_points.yaml`. It was **data, not a
comment** — these two scripts read it and one wrote it into product records. `name` and
`smiles` already identify the species. **Records written before 2026-09-09 carry a `zh`
key and later ones do not**; nothing downstream keys on it.

Three more scripts show an AST difference that is **not** behavioural — a message was
re-wrapped across a different number of `print()` calls, with the same text:
`s0_branch2_cost_model.py`, `s0_qm9_native_reference.py`, `s0_write_orca_inputs.py`.

### One defect introduced and repaired

`calibration/s0_branch2_cost_model.py` lost its nested `import math` — the replacement was
keyed by line number and line 150 was `import math`, not the comment it was assumed to be.
`math.exp` and `math.log` are used sixteen lines later, so the script would have raised
`NameError` at run time while still compiling. Restored 2026-09-09.

**The signal was there and was explained away**: the verification run reported this file as
"236 lines → 237 lines" and that was dismissed as an added comment. The lesson is not about
imports — *a check that fires and is argued with has not been run.* The sweep that finds
this class of fault is: parse both revisions, and report any **non-call statement** present
in the old and absent in the new. It now reports only the two intended `zh` removals.
