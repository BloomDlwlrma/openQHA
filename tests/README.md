# `tests/`

```
python tests/run_tests.py                 # unit only — seconds, no engine
python tests/run_tests.py --all           # every group
python tests/run_tests.py --group regression
```

Each test is a standalone program returning an exit code, and the runner launches each in
its own process. That is deliberate: a real defect once made a script behave
differently when **executed** than when imported, and importing tests into one process
would erase the distinction part of this suite is about. (`regression/t_defect46_shebang.py`
is the test for it. **It is planned, not written** -- it is listed among the
one-off probes worth recovering as permanent tests. This README and `run_tests.py` both
used to speak of it as if it existed; corrected 2026-09-09.)

| Directory | What lives here |
|---|---|
| `unit/` | pure functions and file-level checks; seconds, no engine, no network |
| `integration/` | needs an engine, CREST or ORCA; slow |
| `regression/` | reproduces one numbered defect; the file name carries the number |
| `data/` | inputs and expected outputs for the regression tests, paired in the style of MSTor's `testrun/` and `testo/` |

`_testlib.py` holds the two helpers the repository-hygiene checks share. It is not a test.

**13 tests: 11 unit, 2 regression** (the count of 2026-09-09; `python tests/run_tests.py` prints the current one, 34 unit on 2026-09-16). `integration/` holds the engine-folder chains; `extensions/gromacs/` is not
in `GROUPS` and needs a `gmx` binary.

---

## Every test, by what it protects

### A. The physics is right

| Test | What it pins down |
|---|---|
| `unit/t_qha_harmonic.py` | **Branch B criterion 3.** For a harmonic oscillator `<q^2> = kT/w^2`, so a synthetic harmonic trajectory has a *closed-form* entropy. Checks superposition, mass weighting, rigid-subspace removal, diagonalisation and the entropy sum — **with zero force evaluations** |
| `unit/t_qha_identity_assertion.py` | **Criterion 11.** A bias potential, constraints, or repartitioned H mass do not raise, do not warn, and return a healthy entropy that is *systematically too high*. Feeds one violation at a time and demands a refusal each time |
| `unit/t_mode_match.py` | A synthetic Hessian with a known spectrum and a deliberate four-fold degenerate block. Proves `projected_modes` and `project_and_diagonalise` are the same computation, and that per-mode overlap collapses in a degenerate block while **block** overlap does not — which is why the gate is on the block |
| `unit/t_vdos_chain.py` | Synthetic velocities of known frequency content, judged by the spectral peaks and the sum rules |
| `unit/t_filters_f7.py` | QM9's "uncharacterized" list: exactly **3054** entries with a pinned SHA-256; a two-fragment molecule rejected naming *both* SMILES; a normal molecule passing; and **enabling F7 without an identifier raises** |
| `regression/t_symmetry_planar_determinant.py` | sigma must not collapse on planar or linear molecules |
| `regression/t_mace_translation_invariance.py` | The patched neighbour-list path is exact **and the install is named**. Rewritten because the previous version asserted the wrong thing: the origin-anchored defect lives in the MACE *develop* tree under stage 2, not in the `mace_torch 0.3.16` openQHA imports, and provenance recorded only a version — both trees answer "0.3.x" |

### B. The repository cannot rot

| Test | What it pins down |
|---|---|
| `unit/t_repo_bootstrap.py` | No file reaches the repository root by **counting directory levels**, or via the working directory. `parents[1]` silently started pointing at `scripts/_superseded/` when 26 scripts moved. **This caught a real regression on 2026-09-09**, when a fix for a hard-coded path introduced a level count instead |
| `unit/t_module_entry_points.py` | Every `python -m openqha.<x>` this repository names resolves. The lazy `__getattr__` shim covers the `import` forms but **cannot** cover `-m`: `runpy` asks the finder for a submodule *file* before any package code runs. Branch A died on "No module named openqha.mace_server" while the whole suite passed |
| `unit/t_script_taxonomy.py` | Every script declares its category **and** sits in the matching directory. Two places, because classification is a judgement, and a judgement stored once cannot be checked |

### C. The guards actually fire

| Test | What it pins down |
|---|---|
| `unit/t_artifacts_identity.py` | `write_artifact` can **refuse**. Most cases are cases that *must* raise — a filing helper that accepts everything leaves `analysis/` exactly where it was |
| `unit/t_capabilities.py` | Locally a missing capability **skips**; one declared through `S0_REQUIRE_CAPS` **fails**. Twice, criterion 2 reported "no comparison produced" (GROMACS in a sibling conda environment; `--no-gmx` passed) and went quiet instead of red — nothing distinguished "checked and agreed" from "did not check" |
| `unit/t_crest_settings_reuse.py` | Defect 57: reusing a scratch directory under different settings mixes two sampling conditions and **nothing in the products shows it**. Compares only the keys that change the *result* (thread count excluded). `tstep` was missing until 2026-09-03 — at 5.0 fs, 29 metadynamics runs aborted; at 2.0 fs, none |
| `unit/t_dat_table_sections.py` | The Table (`openqha.store.dat`, 2026-09-16): named sections round-trip with the value fidelity of the single table; every column the schema knows gets its `Type, unit: doc` comment and every unknown one is returned; a section-less file still reads; a sectioned file refuses the single-table reader by naming its sections; a `[name]` opened twice is refused |
| `unit/t_collect_table_columns.py` | Collect's `collect.dat`: the rows the analyse driver builds have exactly the columns `chain_records.COLUMNS` explains, in all three sections, always written; drift (an unexplained column, a missing explained one, in any row) is reported, a section outside `SECTIONS` refused |
| `unit/t_report_reads_collect.py` | The ensemble sums what collect judged: T*S from the `trajectories` section of `collect.dat`, the verdict from `[Criteria]` in `collect.toml`; no Table, or a Table without `[trajectories]`, is refused by name |

### Not in `GROUPS`

`extensions/gromacs/t_qha_gromacs_crosscheck.py` — branch B criterion 2, an independent
implementation. It also records that `gmx anaeig -entropy` **cannot be** that
implementation: measured 2026-09-03 and confirmed against upstream source, it refuses
mass-weighted eigenvalues, uses a formula that expects them anyway, and drops the six
*softest* modes rather than the six rigid ones — a factor of **452** on one case.

---

## The idea running through all of it

> **A criterion nobody has built a failing example for has not been shown to be able to fail.**

That is why so many of these feed deliberate violations and demand a refusal, rather than
checking a happy path. And note how many exist because something **already escaped**:
`t_repo_bootstrap`, `t_module_entry_points`, `t_crest_settings_reuse`, `t_capabilities`,
`t_mace_translation_invariance` — fences built after the horse left. Each docstring names
the incident.

## Where this suite came from

`tests/` held 228 lines (1.6% coverage) while `scripts/` held 10 026. That was never
because checks had not been written — **a great many had been written, and were sitting in
`scripts/` as one-off probes that ran once and were never run again.** The list of those
worth recovering as permanent regression tests is recorded; each has a specific
input and a definite expected output.

## The checks that guard the restructure itself

These are what make branch D's acceptance criteria executable rather than aspirational:

| Test | The criterion it enforces |
|---|---|
| `unit/t_script_taxonomy.py` | every script declares its category, and the declaration matches its directory |
| `unit/t_repo_bootstrap.py` | nothing reaches the repository root by counting directory levels, or by the working directory |
| `unit/t_artifacts_identity.py` | `write_artifact` genuinely refuses an artifact with no identity |

### Retired 2026-09-09 — the migration is finished, so its gates are gone

Four of these were **one-off gates for the branch D migration**, not standing checks.
The migration they policed is complete, so keeping them would mean maintaining a stale
baseline snapshot and a scan that can only ever report "still nothing to do":

| Retired | Why it existed | Why it is gone |
|---|---|---|
| `t_code_is_english.py` | live code must carry no CJK | **The translation is finished**: `ok -- 0 lines`. Its last run cleared 26 files / 1278 lines |
| `t_translation_preserved_numbers.py` | translating must lose no measured number | It verified that translation: **`no number lost`, 0 tokens gone from the tree**. Its baseline is a 2026-09-04 snapshot and goes stale from here |
| `t_superseded_isolated.py` | nothing live imports anything retired | A rule about `_superseded/`, which is itself marked ready to delete |
| `t_legacy_import_forms.py` | every pre-2026-09-07 import form still works | A compatibility shim for one rename, in a fresh process each time |
| `regression/t_defect13_langevin_units.py` | ASE old `Langevin` took temperature in eV, not K | A fixed defect in a route no production driver takes |

To bring the first two back, re-anchor `configs/baseline.yaml` at a fresh snapshot first;
a gate compared against a stale baseline reports noise, not regressions.

Three of these had to be corrected after their first run reported false positives on their
own aperture — a test that scans for a forbidden pattern will find that pattern in its own
source, and in any docstring explaining why the pattern was removed. Each carries a
comment recording what it got wrong, because the next such check will meet the same trap.
