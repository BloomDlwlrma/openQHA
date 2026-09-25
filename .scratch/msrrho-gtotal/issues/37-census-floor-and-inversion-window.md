# 37: The census screen uses the frequency floor and records the inversion window

**What to build:** the census screen admits what the frequency floor allows. A candidate whose lowest frequency lies in (-50, 0) cm⁻¹ becomes a Basin and is recorded as an **inversion window** (count plus lowest frequency); only candidates whose lowest frequency lies below the floor are ejected as saddles. The floor comes from the configuration's package2 section, which mirrors the `crest` preset's `ithr_cm`; a unit test asserts the two are equal, so census and thermochemistry can never drift apart. The council of a molecule whose candidates all lie below the floor still raises, with the message listing every condemned candidate (its lowest frequency) and the tighten residual line, in the floor vocabulary. The Basin record gains the lowest frequency and the window count; acceptance criterion 4 is reworded ("no mode below ithr; the window modes are counted; six rigid modes removed; separation unchanged"). The two dead configuration keys (the old reject-imaginary flags, read by no code) are removed; the live floor key replaces them. CONTEXT.md redefines **Basin** via the floor and adds the term **inversion window**.

The decision shape came from the prototype (`prototype_imaginary_mode_state_machine.html`, `censusVerdict`): `min(nu) < ithr ? "saddle" : "basin"`, with counters `{lowest, n_below_ithr, n_in_window}`.

**Blocked by:** 35 (One production policy on the CREST floor; the three-policy spread removed) — the floor classifier lives there

**Status:** done 2026-09-25 (unit suite 62/62; `t_mace_engine_folder` integration pass with the new section D; py_compile + `--help` clean)

- [x] With the crest floor, a −6.84 cm⁻¹ candidate is admitted with one window mode recorded; a −195.79 cm⁻¹ candidate is ejected; a molecule all of whose candidates lie below the floor refuses with the new message (`t_branch_a_crash` D and C). The wired path was exercised by the MACE integration test on a real Hessian.
- [x] A test asserts the configuration floor equals the preset floor; the dead configuration keys are gone (`t_msrrho_presets`; `package2.ithr_cm` added to `configs/hessian.yaml`, the dead `reject_imaginary` / `reject_imaginary_frequencies` removed).
- [x] Acceptance criterion 4 and the census refusal message use the floor vocabulary; the census record carries the new fields (`ithr_cm`; per candidate `lowest_frequency_cm_inv`, `n_below_ithr`, `n_inversion_window`, `verdict`; `[[Basin]]` gains `N_INVERSION_WINDOW`); the existing message test is updated and passes.
- [x] The census verdict itself is testable without an engine (pure helper `census_verdict`, built on ticket 35's `floor_verdict` / `n_below_ithr` / `n_in_window`); the wired path stays covered by the MACE integration test.

Notes: the package-1 batch screen (`hessian_screen`, `s0_package1_crest_census.py`) is deliberately untouched — the ticket 34 decision names only `census_from_frames`, and ticket 38 extends that same function. CONTEXT.md redefines **Basin** via the floor and adds **Inversion window**; the branch-A report prints `ithr_cm` and the window counts.
