# 28: The three imaginary-mode policies on real data, and `crest_native` for the seam

**What to build:** a third policy `crest_native` in `thermo._apply_imaginary_policy` / `msrrho` that reproduces CREST 3.0.2 exactly (`thermocalc.f90:209`, `thermo.f90:135-138`): modes in (ithr, 0) inverted; modes below ithr kept negative, contributing zero entropy but entering ZPE (`0.5 * sum nu`), H_vib and Cp with the negative frequency as CREST evaluates them; the per-mode record says which modes were inverted and which kept. Every `thermo_msrrho` record (MACE, GFN2, reference) gains `[Imaginary_Spread]` with S_ABS, G_TOTAL, N_INVERTED, N_KEPT_NEGATIVE, N_EXCLUDED under each of `refuse`, `invert_below`, `crest_native`; `[Calculation_Info].ITHR_POLICY` names the policy the `[Result]` block used; the production default stays `refuse` until the ruling is taken on the four molecules' spread. The GFN2 seam (`crest_entropy.evaluate_run`) switches to `crest_native` so that conformer 3 (-68.8 cm^-1 in xtb, -68.42 in CREST's own numerical Hessian, both below ithr = -50) is treated as CREST treated it, and the dS_bar tier of the seam closes. A sub-ithr mode is reported per basin with its frequency whatever the policy. README and CONTEXT: the measured CREST behaviour (keep-with-S=0 below ithr; S_vib 5.035 vs 8.97 / 8.47 for propanal's conformers) and the three regimes.

**Blocked by:** 25 (seam), 26 (reference level records).

**Status:** ready-for-agent

- [ ] synthetic (ticket 22's acetone literals plus one mode at -61.68): under `crest_native` the mode is kept, its S contribution is 0, ZPE is lowered by 30.84 cm^-1, and `n_kept_negative = 1`; under `invert_below` the same input still raises; a mode at -35.74 is inverted under both
- [ ] on the GFN2 seam fixture, `crest_native` reproduces CREST's per-conformer S_vib for conformer 3 (5.035 cal/mol/K) to 0.02 and the seam's dS_bar difference falls below 0.03 cal/mol/K for both runs; the algebraic terms stay at 1e-6
- [ ] at the MACE and reference levels of propanal (0 imaginary modes) the three policies give the same S_ABS to 1e-9 and `[Imaginary_Spread]` says so
- [ ] `[Imaginary_Spread]` is written by every `thermo_msrrho` Calculation and round-trips through `t_toml_record_roundtrip.py`
- [ ] the four molecules' MACE-level records carry `[Imaginary_Spread]` (the user's run; the driver re-writes records without recomputing Hessians); the table of S_ABS under the three policies is in the note for the ruling
- [ ] README and CONTEXT state the three regimes and the measured CREST behaviour with the `thermocalc.f90` / `thermo.f90` lines
