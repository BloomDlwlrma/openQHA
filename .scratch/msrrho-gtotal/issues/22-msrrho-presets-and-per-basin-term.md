# 22: msRRHO presets and the per-basin term

**What to build:** the thermochemistry module computes a basin's msRRHO entropy, heat capacity and free energy from a projected spectrum under a named preset. `crest` (tau 25 cm^-1, free-rotor moment capped by the molecule's mean principal moment, entropy and Cp interpolated, ZPE and enthalpy harmonic, `refuse` as the imaginary-mode policy, `invert_below(ithr)` available only for the CREST seam), `xtb` (tau 50, ithr -20), `grimme2012` (tau 100, B_av = 1e-44 kg m^2, entropy only). The per-mode HO and free-rotor terms follow CREST's `thermodyn` (free rotor with sigma = 1, mu capped as mu B / (mu + B)). `fscal` is 1.0 and declared. Level names follow CONTEXT.md and `composite_notation` is derived from the level name, so the string in every record spells the level the folders spell. A projected mode below 1 cm^-1 is a hard error.

**Blocked by:** None (can start immediately).

**Status:** done 2026-09-16

- [x] acetone's 24 MACE frequencies written as a Turbomole `vibspectrum` and run through `crest --thermo` (environment `s0crest`) give the same T*S_vib as the `crest` preset to 0.001 kcal/mol (target 2.9272; HO 2.9297)
- [x] the three presets on the same spectrum give -0.002 (`crest`), -0.034 (`xtb`) and -0.193 (`grimme2012`) kcal/mol relative to HO, and the preset spread is one line in the returned record
- [x] enthalpy and ZPE are identical across presets (never interpolated)
- [x] a spectrum with a projected mode below 1 cm^-1 raises; a spectrum with an imaginary mode under `refuse` raises with the lowest frequency in the message
- [x] `composite_notation` returns `<reference level> // <engine level>` in the CONTEXT.md spelling (`wb97m-d3bj_def2-tzvppd // mace-off23_medium`) and the one caller that hard-coded a DLPNO string uses the function
- [x] unit tests exist under the repository's test runner and run green
