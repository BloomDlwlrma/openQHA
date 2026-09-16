# 26: Reference level on deimos and level_compare

**What to build:** the `thermo_msrrho` Calculation at the reference level `wb97m-d3bj_def2-tzvppd`: every basin of a molecule re-optimised and its Hessian computed with ORCA 6.1.1 (`! wB97M-D3BJ def2-TZVPPD TightOpt Freq TightSCF`; `NumFreq` substituted if the analytic Hessian is refused for this functional, and the substitution with its reason recorded in the README), the `.hess` parsed and verified by the frequency round-trip, the basins re-deduplicated with branch A's rule at that level, and the same assembly as ticket 24 run on the result. It leaves `levels/wb97m-d3bj_def2-tzvppd/thermo_msrrho.out` and `.toml`, and `merge_map.dat` listing every MACE basin exactly once with the reference basin it landed on (or `saddle`), and the RMSD before and after. A `level_compare` Calculation reads the level folder and leaves `levels/level_compare.out` and `.toml`: one `[[Level]]` per computed level (absent levels stated as PRESENT = false) and `[Tiers]` with model error (MACE vs reference), level error (reference vs experiment) and total, on S_abs and G_total. A propanal dry run on the local ORCA proves keywords, Hessian route and parser before the deimos Batch of all 35 basins of the four shipped molecules.

**Blocked by:** 24 (assembly and record shape).

**Status:** ready-for-agent

- [ ] the propanal dry run leaves a `.hess` that passes `verify_hess_frequencies` and a merge map with three rows; whether the Hessian was analytic or NumFreq is in the README
- [ ] all 35 basins of the four shipped molecules have a reference-level Record on deimos with `provenance.status = production`
- [ ] a MACE basin that merges into another at the reference level appears once in the merge map with the target index and both RMSDs; one that becomes a saddle is marked `saddle` and excluded from the reference ensemble
- [ ] `level_compare.toml` states PRESENT = false for a level with no sub-folder and computes no tier that needs it
- [ ] propanal and ethylene glycol S_abs at the reference level are within 1.0 cal/mol/K of the declared LBH values, and the difference is written to `[Tiers]` either way
- [ ] the Report prints the three tiers for S_abs and G_total with the experimental citation key
