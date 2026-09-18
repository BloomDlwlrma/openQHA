# 33: The CCSD(T) level in `level_compare`: thermochemistry tiers only

**What to build:** once ticket 32's level records exist, `level_compare` treats `dlpno-ccsdt_cc-pvtz` as a second reference: `[[Level]]` gets its row from the level's own `thermo_msrrho.toml` (CCSD(T) geometries, CCSD(T) numerical Hessians, CCSD(T) energies -- one surface, no composite), and `[Tiers]` gains, **for the thermochemistry only** (user ruling 2026-09-17: model error is Hessian vs Hessian at each level; wB97M / CCSD(T) / experiment differences are level errors of the thermochemical quantities): `MODEL_ERROR_S_CCSDT` (MACE S_abs minus the CCSD(T) level's), `LEVEL_ERROR_S_WB97M_VS_CCSDT` (wB97M S_abs minus CCSD(T)'s), `LEVEL_ERROR_S_CCSDT` (CCSD(T) minus experiment where declared) and the same three for S_REF, S_CONF_PRIME, DS_BAR, H_CONF, CP_CONF and G_REL; the Report prints a 4 x 3 table (levels x tiers) and states which reference each model-error column is against. The `[Training_Set]` block and sentence are unchanged. For a molecule with the wB97M level but no CCSD(T) level the CCSD(T) column is absent and stated; no composite level (CCSD(T) energies on wB97M Hessians) is built here -- if it is wanted later it is its own ticket with its own `composite_notation`.

**Blocked by:** 32 (the level records).

**Status:** ready-for-agent

- [ ] `level_compare` on a molecule with three present levels (MACE, wB97M, CCSD(T)) writes the second-reference tiers for every thermochemical term and the Report's 4 x 3 table; on the propanal fixture (no CCSD(T) level) the column is absent and stated
- [ ] every tier is engine-minus-reference or reference-minus-experiment with the sign convention of ticket 26 and the identity MODEL_ERROR_S_* = S_REF + S_CONF_PRIME + DS_BAR terms holds per reference to 1e-9
- [ ] no Hessian metric appears in `level_compare`: the CCSD(T)-vs-wB97M Hessian difference lives only in `hessian_compare` (as REF_NOISE_FLOOR and the two hessian_compare records, one per reference level)
- [ ] OPEN until ticket 32's Batch returns: the three rings' tables in the note, with S_abs at the three levels and the experiment where declared
