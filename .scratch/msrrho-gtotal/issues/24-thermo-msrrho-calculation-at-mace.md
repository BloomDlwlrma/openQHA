# 24: thermo_msrrho at MACE, its records, and the experimental comparison

**What to build:** the `thermo_msrrho` Calculation for one molecule and tag at the MACE level: from `branchA.toml`'s `[[Basin]]` blocks, each basin's `hessian.npy` and `basin.extxyz`, and `degeneracy.toml`, it computes per basin the Eckart-projected spectrum, `S_msRRHO,i` under the `crest` preset, `G_i = E_el + ZPE + [H(T)-H(0)] + G_rot(sigma_i) + G_trans - T S_i`; then populations from `G_i` with `g'_i`, `S'_conf`, `dS_bar` against the lowest basin, `H_conf`, `Cp_conf`, `S_abs`, `G_total` and the 90 %-population basin list. Basins refused for an imaginary mode are excluded and listed. No extrapolation. It leaves `levels/mace-off23_medium/thermo_msrrho.out` and `.toml` with the blocks named in the spec; the Report prints provenance, CREST's per-mode table up to 300 cm^-1 per basin, the six-term breakdown, the ensemble terms, the preset spread, and the experimental value with its citation key when declared. `S_EXPERIMENT` and `S_EXPERIMENT_SOURCE` are per-molecule configuration keys resolved in `docs/cite/cite_openQHA.bib`; propanal 72.75 and ethylene glycol 72.61 cal/mol/K are entered with the LBH, NIST WebBook and Frenkel keys.

**Blocked by:** 22 (presets), 23 (degeneracy), records-redesign 17 (`branchA.toml` `[[Basin]]` blocks).

**Status:** done 2026-09-16

- [x] a one-basin molecule (acetone) gives S'_conf = 0 and dS_bar = 0 exactly and S_abs = S_msRRHO of that basin (70.68 cal/mol/K on the stored Hessian)
- [x] G_total from the partition function equals the Gibbs-Shannon route (S_abs, H_conf) to 1e-9 kcal/mol
- [x] an enantiomer pair entered as two basins with g' = 1 gives the same S'_conf as one basin with g' = 2
- [x] propanal's gauche basin carries g' = 2 into the populations, and the Report shows S'_conf with and without it
- [x] a basin with an imaginary mode is EXCLUDED = true, listed with its lowest frequency, and N_EXCLUDED counts it
- [x] the `.toml` starts with `[Calculation_Status]`, holds only the blocks in the spec, and lives under `levels/mace-off23_medium/`; no file is written under `_records/`
- [x] a molecule with `S_EXPERIMENT` but no `S_EXPERIMENT_SOURCE` (or a key absent from the .bib) is refused at configuration time
- [x] the ensemble report prints one comparison line: msRRHO S_abs next to the trajectory route
