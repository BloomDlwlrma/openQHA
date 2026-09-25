# msRRHO free energy from the basins, against the level MACE was trained to

Status: ready for ticket approval
Date: 2026-09-16 (grilled 2026-09-15/16, three rounds; record in `grilling-rounds.md`)
Vocabulary: CONTEXT.md (Engine, Engine folder, Record, Report, Property file, Calculation,
Batch, Basin, Level, Level folder, Enantiomer degeneracy). Decisions: docs/adr/0004 (new),
`docs/adr/0007` (the 2026-09-25 imaginary-mode ruling),
`S0-B-54/55`, `S0-B-60`. Facts this spec was written from:
`.mem/notes/notes_2026-09-15_msrrho-crest-source.md`,
`.mem/notes/notes_2026-09-15_2_chemsci-msrrho-workflow.md`, and the propanal mirror test
in `grilling-rounds.md` (gauche-propanal is an enantiomer pair, g' = 2).

> **Ruled 2026-09-25 (ticket set 34-40; `docs/adr/0007-one-imaginary-mode-policy-floor-and-the-two-principles.md`).**
> The imaginary-mode regime became: one production policy `invert_below` on the frequency
> floor `ithr = -50 cm^-1` (CREST's `-ithr`); `refuse` and `[Imaginary_Spread]` removed; a
> mode with |omega| < 1 cm^-1 dropped from the thermochemistry sums with the value recorded,
> three or more raising; the census floor with the **inversion window** (a basin whose
> lowest mode lies in [ithr, 0)); the tighten certified against ORCA's default line
> (`TolMaxG = 1.543e-2 eV/A`) with one bounded second optimisation pass; the reference-level
> soft-saddle retry. Every stale sentence below is replaced or annotated with a dated note;
> ticket 34's text is left as written.

## Problem Statement

Branch A leaves every basin with a MACE Hessian, a sigma and a harmonic `G - E_el`, but
no Calculation assembles the molecule's absolute entropy and free energy the way the
published entropy protocol does, and nothing compares any of it to the level MACE-OFF23
was trained to or to an experiment. The one route meant to supply the low-frequency
entropy (Cartesian QHA) is off by 2.87 kcal/mol on acetone; the one convention the
package implements for low modes (Grimme 2012) is not CREST's; and branch A refines one
representative per CREST conformer, so a geometric enantiomer pair (gauche-propanal) is
already one basin and its `R ln 2` is silently missing.

## Solution

Four Calculations per molecule and one seam per implementation:

1. **`thermo_msrrho`** at MACE: from `branchA.toml`'s basins and each basin's
   `hessian.npy`, per basin the Eckart-projected spectrum, `S_msRRHO,i` with CREST's
   conventions, `G_i = E_el + ZPE + [H(T)-H(0)] + G_rot(sigma_i) + G_trans - T S_i`;
   then populations from `G_i` with `g'_i`, `S'_conf` (Gibbs-Shannon), `dS_bar`,
   `H_conf`, `Cp_conf`, `S_abs = S_ref + S'_conf + dS_bar`, `G_total`. One level, so
   the reference subtraction is an identity and the only approximation is the Hessian.
2. **`degeneracy`**: `g'_i` per basin, obtained by running CREST's `intraconfRMSD` on
   `crest_rotamers.xyz` grouped by `cre_members` (core-atom RMSD clustering with the
   rotor-group atoms excluded, threshold 0.125 A; mirror check at 0.094 A), mapped onto
   branch A's basins. Must equal `cre_degen2` from a GFN2 `--entropy` run.
3. **`thermo_msrrho` at the reference level** `wb97m-d3bj_def2-tzvppd`: every basin
   re-optimised at that level and its Hessian computed (ORCA 6.1.1 on deimos;
   `TightOpt Freq`, NumFreq if the analytic Hessian is unsupported for this functional,
   the finding recorded in the README); basins re-deduplicated with branch A's rule; the
   level's own `S_abs` and `G_total` by the same assembly; the merge map from MACE
   basins to reference basins with RMSD before and after.
4. **`level_compare`**: reads the level folder and prints the three tiers: model error
   (MACE vs reference), level error (reference vs experiment, LBH values declared with
   citation keys), total (MACE vs experiment); GFN2 as a fourth level when present;
   absent levels stated.

Seams: `crest --thermo` on acetone's frequencies for the per-basin term; `crest
--entropy` at GFN2 on propanal for the assembly, term by term, and for `g'`.
Everything a level produces lives in `levels/<level>/`; engine files stay in engine
folders. Branch B's trajectory entropy is neither replaced nor spliced.

## User Stories

1. As a user, I want one Calculation that turns a molecule's basins into `S_abs` and
   `G_total` as Pracht & Grimme define them, so that the answer has a published form.
2. As a user, I want every basin's msRRHO entropy computed with CREST's conventions
   (tau 25, rotor cap = mean principal moment, entropy and Cp interpolated, ZPE and
   enthalpy harmonic, `fscal = 1.0` declared), so that our number and `crest --thermo`
   agree on the same frequencies.
3. As a user, I want `grimme2012` (tau 100, B_av = 1e-44) and `xtb` (tau 50, ithr -20)
   as named presets, so that the spread between published parameterisations is one
   error-bar line in the Report.
4. As a user, I want the imaginary-mode policy named in the Property file, so that a
   basin's exclusion or inversion has a stated provenance and is never silently fixed.
   *(Ruled 2026-09-25: `invert_below` with ithr = -50 cm^-1 -- a mode in [ithr, 0) is
   inverted and recorded as an inversion window; only below the floor is the basin
   excluded and listed; tickets 34/35/37, ADR 0007.)*
5. As a user, I want `g'_i` obtained by CREST's own algorithm from CREST's own rotamer
   file, so that gauche-propanal gets `g' = 2` for the same reason CREST gives it.
6. As a user, I want `g'` to agree with `cre_degen2` for every conformer of a GFN2
   `--entropy` run, so that the port is checked against the original, not against my
   reading of it.
7. As a user, I want no rotamer factor anywhere in `S'_conf`, so that a methyl's three
   wells are one harmonic mode in one well; `Cp_conf` and `H_conf` may use the full
   `g`, as the paper does, because a constant factor cancels there.
8. As a user, I want each basin's own sigma from branch A in `G_rot,i`.
9. As a user, I want the reference basin to be the lowest `G_i` and `dS_bar` written out
   even though it is an identity at one level, so that a two-level assembly later drops
   into the same file.
10. As a user, I want a one-basin molecule to give `S'_conf = 0` and `dS_bar = 0`.
11. As a user, I want all 35 basins of the four shipped molecules re-optimised and
    Hessian-computed at `wb97m-d3bj_def2-tzvppd` on deimos, so that the reference
    ensemble is complete and no ensemble-truncation error is labelled "model error".
12. As a user, I want a propanal dry run of the reference level locally first, so that
    the ORCA keywords, the Hessian route (analytic or NumFreq) and the parser are proven
    before a deimos Batch.
13. As a user, I want basins that merge or turn into saddles at the reference level
    reported as a merge map with `N_BASINS` per level, so that the geometry shift is a
    number and the topology disagreements are on record (they are candidate label sites
    for Hessian learning).
14. As a user, I want the three tiers printed in one Report with the experimental value
    and its citation key, so that a residual in tier (i) is read against tier (ii).
15. As a user, I want the experimental values and sources declared per molecule in the
    configuration and resolved in `docs/cite/cite_openQHA.bib`, so that the SI table is
    generated and a value without a citation cannot be entered.
16. As a user, I want GFN2's `S_abs` reported as a fourth level when a GFN2 run exists,
    and never used as a reference for MACE.
17. As a user, I want the Report to print CREST's per-mode table (omega, T*S(HO),
    T*S(FR), the weights) up to 300 cm^-1, so that the interpolation is visible.
18. As a user, I want the 90 %-population basin list in `[Result]`, so that a later
    step (the Hessian-learning set) can select on it without recomputing populations.
19. As a user, I want `composite_notation` derived from the level name, so that the
    string in every record spells the level the folders spell.
20. As a user, I want the level folder to have no sub-folder for a level that was not
    computed, so that absence is stated and never a zero.
21. As a user, I want the Property files to hold only what a later step reads and the
    Reports the tables, conventions and provenance, so that ADR 0003 holds.
22. As a user, I want the MACE Hessian evaluated at the reference geometry of every
    kept basin and compared with the reference `.hess` as a matrix, as a spectrum and
    mode by mode, so that the model error in curvature -- the quantity Hessian learning
    must reduce -- is a number per basin and not inferred from S_abs.
23. As a user, I want every thermochemical quantity the assembly produces (per basin:
    ZPE, thermal enthalpy, S_vib, S_rot, G_i, population, relative energy; ensemble:
    S'_conf, dS_bar, H_conf, Cp_conf, S_abs, G_total) compared between the MACE and
    reference levels at each level's own geometry, so that `level_compare` answers the
    whole thermochemistry question and not one number of it.
24. As a user, I want one production imaginary-mode policy, so that production has no
    choice to get wrong. *(Ruled 2026-09-25: `invert_below` on the frequency floor
    ithr = -50 cm^-1; `crest_native` kept for the GFN2 seam only; the per-record spread
    block was removed with `refuse` (ticket 35) and the four-molecule spread table was
    never built; tickets 34/35, ADR 0007.)*
25. As a user, I want the GFN2 seam to reproduce CREST's per-conformer treatment of a
    sub-ithr mode, so that the Hessian tier of the seam closes to the difference between
    two Hessians of the same geometry and nothing else.
26. As a user, I want a conformer's achirality decided by a continuous self-mirror RMSD
    with the threshold on record, checked against the `procrustes` library, so that
    g' cannot flip between two runs on a point-group label at the edge of a tolerance.

## Implementation Decisions

- **Presets.** The thermochemistry module gains `crest` (tau 25, ithr -50 available,
  rotor cap = mean principal moment, entropy and Cp interpolated), `xtb` (tau 50,
  ithr -20), `grimme2012` (tau 100, B_av = 1e-44, entropy only); per-mode HO and
  free-rotor entropies follow `crest thermo.f90::thermodyn` (free rotor sigma = 1;
  mu capped as mu B / (mu + B)). Enthalpy and ZPE never interpolated. Preset name in
  `[Calculation_Info]`.
- **Spectrum.** `hessian.npy` (raw eV/A^2) Eckart-projected and mass-weighted by the
  existing routine; rigid-body modes identified by overlap with the rigid-body subspace.
  *(2026-09-25: a projected mode with |omega| < 1 cm^-1 is dropped from the
  thermochemistry sums and recorded (`N_BELOW_FLOOR` with its value); three or more in
  one spectrum raise -- ORCA's `CutOffFreq` drop with a tripwire; ticket 35, ADR 0007.)*
- **Imaginary-mode policy** *(ruled 2026-09-25)*: `invert_below` with the frequency floor
  `ithr = -50 cm^-1` (CREST's `-ithr`) is the one production policy, the default of every
  Calculation; `crest_native` remains reachable only by the GFN2 seam; `refuse` is
  removed; records written before that date that name `refuse` stay readable as data.
  Tickets 34/35, ADR 0007.
- **Degeneracy.** Port of `intraconfRMSD`: build rotor groups by CREST's topology
  heuristics (equivalent nuclei bonded to one common neighbour with at most one other
  neighbour; rings once), exclude their atoms, quaternion-Kabsch RMSD on the rest with
  proper rotations only, greedy clustering at 0.125 A -> `g_core`; mirror check
  (x -> -x, 0.094 A) recorded as a flag. Inputs: `crest_rotamers.xyz` and `cre_members`
  from the branch A CREST engine folder; output mapped to basins by the conformer each
  basin came from (`basin.extxyz` header). Both mirror images must have been sampled;
  if a rotamer group has one member the flag says `unsampled`, `g' = 1`.
- **Populations from `G_i`**, not `E_i`; the 90 % mass on those populations, written as
  `N_BASINS_90`.
- **No extrapolation** (paper eq. 15): one branch A search, no iterative series.
- **Reference level.** ORCA input `! wB97M-D3BJ def2-TZVPPD TightOpt Freq TightSCF`,
  `NumFreq` substituted if the analytic Hessian is refused for this functional; the
  substitution and its reason go to the README (and later `docs/tutorials`). Basins
  re-deduplicated with branch A's rule at that level; a MACE basin whose optimisation
  lands on another basin, or on a saddle, is listed in the merge map with its RMSD.
- **Level folder.** `<molecule>/levels/<level>/` holds `thermo_msrrho.out/.toml`,
  `degeneracy.out/.toml` (MACE level only), `merge_map.dat` (reference levels), and
  `level_compare.out/.toml` at `levels/`; engine files (ORCA inputs, outputs, `.hess`,
  xtb `vibspectrum`, `hessian`) stay in `orca/`, `xtb/`, `mace/`. Level names per
  CONTEXT.md; `composite_notation` returns `<reference level> // <engine level>`.
- **Property file blocks** (`thermo_msrrho.toml`): `[Calculation_Status]`;
  `[Calculation_Info]` (molecule, tag, LEVEL, ENGINE, PRESET, TAU, ITHR_POLICY, FSCAL,
  TEMPERATURE, REFERENCE_BASIN); `[[Basin]]` (INDEX, SIGMA, G_PRIME, G_PRIME_SOURCE,
  S_MSRRHO, ZPE, H_THERMAL, G_ROT, G_TRANS, G_I, POPULATION, EXCLUDED); `[Ensemble]`
  (S_CONF_PRIME, DS_BAR, H_CONF, CP_CONF); `[Result]` (S_ABS, G_TOTAL, N_BASINS,
  N_BASINS_90, N_EXCLUDED). `level_compare.toml`: `[[Level]]` (LEVEL, PRESENT, S_ABS,
  G_TOTAL, N_BASINS) and `[Tiers]` (MODEL_ERROR_S, LEVEL_ERROR_S, TOTAL_ERROR_S,
  S_EXPERIMENT, S_EXPERIMENT_SOURCE).
- **Experimental values** declared per molecule: `S_EXPERIMENT`, `S_EXPERIMENT_SOURCE`
  (a BibTeX key in `docs/cite/cite_openQHA.bib`). Propanal 72.75 and ethylene glycol
  72.61 cal/mol/K from the LBH set (Li, Bell, Head-Gordon, JCTC 2016, 12, 2861; values
  from NIST WebBook SRD 69 and Frenkel, TRC 1994); acetone when pinned.
- **GFN2 seam.** A CREST `--entropy` run at GFN2 on propanal (`s0crest`); CREST's
  conformers, `cre_degen2` and per-conformer frequencies are fed to our assembly;
  `S'_conf`, `dS_bar`, `Cp_conf`, `H_conf` must match CREST's printout term by term,
  and our `g'` must match `cre_degen2`. GFN2's `S_abs` then appears as a level.
- **`hessian_compare`** (`levels/hessian_compare.out/.toml`). For every MACE basin
  the merge map marks `kept`: the reference geometry x_r is `$atoms` of `job.hess`; the
  MACE Hessian and residual force are evaluated there (`hessian.hessian(mode="analytic")`)
  and stored as engine files `mace/basinNN/hessian_at_<level>.npy` (ADR 0001). Both
  Hessians are at one point of one space, so no mode assignment is invented; four
  metric families as the MLIP-Hessian literature reports them: (1) element-wise
  Cartesian MAE / RMSE in eV/A^2 (HIP, PFT, Rodriguez, PHL); (2) relative Frobenius
  error of the mass-weighted Eckart-projected `K` (basis-free); (3) sorted-index
  spectrum: eigenvalue / frequency MAE and max over all modes and over modes < 300
  cm^-1, softening slope through the origin (Deng et al.), lowest mode and imaginary
  count of MACE at x_r; (4) mode-resolved on the reference eigenbasis `D = L_r^T K_m
  L_r`: curvature along each DFT mode (the Hessian-vector product PHL trains on),
  `MIXING` from the off-diagonal norm, block-aware eigenvector overlap from
  `mode_match.overlap_matrix` + `degenerate_blocks` (HIP's cosine similarity, made
  safe on degenerate blocks). `mode_match.match` / `hybrid_spectrum` are not used: the
  argmax pairing with counted collisions was built for two bases from different
  estimators (covariance vs Hessian). Curvature in the project's units, still a Hessian
  metric at x_r: per-mode T*S (`crest` preset) on omega_r and on the curvature along
  the reference modes -> `TS_LOW_DELTA`, `S_VIB_CURVATURE_DELTA`, `ZPE_CURVATURE_DELTA`.
  ORCA's own `$vibrational_frequencies` remain the round-trip check of the reference
  side. Thermochemistry proper is NOT compared at x_r: per basin the two levels' own
  `[[Basin]]` rows are set side by side (pairing by the merge map), and the
  `[Ensemble]` / `[Result]` terms likewise. `$dipole_derivatives` are compared only where
  both engines produce them; MACE-OFF23 has no charges, so at that level the record
  states `DIPOLE_DERIVATIVES_PRESENT = false`. `level_compare` grows: `[[Level]]`
  carries S_REF, S_CONF, DS_BAR, H_CONF, CP_CONF, G_TOTAL, N_BASINS, N_BASINS_90 and
  `[Tiers]` a MODEL_ERROR_ line for each of them; the Report prints the per-basin table.
- **Imaginary-mode policies on real data.** *(Ruled 2026-09-25; the four-molecule
  spread table was superseded before it was built.)* `crest_native` reproduces CREST 3.0.2
  (`thermocalc.f90:209`, `thermo.f90:135`): modes in (ithr, 0) inverted, modes below
  ithr kept negative with zero entropy but present in ZPE, H_vib and Cp; it survives
  only for the seam (`crest_entropy`). Production is `invert_below`
  (ithr = -50 cm^-1); `ITHR_POLICY` names the policy in `[Calculation_Info]`; the
  `[Imaginary_Spread]` block (S_ABS, G_TOTAL, N_INVERTED, N_KEPT_NEGATIVE, N_EXCLUDED
  under each policy) was removed with `refuse` (ticket 35). A sub-ithr mode is reported
  per basin with its frequency whatever the policy.
- **Continuous chirality.** `procrustes` (qc-procrustes; Meng et al., Comput. Phys.
  Commun. 2022, 276, 108334, key `meng2022procrustes`) becomes a dependency of the
  `openqha` environment. In `degeneracy.py` the mirror-pair test reports
  `RMSD_ROT = sqrt(rotational(a, b).error / N)` and `RMSD_ORTHO` from `orthogonal`;
  a pair is a mirror pair when RMSD_ORTHO < 0.75 RTHR and RMSD_ROT >= 0.75 RTHR. A
  conformer's achirality is `MIRROR_SELF_RMSD` = min over permutations within its
  RDKit equivalence classes (rotor atoms excluded as in `intraconfRMSD`) of the
  rotational-Procrustes RMSD between the core and its own reflection; achiral iff below
  0.75 RTHR; the threshold and the number are in the record. `symmetry_class` stays in
  the Report as a diagnostic column and decides nothing. The permutation enumeration is
  ours (the library has none); above 10^4 candidates the value is `unresolved` and
  g' falls back to the point-group rule with `G_PRIME_SOURCE = label_fallback`.
  `kabsch_rmsd` must equal the library's rotational RMSD to 1e-8 on every pair it sees.
- **Branch B unchanged**; the ensemble report gains one comparison line.
- **Out of this spec but shaped by it**: the Hessian-learning set reads `N_BASINS_90`
  and the merge map (`.scratch/hessian-learning-set/`).

## Testing Decisions

- A good test reads the Property file a Calculation leaves and asserts on values, or
  feeds a known spectrum / a known rotamer file and asserts the result; never an
  internal dict.
- **`crest --thermo` seam**: acetone's 24 MACE frequencies as a Turbomole `vibspectrum`;
  our `crest` preset equals CREST's T*S_vib to 0.001 kcal/mol (target 2.9272; HO
  2.9297).
- **Degeneracy seams**: propanal `crest_rotamers.xyz` + `cre_members` gives g' = (1, 2,
  1) for (cis, gauche, third); a rotamer group with one member gives `unsampled`;
  `cre_degen2` from the GFN2 `--entropy` run equals our `g'` for every conformer.
- **Algebraic seams**: one-basin molecule -> `S'_conf = dS_bar = 0`; `G_total` from the
  partition function equals the Gibbs-Shannon route to 1e-9 kcal/mol; an enantiomer
  pair given as two basins with g' = 1 equals one basin with g' = 2.
- **GFN2 assembly seam**: term-by-term agreement with CREST's `--entropy` printout to
  1e-4 cal/mol/K on propanal.
- **Experimental seam**: propanal and ethylene glycol `S_abs` at the reference level
  within 1.0 cal/mol/K of LBH (the paper's own MAD is 0.65); the difference is written
  either way.
- **Reference-level seam**: the propanal dry run parses `.hess` and passes
  `verify_hess_frequencies`; the merge map lists every MACE basin exactly once.
- **Hessian-compare seams**: on the propanal fixture (reference `.hess` for basins 0-2,
  a stored `hessian_at_wb97m-d3bj_def2-tzvppd.npy` per basin) the routine that
  diagonalises the reference side reproduces ORCA's `$vibrational_frequencies` to
  0.5 cm^-1; a synthetic pair K_m = K_r gives every error 0, MIXING 0, slope 1, block
  overlap 1, and K_m = 0.81 K_r gives slope 0.9 and curvature-along-reference 0.9
  omega_r; the MACE-at-reference spectrum has 0 imaginary modes or the basin is
  reported with its lowest mode; every kept basin appears once; the low-mode statistics
  are on modes below 300 cm^-1 only; `DIPOLE_DERIVATIVES_PRESENT = false` at the MACE
  level. An integration test evaluates the MACE Hessian at one reference geometry and
  checks the stored `.npy` is what the engine gives today.
- **Imaginary-policy seams**: the synthetic checks of ticket 22 extended with a sub-ithr
  mode under `crest_native` (kept, S = 0, ZPE lowered by |nu|/2); on the GFN2 seam
  fixture `crest_native` reproduces CREST's S_vib for conformer 3 (5.035 cal/mol/K) to
  0.02 and the dS_bar tier of the seam closes below 0.03 cal/mol/K. *(2026-09-25: the
  three-policy-agreement check went with `[Imaginary_Spread]`; the sub-1 rule and the
  window inversion are pinned by `t_msrrho_presets` and `t_census_convergence`; the
  second pass's engine-file append by `t_census_second_pass`; ADR 0007.)*
- **Chirality seams**: the library's own CHFClBr pair reproduces rotational error 26.09
  and orthogonal error 4.4e-8; propanal cis (Cs) gives `MIRROR_SELF_RMSD` below the
  threshold and gauche above; the two `--entropy` runs (c1 / cs label flip) give the
  same g' for conformer 3 and the same `MIRROR_SELF_RMSD` to 1e-3; `kabsch_rmsd` equals
  the library's rotational RMSD to 1e-8.
- **Records seams**: `t_toml_record_roundtrip.py` for the new blocks; an integration test
  asserts the level folder holds exactly the expected stems and no sub-folder for an
  absent level.
- Prior art: `tests/unit/t_report_reads_collect.py`, `t_toml_record_roundtrip.py`,
  `t_mode_match.py`; integration `t_*_engine_folder.py`.

## Out of Scope

- The Hessian-learning set and the judge (own spec after its grilling).
- DLPNO-CCSD(T) single points and the selection rule for which molecules get them.
- SPH (`--bhess`) Hessians at any level.
- Automatic enantiomer detection on basins alone (route ii): rejected; only the port of
  `intraconfRMSD` on CREST's rotamer file (ticket 29 changes the arithmetic of the
  mirror test, not its inputs).
- Comparing thermochemistry at the reference geometry (MACE thermo at DFT minimum):
  only the Hessian is compared there.
- Dipole derivatives at the MACE level (no charges in MACE-OFF23).
- Internal-coordinate / hindered-rotor entropy; any change to branch B.

## Further Notes

- On acetone the `crest` preset moves T*S_vib by -0.002 kcal/mol and `grimme2012` by
  -0.193; the preset spread is the error-bar line of story 3.
- The 23 RI-MP2 conformers keep their status as measurements; no Calculation in this
  spec reads them.
- Tickets: 22 presets + per-basin term; 23 degeneracy port; 24 `thermo_msrrho` at MACE
  + records + experimental comparison; 25 GFN2 seam via `crest --entropy`; 26 reference
  level on deimos + merge map + `level_compare`.
- Round 4 tickets: 27 `hessian_compare` + `level_compare` on every quantity; 28 the
  three imaginary policies on real data (`crest_native`, `[Imaginary_Spread]`) -- closed
  2026-09-25 as superseded: ticket 34's ruling (ticket set 34-40, ADR 0007) made
  `invert_below` the one production policy and removed `refuse` and
  `[Imaginary_Spread]`; 29 continuous chirality with `procrustes`.
- Measured 2026-09-16: CREST's own numerical Hessian gives -68.42 cm^-1 for propanal's
  third `--entropy` conformer, below its ithr; CREST keeps the conformer with S = 0 for
  that mode (S_vib 5.035 vs 8.97 / 8.47), which is the dS_bar residual of the seam.
- Measured 2026-09-17 (tickets 27-29 done): under `crest_native` the GFN2 seam's dS_bar
  closes to -0.002 / -0.022 (conformer 3 kept, S_vib 5.036 vs CREST 5.035). MACE at the
  reference geometry of propanal: Hessian MAE 0.026-0.033 eV/A^2, frequency MAE 2.9-4.7
  cm^-1, softening slope ~1.000; the +0.16 model error in S_abs is S'_conf (+0.158, a
  0.13 kcal/mol relative-energy error of the gauche pair), not curvature (S_REF -0.02).
  Chirality: propanal cis self-mirror 0.0009 A, gauche 0.459; the c1/cs conformer 0.018 /
  0.016 in the two runs (achiral by the number, g' = 1 in both).

- Round 5 tickets: 30 training-set membership of the shipped species (`[Training_Set]`
  in `level_compare`, cached SPICE index); 31 the same over every target QM9 molecule
  (Dataset `data/training_sets/qm9_targets_membership.{dat,toml}`, the SI fraction).
  Story 27: as a user, I want to know whether a molecule's conformers were in
  MACE-OFF23's training data, so that a model-error tier is read as in- or
  out-of-distribution and never mistaken for generalisation.
- Measured 2026-09-17 (ticket 30 done): all four shipped molecules with basins are in
  MACE-OFF23's training data (DES370K monomers, 48-49 conformers each, plus thousands of
  dimer frames); the three ring species are not. Propanal's model-error tiers are
  in-distribution numbers. Index: data/training_sets/mace-off23_spice_index.dat.
- Measured 2026-09-17 (ticket 31 done): 176 of 119,451 QM9 targets (0.15 %) are in
  MACE-OFF23's training data, all at the small end (1-5 heavy atoms: 58 of 172; 9 heavy
  atoms: 23 of 99,098). The shipped 4-heavy-atom molecules are the exception; the bulk of
  the target set is out-of-distribution for MACE-OFF23 by construction.

