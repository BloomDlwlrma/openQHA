# msRRHO free energy from the basins, against the level MACE was trained to

Status: ready for ticket approval
Date: 2026-09-16 (grilled 2026-09-15/16, three rounds; record in `grilling-rounds.md`)
Vocabulary: CONTEXT.md (Engine, Engine folder, Record, Report, Property file, Calculation,
Batch, Basin, Level, Level folder, Enantiomer degeneracy). Decisions: docs/adr/0004 (new),
`S0-B-54/55`, `S0-B-60`. Facts this spec was written from:
`.mem/notes/notes_2026-09-15_msrrho-crest-source.md`,
`.mem/notes/notes_2026-09-15_2_chemsci-msrrho-workflow.md`, and the propanal mirror test
in `grilling-rounds.md` (gauche-propanal is an enantiomer pair, g' = 2).

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
4. As a user, I want the imaginary-mode policy `refuse` applied and named in the
   Property file, so that a basin with an imaginary mode is excluded from that level's
   ensemble, listed with its lowest frequency, and never silently fixed.
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

## Implementation Decisions

- **Presets.** The thermochemistry module gains `crest` (tau 25, ithr -50 available,
  rotor cap = mean principal moment, entropy and Cp interpolated), `xtb` (tau 50,
  ithr -20), `grimme2012` (tau 100, B_av = 1e-44, entropy only); per-mode HO and
  free-rotor entropies follow `crest thermo.f90::thermodyn` (free rotor sigma = 1;
  mu capped as mu B / (mu + B)). Enthalpy and ZPE never interpolated. Preset name in
  `[Calculation_Info]`.
- **Spectrum.** `hessian.npy` (raw eV/A^2) Eckart-projected and mass-weighted by the
  existing routine; rigid-body modes identified by overlap with the rigid-body subspace.
  A projected mode below 1 cm^-1 is a hard error (CREST's `vibthr` as an assertion).
- **Imaginary-mode policy** `refuse` everywhere in this spec; `invert_below(ithr)` exists
  in the module for the seam with `crest --thermo` but is not used by any Calculation.
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
  `intraconfRMSD` on CREST's rotamer file.
- Internal-coordinate / hindered-rotor entropy; any change to branch B.

## Further Notes

- On acetone the `crest` preset moves T*S_vib by -0.002 kcal/mol and `grimme2012` by
  -0.193; the preset spread is the error-bar line of story 3.
- The 23 RI-MP2 conformers keep their status as measurements; no Calculation in this
  spec reads them.
- Tickets: 22 presets + per-basin term; 23 degeneracy port; 24 `thermo_msrrho` at MACE
  + records + experimental comparison; 25 GFN2 seam via `crest --entropy`; 26 reference
  level on deimos + merge map + `level_compare`.
