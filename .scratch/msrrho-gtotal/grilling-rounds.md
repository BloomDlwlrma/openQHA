# msRRHO Step 1: grilling record (2026-09-15)

Answers as given; the spec (`spec.md` beside this file) is rewritten when the frontier is
empty. Facts fetched along the way are marked (fact).

## Round 1 (no folders needed)

- Q1 tiers: all three every time. (i) MACE vs wB97M-D3(BJ)/def2-TZVPPD = model error;
  (ii) wB97M vs experiment (LBH: propanal 72.75, ethylene glycol 72.61 cal/mol/K) =
  level error; (iii) MACE vs experiment = total. Citations go to the SI: LBH set = Li,
  Bell, Head-Gordon, JCTC 2016, 12, 2861; values from NIST WebBook SRD 69 (Pracht &
  Grimme ref. 79) and Frenkel, Thermodynamics of Organic Compounds in the Gas State, TRC
  1994 (ref. 80); acetone from NIST/Chao 1986, still to pin.
- Q2 at the reference level: re-optimise every basin at wB97M, then Hessian, then the
  level's own thermochemistry (paper step 2). "Hessian at the MACE geometry" and "how
  to correct the MACE Hessian with wB97M" are parked in the Hessian-learning grilling.
- Q3 RI-MP2 removed from Step 1. Final tests on a few molecules at RI-CCSD(T); the
  selection rule is defined later. (fact) ORCA has no CCSD(T) analytic gradient, so a
  CCSD(T) Hessian is not feasible; the feasible form is DLPNO-CCSD(T) single points for
  E_el on top of wB97M Hessians (a composite).
- Q4 GFN2 only for (a) validating our msRRHO implementation against CREST on CREST's own
  engine and (b) reporting GFN2 as a fourth level. Never as a reference for MACE.
- Q5 all 35 basins of the four shipped molecules go to wB97M.
- Q6 `crest` preset primary (tau 25, rotor cap = mean principal moment, Cp interpolated,
  ithr -50 available); `grimme2012` and `xtb` as presets whose spread is one error-bar
  line; fscal = 1.0 for MACE, declared.
- Q7 imaginary-mode policy at the reference level: `refuse` (basin excluded from S_abs
  at that level, listed).
- Q8 sigma per basin from branch A. g' from CREST's own algorithm (see round 2, Q10).
- Q9 deimos for production; local ORCA only for a propanal dry run.
  (fact) propanal: CREST at GFN2 found 3 conformers (cre_members 3 / 6 / 1 rotamers;
  0, +0.73, +1.84 kcal/mol); branch A kept all three on MACE. The gauche conformer's six
  rotamers are 3 methyl rotamers x 2 mirror images (heavy-atom RMSD 0.36 A, 0.000-0.008
  A after x -> -x), so gauche-propanal is a geometric enantiomer pair, g' = 2. Branch A
  refines crest_conformers.xyz (one representative per conformer), so the pair is
  already one basin: without g' = 2, S'_conf is short by R ln 2 = 1.38 cal/mol/K.

## Round 2

- Q10 g' obtained by porting CREST `src/entropy/entropic.f90::intraconfRMSD` (core-atom
  RMSD clustering inside each conformer's rotamer group; mirror test x -> -x).
- Q11 `wB97M-D3BJ def2-TZVPPD TightOpt Freq`; if the analytic Hessian is unsupported
  for this meta-GGA range-separated hybrid, accept NumFreq; the finding is recorded in
  the openQHA README and later in docs/tutorials. Verified first with one propanal job
  locally.
- Q12 results saved under a level folder, one per resolution level (gfn2,
  wb97m-d3bj/def2-tzvppd, dlpno-ccsdt, mace-off). How a level folder composes with the
  engine folders of ADR 0001 is round 3, Q17.
- Q13 (b): re-deduplicate at the reference level with branch A's rule; report N_BASINS
  per level and the merge map. The user asks whether merge/saddle sites are
  Hessian-learning sites: yes, they are the geometries where MACE and wB97M disagree
  about the topology of the surface, i.e. where the curvature differs most; parked in
  the Hessian-learning grilling as a candidate selection rule ("topology disagreement
  points"), with the caveat that they are the hardest points and belong in the set with
  weights, not as the whole set.
- Q14 GFN2 implementation seam compared term by term (S'_conf, dS_bar, Cp_conf, H_conf),
  and our g' must equal CREST's cre_degen2 for every propanal conformer.
- Q15 (a): per-molecule config keys S_EXPERIMENT, S_EXPERIMENT_SOURCE resolved in
  docs/cite/cite_openQHA.bib; the SI table is generated from the config. The BibTeX
  entries are delivered in the next conversation.
- Q16 tickets: 22 presets + per-basin term; 23 g' derivation (kept separate); 24 the
  thermo_msrrho Calculation at MACE + records + LBH comparison; 25 GFN2 implementation
  seam via crest --entropy on propanal; 26 wB97M reference Calculations on deimos +
  level_compare with the three tiers.

## Round 3 (open)

- Q17 level folder vs engine folder composition.
- Q18 level naming convention.

## Round 3 (2026-09-16, closed)

- Q17 two folder kinds only: engine folders (ADR 0001) and one level folder `levels/` per molecule holding every computed level in a sub-folder; not every molecule gets every level (dlpno-ccsdt only for the selected few); absence is stated.
- Q18 level names lower-case, method first, basis second, joined by `_`, dispersion in the method token: `wb97m-d3bj_def2-tzvppd`, `dlpno-ccsdt_cc-pvtz`, `gfn2`, `mace-off23_medium`. `composite_notation` (today `"RI-MP2/cc-pVTZ // MACE-OFF23_medium"`, one caller hard-coding DLPNO) is derived from the level name.
- g' port: route (i) only (CREST rotamer file + cre_members); route (ii) rejected.

Frontier empty. CONTEXT.md +3 terms (Level, Level folder, Enantiomer degeneracy); ADR 0004; spec rewritten.

## Round 4 (2026-09-16, after tickets 25-26; closed)

- Q19 `level_compare` compares S_abs / S_conf / G_total only; the reference `.hess`
  (`$hessian`, `$normal_modes`, `$dipole_derivatives`) is read for the frequency
  round-trip and nothing else. Ruling: a `hessian_compare` Calculation, MACE Hessian
  evaluated **at the reference geometry** only (geometry (b): the pure model error in
  curvature, the quantity plan C must reduce); thermochemistry compared per basin **at
  each level's own geometry**; `level_compare` grows to every thermochem quantity.
  `$dipole_derivatives`: MACE-OFF23 has no charges, so `PRESENT = false` at that level.
- Q20 `invert_below` was implemented and unit-tested on synthetic numbers but never run
  on real data; every Calculation uses `refuse`. Measured today with
  `crest --numhess --gfn2 --sthr 25 --ithr -50` on CREST's own propanal conformers
  (`~/runs/openQHA/dryrun_25_26/conf3_thermo/nh{1,2,3}`): conformer 3 has -68.42 cm^-1
  in CREST's own numerical Hessian (ours -68.8), BELOW ithr, and CREST neither inverted
  nor dropped it -- `thermocalc.f90:209` inverts only modes above ithr; a mode below
  stays negative, gets S = 0 in `thermo.f90:135-138`, and still enters ZPE and H_vib.
  Its S_vib is 5.035 vs 8.969 / 8.471 for conformers 1 / 2; H_vib +224 cal/mol. This is
  the origin of the +0.10 / +0.12 dS_bar residual in the seam. CREST has three regimes
  (invert / keep-with-S=0 / never refuse), we have two. Ruling: test all policies on
  real data first; inclination towards CREST's default for production; the decision is
  taken on the numbers ticket 28 produces.
- Q21 chirality: the port's mirror test is the Kearsley test (reflect, proper-rotation
  SVD with the determinant sign fixed), mathematically the `procrustes` library's
  chirality check (`rotational` vs `orthogonal` error; notebook Chirality_Check reads
  26.09 vs 4e-8 for CHFClBr; **no atom-permutation handling** in the library). The
  tolerance point-group label `symmetry_class` is the same kind of label that flipped
  c1/cs in CREST. Ruling: add `procrustes` (qc-procrustes) as a dependency; replace the
  label by a continuous self-mirror RMSD with the threshold on record; the
  equivalence-class permutation stays ours. Tormat: reference still to come.

Tickets 27 (`hessian_compare`), 28 (imaginary policies on real data, `crest_native`),
29 (continuous chirality with `procrustes`).
