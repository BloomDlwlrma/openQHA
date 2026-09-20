---
status: accepted
date: 2026-09-16
---

# The reference level is wB97M-D3(BJ)/def2-TZVPPD, and each level's results live in the level folder

Every number MACE-OFF23 produces is compared to the level it was trained to,
`wb97m-d3bj_def2-tzvppd` (SPICE, PSI4), so that "model error" means the model and not
the model plus a level difference. RI-MP2/RIJK/cc-pVTZ, which stage 0 had used as the
reference, is dropped; a few molecules will later be tested at DLPNO-CCSD(T) single
points on top of the wB97M Hessians, by a selection rule still to be defined. GFN2-xTB is
kept only to validate openQHA's own msRRHO implementation against CREST on CREST's native
engine, and to be reported as a fourth level; it is never a reference for MACE, because it
is softer than MACE and would read the sign of the error backwards.

Molecular entropies and free energies are assembled the way Pracht & Grimme (Chem. Sci.
2021, 12, 6551) define them, inside openQHA, with CREST's conventions (tau = 25 cm^-1,
rotor cap by the molecule's mean principal moment, entropy and Cp interpolated, ZPE and
enthalpy harmonic, imaginary modes refused). CREST's `--entropy` mode is not driven with
MACE: MACE is not a native CREST engine.

Results of a level are written under the molecule directory's `levels/<level>/`, one
sub-folder per level (amended 2026-09-20, ADR 0001 amendment 3: the folder is now
`msrrho/thermo/`, flat, `<level>.<step>.<ext>` per level and the cross-level Records bare;
`layout.level_file / thermo_file / levels_present` are the only spellings); engine files
stay in the engine folders of ADR 0001. Level names
are lower-case, method first, basis second, joined by `_`, dispersion inside the method
token. The `composite_notation` string is derived from the level name.

## Considered options

- Keep RI-MP2 as the reference: rejected; its difference from MACE is model error plus
  the wB97M-to-RI-MP2 gap, which nothing in the repository can separate.
- Run CREST `--entropy` with MACE through the external-calculator interface: rejected;
  the mode is tested at GFN2 only and the numerical Hessians and rotamer logic would be
  exercised on an engine they were never validated on.
- Level sub-folders inside each engine folder (`orca/<level>/`): rejected by the user in
  favour of one level folder per molecule that holds all levels.
- Replace engine folders by level folders: rejected; ADR 0001 stands.

## Consequences

The 23 RI-MP2 conformers and `analysis/committee_calibration.json` keep their status as
measurements but no longer define a reference; `plan_C`'s route 甲 step 4 is to be
rewritten against wB97M. Not every molecule gets every level; a comparison Calculation
reads the level folder and states which levels are absent. Experimental absolute
entropies (LBH set: propanal 72.75, ethylene glycol 72.61 cal/mol/K) are declared per
molecule with a citation key resolved in `docs/cite/cite_openQHA.bib`, so the three
tiers (model error, level error, total) can be printed in one Report. Spec:
`.scratch/msrrho-gtotal/spec.md`; grilling record beside it.
