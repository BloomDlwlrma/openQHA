# msRRHO G_total: closing report (2026-09-17, tickets 22-33)

Written at the end of the msrrho-gtotal proposal, while the DLPNO-CCSD(T) along-mode dry run
of ticket 32 is still running. It states what the workflow computes, what code carries it,
how it is tested, what data was collected, what the numbers say, and what stays open. The
per-ticket detail is in `issues/`, the design in `spec.md`, the rulings in `grilling-rounds.md`;
this file is the summary a reader starts from.

## 1. What the workflow computes

For one Molecule directory (one QM9 species, tag `1_16000/1_1000`), at every Level present:

    G_total(T) = E_el(basin 0) + G_msRRHO(basin 0) + G_conf(T)
    S_abs      = S_ref (lowest basin: S_trans + S_rot + S_vib,msRRHO + S_el)
               + S'_conf (Boltzmann mixing over the kept basins, p_i from G_i, g'_i degeneracy)
               + dS_bar  (population-weighted change of S_vib across basins)

with the CREST conventions of Pracht & Grimme 2021 (equations 7, 10, 13, 14): msRRHO
interpolates only S_vib and Cp_vib (tau 25 cm^-1, alpha 4, rotor moment capped at the mean
principal moment), sigma = 1 with the external symmetry entering through g'; H_conf and
Cp_conf are the population-derived terms. The three imaginary-mode regimes of CREST are
reproduced as `imaginary_policy = refuse | invert_below | crest_native` (ithr -50 grimme2012,
-20 crest preset; crest_native keeps a mode below ithr with S = 0 but present in ZPE/H/Cp).
Enantiomer pairs are two basins on branch A; g' is propagated from a continuous chirality
measure (Procrustes rotational vs orthogonal RMSD), not a point-group label.

The comparison chain is three tiers, each a subtraction of two records at their own
geometries: model error = engine (MACE-OFF23_medium) minus reference (wB97M-D3(BJ)/def2-TZVPPD,
the level MACE-OFF23 was trained to); level error = reference minus experiment; total =
engine minus experiment. A fourth record, `hessian_compare`, measures the engine's Hessian
at the reference geometry against the reference Hessian with four metric families (HIP's
element-wise set, projected Frobenius, sorted spectrum + softening slope + low modes < 300
cm^-1, curvature along each reference mode D_ii with degenerate-block overlap). Training-set
membership of the molecule in SPICE / MACE-OFF23 is stated on every level_compare so a model
error is read as in- or out-of-distribution.

## 2. Code (openqha/, all records as Report .out + Property .toml + Table .dat)

| module | role | record |
|---|---|---|
| `thermochem/thermo.py` | msRRHO per basin, presets (crest / grimme2012 / ...), imaginary policies, `preset_spread` | per-basin dict |
| `thermochem/msrrho_ensemble.py` | ensemble over basins: S_ref, S'_conf, dS_bar, H_conf, Cp_conf, G_total; `[Imaginary_Spread]`; report | `levels/<level>/thermo_msrrho.*` |
| `conformer_search/degeneracy.py` | g' (rotamer + enantiomer), Kabsch + Procrustes chirality, `MIRROR_SELF_RMSD`, permutation cap | `[[Basin]]` CHIRALITY columns |
| `thermochem/crest_entropy.py` | seam to `crest --entropy`: same numbers from CREST's own ensemble under `crest_native`; two tolerance classes (algebraic 1e-6, Hessian 1e-2) | `levels/gfn2/crest_entropy.*` |
| `thermochem/reference_level.py` | ORCA Opt+Freq per MACE basin, re-deduplication (merge map, saddles), level record, level_compare with tiers and `[Training_Set]`, `start_from`, noise floor, `opt_grad_rms` | `levels/<level>/{thermo_msrrho,merge_map}.*`, `levels/level_compare.*` |
| `thermochem/hessian_compare.py` | engine Hessian at the reference geometry (reused only through `.meta.toml` position match), four metric families, per-basin and ensemble deltas | `levels/hessian_compare.*` |
| `thermochem/mode_curvature.py` | higher-level curvature along reference modes from energy lines (the dry run of a numerical Hessian), self-checks vs omega_r and D_ii | `levels/<level>/mode_curvature_dryrun.*` |
| `qm_interfaces/orca.py` | `LEVELS` (analytic wB97M, numerical DLPNO-CCSD(T) with hkuhpc blocks), input writer, single point, `.hess` parser, `hessian_route`, `final_rms_gradient`, `n_single_points`; full `.out` always kept | `orca/<level>/basinNN/job.*` |
| `qm_interfaces/orca_jobs.py` | hkuhpc bundle: inputs, worker.sh, run.sbatch, README with (6N)^2 estimate and days at 93 s/point | `<jobs-dir>/<species>_<level>/` |
| `data/training_set.py` | SPICE / MACE-OFF23 index (Apollo parquet, three strictnesses), shipped-species and QM9-target membership | `data/training_sets/*.dat,toml` |

Drivers: `scripts/production/s0_thermo_msrrho.py --step mace | reference | hessian_compare |
compare | mode_curvature | write_jobs` (`--level`, `--policy`, `--start-from`, `--modes`,
`--basins`, `--nprocs`, `--maxcore`, `--jobs-dir`); `scripts/tooling/s0_training_set_membership.py`
(`--build-index`, `--set shipped|qm9-targets`, `--smiles`); decisions via
`scripts/tooling/s0_mem_decide.py`.

Vocabulary in CONTEXT.md: Level, Level folder, Numerical reference level, Imaginary-mode
policy, Enantiomer degeneracy; ADR 0004 (reference level and level folders).

## 3. Tests

Unit (no ORCA, no engine; 39 files, `tests/unit/`): `t_msrrho_presets`, `t_thermo_msrrho_calculation`,
`t_degeneracy_port`, `t_chirality_procrustes`, `t_crest_entropy_seam`, `t_reference_level`,
`t_hessian_compare`, `t_mode_curvature`, `t_orca_jobs`, `t_training_set_membership`,
`t_training_set_qm9_targets`, `t_mode_match`. Integration (`tests/integration/`):
`t_msrrho_crest_thermo` (CREST `--thermo` line by line), `t_hessian_compare_engine`
(MACE Hessian reuse, max |dH| = 0). Fixtures: `tests/data/propanal_molecule/` (three basins,
full ORCA `.out` 268 KB, `.hess`, MACE `hessian_at_<level>.npy` + `.meta.toml`),
`tests/data/spice_tiny/` (synthetic SPICE parquet), `tests/data/procrustes_chirality/`.
Must-pass / must-fail pairs are in each test docstring; the seam test's tolerance classes are
the acceptance of ticket 25.

## 4. Data collected (WSL `~/runs/openQHA/`, records committed where small)

| set | molecules | levels | records |
|---|---|---|---|
| propanal (in-distribution control) | dsgdb9nsd_000035, 3 basins | gfn2 (CREST), mace-off23_medium, wB97M | thermo_msrrho x3, crest_entropy, hessian_compare, level_compare, mode_curvature_dryrun (running) |
| rings (out of SPICE) | 044 2-methyloxirane (1 basin), 046 cyclopropanol (2), 048 oxetane (1) | mace, wB97M, DLPNO-CCSD(T) dry run | thermo_msrrho x2, hessian_compare, level_compare with `[Training_Set]`, mode_curvature_dryrun |
| shipped four (ticket 28 table) | acetone, acetamide, propanal, N-methylformamide | mace `[Imaginary_Spread]` | user's run, open |
| SPICE index | 951,005 train / 50,195 test frames, 17,132 molecules | -- | `data/training_sets/mace-off23_spice_index.dat` (4.8 MB) |
| QM9 targets | 119,451 gated species | -- | `qm9_targets_membership.{dat,toml}`: 176 in training (0.15 %) |
| hkuhpc bundle | oxetane 1 basin | dlpno-ccsdt_cc-pvtz | `jobs/dsgdb9nsd_000048_dlpno-ccsdt_cc-pvtz/` (4,200 points, ~4.5 days at 8 ranks) |

## 5. What the numbers say

Propanal (in SPICE: 49 monomer + 2,060 dimer frames): S_abs 72.355 (MACE) / 72.193 (wB97M) /
72.75 experiment (LBH). Model error +0.162 cal/mol/K, of which S'_conf +0.158 (relative
energies, populations 0.57/0.21/0.21 vs 0.64/0.19/0.17) and S_ref -0.020 (curvature); level
error -0.557. The GFN2 seam under `crest_native` reproduces CREST to dS_bar -0.002 / -0.022
(was +0.10 before the sub-ithr regime was ported). Hessian at the reference geometry: MAE
0.026-0.033 eV/A^2, low-mode frequency MAE 4-8.5 cm^-1, cos v1 >= 0.997.

Rings (zero SPICE frames at any strictness), all with `IN_TRAINING = false`:

| | 2-methyloxirane | cyclopropanol | oxetane |
|---|---|---|---|
| Hessian MAE, eV/A^2 | 0.074 | 0.053 | 0.067 |
| low-mode freq MAE, cm^-1 | 13 | 57 | 66 |
| lowest mode wB97M -> MACE | 203 -> 190 | 292 -> 235 | 16.7 -> 82.6 |
| MODEL_ERROR_S, cal/mol/K | +0.098 (S_ref) | +0.173 (S_ref) | -1.386 (S_ref) |

Out-of-distribution Hessian error is 2-3x the in-distribution one and sits in the low modes;
mode shapes are right, curvature magnitudes are not; the sign of the lesson flips between the
classes (propanal: relative energy; rings: curvature).

Oxetane's puckering mode at DLPNO-CCSD(T)/cc-pVTZ (TightPNO), 7 points along the wB97M mode
at the wB97M geometry, delta_q 0.35 amu^1/2 A, 93 s per point at 8 ranks:

    k      CCSD(T) E-E0 / Eh    wB97M       MACE        RI-MP2
    +-1        -0.0000657      +0.0000492  +0.0001035  -0.0000148
    +-2        +0.0002688      +0.0007254  +0.0009576  +0.0004650
    +-3        +0.0024272      +0.0034954  +0.0040648  +0.0028950

Read with the second-pass results (ticket 32, findings 1-5): the correlated levels see a
double well where wB97M and MACE see a single well -- RI-MP2 optimised from this geometry
(0.003 A away) is planar with an imaginary puckering mode (-79.6 cm^-1) and a well ~3
cm^-1 deep; the far-IR barrier is 15.5 cm^-1 (puckering fundamental 52.92) -- but the
-14.4 cm^-1 of the CCSD(T) line at the wB97M geometry is not the barrier: a curvature
measured at a foreign geometry contains a gradient term (a puckering displacement
lengthens the ring bonds at second order, and a level whose bonds want to be longer
than wB97M's gains energy from it), which on propanal's torsion turns RI-MP2's 136.2
cm^-1 (own minimum; experiment 135.1) into 93 at the wB97M geometry and canonical
CCSD(T) into -65. The harmonic numbers on the puckering mode are meaningless at every
level (wB97M's own finite-difference self-check is 8.4 cm^-1 there, quartic-dominated;
MACE reproduces its D_ii to 1.1). The -1.39 cal/mol/K "model error" of oxetane is the
difference of two harmonic numbers on a mode that is not harmonic: approximation error
of the model (msRRHO) first, model error (MACE's stiff single well, 93 at its own
minimum vs 16.7) second, and the judge of the Hessian-learning step must separate them.

The dry run also refuted the Batch as designed: DLPNO energies carry ~5e-6 Eh of PNO
noise between distinct geometries, so a NumFreq (0.005 bohr displacements) at that level
would be noise on every low mode; canonical CCSD(T) is smooth but 6 min a point; and the
analytic wB97M Hessian itself moves 6 cm^-1 on the softest mode between DefGrid2 and
DefGrid3 (131.3 / 137.2; the energies' finite-difference curvature says 140.8). Ticket
32 is blocked on the round-2 rulings.

## 6. Conventions decided (decisions log)

S0-B-61 (tickets 22-24, propanal 72.355), S0-B-62 (seam tolerance classes), S0-B-63
(three tiers on propanal), S0-B-64 (CREST sub-ithr regime, tickets 27-29), S0-B-65
(crest_native, Procrustes chirality); S0-C-33 (MACE-OFF23's reference is wB97M, not
RI-MP2), S0-C-38 (analytic wB97M Hessian runs on ORCA 6.0.1), S0-C-39 (propanal's model
error is not curvature), S0-C-40 (four shipped molecules are in SPICE), S0-C-41 (QM9
targets are out-of-distribution by construction; rings' error is curvature). Standing
rules: full ORCA `.out` kept everywhere; `.npy` Hessians reused only through a
position-matching `.meta.toml`; every level records `HESSIAN_ROUTE` and, if numerical,
`NOISE_FLOOR_CM`; training-set membership stated on every comparison.

## 7. Open

- Ticket 32: local criteria closed; the Batch (DLPNO Opt + NumFreq) is refuted by the dry
  run and the replacement (canonical CCSD(T) optimisation + along-mode lines for the low
  modes at the CCSD(T) minimum, or RI-MP2 as the wavefunction tier) needs the round-2
  rulings; no bundle is submitted.
- Ticket 33: CCSD(T) tier in level_compare once the Batch returns; experimental S for the
  rings is not in the li2016lbh table (to be sourced or the tier stays "absent").
- Ticket 28 open item: the four shipped molecules' `[Imaginary_Spread]` at the MACE level.
- Oxetane puckering: the 1-D anharmonic treatment is not in this proposal; it is the first
  question of the Hessian-learning grilling (round 2).
