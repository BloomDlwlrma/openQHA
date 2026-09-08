# 02c — is MACE-OFF's curvature good enough to do thermochemistry with?

Three levels, the same two molecules, energies and forces through to `G − E_el`.

```bash
# seconds: MACE + GFN2
python examples/02c_hessian_benchmark_levels/s0_level_benchmark.py \
    --species dsgdb9nsd_000018 --tag prod --levels mace,gfn2

# hours: adds the reference level. Detached, both molecules.
bash examples/02c_hessian_benchmark_levels/run_reference.sh
```

| level | what it is here |
|---|---|
| `MACE-OFF23_medium` | the production potential — every branch B trajectory and every branch A refinement runs on it |
| `GFN2-xTB` | branch A's CREST workhorse — **the level that picks the geometries** |
| `RI-MP2/RIJK/cc-pVTZ` | the reference (`plan_C` §5.3, user-specified) |

---

## What this example is for

**A scale.** Every other question in this project — whether a quasi-harmonic `ν` may
stand in for a Hessian `ω` ([02d](../02d_qha_frequency_identity/README.md)), whether a
0.41 kcal/mol symmetry term matters, whether a correction earns its cost — is a
comparison against how far MACE's own thermochemistry sits from the reference. Without
that number, 0.3 kcal/mol is either negligible or fatal depending on an assumption
nobody has written down.

---

## Three things that would make the number mean less than it looks

**1. MACE-OFF23 was never trained towards RI-MP2.** Its reference level is
**ωB97M-D3(BJ)/def2-TZVPPD**, computed with PSI4 on a subset of SPICE — that is what the
[MACE-OFF23 data release](../../../source-code/final-workflow-design/hessian-train/SPICE/Discription.txt)
says. So `MACE − RI-MP2` is a **model error plus a level difference**, and this script
cannot separate them. It reports the total. Separating them needs a fourth column at
MACE's own level, which is not run here.

**2. A Hessian is only a Hessian at a stationary point of the same surface.** Each level
is relaxed to its **own** minimum before its Hessian is taken, and the displacement from
MACE's minimum is reported (`dx_max/A`). Taking GFN2's second derivatives at MACE's
geometry would fold the distance between two minima into what reads as a curvature
error, and the two could not be separated afterwards.

**3. The spectra are paired by ascending index, not by eigenvector overlap.** They come
from different geometries, so their eigenvectors are not expressed in a common frame and
an overlap between them would be measuring the displacement. Eigenvector pairing is
[02d](../02d_qha_frequency_identity/README.md)'s business, where both mode sets belong to
one geometry.

---

## Measured 2026-09-08 — acetone, MACE against GFN2

```
                 level  n_imag    nu_min    nu_max  dx_max/A       ZPE       T*S     G-Eel
     MACE-OFF23_medium       0     79.72   3182.60    0.0000   52.6463    2.9297   35.5092
              GFN2-xTB       1    -61.68   3060.33    0.0184   refused   refused   refused

signed deviation, MACE − GFN2, by band (cm^-1)
      below 500      +4.7 (n=3)
      500 - 1500    +10.9 (n=13)
      above 1500    +74.0 (n=7)          rms over all modes 56.4
```

**GFN2 does not agree that acetone's structure is a minimum.** Starting from MACE's
minimum and relaxing at `--ohess vtight`, GFN2 lands 0.018 Å away on a structure carrying
a **−61.68 cm⁻¹** mode. Loosening the optimiser was checked first: at the default level
the same mode is −62.82 cm⁻¹, so this is GFN2's own verdict on the methyl torsion and not
a convergence artefact.

The thermochemistry columns say `refused` because `thermo.vibrational` will not compute a
free energy on an imaginary mode, and will not take its absolute value either. **That is
the result for that level, not a gap in the table.**

It does not invalidate branch A: CREST uses GFN2 to *search*, and MACE `refine="opt"`
settles the geometry afterwards. It does say that a GFN2 column in any thermochemistry
comparison will often be missing rather than wrong.

**MACE is stiffer than GFN2 in every band**, most of all above 1500 cm⁻¹. Whether MACE is
stiffer or softer than the *reference* is the question `plan_C` §8.5 predicts a sign for
(systematic softening: `ν` low → `S` high → `G` low), and it needs the RI-MP2 column.

---

## The RI-MP2 column

`run_reference.sh` runs it detached for both molecules — acetone (1 basin) and propanal
(3 basins). Cost from this repository's own measurement: **1613 s per 10-atom molecule
at 4 processes** (`analysis/branch2_cost_model.json`), so roughly two hours for the four.

The route is
`! RI-MP2 cc-pVTZ cc-pVTZ/C cc-pVTZ/JK RIJK TightOpt NumFreq TightSCF`, and
`assert_route_matches_production()` reads it back out of
`scripts/production/s0_branch2_opt_freq.py` and refuses to run if the two have drifted.
Two copies of a route line is one copy too many, and a drifted copy would silently
benchmark a different level from the production numbers it exists to calibrate.

**`NumFreq`, not `FREQ`** — ORCA has no analytic MP2 Hessian and exits with code 25 if
asked (measured 2026-09-02).

**ORCA's own `THERMOCHEMISTRY` block is read for nothing.** It guesses the symmetry
number, and this repository has measured it guessing acetone wrong — `C1, σ = 1` instead
of `σ = 2`, worth `RT ln 2 = 0.411` kcal/mol. Frequencies come out of ORCA; free energies
are assembled by `openqha.thermochem.thermo` from a σ that branch A declared.

---

## Output

```
analysis/levels/<tag>/<species>_02c_level_benchmark.json
$S0_RUNS_ROOT/level_benchmark/<tag>/<species>/basinNN/{gfn2,rimp2}/
```

Read in this order:

1. **`n_imaginary`** per level. A level that did not find a minimum has no
   thermochemistry, and everything below is empty for it.
2. **`max_displacement_A`** — how far that level's minimum is from MACE's. A large
   displacement means the frequency comparison is between two different structures.
3. **signed deviation by band** — the sign is what carries the softening prediction, so
   it is never made absolute.
4. **`d(G-Eel)`** — the number the rest of the project uses as its scale.
