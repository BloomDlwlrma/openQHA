# 02c — is MACE-OFF's curvature good enough to do thermochemistry with?

Three levels, the same two molecules, energies and forces through to `G − E_el`.

```bash
# step 1, once, on the CPU cluster: the basins, under THIS example's tag (02c_prod)
bash examples/run_chain.sh examples/02c_hessian_benchmark_levels/branchA.conf deimos

# seconds of compute: MACE + GFN2 -- through Slurm, on the 30-minute CPU queue
bash examples/run_chain.sh examples/02c_hessian_benchmark_levels/quick.conf debug

# hours: MACE + GFN2 + RI-MP2/RIJK/cc-pVTZ, every basin -- the production CPU queue
bash examples/run_chain.sh examples/02c_hessian_benchmark_levels/chain.conf deimos

# on a workstation, no scheduler -- same script the job runs
python examples/02c_hessian_benchmark_levels/s0_level_benchmark.py \
    --species dsgdb9nsd_000018 --tag 02c_prod --levels mace,gfn2,rimp2
bash examples/02c_hessian_benchmark_levels/run_reference.sh      # both molecules, detached
```

The two confs differ in one line, `LEVELS`, and share the tag, so `quick.conf` fills
`analysis/levels/02c_prod/<species>_02c_level_benchmark.json` with the MACE and GFN2
columns and `chain.conf` adds RI-MP2 to the same record. Both are `sbatch` jobs on
TianheXY-CN (`debug`: 30 min; `deimos`: 3 days, `--exclusive` whole node -- the RI-MP2
`TightOpt NumFreq` is 1613 s per 10-atom basin at 4 processes -- `%pal nprocs 4 end`,
the count the cost model was measured at and the one used here). The job is named
`openqha_<species>_levels_02c_prod`, which is also its log file under `logs/`.

**The tag is `02c_prod`**, because that is what `branchA.conf` here sets, and a branch A
product is filed under its tag: `<root>/02c_prod/…/<qid>/mace/basinNN/` (since 2026-09-14). `--tag prod` answers
`no branch A product for dsgdb9nsd_000018 under tag 'prod'` even when `02c_prod` is
sitting right there. Propanal under the same tag needs its own step 1:
`bash examples/run_chain.sh examples/02c_hessian_benchmark_levels/branchA-propanal.conf deimos`
(a separate file: a conf's own `SPECIES=` line overrides anything set in the shell, and
`run_chain.sh` refuses the `SPECIES=... bash ...` form for that reason).

**`CHAIN=levels` runs on CPU partitions only.** ORCA has no GPU path in this repository
and production quantum chemistry runs on deimos; `run_chain.sh` refuses a GPU
partition for it on the login node rather than wasting the allocation in the queue.
A finished `.hess` is reused on a re-run — and only if `verify_hess_frequencies` still
passes on it, so a truncated file fails rather than being trusted for existing. Use
`--refresh` to force the ORCA job.

| level | what it is here |
|---|---|
| `MACE-OFF23_medium` | the production potential — every branch B trajectory and every branch A refinement runs on it |
| `GFN2-xTB` | branch A's CREST workhorse — **the level that picks the geometries** |
| `RI-MP2/RIJK/cc-pVTZ` | the reference (user-specified) |

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

## The number this example exists to produce

Propanal, all three basins, all three levels, **zero imaginary modes anywhere** — so the
thermochemistry comparison is complete. kcal/mol, against RI-MP2:

```
                 dZPE     dE_vib       dT*S    d(G-Eel)
 MACE - RI-MP2
   basin 0    -0.3345    -0.2992    +0.0941     -0.3930
   basin 1    -0.3973    -0.3671    +0.0777     -0.4346
   basin 2    -0.3897    -0.3598    +0.0766     -0.4262
 GFN2 - RI-MP2
   basin 0    -1.6842    -1.5868    +0.2831     -1.8672
   basin 1    -1.6283    -1.5719    +0.1435     -1.6951
   basin 2    -1.6207    -1.5647    +0.1412     -1.6856
```

**MACE-OFF's thermochemistry is 0.39–0.43 kcal/mol below the reference** — inside the
1.0 kcal/mol target. GFN2 is 1.69–1.87 below it, outside.

**And the part that matters more: it cancels.** `Δ(G − E_el)` spans only **0.042
kcal/mol** across the three basins. The error is systematic, so in a *relative*
conformational free energy — which is what branch A and branch B actually deliver — MACE
costs about **0.04** kcal/mol, not 0.4. An absolute `G_total` pays the full 0.4.

The signs are internally consistent end to end: **ν low → ZPE low, S
high, G low**. `dZPE` negative, `dT·S` positive, `d(G−E_el)` negative, on every basin.

```
signed frequency deviation against RI-MP2, propanal (cm^-1)
                   below 500    500 - 1500    above 1500    rms all
   MACE − RI-MP2  -7.5 .. -7.8   -5.8 .. -8.8   -15.6 .. -16.7   ~20-22
   GFN2 − RI-MP2 -13.7 .. -27.4  -24.8 .. -25.4  -88.0 .. -88.7   ~71-75
```

**Softening is now confirmed on four structures** (acetone plus propanal's three basins),
in every band, on every one. Still one *molecule pair*, not the 46-structure criterion 15.

---

## Measured — acetone, all three levels

```
                 level  n_imag    nu_min    nu_max  dx_max/A       ZPE       T*S     G-Eel
     MACE-OFF23_medium       0     79.72   3182.60    0.0000   52.6463    2.9297   35.5092
              GFN2-xTB       1    -61.68   3060.33    0.0184   refused   refused   refused
   RI-MP2/RIJK/cc-pVTZ       1    -35.74   3206.42    0.2230   refused   refused   refused

signed deviation against RI-MP2, by band (cm^-1)
                   below 500    500 - 1500    above 1500    rms all
   MACE − RI-MP2        -0.9          -7.0         -16.9       30.2
   GFN2 − RI-MP2        -5.6         -17.7         -83.2       61.9
```

### MACE is softer than the reference in every band

That is the predicted sign for the systematic softening of universal MLIPs
(Deng et al., *npj Comput. Mater.* 2025): **curvature soft → ν low → S high → G low**.
Acceptance criterion 15 holds in sign, on this molecule.

**It also corrects a reading from the run before the reference existed.** Against GFN2
alone, MACE looked *stiffer* — +4.7 / +10.9 / +74.0 cm⁻¹. Both statements are true and
they are not in conflict: GFN2 is softer still. **Comparing against the wrong reference
reverses the sign of the headline**, which is why the GFN2 column is a third data point
and never the yardstick.

Not yet general: **one molecule, one basin**. The full criterion 15 over 46 structures is
still not done. And the deviation is model error *plus* level difference (see above), so
the sign is trustworthy and the magnitude is not attributable on its own.

### Two levels out of three say MACE's minimum is a saddle

```
level                    lowest mode   n_imag   displacement from MACE's minimum
MACE-OFF23_medium            +79.72        0                 0.0000 Å
GFN2-xTB                     -61.68        1                 0.0184 Å
RI-MP2/RIJK/cc-pVTZ          -35.74        1                 0.2230 Å
```

GFN2 and RI-MP2 independently find a first-order saddle, in the methyl-torsion direction,
where MACE finds a minimum. The GFN2 result survives `--ohess vtight` (the default level
gives −62.82), so neither is a convergence artefact.

**The boundary of that statement matters.** `TightOpt` is a local minimiser and stays in
MACE's well by construction — it will not cross the torsional barrier. So this says *the
stationary point MACE found is a saddle on the other two surfaces*, **not** that acetone
has no minimum at RI-MP2. The proper next step is to displace along the imaginary mode
and re-optimise. **That has not been run**, and nothing here should be read as if it had.

**Propanal is the clean case**: basins 0 and 1 are true minima at RI-MP2 (lowest modes
141.76 and 78.31 cm⁻¹, zero imaginary), so the full three-level thermochemistry comparison
is obtainable there.

The `refused` cells are `thermo.vibrational` declining to compute a free energy on an
imaginary mode, and declining to take its absolute value. **That is the result for that
level, not a gap in the table.**

None of this invalidates branch A: CREST uses GFN2 to *search*, and MACE `refine="opt"`
settles the geometry. It does mean a GFN2 or RI-MP2 thermochemistry column will sometimes
be missing rather than merely inaccurate.

### A production defect this example surfaced

`orca.verify_hess_frequencies` compared our **unprojected** spectrum against ORCA's
**projected** one, aligning by "keep the entries that are not zero". On a structure with
no imaginary mode the two agree to ~0.05 cm⁻¹, which is why it passed on 46 structures
and looked correct. On acetone at RI-MP2, ORCA's six lowest are `[-35.74, 0, 0, 0, 0, 0]`
— the rule keeps the imaginary mode and drops only **five** zeros, the lists shift by one,
and it reported a **48.03 cm⁻¹** disagreement that was entirely its own doing. Eckart-
projected, the same file agrees to **0.058 cm⁻¹**.

Same shape as the bug fixed in `xtb.verify_frequencies` hours earlier: **"the six
smallest" and "the ones that are not zero" both mis-identify rigid modes as soon as an
imaginary mode exists.** `s0_C_committee_calibration.py` calls the same function; its 46
structures had no imaginary modes, so the historical results stand — but that was luck,
not design.

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
