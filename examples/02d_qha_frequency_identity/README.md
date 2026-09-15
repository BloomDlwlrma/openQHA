# 02d — can a quasi-harmonic `ν` stand in for a Hessian `ω`?

If it can, then

```
G_total = E_el + G_trans + G_rot + Σ_i [ ZPE_i(ω_i) + H_i(ω_i,T) − T·S_i ]
```

can be built from **one** spectrum instead of two, and the hybrid splice — Hessian for
the stiff modes, QHA for the soft ones — is unnecessary machinery.

```bash
# step 1, once, on the CPU cluster: the basins, under THIS example's tag (02d_prod)
bash examples/run_chain.sh examples/02d_qha_frequency_identity/branchA.conf deimos

# seconds, no MD at all: the harmonic-limit control. --tag is the one branchA.conf wrote.
python examples/02d_qha_frequency_identity/s0_frequency_identity.py \
    --species dsgdb9nsd_000018 --tag 02d_prod --stage harmonic

# PRODUCTION: 500 ps, one frame per 2 fs, every basin, three seeds -- one card, one job
source /APP/u22/ai_x86/toolshs/set-XY-I.sh                 # once per shell, TianheXY-A
bash examples/run_chain.sh examples/02d_qha_frequency_identity/chain.conf ai
bash examples/run_chain.sh examples/02d_qha_frequency_identity/chain.conf   # or here

# a trajectory you already have
python examples/02d_qha_frequency_identity/s0_frequency_identity.py \
    --species dsgdb9nsd_000018 --tag 02d_prod --stage all --traj-tag ex02d
```

The tag is `02d_prod` because that is what `branchA.conf` here sets and a branch A
product is filed under its tag (`<root>/02d_prod/…/<qid>/mace/basinNN/`, since 2026-09-14); `--tag prod` fails with
`no branch A product … under tag 'prod'`. Acetone's branch B for this example is 1 basin
× 3 seeds = 3 dense trajectories, which is one card.

---

## The answer, in one line

**`ν_k` may replace `ω_i` for the entropy. It may not for the zero-point energy or the
enthalpy — and `G_trans` and `G_rot` never depended on the spectrum at all.**

Substituting the whole quasi-harmonic spectrum into `G_total` moves it by **−6.8
kcal/mol** on acetone and **−6.7** on propanal, against a potential whose own error
against RI-MP2 is **0.4** (02c). Where that comes from, term by term, is stage 3.

---

## A 30-minute test, on the compute node, in the foreground

Before a 9-hour submission, run the same chain for 1 + 5 ps where you can watch it:

```bash
source /APP/u22/ai_x86/toolshs/set-XY-I.sh
bash hpc/tools/gpu_shell.sh 1 02:00:00        # a card; two hours covers a few attempts
conda activate openqha-gpu
bash hpc/tools/test30.sh examples/02d_qha_frequency_identity/test30.conf
```

`examples/02d_qha_frequency_identity/test30.conf` writes under **`02d_t30`** and reads branch A from `02d_prod` -- never under the
production tag, because the trajectory driver *resumes* from whatever frames it finds
there. `hpc/tools/test30.sh` refuses a conf whose TAG already has basins for that reason.
Expect 10-15 min, most of it the CPU-side minimisation per basin; delete
`analysis/qha/02d_t30/` afterwards. Its numbers are not science (5 ps is not an entropy),
they are proof that every step ran and wrote where it should.

## Stage 1 — the harmonic limit, where the answer is known

A synthetic trajectory sampled analytically from the molecule's own MACE Hessian. The
surface **is** harmonic, so `ν = ω` exactly and every deviation is sampling noise in the
covariance. Nothing else can contribute, which is what makes the frame-count sweep a
measurement of **how many frames each thermodynamic term needs**.

Acetone, measured 2026-09-08 (`prod` basin 0, 24 modes, ω from 79.72 to 3182.60 cm⁻¹):

```
 frames   max|dnu|       dZPE     dE_vib       dT*S    minO   minBlk   dup
    200     861.74    +3.5393    +3.5719    +0.0690   0.334    0.403     1
    500     381.22    +1.4246    +1.4357    +0.0226   0.275    0.275     1
   1000     314.16    +0.8319    +0.8311    -0.0078   0.331    0.499     2
   2500     102.06    +0.4349    +0.4305    -0.0071   0.260    0.365     2
   5000      83.21    +0.1000    +0.0985    -0.0010   0.395    0.573     2
  12500      40.83    -0.0035    -0.0054    -0.0037   0.434    0.731     0
  50000      16.16    -0.0145    -0.0140    -0.0012   0.427    0.897     1
```

Propanal's three basins give the same curve to within 0.01 kcal/mol at every row, so this
is a property of the estimator and the number of degrees of freedom, not of the molecule.

**Read the `dT*S` and `dZPE` columns against each other.** At 200 frames — 8.3 frames per
degree of freedom — the entropy is already within **0.07 kcal/mol** while the zero-point
energy is out by **3.54**. To reach 0.01 kcal/mol the entropy needs about 1000 frames and
the ZPE about 12 500.

The reason is not subtle once seen: entropy weights the **soft** end of the spectrum,
where the variance is large and easy to resolve; ZPE and enthalpy are `½hcΣν` and are
dominated by the **stiff** end, whose variances are tiny and need far more samples before
their eigenvalues settle. A finite-sample covariance has systematically too-small
eigenvalues, so `ν` comes out too **high** and the ZPE too **large** — a positive bias
that only disappears with sample count.

### What this does to the production protocol

`configs/branchB_protocol.yaml` samples one frame per **1.0 ps**, chosen to filter
high-frequency noise out of the entropy. A 500 ps production therefore yields **500
frames**, and the table says the harmonic-limit ZPE error at 500 frames is **+1.42
kcal/mol** — from sampling alone, before any anharmonicity, on a surface that is exactly
harmonic.

**The interval that makes the entropy clean is the same interval that makes the ZPE
unobtainable.** The two terms want opposite sampling. That is why `run_trajectory.sh`
samples every 2 fs: 12 500 frames is 12.5 ns at 1.0 ps and 25 ps at 2 fs.

---

## The overlap gate in the formula is unsound as written

The natural gate is `max_j O_ij ≥ 0.7`. On acetone's **exactly harmonic** surface at
50 000 frames it rejects modes reproduced to a fraction of a wavenumber:

```
mode      nu        omega    max_j O_ij   block O
  13   1455.60    1455.49       0.774      0.996
  14   1460.09    1459.89       0.427      0.996
  15   1468.79    1459.89       0.499      0.993
```

Acetone's two methyl groups make a **four-fold near-degenerate block** at 1455–1481 cm⁻¹,
and inside a degenerate block the eigenvectors are an arbitrary rotation. The pairing
between individual members is meaningless there; what is invariant is the overlap with
the whole block, and it stays above **0.88** at every block threshold from 5 to 50 cm⁻¹.

So `mode_match` gates on the **block** overlap and reports the per-mode value beside it,
where its collapse is a useful signal that says "these modes are degenerate" rather than
a verdict.

A second reason the per-mode gate cannot be used: **it is itself sampling-limited.** The
`minBlk` column above only reaches 0.7 at 12 500 frames. Eigenvectors converge more
slowly than eigenvalues, so at any trajectory length a real run can afford, an overlap
gate rejects modes whose frequencies are perfectly good.

---

## Stage 2 — the real trajectory, and the answer

The same analysis on molecular dynamics on the real surface, over prefixes of **one**
trajectory so that only the length changes. The deviation here is stage 1's sampling
noise **plus** the physics: anharmonicity, internal rotation, basin escape. Stage 1 is
what makes those separable.

Measured 2026-09-09. Acetone, 12 500 frames at 2 fs = 25 ps, Nosé–Hoover chain,
tdamp 20 fs — **the frame count at which stage 1's ZPE error is 0.004 kcal/mol**:

```
 length   max|dnu|       dZPE     dE_vib       dT*S    minBlk   dup
  5.0ps    3007.04    +6.2690    +6.3815    +0.5851     0.277     2
 10.0ps    3003.86    +0.1240    +0.6542    +1.8003     0.194     2
 15.0ps    3002.99    -2.9329    -2.2789    +2.1912     0.244     3
 20.0ps    3004.92    -3.9129    -3.2420    +2.0784     0.309     4
 25.0ps    3003.93    -5.4246    -4.6985    +2.1758     0.261     2
  crossings: 0 distinct, 137 symmetry-equivalent
```

**Sampling has been eliminated as the explanation, and the ZPE is still wrong by
−5.42 kcal/mol — and still moving.** Propanal's three basins give −4.59, −6.12, −5.9.

Two things in that table are worth reading slowly.

**`max|dnu|` sits at ~3000 cm⁻¹ at every length.** A quasi-harmonic mode's best-overlap
partner is a C–H stretch it is nowhere near. The two bases have stopped corresponding —
`minBlk` is 0.18–0.31 and there are 2–5 duplicate pairings — so on the real surface there
is no pairing `π` to build `S_i^final` on. That is not a threshold being missed; it is
the mapping not existing.

**The `dT·S` and `dZPE` columns are not the same kind of error.** The entropy excess is
**physical**: 137 symmetry-equivalent methyl turns, zero conformer changes, and the same
+2.2 kcal/mol this repository measured before. The ZPE deviation is **an artefact of the
question**: a classical trajectory has no zero-point motion at all, so reading `½hcΣν`
off a classical variance is not a bad estimate of the ZPE, it is not an estimate of it.

**Propanal basin 0 left its basin**: 16 distinct crossings in 25 ps. Its intra-basin
quasi-harmonic entropy is not an intra-basin quantity, and `basin_residence` says so
before anything downstream reads the number.

The trajectory must carry branch A's own atom order. It is checked, never repaired — the
basin `xyz` and a stored trajectory were measured on 2026-09-08 to disagree for acetone,
and a silent reordering scrambles the mass weighting and every mode with it.
`qha.assert_trajectory_identity` runs first, so a biased, constrained or
mass-repartitioned trajectory is refused rather than analysed.

`basin_residence` is reported at every prefix: a trajectory that left its basin has an
inflated `T·S` however smooth its saturation curve looks.

---

## Stage 3 — the substitution itself: `G_total` with `ν_k` in place of `ω_i`

**This is the question, and it needs no mode pairing.** Every term is a *sum* over its
spectrum, so swapping one spectrum for the other is a set-to-set substitution: no
permutation `π`, no overlap, no cutoff. (Whether a pairing could be built is stage 4's
business, and stage 3 does not depend on the answer.)

Acetone, 25 ps, 12 500 frames, kcal/mol:

```
            term  Hessian omega         QHA nu    difference
------------------------------------------------------------
            E_el   -121286.9509   -121286.9509       +0.0000
             ZPE        52.6463        47.2217       -5.4246
   H(T)-H(0) vib         1.5698         2.2959       +0.7261
           E_rot         0.8887         0.8887       +0.0000
         E_trans         0.8887         0.8887       +0.0000
              pV         0.5925         0.5925       +0.0000
         G_trans        -9.8856        -9.8856       +0.0000
   G_rot (sigma)        -5.8917        -5.8470       +0.0446
        -T*S_vib        -2.9297        -5.1055       -2.1758
      (G - E_el)        35.5092        28.6795       -6.8297
         G_total   -121251.4417   -121258.2714       -6.8297
```

Propanal basin 1: `G_total` moves **−6.6555** (ZPE −6.1241, H +0.5649, −T·S −1.1137).

### `G_trans` and `G_rot` cannot move, and the code asserts it

**`G_trans` is identical, exactly.** Sackur–Tetrode depends on the molecular mass, the
temperature and the pressure. No frequency enters it, so a difference there would be an
input error, and `gtotal.compare` raises rather than reports one.

**`G_rot` moved +0.0446 — and that is geometry, not spectrum.** The rigid rotor depends
on the moments of inertia and σ; no frequency enters it either. The Hessian route is
evaluated at the minimum, the quasi-harmonic route at the trajectory's mean structure, and
those are not the same point. The record labels it `rotational_geometry_shift_kcal` so it
can never be read as an effect of the substitution.

So of the six terms, **exactly three can move**: ZPE, the thermal enthalpy, and `−T·S`.

### Which band carries which term — the argument, measured

Acetone, from the Hessian spectrum:

```
 band / cm^-1    n          ZPE    share      H(T)-H(0)    share          T*S    share
--------------------------------------------------------------------------------------
    below 500    4       1.5404     2.9%         1.2616    80.4%       2.5358    86.6%
  500 to 1500   13      21.7303    41.3%         0.3074    19.6%       0.3929    13.4%
   above 1500    7      29.3755    55.8%         0.0008     0.1%       0.0009     0.0%
        total   24      52.6463   100.0%         1.5698   100.0%       2.9297   100.0%
```

**ZPE is 55.8% from above 1500 cm⁻¹ and 97.1% from above 500. The thermal enthalpy is
80.4% from below 500, and above 1500 contributes 0.1%.** Entropy behaves like the
enthalpy: 86.6% from below 500.

That is the whole argument in one table. `ZPE = ½hcΣν` weights every mode by its
frequency, so the stiff end dominates — one C–H stretch at 3000 cm⁻¹ is worth 4.29
kcal/mol of ZPE, a torsion at 100 cm⁻¹ worth 0.14. `H(T)−H(0) = Σ hcν/(e^{hcν/kT}−1)`
weights by thermal occupation, and a 3000 cm⁻¹ mode is not excited at 300 K.

**A trajectory resolves the soft end well and the stiff end badly.** So it can carry the
enthalpy and the entropy, and it cannot carry the ZPE — not because the estimator is poor
but because the ZPE is a question about the part of the spectrum a classical trajectory
has least to say about, and no zero-point motion to say it with.

### Against the yardstick

02c measures MACE-OFF's own thermochemistry as **0.39–0.43 kcal/mol** from RI-MP2. The
substitution costs **6.8**. It is **16× the error of the potential the whole project is
built on** — so this is not a close call that better sampling might settle.

---

## Stage 4 — the hybrid, priced

`mode_match.hybrid_spectrum` builds `S_i^final` and `G − E_el` is reported for the
pure-Hessian, pure-QHA and hybrid spectra side by side, over a sweep of `ν_cut`.

Measured 2026-09-09, kcal/mol, at 25 ps:

```
                        acetone   propanal b0   propanal b1
 hessian                35.5092       36.0252       35.7700
 qha                    28.6348       30.1118       29.0972
 hybrid nu_cut = 25     35.5092       36.0252       35.7700   (0 modes taken)
 hybrid nu_cut = 50     35.5092       36.0252       35.7700   (0 modes taken)
 hybrid nu_cut = 100    30.5733       36.0252       30.9618   (1, 0, 1 taken)
 hybrid nu_cut = 150    30.5733       36.0252       31.0627   (1, 0, 3 taken)
```

**The hybrid is not a rescue, and the table shows both ways it fails.** At the cuts
where the block-overlap gate passes nothing, the hybrid *is* the Hessian answer and the
trajectory contributed nothing. At the cuts where it passes something, taking **one**
soft mode from the quasi-harmonic spectrum moves `G − E_el` by about **5 kcal/mol** —
five times the target accuracy, from one mode.

The pure-QHA column is 5.9 to 6.9 kcal/mol below the Hessian on every basin. That is
what the spectrum substitution costs when it is taken at face value.

`ν_cut` was swept, not assumed: `thermo.QRRHO_NU0_CM` is 100 cm⁻¹, Grimme's convention
for free energies, while CREST's entropy mode uses `--sthr 25.0`. **The two differ by a
factor of four, and the table above differs by 5 kcal/mol across that range** — so the
value is a result, not a convention.

---

## Verdict

| term | can `ν(QHA)` replace `ω(Hessian)`? |
|---|---|
| entropy, **as a total** | **yes, and it is arguably the better quantity** — it carries hindered internal rotation the harmonic spectrum omits. It is a different number, not a worse one, and must be entered with the internal symmetry number (see [`plan_AB`](../../.mem/plan/plan_AB_total-free-energy.md)) |
| entropy, **mode by mode** | **no.** On the real surface the pairing does not exist: block overlap 0.18–0.31, 2–5 duplicate pairings, largest paired discrepancy about 3000 cm⁻¹ |
| **ZPE** | **no, and not for a fixable reason.** A classical trajectory has no zero-point motion |
| **enthalpy** | **no**, same reason; it tracks the ZPE column row for row |

So `G_total` keeps two spectra: **ZPE, enthalpy, `G_rot` and `G_trans` from the analytic
Hessian; the entropy from the trajectory, as a total.** That is not the complicated
splice — there is no mode-by-mode `π` in it, and no `ν_cut` and no overlap gate, because
stage 2 says none of those three can be made to work.

---

## Output

```
<molecule>/_records/md_openmm/<setting>/02d_frequency_identity.json
```
(`<molecule>` is the molecule directory of the BASIN tag; since 2026-09-14, before that
`analysis/qha/<tag>/<species>_02d_frequency_identity.json`)

Read in this order:

1. **stage 1 at the frame count your real trajectory actually has** — that is the floor
   on every deviation stage 2 reports.
2. **`distinct_basin_crossings` in stage 2.** Non-zero and the entropy is not an
   intra-basin quantity.
3. **stage 2 minus stage 1, term by term** — the physics, separated from the noise.
4. **stage 3 against 02c's `d(G-Eel)`** — the deviation only matters relative to how far
   the potential itself is from the reference. See
   [`../02c_hessian_benchmark_levels/`](../02c_hessian_benchmark_levels/README.md).
