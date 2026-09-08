# 02d — can a quasi-harmonic `ν` stand in for a Hessian `ω`?

If it can, then

```
G_total = E_el + G_trans + G_rot + Σ_i [ ZPE_i(ω_i) + H_i(ω_i,T) − T·S_i ]
```

can be built from **one** spectrum instead of two, and the hybrid splice — Hessian for
the stiff modes, QHA for the soft ones — is unnecessary machinery.

```bash
# seconds, no MD at all: the harmonic-limit control
python examples/02d_qha_frequency_identity/s0_frequency_identity.py \
    --species dsgdb9nsd_000018 --tag prod --stage harmonic

# needs a trajectory whose atom order is branch A's own
BASINS=data/basins/prod/1_16000/1_4000/dsgdb9nsd_000018.basins.xyz \
SPECIES=dsgdb9nsd_000018 bash examples/02d_qha_frequency_identity/run_trajectory.sh

python examples/02d_qha_frequency_identity/s0_frequency_identity.py \
    --species dsgdb9nsd_000018 --tag prod --stage all --traj-tag ex02d
```

---

## The answer, in one line

**For entropy, yes. For zero-point energy and enthalpy, no — and the obstacle is the
frame count, not the physics.**

---

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

## Stage 2 — the real trajectory

The same analysis on molecular dynamics on the real surface, over prefixes of **one**
trajectory so that only the length changes. The deviation here is stage 1's sampling
noise **plus** the physics: anharmonicity, internal rotation, basin escape. Stage 1 is
what makes those separable.

The trajectory must carry branch A's own atom order. It is checked, never repaired — the
basin `xyz` and a stored trajectory were measured on 2026-09-08 to disagree for acetone,
and a silent reordering scrambles the mass weighting and every mode with it.
`qha.assert_trajectory_identity` runs first, so a biased, constrained or
mass-repartitioned trajectory is refused rather than analysed.

`basin_residence` is reported at every prefix: a trajectory that left its basin has an
inflated `T·S` however smooth its saturation curve looks.

---

## Stage 3 — the hybrid, priced

`mode_match.hybrid_spectrum` builds `S_i^final` and `G − E_el` is reported for the
pure-Hessian, pure-QHA and hybrid spectra side by side, over a sweep of `ν_cut`. The
price of each choice is then a number rather than an argument.

`ν_cut` is swept and not assumed. This package's `thermo.QRRHO_NU0_CM` is 100 cm⁻¹,
Grimme's convention for free energies; CREST's entropy mode uses `--sthr 25.0`. **The two
differ by a factor of four**, so the value has to come out of the scan.

---

## Output

```
analysis/qha/<tag>/<species>_02d_frequency_identity.json
```

Read in this order:

1. **stage 1 at the frame count your real trajectory actually has** — that is the floor
   on every deviation stage 2 reports.
2. **`distinct_basin_crossings` in stage 2.** Non-zero and the entropy is not an
   intra-basin quantity.
3. **stage 2 minus stage 1, term by term** — the physics, separated from the noise.
4. **stage 3 against 02c's `d(G-Eel)`** — the deviation only matters relative to how far
   the potential itself is from the reference. See
   [`../02c_hessian_benchmark_levels/`](../02c_hessian_benchmark_levels/README.md).
