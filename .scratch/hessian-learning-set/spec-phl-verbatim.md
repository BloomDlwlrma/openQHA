# Spec: PHL verbatim -- deleting the projected path, one algorithm per step (tickets 35-38)

Label: `ready-for-agent`. Tracker: `.scratch/hessian-learning-set/`. **Rulings: S0-C-64** and **S0-C-67** (the validation probes are stored in the dataset, not seeded from the Label; step 4)
**S0-C-64**
(2026-09-23) -- *the training loss is PHL's as published: the Cartesian Hessian's MSE, per
structure over (3N)^2, estimated by Hutchinson random probes through Hessian-vector products;
"projected" means the random-vector projection and nothing else; mass-weighting and the Eckart
projection exist only in evaluation, where they are the standard vibrational analysis.* This
spec supersedes the loss side of `spec-fine-tune-basins.md` (its "Loss = PHL's Cartesian term"
decision, which kept the projected variants as diagnostics and scan rows) and
`design-phl-loss.md` (the projected, mass-weighted, entropy-weighted loss of 2026-09-18,
ticket 11). Everything else of `spec-fine-tune-basins.md` stands.

**What is already settled, and is not this spec's work.** S0-C-53 made the Cartesian matrix the
target; S0-C-54 basin frames only; S0-C-55 the fixed-probe validation; S0-C-56/57/60 the Replay
and the single production row R4; S0-C-58/59 the judge's gate on the matrix itself and its
closure. On 2026-09-23, under **S0-C-65**: ticket 33 moved the split to MACE-OFF's granularity
(whole test molecules at 5 %, validation by frame from the training molecules), ticket 34 added
the exact anchors on the validation file before and after training (path A, no fork change), and
ticket 32's probe calibration lost its verdict columns -- whether K = 4 is enough is read from a
real run, not extrapolated. T04 and T05 were rewritten to PHL verbatim on 2026-09-22 and are
re-executed at the end of this work, because the probe module's signatures change under it.

What remains is the deletion itself: the code still carries both loss paths, and the fork still
offers the flags that select the one that no longer exists. Six steps, one algorithm each, with
the derivation each step rests on; steps 0 and 4 are done (tickets 33/34/32) and are kept here
because the derivations are what the remaining tickets are checked against.

## Problem statement

The training code carries two loss paths: the one that trains (S0-C-53/64: PHL's Cartesian
Hessian, sampled by Hessian-vector products) and one that never trains -- the projected,
mass-weighted, entropy-weighted norms designed on 2026-09-18 when the target was the msRRHO
entropy, kept after S0-C-53 as "diagnostics and scan rows". The second path is two thirds of
`phl.py` (7 functions and a `metric` argument threaded through 3 more), a branch in every
`FrameConstants` with 7 extra slots, a three-way `mode_weighting` in the loss, the driver, the
Slurm script and every Record, a `modes` probe set that needs the reference eigenvectors, a
three-row balance ladder in the smoke fit, a `LOSS_EXACT` column in the judge, and two flags in
the fork's argument parser.

Nobody trains on it: Rodriguez 2025 (element-wise RMSE), PHL (eq. 6, MSE with Hutchinson probes),
PFT (MAE on a randomly sampled column) and HIP (element-wise MAE/MSE plus a subspace term in the
REFERENCE Cartesian eigenbasis) all train on the Cartesian matrix, and HIP's mass-weighting and
Eckart projection appear only where frequencies are read. Keeping the path costs tests, reading
time -- it was the hardest part of T04 and T05 until they were rewritten -- and carries a
permanent risk: a flag that mace parses but our loss no longer reads would sit in every
`config.yaml` claiming a target that did not run, which is the failure S0-C-56 was written
about. It buys nothing the judge does not already read from the trained matrix by the standard
analysis.

## Solution

Delete the projected path in the order the algorithms run, so that every step leaves the suite
green and the Record honest: **35** Algorithms 1-2 (the frame's constants become the Label, the
normalisation and the seed; the probes become the draw and one matvec; the seven projected
functions and the `metric` argument go, with the derivations of Step 2 as the tests), **36**
Algorithm 3 (the training step loses `mode_weighting` in the loss, the driver, the Slurm script
and the Record; fork commit D removes `--hessian_mode_weighting` and the `modes` choice of
`--hessian_probe`; the smoke fit's ladder becomes probe kind x K on the one target), **37**
Algorithm 4 (the fixed-probe evaluation on the reduced constants), **38** Algorithm 5 and the
documents (the judge's `LOSS_EXACT` column goes, the frequency rows are named for what they are,
CONTEXT and ADR 0006 say it once, T03 is archived with a banner and T04/T05 are re-executed
against the new signatures).

At the end: one probe module without a metric switch, a loss module about half its size, no
`mode_weighting` anywhere, a fork whose parser offers only what the loss reads, and a judge table
in which every column is a quantity the loss trains or the standard analysis computes.

## User stories

1. As the trainer, I want the Hessian term of the loss to be PHL's eq. 6 sampled by PHL's Algorithm 1 -- and nothing else -- so that the code trains what the literature trains and what S0-C-53 rules.
2. As the trainer, I want the probe the model sees to be the raw random draw and the reference side a matvec on the stored Label, so that no mass matrix, projector or mode basis sits between the label and the loss.
3. As the trainer, I want the `mode_weighting` switch gone from the loss, the driver, the Slurm script, the fork's parser and the Record, so that no run can be configured onto a target that is not the ruling's.
4. As the trainer, I want the `modes` probe gone (it needs the reference eigenvectors), leaving `gaussian` (PHL's Algorithm 1 and the default since S0-C-68) and `rademacher` (the smaller variance, kept as a choice), so that every probe set is a random draw or the deterministic unit set.
5. As the trainer, I want `w_H` measured by the balance rule ~~on the full Cartesian matrix only~~ **with the run's probe setting (gaussian k=4 by default; the exact full-matrix reading remains the anchors' and the judge's job)**, so that ~~the driver's default and the smoke fit's balance table are one number~~ **the balance is measured under the trained term's own estimator** *(amended 2026-10-01 -- hessian-learn-framework ticket 15: the balance on the probe estimator)*.
6. As the reader of a Record, I want `LOSS phl`, `PROBE`, `N_PROBES`, `VALID_PROBES`, `HESSIAN_WEIGHT`, `HESSIAN_WEIGHT_RULE`, `BALANCE_L_*` and no `MODE_WEIGHTING`, so that the Record names only quantities that exist.
7. As the judge's reader, I want the gate row to be the target itself (the frame-weighted held-out `||dH||^2/(9N^2)`, engine over base) and the frequency rows to be stated as the standard vibrational analysis of the trained matrix, so that "projected" never appears in a training claim.
8. As the judge's reader, I want the `LOSS_EXACT` column (the projected norm) gone, so that the judge table carries no quantity the loss does not train.
9. As a reader of T04/T05, I want each algorithm's derivation to be exactly what the code does -- unbiasedness, variance, exactness, the HVP, the third-order graph, the masked mean, the balance rule, the Stage Two rescaling, Weyl's bound -- so that the notebooks are the specification of the code and not of a superseded design.
10. As a reader of T03, I want it archived with a banner (the projected design; superseded by S0-C-53), not deleted, so that the record of why the projected loss was designed and why it was dropped survives.
11. As a maintainer, I want the fork's `--hessian_mode_weighting` flag removed (commit D), so that a flag mace does not read never enters a Record (the S0-C-56 principle).
12. As a maintainer, I want the unit tests of the probe and loss modules to test the PHL estimator's properties (unbiased, exact with unit probes, fixed at validation, masked in a batch) on the stored fixture Hessians, so that the tests are the derivations run.
13. As a maintainer, I want the smoke fit's ladder to be probe kind x k on the Cartesian target, so that the cost table compares what can be run.
14. As the campaign, I want the Replay's Record to carry the expected coverage of the draw (the number of distinct SPICE molecules a uniform-by-frame draw of n frames touches), so that Algorithm 0's arithmetic is a Record field and not a notebook calculation.
15. As a reader of CONTEXT.md, I want the Loss entry to say "PHL's full-Hessian loss sampled by random probes" and the Judge entry to name the standard vibrational analysis for the frequency rows, so that the glossary matches the code.
16. As a future reader, I want ADR 0006 to say why the loss is PHL verbatim and why the Eckart projection is evaluation-only, so that the projected loss is not re-adopted by accident.
17. As a reader of CONTEXT.md, I want the distribution entry to say that the production split is by MOLECULE (S0-C-65) and that `interpolation` is therefore empty in production, so that the glossary stops describing the split ticket 33 replaced.
18. As the person paying for the campaign, I want every step of the deletion to leave the unit suite and both integration suites green, so that the deletion can stop half-way without leaving a broken tree.
19. As a maintainer, I want `phl.estimator_variance` and `phl.make_probes` to keep the names and the meanings the tutorials cite, so that T04's equation numbers still point at the functions after the signatures shrink.

## Seams (where the behaviour is tested)

Existing, and unchanged by this spec: **the driver's command line and Record** (which flags are
and are not passed; the Record's keys), **the loss module's `forward` on a synthetic batch** (the
probe path, the masked mean, the fixed validation probes), **the probe module on the stored
fixture pair** (propanal: the ORCA Hessian and the stored MACE Hessian at the same geometry --
every identity of Algorithm 2 is a numerical check on it), **the judge's Record** (its columns and
verdict lines), and **the fork's own test suite** (the external loss built in multihead mode;
the parser's choices). No new seam. The deletions are visible at every one of them as an absence:
a flag not emitted, a key not written, a column not present, a function not importable.

## The six steps, one algorithm each: principle, derivation, decision

Notation: $N$ atoms, $x\in\mathbb R^{3N}$, $E$ the energy, $g=\nabla E=-F$, $H=\nabla^2E$ (eV Å$^{-2}$),
$\Delta H=H_\theta-H_r$; $\|\cdot\|_F$ Frobenius, $\|\cdot\|_2$ spectral; $\theta$ the model, $r$ the reference.

### Step 0 -- Algorithm 0: who trains on what (the Dataset and the Replay)

```
Algorithm 0  dataset(frames, train_generators = (basin,))            -- dataset.build, S0-C-54
  1  for every labelled frame:  if generator ∉ train_generators:  split ← test, held_out_generator ← yes   # above the draw
     else: the by-frame draw (90 / 5 / 5) or the previous index's decision
  2  Replay ← s0_spice_pt_draw.py --n N --seed S --weight W          # one permutation of SPICE's train split; the first N eligible
     frames (no forgetting-draw molecule, no in_distribution molecule); config_weight = W; <stem>.valid.extxyz from the end
  3  N_TRAIN_HESSIAN ← |{train frames with a Label}|;  REPLAY_R4_FRAMES ← 4 × N_TRAIN_HESSIAN   # the Dataset Record; R4's N and W = 10 (S0-C-60)
  Output  train / valid (basin frames), test (drawn + held out), the Replay file and its companion
```

**Principle.** Three populations enter one training loop: the labelled basin frames (E, F, H),
the Replay (SPICE E/F frames, to keep the base model's knowledge), and nothing else; every other
generator's frame is held out for the judge. Two properties make the Dataset a measuring
instrument rather than a shuffle: a frame's split depends on nothing but itself, and the Replay
of one size is a prefix of the Replay of a larger size.

**Derivation 0.1 (the by-frame split is a per-frame Bernoulli trial).** For a non-pinned
labelled frame $f$ with identity $(\text{molecule},\text{generator},\text{basin},k)$, one number
$u_f\in[0,1)$ is drawn from a generator seeded by $(\text{seed},\text{"frame"},\text{identity})$;
the split is test if $u_f<p_t$, valid if $p_t\le u_f<p_t+p_v$, train otherwise ($p_t=p_v=0.05$).
Since the seed is a function of the identity alone, $u_f$ is the same whenever the frame is
labelled and whatever else is labelled: the split at round $t$ of the label stream is a prefix
of the split at round $t+1$, and the fractions are expectations with binomial error
$\sqrt{p(1-p)/n_{\rm frames}}$ ($\pm0.1\,\%$ over $10^5$ frames). Held-out generators are routed
*above* the draw (step 1), so their frames never consume a draw and the basin frames' decisions
do not move when `train_generators` changes.

**Derivation 0.2 (the Replay is a prefix of one permutation).** Let $\pi$ be the permutation of
SPICE's train frames from `default_rng(S)`. The draw of size $N$ is the first $N$ *eligible*
frames of $\pi$ (eligible: no fragment of the frame's molecule, at connectivity level, in the
forgetting draw's ids nor among the in_distribution molecules). Eligibility does not depend on
$N$, so the draw of size $N$ is a prefix of the draw of size $N'>N$: the rows R1 ⊂ R2 ⊂ R3 and the
coverage of R4 at every label round are nested by construction (S0-C-56). The companion
validation file takes `--n-valid` frames from the *end* of $\pi$, disjoint from any prefix that
the campaign will ever draw ($N\ll|\pi|$).

**Derivation 0.3 (expected coverage of a uniform-by-frame draw).** SPICE holds $n_m$ frames of
molecule $m$, $\sum_mn_m=N_{\rm src}$. Drawing $n$ frames uniformly without replacement, molecule
$m$ is untouched with probability $\binom{N_{\rm src}-n_m}{n}/\binom{N_{\rm src}}{n}\approx(1-n/N_{\rm src})^{n_m}$
(the relative error of the approximation is $O(n\,n_m^2/N_{\rm src}^2)$, below $10^{-2}$ for every SPICE molecule at $n\le68{,}528$), so by linearity of expectation -- no independence between molecules is needed -- the expected number of distinct molecules is
$$C(n)=\sum_m\Big[1-\Big(1-\frac{n}{N_{\rm src}}\Big)^{n_m}\Big].$$
On the SPICE index (14,777 monomer molecules, 682,607 monomer frames, 46 per molecule) this gives
29 % of the molecules at $n=5{,}000$, 69 % at 17,132, 99 % at 68,528 (T05 §5): R3 and R4 differ
by the pull, not the coverage.

**Derivation 0.4 (R4's arithmetic).** With $N_H$ labelled train frames, Replay $=4N_H$ frames at
`config_weight` $=10$: per epoch the E/F terms see $N_H$ frames at weight 1 and $4N_H$ at weight 10,
so the Replay carries $40/41=97.6\,\%$ of the E/F weight and the labelled frames all of the
Hessian weight. The two numbers are PFT's ($K=4$ replay structures per phonon structure, upstream
E/F at 10x), read as dataset ratios because mace's multihead loop sees every frame of either head
once per epoch (`--num_samples_pt` is not read with a user file; the duplication threshold is
passed as 0).

**Decision (step 0), amended 2026-09-23 (S0-C-65).** The split's GRANULARITY changes to MACE-OFF's: whole molecules on the test side (5 %, drawn per structure class, plus the pinned seven), frames on the validation side (5 % of the training molecules' labelled frames -- the 95 % pool split by configuration, as mace's `--valid_fraction` does it). Derivation 0.1 therefore runs at two levels: `u_m` from the molecule's identity decides test, `u_f` from the frame's identity decides valid inside the training molecules, and the prefix property holds at both. Ticket 33 carries it; the by-frame mode stays for the smoke and fit Datasets. The rest of Algorithm 0 is unchanged. The draw tool's `[Replay]` Record gains
`EXPECTED_MOLECULES` = $C(n)$ from the SPICE index (Derivation 0.3) beside the actual
`N_MOLECULES`, and its unit test asserts $C(n)$ on the tiny fixture against the closed form.
Everything else is a verification: the split's prefix property, the draw's prefix property, and
the R4 numbers of the Dataset Record are asserted by the existing tests (ticket 19, 20) and
stated in T04 §6.

### Step 1 -- Algorithm 1: the frame's constants

```
Algorithm 1  frame_constants(H_r)                                  -- once per labelled frame, no autograd (phl_loss.FrameConstants)
  Input   H_r [3N,3N] the raw Cartesian Label, as stored
  1  ν ← 9 N²                                                             # PHL's (3N)² normalisation
  Output  (H_r, ν)                                                          # nothing is diagonalised, projected or hashed
```

**Principle.** The loss needs, per labelled frame, the Label and one number. The Label is used
as stored -- symmetric to the precision of the reference program, never symmetrised, projected or
mass-weighted by the loss (the fork's Label check refuses a non-symmetric or wrongly sized
Hessian at data-loading time; that is a data check, not a transformation). The normalisation is
PHL's $(3N)^2$: the mean squared error per matrix element, so that molecules of different size
contribute on the same scale. The seed makes the *validation* probes a deterministic function of
the Label.

**Derivation 1.1 (why the validation probes are stored, not seeded).** Validation must be
reproducible across epochs, processes and machines, and the probe errors must be independent
between frames (that is what makes $\mathrm{Var}(\text{mean})=\sum_n \mathrm{Var}_n/n^2$ exact
rather than an assumption about the frames). A seed taken from the frame's position in a shuffled
loader gives neither. Until S0-C-67 the seed was `SHA-1(bytes(H_r))[:8]`, which gives both -- but
it is a hash of the *measured object*: a Label recomputed at the same level (another program
version, convergence threshold, dtype or write precision) has different bytes, so the probe set
changes silently and two runs' validation curves stop being comparable with nothing said in the
Record. S0-C-67 therefore follows PHL's fixed-vector protocol: the vectors are drawn once when the
Dataset is built, from a generator seeded by the frame's IDENTITY
(`_rng(seed, "probe", qm9_index, generator, basin, k)` -- the rule the split already uses), and
written into the labelled splits as `valid_probes` $[k_{\max},3N]$ with $k_{\max}=16$
(every labelled frame, not only the valid ones: mace evaluates the loss on the training split
too, and in eval mode the loss draws nothing). The set is
nested: the $K=4$ reading is its first four rows, so a $K$ scan is a flag and not a rebuild. The
loss reads them; nothing in the training path hashes anything. The training probes are unchanged:
they come from mace's own generator, fresh every step (Algorithm 2).

**Decision (step 1).** `FrameConstants` holds the Label (as a tensor on the batch's device and
dtype) and $\nu=9N^2$. Removed by 35: the metric switch, the projector, the reference modes and
eigenvalues, the entropy weights, the weighted projector, `inv_sqrt_m`, `n_vib`, the temperature
and the preset. Removed by 37 (S0-C-67): the seed, the seeded probe draw and BOTH `hashlib` calls
-- `frame_seed` and the `constants()` cache key. The validation probes arrive with the graph; the
cache either keys on the frame's identity or goes, a `FrameConstants` now costing an `asarray`
and two integers.

### Step 2 -- Algorithm 2: the probes and the reference matvec

```
Algorithm 2  probes(frame constants, mode, K, rng | seed)           -- per frame per batch, no autograd (phl.make_probes)
  1  if mode = gaussian:    v_j ~ N(0, I),               j = 1..K          denominator ← ν · K      (PHL's Algorithm 1; the default, S0-C-68)
     if mode = rademacher:  v_j ~ Uniform{−1,+1}^{3N}, j = 1..K          denominator ← ν · K      (the smaller variance; a choice, not a default)
     if mode = cartesian:   v_j ← e_j,                   j = 1..3N        denominator ← ν          (exact: = get_hessian)
     the draw comes from rng in training (mace's seed); at validation it is the frame's STORED set, read from the batch (S0-C-67)
  2  r_j ← H_r v_j                               # the reference side: a matvec on the Label (PHL's bmm(H_ref, v))
  Output  {v_j}, {r_j}, denominator              # the probe the model sees is the draw itself
```

**Principle (PHL eq. 6 and its stochastic form).** The Hessian term is
$$\mathcal L_H=\frac{1}{(3N)^2}\|H_\theta-H_r\|_F^2, \qquad
\hat{\mathcal L}_H^{(K)}=\frac{1}{9N^2K}\sum_{j=1}^K\|H_\theta v_j-H_rv_j\|^2 ,$$
with $v_j$ i.i.d., components independent, zero mean, unit variance.

**Derivation 2.1 (unbiased).** With $A=\Delta H$ and $E[vv^{\!\top}]=I$,
$E\|Av\|^2=E[v^{\!\top}A^{\!\top}Av]=\operatorname{tr}(A^{\!\top}A\,E[vv^{\!\top}])=\operatorname{tr}(A^{\!\top}A)=\|A\|_F^2$.
Averaging $K$ independent probes and dividing by $9N^2$ gives $E[\hat{\mathcal L}_H^{(K)}]=\mathcal L_H$
for every $K\ge1$. The gradient is unbiased too: $\partial_\theta$ and $E_v$ commute because the
probes do not depend on $\theta$.

**Derivation 2.2 (variance; what the Rademacher draw would have bought).** Let $B=A^{\!\top}A$ (symmetric,
$\operatorname{tr}B=\|A\|_F^2$) and $X=v^{\!\top}Bv$. Then
$E[X^2]=\sum_{ijkl}B_{ij}B_{kl}E[v_iv_jv_kv_l]$. For Rademacher components
$E[v_iv_jv_kv_l]=\delta_{ij}\delta_{kl}+\delta_{ik}\delta_{jl}+\delta_{il}\delta_{jk}-2\,\delta_{ijkl}$
(the all-equal case is counted three times by the pairings but equals 1), so
$E[X^2]=(\operatorname{tr}B)^2+2\|B\|_F^2-2\sum_iB_{ii}^2$ and
$$\operatorname{Var}_{\rm Rad}[X]=2\Big(\|B\|_F^2-\sum_iB_{ii}^2\Big).$$
For Gaussian components the fourth moment has no correction term ($E[v_i^4]=3$), giving
$\operatorname{Var}_{\rm Gau}[X]=2\|B\|_F^2\ge\operatorname{Var}_{\rm Rad}[X]$. With $K$ probes and
the $1/(9N^2)$ normalisation both variances divide by $(9N^2)^2K$. Rademacher is therefore the
default (Hutchinson's own choice); Gaussian stays as PHL's Algorithm 1 draw. Measured on the
propanal pair: Rademacher variance 1.41 against the formula's 1.32; Gaussian 1.76x that (T04 §2).

**S0-C-68 does not take that bargain.** The user's instruction is to follow PHL as published, and
PHL's Algorithm 1 draws the standard normal, so `gaussian` is the default everywhere (the dataset's
stored sets, the training draw, the driver, the Slurm variable, the fork's parser) and `rademacher`
is a flag nobody sets. The accepted cost, stated: at the same $K$ the validation reading's standard
deviation is $\sqrt{1.76}\approx1.33$ times the Rademacher one -- $K$ stays 4, so the reading is
noisier by that factor and nothing else changes. Two consequences to keep in mind: the predicted
spread must be read from `estimator_variance(...)['gaussian']` (the calibration tool takes it from
`phl_loss.VALID_PROBE`), and a check on a MEASURED spread cannot use a fixed percentage band --
with a normal draw $X=v^\top Bv$ is a weighted sum of $\chi^2_1$, so the band is computed from the
draws' own kurtosis, as `t_phl` and `t_probe_calibration` now do.

**Derivation 2.3 (exact with an orthonormal set).** For any $U$ with $UU^{\!\top}=I$,
$\sum_j\|AU_j\|^2=\operatorname{tr}(A^{\!\top}AUU^{\!\top})=\|A\|_F^2$ with zero variance. With $U=I$
the $3N$ unit probes give $\mathcal L_H$ exactly, and $H_\theta e_j$ is column $j$ of the analytic
Hessian, so the deterministic limit of the estimator *is* `get_hessian` at $3N$ times the cost of
one probe (Step 3's cost model). No other orthonormal set is exact for the Cartesian target unless
it spans $\operatorname{range}(A^{\!\top})$; in particular the reference vibrational modes are
not, which is why the `modes` probe set goes with the projected path.

**Decision (step 2).** `make_probes(H_r, k, mode, rng)` returns the draw, the matvecs and the
denominator; modes `rademacher`, `gaussian`, `cartesian`. `loss_full(H_theta, H_r)` (the exact
value), `estimator_from_products`, `estimator_variance(H_theta, H_r, k)` (Derivation 2.2, both
kinds). Removed: the `metric` argument, `projector`, `reference_modes`, `entropy_weights`,
`weighted_projector`, `error_operator`, `projected_loss_full`, `mode_basis_terms`, the `modes`
probe set, masses and positions from every signature. The fork's `--hessian_probe` choices become
`rademacher | gaussian | cartesian` (commit D, with step 3).

### Step 3 -- Algorithm 3: the training step

```
Algorithm 3  training_step(batch 𝓑, θ, K, w_E, w_F, w_H)          -- one optimiser step (phl_loss.forward inside mace's train loop)
  1  for n in 𝓑 with a Label:  ({v_{n,j}}, {r_{n,j}}, den_n) ← Algorithm 2      # Replay frames and any unlabelled graph: v_{n,j} ← 0
  2  E_θ, F_θ ← model(𝓑, training=True)          # forces with create_graph=True; batch.positions is the leaf
  3  for j = 1..K:
  4      g_j ← autograd.grad( −Σ_n F_{θ,n} · v_{n,j},  positions,  create_graph=True )   # = [H_{θ,n} v_{n,j}]_n, one backward for the whole batch  (hvp.hvp_from_forces)
  5  for n in 𝓑_H:  L̂_n ← Σ_j ‖ g_{n,j} − r_{n,j} ‖² / den_n                     # per-graph slice by ptr  (phl_loss.hvp_error)
  6  L_H ← (1/|𝓑_H|) Σ_{n ∈ 𝓑_H} L̂_n                                               # masked mean; 0 · Σ F if 𝓑_H = ∅
  7  L ← w_E L_E(𝓑) + w_F L_F(𝓑) + w_H L_H                                       # mace's E/F terms, each frame × its config_weight
  8  θ ← optimizer.step( ∂L/∂θ )                 # the parameter backward runs through the third-order graph of step 4
  (multihead: the Replay's frames are in the same batch and pay E/F only -- commit C keeps this loss there;
   Stage Two at 3/4 of the epochs rebuilds it from the swa_* weights, w_H^II = w_H w_F^II / w_F -- run.control_settings;
   w_H itself = w_F L_F / L_H on the base over the train file, the driver's default rule -- run.hessian_weight_balance, S0-C-60)
```

**Derivation 3.1 (the Hessian-vector product from the force graph).** For a constant $v$,
$\frac{d}{d\varepsilon}g(x+\varepsilon v)\big|_0=Hv$, and since $v$ is constant
$\nabla_x\big(g(x)\cdot v\big)=H^{\!\top}v=Hv$ ($H$ symmetric). With $F=-g$:
$$H_\theta v=-\nabla_x\big(F_\theta\cdot v\big),$$
one backward pass over the force graph (`create_graph=True` on the forces). Nothing in MACE's
forward changes; in training mode the forces already carry the graph and `batch.positions` is the
leaf they were taken against.

**Derivation 3.2 (one backward serves the whole batch).** A batch is a block-diagonal graph:
$E(\mathcal B)=\sum_nE_n(x_n)$, so $\nabla_{x_m}\sum_nF_n\cdot v_n=\nabla_{x_m}(F_m\cdot v_m)$ -- graph
$m$'s HVP with its own probe, in the same backward as every other graph's. The Hessian term
therefore costs $K$ backward passes per *batch*, independent of $|\mathcal B|$; an unlabelled
graph (a Replay frame) receives the zero probe and contributes nothing to the sum and nothing to
the gradient through this term. Slicing the result by `ptr` gives each graph's product.

**Derivation 3.3 (the masked mean is unbiased).** With $\mathcal B_H$ the labelled graphs,
$E\big[\frac1{|\mathcal B_H|}\sum_{n\in\mathcal B_H}\hat{\mathcal L}_{H,n}\big]=\frac1{|\mathcal B_H|}\sum_{n\in\mathcal B_H}\mathcal L_{H,n}$
by Derivation 2.1 applied per graph; the batch's population loss is the mean over labelled
frames, which is what the Dataset's `N_TRAIN_HESSIAN` counts. An empty $\mathcal B_H$ gives
$0\cdot\sum F$ -- a zero that keeps the graph connected so autograd finds every parameter.

**Derivation 3.4 (the gradient runs through a third derivative).** With $\rho_j=H_\theta v_j-r_j$,
$$\frac{\partial\hat{\mathcal L}_H^{(K)}}{\partial\theta}=\frac{2}{9N^2K}\sum_j\rho_j^{\!\top}\frac{\partial(H_\theta v_j)}{\partial\theta},\qquad
\frac{\partial(H_\theta v_j)}{\partial\theta}=-\frac{\partial}{\partial\theta}\nabla_x(F_\theta\cdot v_j)=-\frac{\partial^3E_\theta}{\partial\theta\,\partial x\,\partial x}v_j .$$
Autograd delivers it when the HVP itself is built with `create_graph=True` (the "third-order
graph"; PFT's "triple backward"); the E/F terms need only $\partial^2E/\partial\theta\partial x$.
Checked on a toy potential by finite differences to $7\times10^{-10}$ (T04 §5).

**Derivation 3.5 (the cost model).** Reverse-mode differentiation of a scalar costs a fixed small
multiple of the forward it differentiates (Baur-Strassen). With one energy-plus-forces pass with
graph as the unit $u$, one HVP is one more backward over a graph of the same size ($\approx1\,u$;
measured 1.0-1.2 $u$ per probe on MACE-OFF23_medium at $N=10$), the probes share the force graph,
so a step's Hessian term costs $\approx Ku$ on top of the $1u$ of E/F; the full matrix by `vmap`
is $3Nu$ (batched backwards are not shorter). The ratio full : $K$-probe is $3N:K$ -- $57:4$ for
the campaign's 19-atom molecules at $K=4$.

**Derivation 3.6 (the balance rule for $w_H$).** $\mathcal L_H$ is in eV$^2$ Å$^{-4}$ per element,
$\mathcal L_F$ in eV$^2$ Å$^{-2}$ per component; no literature ratio transfers across units, models
and data sets (PHL's $0.09/0.30$ would put the Hessian term at $10\times$ the force term on the
fixture). The rule that does transfer sets the two terms to the same magnitude on the base model
before the first step,
$$w_H=\frac{w_F\,\mathcal L_F}{\mathcal L_H}\Big|_{\theta=\theta_{\rm base}},$$
with $\mathcal L_H$ ~~the *exact* full-matrix value on the train file's labelled frames (the estimator's
noise must not enter a weight)~~ **measured by the balance rule with the run's probe setting (gaussian
$k=4$ by default), drawn per frame from a dedicated generator seeded by the run's `SEED`; the exact
full-matrix reading remains the anchors' and the judge's job. A fixed-seed $k=4$ estimate carries a
$\approx$0.1--0.5 % offset on the full train file -- far below the $\approx$5--35 % per-step noise the
loss itself trains through -- and the weight is thereby measured under the trained term's own
estimator, with the Record keeping the evidence to re-identify it (`BALANCE_PROBE` /
`BALANCE_N_PROBES`)** *(amended 2026-10-01 -- hessian-learn-framework ticket 15: the balance on the
probe estimator)*.
It is a loss-magnitude balance, not a gradient-norm balance; the
Record keeps `HESSIAN_WEIGHT_RULE = balance` and the three terms so that the ratio actually run is
on file. Measured (pre-amendment, the exact reading): 2.948 on the fixture, the driver's rule against
the stored values to 4 digits.

**Derivation 3.7 (Stage Two keeps the share).** When mace rebuilds the loss with the `swa_*`
weights, $w_H^{\rm II}=w_H\,w_F^{\rm II}/w_F$ keeps $w_H\mathcal L_H/(w_F\mathcal L_F)$ unchanged at the
switch (with mace's defaults $w_F^{\rm II}=w_F=100$, so $w_H^{\rm II}=w_H$); both sets are in the Record.

**Decision (step 3).** The loss module's `forward` keeps its logic; its constructor loses
`mode_weighting`, `temperature_K`, `preset`, and refuses nothing about probes but an unknown
name. Fork commit D removes `--hessian_mode_weighting` from the parser and `modes` from
`--hessian_probe`; the fork's tests follow. The driver never emits the flag; `05_train.py` and
`hl_train.slurm` lose `--mode-weighting` / `MODE_WEIGHTING`; the Record loses `MODE_WEIGHTING`
and keeps `LOSS phl`. `epoch_zero_balance` ~~computes the Cartesian value only~~ **reads
$\mathcal L_H$ with the run's probe setting beside the exact path (`cartesian` keeps the full-matrix
reading; the balance uses the run's probe setting)** *(amended 2026-10-01 -- hessian-learn-framework
ticket 15: the balance on the probe estimator)*; the smoke fit's balance table is one row and its
ladder is probe kind x $K\in\{2,4\}$ on the target.

### Step 4 -- Algorithm 4: evaluation on fixed probes

```
Algorithm 4  evaluate(validation set, θ)                            -- the fixed-probe estimator (S0-C-55; phl_loss in eval mode)
  1  loss.eval();  for each batch:  E_θ, F_θ ← model(𝓑, training=True)   # the force graph kept: wants_force_graph_at_eval (commit C)
  2  for each labelled frame:  {v_j} ← the first K = 4 rows of its STORED valid_probes (S0-C-67); r_j ← H_r v_j; den ← v·K   # the same 4 every epoch
  3      L̂_n ← Σ_j ‖ g_j − r_j ‖² / den                                    # under enable_grad (torchmetrics runs update under no_grad)
  4  accumulate w_E L_E, w_F L_F, mean_n L̂_n; total ← their sum            # what ReduceLROnPlateau, the best checkpoint, Stage Two read
  5  eval_summary() → valid_energy_term, valid_forces_term, valid_hessian_term → results/*.txt → the Record's three curves (run.parse_results)
  (the full matrix is never asked for: wants_hessian_at_eval = False; it is the judge's tool, Algorithm 5)
```

**Principle.** The validation Hessian term is the *same* estimator as the training term
(Derivation 2.1: unbiased for every $K$), with the probes fixed per frame (Derivation 1.1) so that
the curve across epochs measures the model and not the draw. It enters the total validation loss
that the scheduler, the checkpoint and the Stage Two switch read.

**Derivation 4.1 (cost and resolution).** Fixed $K=4$ probes cost $4u$ per labelled frame against
$3Nu$ for the full matrix ($4/57$ at $N=19$). The reading's standard error per frame is
Derivation 2.2's at $K=4$; over $n_v$ validation frames the mean's error falls as $n_v^{-1/2}$,
and because the probes are fixed, the *difference* between two epochs on the same frames is free
of draw noise -- the curve's movement is the model's. A flat curve is the warning "the term did
not act" (`HESSIAN_CURVE_MOVED`).

**Derivation 4.2 (why the HVP runs under `enable_grad`).** mace evaluates through torchmetrics,
whose `update` runs under `no_grad`; the HVP is a gradient of the forces with respect to the
positions and needs a graph even when no parameter gradient is wanted. The term is therefore
taken under `torch.enable_grad()` with `create_graph=False` and detached; the force graph itself
is kept by the fork's `evaluate` calling the model with `training=True` when the loss says
`wants_force_graph_at_eval`. torchmetrics' full-state update calls the loss twice per batch, so
the summary accumulates each batch object once.

**Decision (step 4).** The estimator and the schedule are unchanged; where the fixed probes COME
FROM changes (S0-C-67, ticket 37; S0-C-68 for the draw). `04_dataset` draws $[k_{\max}=16,3N]$ standard-normal rows per
labelled VALID frame from the frame's identity and writes them into the valid file; fork commit D
carries them into the batch beside `hessian` / `has_hessian`; the loss takes the first
`valid_n_probes` rows. A labelled valid frame with no stored probes is REFUSED, not drawn for: that
state means a stale valid file, and a silent draw is the failure this ruling removes. The tests of
the fixed probes (two calls on one frame agree to $10^{-12}$; another mace seed gives the same
value; different frames differ; training draws differ) are re-pointed at the stored set and gain
the regression this ruling exists for -- **rewriting the Label bit-for-bit differently at the same
level leaves the probes untouched**. `eval_summary`'s `valid_target` field is dropped (35).

### Step 5 -- Algorithm 5: the judge

```
Algorithm 5  judge(test split, θ, base)                              -- the ruler (judge.run; ticket 22's rows)
  1  for each labelled frame of test, for engine ∈ {θ, base}:
  2      E, F ← get_potential_energy / get_forces:  |ΔE|/N, F RMSE                (every frame, held out or drawn)
  3      if the frame has a Label Hessian:  H ← get_hessian (= the 3N unit probes);
             ‖H − H_r‖²_F/(9N²)  (the target: loss_cartesian);
             hessian_compare(H, H_r, M, x): element MAE, and -- the standard vibrational analysis, mass-weighting + Eckart
             projection, an evaluation step -- frequency MAE all / low, mode overlap
  4  aggregate: per distribution and per class over the Hessian frames (judge.aggregate)
  5  reference bins: (distribution, rms_bin ∈ {0, <0.08, <0.15, ≥0.15 Å}) over ALL frames: H over the Hessian frames, E/F over all (judge.aggregate_displacement)
  6  thermochemistry: MODEL_ERROR_S_REF per pinned molecule, READ from the engine's own msRRHO Records under its tag (--thermo-tag)
  7  forgetting: E/F RMSE on the fixed SPICE test draw, ratio to base
  8  MD ramp (reference, only when asked for): Rodriguez's protocol from each model's own minimum (judge.ramp_one / md_ramp)
  9  verdict lines with GATE (S0-C-58/59):  yes -- held_out_hessian_cartesian (engine ‖ΔH‖²/(9N²) / base − 1 ≤ 0),
                                                     in_distribution_degradation (≤ 15 %), forgetting (≤ 1.15×)
                                           no  -- held_out_low_mode_mae_cm, model_error_s_ref, held_out_generator_hessian_vs_base, md_ramp_K
     VERDICT ← REPORTED while the gate is closed (S0-C-60, the default); with --gate: PASS iff every gate row is PASS or '-'
```

**Settled before this spec, not reopened by it.** *Why a gate exists at all* (a constrained
selection rule over candidate rows, so that "the row that passes every gate row and replays least"
is the registered model without a human choosing) was argued and then answered on 2026-09-22:
with one production row there is nothing to select, so the gate is CLOSED (S0-C-60) and the judge
reports. *Which rows carry the veto when it is reopened* is S0-C-58/59: the held-out Hessian ratio
(the training target itself), `in_distribution_degradation`, `forgetting`. *Where their numbers come
from*: the ratio's threshold is 0 by construction (Derivation 5.1, no free parameter); the 15 % and
the 1.15x are round-2 placeholders (Q7 design-approved, Q8 "OPEN, assumed as stated"), never ruled
and never measured -- marked as such in `spec-fine-tune-basins.md`'s amendment box, to be replaced
by a measured basis (the bootstrap s.e. of the held-out row over the test frames) if and when the
gate is reopened. This spec changes none of it; step 5's only code change is the removal of the
`LOSS_EXACT` column. The derivations below state what the rows *are*, not why they gate.

**Derivation 5.1 (the gate quantity is the target).** Over the held-out Hessian frames $f$ of a
distribution, $\bar{\mathcal L}=\frac1{n_f}\sum_f\|\Delta H_f\|_F^2/(9N_f^2)$ -- the population value
of what Algorithm 3 minimises, computed exactly (Derivation 2.3) on frames the fine-tune never
saw. The gate line is $\bar{\mathcal L}_\theta/\bar{\mathcal L}_{\rm base}-1\le0$: the base against
itself reads exactly 0 and passes; a potential with the Hessian scaled by $0.81$ reads
$\|0.81H_b-H_r\|^2$ against $\|H_b-H_r\|^2$ and fails (the calibration of ticket 22).

**Derivation 5.2 (the frequency rows are the standard analysis, and what the target bounds).**
Frequencies are the eigenvalues of the mass-weighted Hessian $K=M^{-1/2}HM^{-1/2}$ restricted to the
vibrational subspace: with $V$ the orthonormalised translation vectors $M^{1/2}\hat{\mathbf 1}_\alpha$
and rotation vectors $M^{1/2}\hat\Omega_\alpha(x-x_{\rm com})$, $P=I-VV^{\!\top}$ and
$\tilde K=PKP$ has $3N-6$ (or $3N-5$) non-trivial eigenvalues $\lambda_i$,
$\omega_i\propto\sqrt{\lambda_i}$. At a stationary point the six vectors are exact null vectors of
$K$ (translation invariance; rotation invariance with zero gradient); off a stationary point
projecting them out is the convention of every vibrational-analysis program (Miller-Handy-Adams;
gaussian.com/vib; HIP §5). This analysis is applied to *both* matrices by `hessian_compare`; it is
an evaluation step and enters no loss. What the target says about it is Weyl's inequality for
symmetric matrices, $|\lambda_i(K_\theta)-\lambda_i(K_r)|\le\|\Delta K\|_2$, and
$\|\Delta K\|_2\le\|\Delta K\|_F\le\|M^{-1/2}\|_2^2\|\Delta H\|_F=\|\Delta H\|_F/m_{\min}$:
driving the target to zero drives every frequency to the reference's. The bound is loose
(measured: orders of magnitude on the soft modes, T04 §2), which is why the low-mode line is
measured and reported, never inferred -- and never gated (S0-C-59).

**Derivation 5.3 (the held-out generator's reading).** Along a displacement $d$ from a basin
$x_0$, $H(x_0+d)=H(x_0)+\mathcal T[d]+O(\|d\|^2)$ with $\mathcal T[d]_{ij}=\sum_kE_{ijk}d_k$: a
Hessian at a displaced frame differs from the basin's by a slice of the cubic tensor that a
basin-only label set never constrains and that E/F labels constrain in one direction only
(Proposition 1, T04 §1). The RMS bins read
$\rho=\bar{\mathcal L}(\text{bin})/\bar{\mathcal L}(\text{basin})$, engine against base -- the
target's own error off the minimum. It is reported and triggers nothing (ADR 0005).

**Decision (step 5).** The judge's per-frame `loss_exact` / `LOSS_EXACT` columns (the projected
norm) are removed; `loss_cartesian` / `LOSS_CARTESIAN` stay as the gate's quantity. The
frequency rows are unchanged in code and restated in the docstring, CONTEXT's Judge entry and
T05 §6 as the standard vibrational analysis. The msRRHO branch (`thermochem.hessian`,
`hessian_compare`) keeps its Eckart projection -- that is what a frequency is.

## Implementation decisions: what goes, file by file

The inventory below is the survey of 2026-09-23, so that the tickets are bounded and nothing is
found later "still importing it".

**It is the END state of tickets 35-38, not a list for any one of them.** A name is deleted by
the ticket that removes its LAST caller, so the vibrational-analysis half of `phl.py`
(`mass_weighted` .. `mode_basis_terms`) survives 35 for the smoke fit's diagnostic (36) and the
judge's `loss_exact` row (38), under a banner saying it is not a training path; and
`MODE_WEIGHTINGS` / `DEFAULT_MODE_WEIGHTING` survive 35 as the DRIVER's vocabulary for a flag
that only 36 can take out of the fork's parser -- reduced to `("cartesian",)`, with
`build(args)` raising on anything else rather than accepting a flag it ignores. Ticket 35's
closing note states both.

**`openqha/training/phl.py`** (261 lines). Goes: `mass_weighted`, `projector`, `reference_modes`,
`entropy_weights`, `weighted_projector`, `error_operator`, `projected_loss_full`,
`mode_basis_terms`; the `metric` argument of `make_probes` and `estimator_variance`; the `modes`
probe set; `masses` and `positions` from every signature; `METRICS`. Stays, with the same names:
`make_probes(H_r, k, mode, rng)` returning the draw, `r_j = H_r v_j` and the denominator;
`cartesian_loss_full` (renamed `loss_full`, the old name kept as an alias for one release);
`estimator_from_products`; `estimator_variance(H_theta, H_r, k)` with both probe kinds. The
module docstring stops being "Algorithm 1 of the projected Hessian loss".

**`openqha/training/phl_loss.py`** (402 lines). Goes: `MODE_WEIGHTINGS`,
`DEFAULT_MODE_WEIGHTING`, the `mode_weighting` / `temperature_K` / `preset` constructor
arguments, the projected branch of `FrameConstants` and its slots (`masses`, `positions`,
`metric`, `modes_r`, `lam_r`, `weights`, `projector`, `inv_sqrt_m`, `n_vib`, `projector_t`,
`inv_sqrt_m_t`), the `probe == "modes"` refusal (there is no such probe), and `valid_target` in
`eval_summary`. `build(args)` stops reading `args.hessian_mode_weighting`. With 37 (S0-C-67) go
`frame_seed`, the `seed` slot, the seeded `_fixed` draw and the `constants()` cache key: `import
hashlib` leaves the module, and `graph_labels` returns the graph's stored `valid_probes` beside
its Label.

**The fork, commit D** (`openQHA-Hessian`): `mace/tools/arg_parser.py` loses
`--hessian_mode_weighting` and the `"modes"` choice of `--hessian_probe`; `tests/
test_external_loss.py` loses the three assertions that named them. It also GAINS one field
(S0-C-67): a per-graph `valid_probes` `[k_max, 3n]` read from the config's flat key, with
`has_valid_probes` false when it is absent -- the same pattern commit A used for `hessian` /
`has_hessian`, no default draw and no silent zero. Nothing else in the fork changes: commits A-C
stand.

**`openqha/data/dataset.py` and `workflows/hessian_learning/04_dataset.py`** (S0-C-67, ticket 37).
Gains: the valid split's labelled frames carry `valid_probes` drawn from
`_rng(seed, "probe", qid, generator, basin, k)`; Record keys `VALID_PROBE_SOURCE`,
`VALID_PROBE_KMAX`, `VALID_PROBE_SEED`, `N_VALID_PROBE_FRAMES`. `train` and `test` are untouched:
training draws fresh probes every step and the judge reads the full matrix.

**`openqha/training/run.py`**: `mace_argv` stops emitting `--hessian_mode_weighting` and loses
the `mode_weighting` parameter; the Record loses `MODE_WEIGHTING` (schema, info dict, report
line, `registry_entry`'s note); `hessian_weight_balance` and `exact_valid_hessian` stop passing
`mode_weighting=` to `epoch_zero_balance`.

**`openqha/training/smoke_fit.py`**: `epoch_zero_balance` loses `mode_weighting` and computes
only `phl.loss_full`; `MODE_WEIGHTING` leaves its Record.

**`workflows/hessian_learning/05_train.py`** and **`hpc/slurm/hl_train.slurm`**:
`--mode-weighting` / `MODE_WEIGHTING` go; the printed summary line loses the column.

**`scripts/production/s0_hl_smoke_fit.py`**: the balance table becomes one row (there is one
target); the ladder becomes probe kind (`gaussian`, `rademacher`) x K in {1, 2, 4}, with the
`modes` and `none` rows gone; both `MODE_WEIGHTING` schema entries go.

**`openqha/training/judge.py`**: `loss_exact` / `base_loss_exact` (frame rows), `LOSS_EXACT` /
`BASE_LOSS_EXACT` (the distribution and class blocks) and the two `phl.projected_loss_full`
calls go; `loss_cartesian` / `LOSS_CARTESIAN` stay -- that is the gate's quantity. The
`MODE_WEIGHTING` key disappears from the train-Record echo. `hessian_compare` is untouched: its
mass-weighting and Eckart projection are the standard vibrational analysis and stay exactly
where they are.

**`scripts/tooling/s0_probe_calibration.py`**: the one `metric="cartesian"` call site follows the
new signature.

## Documentation decisions (with step 5)

- T03 (`T03_openQHA_Theory_Projected_Hessian_Loss.ipynb`) moves to `docs/tutorials/archive/`
  with a banner cell: the projected design of 2026-09-18, superseded by S0-C-53 and this spec;
  not executed again; `examples/README.md` row updated.
- T04 and T05 are re-executed after steps 2-3 land (`phl.make_probes` loses `metric=`).
- CONTEXT.md: the Loss entry says PHL's full-Hessian loss sampled by random probes (Algorithms
  1-4); the Judge entry names the standard vibrational analysis for the frequency rows; the
  "projected" wording is removed from the training entries. Carried over from ticket 33: the
  DISTRIBUTION entry still describes the by-frame production split ("held out by FRAME (the
  production split, round-5 Q4)") -- it becomes the by-molecule split of S0-C-65, with
  `interpolation` named as empty in production and kept for the smoke / fit Datasets.
- ADR 0006 "PHL verbatim: the Cartesian Hessian trained by random-probe HVPs; the Eckart
  projection is evaluation-only" (context: design-phl-loss.md and ticket 11; decision; consequences:
  what was removed, where the frequency analysis lives). `design-phl-loss.md` marked superseded.
- The ruling recorded as S0-C-61 by `s0_mem_decide.py` with the user's words.

## Testing decisions

- A good test runs a derivation on stored numbers, or reads an argv, a Record, an index or a
  tool's exit -- never mace's internals. Prior art, all of it already in the tree:
  `t_phl.py` (the propanal pair: the ORCA Hessian and the stored MACE Hessian at one geometry),
  `t_phl_loss.py` (forward on synthetic batches), `t_train_run.py` (argv pairs and the Record),
  `t_smoke_fit.py`, `t_judge.py`, `t_probe_calibration.py`, and the fork's `test_external_loss.py`.
- **35**: the estimator's three properties on the fixture pair -- unbiased for Rademacher and
  Gaussian over many draws; the Rademacher variance equal to eq. 2.3 and strictly below the
  Gaussian one; the 3N unit probes exact to 1e-12 -- plus: `make_probes` takes no `metric` and
  no masses (a `TypeError`), `mode="modes"` is not a probe, and the seven deleted functions are
  not importable. `t_phl.py` shrinks from 27 checks to the ones that test what is left;
  `t_phl_loss.py` keeps the fixed-probe and masking checks with the reduced constructor.
- **36**: the driver never emits `--hessian_mode_weighting`; no Record carries `MODE_WEIGHTING`;
  `epoch_zero_balance` takes no `mode_weighting`; the fork's parser REFUSES both the flag and
  `--hessian_probe modes` (the fork's own test); the smoke fit's ladder rows are the new ones.
- **37**: the fixed-probe tests of ticket 21 re-pointed at the STORED set (two calls on one frame
  agree to 1e-12, another mace seed gives the same value, different frames differ, training draws
  differ), plus S0-C-67's own: a Label rewritten bit-for-bit differently at the same level leaves
  the probes untouched, the first 4 rows of the k_max set are the K = 4 set, a labelled valid frame
  without stored probes is refused, and `grep -rn hashlib openqha/training/phl_loss.py` is empty.
  The integration fine-tune still yields three validation curves, a moving Hessian curve and the
  exact anchors of ticket 34, reading a valid file that carries probes.
- **38**: the judge's frame rows carry no `loss_exact`; the gate row still reads exactly 0 for
  base-against-base and FAIL for the 0.81x potential (`t_judge.py`, `t_judge_engine.py`);
  `examples/README.md` points at the archived T03; T04 and T05 execute with 0 errors.
- Every ticket ends with the full unit suite (59 files) and both integration suites green before
  its commit is handed over.

## Out of scope

The judge's rows and thresholds beyond the `LOSS_EXACT` removal (S0-C-58/59/60 stand; the 15 %,
1.15x, 8.5 cm^-1 and 0.2 cal/mol/K numbers remain un-ruled placeholders); the R4 recipe and the
campaign (ticket 16); active learning (round 12, parked); the exclusion-versus-refusal letter of
ticket 19; the msRRHO pipeline's own Hessian analysis and `hessian_compare` (they keep their
mass-weighting and Eckart projection -- that is what a frequency is); `openqha/quasi_harmonic`
and the branch B tools, whose `mass_weighted*` names are their own; upstreaming the fork; any
change to `hvp.hvp_from_forces` or to the fork's commits A-C beyond the parser's two choices;
whether K = 4 is enough (ticket 34's anchors answer it on a real run).

## Further notes

The projected loss was not wrong; it was a different target (the msRRHO entropy's own metric), and
S0-C-53 chose the literature's target instead because the entropy is computed from the matrix
afterwards and the judge measures it there. What this spec removes is the *carrying* of the other
target as an option -- the option is what made T04 hard to read and the code twice its size. The
propanal fixture pair and the 2-methyloxirane frames remain the numerical bench for every
derivation above; the same numbers appear in T04/T05 and in the tests.

PHL's own weights are not adopted with its form. `lambda_F = 0.30` and `lambda_H = 0.09` were,
in its words, "tuned to balance the relative contributions of forces and Hessian information
against energies" on ANI, from scratch, in its units -- and the ratio is dimensional (the units
of `lambda_H / lambda_F` are a length squared, so the same balance reads 0.30 in Angstrom and
1.07 in Bohr). On our fixture, copied verbatim, it would put the Hessian term at 10.2x the force
term at epoch 0, because the base model already fits E and F at its own level while its Hessians
were never supervised (L_H / L_F ~ 34). We adopt the rule the sentence states, not the instance:
`w_H = w_F L_F / L_H` measured on the base model over the run's own train file ~~with the full
matrix~~ **with the run's probe setting (gaussian k=4 by default; `BALANCE_PROBE` /
`BALANCE_N_PROBES` in the Record)** *(amended 2026-10-01 -- hessian-learn-framework ticket 15:
the balance on the probe estimator)*, recorded with its three terms (S0-C-60). The same reasoning
voids any attempt to carry Rodriguez's `eta_H = 0.02`, which multiplies an RMSE rather than an MSE.

## Tickets, in order (`issues/`; proposed, for `/to-tickets`)

| # | step | ticket | status | blocked by |
|---|---|---|---|---|
| 32 | 4 | the probe calibration (`s0_probe_calibration.py`): checks 1 and 2, the funnel diagnostic; its verdict columns voided the same day | done 2026-09-23; the tianhe run is the user's | -- |
| 33 | 0 | the MACE-OFF split granularity (S0-C-65): test = whole molecules 5 % (stratified) + the pinned seven, valid = 5 % of the training molecules' frames, `--split-by molecule` the default with a `--resplit` guard, the judge's empty `interpolation` row | done 2026-09-23 (CONTEXT's distribution entry carried into 38) | -- |
| 34 | 4 | the exact anchors (path A, no fork change): the full matrix on the validation file before and after training, `VALID_HESSIAN_EXACT_BEFORE` / `_AFTER` and the last epoch's probe reading beside them | done 2026-09-23 | -- |
| 35 | 1-2 | Algorithms 1-2: the frame's constants and the probes, PHL verbatim; the projected path removed from the probe and loss modules; the derivations as tests | done 2026-09-23 | -- |
| 36 | 3 | Algorithm 3: the training step without `mode_weighting`; fork commit D -- the parser's two options out, the `valid_probes` field in (S0-C-67); driver, Slurm, Record, smoke fit, balance | done 2026-09-23 | -- |
| 37 | 4 | Algorithm 4 (S0-C-67): the dataset writes the fixed validation probes, the loss reads them from the batch, both `hashlib` calls die; the calibration tool follows; tests re-pointed | done 2026-09-23 | -- |
| 38 | 5 | Algorithm 5: `LOSS_EXACT` removed; the frequency rows stated as the standard analysis; CONTEXT, ADR 0006, T03 archived, T04/T05 re-executed | ready-for-agent | 35, 36, 37 |
| 39 | 2 | S0-C-68: the probes are PHL's standard normal everywhere a default is chosen (dataset, loss, driver, Slurm, fork parser); the measured-spread band is read off the draws | done 2026-09-23 | -- |

Frontier: **35** (nothing blocks it); then 36, 37, 38 in order -- each leaves the suite green, so
the sequence can stop anywhere. The numbers 27-31 went to the other session's tickets (the energy
window, the ANI-1 draw, the displaced frames) while this spec was being written; the work is the
same, the labels moved. The Replay Record's `EXPECTED_MOLECULES` (Derivation 0.3) was proposed
with ticket 33 and is not in it: it is a one-field addition to `s0_spice_pt_draw.py` and can ride
with 36 or be dropped -- it records what the draw's coverage is, which is otherwise a notebook
calculation.
