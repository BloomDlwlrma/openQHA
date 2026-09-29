# Branch B: seeds, length, and what the literature and this repository's own measurements say

*Written 2026-09-13, after the first complete chains on a Tianhe card (02b `qha`, 02d
`identity`, both on an113 / a100x). Every number below is either cited or measured here;
the arithmetic at the end is spelled out so it can be redone when a number changes.*

## 0. The chosen protocol -- supersedes the recommendation in section 5

**One trajectory per (molecule, basin), everywhere: examples and campaign alike.** The
velocity seed follows OpenMM's own convention for `randomNumberSeed=0` -- a fresh seed
per run -- with one addition: the number drawn is written to `md.toml` (`SEED`)
(`seed`, `seed_formula`, `segments`), so every trajectory stays re-derivable.

What this changes in the code (`--seeds 1`, `--seed0 0` are the defaults; 2+ seeds still
accepted):

* `s0_B_qha_trajectory_openmm.py`: `choose_seed()` draws from the OS entropy source when
  `seed0 == 0`; `segments_record()` writes which frames came from which draw, because a
  resumed trajectory restarts from the relaxed geometry with new velocities and appends
  (with a fixed seed the second segment even re-traced the first one's opening steps).
* `s0_B_qha_analyse.py`: criterion 5 (the blank control) reads NOT APPLICABLE with one
  seed per basin and does not fail; the noise floor is not measured in production.
* 02d-2's grid now measures length x sampling through saturation (criterion 1), not the
  seed spread.
* `configs/branchB_protocol.yaml`: `trajectories_per_basin: 1`, `velocity_seed` recorded.

Sections 1-5 below are the argument as it was first written and are kept as such.

## 1. What the quasi-harmonic method is, and what it is not

The entropy this branch computes is the quasi-harmonic (QH) estimate of Karplus & Kushick
(1981) in the form given by Andricioaei & Karplus (2001): the mass-weighted covariance of
the atomic fluctuations is diagonalised, each eigenvalue is turned into an effective
frequency, and the entropy is the sum of quantum harmonic-oscillator entropies over those
frequencies (`openqha/quasi_harmonic/qha.py`). The Schlitter (1993) formula is the
analytic upper bound to it, and the code checks `S_QH <= S_Schlitter` on every trajectory.

Two properties decide how it must be sampled:

* **It converges from below with sampling.** A covariance estimated from T frames has at
  most T-1 non-zero eigenvalues, and the fluctuation of each mode is under-estimated until
  that mode has been sampled through several of its own periods. Baron, van Gunsteren &
  Hünenberger (JCTC 2009) measured this on a peptide in water: 99 % of the final QH entropy
  needed 0.7 µs, and 1.1 µs once anharmonic and pairwise-correlation corrections were
  included.
* **In the limit it is an upper bound to the true configurational entropy**, because the
  effective harmonic model ignores anharmonicity and mode correlations (Baron et al.
  2009). Chang, Chen & Gilson (JCTC 2005) showed the over-estimate is large whenever
  *several wells are occupied* and larger still in Cartesian than in internal
  coordinates. The method was introduced for single-well systems.

The second point is what 02d showed at 5 ps on an113: `32 symmetry-equivalent crossings`
(methyl rotations) in one basin of dsgdb9nsd_000018, and ZPE from the QH frequencies
15.9 kcal/mol below the Hessian value. That is not seed noise and no number of seeds
reduces it; it is why the chain records basin residence per trajectory and why
`--atoms heavy` exists. The per-molecule question "does this basin contain an internal
rotor" is the dominant systematic of this branch, ahead of anything in this document.

## 2. The random seed: OpenMM, GROMACS, and this repository

* **OpenMM.** `randomNumberSeed` defaults to 0, and 0 means "choose a unique seed when
  the Context is created" -- for every stochastic integrator (Langevin, LangevinMiddle,
  Brownian, Custom) and for `setVelocitiesToTemperature` called without a seed. Two runs
  of the same script give two different trajectories.
* **GROMACS.** `ld-seed = -1` (default) draws a pseudo-random seed for SD/BD thermal
  noise; `gen-seed = -1` does the same for generated velocities.
* **This repository.** The protocol (`configs/branchB_protocol.yaml`, Rinaldo & Field
  2003) integrates with a Nosé--Hoover chain, which is deterministic. The only stochastic
  input is the initial velocity draw, and it is given an explicit seed:
  `seed = seed0 + 1000*basin + seed_index`, `seed0 = 20260903`
  (`scripts/production/s0_B_qha_trajectory_openmm.py`). The seed is recorded in
  `md.toml`.

So the codes' defaults ("different every run") are the right thing for interactive work
and the wrong thing for a campaign of 133 885 molecules, where every trajectory must be
re-derivable from its record. The explicit seed stays. (Bitwise reproducibility on a card
is a separate matter: CUDA reductions are not order-deterministic; the seed makes a run
re-derivable, not bit-identical.)

## 3. Do the examples need three seeds? Yes.

Knapp, Ospina & Deane (JCTC 2018, 310 µs over 100-replica sets) make the general point
that a single trajectory carries no information about its own variance; conclusions
drawn from one replica are false-positive-prone, and the spread between replicas is the
only estimate of the error that does not rest on an assumption. 02b, 02d and 02d-2 are
exactly the places where that spread is *measured* -- 02d-2 is a length x sampling-interval
grid whose whole purpose is to see when the seed spread falls under the 0.3 kcal/mol
saturation budget. Their `SEEDS=3` stays.

## 4. Does the 133 885-molecule campaign need three seeds? The arithmetic first.

Measured (an113, A100-SXM4-80GB, MACE-OFF23_medium, 10 atoms, 1 fs, CUDA double):

| condition | s/ps per trajectory | card throughput |
|---|---|---|
| 6 trajectories sharing one card, standard sampling | 43.6 | 0.138 ps/s |
| 3 trajectories sharing one card, one frame per step (02d) | 59.0 | 0.051 ps/s |

(The A800 sweep on an45 found 12 workers per card the optimum -- ~1.3x the 6-worker
throughput -- but that has not been re-measured on an A100.)

One protocol trajectory is 50 + 500 = 550 ps. At 0.138 ps/s that is **1.1 card-hours per
trajectory**; 0.9 trajectories per card-hour.

| campaign | trajectories | card-hours | on 48 cards |
|---|---|---|---|
| 133 885 x 2.5 basins x 3 seeds | 1.0 M | 1.1 M | 2.6 years |
| 133 885 x 2.5 basins x 1 seed | 0.33 M | 0.37 M | 0.9 years |

(2.5 basins per molecule is a placeholder; the branch A census gives the real number.
48 cards is 6 nodes of 8; the AI cluster's node quota is 6.)

Two conclusions that do not depend on the placeholder:

1. **Seeds are a factor of 3; length is the larger lever.** 500 ps at 1 fs is 5 x 10^5
   force calls per trajectory. Whether the protocol length can be shortened is what
   02d-2's grid measures (p500/p1000/p1500 at s1/s2/s5); the campaign should not be
   sized before that grid has run once at production length.
2. **At fixed card-time, one trajectory of 3L converges the QH covariance better than
   three of L.** The estimate converges with the sampling of *one continuous well*
   (section 1); splitting the same budget into replicas re-pays the equilibration and
   the slow initial rise three times. Replicas buy two things only: an error bar, and a
   second chance to see a basin crossing.

## 5. Recommendation

* **Examples 02b / 02d / 02d-2: `SEEDS=3`**, unchanged. They are the calibration.
* **Campaign: `SEEDS=1` per (molecule, basin), plus a calibration subset at `SEEDS=3`.**
  Choose the subset stratified by the branch A flexibility ranking
  (`analysis/package1_flexibility_ranking.parquet`) -- roughly 1 % of molecules, ~1 300,
  spanning rigid to floppy -- and report the seed-to-seed spread measured there, as a
  function of flexibility class, as the error bar on every single-seed molecule of that
  class. This is the Knapp et al. argument applied at the level where it is affordable:
  the spread is a property of a class of molecules, measured with replicas, and not
  re-measured 133 885 times.
* **If a per-molecule error bar is a requirement of the dataset** (rather than a
  per-class one), the honest cost is `SEEDS=2` everywhere (spread from two is a rough
  bar) and the campaign doubles. That is a decision for the user, not a default.
* Whatever the seed count, the crossing detector's verdict (`distinct_crossings`) is
  recorded per trajectory and a basin whose trajectories left it is not an intra-basin
  entropy; that flag, not the seed spread, is the first thing to read in the campaign's
  output.

## Sources

* Karplus & Kushick, *Macromolecules* 1981, 14, 325 (the method).
* Andricioaei & Karplus, *J. Chem. Phys.* 2001, 115, 6289 (the quantum QH form used here).
* Schlitter, *Chem. Phys. Lett.* 1993, 215, 617 (the upper bound checked on every run).
* Baron, van Gunsteren & Hünenberger, *JCTC* 2009 -- https://pubs.acs.org/doi/10.1021/ct900373z
* Chang, Chen & Gilson, *JCTC* 2005, 1, 1017 -- https://pubs.acs.org/doi/10.1021/ct0500904
* Knapp, Ospina & Deane, *JCTC* 2018, 14, 6127 -- https://pubs.acs.org/doi/10.1021/acs.jctc.8b00391
* OpenMM `LangevinMiddleIntegrator` (seed 0 -> unique per Context) --
  https://docs.openmm.org/latest/api-python/generated/openmm.openmm.LangevinMiddleIntegrator.html
* GROMACS mdp options (`ld-seed`, `gen-seed`) --
  https://manual.gromacs.org/current/user-guide/mdp-options.html
* Rinaldo & Field, *Biophys. J.* 2003 (the integration protocol adopted 2026-09-07).
