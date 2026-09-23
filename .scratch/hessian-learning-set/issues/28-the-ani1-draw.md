# 28: The displaced draw becomes NORMAL-MODE SAMPLING at 450 K (`nms`), the equipartition and zero-point draws leave the workflow, and every Frame set is rebuilt with it (`thermochem/hessian.py`, `data/frames.py`, `02_frames.py`, `t_frames.py`)

**Why (user ruling 2026-09-23):** the displaced generator has drawn from the classical
equipartition Gaussian (round-2 Q14, 2026-09-18): per-mode sigma_i = sqrt(k_B T)/omega_i,
total harmonic energy a chi-square with mean (3N-6)/2 k_B T and an unbounded tail. The user
rules the draw over to ANI-1's scheme (Smith, Isayev, Roitberg, Chem. Sci. 2017, 8, 3192;
the ANI-1 data set, Sci. Data 2017):

    R_i = +/- sqrt(3 N_a c_i k_B T / K_i),   c_i >= 0,  sum_i c_i <= 1

so that E_i = (1/2) K_i R_i^2 = (3/2) c_i N_a k_B T and the TOTAL harmonic energy is
rho (3/2) N_a k_B T with rho = sum c_i <= 1: a uniform random partition of at most the
classical total vibrational energy, with random signs. `c` is drawn as
`rng.dirichlet(ones(M)) * rng.random()` (a uniform direction on the simplex times a uniform
radius); the draw is recorded per frame.

**What this changes and what it does not.** msRRHO thermochemistry is computed at BASINS
(basin geometry + basin Hessian) and the training set is basin frames only (S0-C-54), so
neither the thermochemistry nor the training data are touched. The displaced frames are
diagnostics: they carry a reference EnGrad label (no reference Hessian -- round 5 Q7 (b)) and
serve the judge's held-out rows, the in-distribution and forgetting checks, and rho_k (the
curvature change from the basin). The draw is therefore **the scale of those diagnostics**,
and it must be one scale across the campaign -- hence the rebuild.

**It does not fix a near-zero mode** (S0-C-61, dsgdb9nsd_013068 / 025659, 4-6 cm^-1 surviving
the Eckart projection): ANI-1 bounds the ENERGY, not the geometry, and the amplitude is still
sqrt(3 N_a c_i k_B T)/omega_i -- at c_i -> 1 it is sqrt(3 N_a) ~ 6.5 times the Gaussian sigma.
Those molecules keep losing their displaced frames to the energy window, which is the only
filter (ticket 27).

**What to build:**

- `hessian.thermal_displacements(distribution="ani1")`: the per-sample draw moves inside the
  loop (the Gaussian branch unchanged); `q_i = sign * sqrt(2 c_i rho E_tot)/omega_i` in
  mass-weighted amu^1/2 A, `E_tot = (3/2) N_a k_B T`. The record gains
  `energy_fraction` (rho per sample), `harmonic_energy_kcal` per sample and
  `ani1_energy_cap_kcal`; `sigma_q_amu_half_A` stays (it is the Gaussian's scale, nan-free,
  and still describes the modes) with a note that ANI-1 does not use it.
- `frames.DISTRIBUTION = "ani1"`; the ENERGY OF A DISPLACED FRAME paragraph rewritten with the
  three draws side by side and the ruling; `02_frames.py --distribution` gains `ani1`.
- CONTEXT.md, Frame entry: one sentence -- the displaced draw is the scale of the diagnostics,
  independent of msRRHO (basins) and of the training set (basin frames).
- `t_frames` / a new `t_ani1_draw`: the total harmonic energy of every drawn frame is
  <= (3/2) N_a k_B T (to 1e-9); the same seeds reproduce the same frames and different seeds
  do not; the Record says `DISTRIBUTION = ani1`; the Gaussian branch is untouched.

**The rebuild (user's ruling: redo every molecule, do not mix).** `02_frames.py --force`
rewrites a Frame set whose Record exists. The campaign's 1,250 Frame sets were drawn from the
Gaussian; they are rebuilt with the same command the fresh ones use, so after the rebuild
every `frames.toml` says `DISTRIBUTION = ani1`. Cost at the measured 25 min/molecule: 1,250
molecules ~ 520 core-h ~ one day of 12 nodes for the rebuild, plus the ~4,000 not yet built.
The ORCA labels already computed on Gaussian displaced frames (day-1 has none: it labels
basins only, `--generators basin`) would be orphaned -- to be checked before the rebuild runs.

**Blocked by:** nothing. **Unblocks:** the 02 completion of draw300 on one scale.

**Status:** implemented 2026-09-23; the tianhe rebuild is the user's.

- [x] unit (`t_frames`, +1 and two restated): every ANI-1 frame's harmonic energy <= (3/2) N_a k_B T
      (8.9 kcal/mol on propanal) and equal to rho x cap to 1e-6, rho in [0,1], same seeds reproduce,
      other seeds do not, the classical branch untouched (its `energy_fraction` is nan)
- [x] unit group green (58/58)
- [x] `FORCE=1` in `hl_frames.slurm` (`hl_list --force` + `02_frames --force`), the header and the
      campaign page carry the rebuild command
- [ ] tianhe (user): rebuild + finish 02 with `FORCE=1 TAG=draw300 sbatch --array=0-11 --time=1-00:00:00 hpc/slurm/hl_frames.slurm`


## Amendment (user ruling 2026-09-23, after the citation sweep): the method is the name

Three things were ruled after the first implementation, which had called the draw `ani1` at
298 K and kept the other two as options:

1. **The name is the method, everywhere.** `DISTRIBUTION = "nms"` (normal-mode sampling); no
   constant, schema description, CLI help or Record note is named after a paper or carries a
   citation shorthand. The source is cited once where the module surveys the literature
   (`frames.py` FILTER and THE DISPLACED DRAW, `hessian.thermal_displacements`'s docstring):
   Smith, Isayev, Roitberg, Sci. Data 4, 170193 (2017), section "Normal mode sampling", eq. 1
   -- the data-set paper, which carries both the formula and the temperature table, not the
   potential paper (Chem. Sci. 8, 3192) the first implementation cited.
2. **450 K** (`frames.TEMPERATURE_K`), the published setting for 8-heavy-atom molecules and
   the temperature at which this draw's mean energy (3/4) N_a k_B T matches what equipartition
   put in at 298 K: on propanal (4000 draws, MACE Hessian, `design-nms-frames.md` §4) NMS at
   450 K gives RMS 0.113 A mean / 0.266 max against equipartition 298 K's 0.118 / 0.365.
3. **The equipartition and zero-point draws leave the workflow.** `02_frames.py` has no
   `--distribution`; `frames.generate` still takes the keyword (tests, calibration) but the
   workflow passes `frames.DISTRIBUTION`. `hessian.thermal_displacements` keeps all three
   branches -- the other two have callers in calibration and branch B.

The Record's keys follow the method: `c_sum` (the frame's energy share s),
`harmonic_energy_kcal` per frame, `harmonic_energy_cap_kcal` = (3/2) N_a k_B T.
This supersedes the `ani1` naming; `design-nms-frames.md` (2026-09-21) is the design this
implements, with its §4 measurements and its "4 frames per basin at 450 K" proposal (which is
`N_DISPLACED = 4`, unchanged).

- [x] unit `t_frames` (58/58 in the group): the Record says `nms` / 450 K; every frame's
      harmonic energy is under (3/2) N_a k_B T (13.4 kcal/mol on propanal at 450 K) and equals
      `c_sum` x the bound to 1e-6; the mean over 400 draws is (3/4) N_a k_B T within 5 %;
      seeds reproduce; the equipartition branch is untouched (`c_sum` nan). The redraw checks
      now take the temperature from the Record -- they caught the 298 K default when the
      default moved
