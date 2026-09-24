# 27: The energy window is the only filter -- the RMS displacement ceiling comes off the displaced draw (`data/frames.py`, `02_frames.py`, `t_frames.py`)

**Why (2026-09-23, the draw300 Frame sets):** two molecules of the 02 array died with
`RuntimeError: 50 consecutive draws all exceeded the displacement ceiling of 0.15 A`, leaving
no Frame set and no record -- pending forever. Measured on their basins (login node):

| molecule | N | three lowest wavenumbers | RMS of the classical draw (median) | fraction <= 0.15 A | isotropic scaling c (median) -> T_eff |
|---|---:|---|---:|---:|---|
| dsgdb9nsd_013068 | 14 | **6.1**, 83.8, 86.0 cm^-1 | 1.369 A | 3 % | 0.11 -> **4 K** |
| dsgdb9nsd_025659 | 15 | **4.2**, 86.8, 95.6 cm^-1 | 1.971 A | 2 % | 0.08 -> **2 K** |

The 4-6 cm^-1 mode is not a soft vibration, it is a near-zero eigenvalue surviving the Eckart
projection; the classical amplitude sigma = sqrt(k_B T)/omega then diverges and the whole draw
leaves the molecule. Isotropic scaling (`dx *= ceiling/rms`) is therefore out: it would put
these frames at 2-4 K, not 298 K (the harmonic energy scales as c^2, so T_eff = c^2 T).

**The ruling (user, 2026-09-23; decision S0-C-61): the energy window is the only filter.** That is what ANI-1
(Smith, Isayev, Roitberg, Sci. Data 2017; 275 kcal/mol) and SPICE (Eastman 2023; 1e4 kJ/mol)
do, and it is what `frames.py`'s FILTER paragraph already says about the KEPT frames -- the
0.15 A ceiling was a second, undeclared filter inside the draw. It comes off: the first draw
is taken whatever its RMS, the frame is built, the engine's own energy decides.

**What to change:**

- `frames.MAX_RMS_A = None` (the ceiling off; `hessian.MAX_RMS_DISPLACEMENT_A` stays as that
  function's own default for other callers). `thermal_displacements` already accepts None and
  then takes the first draw -- no rejection loop, so the RuntimeError cannot arise.
- The Record: `MAX_RMS_A` is `nan` when there is no ceiling (the key stays, a reader sees it);
  `RMS_DISPLACEMENT_A` per frame is unchanged -- it was always reported, now it is only reported.
- `02_frames.py --max-rms` keeps working (a caller may set a ceiling); its help says the
  default is no ceiling.
- Docstrings: the GENERATORS and FILTER paragraphs of `frames.py`; `thermal_displacements`
  gains the near-zero-mode warning with the two measured molecules.
- `t_frames`: the ceiling assertion becomes "the draw is taken as it comes, each frame with
  its own seed" plus a check that a tight ceiling still works when asked for explicitly.

**Consequence, stated:** for these two molecules the displaced frames will be drawn at
1.4-2.0 A RMS, the engine energy will be far above 275 kcal/mol, and all four per basin will
be DROPPED by the window -- the molecule gets a Frame set (basin / merged / saddle) and no
displaced frames, at the cost of one MACE finite-difference Hessian per dropped frame. Giving
such a molecule displaced frames again needs a separate ruling: skip the modes below ~50-100
cm^-1 in the draw (they are the conformational degrees of freedom CREST already enumerated as
separate basins, and they are the ones whose harmonic sigma is meaningless). Not built.

**Blocked by:** nothing. **Unblocks:** the 02 resubmission that completes draw300.

**Status:** implemented 2026-09-23; the tianhe item is the user's.

- [x] unit `t_frames`: no ceiling by default (MAX_RMS_A None, the Record nan, each frame's rms as drawn);
      an explicit 0.05 A ceiling still honoured and an unreachable one still raises after 50 draws
- [x] unit group green (58/58)
- [ ] tianhe (user): resubmit 02; the two molecules build a Frame set (displaced dropped by the window)
