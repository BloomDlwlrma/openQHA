# 29: No engine Hessian at a displaced frame -- energy and forces only, `LOWEST_FREQ` blank there (`data/frames.py`, `02_frames.py`, `t_frames.py`)

**Why (user ruling 2026-09-23):** 02 costs ~25 min/molecule on draw300 and the engine
Hessian is the whole bill. Measured single-threaded on the workstation (one worker's share):
energy + forces 0.21 s, **analytic Hessian 13.4 s** at 19 atoms (3.7 s at 10 -- the cost is
3N backward passes). A molecule of the draw carries 30.6 frames, of which ~22 are displaced
(4 per basin, ~5.5 basins), so

    today   30.6 x 13.6 s = 416 s/molecule   -> 747 core-h for 6,458 molecules
    after   8.6 x 13.6 + 22 x 0.21 = 122 s   -> 219 core-h        (3.4x cheaper)

and nothing downstream needs it. The consumers of a displaced frame's ENGINE Hessian, read
off the code: `frames.consider` computes `LOWEST_FREQ` from it (one diagnostic column of the
Frame set Record), `_write_frames` puts it in `displaced.<mace level>.extxyz`, and
`dataset._frames_of` carries it into the `pool` split (the not-yet-labelled work list).
Nobody else: a labelled frame enters the Dataset as its REFERENCE label (`dataset.py:617`,
`atoms = lab`), a displaced frame has no reference Hessian (round 5, Q7 (b): `EnGrad` only)
and so reaches the judge with `has_hessian=False`, where the Hessian rows filter it out
(`judge.py:446`); training predicts its own Hessian (`phl_loss`); `hessian_compare`,
`mode_curvature` and `smoke_fit` all need a reference Hessian. `frame_labels.load_frame`
reads the MACE file for the GEOMETRY only.

**What to change:**

- `frames.generate`: the displaced frames are considered with `engine_efh(..., hessian=False)`
  -- energy and forces only. `consider` accepts `hessian=None` for a frame that has none:
  `LOWEST_FREQ` is `nan` there, `_write_frames` already writes `has_hessian=False` and no
  `hessian` key.
- `engine_efh(atoms, calc, want_hessian=True)`.
- The Record: `N_ENGINE_HESSIAN` beside `N_FRAMES` (how many kept frames carry one), and the
  `LOWEST_FREQ` schema line says it is blank where the frame has no engine Hessian.
- `02_frames.py --displaced-hessian`: off by default, on to rebuild a molecule with them (the
  seven pinned molecules, if a basin -> displaced curvature comparison at the engine level is
  ever wanted -- the reference-level one is impossible since Q7 (b)).
- Docstrings: the GENERATORS paragraph, the report note.

**Blocked by:** nothing. **Unblocks:** the `FORCE=1` rebuild at 1/3.4 of the cost -- and it
must land BEFORE that rebuild, or the rebuild pays the old bill.

**Status:** implemented 2026-09-23; the tianhe rebuild is the user's.

- [x] unit `t_frames`: displaced frames have no `hessian` key and `has_hessian=False`, their
      `LOWEST_FREQ` is nan, basin / merged / saddle still carry one; `--displaced-hessian`
      restores them; `DISPLACED_HESSIAN` and `N_ENGINE_HESSIAN` in the Record follow
- [x] unit group green (59/59). Two fixtures had been building their FAKE reference labels out
      of the engine Hessian of a displaced frame (`t_dataset.fake_labels`) and asserting every
      indexed frame carries one; both now say what they mean -- a label Hessian where the label
      has one, `has_hessian` false where it does not
- [ ] tianhe (user): the rebuild; 02's per-molecule seconds drop by ~3x
