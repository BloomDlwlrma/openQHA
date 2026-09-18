# 02: The Frame set of one molecule at the MACE level (`openqha/data/frames.py`, `02_frames.py`)

**What to build:** `frames.generate(molecule, n_displaced=4, temperature_K=298.15, max_rms_A=0.15, engine_name=None)`: per basin, `basin` (1), `displaced` (n, `hessian.thermal_displacements` with `hessian=` the stored MACE Hessian, seed = hash(qm9_index, basin, generator, k)), and from branch A's merge map / CREST engine files `merged` and `saddle` (one each); MACE E, F, H at every frame (`get_hessian`, eV/A^2 flattened); filter |F|max > 10 eV/A and bond-graph change vs the basin's SMILES (RDKit) -> dropped, counted; writes `<molecule>/frames/<generator>.<engine level>.extxyz` (keys `qm9_index`, `basin`, `generator`, `k`, `seed`, `level`, `energy`, `forces`, `hessian`, `smiles`) and the Record `frames/frames.{out,toml}` (counts, dropped with reason, seeds, `provenance()` incl. `params_sha256`). Driver `workflows/hessian_learning/02_frames.py --tag --species|--all`. `thermal_displacements` gains `hessian=` and returns the per-sample seed.

**Blocked by:** 01 (the fingerprint in the Record).

**Status:** done 2026-09-18

**Delivers:** the frames of the smoke set on disk; the propanal fixture's frames.

- [x] propanal fixture: 3 basins -> 3 basin + 12 displaced (+ merged if any) frames, RMS <= 0.15 A, keys and seeds present; same seed -> same positions to 1e-12
- [x] a frame with a broken bond graph is dropped and counted in the Record; |F|max filter likewise
- [x] the basin frame's Hessian equals `hessian_at_<level>.npy` to 0 (integration, engine)
- [x] CONTEXT.md Frame / Frame set entries match what the code writes (keys, folder)

**Closing:** `openqha/data/frames.py` (`generate`, `frame_seed`, `bond_change`, `read_frames`, SCHEMA), `layout.frames_dir / frames_file / orca_frame_dir`, `hessian.thermal_displacements(hessian=, seeds=)` (per-sample generators), branch-A Property `DUPLICATE_MAP` + `SADDLE_CONFORMER_IDS` (new records; old ones give no merged/saddle frames and say so), `workflows/hessian_learning/02_frames.py` (`--species | --all`, skip-if-Record, `--force`). Frame sets built with the real engine: propanal 3 + 12, oxetane 1 + 4, 2-methyloxirane 1 + 4, cyclopropanol 2 + 8 (no merged/saddle: those runs predate the map). `tests/unit/t_frames.py` (harmonic surrogate, 11 checks) and `tests/integration/t_frames_engine.py` (basin frame = hessian.npy to 0; displaced file H vs engine 2.5e-6 eV/A^2 from the file's 8-decimal positions).

**Two things measured that the rulings did not foresee:** (1) the bond-graph filter at a single 1.2 x covalent cutoff dropped 2 of 12 propanal frames for a 0.2 A C-H stretch at 0.12 A RMS -- replaced by an asymmetric test (broken > 1.35 x, formed < 0.95 x); (2) the harmonic QUANTUM draw at 298 K puts the zero-point energy into every stretch: displaced frames sit 12-42 kcal/mol above their basins at 0.08-0.15 A RMS (propanal mean 30, rings 12-40; Rodriguez's NMS test frames: mean 39). A classical draw (kT/omega^2) would give ~0.6 kcal/mol per mode. Recorded per frame as `ENERGY_ABOVE_BASIN`; which distribution the training should see is a round-2 question (added there as Q14).

**Amendment (user ruling 2026-09-18, after the literature check):** no published set (ANI-1, SPICE, OpenREACT, HORM/Transition1x, PFT) applies a bond-graph test to displaced frames; ANI-1 and SPICE filter by an energy window. openQHA does the same: `ENERGY_WINDOW_KCAL = 275` (ANI-1's) is the only filter; the engine |F|max filter is removed; `bond_change` (RDKit 1.3 / Open Babel ~1.4 perception tolerances) is still evaluated and written per frame as `BOND_CHANGE`, with `N_BOND_CHANGED` in the Record, as information. Frame sets rebuilt for the four molecules (0 dropped, 0 bond changes). Note `notes_2026-09-18_3_frame-filters-in-the-literature.md`.
