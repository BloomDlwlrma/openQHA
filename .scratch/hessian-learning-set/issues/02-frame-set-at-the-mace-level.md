# 02: The Frame set of one molecule at the MACE level (`openqha/data/frames.py`, `02_frames.py`)

**What to build:** `frames.generate(molecule, n_displaced=4, temperature_K=298.15, max_rms_A=0.15, engine_name=None)`: per basin, `basin` (1), `displaced` (n, `hessian.thermal_displacements` with `hessian=` the stored MACE Hessian, seed = hash(qm9_index, basin, generator, k)), and from branch A's merge map / CREST engine files `merged` and `saddle` (one each); MACE E, F, H at every frame (`get_hessian`, eV/A^2 flattened); filter |F|max > 10 eV/A and bond-graph change vs the basin's SMILES (RDKit) -> dropped, counted; writes `<molecule>/frames/<generator>.<engine level>.extxyz` (keys `qm9_index`, `basin`, `generator`, `k`, `seed`, `level`, `energy`, `forces`, `hessian`, `smiles`) and the Record `frames/frames.{out,toml}` (counts, dropped with reason, seeds, `provenance()` incl. `params_sha256`). Driver `workflows/hessian_learning/02_frames.py --tag --species|--all`. `thermal_displacements` gains `hessian=` and returns the per-sample seed.

**Blocked by:** 01 (the fingerprint in the Record).

**Delivers:** the frames of the smoke set on disk; the propanal fixture's frames.

- [ ] propanal fixture: 3 basins -> 3 basin + 12 displaced (+ merged if any) frames, RMS <= 0.15 A, keys and seeds present; same seed -> same positions to 1e-12
- [ ] a frame with a broken bond graph is dropped and counted in the Record; |F|max filter likewise
- [ ] the basin frame's Hessian equals `hessian_at_<level>.npy` to 0 (integration, engine)
- [ ] CONTEXT.md Frame / Frame set entries match what the code writes (keys, folder)
