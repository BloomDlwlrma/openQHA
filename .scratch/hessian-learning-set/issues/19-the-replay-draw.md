# 19: The Replay draw -- `s0_spice_pt_draw.py`, one seed, one file, two assertions that fail

**What to build:** the tool that makes the Replay file nobody writes today (`README` and `05_train.py` name `spice_pt_5000.extxyz`; only the judge's test-split draw exists). From SPICE's TRAIN split (`training_set.settings()`, `train_large_neut_no_bad_clean.xyz`, 951,005 frames, 17,132 molecules) it draws `--n` frames uniformly BY FRAME from one seeded permutation (so a smaller `--n` is a prefix of a larger one and the scan rows R0-R4 are nested), writes every frame with `config_weight = --weight` (default 1.0), writes an ids file (source file, frame index, SMILES, and a per-molecule frame count so the coverage of the draw is a number), and a Record with DOI, split, seed, n, weight, distinct molecules. Two assertions, each of which makes the tool exit non-zero and write nothing: (a) no molecule (by SMILES at the index's canonical level) is shared with the forgetting draw's ids file (`spice_test_<n>.ids.dat`); (b) none of the in_distribution molecules (`judge.IN_DISTRIBUTION`, the four shipped) is present. When the release is absent it refuses with the DOI, as the test-split tool does. Rulings: S0-C-56, CONTEXT "Replay".

**Blocked by:** None. **Unblocks:** 16.

**Status:** ready-for-agent

- [ ] `scripts/tooling/s0_spice_pt_draw.py --n N --seed S --weight W [--source ...] [--forgetting-ids ...] [--out ...]`: the permutation is `default_rng(seed).permutation(n_frames)`, the file holds the first N; `config_weight` on every frame; ids file beside it; Record `[Replay]` with `DOI, SPLIT=train, SEED, N, WEIGHT, N_MOLECULES, FRAMES_PER_MOLECULE_MAX`
- [ ] assertion (a): SMILES overlap with the forgetting ids file -> exit 2, message names the offending molecules, no file written; assertion (b): any of the four in_distribution molecules -> exit 2 likewise; both exercised on `tests/data/spice_tiny` with a constructed overlap
- [ ] nesting: `--n 3` from seed 0 is the first three frames of `--n 5` from seed 0 (unit)
- [ ] `05_train.py` and `README` name this tool as the only source of a Replay file; the ratio the file means on a given Dataset is printed by `run.py` (ticket 18), not here
- [ ] unit (`tests/unit/t_spice_pt_draw.py`): the four items above on `spice_tiny`; absence of the release refuses with the DOI
- [ ] `.mem/notes` round note
