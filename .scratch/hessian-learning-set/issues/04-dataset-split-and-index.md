# 04: The Dataset: split, files, index (`openqha/data/dataset.py`, `01_select.py`, `04_dataset.py`)

**What to build:** `01_select.py`: the molecule list under a tag with branch A done, SPICE membership from `qm9_targets_membership.dat`, stratification keys (ring count, heteroatom pattern, heavy atoms), the 7 known molecules pinned to `test`; `select.{out,toml,dat}` under `<root>/<tag>/_datasets/<name>/`. `dataset.build(root, tag, name, level, valid_fraction=0.1, test_fraction=0.1, seed)`: reads every Frame set, assigns split (test = whole molecules incl. pinned, stratified; valid = by frame from the training molecules; train = rest; pool = frames without the reference level), writes `{train,valid,test,pool}.<level>.extxyz` (MACE-loader keys: `energy`, `forces`, `hessian`), `index.dat` (molecule, basin, generator, k, split, levels present, seed, params_sha256, orca version), `dataset.{out,toml}`; `--export openreact` writes `molecules-<name>.h5` in OpenREACT's layout (Eh, Eh/bohr^2). CONTEXT Dataset / Workflow entries verified against the files; `workflows/hessian_learning/README.md` + `run.sh` (steps 01-04 in order; 05/06 stubs that refuse with "round 2 open").

**Blocked by:** 02 (frames), 03 (labels; pool works without them).

**Delivers:** the first Dataset from the smoke set; the workflow folder.

- [ ] two fake molecules with frames at two levels: valid frames come only from train molecules, test holds whole molecules, pool = frames lacking the level; fractions honoured; the split never changes on rebuild with the same seed
- [ ] `index.dat` round-trips through `dat.read_table`; every extxyz frame is findable from its row
- [ ] `--export openreact` writes a group per molecule with `coordinates/energies/forces/hessian/species` in Eh and Eh/bohr^2; read back equals the extxyz to 1e-10
- [ ] `run.sh --tag rings --limit 7` runs 01 -> 02 -> 04 locally (03 skipped) and produces a Dataset with a non-empty pool
