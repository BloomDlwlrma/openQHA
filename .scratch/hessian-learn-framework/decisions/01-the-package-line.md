# The package line: what moves, what stays, and what the package is called

Type: grilling
Status: resolved
Part of: [hessian-learn-framework](../map.md)

## Question

Q2 of the split ruled that the training side moves out of openQHA into `openQHA-Hessian` and openQHA imports it. Settle the boundary and the names before anything moves (the move itself is [The move](07-the-move.md)).

Decide:

1. **What moves.** Candidates: `openqha/training/{__init__,hvp,phl,phl_loss,judge,run,smoke_fit}.py`; the drivers `workflows/hessian_learning/{05_train,06_judge}.py`; `hpc/slurm/hl_train.slurm`; the training tests (`tests/unit/{t_hvp,t_phl,t_phl_loss,t_judge,t_smoke_fit,t_train_run,t_probe_calibration}.py`, `tests/integration/{t_hvp_engine,t_judge_engine,t_train_engine}.py`); `docs/hessian_learning_campaign.md`; tutorial cells. What stays in openQHA, and where the line is drawn *inside* `openqha/training/`.
2. **The import direction.** Only `hvp.py`, `phl.py`, `__init__.py` are clean today (no openQHA imports); `phl_loss.py` needs `hvp`/`phl` only. `judge.py`, `run.py`, `smoke_fit.py` import `..data` (`dataset`), `..store` (`dat`, `layout`, `property`, `report`), `..thermochem` (`hessian`, `hessian_compare`) and `..potentials` (`engine`, `mace_patch`). Moving those out can mean: (a) the package depends on `openqha` while openQHA depends on the package — a cycle to design away; (b) they stay in openQHA and only the clean core moves; (c) openQHA passes what they need through an interface the package owns. Pick one; say how any cycle is broken.
3. **Package identity.** Distribution name and import name (e.g. `openqha-hessian` / `openqha_hessian`), versioning, layout (`pyproject.toml`, one top-level package), console scripts or none, and what `--loss_module openqha.training.phl_loss:build` becomes (that string is printed into Records and Slurm scripts).
4. **Compatibility.** `docs/tutorials/T04`/`T05` import `openqha.training`; `openqha/__init__.py` lists `training` in its subpackage map; old Records name the old module string. Shim (`openqha.training` re-exporting from the package) or update everything?
5. **Test ownership.** Where the moved tests live, and how openQHA's `tests/run_tests.py` groups and the package's suite relate (does openQHA still run them; is there a twin command).

## Facts to stand on

- Import map of `openqha/training/` (verified read-only): as in point 2; `judge` also lazy-imports `..potentials.engine` and `mace`; `run` lazy-imports `..data.dataset`, `smoke_fit`, `..potentials.mace_patch`.
- `openqha/training/__init__.py` docstring: the mace-md pattern — public API here, mace internals in the fork; `openqha.extensions` explicitly not involved.
- Of `workflows/hessian_learning/`, only `05_train.py` (line 57) and `06_judge.py` (line 54) import `openqha.training`.

## Answer (2026-09-26, grilling; user-ruled)

The boundary sits at the directory: **all seven modules of `openqha/training/` move; nothing is drawn inside it, and nothing of it stays.** The consumers stay in openQHA and change addresses only.

1. **What moves.** `openqha/training/{__init__,hvp,phl,phl_loss,judge,run,smoke_fit}.py` (2,662 lines) enter the package under the same module names; `__init__.py` stays docstring-only (rewritten for the package). The nine tests whose subject is that code move too: `t_hvp`, `t_phl`, `t_phl_loss`, `t_judge`, `t_smoke_fit`, `t_train_run` (unit) and `t_hvp_engine`, `t_judge_engine`, `t_train_engine` (integration). Everything else stays: the drivers `05_train.py`/`06_judge.py`, `hl_train.slurm`, the `s0_*.py` scripts that reach into training, `docs/hessian_learning_campaign.md` and the T04/T05 cells are openQHA's campaign surface and only get their addresses rewritten (07/12). `t_probe_calibration` stays as well — its subject is the staying `scripts/tooling/s0_probe_calibration.py`; so do `t_spice_pt_draw` and the `t_dataset*` tests, whose in-function training imports are rewired.

2. **The import direction: package → openqha, never the reverse.** `openqha_hessian` imports `openqha.{data,store,thermochem,potentials}` and the mace fork; openQHA's consumer layer (workflows, scripts, the Slurm gate, tutorials, staying tests) imports the package; **the `openqha` library itself never imports it** (today zero imports outside `openqha/training/`, comments only — the graph stays acyclic: `openqha` ← `openqha_hessian` ← consumers). Interface inversion (the package owning a data/record/calculator seam) was rejected: the package's semantics *are* openQHA's — molecule trees, Record formats, engine registry — so the seam would be a second openQHA bought to fix a cycle that does not exist. Handed to [Install and transport](05-install-and-transport.md): the package needs `openqha` importable, and openQHA has no packaging files today (checkout + `sys.path`); 05 owes that answer, and the working state is three things — openQHA checkout, package, fork.

3. **Identity.** Distribution `openqha-hessian`, import `openqha_hessian`, version from `0.1.0` (the fork's `0.3.16+openqha` is mace's version and stays with the fork). One top-level package, flat layout, `pyproject.toml` (setuptools, the fork's stack), `tests/`, `install.sh`, rewritten README; **no console scripts** — the entry points stay `workflows/hessian_learning/05_train.py`/`06_judge.py` in openQHA. The `--loss_module` address becomes `openqha_hessian.phl_loss:build`; it is the only mace-argv change. Old Records keep the old string untouched (data; nothing resolves it — the fork's `load_external_loss` reads the CLI argument only); new Records carry the new one. Vocabulary: "openQHA-Hessian" means the package, the mace side is "the mace fork" (`BloomDlwlrma/mace`).

4. **Compatibility: no shim.** `openqha.training` ceases to exist as an import path; the in-repo references are swept by 07/12 and the `SUBPACKAGES` map and docstring line go with them. The `_MOVED` compatibility layer is not extended — its values are in-package subpackages and cannot express a departure; there is no out-of-repo consumer to keep alive (user ruling). Old tickets and notes get pointer notes, not rewrites (12).

5. **Test ownership.** The moved tests live in `openQHA-Hessian/tests/{unit,integration}/` and run under the package's twin of `tests/run_tests.py` — same conventions (groups `unit`/`integration`, one subprocess per file, `--group`/`--all`, unit by default) plus a provenance header (python, `openqha.__file__`, `openqha_hessian.__file__`, mace version). openQHA's runner no longer runs them and gains no forwarding flag; acceptance is two commands, one per repository. Fixtures are reused, not copied: the moved tests locate the openQHA checkout via `Path(openqha.__file__).resolve().parents[1]` (`OPENQHA_SRC` overrides) and keep reading `tests/data/{propanal_molecule,methyloxirane_frames,spice_tiny}`; the `_repo_root()` walk (which looks for `openqha/__init__.py`) is replaced. Caveat for 05: a wheel-only openQHA would need this revisited.

The durable record is [ADR 0010](../../../docs/adr/0010-hessian-learning-lives-in-openqha-hessian.md) (in force — the move landed as the slices 01b–01d; [The move](07-the-move.md) is resolved), and the `/to-spec` output is [spec-the-package-line.md](../spec-the-package-line.md). **Amended 2026-09-29:** the spec is re-homed from `implementation/01a` to the effort root in the `spec-<slug>` form, like [spec-install-and-transport.md](../spec-install-and-transport.md); [The repo swap](11-repo-swap.md) and [References sweep](12-references-sweep.md) carry the remainder.
