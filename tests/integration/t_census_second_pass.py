"""The census's one second optimisation pass really appends.

INTEGRATION. Runs `census_from_frames` with the real MACE calculator on two prepared
frames of the propanal fixture (no CREST binary involved); tens of seconds. Skips with a
message when MACE is absent.

**Why a forcing fixture.** Every shipped frame's tighten lands at or below ORCA's default
line (`TolMaxG = 3e-4 Eh/bohr = 1.543e-2 eV/A`), so no shipped fixture ever triggers the
second pass and its engine-file path -- `Trajectory(..., mode="a")` on the conformer's own
`opt.traj`, the same `opt.log` -- had no end-to-end run until here. With
`max_opt_steps = 1` the displaced frame's first pass stops at a residual of ~8 eV/A, far
above the line; the pass runs (once), the frame still fails, and is rejected as
`not_certified`. The assertions read what the two passes left on disk:

  * `opt.log` carries BOTH runs: two `Step ... fmax` headers, and LBFGS step numbers
    `0..n` restarting once (ASE logs the initial state and every step, per run);
  * `opt.traj` still holds the first pass's frames -- the file's FIRST frame is the input
    geometry and the count is n1 + n2 + 1. ASE writes a run's initial state only into an
    EMPTY trajectory (`Optimizer.irun`, `_traj_is_empty()`), so the appended second pass
    contributes exactly its steps;
  * `conf.extxyz` is the final geometry (the trajectory's last frame) and carries the
    same residual the record rejected the frame with;
  * `mace/confNN/` still holds exactly `layout.MACE_CONFORMER_FILES` -- no `opt2.*` was
    born.

The pair: the fixture's own MACE basin (`mace/basin00/basin.extxyz`, residual 6.7e-5 eV/A)
certifies without a step, so the census returns a record; the same geometry displaced by a
fixed +-0.05 A pattern (seed 7) is the frame that fails twice and is rejected.

**The `converged_orca_default` section.** It was the one certification
class with no real-calculator end-to-end run. The same displaced frame with a larger cap
lands inside ORCA's line instead: measured on this fixture (real MACE-OFF23_medium), the
first-pass residual against the cap is 1 -> 8.438, 2 -> 3.577, 4 -> 0.335, 12 -> 0.048,
24 -> 0.030, 40 -> 0.0144, 48 -> **0.00143**, 50 -> 5.65e-4, 60 -> 6.56e-4, 70 ->
8.85e-5 (converged; the optimiser stops at 69 steps). Cap 48 sits near the geometric
middle of (1e-4, 1.543e-2] with ~14x margin above the target and ~11x below the line, so
the class does not hinge on machine noise. The frame is admitted with
`tighten_converged = false` and no second pass runs; the census counts the class, and the
run's engine files carry the single pass (one log header, LBFGS 0..48 once).
"""
import sys
import tempfile
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
QID = "dsgdb9nsd_000035"
FIX = ROOT / "tests" / "data" / "propanal_molecule" / "mace" / "basin00" / "basin.extxyz"
FAIL = []


def check(label, ok, detail=""):
    print("  {:64s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:300]))
    if not ok:
        FAIL.append(label)


def main():
    try:
        import numpy as np
        from ase.io import Trajectory, read
        from openqha import engine
        from openqha.conformer_search import crest_census
        from openqha.store import layout
        calc, _name, _prov = engine.calculator(device="cpu")
    except Exception as exc:                          # noqa: BLE001
        print("SKIP: {}".format(exc))
        return 0

    ref = read(str(FIX), format="extxyz")
    displaced = read(str(FIX), format="extxyz")
    rng = np.random.RandomState(7)
    displaced.positions = displaced.positions + 0.05 * rng.choice(
        [-1.0, 1.0], size=displaced.positions.shape)
    input_pos = displaced.positions.copy()

    tmp = Path(tempfile.mkdtemp(prefix="openqha_t38sp_"))
    try:
        mol = layout.molecule_dir(tmp, "t40", QID)
        rec, basins, _m = crest_census.census_from_frames(
            "CCC=O", [ref, displaced], calc, name=QID, fmax=1e-4, threshold_A=0.30,
            temperature_K=298.15, do_hessian=True,
            comments=["fixture basin", "fixture basin, 0.05 A pattern"],
            hessian_mode="analytic", molecule_dir=mol, max_opt_steps=1)

        print("A. the second pass fired; the record names what failed twice")
        check("the rejected frame folded 2 steps (one per pass); the certified frame 0-1",
              rec["opt_steps_per_frame"][1] == 2 and rec["opt_steps_per_frame"][0] <= 1,
              rec["opt_steps_per_frame"])
        check("one certified basin, one not_certified, nothing in the other classes",
              rec["n_converged"] + rec["n_converged_orca_default"] == 1
              and rec["n_converged_second_pass"] == 0 and rec["n_not_certified"] == 1,
              {k: rec.get(k) for k in ("n_converged", "n_converged_orca_default",
                                       "n_converged_second_pass", "n_not_certified")})
        check("the rejected frame is conformer 1 with its FINAL residual, above ORCA's line",
              rec["not_certified"][0]["conformer_id"] == 1
              and rec["not_certified"][0]["residual_eV_A"]
              > crest_census.ORCA_DEFAULT_TOLMAXG_EV_A,
              rec["not_certified"])
        h0 = rec["hessian"][str(rec["basin_conformer_ids"][0])]
        check("the certified basin is the only one that survives, with its class recorded",
              rec["basin_conformer_ids"] == [0] and rec["n_basins"] == 1
              and h0["convergence_class"] in ("converged", "converged_orca_default")
              and h0["tighten_converged"] == (h0["convergence_class"] == "converged")
              and len(basins) == 1,
              (rec["basin_conformer_ids"], h0["convergence_class"]))

        print("B. both runs wrote the same opt.log")
        conf01 = mol / "mace" / "conf01"
        log = (conf01 / "opt.log").read_text(encoding="utf-8")
        headers = sum(1 for ln in log.splitlines()
                      if ln.split()[:2] == ["Step", "Time"])
        steps = [int(ln.split()[1]) for ln in log.splitlines() if ln.startswith("LBFGS:")]
        check("two 'Step ... fmax' headers -- one optimisation per pass",
              headers == 2, headers)
        check("LBFGS step numbers 0, n, 0, n -- the second pass ran its own steps",
              steps == [0, 1, 0, 1], steps)

        print("C. the trajectory kept the first pass (append, not truncation)")
        traj_path = conf01 / "opt.traj"
        tr = Trajectory(str(traj_path))
        first, last = tr[0], tr[-1]
        conf = read(str(conf01 / "conf.extxyz"), format="extxyz")
        n1, n2 = max(steps[:2]), max(steps[2:])
        check("frames == n1 + n2 + 1: the first pass's initial state and both steps survive",
              len(tr) == n1 + n2 + 1, len(tr))
        check("the file's FIRST frame is the input geometry (pass 1's start)",
              float(np.abs(first.positions - input_pos).max()) < 1e-8,
              float(np.abs(first.positions - input_pos).max()))
        check("the LAST frame is conf.extxyz, to the writer's precision",
              float(np.abs(last.positions - conf.positions).max()) < 1e-6,
              float(np.abs(last.positions - conf.positions).max()))
        check("conf.extxyz carries the residual the record rejected the frame with",
              abs(float(np.abs(conf.get_forces()).max())
                  - rec["not_certified"][0]["residual_eV_A"]) < 1e-6,
              (float(np.abs(conf.get_forces()).max()),
               rec["not_certified"][0]["residual_eV_A"]))

        print("D. no new engine-file name was born")
        names01 = sorted(p.name for p in conf01.iterdir())
        check("conf01 holds exactly opt.traj opt.log conf.extxyz",
              names01 == sorted(layout.MACE_CONFORMER_FILES), names01)
        names00 = sorted(p.name for p in (mol / "mace" / "conf00").iterdir())
        check("conf00 (no second pass) holds exactly the same three files",
              names00 == sorted(layout.MACE_CONFORMER_FILES), names00)
        extra = sorted(p.name for p in (mol / "mace").rglob("*") if "opt2" in p.name)
        check("no opt2.* anywhere under mace/", extra == [], extra)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    # E -- `converged_orca_default`: the cap chosen so the displaced
    # frame's FIRST-pass residual lands inside ORCA's line but above our target. The
    # frame is admitted with `tighten_converged = false`, no second pass runs, and the
    # run's engine files carry exactly the one pass.
    tmp_e = Path(tempfile.mkdtemp(prefix="openqha_t41_cod_"))
    try:
        mol_e = layout.molecule_dir(tmp_e, "t41", QID)
        rec_e, basins_e, _m_e = crest_census.census_from_frames(
            "CCC=O", [ref.copy(), displaced.copy()], calc, name=QID, fmax=1e-4,
            threshold_A=0.30, temperature_K=298.15, do_hessian=True,
            comments=["fixture basin", "fixture basin, 0.05 A pattern"],
            hessian_mode="analytic", molecule_dir=mol_e, max_opt_steps=48)

        print("E. a frame inside ORCA's line is admitted as converged_orca_default, no pass")
        check("one converged frame, one converged_orca_default, no pass, no rejection",
              rec_e["n_converged"] == 1 and rec_e["n_converged_orca_default"] == 1
              and rec_e["n_converged_second_pass"] == 0 and rec_e["n_not_certified"] == 0
              and rec_e["not_certified"] == [],
              {k: rec_e.get(k) for k in ("n_converged", "n_converged_orca_default",
                                         "n_converged_second_pass", "n_not_certified")})
        check("the final residual sits inside (fmax, ORCA's line] -- the class's window",
              1e-4 < rec_e["max_residual_force_eV_A"]
              <= crest_census.ORCA_DEFAULT_TOLMAXG_EV_A,
              rec_e["max_residual_force_eV_A"])
        check("the displaced frame consumed the whole cap: its first pass stopped inside "
              "the line, not at the target",
              rec_e["opt_steps_per_frame"][1] == 48
              and rec_e["opt_steps_per_frame"][0] <= 1,
              rec_e["opt_steps_per_frame"])

        conf_e = mol_e / "mace" / "conf01"
        log_e = (conf_e / "opt.log").read_text(encoding="utf-8")
        heads_e = sum(1 for ln in log_e.splitlines()
                      if ln.split()[:2] == ["Step", "Time"])
        steps_e = [int(ln.split()[1]) for ln in log_e.splitlines()
                   if ln.startswith("LBFGS:")]
        check("exactly one optimisation ran: one header, LBFGS 0..48 once",
              heads_e == 1 and steps_e == list(range(49)),
              (heads_e, steps_e[:3], steps_e[-3:]))
        tr_e = Trajectory(str(conf_e / "opt.traj"))
        check("the trajectory holds the one pass (initial state + 48 steps)",
              len(tr_e) == rec_e["opt_steps_per_frame"][1] + 1, len(tr_e))

        print("E2. the certified pair deduplicates; the survivor keeps its own class")
        check("two certified frames merge into one basin",
              rec_e["n_basins"] == 1 and len(basins_e) == 1
              and 1 in rec_e["duplicate_map"],
              (rec_e["n_basins"], rec_e["duplicate_map"]))
        h_e = rec_e["hessian"][str(rec_e["basin_conformer_ids"][0])]
        check("the surviving basin's class and tighten_converged answer to the two lines",
              h_e["convergence_class"] in ("converged", "converged_orca_default")
              and h_e["tighten_converged"]
              == (h_e["convergence_class"] == "converged"),
              (h_e["convergence_class"], h_e["tighten_converged"]))
    finally:
        import shutil as _sh
        _sh.rmtree(tmp_e, ignore_errors=True)

    print("\n{}".format("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL)))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
