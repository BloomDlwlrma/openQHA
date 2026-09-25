"""The census's one second optimisation pass really appends (ticket 38; closed by ticket 40).

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
    born (ADR 0001).

The pair: the fixture's own MACE basin (`mace/basin00/basin.extxyz`, residual 6.7e-5 eV/A)
certifies without a step, so the census returns a record; the same geometry displaced by a
fixed +-0.05 A pattern (seed 7) is the frame that fails twice and is rejected.
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

    print("\n{}".format("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL)))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
