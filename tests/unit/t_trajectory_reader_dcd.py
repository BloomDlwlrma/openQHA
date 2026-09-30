"""One reader turns an OpenMM engine folder back into positions, symbols, masses.

UNIT. Writes a three-frame engine folder through `openmm_files.EngineFolder`
with known float64 positions, then reads it back through
`openqha.quasi_harmonic.trajectory_reader`; needs openmm and mdtraj, skips otherwise;
seconds.

collect, the ensemble report and 02d read the
trajectory from `traj.dcd` + `start.pdb` + `state.csv`, never from `frames.npy`. So:

    A. positions read back equal the ones written to float32 precision (DCD is float32);
       shape (n_frames, N, 3), float64, in angstrom
    B. symbols and masses come from start.pdb's elements, in the engine's atom order
    C. the per-frame table has one row per frame and the CSV's five columns
    D. a folder without traj.dcd, or without start.pdb, is refused with the folder named
    E. the records folder's md.toml rides along when it exists, and is None when not
"""
import json
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
FAIL = []


def check(label, ok, detail=""):
    print("  {:60s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


def main():
    try:
        import numpy as np
        import openmm  # noqa: F401
        import mdtraj  # noqa: F401
        from openqha.quasi_harmonic import openmm_files, trajectory_reader as tr
    except Exception as exc:                          # noqa: BLE001
        print("SKIP: {}".format(exc))
        return 0

    numbers = [8, 1, 1, 6]                            # O H H C: symbols by element
    rng = np.random.RandomState(3)
    frames = 2.0 + 0.3 * rng.standard_normal((3, 4, 3))
    with tempfile.TemporaryDirectory() as t:
        eng = Path(t) / "md_openmm" / "basin00"       # <molecule>/md_openmm/basinNN: the route is the parent name
        rec = Path(t) / "_records" / "md_openmm" / "default" / "basin00"
        top = openmm_files.topology_for(numbers)
        f = openmm_files.EngineFolder(eng, top, frame_spacing_ps=0.01, setting="s1")
        openmm_files.write_start_pdb(f.paths["start.pdb"], top, frames[0])
        check("a non-default setting writes traj_s1.dcd beside the folder's other settings",
              f.paths["traj.dcd"].name == "traj_s1.dcd", f.paths["traj.dcd"].name)
        f.open_frames(append=False)
        for i in range(3):
            f.write_frame(frames[i], step=10 * (i + 1), time_ps=0.01 * (i + 1),
                          potential_kJ=-1.0 * i, kinetic_kJ=2.0, temperature_K=300.0 + i)
        f.close_frames()

        print("A/B/C. read back:")
        r = tr.read_trajectory(eng, setting="s1")
        pos = r["positions_A"]
        check("shape (3, 4, 3) float64", pos.shape == (3, 4, 3) and pos.dtype == np.float64, (pos.shape, pos.dtype))
        d = float(np.abs(pos - frames).max())
        check("positions equal to float32 precision (max {:.1e} A)".format(d), d < 1e-4, d)
        check("symbols from start.pdb in atom order", r["symbols"] == ["O", "H", "H", "C"], r["symbols"])
        m = r["masses_amu"]
        check("masses from the elements (O ~16, H ~1, C ~12)",
              abs(m[0] - 16.0) < 0.1 and abs(m[1] - 1.0) < 0.1 and abs(m[3] - 12.0) < 0.1, m)
        tab = r["table"]
        check("table: 3 rows, five columns", len(tab["step"]) == 3 and set(tab) >= {
            "step", "time_ps", "potential_kJ", "kinetic_kJ", "temperature_K"}, list(tab))
        check("table values are the ones written", list(tab["step"]) == [10, 20, 30]
              and abs(tab["temperature_K"][2] - 302.0) < 1e-9, tab)
        check("frame spacing read from the CSV (0.01 ps)", abs(r["frame_spacing_ps"] - 0.01) < 1e-9,
              r["frame_spacing_ps"])
        check("n_frames = 3", r["n_frames"] == 3, r["n_frames"])

        print("E. the record rides along:")
        check("no records folder -> meta None", r["meta"] is None, r["meta"])
        rec.mkdir(parents=True)
        from openqha.quasi_harmonic import md_record
        md_record.write(dict(seed=5, symbols=["O", "H", "H", "C"], masses_amu=[16.0, 1.0, 1.0, 12.0]), rec, "s1")
        r2 = tr.read_trajectory(eng, records_dir=rec, setting="s1")
        check("md_s1.toml read when present", r2["meta"] is not None and r2["meta"]["seed"] == 5
              and r2["meta"]["symbols"] == ["O", "H", "H", "C"], r2["meta"])
        check("the setting is in the stem, not a folder", (rec / "md_s1.toml").is_file() and not (rec / "s1").exists())

        print("D. refusals name the folder:")
        for missing in ("traj.dcd", "start.pdb"):
            bad = Path(t) / "md_openmm" / ("bad_" + missing)
            bad.mkdir()
            for n in ("traj.dcd", "start.pdb", "state.csv"):
                if n != missing:
                    (bad / n).write_bytes(f.paths[n].read_bytes())
            try:
                tr.read_trajectory(bad)                 # default setting: bare names
                check("missing {} refused".format(missing), False, "no error")
            except FileNotFoundError as exc:
                check("missing {} refused, folder named".format(missing),
                      str(bad) in str(exc) and missing in str(exc), str(exc))

    print("\n{}".format("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL)))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
