"""The ASE-route driver leaves exactly ASE's three files in the engine folder (ticket 10).

INTEGRATION. Runs the real ASE driver (MACE on CPU) twice on a shipped molecule with a
protocol of a few hundred steps, then collect on the result; a minute or two. Skips when
MACE is absent.

The decision (ADR 0001; ticket 10, 2026-09-15): `md_ase/basinNN/` holds `start.extxyz`,
`md.traj` (ASE Trajectory) and `md.log` (ASE MDLogger) and nothing else; a partial
trajectory resumes from the last frame of `md.traj`; the driver's records go to
`_records/md_ase/<setting>/basinNN/`; the same reader reads it and collect finds it with
`--route auto`.

    run 1  prod 0.10 ps, frame every 10 fs  -> 10 frames, the three files, nothing else
    run 2  prod 0.20 ps, same folder        -> resumes, 20 frames, record says resumed
    run 3  prod 0.20 ps again               -> nothing to run
    then   s0_B_qha_analyse.py --route auto -> finds ase/, writes _records/md_ase/default/collect__*.parquet
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from openqha.store import layout   # noqa: E402

DRIVER = ROOT / "scripts" / "production" / "s0_B_qha_trajectory.py"
ANALYSE = ROOT / "scripts" / "production" / "s0_B_qha_analyse.py"
QID, TAG = "dsgdb9nsd_000018", "t10"
FAIL = []


def check(label, ok, detail=""):
    print("  {:62s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:300]))
    if not ok:
        FAIL.append(label)


def run(cmd, root):
    env = dict(os.environ, S0_RUNS_ROOT=str(root), OPENQHA_SMOKE="1")
    t0 = time.time()
    p = subprocess.run(cmd, env=env, cwd=str(ROOT), text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.returncode, p.stdout, time.time() - t0


def run_driver(root, prod_ps):
    return run([sys.executable, str(DRIVER), "--species", QID, "--tag", TAG, "--seeds", "1",
                "--equil-ps", "0.02", "--prod-ps", str(prod_ps), "--sample-every", "10",
                "--device", "cpu"], root)


def main():
    try:
        import mace  # noqa: F401
        import numpy as np
        from ase.io import read
        from openqha.quasi_harmonic import trajectory_reader
    except Exception as exc:                          # noqa: BLE001
        print("SKIP: {}".format(exc))
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="openqha_t10_"))
    try:
        mol = layout.molecule_dir(tmp, TAG, QID)
        eng = layout.ase_dir(mol, "default", 0)
        rec = layout.records_for(mol, "ase", "default") / "basin00"

        print("A. first run: the three files, 10 frames")
        rc, out, dt = run_driver(tmp, 0.10)
        check("driver exit 0 ({:.0f} s)".format(dt), rc == 0, out[-1500:])
        names = sorted(p.name for p in eng.iterdir()) if eng.is_dir() else []
        check("md_ase/basin00 holds exactly start.extxyz md.traj md.log",
              names == sorted(layout.ASE_FILES), names)
        n_traj = len(read(str(eng / "md.traj"), index=":")) if (eng / "md.traj").exists() else 0
        check("md.traj has 10 frames (ase.io.read)", n_traj == 10, n_traj)
        rows = [l for l in (eng / "md.log").read_text().splitlines() if l.strip() and not l.startswith("Time")] \
            if (eng / "md.log").exists() else []
        check("md.log has 10 rows", len(rows) == 10, len(rows))
        check("start.extxyz carries energy and forces",
              (eng / "start.extxyz").exists() and read(str(eng / "start.extxyz")).calc is not None)
        check("records in _records/md_ase/default/basin00 (meta, progress, frames.npy)",
              all((rec / n).exists() for n in ("meta.json", "progress.json", "frames.npy")),
              sorted(p.name for p in rec.iterdir()) if rec.is_dir() else None)
        check("no state.npz anywhere", not list(mol.rglob("state.npz")))
        r = trajectory_reader.read_trajectory(eng, records_dir=rec, setting="default")
        if (rec / "frames.npy").exists() and n_traj == 10:
            d = float(np.abs(r["positions_A"] - np.load(rec / "frames.npy")).max())
            check("reader's positions equal the driver's frames (max {:.1e} A)".format(d), d < 1e-6, d)
        check("reader table: 10 rows, temperature column", len(r["table"]["temperature_K"]) == 10, r["table"])
        check("reader route = ase", r.get("route") == "ase", r.get("route"))

        print("B. second run resumes from the last md.traj frame and appends")
        rc, out, dt = run_driver(tmp, 0.20)
        check("driver exit 0 ({:.0f} s)".format(dt), rc == 0, out[-1500:])
        n_traj = len(read(str(eng / "md.traj"), index=":"))
        check("20 frames in md.traj", n_traj == 20, n_traj)
        meta = json.loads((rec / "meta.json").read_text(encoding="utf-8"))
        prod = meta.get("production") or {}
        check("record says resumed with 10 already on disk", prod.get("resumed") is True
              and prod.get("n_frames_already_on_disk") == 10, prod)
        check("resumed from md.traj", prod.get("resumed_from") == "md.traj", prod.get("resumed_from"))
        names = sorted(p.name for p in eng.iterdir())
        check("still exactly the three files", names == sorted(layout.ASE_FILES), names)

        print("C. third run does nothing")
        mtimes = {p.name: p.stat().st_mtime_ns for p in eng.iterdir()}
        rc, out, dt = run_driver(tmp, 0.20)
        check("exit 0 and engine files untouched", rc == 0
              and all(p.stat().st_mtime_ns == mtimes[p.name] for p in eng.iterdir()), out[-600:])

        print("D. collect finds the ase route")
        rc, out, dt = run([sys.executable, str(ANALYSE), "--species", QID, "--tag", TAG,
                           "--route", "auto", "--no-gmx"], tmp)
        check("analyse ran (exit {} is a verdict, not a crash)".format(rc),
              "[PASS]" in out or "[FAIL]" in out, out[-1500:])
        tab = layout.records_for(mol, "ase", "default") / "collect__trajectories.parquet"
        check("collect tables under _records/md_ase/default/", tab.exists(), tab)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n{}".format("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL)))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
