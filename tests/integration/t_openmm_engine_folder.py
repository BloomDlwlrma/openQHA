"""The trajectory driver leaves exactly OpenMM's seven files in the engine folder (ticket 03).

INTEGRATION. Runs the real driver three times on the CPU platform with a protocol of a
few hundred steps on a shipped molecule; each run traces the MACE model, so it is a
minute or two, not a second. Skips with a message when OpenMM or MACE is absent.

The decision (ADR 0001, user 2026-09-14): `openmm/<setting>/basinNN/` under the molecule
directory holds `start.pdb system.xml integrator.xml traj.dcd state.csv state.xml
state.chk` and nothing else; the DCD and the CSV are written at the sampling interval so
frames and rows align; state XML and checkpoint at every flush; a partial trajectory is
resumed from its state and appended to; the driver's records go to `_records/`.

    run 1  prod 0.10 ps, frame every 10 fs  -> 10 frames, the seven files, nothing else
    run 2  prod 0.20 ps, same folder        -> resumes, 20 frames, the record says resumed
    run 3  prod 0.20 ps again               -> nothing to run; nothing rewritten

The DCD is read back with mdtraj, a reader the driver does not use, and its positions
must match the driver's own float64 frames to float32 precision.
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

DRIVER = ROOT / "scripts" / "production" / "s0_B_qha_trajectory_openmm.py"
QID, TAG, SETTING = "dsgdb9nsd_000018", "t03", "default"
FAIL = []


def check(label, ok, detail=""):
    print("  {:60s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:300]))
    if not ok:
        FAIL.append(label)


def run_driver(root, prod_ps, setting=SETTING):
    env = dict(os.environ, S0_RUNS_ROOT=str(root))
    cmd = [sys.executable, str(DRIVER), "--species", QID, "--tag", TAG, "--setting", setting,
           "--platform", "CPU", "--equil-ps", "0.02", "--prod-ps", str(prod_ps),
           "--sample-every", "10", "--timestep-fs", "1.0"]
    t0 = time.time()
    p = subprocess.run(cmd, env=env, cwd=str(ROOT), text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.returncode, p.stdout, time.time() - t0


def load_dcd(eng, setting="default"):
    """mdtraj's reading of the DCD, or None with the reason when a file is missing."""
    import mdtraj as md
    dcd = eng / layout.openmm_file_name("traj.dcd", setting)
    pdb = eng / layout.openmm_file_name("start.pdb", setting)
    if not (dcd.exists() and pdb.exists()):
        return None
    return md.load(str(dcd), top=str(pdb))


def csv_rows(path):
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return len([l for l in lines if not l.startswith("#")]), lines[0] if lines else ""


def main():
    try:
        import openmm, mace, mdtraj, numpy as np      # noqa: F401
    except Exception as exc:                          # noqa: BLE001
        print("SKIP: {}".format(exc))
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="openqha_t03_"))
    try:
        mol = layout.molecule_dir(tmp, TAG, QID)
        eng = layout.openmm_dir(mol, SETTING, 0)
        rec = layout.records_dir(mol)

        print("A. first run: 10 frames, the seven files, nothing else")
        rc, out, dt = run_driver(tmp, 0.10)
        check("driver exit 0 ({:.0f} s)".format(dt), rc == 0, out[-1500:])
        names = sorted(p.name for p in eng.iterdir()) if eng.is_dir() else []
        check("engine folder holds exactly the seven files", names == sorted(layout.OPENMM_FILES), names)
        n_rows, header = csv_rows(eng / "state.csv") if (eng / "state.csv").exists() else (0, "")
        check("state.csv has 10 rows", n_rows == 10, n_rows)
        check("state.csv header is OpenMM's StateDataReporter form",
              header.startswith('#"Step","Time (ps)"'), header)
        t = load_dcd(eng)
        check("traj.dcd has 10 frames (mdtraj)", t is not None and t.n_frames == 10,
              t.n_frames if t is not None else "no dcd/pdb")
        check("no seed level, no setting level: openmm/basin00",
              eng.parent.name == "openmm" and eng.name == "basin00")
        meta_p = rec / "openmm" / SETTING / "basin00" / "meta.json"
        check("record in _records/", meta_p.exists(), meta_p)
        check("no record or .npy in the engine folder",
              not any(n.endswith((".npy", ".json")) for n in names))
        frames_p = rec / "openmm" / SETTING / "basin00" / "frames.npy"
        if t is not None and frames_p.exists() and t.n_frames == 10:
            f64 = np.load(frames_p)
            d = np.abs(t.xyz * 10.0 - f64).max()
            check("DCD positions equal the float64 frames to float32 precision (max {:.1e} A)".format(d),
                  d < 1e-3, d)
        summ = rec / "openmm" / SETTING / "summary.json"
        check("summary.json in _records/", summ.exists(), summ)

        print("B. second run with a longer protocol resumes from the state and appends")
        rc, out, dt = run_driver(tmp, 0.20)
        check("driver exit 0 ({:.0f} s)".format(dt), rc == 0, out[-1500:])
        n_rows, _ = csv_rows(eng / "state.csv")
        t = load_dcd(eng)
        check("20 frames in traj.dcd", t is not None and t.n_frames == 20,
              t.n_frames if t is not None else "no dcd/pdb")
        check("20 rows in state.csv", n_rows == 20, n_rows)
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        prod = meta.get("production") or {}
        check("record says resumed, 10 already on disk, 10 generated",
              prod.get("resumed") is True and prod.get("n_frames_already_on_disk") == 10
              and prod.get("n_frames_generated_this_run") == 10, prod)
        check("resumed from the state, not re-equilibrated",
              str(prod.get("resumed_from", "")).startswith("state."), prod.get("resumed_from"))
        check("one velocity draw only (segments share the seed)",
              len({s["seed"] for s in meta.get("segments", [])}) == 1, meta.get("segments"))
        names = sorted(p.name for p in eng.iterdir())
        check("still exactly the seven files", names == sorted(layout.OPENMM_FILES), names)

        print("D. a second setting lives in the SAME folder under its own file names")
        rc, out, dt = run_driver(tmp, 0.05, setting="s2")
        check("driver exit 0 ({:.0f} s)".format(dt), rc == 0, out[-1500:])
        names = sorted(p.name for p in eng.iterdir())
        want = sorted(list(layout.OPENMM_FILES) + [layout.openmm_file_name(n, "s2") for n in layout.OPENMM_FILES])
        check("fourteen files: seven bare, seven with _s2", names == want, names)
        t2 = load_dcd(eng, "s2")
        check("traj_s2.dcd has 5 frames", t2 is not None and t2.n_frames == 5, t2.n_frames if t2 else None)
        t = load_dcd(eng)
        check("the default trajectory still has 20", t is not None and t.n_frames == 20, t.n_frames if t else None)
        check("records under _records/openmm/s2/", (rec / "openmm" / "s2" / "basin00" / "meta.json").exists())

        print("C. third run over a finished basin does nothing")
        mtimes = {p.name: p.stat().st_mtime_ns for p in eng.iterdir()}
        rc, out, dt = run_driver(tmp, 0.20)
        check("driver exit 0 ({:.0f} s)".format(dt), rc == 0, out[-800:])
        check("says nothing to run", "nothing to run" in out, out[-400:])
        check("engine files untouched", all(p.stat().st_mtime_ns == mtimes[p.name] for p in eng.iterdir()))
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n{}".format("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL)))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
