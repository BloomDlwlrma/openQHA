"""The ASE-route driver leaves exactly ASE's three files in the engine folder.

INTEGRATION. Runs the real ASE driver (MACE on CPU) twice on a shipped molecule with a
protocol of a few hundred steps, then collect on the result; a minute or two. Skips when
MACE is absent.

The layout: `md_ase/basinNN/` holds `start.extxyz`,
`md.traj` (ASE Trajectory) and `md.log` (ASE MDLogger) and nothing else; a partial
trajectory resumes from the last frame of `md.traj`; the driver's records go to
`_records/md_ase/basinNN/`; the same reader reads it and collect finds it with
`--route auto`.

    run 1  prod 0.10 ps, frame every 10 fs  -> 10 frames, the three files, nothing else
    run 2  prod 0.20 ps, same folder        -> resumes, 20 frames, record says resumed
    run 3  prod 0.20 ps again               -> nothing to run
    then   s0_B_qha_analyse.py --route auto -> finds md_ase/, leaves _records/md_ase/collect.{out,toml,dat}
           (one Table, three sections, every column explained)
    then   s0_B_report_ensemble.py --route ase -> ensemble.toml carries the verdict of collect.toml
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
REPORT = ROOT / "scripts" / "production" / "s0_B_report_ensemble.py"
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
        rec = layout.basin_records_dir(mol, "ase", 0)

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
        from openqha.store import report as _report, property as _prop
        from openqha.quasi_harmonic import md_record as _md
        rnames = sorted(p.name for p in rec.iterdir()) if rec.is_dir() else None
        check("records in _records/md_ase/basin00: exactly md.out and md.toml",
              rnames == ["md.out", "md.toml"], rnames)
        check("md.out ends with the terminal line", _report.terminated_normally(rec / "md.out", "md"))
        check("md.toml starts with the status block, STATUS NORMAL TERMINATION",
              (rec / "md.toml").read_text(encoding="utf-8").startswith("[Calculation_Status]")
              and _prop.status_of(rec / "md.toml") == "NORMAL TERMINATION")
        check("no directory named default under _records", not [p for p in (mol / "_records").rglob("default") if p.is_dir()])
        check("no state.npz, frames.npy or progress.json anywhere",
              not [p for p in mol.rglob("*") if p.name in ("state.npz", "frames.npy", "progress.json")])
        r = trajectory_reader.read_trajectory(eng, records_dir=rec, setting="default")
        check("reader gets the record from md.toml", r["meta"] is not None and r["meta"].get("route") == "ase",
              (r["meta"] or {}).get("route"))
        check("reader table: 10 rows, temperature column", len(r["table"]["temperature_K"]) == 10, r["table"])
        check("reader route = ase", r.get("route") == "ase", r.get("route"))

        print("B. second run resumes from the last md.traj frame and appends")
        rc, out, dt = run_driver(tmp, 0.20)
        check("driver exit 0 ({:.0f} s)".format(dt), rc == 0, out[-1500:])
        n_traj = len(read(str(eng / "md.traj"), index=":"))
        check("20 frames in md.traj", n_traj == 20, n_traj)
        meta = _md.read(rec) or {}
        prod = meta.get("production") or {}
        check("record says resumed with 10 already on disk", prod.get("resumed") is True
              and prod.get("n_frames_already_on_disk") == 10, prod)
        check("two segments (one per run)", len(meta.get("segments") or []) == 2, meta.get("segments"))
        check("md.out says resumed from md.traj", "md.traj" in (rec / "md.out").read_text(encoding="utf-8", errors="replace"))
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
        from openqha.quasi_harmonic import chain_records as _cr
        _cstem = _cr.stem(mol, "ase", "default", _cr.COLLECT)
        _cpaths = _cr.collect_paths(_cstem)
        check("the Table collect.dat under _records/md_ase/", _cpaths["dat"].exists(), _cpaths["dat"])
        check("no other .dat beside it (one Table)",
              sorted(p.name for p in _cstem.parent.glob("*.dat")) == ["collect.dat"],
              sorted(p.name for p in _cstem.parent.glob("*.dat")))
        _tables = _cr.read_collect_table(_cstem)
        check("the three sections in order, one row each (one basin, one seed)",
              list(_tables) == list(_cr.SECTIONS) and [len(_tables[k]) for k in _cr.SECTIONS] == [1, 1, 1],
              {k: len(v) for k, v in _tables.items()})
        _lines = _cpaths["dat"].read_text(encoding="utf-8").splitlines()
        _blocks, _cur = {}, None
        for _l in _lines:                        # the comment lines of each section, by section
            if _l.startswith("[") and _l.endswith("]"):
                _cur = _l[1:-1]
                _blocks[_cur] = []
            elif _cur and _l.startswith("#   "):
                _blocks[_cur].append(_l.split()[1])
        check("every column of every section has its comment line, in its own section",
              all(_blocks.get(s) == list(_cr.COLUMNS[s]) for s in _cr.SECTIONS), _blocks)
        from openqha.store import report as _rep
        check("collect.out ends with the terminal line (the marker)",
              _rep.terminated_normally(_cpaths["out"], "collect"))
        _out_text = _cpaths["out"].read_text(encoding="utf-8", errors="replace")
        check("each criterion sentence appears once in collect.out",
              _out_text.count("the saturation curve is drawn") == 1
              and _out_text.count("the symmetry number is read from the declaration") == 1)
        check("collect.toml beside it, STATUS NORMAL TERMINATION (the collect Batch's marker)", _cr.collect_done(_cstem))
        check("collect's files are in _records/md_ase/, no setting level",
              _cstem.parent == mol / "_records" / "md_ase" and not (mol / "_records" / "md_ase" / "default").exists())

        print("E. the ensemble reads the Table and the Property file")
        # This chain starts from the QM9 geometry, so there is no branch A Record; the
        # ensemble needs only [[Basin]] RELATIVE from branchA.toml, so a one-basin stand-in
        # is written here (the same shape branch_a_property reads).
        from openqha.store import property as _prop
        _prop.write(mol / "_records" / "branchA.toml",
                    {_prop.INFO_BLOCK: {"QM9_INDEX": QID}, "Basin": [{"INDEX": 0, "RELATIVE": 0.0}]},
                    {"Basin": {"RELATIVE": ("Double", "kcal/mol", "stand-in: one basin at zero")}},
                    status=_prop.NORMAL_TERMINATION, progname="openQHA branchA")
        rc, out, dt = run([sys.executable, str(REPORT), "--species", QID, "--tag", TAG, "--route", "ase"], tmp)
        _epaths = _cstem.parent / "ensemble.toml"
        check("ensemble ran and left ensemble.toml (exit {} is collect's verdict, not a crash)".format(rc),
              _epaths.is_file(), out[-1500:])
        _edoc = _prop.load(_epaths) if _epaths.is_file() else {}
        _cdoc = _prop.load(_cpaths["toml"])
        check("COLLECT_PASSED / COLLECT_TOTAL equal [Criteria] of collect.toml",
              (_edoc.get("Calculation_Info") or {}).get("COLLECT_PASSED") == _cdoc["Criteria"]["N_PASSED"]
              and (_edoc.get("Calculation_Info") or {}).get("COLLECT_TOTAL") == _cdoc["Criteria"]["N_TOTAL"],
              (_edoc.get("Calculation_Info"), _cdoc.get("Criteria")))
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n{}".format("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL)))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
