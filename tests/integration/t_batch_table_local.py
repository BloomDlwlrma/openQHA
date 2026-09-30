"""A Batch leaves nothing but its table: the branch B parsl driver on the local resource config.

INTEGRATION (OpenMM + MACE on CPU, parsl; a minute).

Runs `s0_E_branchB_parsl.py --resource local` for one molecule (one basin: no branch A
under this tag, so the trajectory starts from the QM9 reference geometry, which the
driver says loudly) at smoke length, and holds:

    A. the driver's stdout carries the Batch table: the header with the seven common
       columns, one line whose `record` is the absolute path of an existing md.toml with
       STATUS NORMAL TERMINATION, and the footer with the batch wall and the Slurm job id
    B. under <root>/<tag>/_records/ there is nothing but parsl/ (no JSON summary), and the
       repository checkout gained no analysis/branchE/<tag>/
    C. the Calculation's own Record is in _records/md_openmm/basin00/
       md.out, md.toml, driver.log
"""
import os
import re
import shutil
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

DRIVER = ROOT / "scripts" / "production" / "s0_E_branchB_parsl.py"
QID, TAG = "dsgdb9nsd_000018", "tbatch"
FAIL = []


def check(label, ok, detail=""):
    print("  {:64s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:400]))
    if not ok:
        FAIL.append(label)


def main():
    try:
        import openmm, mace, mdtraj, parsl      # noqa: F401
    except Exception as exc:                    # noqa: BLE001
        print("SKIP: {}".format(exc))
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="openqha_t20_"))
    branch_e = ROOT / "analysis" / "branchE" / TAG
    try:
        env = dict(os.environ, S0_RUNS_ROOT=str(tmp))
        env.pop("SLURM_JOB_ID", None)
        cmd = [sys.executable, str(DRIVER), "--species", QID, "--tag", TAG, "--resource", "local",
               "--route", "openmm", "--platform", "CPU", "--equil-ps", "0.02", "--prod-ps", "0.05",
               "--sample-every", "10", "--max-workers", "1"]
        t0 = time.time()
        p = subprocess.run(cmd, env=env, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = p.stdout
        dt = time.time() - t0
        lines = out.splitlines()

        print("A. the Batch table in stdout")
        check("driver exit 0 ({:.0f} s)".format(dt), p.returncode == 0, out[-2000:])
        hdr = [i for i, l in enumerate(lines) if l.split()[:7] == ["species", "basin", "seed", "rc", "seconds", "STATUS", "record"]]
        check("header with the seven common columns", len(hdr) == 1, [l for l in lines if l.startswith("species")])
        row = lines[hdr[0] + 1] if hdr else ""
        check("one line for the one Calculation, rc 0, NORMAL TERMINATION",
              row.startswith(QID) and " 0 " in row and "NORMAL TERMINATION" in row, row)
        m = re.search(r"(/\S+md\.toml)", row)
        rec = Path(m.group(1)) if m else None
        mol = layout.molecule_dir(tmp, TAG, QID)
        check("record is the absolute path of an existing md.toml under _records/md_openmm/basin00/",
              rec is not None and rec.is_file() and rec == layout.basin_records_dir(mol, "openmm", 0) / "md.toml", row)
        from openqha.store import property as prop
        check("...with STATUS NORMAL TERMINATION", rec is not None and prop.status_of(rec) == prop.NORMAL_TERMINATION)
        check("footer: batch wall and the pass count", any(l.startswith("batch wall ") and "1/1" in l for l in lines),
              [l for l in lines if l.startswith("batch wall")])
        check("footer: the Slurm job id line (none here)", any(l.startswith("slurm job (none") for l in lines),
              [l for l in lines if l.startswith("slurm job")])
        check("no 'written ...json' line", not any("written" in l and ".json" in l for l in lines))

        print("B. nothing but parsl/ under <root>/<tag>/_records/, no analysis/branchE/")
        tag_rec = layout.tag_records_dir(tmp, TAG)
        names = sorted(q.name for q in tag_rec.iterdir()) if tag_rec.is_dir() else []
        check("<root>/<tag>/_records/ holds only parsl/", names in ([], ["parsl"]), names)
        check("no JSON anywhere under the root", not list(tmp.rglob("*.json")), list(tmp.rglob("*.json"))[:5])
        check("no analysis/branchE/<tag>/ in the checkout", not branch_e.exists())

        print("C. the Calculation's Record")
        rdir = layout.basin_records_dir(mol, "openmm", 0)
        rnames = sorted(q.name for q in rdir.iterdir()) if rdir.is_dir() else []
        check("_records/md_openmm/basin00: driver.log md.out md.toml", rnames == ["driver.log", "md.out", "md.toml"], rnames)
        check("no directory named default under _records", not [q for q in (mol / "_records").rglob("default") if q.is_dir()])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(branch_e, ignore_errors=True)

    print()
    print("FAILED: {}".format(FAIL) if FAIL else "PASS")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
