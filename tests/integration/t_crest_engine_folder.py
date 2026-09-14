"""CREST runs node-local and its finished directory is moved once into crest/ (ticket 06).

INTEGRATION. Runs the branch A pipeline with a STAND-IN crest binary (a shell script that
writes the known file set) and the real MACE for the tightening and Hessian that follow;
tens of seconds. Skips when MACE is absent or the platform has no bash.

The decision (ADR 0002 + Q20, user 2026-09-14): CREST writes dozens of small files and
must not do so on Lustre, so it runs under the node-local scratch
(`$S0_SCRATCH/openqha_crest/<qid>/`) and, when it returns, that directory is copied ONCE
into `<molecule>/crest/`; the SHAKE fallback's into `crest_shake1/`; the node-local copy
is then removed. Reuse, settings matching and the record parser read `crest/`.

    A. a normal run: crest/ holds crest's own files plus input.toml, <qid>.xyz and
       crest.out; no crest_shake1/; the scratch directory is gone; the record's workdir
       is crest/
    B. a run whose SHAKE=2 attempt terminates early: crest/ (the first attempt, kept)
       AND crest_shake1/ (the retry) side by side, both moved, scratch gone
    C. a second run under the same settings reuses crest/ and starts no scratch
    D. a crest that fails outright still leaves its crest.out in crest/

The stand-in reads `shake` from input.toml so B can be driven from the environment
(FAKE_CREST_EARLY_AT_SHAKE=2) and D from FAKE_CREST_FAIL=1.
"""
import json
import os
import shutil
import subprocess
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
from openqha.store import layout   # noqa: E402

PIPE = ROOT / "scripts" / "production" / "s0_A_pipeline.py"
QID = "dsgdb9nsd_000018"
FAIL = []

FAKE = r'''#!/bin/bash
# stand-in crest for tests/integration/t_crest_engine_folder.py
if [ "$1" = "--version" ]; then
  echo "       Version 3.0.2, Fri 1 Jan 00:00:00 UTC 2026"
  echo "       commit (fake000) compiled by 'test'"
  exit 0
fi
toml="$1"
shake=$(sed -n 's/^ *shake *= *\([0-9]*\).*/\1/p' "$toml" | head -1)
xyz=$(ls *.xyz 2>/dev/null | grep -v crest | head -1)
if [ -n "$FAKE_CREST_FAIL" ]; then
  echo "some crest banner"
  echo "error: the stand-in was told to fail"
  exit 1
fi
n=$(head -1 "$xyz")
body=$(tail -n +3 "$xyz" | head -n "$n")
{ echo "$n"; echo "     -13.28497543"; echo "$body"; echo "$n"; echo "     -13.28400000"; echo "$body"; } > crest_conformers.xyz
{ echo "$n"; echo "     -13.28497543"; echo "$body"; } > crest_best.xyz
cp crest_conformers.xyz crest_rotamers.xyz
printf '    1   -13.28497543\n    2   -13.28400000\n' > crest.energies
echo "restart" > crest.restart; echo "wbo" > wbo; echo "topo" > gfnff_topo; echo "x" > cre_members
echo "  CREST banner (stand-in)"
if [ -n "$FAKE_CREST_EARLY_AT_SHAKE" ] && [ "$shake" = "$FAKE_CREST_EARLY_AT_SHAKE" ]; then
  echo " *** MD run terminated EARLY ***"
fi
echo " Total number of energy+grad calls: 12"
echo " CREST terminated normally."
'''


def check(label, ok, detail=""):
    print("  {:64s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:300]))
    if not ok:
        FAIL.append(label)


def run_pipeline(root, scratch, fake, tag, extra_env=None):
    env = dict(os.environ, S0_RUNS_ROOT=str(root), S0_SCRATCH=str(scratch),
               S0_CREST_BIN=str(fake), S0_MACE_SOCKET=str(root / "unused.sock"))
    env.update(extra_env or {})
    cmd = [sys.executable, str(PIPE), "--species", QID, "--tag", tag, "--threads", "1",
           "--hessian-mode", "analytic"]
    p = subprocess.run(cmd, env=env, cwd=str(ROOT), text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.returncode, p.stdout


def crest_set(d):
    return sorted(p.name for p in d.iterdir()) if d.is_dir() else None


CREST_OWN = {"crest_conformers.xyz", "crest_best.xyz", "crest_rotamers.xyz", "crest.energies",
             "crest.restart", "wbo", "gfnff_topo", "cre_members"}
OURS = {"input.toml", QID + ".xyz", "crest.out"}


def main():
    if shutil.which("bash") is None:
        print("SKIP: no bash for the stand-in crest")
        return 0
    try:
        import mace  # noqa: F401
    except Exception as exc:                          # noqa: BLE001
        print("SKIP: {}".format(exc))
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="openqha_t06_"))
    try:
        fake = tmp / "crest"
        fake.write_text(FAKE, encoding="utf-8")
        fake.chmod(0o755)
        scratch = tmp / "scratch"
        root = tmp / "root"

        print("A. a normal run: node-local, then moved once into crest/")
        rc, out = run_pipeline(root, scratch, fake, "a")
        mol = layout.molecule_dir(root, "a", QID)
        crest = layout.crest_dir(mol)
        names = crest_set(crest)
        check("pipeline ran to its record (exit {})".format(rc),
              (layout.records_dir(mol) / "basins.json").exists(), out[-1500:])
        check("crest/ holds crest's own files and ours, nothing else",
              names is not None and set(names) == CREST_OWN | OURS, names)
        check("no crest_shake1/", not layout.crest_dir(mol, fallback_shake=1).exists())
        check("the scratch copy is gone", not (scratch / "openqha_crest").exists()
              or not any((scratch / "openqha_crest").iterdir()),
              crest_set(scratch / "openqha_crest"))
        rec = json.loads((layout.records_dir(mol) / "basins.json").read_text(encoding="utf-8"))
        check("record's crest workdir is crest/", rec["crest"]["workdir"] == str(crest), rec["crest"].get("workdir"))
        check("record says where it ran", "ran_in" in rec["crest"] and str(scratch) in rec["crest"]["ran_in"],
              rec["crest"].get("ran_in"))

        print("C. a second run under the same settings reuses crest/ without a scratch")
        rc, out = run_pipeline(root, scratch, fake, "a")
        rec = json.loads((layout.records_dir(mol) / "basins.json").read_text(encoding="utf-8"))
        check("reused_scratch is True", rec["crest"].get("reused_scratch") is True, rec["crest"].get("reused_scratch"))
        check("no scratch directory was created", not (scratch / "openqha_crest" / QID).exists())

        print("B. SHAKE=2 terminates early: the first attempt is kept beside the retry")
        rc, out = run_pipeline(root, scratch, fake, "b", {"FAKE_CREST_EARLY_AT_SHAKE": "2"})
        mol_b = layout.molecule_dir(root, "b", QID)
        c1, c2 = layout.crest_dir(mol_b), layout.crest_dir(mol_b, fallback_shake=1)
        check("crest/ (first attempt) exists with crest.out", (c1 / "crest.out").exists(), crest_set(c1))
        check("crest_shake1/ (retry) exists with the file set",
              crest_set(c2) is not None and set(crest_set(c2)) == CREST_OWN | OURS, crest_set(c2))
        def _out(d):
            return (d / "crest.out").read_text(encoding="utf-8", errors="replace") if (d / "crest.out").exists() else ""
        check("first attempt's crest.out says EARLY", "terminated EARLY" in _out(c1))
        check("retry's does not", (c2 / "crest.out").exists() and "terminated EARLY" not in _out(c2))
        check("scratch gone after the move", not (scratch / "openqha_crest" / QID).exists()
              and not (scratch / "openqha_crest" / (QID + "_shake1")).exists())
        rb = (layout.records_dir(mol_b) / "basins.json")
        rec = json.loads(rb.read_text(encoding="utf-8")) if rb.exists() else {"crest": {}}
        check("record: used_shake_fallback, workdir crest_shake1/, first_attempt crest/",
              rec["crest"].get("used_shake_fallback") is True and rec["crest"].get("workdir") == str(c2)
              and (rec["crest"].get("first_attempt") or {}).get("workdir") == str(c1), rec["crest"])

        print("D. a crest that fails still leaves its crest.out in crest/")
        rc, out = run_pipeline(root, scratch, fake, "d", {"FAKE_CREST_FAIL": "1"})
        mol_d = layout.molecule_dir(root, "d", QID)
        cd = layout.crest_dir(mol_d)
        check("pipeline stopped (exit {})".format(rc), rc != 0)
        check("crest.out in crest/", (cd / "crest.out").exists(), crest_set(cd))
        check("...with the stand-in's error line",
              (cd / "crest.out").exists() and "told to fail" in (cd / "crest.out").read_text(encoding="utf-8"))
        check("scratch gone", not (scratch / "openqha_crest" / QID).exists())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n{}".format("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL)))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
