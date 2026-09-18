"""The delete-old-layout script lists before it removes, and never touches the new root (ticket 08).

UNIT. Builds a throwaway copy of the OLD layout (repository trees and the runs roots of
every period) inside a temporary directory, then runs the script on it; under a second.

The ruling (user, 2026-09-14): the old trees are not migrated, they are deleted -- after
a plan that lists exactly what will go. So `--plan` (the default) must list every old
tree with its size and file count and remove nothing; `--delete` must remove those trees
and nothing else; and a directory under the NEW root is never listed even when it is
called `analysis` or `runs`, because that name is not the layout.

Old trees, by period:
    <repo>/analysis, <repo>/data/basins, <repo>/logs/node_local, <repo>/runs
    <home>/runs/openQHA                          off-cluster default; Tianhe before 2026-09-12
    <home>/runs/<jobid>[_cardK_rowJ]             Tianhe scratch, 2026-09-12 .. 2026-09-14
    <home>/HDD_POOL/runs/openQHA                 tianhe_cpu default
"""
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
SCRIPT = ROOT / "scripts" / "tooling" / "s0_delete_old_layout.py"
FAIL = []


def touch(p, n=1):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * n)


def build(tmp):
    repo, home, new = tmp / "repo", tmp / "home", tmp / "newroot"
    (repo / "openqha").mkdir(parents=True)
    touch(repo / "openqha" / "__init__.py")
    touch(repo / "analysis" / "qha" / "fixed" / "x.log", 10)
    touch(repo / "analysis" / "branchA" / "prod" / "m" / "basins.json", 20)
    touch(repo / "data" / "basins" / "fixed" / "1_16000" / "1_4000" / "m.basins.json", 30)
    touch(repo / "logs" / "node_local" / "247785" / "MANIFEST.txt", 5)
    touch(repo / "logs" / "openqha_247785.out", 7)                 # Slurm output: stays
    touch(repo / "runs" / "branchA" / "t" / "m" / "crest.out", 9)
    touch(home / "runs" / "openQHA" / "branchA" / "t" / "m" / "crest.out", 11)
    touch(home / "runs" / "247785" / "runs" / "qha" / "t" / "m" / "frames.npy", 13)
    touch(home / "runs" / "247790_card0_row1" / "s0.sock.log", 1)
    touch(home / "runs" / "notes.txt", 2)                            # not a job dir: stays
    touch(home / "HDD_POOL" / "runs" / "openQHA" / "logs" / "b.log", 3)
    touch(new / "02d_prod" / "1_16000" / "1_1000" / "m" / "crest" / "crest.out", 4)
    touch(new / "analysis" / "trap.txt", 4)                          # under the new root: stays
    # Step-2 leftovers inside the new root (records redesign, 2026-09-15): listed by name.
    mol = new / "02d_prod" / "1_16000" / "1_1000" / "dsgdb9nsd_000018"
    touch(mol / "_records" / "basins.toml", 6)
    touch(mol / "_records" / "branchA.toml", 6)                      # the replacement: stays
    touch(mol / "_records" / "md_openmm" / "default" / "basin00" / "md.toml", 6)
    touch(mol / "_records" / "md_openmm" / "s2" / "collect.out", 6)
    touch(mol / "_records" / "md_openmm" / "basin00" / "md.toml", 6)  # the new form: stays
    touch(new / "02d_prod" / "_records" / "branchB_parsl_summary.json", 6)
    touch(new / "02d_prod" / "_records" / "collect_batch.json", 6)
    touch(new / "02d_prod" / "_records" / "parsl" / "1.2" / "parsl.log", 6)   # parsl's own: stays
    return repo, home, new


def run(args, repo, home, new):
    cmd = [sys.executable, str(SCRIPT), "--repo", str(repo), "--home", str(home),
           "--new-root", str(new)] + args
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.returncode, p.stdout


def check(label, ok, detail=""):
    print("  {:56s} {}".format(label, "ok" if ok else "FAIL " + detail))
    if not ok:
        FAIL.append(label)


def main():
    if not SCRIPT.is_file():
        print("FAIL: {} does not exist".format(SCRIPT))
        return 1
    tmp = Path(tempfile.mkdtemp(prefix="openqha_old_"))
    try:
        repo, home, new = build(tmp)
        old = [repo / "analysis", repo / "data" / "basins", repo / "logs" / "node_local",
               repo / "runs", home / "runs" / "openQHA", home / "runs" / "247785",
               home / "runs" / "247790_card0_row1", home / "HDD_POOL" / "runs" / "openQHA"]
        mol = new / "02d_prod" / "1_16000" / "1_1000" / "dsgdb9nsd_000018"
        old += [mol / "_records" / "basins.toml", mol / "_records" / "md_openmm" / "default",
                mol / "_records" / "md_openmm" / "s2",
                new / "02d_prod" / "_records" / "branchB_parsl_summary.json",
                new / "02d_prod" / "_records" / "collect_batch.json"]
        stays = [repo / "logs" / "openqha_247785.out", home / "runs" / "notes.txt",
                 new / "02d_prod" / "1_16000" / "1_1000" / "m", new / "analysis" / "trap.txt",
                 mol / "_records" / "branchA.toml", mol / "_records" / "md_openmm" / "basin00" / "md.toml",
                 new / "02d_prod" / "_records" / "parsl" / "1.2" / "parsl.log"]

        print("A. --plan lists every old tree with size and file count, removes nothing:")
        rc, out = run([], repo, home, new)
        check("exit 0", rc == 0, out[-300:])
        for p in old:
            line = [l for l in out.splitlines() if str(p) in l]
            check("listed  " + str(p.relative_to(tmp)), bool(line), "not in output")
            if line:
                l = line[0]
                check("  ...with a size and a count", ("files" in l and "B" in l), l)
        for p in stays:
            check("not listed  " + str(p.relative_to(tmp)), str(p) not in out)
        check("nothing removed by --plan", all(p.exists() for p in old + stays))
        check("plan says it removed nothing", "nothing removed" in out.lower(), out[-200:])

        print("B. --delete removes exactly the listed trees:")
        rc, out = run(["--delete"], repo, home, new)
        check("exit 0", rc == 0, out[-300:])
        check("old trees gone", not any(p.exists() for p in old),
              str([str(p) for p in old if p.exists()]))
        check("everything else still there", all(p.exists() for p in stays),
              str([str(p) for p in stays if not p.exists()]))
        check("reports what it removed", "removed" in out.lower() and str(repo / "analysis") in out)

        print("C. a second --delete finds nothing and says so:")
        rc, out = run(["--delete"], repo, home, new)
        check("exit 0, nothing to do", rc == 0 and "nothing" in out.lower(), out[-200:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n{}".format("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL)))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
