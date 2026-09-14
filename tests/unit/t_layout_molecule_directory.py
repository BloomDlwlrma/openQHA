"""The layout module answers every path question of the molecule tree (ticket 01).

UNIT. Pure path arithmetic; no filesystem, no configuration; under a second.

The decision (ADR 0001, 2026-09-14)
-----------------------------------
One molecule directory per (root, tag, qid), one folder per engine inside it, engine
files only in those folders, this repository's records in `_records/`. The tree is
sharded as range 16 000 / chunk 1 000 (user, 2026-09-14: chunk 1 000, not the retired
store's 4 000). Every writer and reader asks `openqha.store.layout`; nothing else
composes these paths, which is what makes the tree shape a checked fact.

Expected values below are literals worked out by hand from the rule, not recomputed
from the module.
"""
import sys
from pathlib import Path, PurePosixPath


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from openqha.store import layout    # noqa: E402

FAIL = []
R = PurePosixPath("/XYAIFS00/HDD_POOL/acct/user/sherwin/runs")


def check(label, got, want):
    ok = str(got) == str(want)
    print("  {:56s} {}".format(label, "ok" if ok else "FAIL\n      got  {}\n      want {}".format(got, want)))
    if not ok:
        FAIL.append(label)


def main():
    print("A. shards, worked by hand from range 16 000 / chunk 1 000:")
    for qid, want in (("dsgdb9nsd_000018", "1_16000/1_1000"),
                      ("dsgdb9nsd_001000", "1_16000/1_1000"),
                      ("dsgdb9nsd_001001", "1_16000/1001_2000"),
                      ("dsgdb9nsd_016000", "1_16000/15001_16000"),
                      ("dsgdb9nsd_016001", "16001_32000/16001_17000"),
                      ("dsgdb9nsd_133885", "128001_144000/133001_134000")):
        check(qid, "/".join(layout.shard(qid)), want)

    print("B. the seven shipped species share one chunk:")
    for p in sorted((ROOT / "data" / "reference-geometries").glob("dsgdb9nsd_*.xyz")):
        check(p.stem, "/".join(layout.shard(p.stem)), "1_16000/1_1000")

    print("C. the molecule directory and its folders:")
    m = layout.molecule_dir(R, "02d_prod", "dsgdb9nsd_000018")
    check("molecule_dir", m, R / "02d_prod/1_16000/1_1000/dsgdb9nsd_000018")
    check("crest", layout.crest_dir(m), m / "crest")
    check("crest fallback", layout.crest_dir(m, fallback_shake=1), m / "crest_shake1")
    check("mace conformer 3", layout.mace_conformer_dir(m, 3), m / "mace/conf03")
    check("mace basin 0", layout.mace_basin_dir(m, 0), m / "mace/basin00")
    check("openmm default basin 1", layout.openmm_dir(m, "default", 1), m / "openmm/default/basin01")
    check("openmm row p1500_s5 basin 0", layout.openmm_dir(m, "p1500_s5", 0), m / "openmm/p1500_s5/basin00")
    check("xtb basin 2", layout.xtb_dir(m, 2), m / "xtb/basin02")
    check("orca basin 2", layout.orca_dir(m, 2), m / "orca/basin02")
    check("records", layout.records_dir(m), m / "_records")

    print("D. engine file names are fixed by the module, not by callers:")
    check("openmm files", " ".join(layout.OPENMM_FILES),
          "start.pdb system.xml integrator.xml traj.dcd state.csv state.xml state.chk")
    check("mace conformer files", " ".join(layout.MACE_CONFORMER_FILES), "opt.traj opt.log conf.extxyz")
    check("mace basin files", " ".join(layout.MACE_BASIN_FILES), "basin.extxyz hessian.npy")

    print("E. a qid that carries no number is refused, not sharded into nonsense:")
    try:
        layout.shard("acetone")
        FAIL.append("shard('acetone') did not raise")
        print("  acetone                                                  FAIL (no error)")
    except ValueError as exc:
        print("  acetone -> ValueError: {}".format(str(exc)[:60]))

    print("\n{}".format("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL)))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
