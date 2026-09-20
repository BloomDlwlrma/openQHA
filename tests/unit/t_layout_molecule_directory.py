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
    print("A. the tag directory is flat (ADR 0001 amendment 3, 2026-09-20): no shard layers, no shard():")
    for qid in ("dsgdb9nsd_000018", "dsgdb9nsd_001001", "dsgdb9nsd_133885"):
        check(qid, layout.molecule_dir(R, "t", qid), R / "t" / qid)
    check("no shard / CHUNK / RANGE any more", any(hasattr(layout, n) for n in ("shard", "CHUNK", "RANGE")), False)

    print("B. the seven shipped species sit beside each other under the tag:")
    for p in sorted((ROOT / "data" / "reference-geometries").glob("dsgdb9nsd_*.xyz")):
        check(p.stem, layout.molecule_dir(R, "t", p.stem).parent, R / "t")

    print("C. the molecule directory and its folders:")
    m = layout.molecule_dir(R, "02d_prod", "dsgdb9nsd_000018")
    check("molecule_dir", m, R / "02d_prod/dsgdb9nsd_000018")
    check("crest", layout.crest_dir(m), m / "crest")
    check("crest fallback", layout.crest_dir(m, fallback_shake=1), m / "crest_shake1")
    check("mace conformer 3", layout.mace_conformer_dir(m, 3), m / "mace/conf03")
    check("mace basin 0", layout.mace_basin_dir(m, 0), m / "mace/basin00")
    # One md_openmm/basinNN per basin, every setting inside it (user ruling 2026-09-14, which
    # replaced openmm/<setting>/basinNN); the setting is in the FILE NAME, and the
    # default setting keeps the bare names.
    # Named by ROLE (user, 2026-09-15): md_openmm/ and md_ase/ are the two MD routes,
    # mace/ is the relax + Hessian; the route identifiers stay "openmm" / "ase".
    check("openmm default basin 1", layout.openmm_dir(m, "default", 1), m / "md_openmm/basin01")
    check("openmm row p1500_s5 basin 0", layout.openmm_dir(m, "p1500_s5", 0), m / "md_openmm/basin00")
    check("ase basin 0", layout.ase_dir(m, "default", 0), m / "md_ase/basin00")
    check("md_folder(openmm)", layout.md_folder("openmm"), "md_openmm")
    check("md_folder(ase)", layout.md_folder("ase"), "md_ase")
    check("route_of_folder(md_ase)", layout.route_of_folder("md_ase"), "ase")
    # No setting level under _records (user ruling 2026-09-15): the setting is in the stem.
    check("md_records_dir openmm", layout.md_records_dir(m, "openmm"), m / "_records/md_openmm")
    check("basin_records_dir ase 1", layout.basin_records_dir(m, "ase", 1), m / "_records/md_ase/basin01")
    check("record_file_name s2", layout.record_file_name("collect.out", "s2"), "collect_s2.out")
    check("no records_for any more", hasattr(layout, "records_for") or hasattr(layout, "openmm_records_dir"), False)
    check("default file name", layout.openmm_file_name("traj.dcd", "default"), "traj.dcd")
    check("setting file name", layout.openmm_file_name("traj.dcd", "p1500_s5"), "traj_p1500_s5.dcd")
    check("setting file name, pdb", layout.openmm_file_name("start.pdb", "p1500_s5"), "start_p1500_s5.pdb")
    check("openmm_file composes both", layout.openmm_file(m, "p1500_s5", 0, "state.csv"),
          m / "md_openmm/basin00/state_p1500_s5.csv")
    check("tag records dir", layout.tag_records_dir(R, "02d_prod"), R / "02d_prod/_records")
    check("xtb basin 2", layout.xtb_dir(m, 2), m / "xtb/basin02")
    check("no orca_dir / orca_level_dir / level_dir / LEVELS any more (ticket 09b)",
          any(hasattr(layout, n) for n in ("orca_dir", "orca_level_dir", "level_dir", "LEVELS")), False)
    check("msrrho basin job stem", layout.orca_level_stem("wb97m-d3bj_def2-tzvppd", 0), "orca.wb97m-d3bj_def2-tzvppd.basin00")
    check("msrrho basin job file", layout.orca_level_file(m, "wb97m-d3bj_def2-tzvppd", 2, ".hess"),
          m / "msrrho/orca.wb97m-d3bj_def2-tzvppd.basin02.hess")
    check("crest entropy run 1", layout.crest_entropy_dir(m, 1), m / "msrrho/crest_entropy/run01")
    check("xtb entropy run 2 conf 99", layout.xtb_entropy_dir(m, 2, 99), m / "msrrho/xtb/entropy_run02/conf99")
    check("thermo record of a level", layout.level_file(m, "mace-off23_medium", "thermo_msrrho.toml"),
          m / "msrrho/thermo/mace-off23_medium.thermo_msrrho.toml")
    check("thermo record across levels", layout.thermo_file(m, "level_compare.out"), m / "msrrho/thermo/level_compare.out")
    check("frame label stem (ticket 09)", layout.orca_frame_stem("wb97m-d3bj_def2-tzvppd", "displaced", 0, 3),
          "orca.wb97m-d3bj_def2-tzvppd.displaced_b00_k3")
    check("frame label file, in frames/", layout.orca_frame_file(m, "hf_cc-pvtz", "basin", 12, 0, ".engrad"),
          m / "frames/orca.hf_cc-pvtz.basin_b12_k0.engrad")
    check("records", layout.records_dir(m), m / "_records")

    print("D. engine file names are fixed by the module, not by callers:")
    check("openmm files", " ".join(layout.OPENMM_FILES),
          "start.pdb system.xml integrator.xml traj.dcd state.csv state.xml state.chk")
    check("mace conformer files", " ".join(layout.MACE_CONFORMER_FILES), "opt.traj opt.log conf.extxyz")
    check("mace basin files", " ".join(layout.MACE_BASIN_FILES), "basin.extxyz hessian.npy")

    print("E. a qid that carries no number: qid_number refuses it, molecule_dir uses the label as it is:")
    try:
        layout.qid_number("acetone")
        FAIL.append("qid_number('acetone') did not raise")
        print("  acetone                                                  FAIL (no error)")
    except ValueError as exc:
        print("  acetone -> ValueError: {}".format(str(exc)[:60]))
    check("molecule_dir for a label-only molecule", layout.molecule_dir(R, "t", "acetone"), R / "t/acetone")

    print("\n{}".format("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL)))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
