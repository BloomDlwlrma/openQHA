"""Branch A's basins are read from the molecule directory, not from a store.

UNIT. Builds a molecule directory by hand in a temporary root (two `mace/basinNN/
basin.extxyz` files and a `_records/basins.json`), then reads it back through
`openqha.store.basins`; under a second.

Branch B reads
`<molecule>/mace/basinNN/basin.extxyz`; the byte-identical second copy under
`data/basins/<tag>/<range>/<chunk>/` is retired. What every reader needs from
the module: does the product exist, how many basins, their geometries in branch A's
atom order, the record, a message naming the folder when it is absent, and the set of
finished molecules under a tag (what a resume subtracts from its worklist).
"""
import json
import os
import shutil
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
from openqha.store import basins, layout   # noqa: E402

FAIL = []
EXTXYZ = ('3\nProperties=species:S:1:pos:R:3 conformer={c} crest_comment="x" basin={b} energy_eV={e}\n'
          'O 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n')


def check(label, ok, detail=""):
    print("  {:58s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="openqha_t07_"))
    os.environ["S0_RUNS_ROOT"] = str(tmp)
    try:
        qid, tag = "dsgdb9nsd_000018", "t07"
        mol = layout.molecule_dir(tmp, tag, qid)
        for b in (0, 1):
            d = layout.mace_basin_dir(mol, b)
            d.mkdir(parents=True)
            (d / "basin.extxyz").write_text(EXTXYZ.format(c=b, b=b, e=-10.0 - b), encoding="utf-8")
        layout.records_dir(mol).mkdir()
        # The Property file, branchA.toml: the status
        # block first, then the blocks a later step reads.
        from openqha.store import property as prop
        prop.write(layout.records_dir(mol) / "branchA.toml",
                   {"Calculation_Info": {"QM9_INDEX": qid},
                    "Basin": [{"INDEX": 0, "RELATIVE": 0.0}, {"INDEX": 1, "RELATIVE": 1.2}]},
                   {}, status=prop.NORMAL_TERMINATION, progname="openQHA branchA")

        print("A. existence and count come from mace/basinNN/basin.extxyz:")
        check("exists(qid, tag)", basins.exists(qid, tag=tag) is True)
        check("not exists for another tag", basins.exists(qid, tag="other") is False)
        files = basins.basin_files(mol)
        check("two basin files, in basin order", [p.parent.name for p in files] == ["basin00", "basin01"],
              [str(p) for p in files])

        print("B. the geometries, in branch A's atom order:")
        try:
            atoms = basins.read_basins(qid, tag=tag)
            check("two Atoms", len(atoms) == 2 and list(atoms[0].get_chemical_symbols()) == ["O", "H", "H"],
                  [a.get_chemical_symbols() for a in atoms])
            check("header info survives", int(atoms[1].info.get("basin", -1)) == 1, atoms[1].info)
        except ImportError as exc:
            print("  (ase absent: {}; geometry read not checked)".format(exc))

        print("C. the record and the message:")
        rec = basins.read_record(qid, tag=tag)
        check("record read from _records/branchA.toml", rec is not None and len(rec["Basin"]) == 2, rec)
        from openqha.store import branch_a_property
        check("relative_kcal from the Basin blocks", branch_a_property.relative_kcal(rec) == [0.0, 1.2])
        check("status() reads the marker", basins.status(qid, tag=tag) == "NORMAL TERMINATION")
        check("done() = marker and basins", basins.done(qid, tag=tag) is True and basins.done(qid, tag="other") is False)
        check("no record for another tag", basins.read_record(qid, tag="other") is None)
        msg = basins.missing_message(qid, "other")
        check("missing message names the mace folder it looked for",
              "mace" in msg and "other" in msg and qid in msg, msg)
        check("...and the tag that does have it", "t07" in msg, msg)

        print("D. the finished set under a tag (resume subtracts it):")
        done = basins.completed(tag=tag)
        check("completed = {18}", done == {18}, done)
        # a molecule with a Property file but no basin file is NOT finished, and neither
        # is one with basins but no Property file (killed between the census and the record)
        qid2 = "dsgdb9nsd_000019"
        mol2 = layout.molecule_dir(tmp, tag, qid2)
        layout.records_dir(mol2).mkdir(parents=True)
        (layout.records_dir(mol2) / "branchA.toml").write_text("", encoding="utf-8")
        check("record without mace/basin00 is not finished", basins.completed(tag=tag) == {18})
        qid3 = "dsgdb9nsd_000020"
        d3 = layout.mace_basin_dir(layout.molecule_dir(tmp, tag, qid3), 0)
        d3.mkdir(parents=True)
        (d3 / "basin.extxyz").write_text(EXTXYZ.format(c=0, b=0, e=-10.0), encoding="utf-8")
        check("basins without branchA.toml are not finished", basins.completed(tag=tag) == {18})
        check("is_complete agrees", __import__("openqha.store.worklist", fromlist=["x"]).is_complete(qid3, tag=tag) is False)
        c = basins.census(tag=tag)
        # census counts molecules WITH BASINS (18 and 20), done() and completed() ask for the marker too
        check("census counts 2 under the flat tag directory", c["total"] == 2 and "chunks" not in c, c)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        os.environ.pop("S0_RUNS_ROOT", None)

    print("\n{}".format("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL)))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
