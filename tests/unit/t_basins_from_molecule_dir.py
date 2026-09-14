"""Branch A's basins are read from the molecule directory, not from a store (ticket 07).

UNIT. Builds a molecule directory by hand in a temporary root (two `mace/basinNN/
basin.extxyz` files and a `_records/basins.json`), then reads it back through
`openqha.store.basins`; under a second.

The decision (ADR 0001, user 2026-09-14, Q6): branch B reads
`<molecule>/mace/basinNN/basin.extxyz`; `data/basins/<tag>/<range>/<chunk>/` was a
byte-identical second copy of the record and is retired. What every reader needs from
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
        (layout.records_dir(mol) / "basins.json").write_text(
            json.dumps(dict(qm9_index=qid, basins=[dict(relative_kcal=0.0), dict(relative_kcal=1.2)])),
            encoding="utf-8")

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
        check("record read from _records/basins.json", rec is not None and len(rec["basins"]) == 2, rec)
        check("no record for another tag", basins.read_record(qid, tag="other") is None)
        msg = basins.missing_message(qid, "other")
        check("missing message names the mace folder it looked for",
              "mace" in msg and "other" in msg and qid in msg, msg)
        check("...and the tag that does have it", "t07" in msg, msg)

        print("D. the finished set under a tag (resume subtracts it):")
        done = basins.completed(tag=tag)
        check("completed = {18}", done == {18}, done)
        # a molecule with a record but no basin file is NOT finished
        qid2 = "dsgdb9nsd_000019"
        mol2 = layout.molecule_dir(tmp, tag, qid2)
        layout.records_dir(mol2).mkdir(parents=True)
        (layout.records_dir(mol2) / "basins.json").write_text("{}", encoding="utf-8")
        check("record without mace/basin00 is not finished", basins.completed(tag=tag) == {18})
        c = basins.census(tag=tag)
        check("census counts 1 under 1_16000/1_1000", c["total"] == 1 and c["chunks"].get("1_16000/1_1000") == 1, c)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        os.environ.pop("S0_RUNS_ROOT", None)

    print("\n{}".format("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL)))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
