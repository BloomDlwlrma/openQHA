"""The structure-class census over the curated QM9 files against the counts the user quoted.

TOOLING. Reads every curated QM9 xyz (133,660 files; the relaxed SMILES line), classifies
it with `configs/structure_classes.yaml`, prints one row per class (census, quoted, ratio)
and flags a class more than 10 % off its quoted count as SUSPECT -- the check of the
SMARTS before any draw. `--membership` runs the same census
over the gated-target table instead (119,450 rows, seconds).

    python scripts/tooling/s0_structure_census.py                 # all curated QM9 (minutes)
    python scripts/tooling/s0_structure_census.py --membership    # the gated targets only
    python scripts/tooling/s0_structure_census.py --limit 5000    # a quick look
    python scripts/tooling/s0_structure_census.py --write         # + the per-molecule table (below)

`--write` also writes the per-molecule table `data/training_sets/qm9_structure_classes.dat`
(the repository's .dat form; a `.csv` twin beside it): one row per curated QM9
file -- qm9_index, gdb17_smiles, relaxed_smiles, n_heavy, target (passes the branch A
gates: a row of qm9_targets_membership.dat), in_training (SPICE, for targets), classes
(';'-joined). 133,660 rows, ~15 MB; the answer to "which QM9 molecules does each class
contain".
"""
import argparse
import sys
import time
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from openqha import config                                               # noqa: E402
from openqha.data import curated_qm9, dataset, structure_classes as sc   # noqa: E402
from openqha.store import dat                                            # noqa: E402


def smiles_all(limit=None):
    for _qid, _gdb, relaxed in molecules_all(limit):
        yield relaxed


def molecules_all(limit=None):
    """(qm9_index, gdb17 SMILES, relaxed SMILES) for every curated QM9 file, in index order."""
    index = curated_qm9._index()
    for i, n in enumerate(sorted(index)):
        if limit and i >= limit:
            return
        gdb, relaxed = config.qm9_smiles_from_xyz(index[n][1])
        yield "dsgdb9nsd_{:06d}".format(n), gdb, relaxed


TABLE = Path(dataset.MEMBERSHIP_FILE).with_name("qm9_structure_classes.dat")
TABLE_SCHEMA = {
    "qm9_index": ("String", None, "the molecule"),
    "gdb17_smiles": ("String", None, "the GDB-17 SMILES of the QM9 file"),
    "relaxed_smiles": ("String", None, "the SMILES parsed back from the deposited geometry (what the classes are matched on)"),
    "n_heavy": ("Integer", None, "heavy atoms (-1 when RDKit cannot parse)"),
    "target": ("Boolean", None, "passes the branch A gates F0-F7 (a row of qm9_targets_membership.dat)"),
    "in_training": ("String", None, "true / false: in MACE-OFF23's SPICE training file (targets only); - otherwise"),
    "classes": ("String", None, "structure classes, ';'-joined (- when none)"),
}


def write_table(cf, limit=None, path=TABLE):
    from rdkit import Chem
    mem = {r["qm9_index"]: r for r in dat.read_table(dataset.MEMBERSHIP_FILE)}
    rows = []
    for qid, gdb, relaxed in molecules_all(limit):
        m = Chem.MolFromSmiles(relaxed)
        cls = sorted(cf.classify(relaxed)) if m is not None else []
        t = mem.get(qid)
        rows.append(dict(qm9_index=qid, gdb17_smiles=gdb, relaxed_smiles=relaxed,
                         n_heavy=int(m.GetNumHeavyAtoms()) if m is not None else -1, target=t is not None,
                         in_training=("true" if t["in_training"] else "false") if t else "-", classes=";".join(cls) or "-"))
    dat.write_table(path, rows, list(TABLE_SCHEMA), TABLE_SCHEMA)
    import csv
    with open(str(path)[:-4] + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(TABLE_SCHEMA))
        w.writeheader()
        w.writerows(rows)
    return path, len(rows)


def smiles_membership(limit=None):
    for i, r in enumerate(dat.read_table(dataset.MEMBERSHIP_FILE)):
        if limit and i >= limit:
            return
        yield r["smiles"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--membership", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--classes", default=str(sc.CLASSES_FILE))
    ap.add_argument("--write", action="store_true", help="also write the per-molecule table (all curated QM9)")
    args = ap.parse_args()
    classes = sc.load_classes(args.classes)
    cf = sc.Classifier(classes)
    t0 = time.time()
    counts = sc.census(smiles_membership(args.limit) if args.membership else smiles_all(args.limit), cf)
    bad = {n for n, _c, _q in sc.suspects(counts, classes)}
    print("{:22s} {:>8s} {:>8s} {:>7s}  {}".format("class", "census", "quoted", "ratio", "definition"))
    for c in classes:
        n, q = counts[c["name"]], c.get("qm9_count") or 0
        print("{:22s} {:8d} {:>8s} {:>7s}  {}{}".format(
            c["name"], n, str(q) if q else "-", "{:.2f}".format(n / q) if q else "-", sc.definition(c),
            "   SUSPECT" if c["name"] in bad else ""))
    print("molecules {}  unparsed {}  {:.0f} s".format(counts["_n"], counts["_unparsed"], time.time() - t0))
    if args.write:
        path, n = write_table(cf, args.limit)
        print("table  {}  ({} rows; .csv twin beside it)".format(path, n))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
