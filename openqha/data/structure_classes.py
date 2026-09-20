"""Structure classes of QM9 molecules and the draw of the Hessian-learning campaign
(ticket 05; round 5, Q1 (a)+(c), Q2 (a), Q5 (a)).

CLASSES
-------
`configs/structure_classes.yaml` names each class with a SMARTS (RDKit, matched on the
relaxed SMILES with explicit hydrogens removed) or a ring rule (`min_rings`, `fused`,
`aromatic_rings`, `ring_size` on RDKit's ring info). `classify(smiles)` returns the set of
class names a molecule belongs to -- a molecule is usually in several (a heterocyclic
epoxide is small_ring, heterocyclic, epoxide and dialkyl_ether), and that is kept:
the judge asks per class. `census(smiles_iter)` counts per class; the tooling script
runs it over the curated QM9 files against the counts the user quoted, and a class more
than SUSPECT_TOL off is flagged before any draw.

THE DRAW (`draw`)
-----------------
Candidates: the gated targets of `data/training_sets/qm9_targets_membership.dat` with
`in_training = false` -- not in MACE-OFF23's SPICE training file at ANY match level
(isomeric, stereo-removed, InChIKey connectivity; `training_set.py`) -- 119,275 of
119,450 (2026-09-19). Per class, `per_class` molecules are drawn with one generator
seeded from `(seed, class)` over the sorted candidates of that class; the draw is the
UNION over classes (a molecule counts for every class it is in); a class with fewer
candidates than `per_class` takes them all and its shortfall is a row of the Record
(eight-membered rings: 36 in all of QM9). The pinned seven of the msRRHO study are
always rows. Written under `<root>/<tag>/_datasets/<name>/` as `draw.{out,toml,dat}`;
`draw.dat` is the molecule list the campaign's stage scripts read (one `qm9_index` per
line, with `smiles`, `n_heavy`, `classes`, `in_training`).
"""
import time
from pathlib import Path

import numpy as np

from ..store import dat, property as prop, report
from . import dataset as dataset_mod

STEP = "draw"
PROGNAME = "openQHA structure draw"
CLASSES_FILE = Path(__file__).resolve().parents[2] / "configs" / "structure_classes.yaml"
PER_CLASS = 500
SEED = 0
#: a class whose census count differs from the quoted QM9 count by more than this is suspect
SUSPECT_TOL = 0.10

ROW_SCHEMA = {
    "qm9_index": ("String", None, "the molecule"),
    "smiles": ("String", None, "relaxed SMILES (the membership table's)"),
    "n_heavy": ("Integer", None, "heavy atoms"),
    "classes": ("String", None, "the structure classes it belongs to, ';'-joined (- when none)"),
    "in_training": ("Boolean", None, "in MACE-OFF23's SPICE training file at any match level (false for every drawn row)"),
    "pinned": ("Boolean", None, "one of the seven msRRHO molecules, always in"),
}

SCHEMA = {
    "Calculation_Info": {
        "NAME": ("String", None, "the Dataset name"),
        "TAG": ("String", None, "the tag the campaign's molecule directories go under"),
        "CLASSES_FILE": ("String", None, "the class definitions"),
        "MEMBERSHIP_FILE": ("String", None, "the gated-target table with SPICE membership"),
        "PER_CLASS": ("Integer", None, "molecules drawn per class (fewer when the class is short)"),
        "SEED": ("Integer", None, "the draw's seed"),
        "EXCLUDE_TRAINING": ("Boolean", None, "candidates with in_training = true excluded"),
        "N_TARGETS": ("Integer", None, "rows of the membership table"),
        "N_CANDIDATES": ("Integer", None, "rows kept as candidates (outside SPICE)"),
        "N_CLASSES": ("Integer", None, "classes defined"),
        "N_DRAWN": ("Integer", None, "molecules in the union (pinned included)"),
        "N_PINNED": ("Integer", None, "pinned molecules present in the table"),
        "N_SHORT_CLASSES": ("Integer", None, "classes with fewer candidates than PER_CLASS"),
        "SECONDS": ("Double", "s", "wall time"),
    },
    "Class": {
        "CLASS": ("String", None, "the class"),
        "DEFINITION": ("String", None, "SMARTS or ring rule"),
        "QM9_COUNT_QUOTED": ("Integer", None, "the count the user quoted for QM9 (0: none quoted)"),
        "N_TARGETS": ("Integer", None, "molecules of the class among the gated targets"),
        "N_CANDIDATES": ("Integer", None, "of those, outside SPICE"),
        "N_DRAWN": ("Integer", None, "drawn for this class"),
        "SHORTFALL": ("Integer", None, "PER_CLASS - N_DRAWN when the class is short, else 0"),
        "N_IN_UNION": ("Integer", None, "molecules of the union that belong to the class (drawn for it or for another)"),
    },
}


# ====================================================================== classes
def load_classes(path=CLASSES_FILE):
    """The class table of the YAML: a list of dicts (name, smarts | rule, note, qm9_count)."""
    import yaml
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out = []
    for c in doc["classes"]:
        if not (("smarts" in c) ^ ("rule" in c)):
            raise ValueError("class {!r} must have exactly one of smarts / rule".format(c.get("name")))
        out.append(dict(name=str(c["name"]), smarts=c.get("smarts"), rule=c.get("rule"),
                        note=str(c.get("note", "")), qm9_count=int(c.get("qm9_count", 0))))
    return out


def definition(cls):
    if cls.get("smarts"):
        return cls["smarts"]
    return "rule " + " ".join("{}={}".format(k, v) for k, v in sorted(cls["rule"].items()))


class Classifier:
    """SMARTS compiled once; `classify(smiles)` -> set of class names."""

    def __init__(self, classes=None):
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        self.classes = classes if classes is not None else load_classes()
        self._patterns = {}
        for c in self.classes:
            if c.get("smarts"):
                pat = Chem.MolFromSmarts(c["smarts"])
                if pat is None:
                    raise ValueError("class {!r}: RDKit cannot parse SMARTS {!r}".format(c["name"], c["smarts"]))
                self._patterns[c["name"]] = pat

    def names(self):
        return [c["name"] for c in self.classes]

    def classify(self, smiles):
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return set()
        out = set()
        ring = mol.GetRingInfo()
        rings = [set(r) for r in ring.AtomRings()]
        for c in self.classes:
            if c.get("smarts"):
                if mol.HasSubstructMatch(self._patterns[c["name"]]):
                    out.add(c["name"])
                continue
            rule = c["rule"]
            ok = True
            if "min_rings" in rule and len(rings) < int(rule["min_rings"]):
                ok = False
            if ok and rule.get("fused"):
                ok = any(rings[i] & rings[j] for i in range(len(rings)) for j in range(i + 1, len(rings)))
            if ok and "aromatic_rings" in rule:
                from rdkit.Chem import rdMolDescriptors
                ok = rdMolDescriptors.CalcNumAromaticRings(mol) >= int(rule["aromatic_rings"])
            if ok and "ring_size" in rule:
                ok = any(len(r) == int(rule["ring_size"]) for r in rings)
            if ok:
                out.add(c["name"])
        return out


def classify(smiles, classes=None):
    return Classifier(classes).classify(smiles)


def census(smiles_iter, classifier=None):
    """Counts per class over an iterable of SMILES; also `_n` (molecules) and `_unparsed`."""
    cf = classifier or Classifier()
    counts = {n: 0 for n in cf.names()}
    n = unparsed = 0
    from rdkit import Chem
    for s in smiles_iter:
        n += 1
        if Chem.MolFromSmiles(s) is None:
            unparsed += 1
            continue
        for name in cf.classify(s):
            counts[name] += 1
    counts["_n"] = n
    counts["_unparsed"] = unparsed
    return counts


def suspects(counts, classes, tol=SUSPECT_TOL):
    """Classes whose census count is more than `tol` off the quoted QM9 count."""
    out = []
    for c in classes:
        q = c.get("qm9_count") or 0
        if q and abs(counts.get(c["name"], 0) - q) > tol * q:
            out.append((c["name"], counts.get(c["name"], 0), q))
    return out


# ====================================================================== the draw
def draw(root, tag, name, per_class=PER_CLASS, seed=SEED, classes_file=CLASSES_FILE,
         membership_file=dataset_mod.MEMBERSHIP_FILE, exclude_training=True, pinned=dataset_mod.PINNED):
    """The molecule list of the campaign (module docstring). Returns the rows written to
    `draw.dat`, sorted by qm9_index."""
    t0 = time.time()
    classes = load_classes(classes_file)
    cf = Classifier(classes)
    rows = dat.read_table(Path(membership_file))
    n_targets = len(rows)
    per_mol = {}
    for r in rows:
        per_mol[r["qm9_index"]] = dict(qm9_index=r["qm9_index"], smiles=r["smiles"], n_heavy=int(r["n_heavy"]),
                                       in_training=bool(r["in_training"]), classes=cf.classify(r["smiles"]))
    candidates = [m for m in per_mol.values() if not (exclude_training and m["in_training"])]
    by_class_targets = {c["name"]: sum(1 for m in per_mol.values() if c["name"] in m["classes"]) for c in classes}
    chosen = {q for q in pinned if q in per_mol}
    class_rows = []
    for c in classes:
        ids = sorted(m["qm9_index"] for m in candidates if c["name"] in m["classes"])
        n = min(int(per_class), len(ids))
        picked = sorted(dataset_mod._rng(seed, "draw", c["name"]).choice(ids, size=n, replace=False).tolist()) if n else []
        chosen.update(picked)
        class_rows.append(dict(CLASS=c["name"], DEFINITION=definition(c), QM9_COUNT_QUOTED=int(c.get("qm9_count") or 0),
                               N_TARGETS=by_class_targets[c["name"]], N_CANDIDATES=len(ids), N_DRAWN=n,
                               SHORTFALL=max(0, int(per_class) - n), N_IN_UNION=0))
    out_rows = []
    for q in sorted(chosen):
        m = per_mol[q]
        out_rows.append(dict(qm9_index=q, smiles=m["smiles"], n_heavy=m["n_heavy"],
                             classes=";".join(sorted(m["classes"])) or "-", in_training=m["in_training"],
                             pinned=q in pinned))
    for cr in class_rows:
        cr["N_IN_UNION"] = sum(1 for r in out_rows if cr["CLASS"] in r["classes"].split(";"))
    d = dataset_mod.datasets_dir(root, tag, name)
    d.mkdir(parents=True, exist_ok=True)
    info = dict(NAME=str(name), TAG=str(tag), CLASSES_FILE=str(classes_file), MEMBERSHIP_FILE=str(membership_file),
                PER_CLASS=int(per_class), SEED=int(seed), EXCLUDE_TRAINING=bool(exclude_training),
                N_TARGETS=n_targets, N_CANDIDATES=len(candidates), N_CLASSES=len(classes), N_DRAWN=len(out_rows),
                N_PINNED=sum(1 for r in out_rows if r["pinned"]),
                N_SHORT_CLASSES=sum(1 for cr in class_rows if cr["SHORTFALL"]), SECONDS=time.time() - t0)
    missing = prop.write(d / (STEP + ".toml"), {"Calculation_Info": info, "Class": class_rows}, SCHEMA,
                         prop.NORMAL_TERMINATION, PROGNAME)
    if missing:
        raise RuntimeError("draw.toml keys outside the schema: {}".format(missing))
    dat.write_table(d / (STEP + ".dat"), out_rows, list(ROW_SCHEMA), ROW_SCHEMA)
    rep = report.Report(PROGNAME, "the structure-class draw of Dataset {!r}".format(name))
    rep.section("conventions")
    for k in ("TAG", "PER_CLASS", "SEED", "EXCLUDE_TRAINING", "N_TARGETS", "N_CANDIDATES", "N_CLASSES", "N_DRAWN",
              "N_PINNED", "N_SHORT_CLASSES"):
        rep.kv(k, info[k])
    rep.section("per class (targets: gated QM9; candidates: outside SPICE; union: drawn for this or another class)")
    rep.table(["class", "quoted", "targets", "candidates", "drawn", "shortfall", "in union", "definition"],
              [[cr["CLASS"], cr["QM9_COUNT_QUOTED"] or "-", cr["N_TARGETS"], cr["N_CANDIDATES"], cr["N_DRAWN"],
                cr["SHORTFALL"] or "-", cr["N_IN_UNION"], cr["DEFINITION"]] for cr in class_rows])
    rep.note("per class, PER_CLASS molecules drawn with a generator seeded from (seed, class) over the sorted "
             "candidates; the union over classes is the campaign's list, a molecule counting for every class it "
             "belongs to; a short class takes all its candidates and its shortfall is stated (round 5, Q1 (a)+(c)). "
             "Candidates are the gated QM9 targets outside MACE-OFF23's SPICE training file at any match level "
             "(Q5 (a)). The pinned seven are always in.")
    rep.write(d / (STEP + ".out"), step=STEP)
    return out_rows


def read_draw(root, tag, name):
    p = dataset_mod.datasets_dir(root, tag, name) / (STEP + ".dat")
    if not p.is_file():
        raise FileNotFoundError("no draw at {} (run 00_draw first)".format(p))
    return dat.read_table(p)
