"""Ticket 05 of the Hessian-learning set: structure classes and the draw.

Asserted: `classify` on known SMILES gives the expected class sets (oxetane, 2-methyl-
oxirane, cyclopropanol, acetamide, propanal, acetone, benzene, cyclooctane,
bicyclo[1.1.0]butane, acetonitrile, propyne, dimethyl ether, trimethylamine, acetic acid,
methyl acetate); every class of the YAML has exactly one of smarts / rule and every SMARTS
compiles; `census` counts and flags a suspect against a quoted count; `draw` on a fake
membership table: per class min(per_class, candidates) molecules, none in_training, the
union with multi-class molecules counted for every class, the shortfall row, the pinned
molecule always in, the same seed reproducing, `draw.dat` round-tripping.
"""
import sys
import tempfile
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))
from openqha.data import dataset, structure_classes as sc     # noqa: E402
from openqha.store import dat, property as prop               # noqa: E402

FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:220]))
    if not ok:
        FAIL.append(label)


CASES = {
    "C1COC1": {"small_ring", "heterocyclic", "dialkyl_ether"},                               # oxetane
    "CC1CO1": {"three_membered_ring", "small_ring", "heterocyclic", "dialkyl_ether", "epoxide"},  # 2-methyloxirane
    "OC1CC1": {"three_membered_ring", "small_ring", "cyclopropane", "secondary_alcohol"},   # cyclopropanol
    "CC(N)=O": {"amide"},                                                                    # acetamide
    "CCC=O": {"aldehyde"},                                                                   # propanal
    "CC(C)=O": {"ketone"},                                                                   # acetone
    "c1ccccc1": {"aromatic"},                                                                # benzene
    "C1CCCCCCC1": {"eight_membered_ring"},                                                   # cyclooctane
    "C1C2C1C2": {"three_membered_ring", "small_ring", "bicyclic", "cyclopropane"},          # bicyclo[1.1.0]butane
    "CC#N": {"carbonitrile"},                                                                # acetonitrile
    "CC#C": {"alkyne"},                                                                      # propyne
    "COC": {"dialkyl_ether"},                                                                # dimethyl ether
    "CN(C)C": {"trialkylamine", "tertiary_amine", "aliphatic_amine"},                        # trimethylamine
    "CC(=O)O": {"carboxylic_acid"},                                                          # acetic acid
    "CC(=O)OC": {"ester"},                                                                   # methyl acetate
    "CCO": {"primary_alcohol"},                                                              # ethanol
    "Nc1ccccc1": {"aromatic", "aromatic_amine"},                                             # aniline
    "CCN": {"aliphatic_amine"},                                                              # ethylamine
}


def main():
    classes = sc.load_classes()
    cf = sc.Classifier(classes)
    check("the YAML: {} classes, each with exactly one of smarts / rule, every SMARTS compiled".format(len(classes)),
          len(classes) >= 20 and all(bool(c["smarts"]) ^ bool(c["rule"]) for c in classes))
    bad = {s: (cf.classify(s), want) for s, want in CASES.items() if cf.classify(s) != want}
    check("classify on {} known SMILES gives exactly the expected class sets".format(len(CASES)), not bad, bad)

    counts = sc.census(CASES.keys(), cf)
    fake = [dict(c, qm9_count=0) for c in classes]       # only the two below carry a quoted count here
    for c in fake:
        if c["name"] == "aromatic":
            c["qm9_count"] = 2                 # census says 2 (benzene, aniline): not suspect
        if c["name"] == "amide":
            c["qm9_count"] = 10                # census says 1: suspect
    sus = sc.suspects(counts, fake)
    check("census counts (aromatic 2, amide 1, eight-membered 1) and the suspect test flags amide (1 vs quoted 10), not aromatic",
          counts["aromatic"] == 2 and counts["amide"] == 1 and counts["eight_membered_ring"] == 1 and counts["_n"] == len(CASES)
          and [s[0] for s in sus] == ["amide"], (counts, sus))

    # --- the draw on a fake membership table -------------------------------------------
    with tempfile.TemporaryDirectory(prefix="draw_") as tmp:
        root = Path(tmp)
        rows = []
        i = 0
        def add(smiles, in_training=False, n=1):
            nonlocal i
            for _ in range(n):
                i += 1
                rows.append(dict(qm9_index="dsgdb9nsd_{:06d}".format(i), smiles=smiles, n_heavy=4, in_training=in_training,
                                 in_test_only=False, match_level="none" if not in_training else "isomeric",
                                 n_train_frames=0, n_test_frames=0, n_train_dimer_frames=0, n_test_dimer_frames=0, config_types=""))
        add("CC1CO1", n=6)                 # epoxide + ether + heterocyclic + small ring: 6 candidates
        add("CC1CO1", in_training=True)    # excluded
        add("CCC=O", n=3)                  # aldehyde: 3 candidates (short of 4)
        add("CC(C)=O", n=5)                # ketone: 5
        add("C1CCCCCCC1", n=1)             # eight-membered: 1 (short)
        rows.append(dict(qm9_index="dsgdb9nsd_000035", smiles="CCC=O", n_heavy=4, in_training=True, in_test_only=False,
                         match_level="isomeric", n_train_frames=49, n_test_frames=1, n_train_dimer_frames=0,
                         n_test_dimer_frames=0, config_types=""))     # propanal: pinned, in SPICE, still in
        mem = root / "membership.dat"
        dat.write_table(mem, rows, list(rows[0]))
        small = [c for c in classes if c["name"] in ("epoxide", "dialkyl_ether", "aldehyde", "ketone", "eight_membered_ring", "amide")]
        import yaml
        cfile = root / "classes.yaml"
        cfile.write_text(yaml.safe_dump({"classes": [{k: v for k, v in c.items() if v not in (None, "", 0)} for c in small]}), encoding="utf-8")
        out = sc.draw(root, "t", "d", per_class=4, seed=3, classes_file=cfile, membership_file=mem, pinned=("dsgdb9nsd_000035",))
        d = dataset.datasets_dir(root, "t", "d")
        rec = prop.load(d / "draw.toml")
        by = {r["CLASS"]: r for r in rec["Class"]}
        drawn_epox = [r for r in out if "epoxide" in r["classes"].split(";")]
        check("per class min(4, candidates): epoxide 4 of 6 (the SPICE one excluded), ketone 4 of 5, aldehyde 3 of 3 (short 1), eight-membered 1 (short 3), amide 0 (short 4)",
              (by["epoxide"]["N_CANDIDATES"], by["epoxide"]["N_DRAWN"]) == (6, 4) and by["ketone"]["N_DRAWN"] == 4
              and (by["aldehyde"]["N_DRAWN"], by["aldehyde"]["SHORTFALL"]) == (3, 1)
              and (by["eight_membered_ring"]["N_DRAWN"], by["eight_membered_ring"]["SHORTFALL"]) == (1, 3)
              and (by["amide"]["N_DRAWN"], by["amide"]["SHORTFALL"]) == (0, 4) and rec["Calculation_Info"]["N_SHORT_CLASSES"] == 3,
              {k: (v["N_CANDIDATES"], v["N_DRAWN"], v["SHORTFALL"]) for k, v in by.items()})
        check("the union: a molecule drawn for epoxide counts for dialkyl_ether too (N_IN_UNION >= N_DRAWN); no drawn row in_training except the pinned one; propanal pinned and present",
              by["dialkyl_ether"]["N_IN_UNION"] >= by["dialkyl_ether"]["N_DRAWN"]
              and all(not r["in_training"] or r["pinned"] for r in out)
              and any(r["qm9_index"] == "dsgdb9nsd_000035" and r["pinned"] for r in out)
              and len(drawn_epox) == by["epoxide"]["N_IN_UNION"], (len(out), by["dialkyl_ether"]))
        again = sc.draw(root, "t", "d", per_class=4, seed=3, classes_file=cfile, membership_file=mem, pinned=("dsgdb9nsd_000035",))
        other = sc.draw(root, "t", "d2", per_class=4, seed=4, classes_file=cfile, membership_file=mem, pinned=("dsgdb9nsd_000035",))
        ids = lambda rs: [r["qm9_index"] for r in rs]    # noqa: E731
        check("the same seed reproduces the draw; another seed changes it", ids(again) == ids(out) and ids(other) != ids(out))
        back = sc.read_draw(root, "t", "d")
        check("draw.dat round-trips (classes ';'-joined, pinned Boolean, n_heavy Integer)",
              ids(back) == ids(out) and back[0]["classes"] == out[0]["classes"] and isinstance(back[0]["pinned"], bool)
              and back[0]["n_heavy"] == 4 and prop.status_of(d / "draw.toml") == prop.NORMAL_TERMINATION)

    print("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
