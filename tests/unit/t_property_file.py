"""A Calculation's Property file: ORCA `.property.txt` carried into TOML (records redesign, ticket 16).

UNIT. No engine; under a second. Needs Python 3.11 (tomllib).

The ruling (user, 2026-09-15, grilling Q2 and Q3): the `.toml` beside a Report holds a
`[Calculation_Status]` block first (PROGNAME, VERSION, STATUS), a `[Calculation_Info]`
block with the inputs, and the result blocks a later step reads; nothing else. Keys are
upper-case, each followed by a `# Type, unit: doc` comment taken from one schema table
so every file carries the same comments; repeated things are arrays of tables with an
INDEX key. `openqha.store.property` is the writer; the standard library reads it back
and ignores the comments.

    A. the file starts with [Calculation_Status]; STATUS, PROGNAME, VERSION read back
    B. every value reads back unchanged (scalars, arrays, arrays of tables with INDEX)
    C. every key carries its schema comment; a key outside the schema has none and is reported
    D. status_of() reads STATUS from a file, None when the file or the block is absent
    E. a Report and its Property file share a stem, the setting in the stem (layout rule)
"""
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
FAIL = []


def check(label, ok, detail=""):
    print("  {:60s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:300]))
    if not ok:
        FAIL.append(label)


def main():
    if sys.version_info < (3, 11):
        print("  needs Python 3.11 (tomllib); skipped")
        return 0
    import tomllib
    from openqha.store import property as prop
    from openqha.store import layout

    schema = {
        "Calculation_Info": {
            "QM9_INDEX": ("String", None, "the molecule"),
            "TSTEP": ("Double", "fs", "CREST MD time step"),
            "SHAKE": ("Integer", None, "SHAKE on all bonds"),
            "THREADS": ("Integer", None, ""),
        },
        "Census": {
            "N_BASINS": ("Integer", None, "after tightening and deduplication"),
            "BASIN_CONFORMER_IDS": ("ArrayOfIntegers", None, "input frame of each basin"),
        },
        "Basin": {
            "ENERGY": ("Double", "eV", "electronic energy after tightening"),
            "SIGMA": ("Integer", None, "symmetry number"),
        },
        "Criteria": {
            "ALL_PASSED": ("Boolean", None, "what the batch driver reads"),
        },
    }
    blocks = {
        "Calculation_Info": {"QM9_INDEX": "dsgdb9nsd_000035", "TSTEP": 5.0, "SHAKE": 2, "THREADS": 4,
                             "NOT_IN_SCHEMA": "x"},
        "Census": {"N_BASINS": 2, "BASIN_CONFORMER_IDS": [0, 3]},
        "Basin": [{"INDEX": 0, "ENERGY": -5259.18895637, "SIGMA": 1},
                  {"INDEX": 1, "ENERGY": -5259.15270342, "SIGMA": 3}],
        "Criteria": {"ALL_PASSED": True},
    }

    print("A. the file starts with [Calculation_Status]:")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "branchA.toml"
        missing = prop.write(path, blocks, schema, status=prop.NORMAL_TERMINATION,
                             progname="openQHA branchA")
        text = path.read_text(encoding="utf-8")
        first = [l for l in text.splitlines() if l.strip() and not l.startswith("#")][0]
        check("first block is [Calculation_Status]", first == "[Calculation_Status]", first)
        back = tomllib.loads(text)
        st = back.get("Calculation_Status", {})
        check("STATUS reads back", st.get("STATUS") == "NORMAL TERMINATION", st)
        check("PROGNAME reads back", st.get("PROGNAME") == "openQHA branchA", st)
        check("VERSION is openqha's", st.get("VERSION") == prop.VERSION and prop.VERSION, st)
        check("no .part left behind", not (Path(td) / "branchA.toml.part").exists())

        print("B. every value reads back unchanged:")
        check("Calculation_Info scalars", back["Calculation_Info"]["TSTEP"] == 5.0
              and back["Calculation_Info"]["SHAKE"] == 2 and back["Calculation_Info"]["QM9_INDEX"] == "dsgdb9nsd_000035")
        check("array", back["Census"]["BASIN_CONFORMER_IDS"] == [0, 3])
        check("[[Basin]] is an array of tables", isinstance(back["Basin"], list) and len(back["Basin"]) == 2)
        check("INDEX and values per basin", back["Basin"][1] == {"INDEX": 1, "ENERGY": -5259.15270342, "SIGMA": 3},
              back["Basin"][1])
        check("boolean", back["Criteria"]["ALL_PASSED"] is True)
        check("block order is the given order",
              [k for k in back] == ["Calculation_Status", "Calculation_Info", "Census", "Basin", "Criteria"],
              list(back))

        print("C. schema comments:")
        lines = {l.split("=")[0].strip(): l for l in text.splitlines() if "=" in l}
        check("type, unit and doc", lines["TSTEP"].rstrip().endswith("# Double, fs: CREST MD time step"), lines["TSTEP"])
        check("type and doc, no unit", lines["SHAKE"].rstrip().endswith("# Integer: SHAKE on all bonds"), lines["SHAKE"])
        check("type only when the doc is empty", lines["THREADS"].rstrip().endswith("# Integer"), lines["THREADS"])
        check("array of tables keys get comments too", lines["ENERGY"].rstrip().endswith("# Double, eV: electronic energy after tightening"),
              lines["ENERGY"])
        check("INDEX needs no schema entry", "#" not in lines["INDEX"] or "Integer" in lines["INDEX"], lines["INDEX"])
        check("a key outside the schema has no comment", "#" not in lines["NOT_IN_SCHEMA"], lines["NOT_IN_SCHEMA"])
        check("and is reported", missing == ["Calculation_Info.NOT_IN_SCHEMA"], missing)
        check("keys are upper-case in the file", all(k == k.upper() for k in lines), [k for k in lines if k != k.upper()])

        print("D. status_of():")
        check("reads STATUS", prop.status_of(path) == "NORMAL TERMINATION")
        prop.write(path, blocks, schema, status=prop.RUNNING, progname="openQHA md")
        check("RUNNING after a rewrite", prop.status_of(path) == "RUNNING")
        check("None for an absent file", prop.status_of(Path(td) / "nope.toml") is None)
        (Path(td) / "old.toml").write_text('name = "x"\n[settings]\nshake = 2\n', encoding="utf-8")
        check("None for a file without the block", prop.status_of(Path(td) / "old.toml") is None)
        check("load() gives the dict", prop.load(path)["Census"]["N_BASINS"] == 2)

    print("E. the records path rule (no setting level; the setting is in the stem):")
    m = Path("/r/tag/dsgdb9nsd_000035")
    check("md records dir openmm", layout.md_records_dir(m, "openmm"), m / "_records/md_openmm")
    check("md records dir ase", layout.md_records_dir(m, "ase"), m / "_records/md_ase")
    check("basin records dir", layout.basin_records_dir(m, "openmm", 1), m / "_records/md_openmm/basin01")
    check("record file name, default", layout.record_file_name("md.toml", "default"), "md.toml")
    check("record file name, s2", layout.record_file_name("md.toml", "s2"), "md_s2.toml")
    check("record file name, two dots", layout.record_file_name("thermo.msrrho.dat", "s2"),
          "thermo_s2.msrrho.dat")
    check("driver log follows the rule", layout.record_file_name("driver.log", "p1500_s5"), "driver_p1500_s5.log")
    check("report and property file share a stem",
          layout.record_file_name("md.out", "s2")[:-4] == layout.record_file_name("md.toml", "s2")[:-5])
    paths = [layout.md_records_dir(m, "openmm"), layout.basin_records_dir(m, "ase", 0),
             layout.md_records_dir(m, "openmm") / layout.record_file_name("collect.out", "default")]
    check("the literal 'default' is in no path", all("default" not in str(p) for p in paths), paths)

    print()
    print("FAILED: {}".format(FAIL) if FAIL else "all checks passed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
