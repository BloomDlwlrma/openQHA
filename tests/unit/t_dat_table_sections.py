"""A Table holds named sections with a comment per column, and reads back (ticket 01,
collect-one-table, 2026-09-16).

UNIT. No engine; under a second.

The ruling (user, 2026-09-16, ADR 0003 amendment): collect leaves ONE `.dat` with a
`[section]` line before each of its tables and, above each header, one comment line per
column in the Property file's form (`Type, unit: doc`). `openqha.store.dat` writes and
reads it. The rules checked here: `read_tables(write_tables(sections)) == sections` with
the value fidelity of the single-table round trip; every column the schema knows gets a
comment and every column it does not know is returned; a file without sections still
reads through both readers; the single-table reader refuses a sectioned file by naming
its sections; a string beginning with `[` cannot be mistaken for a section line.
"""
import math
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


SCHEMA = {
    "trajectories": {
        "species": ("String", None, "the molecule"),
        "n_frames": ("Integer", None, "frames the covariance was formed from"),
        "TS_QH_kcal": ("Double", "kcal/mol", "T*S from the quasi-harmonic entropy"),
        "note": ("String", None, "free text"),
    },
    "blank": {
        "species": ("String", None, "the molecule"),
        "standard_error_kcal": ("Double", "kcal/mol", "sample standard deviation over sqrt(n_seeds); NA with one seed"),
    },
    "assembly": {"species": ("String", None, "the molecule"), "ok": ("Boolean", None, "")},
}


def main():
    from openqha.store import dat
    traj = [dict(species="dsgdb9nsd_000018", n_frames=3000, TS_QH_kcal=3.215, note="[not a section] two  spaces"),
            dict(species="dsgdb9nsd_000018", n_frames=5, TS_QH_kcal=float("nan"), note='say "hi"')]
    blank = [dict(species="dsgdb9nsd_000018", standard_error_kcal=None, undocumented=float("inf"))]
    assembly = []
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "collect.dat"
        unknown = dat.write_tables(p, [("trajectories", traj), ("blank", blank),
                                       ("assembly", (assembly, ["species", "ok"]))], SCHEMA)
        text = p.read_text(encoding="utf-8")
        lines = text.splitlines()
        print("A. the file")
        check("three [section] lines in order",
              [l for l in lines if l.startswith("[")] == ["[trajectories]", "[blank]", "[assembly]"], lines)
        i = lines.index("[trajectories]")
        check("one comment per column, Property-file form, before the header",
              lines[i + 1].startswith("#   species") and "String: the molecule" in lines[i + 1]
              and "Double, kcal/mol: T*S" in lines[i + 3] and lines[i + 5].split() == ["#", "species", "n_frames", "TS_QH_kcal", "note"],
              lines[i:i + 7])
        check("an empty section still has its comments and header",
              lines[lines.index("[assembly]") + 3].split() == ["#", "species", "ok"], lines[-4:])
        check("the writer returns the column the schema does not know",
              unknown == [("blank", "undocumented")], unknown)
        check("a string beginning with [ is quoted", '"[not a section]' in text, text)
        print("B. the round trip")
        back = dat.read_tables(p)
        check("sections come back in file order", list(back) == ["trajectories", "blank", "assembly"], list(back))
        check("rows, keys and values of the trajectories section",
              len(back["trajectories"]) == 2 and back["trajectories"][0]["n_frames"] == 3000
              and back["trajectories"][0]["TS_QH_kcal"] == 3.215 and math.isnan(back["trajectories"][1]["TS_QH_kcal"])
              and back["trajectories"][0]["note"] == traj[0]["note"] and back["trajectories"][1]["note"] == traj[1]["note"],
              back["trajectories"])
        check("NA and inf survive", back["blank"][0]["standard_error_kcal"] is None
              and math.isinf(back["blank"][0]["undocumented"]), back["blank"])
        check("the empty section reads as []", back["assembly"] == [], back["assembly"])
        check("read_table on a sectioned file refuses by naming the sections", _refuses_sections(dat, p))
        print("C. a file without sections")
        q = Path(t) / "plain.dat"
        dat.write_table(q, traj, schema=SCHEMA["trajectories"])
        qtext = q.read_text(encoding="utf-8")
        check("no [section] line, comments then header", not qtext.startswith("[")
              and qtext.splitlines()[0].startswith("#   species") and qtext.splitlines()[4].startswith("# species"), qtext)
        check("read_table reads it as before", dat.read_table(q)[1]["n_frames"] == 5, dat.read_table(q))
        check("read_tables gives it under the empty name", list(dat.read_tables(q)) == [""] and len(dat.read_tables(q)[""]) == 2,
              dat.read_tables(q))
        old = Path(t) / "old.dat"
        old.write_text("# a b\n1 x\n# a trailing comment\n2 y\n", encoding="utf-8")
        check("a header-only file of the old form, with a comment after the rows",
              dat.read_table(old) == [dict(a=1, b="x"), dict(a=2, b="y")], dat.read_table(old))
        check("a header-only table reads as [] through both readers",
              dat.read_table(dat.write_table(Path(t) / "e.dat", [])) == [] and dat.read_tables(Path(t) / "e.dat") == {"": []})
        (Path(t) / "none.dat").write_text("", encoding="utf-8")
        check("an empty file is no table at all", dat.read_tables(Path(t) / "none.dat") == {}
              and dat.read_table(Path(t) / "none.dat") == [])
        check("a bad section name is refused", _refuses_name(dat, Path(t) / "bad.dat"))
    print("\n{}".format("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL)))
    return 0 if not FAIL else 1


def _refuses_sections(dat, p):
    try:
        dat.read_table(p)
        return False
    except ValueError as exc:
        return "trajectories" in str(exc) and "blank" in str(exc)


def _refuses_name(dat, p):
    try:
        dat.write_tables(p, [("two words", [])])
        return False
    except ValueError as exc:
        return "two words" in str(exc)


if __name__ == "__main__":
    raise SystemExit(main())
