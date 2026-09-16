"""Collect's tables are whitespace .dat files that read back as they were written (ticket 13).

UNIT. No engine; under a second.

The ruling (user, 2026-09-15): the tables collect writes (per-trajectory entropies, the
criteria, the assembly, the blank control) take the form CREST's `crest.energies` has --
whitespace-separated columns, one header line -- instead of parquet, and the ensemble
report reads them with plain Python. `openqha.store.dat` is the writer and reader; the
rule: `read_table(write_table(rows)) == rows`, with ints, floats (nan, inf), booleans,
None and strings that contain spaces or quotes all surviving.
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


def main():
    from openqha.store import dat
    rows = [
        dict(species="dsgdb9nsd_000018", basin="basin00", seed="seed00", n_frames=3000,
             TS_QH_kcal=3.2150, passed=True, note="the saturation curve is drawn", missing=None),
        dict(species="dsgdb9nsd_000018", basin="basin01", seed="seed00", n_frames=3000,
             TS_QH_kcal=float("nan"), passed=False, note='say "hi" to  two  spaces', missing=1.5),
    ]
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "plain.dat"
        dat.write_table(p, rows)
        text = p.read_text(encoding="utf-8")
        print("A. the file")
        check("first line is a # header with the column names in order",
              text.splitlines()[0].split() == ["#"] + list(rows[0].keys()), text.splitlines()[0])
        check("one row per line after the header", len(text.splitlines()) == 3, text)
        back = dat.read_table(p)
        print("B. the round trip")
        check("two rows, same keys", len(back) == 2 and list(back[0].keys()) == list(rows[0].keys()), back)
        check("int, float, bool, str", back[0]["n_frames"] == 3000 and back[0]["TS_QH_kcal"] == 3.215
              and back[0]["passed"] is True and back[1]["passed"] is False and back[0]["basin"] == "basin00", back)
        check("nan survives", math.isnan(back[1]["TS_QH_kcal"]), back[1]["TS_QH_kcal"])
        check("None survives (NA)", back[0]["missing"] is None and back[1]["missing"] == 1.5, back)
        check("strings with spaces and quotes survive", back[0]["note"] == rows[0]["note"]
              and back[1]["note"] == rows[1]["note"], [r["note"] for r in back])
        print("C. edge cases")
        check("empty table: header only, reads back as []", dat.read_table(dat.write_table(Path(t) / "e.dat", [], columns=["a", "b"])) == [])
        check("a missing file is refused by name", _refuses(dat, Path(t) / "nope.dat"))
    print("\n{}".format("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL)))
    return 0 if not FAIL else 1


def _refuses(dat, p):
    try:
        dat.read_table(p)
        return False
    except FileNotFoundError as exc:
        return str(p) in str(exc)


if __name__ == "__main__":
    raise SystemExit(main())
