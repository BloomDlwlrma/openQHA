"""Records are written as TOML and read back with the standard library.

UNIT. No engine; under a second. Needs Python 3.11 (tomllib).

The five records this repository writes about a run take
CREST's own settings format, TOML, for the fields a program reads back, and a `.out`
text report for a person. The standard library reads TOML but does not write it, so
`openqha.store.toml_out.dumps` is the writer, and this test holds it to one rule: what
`tomllib.loads` reads back equals what was given, minus the Nones (TOML has no null:
a None key is simply absent, and readers already use `.get`).

    A. scalars of every kind, nested tables, arrays of tables, nested arrays, nan
    B. a key that is not a bare TOML key is quoted, and a value with quotes and newlines survives
    C. None is dropped at every depth; None inside a list becomes the string "None"
    D. a real branch A record shape (basins as an array of tables with sub-tables)
    E. the .out report ends with the terminal line, and `terminated_normally` finds it
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
    try:
        import tomllib
    except ImportError as exc:
        print("SKIP: {}".format(exc))
        return 0
    from openqha.store import toml_out, report

    print("A. every value kind round-trips")
    rec = dict(name="dsgdb9nsd_000018", n=3, x=-5697.0238581204, ok=True, nan=float("nan"),
               inf=float("inf"), words=["a", "b"], numbers=[1, 2.5, 3],
               matrix=[[1.0, 2.0], [3.0, 4.0]], nested=dict(deep=dict(k="v", z=0)),
               basins=[dict(index=0, energy_eV=-1.0, hessian=dict(n_imaginary=0, freqs=[10.0, 20.0])),
                       dict(index=1, energy_eV=-0.5, hessian=dict(n_imaginary=1, freqs=[-5.0]))])
    text = toml_out.dumps(rec)
    back = tomllib.loads(text)
    check("scalars, lists, nested tables, arrays of tables",
          back["name"] == rec["name"] and back["n"] == 3 and back["x"] == rec["x"] and back["ok"] is True
          and back["words"] == ["a", "b"] and back["numbers"] == [1, 2.5, 3]
          and back["matrix"] == rec["matrix"] and back["nested"]["deep"] == dict(k="v", z=0)
          and back["basins"][1]["hessian"]["freqs"] == [-5.0] and back["basins"][0]["hessian"]["n_imaginary"] == 0,
          text)
    check("nan and inf", math.isnan(back["nan"]) and back["inf"] == float("inf"), (back.get("nan"), back.get("inf")))

    print("B. keys and strings")
    rec = {"a key with spaces": 1, "unicode-ключ": 2, "quote": 'say "hi"\nsecond line', "path": "C:\\x\\y"}
    back = tomllib.loads(toml_out.dumps(rec))
    check("non-bare keys are quoted and read back", back["a key with spaces"] == 1 and back["unicode-ключ"] == 2, back)
    check("quotes, newline and backslash survive", back["quote"] == rec["quote"] and back["path"] == rec["path"], back)

    print("C. None")
    rec = dict(a=None, b=dict(c=None, d=1), e=[dict(f=None, g=2)], h=[1, None])
    back = tomllib.loads(toml_out.dumps(rec))
    check("None keys absent at every depth", "a" not in back and "c" not in back["b"] and "f" not in back["e"][0], back)
    check("kept siblings", back["b"]["d"] == 1 and back["e"][0]["g"] == 2, back)
    check("None inside a list becomes the string 'None'", back["h"] == [1, "None"], back["h"])

    print("D. a branch A record shape")
    rec = dict(generated_by="s0_A_pipeline.py", qm9_index="dsgdb9nsd_000018",
               settings=dict(shake=2, tstep_fs=5.0, hydrogen_mass_amu=None),
               crest=dict(workdir="/x/crest", n_conformers=2, products={"crest_conformers.xyz": 610}),
               basins=[dict(index=0, energy_eV=-5697.02, relative_kcal=0.0, symmetry_number=1,
                            electronic_degeneracy=1, point_group="C1",
                            hessian=dict(n_imaginary=0, lowest_frequency_cm_inv=35.2)),
                       dict(index=1, energy_eV=-5697.01, relative_kcal=0.6, symmetry_number=1,
                            electronic_degeneracy=1, point_group="C1",
                            hessian=dict(n_imaginary=0, lowest_frequency_cm_inv=34.7))],
               criteria=[dict(criterion="workhorse identity", passed=True, measured="gfn2"),
                         dict(criterion="cost", passed=False, measured=None)],
               all_criteria_passed=False)
    back = tomllib.loads(toml_out.dumps(rec))
    check("basins read back as a list of dicts with sub-tables",
          [b["relative_kcal"] for b in back["basins"]] == [0.0, 0.6]
          and back["basins"][1]["hessian"]["lowest_frequency_cm_inv"] == 34.7, back.get("basins"))
    check("criteria list, None measured dropped", back["criteria"][1]["passed"] is False
          and "measured" not in back["criteria"][1], back["criteria"])
    check("dotted file name as a key survives", back["crest"]["products"]["crest_conformers.xyz"] == 610,
          back["crest"])
    check("None in settings dropped", "hydrogen_mass_amu" not in back["settings"], back["settings"])

    print("E. the .out report's terminal line")
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "branchA.out"
        r = report.Report("openQHA branch A", subtitle="test")
        r.kv("species", "dsgdb9nsd_000018")
        r.write(p, step="branch A")
        lines = p.read_text(encoding="utf-8").rstrip("\n").splitlines()
        check("last line is the terminal line", lines[-1] == "openQHA branch A terminated normally", lines[-3:])
        check("terminated_normally(path) finds it", report.terminated_normally(p, "branch A") is True)
        check("...and not another step's", report.terminated_normally(p, "collect") is False)
        q = Path(t) / "cut.out"
        q.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        check("a report cut before its last line is not finished", report.terminated_normally(q, "branch A") is False)

    print("\n{}".format("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL)))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
