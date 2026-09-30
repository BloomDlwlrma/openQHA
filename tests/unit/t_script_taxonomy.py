"""Unit test: every script declares its category, and the declaration matches its directory.

UNIT. Seconds; reads files only.

Why a declaration AND a directory, when either alone would place the script
--------------------------------------------------------------------------
Because classification is a JUDGEMENT, not a fact, and a
judgement that lives in only one place cannot be checked. A script that says
CALIBRATION while sitting in `production/` is a disagreement between two people --
or between the same person three weeks apart -- and this test is what makes that
disagreement visible instead of letting the directory win silently.

The categories, and the test each one has to pass:

    production/   its output goes into a deliverable; delete it and some edge has no number
    calibration/  its output is a number used to make a decision; it reaches a
                  checkpoint, not a deliverable
    diagnostics/  written to locate one numbered defect; still runnable, rarely run
    tooling/      it manages the repository itself (identifiers, format conversion,
                  notebook checks, migration) and produces no science

`diagnostics/` is currently EMPTY, and that is not an oversight: the closed-defect
probes live in `scripts/_superseded/closed-defects/` instead; each is meant to end as a
named regression test under `tests/regression/`. This test therefore accepts an
absent `diagnostics/` but rejects a script inside one that is not declared.

Run::  python tests/unit/t_script_taxonomy.py
"""
import re
import sys
from pathlib import Path


def _repo_root():
    """Directory holding the openQHA package, found by walking up from this file."""
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
SCRIPTS = ROOT / "scripts"
CATEGORIES = ("production", "calibration", "diagnostics", "tooling")

#: The declaration is a bare upper-case word on its own, inside the module docstring,
#: within the first few lines -- so a reader sees it before anything else.
_DECL = re.compile(r"^(PRODUCTION|CALIBRATION|DIAGNOSTIC|DIAGNOSTICS|TOOLING)\b", re.M)
_ALIAS = {"DIAGNOSTIC": "diagnostics", "DIAGNOSTICS": "diagnostics"}

FAIL = []


def declared_category(path, head_lines=12):
    """The category a script declares, or None. Only the head of the file counts."""
    text = "\n".join(path.read_text(encoding="utf-8").splitlines()[:head_lines])
    m = _DECL.search(text)
    if not m:
        return None
    word = m.group(1)
    return _ALIAS.get(word, word.lower())


def main():
    loose = sorted(p.name for p in SCRIPTS.glob("*.py"))
    print("A. no script sits loose directly in scripts/")
    if loose:
        FAIL.append("loose scripts: " + ", ".join(loose))
        print("  FAIL -- {} file(s) not in a category directory: {}".format(
            len(loose), ", ".join(loose)))
    else:
        print("  ok -- every script is inside a category directory")

    print("\nB. declaration present and matching its directory")
    n_checked = 0
    for category in CATEGORIES:
        d = SCRIPTS / category
        if not d.is_dir():
            print("  {:<14s} (directory absent)".format(category + "/"))
            continue
        files = sorted(d.glob("*.py")) + sorted(d.glob("*.sh"))
        print("  {:<14s} {} file(s)".format(category + "/", len(files)))
        for p in files:
            n_checked += 1
            got = declared_category(p)
            rel = p.relative_to(ROOT)
            if got is None:
                FAIL.append(str(rel) + ": no declaration")
                print("      FAIL  {:<46s} declares nothing".format(p.name))
            elif got != category:
                FAIL.append("{}: declares {}, sits in {}".format(rel, got, category))
                print("      FAIL  {:<46s} declares {}".format(p.name, got.upper()))
    print("  {} script(s) checked".format(n_checked))

    print("\nC. retired scripts are separated, not deleted")
    sup = SCRIPTS / "_superseded"
    n_sup = len(list(sup.rglob("*.py"))) + len(list(sup.rglob("*.sh"))) if sup.is_dir() else 0
    print("  _superseded/   {} file(s) kept as evidence".format(n_sup))
    if sup.is_dir() and not (sup / "README.md").is_file():
        FAIL.append("scripts/_superseded/README.md missing")
        print("      FAIL  no README saying why each group is retired")

    print()
    if FAIL:
        print("{} problem(s):".format(len(FAIL)))
        for f in FAIL:
            print("  - " + f)
        return 1
    print("taxonomy consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
