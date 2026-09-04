"""Unit test: live code carries no CJK characters.

UNIT. Branch D, plan_D section 3 / acceptance criterion 2. Seconds; reads files only.

The user's ruling (2026-09-03, verbatim): "change all code in english(included Code
comments)". `.mem/` and `docs/` are out of scope -- decision logs, plans, checkpoints
and lecture notes are records for people to read, not code.

Configuration files ARE in scope: `configs/*.yaml` and the `.mdp` files are read by
code and carry measurements in their comments (plan_D section 3.1 rule 4).

Two populations, reported separately and on purpose
---------------------------------------------------
A. LIVE code -- the package, the four script categories, tests, examples, hpc,
   configs. This must be zero. It is the criterion.

B. RETIRED code under `_superseded/` -- kept as evidence and marked ready to delete
   (S0-D-2). Translating files that are scheduled for deletion spends effort on
   throwaway work and risks corrupting evidence whose only value is being the
   original. So it is COUNTED and PRINTED rather than silently exempted -- the number
   stays visible, and whether to translate or delete it is a ruling to be made, not a
   thing to let drift.

Run::  python tests/unit/t_code_is_english.py
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

#: CJK ideographs plus the full-width punctuation that comes with them. Full-width
#: punctuation matters on its own: a stray full-width comma or bracket left behind
#: after a translation is exactly the kind of residue a plain ideograph scan misses.
#: Written with escapes rather than literal endpoints, so this file does not trip its
#: own check -- the first version did exactly that (skills section 2.4(b)).
CJK = re.compile("[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")

SEARCH = ("openQHA", "scripts", "tests", "examples", "hpc", "configs")
EXTS = {".py", ".yaml", ".yml", ".toml", ".template", ".sh", ".mdp", ".json"}
SKIP_PARTS = {"__pycache__", "source-code", "_backup", ".mem", "docs"}
RETIRED = "_superseded"

FAIL = []


def scan():
    live, retired = [], []
    for top in SEARCH:
        d = ROOT / top
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.suffix not in EXTS:
                continue
            if set(p.parts) & SKIP_PARTS or p.name.endswith(".bak"):
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            n = sum(1 for line in text.splitlines() if CJK.search(line))
            if not n:
                continue
            (retired if RETIRED in p.parts else live).append((n, p))
    return live, retired


def main():
    live, retired = scan()

    print("A. live code must be free of CJK characters")
    if live:
        total = sum(n for n, _ in live)
        print("  {} file(s), {} line(s) still to translate:".format(len(live), total))
        for n, p in sorted(live, reverse=True):
            print("      {:>5d}  {}".format(n, p.relative_to(ROOT)))
            FAIL.append("{}: {} CJK line(s)".format(p.relative_to(ROOT), n))
    else:
        print("  ok -- 0 lines")

    print("\nB. retired code under {}/ (counted, not failed)".format(RETIRED))
    if retired:
        print("  {} file(s), {} line(s) left in the original language.".format(
            len(retired), sum(n for n, _ in retired)))
        print("  These are kept as evidence and marked ready to delete (S0-D-2).")
        print("  Whether to translate or to delete them is an open ruling.")
    else:
        print("  none")

    print()
    if FAIL:
        print("{} file(s) not yet translated".format(len(FAIL)))
        return 1
    print("live code is English throughout")
    return 0


if __name__ == "__main__":
    sys.exit(main())
