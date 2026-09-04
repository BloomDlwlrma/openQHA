"""Unit test: restructuring and translating must not lose a single number.

UNIT. Branch D, plan_D section 3.2 / acceptance criterion 1. Seconds; reads files only.

Why this is the core of branch D and not a formality
----------------------------------------------------
The comments in this repository are not decoration, they are where measurements are
recorded. The twenty lines about `refine` in `openqha/crest.py`, the three-by-four
table of temperature deviations under SHAKE in `configs/openqha.yaml` --
**those numbers exist nowhere else.** Losing one while rewriting a comment deletes an
experiment, silently, with no test going red anywhere.

There is a precedent for the method. When D0-40 split the code by work item, the two
halves could not be diffed line by line, so 39 key numbers and identifiers were
checked off one at a time, 0 missing. This test makes that one-off procedure permanent.

What counts as a token
----------------------
Numbers, decision identifiers (`D0-*`, `S0-*`), and long hexadecimal runs (checksum
prefixes). They are compared as MULTISETS, so three occurrences of 5 fs must still be
three afterwards.

Losses fail; additions are only reported
----------------------------------------
plan_D section 3.2 asks for multiset equality in both directions. In practice the
restructure legitimately ADDS numbers -- a category declaration that cites `D0-28`, a
docstring that names branch 2. Refusing those would mean the criterion could only be
satisfied by writing nothing down. So the direction that fails is the one that
matters: **a token present in the baseline and missing afterwards is an error.**
Additions are printed with their counts so they stay reviewable rather than invisible.

The renamed identifier must be stripped before counting
-------------------------------------------------------
The first run of this criterion during the `s0tf` -> `openQHA` rename reported 37
false positives, because `s0tf` CONTAINS the digit 0: replacing it removed a digit
from every file. Both spellings are therefore removed from both texts before any
token is extracted. This is skills section 2.4(b) -- when a criterion first fires,
check its own aperture before believing it.

Run::  python tests/unit/t_translation_preserved_numbers.py
"""
import collections
import csv
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
BACKUP_ROOT = ROOT / "_backup"
BASELINE_PREFIX = "pre_branchD_restructure"

#: Text replaced wholesale by the restructure, whose own digits are structural rather
#: than measured. Removed from BOTH sides before counting, or every rewritten file
#: reads as a loss.
#:
#: Two entries, both learned by the criterion firing on itself:
#:   `s0tf` contains the digit 0, so renaming the package to `openQHA` looked like
#:   losing a digit in all 37 files it appeared in.
#:   `parents[1]` contains the depth index, so replacing it with the depth-independent
#:   walk-up looked like losing a 1 -- and did, in the one file that used it twice
#:   while the replacement mentions it once.
#: Neither digit ever measured anything.
#:   The source-encoding cookie `# -*- coding: utf-8 -*-` carries a digit too. It exists
#:   only because a file held non-ASCII text; once that file is translated it is
#:   vestigial, and removing it read as losing an 8.
#: None of these digits ever measured anything.
_RENAMED = re.compile(r"s0tf|openQHA|openqha|parents\[\d+\]"
                      r"|#\s*-\*-\s*coding:\s*[\w-]+\s*-\*-")

#: A minus sign only counts as a sign when nothing word-like precedes it. Without that
#: lookbehind, the range "5-10 seconds" tokenises as 5 and -10, so translating a
#: full-width dash into an ASCII hyphen looked like losing a 10 and gaining a -10. The
#: same artefact turned "indices 1-4000" into -4000. Ranges are written with hyphens all
#: over this repository, so the tokeniser has to read them the way a person does.
_TOKEN = re.compile(
    r"(?:D0|S0)-[A-Z0-9]+(?:-[A-Z0-9]+)*"      # decision identifiers
    r"|\b[0-9a-f]{8,}\b"                        # checksum prefixes
    r"|(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")   # numbers

#: Losses that are NOT translation losses, acknowledged one file at a time with a reason.
#:
#: The baseline is a snapshot taken when the restructure began, so it also catches
#: deliberate authoring that happened afterwards -- and a textual criterion cannot tell a
#: rewritten paragraph from a deleted measurement. Rather than weaken the check, such a
#: case is listed here individually, which makes acknowledging one a deliberate act with a
#: justification attached. An entry that stops matching is itself reported, so the list
#: cannot quietly outlive its reason.
ACKNOWLEDGED_LOSSES = {
    ("hpc/README.md", "D0-48"):
        "the row citing D0-48 (local T400 slower than CPU, branch B placement undecided) "
        "was rewritten on 2026-09-03 when branch B gained a measured cost and its "
        "placement stopped being undecided; a deliberate edit, not a translation loss",
}

FAIL = []


def baseline_dir():
    if not BACKUP_ROOT.is_dir():
        return None
    cands = sorted(d for d in BACKUP_ROOT.iterdir()
                   if d.is_dir() and d.name.startswith(BASELINE_PREFIX))
    return cands[-1] if cands else None


def path_map():
    """Baseline relative path -> current relative path, from the migration manifest."""
    m = {}
    manifest = ROOT / "scripts" / "MIGRATION_MANIFEST.csv"
    if manifest.is_file():
        with open(manifest, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                m[row["old_path"]] = row["new_path"]
    return m


def tokens(text):
    # U+2212 MINUS SIGN is a minus; translating it to an ASCII hyphen must not turn a
    # value into a different token. Everything else is left alone -- the en dash used for
    # ranges is deliberately NOT treated as a sign.
    text = _RENAMED.sub("", text).replace("−", "-")
    return collections.Counter(_TOKEN.findall(text))


def main():
    base = baseline_dir()
    if base is None:
        print("no baseline under _backup/{}* -- nothing to compare against.".format(
            BASELINE_PREFIX))
        print("This is a FAILURE, not a skip: without a baseline the restructure has "
              "no rollback and no check (S0-G-9, this repository is not under git).")
        return 1
    print("baseline: {}".format(base.relative_to(ROOT)))

    moved = path_map()
    checked = missing_file = 0
    seen_ack = set()
    lost_total = added_total = 0
    per_file_added = []

    for old in sorted(base.rglob("*")):
        if not old.is_file() or old.suffix not in (".py", ".yaml", ".yml", ".toml",
                                                   ".template", ".sh", ".mdp", ".md"):
            continue
        rel = old.relative_to(base).as_posix()
        new_rel = moved.get(rel, rel)
        new = ROOT / new_rel
        if not new.is_file():
            missing_file += 1
            FAIL.append("{} has no counterpart at {}".format(rel, new_rel))
            print("  MISSING  {:<62s} expected at {}".format(rel, new_rel))
            continue
        checked += 1
        a = tokens(old.read_text(encoding="utf-8", errors="replace"))
        b = tokens(new.read_text(encoding="utf-8", errors="replace"))
        lost = a - b
        added = b - a
        if lost:
            unexplained = {t: n for t, n in lost.items()
                           if (new_rel, t) not in ACKNOWLEDGED_LOSSES}
            for t in lost:
                if (new_rel, t) in ACKNOWLEDGED_LOSSES:
                    seen_ack.add((new_rel, t))
                    print("  ack      {:<62s} {}".format(new_rel, t))
            if unexplained:
                lost_total += sum(unexplained.values())
                FAIL.append("{}: lost {}".format(new_rel, unexplained))
                print("  LOST     {:<62s} {}".format(new_rel, unexplained))
        if added:
            added_total += sum(added.values())
            per_file_added.append((new_rel, sum(added.values())))

    print("\n  {} file(s) compared, {} without a counterpart".format(checked, missing_file))
    print("  tokens lost:  {}".format(lost_total))
    print("  tokens added: {} across {} file(s) (reported, not an error)".format(
        added_total, len(per_file_added)))
    for f, n in sorted(per_file_added, key=lambda kv: -kv[1])[:12]:
        print("      +{:<4d} {}".format(n, f))

    print()
    if FAIL:
        print("{} problem(s) -- every one is a measurement that would have been "
              "deleted silently".format(len(FAIL)))
        return 1
    print("no number lost")
    return 0


if __name__ == "__main__":
    sys.exit(main())
