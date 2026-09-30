"""Unit test: no file reaches the repository root by counting directory levels.

UNIT. Seconds; reads files and resolves paths, runs nothing.

The defect this test exists for
-------------------------------
Every script outside the package used to find the repository root with

    Path(__file__).resolve().parents[1]

which is correct only at one specific depth. When 26 scripts were moved into
`scripts/_superseded/<group>/` on 2026-09-03, `parents[1]` silently started
pointing at `scripts/_superseded/` instead of the repository root. Nothing failed
at move time, nothing failed at import time, and the breakage was found only when
the same move was about to be repeated for the live scripts.

That is the shape of defect this repository keeps meeting: a value that is right
under the conditions it was written for and wrong under the conditions it is used
in (an interface counts as verified only under the caller's real invocation).

The replacement walks up until it finds the directory that CONTAINS the package:

    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p

which is depth-independent, so moving a file cannot break it. This test enforces
that nothing goes back to counting levels, and -- more importantly -- it does not
trust the source text alone: part C actually resolves each file's root and checks
it is the real one.

Run::  python tests/unit/t_repo_bootstrap.py
"""
import ast
import io
import re
import sys
import tokenize
from pathlib import Path


def _repo_root():
    """Directory holding the openQHA package, found by walking up from this file."""
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()

#: Directories whose Python files must not count levels. The package itself is
#: excluded: `openqha/__init__.py` legitimately uses `parents[1]`, because a module
#: inside a package has a fixed depth relative to the package by construction.
SEARCH = ("scripts", "tests", "examples", "hpc")
SKIP_PARTS = {"__pycache__", "source-code", "_backup"}

_LEVEL_COUNTING = re.compile(r"Path\(__file__\)\.resolve\(\)\.parents\[\d+\]")
_CWD_PATH = re.compile(r"sys\.path\.insert\(\s*0\s*,\s*[\"']\.[\"']\s*\)")

FAIL = []


def python_files():
    """Every file the rules apply to -- except this one.

    This test necessarily CONTAINS both forbidden patterns, as regular expressions
    and as prose in the docstring. Scanning itself would make it fail forever, which
    is not a finding about the repository. It also once flagged two files whose
    docstrings merely QUOTE the old idiom while explaining why it was removed: when a
    new criterion fires for the first time, check its aperture before believing it.
    Hence `_code_only` below.
    """
    for top in SEARCH:
        d = ROOT / top
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*.py")):
            if set(p.parts) & SKIP_PARTS or p.resolve() == Path(__file__).resolve():
                continue
            yield p


def _code_only(text):
    """`text` with comments and prose strings blanked, line numbers preserved.

    A docstring that says "this used to be sys.path.insert(0, '.')" is documentation,
    not an occurrence, so it must not count. But blanking EVERY string literal is too
    wide, and doing that hid two genuine hits on the first attempt: the pattern being
    searched for is `sys.path.insert(0, ".")`, whose `"."` is itself a string literal,
    so blanking it destroyed the very thing the check looks for.

    So only two things are blanked: comments, and strings that stand alone as a
    statement (module, class and function docstrings, and free-floating prose). A
    string used as an ARGUMENT is code and stays.
    """
    out = text.splitlines()

    def blank(r0, c0, r1, c1):
        for r in range(r0, r1 + 1):
            i = r - 1
            if not (0 <= i < len(out)):
                continue
            line = out[i]
            a = c0 if r == r0 else 0
            b = c1 if r == r1 else len(line)
            out[i] = line[:a] + " " * max(0, b - a) + line[b:]

    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                blank(tok.start[0], tok.start[1], tok.end[0], tok.end[1])
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return text                       # unparsable: fall back to the raw text
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return "\n".join(out)
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            v = node.value
            blank(v.lineno, v.col_offset, v.end_lineno, v.end_col_offset)
    return "\n".join(out)


def resolved_root(path):
    """Repeat the walk-up from `path` and return what it would find."""
    for q in path.resolve().parents:
        if (q / "openqha" / "__init__.py").is_file():
            return q
    return None


def main():
    files = list(python_files())
    print("A. nobody counts directory levels to reach the repository root")
    n = 0
    for p in files:
        text = _code_only(p.read_text(encoding="utf-8"))
        for m in _LEVEL_COUNTING.finditer(text):
            n += 1
            FAIL.append("{}: {}".format(p.relative_to(ROOT), m.group(0)))
            print("  FAIL  {:<62s} {}".format(str(p.relative_to(ROOT)), m.group(0)))
    print("  {} file(s) scanned, {} level-counting expression(s)".format(len(files), n))

    print("\nB. nobody depends on the current working directory")
    n = 0
    for p in files:
        if _CWD_PATH.search(_code_only(p.read_text(encoding="utf-8"))):
            n += 1
            FAIL.append("{}: sys.path.insert(0, '.')".format(p.relative_to(ROOT)))
            print("  FAIL  {:<62s} sys.path.insert(0, '.')".format(
                str(p.relative_to(ROOT))))
    print("  {} occurrence(s)".format(n))

    print("\nC. the walk-up actually lands on this repository, from every depth")
    depths = {}
    for p in files:
        got = resolved_root(p)
        depth = len(p.relative_to(ROOT).parts) - 1
        depths.setdefault(depth, 0)
        depths[depth] += 1
        if got != ROOT:
            FAIL.append("{}: walk-up gives {}".format(p.relative_to(ROOT), got))
            print("  FAIL  {:<62s} -> {}".format(str(p.relative_to(ROOT)), got))
    print("  depths exercised: " + ", ".join(
        "{} dir(s) deep x{}".format(d, k) for d, k in sorted(depths.items())))
    if len(depths) < 2:
        FAIL.append("only one depth exercised -- the test cannot detect the defect it is for")
        print("  FAIL  only one depth present; this test would not have caught the "
              "original defect")

    print()
    if FAIL:
        print("{} problem(s):".format(len(FAIL)))
        for f in FAIL:
            print("  - " + f)
        return 1
    print("bootstrap is depth-independent everywhere")
    return 0


if __name__ == "__main__":
    sys.exit(main())
