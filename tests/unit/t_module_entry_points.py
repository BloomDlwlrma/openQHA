"""Unit test: every `python -m openqha.<x>` this repository names actually resolves.

WHY THIS EXISTS
---------------
On 2026-09-07 `openqha/` was split into subpackages and `mace_server` moved to
`openqha/potentials/`. `openqha/__init__.py` carries a lazy PEP 562 `__getattr__` so that
every old top-level name keeps working -- and it does, for `from openqha import X`,
`import openqha.X` and `from openqha.X import Y`.

**It cannot cover `python -m openqha.X`.** `runpy` asks the finder for a submodule FILE
before any code in this package runs, so there is nothing to intercept. Branch A died at
`crest.start_servers()` with "No module named openqha.mace_server", and the whole test
suite passed, because nothing in it starts the resident MACE server.

So the entry points are checked as STRINGS, by finding them in the source and resolving
each one. That catches the next rename without needing a test that starts a server.
"""
import importlib.util
import re
import sys
from pathlib import Path


def _repo_root():
    for p in Path(__file__).resolve().parents:
        if (p / "openqha" / "__init__.py").is_file():
            return p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))

#: `python -m openqha.a.b` in a shell line, a docstring, or a subprocess argument list.
PATTERNS = (
    re.compile(r"-m\s+(openqha(?:\.[A-Za-z_][A-Za-z0-9_]*)+)"),
    re.compile(r'"-m",\s*"(openqha(?:\.[A-Za-z_][A-Za-z0-9_]*)+)"'),
)

#: Directories that are kept as evidence and are not expected to import.
SKIP = ("_superseded", "_backup", "__pycache__", ".git", "_to_delete", ".mem")

#: (file, module) pairs where a RETIRED invocation is shown ON PURPOSE, because the
#: documentation is about the fact that it no longer works. Exact pairs, so a genuinely
#: stale command anywhere else still fails. Prose that documents a trap is not the trap.
DELIBERATE = {
    ("tests/unit/t_module_entry_points.py", "openqha.X"),
    ("tests/unit/t_module_entry_points.py", "openqha.a.b"),
    ("examples/02_qha_openmm_acetone/README.md", "openqha.mace_server"),
    ("openqha/__init__.py", "openqha.mace_server"),
}


def entry_points():
    found = {}
    for path in ROOT.rglob("*"):
        if path.is_dir() or path.suffix not in (".py", ".sh", ".md", ".slurm", ".ipynb"):
            continue
        if any(s in path.parts for s in SKIP):
            continue
        if path.name == Path(__file__).name:
            # This file names `openqha.X` and `openqha.a.b` in its own docstring as
            # placeholders. Scanning itself made it fail on its own prose -- a test
            # reporting a defect in its own explanation of the defect.
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for rx in PATTERNS:
            for m in rx.finditer(text):
                found.setdefault(m.group(1), set()).add(
                    str(path.relative_to(ROOT)).replace("\\", "/"))
    return found


def main():
    found = entry_points()
    print("A. every `python -m openqha.<x>` named in the repository resolves")
    if not found:
        print("  none found -- nothing to check")
        return 0

    # Drop the deliberate counter-examples before resolving anything.
    for path_module in list(DELIBERATE):
        rel, module = path_module
        if module in found:
            found[module].discard(rel)
            if not found[module]:
                del found[module]

    bad = []
    for module in sorted(found):
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, AttributeError, ValueError) as exc:
            spec = None
            reason = "{}: {}".format(type(exc).__name__, exc)
        else:
            reason = "find_spec returned None"
        if spec is None:
            bad.append((module, sorted(found[module]), reason))
            print("  FAIL  {:44s} {}".format(module, reason))
        else:
            print("  ok    {:44s} {} caller(s)".format(module, len(found[module])))

    if bad:
        print()
        print("{} entry point(s) do not resolve. The lazy compatibility shim in".format(
            len(bad)))
        print("openqha/__init__.py does NOT cover `python -m` -- runpy asks the finder")
        print("for a submodule file before any of this package's code runs. Use the")
        print("real dotted path.")
        for module, callers, _r in bad:
            print("  {}".format(module))
            for c in callers:
                print("      {}".format(c))
        return 1

    print()
    print("{} entry point(s) checked".format(len(found)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
