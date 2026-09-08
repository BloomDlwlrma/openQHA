"""Unit test: every pre-2026-09-07 import form still works, each in a FRESH process.

WHY A SUBPROCESS PER FORM
-------------------------
This is the check that failed to catch a real defect on 2026-09-07, and it failed for a
reason worth encoding. The compatibility layer registers a module in `sys.modules` under
its old dotted name the first time it is reached as an ATTRIBUTE. So a check written as

    from openqha import qha, engine, thermo      # <- registers openqha.thermo
    from openqha.thermo import KB_KCAL           # <- cannot fail now

passes whatever the state of the import machinery. Branch B then died in a fresh process
with `ModuleNotFoundError: No module named 'openqha.thermo'`, because
`from a.b import c` asks the import system for a submodule `b` and never consults
`__getattr__` -- only the trailing `c` gets an attribute fallback.

Each form below therefore runs in its own interpreter, with that form FIRST.
"""
import subprocess
import sys
from pathlib import Path


def _repo_root():
    for p in Path(__file__).resolve().parents:
        if (p / "openqha" / "__init__.py").is_file():
            return p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()

#: (description, first statement of a fresh interpreter)
FORMS = [
    ("from openqha import <name>",
     "from openqha import qha, engine, thermo, basin_store, filters, orca; "
     "assert qha.__name__ == 'openqha.quasi_harmonic.qha'"),
    ("from openqha.<name> import <thing>",
     "from openqha.thermo import KB_KCAL; assert KB_KCAL > 0"),
    ("import openqha.<name>",
     "import openqha.qha; assert openqha.qha.__name__ == 'openqha.quasi_harmonic.qha'"),
    ("the alias IS the real module, not a copy",
     "import openqha.engine as a; from openqha.potentials import engine as b; "
     "assert a is b, 'two module objects would mean two caches'"),
    ("the new explicit path",
     "from openqha.quasi_harmonic import qha; from openqha.conformer_search import "
     "filters; from openqha.qm_interfaces import orca"),
    ("import openqha pulls in no heavy dependency",
     "import sys, openqha; "
     "heavy = [m for m in ('torch', 'ase', 'rdkit', 'openmm') if m in sys.modules]; "
     "assert not heavy, heavy"),
    ("an unknown name still raises",
     "import openqha\n"
     "try:\n"
     "    openqha.no_such_module\n"
     "except AttributeError:\n"
     "    pass\n"
     "else:\n"
     "    raise AssertionError('an unknown attribute must raise')"),
]


def main():
    print("A. every legacy import form, each in a fresh interpreter")
    bad = []
    for label, code in FORMS:
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                              capture_output=True, text=True)
        if proc.returncode == 0:
            print("  ok    {}".format(label))
        else:
            tail = (proc.stderr or "").strip().splitlines()
            print("  FAIL  {}".format(label))
            print("        {}".format(tail[-1] if tail else "no stderr"))
            bad.append((label, tail[-1] if tail else ""))

    if bad:
        print()
        print("{} form(s) broken. The compatibility layer in openqha/__init__.py is "
              "what these depend on.".format(len(bad)))
        return 1
    print()
    print("{} form(s) checked".format(len(FORMS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
