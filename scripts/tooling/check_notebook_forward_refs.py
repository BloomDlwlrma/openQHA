"""Static forward-reference check across notebook cells.

TOOLING. Static check over notebook cells. Produces no science.

Walks the code cells in order, tracking names bound AT MODULE LEVEL, and reports any global name
used before it is bound.

Two defects of my own that this file has already had, both recorded so they are not repeated:

  (1) The first version missed `EV_TO_KCAL`, used in the molecular-dynamics cell but defined
      three cells later -- that one it did catch once written.
  (2) The second version MISSED `Stationary`, because it collected imports with ast.walk over the
      whole tree, so an import inside one function body counted as a cell-level binding and made a
      use inside a DIFFERENT function look satisfied. Module-level bindings only, now.
"""
import ast
import builtins
import json
import sys
from pathlib import Path

import sys as _sys

def _repo_root():
    """Directory holding the openQHA package, found by walking up from this file.

    Depth-independent on purpose: this file keeps working wherever under the
    repository it is moved to. The earlier move into `scripts/_superseded/` broke
    every `parents[1]` in the moved files silently, which is what this replaces.
    """
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)

nb_path = Path(_sys.argv[1]) if len(_sys.argv) > 1 else _repo_root() / "docs" / "s0-1_conformer-to-free-energy.ipynb"
_unused = (  # the old hard-coded paths, kept only to record where they came from
    "/mnt/c/Users/10704/Documents/01_Free-Energy-alchemical/"
               "lambda-qm9-reaction-deltan_0-mace_v1/stage0-tradition-free-energy-calc/"
               "docs/s0-1_conformer-to-free-energy.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

bound = set(dir(builtins))
problems = []
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)

# Parse the SPECIES keys out of the notebook -- the hard-coded-species check applies only to these strings.
SPECIES_KEYS = set()
for _cell in nb["cells"]:
    if _cell["cell_type"] != "code":
        continue
    try:
        _tree = ast.parse("".join(_cell["source"]))
    except SyntaxError:
        continue
    for _n in ast.walk(_tree):
        if isinstance(_n, ast.Assign) and any(
                isinstance(tg, ast.Name) and tg.id == "SPECIES" for tg in _n.targets):
            if isinstance(_n.value, ast.Dict):
                SPECIES_KEYS |= {k.value for k in _n.value.keys
                                 if isinstance(k, ast.Constant) and isinstance(k.value, str)}
print("SPECIES keys found for the hard-coded-name check:", sorted(SPECIES_KEYS))


def module_level_bindings(tree):
    """Names bound at module level only -- do not descend into function or class bodies."""
    out = set()

    def visit(node, top):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and top:
            for al in node.names:
                out.add((al.asname or al.name).split(".")[0])
            return
        if isinstance(node, SCOPES):
            if top:
                out.add(getattr(node, "name", ""))     # def/class binds its own name
            return                                      # never descend
        if top and isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign,
                                     ast.For, ast.With, ast.NamedExpr)):
            for n in ast.walk(node):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    out.add(n.id)
        for child in ast.iter_child_nodes(node):
            visit(child, top)

    for stmt in tree.body:
        visit(stmt, True)
    out.discard("")
    return out


def local_names(tree):
    """Names that are local to some inner scope, so a Load of them is not a global reference."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = node.args
            for a in (list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)):
                out.add(a.arg)
            if args.vararg:
                out.add(args.vararg.arg)
            if args.kwarg:
                out.add(args.kwarg.arg)
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                    out.add(sub.id)
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for al in sub.names:
                        out.add((al.asname or al.name).split(".")[0])
                if isinstance(sub, ast.ExceptHandler) and sub.name:
                    out.add(sub.name)
                # a nested def / class binds its own name too (for example the closure grab inside run_md)
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    out.add(sub.name)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for gen in node.generators:
                for n in ast.walk(gen.target):
                    if isinstance(n, ast.Name):
                        out.add(n.id)
        if isinstance(node, ast.ExceptHandler) and node.name:
            out.add(node.name)
    return out


for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        problems.append((idx, "SYNTAX", "{}: line {}".format(e.msg, e.lineno)))
        continue

    this_cell = module_level_bindings(tree)
    locals_ = local_names(tree)
    used = {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    for u in sorted(used - bound - this_cell - locals_):
        problems.append((idx, "FORWARD-REF", u))
    bound |= this_cell

    # Hard-coded species names: they turn into a KeyError when the edge changes, which has happened once (KeyError: 'propanal').
    # Only species names are reported -- string keys in a record dict are normal and must not all be flagged.
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                and isinstance(node.slice.value, str) and node.slice.value in SPECIES_KEYS:
            problems.append((idx, "HARD-CODED-SPECIES",
                             "{}[{!r}] -- use REACTANT / PRODUCT, or iterate over SPECIES".format(
                                 getattr(node.value, "id", "<expr>"), node.slice.value)))

if problems:
    for idx, kind, what in problems:
        print("cell {:3d}  {:12s} {}".format(idx, kind, what))
    print()
    print("{} problem(s)".format(len(problems)))
    sys.exit(1)
print("no forward references, no syntax errors across {} code cells".format(
    sum(1 for c in nb["cells"] if c["cell_type"] == "code")))
