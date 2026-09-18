"""Report what openQHA can and cannot do with what is installed here.

    python check_dependency.py

It never guesses. Everything below is imported, executed or hashed, and each row says
what stops working if it is missing — because "dependency X is absent" is not useful and
"branch A cannot run" is.

Exit code 0 if branch A can run, 1 otherwise. Optional things being absent is not a
failure; it is a smaller set of things you can do.
"""
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root():
    for p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents):
        if (p / "openqha" / "__init__.py").is_file():
            return p
    return Path(__file__).resolve().parent


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))

#: (module, minimum, required_for, what breaks without it)
PYTHON_DEPS = [
    ("numpy", "1.24", "core", "nothing works"),
    ("scipy", "1.10", "core", "nothing works"),
    ("yaml", "6.0", "core", "no configuration can be read"),
    ("ase", "3.22", "core", "no structures, no optimiser"),
    ("rdkit", "2023.03", "branch A", "no SMILES, no automorphism RMSD, no dedup"),
    ("torch", "2.0", "the potential", "no energies or forces"),
    ("mace", "0.3.6", "the potential", "no energies or forces"),
    ("pymsym", None, "branch A", "point-group labels only; sigma still works"),
    # Every chain's last step writes parquet. Listed as optional until 2026-09-13,
    # when an113's openqha-gpu turned out to have no pyarrow and the first complete
    # branch B analysis on a card died on its final line. `import openqha` survives
    # without them; no chain does.
    # pandas and pyarrow left the chain on 2026-09-15 (collect's tables are whitespace
    # .dat, the records TOML); they remain optional for the parquet writers outside it.
    ("parsl", None, "branch E", "no fan-out; one molecule at a time still works"),
    ("matplotlib", "3.5", "tutorials", "no plots"),
    ("h5py", "3.8", "branch C", "no Hessian dataset export"),
    # ---- branch B, the OpenMM route ---------------------------------------------------
    # Optional because branch B has two routes and the ASE one needs none of these. What
    # they buy is the Nose-Hoover chain at openmmtools' own defaults, and an independent
    # superposition to check ours against.
    ("openmm", "8.1", "branch B (OpenMM route)",
     "no Nose-Hoover chain route; the ASE route still runs"),
    ("openmmtorch", None, "branch B (OpenMM route)",
     "MACE cannot be carried into OpenMM"),
    ("openmmtools", None, "branch B (OpenMM route)",
     "falls back to openmm.NoseHooverIntegrator, which uses a DIFFERENT Yoshida-Suzuki "
     "order (7 rather than 5); the fallback is recorded in the product"),
    ("mdtraj", "1.9", "branch B cross-check",
     "no mdtraj superposition check"),
    ("MDAnalysis", "2.4", "branch B cross-check",
     "no MASS-WEIGHTED independent superposition; mdtraj alone cannot mass-weight and "
     "differed from us by 2.41 kcal/mol on a real trajectory"),
]

OPTIONAL_MODULES = {"pymsym", "parsl", "matplotlib", "h5py",
                    "openmm", "openmmtorch", "openmmtools", "mdtraj", "MDAnalysis"}

#: (executable, env var that overrides, required_for, what breaks)
BINARIES = [
    ("crest", "S0_CREST_BIN", "branch A", "no conformer search at all"),
    ("xtb", None, "branch C", "no GFN2 labels"),
    # `gmx` may live in a SIBLING conda environment; openqha/gmx_io.gmx_binary() searches
    # for it, because acceptance criterion 2 once failed for no reason but which shell was
    # active. This check looks only on PATH, so a "missing" here is not conclusive.
    ("gmx", "S0_GMX_BIN", "branch B cross-check",
     "no gmx covar -mwa check of the covariance spectrum (the mass-weighted one, which "
     "mdtraj cannot provide)"),
    ("orca", None, "branch C", "no RI-MP2 reference labels"),
]


def _version(mod):
    return getattr(mod, "__version__", None) or getattr(mod, "version", None) or "?"


def _cmp(have, want):
    """True if `have` >= `want`, comparing numeric prefixes only."""
    if want is None or have in (None, "?"):
        return True
    def parts(s):
        out = []
        for chunk in str(s).replace("-", ".").split("."):
            if chunk.isdigit():
                out.append(int(chunk))
            else:
                break
        return out
    h, w = parts(have), parts(want)
    return h + [0] * (len(w) - len(h)) >= w + [0] * (len(h) - len(w))


def main():
    print("=" * 78)
    print("openQHA dependency check")
    print("=" * 78)
    print("python  %s" % sys.version.split()[0])
    print("repo    %s" % ROOT)
    print()

    print("%-12s %-12s %-10s %-16s %s" % ("module", "found", "minimum", "needed for",
                                          "status"))
    print("-" * 78)
    missing_required, missing_optional = [], []
    for name, minimum, need, breaks in PYTHON_DEPS:
        try:
            mod = importlib.import_module(name)
            ver = _version(mod)
            ok = _cmp(ver, minimum)
            status = "ok" if ok else "TOO OLD"
            if not ok and name not in OPTIONAL_MODULES:
                missing_required.append((name, breaks))
        except ImportError:
            ver, status = "-", "MISSING"
            (missing_optional if name in OPTIONAL_MODULES
             else missing_required).append((name, breaks))
        print("%-12s %-12s %-10s %-16s %s" % (name, ver, minimum or "-", need, status))

    print()
    print("%-10s %-42s %-16s %s" % ("binary", "path", "needed for", "status"))
    print("-" * 78)
    missing_bin = []
    for name, env, need, breaks in BINARIES:
        path = (os.environ.get(env) if env else None) or shutil.which(name)
        if path and Path(path).exists():
            status = "ok"
        else:
            status = "MISSING"
            missing_bin.append((name, need, breaks))
            path = "-"
        print("%-10s %-42s %-16s %s" % (name, str(path)[-42:], need, status))

    # ---- the potential: a path AND a checksum, not just an import --------------------
    print()
    engine_ok = False
    try:
        from openqha import engine
        prov = engine.provenance()
        print("potential  %s" % prov["engine"])
        print("  weights  %s" % prov["weights_path"])
        print("  source   %s" % prov["source"])
        engine_ok = True
    except Exception as exc:
        print("potential  UNAVAILABLE: %s: %s" % (type(exc).__name__, str(exc)[:180]))
        print("  The weights are not shipped with this repository. Fetch them:")
        print("    bash install_dependency.sh            # downloads them")
        print("  or copy MACE-OFF23_medium.model into that directory yourself --")
        print("  FLAT, no subdirectory. The error above names the exact path.")
        print("  S0_MACE_ROOT moves the directory; S0_MACE_MODEL overrides one file.")

    # ---- data ------------------------------------------------------------------------
    print()
    try:
        from openqha import config, curated_qm9
        cfg = config.load()
        print("config     %s" % Path(cfg["_path"]).name)
        for inc in cfg.get("_includes", []):
            print("  include  %s" % Path(inc).name)
        c = curated_qm9.census(cfg)
        if c["available"]:
            print("curatedQM9 %d files (%d repaired) at %s"
                  % (sum(c["counts"].values()), c["n_repaired"], c["root"]))
        else:
            print("curatedQM9 absent -- f7_mode='curated' will refuse to run")
        vend = ROOT / "data" / "reference-geometries"
        print("shipped    %d reference geometries"
              % len(list(vend.glob("dsgdb9nsd_*.xyz"))))
    except Exception as exc:
        print("config     UNREADABLE: %s: %s" % (type(exc).__name__, str(exc)[:150]))

    # ---- verdict ---------------------------------------------------------------------
    print()
    print("=" * 78)
    crest_missing = any(n == "crest" for n, _, _ in missing_bin)
    branch_a = not missing_required and engine_ok and not crest_missing
    if missing_required:
        print("REQUIRED AND MISSING:")
        for n, breaks in missing_required:
            print("  %-12s %s" % (n, breaks))
    if crest_missing:
        print("  %-12s %s" % ("crest", "no conformer search at all"))
        print("               conda env create -f environment.yml   # crest is IN it")
        print("               S0_CREST_BIN overrides, for a CREST from elsewhere")
    if missing_optional:
        print("optional, absent (each line is a capability you do not have):")
        for n, breaks in missing_optional:
            print("  %-12s %s" % (n, breaks))
    for n, need, breaks in missing_bin:
        if n != "crest":
            print("  %-12s absent -- %s: %s" % (n, need, breaks))
    print()
    print("branch A (conformer search): %s" % ("READY" if branch_a else "NOT READY"))
    print("=" * 78)
    return 0 if branch_a else 1


if __name__ == "__main__":
    sys.exit(main())
