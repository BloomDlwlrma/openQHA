"""openQHA -- conformational free energy for small organic molecules on a MACE potential.

THE PACKAGE LAYOUT
------------------
Modules are grouped by ROLE, the way ALF (`alframework/{builders,samplers,qm_interfaces,
ml_interfaces,tools}`) and MACE (`mace/{calculators,cli,data,modules,tools}`) are grouped.
Before 2026-09-07 all 29 lived in one flat directory, which is the shape a package has
before anyone has had to find anything in it.

    openqha/
        config.py        this repository's own configuration (configs/openqha.yaml)
        capabilities.py  what is installed here, and what each absence costs

        conformer_search/   CREST iMTD-GC, deduplication, basins
            crest, conformers, crest_census, refine_analysis, filters, symmetry
        quasi_harmonic/     trajectories, covariance, spectra, entropy
            qha, openmm_mace, vdos, torsion_cv, perturb, mdtraj_io, gmx_io
        potentials/         the potential and what carries it  (ALF: ml_interfaces)
            engine, mace_patch, mace_server
        thermochem/         thermochemistry shared by both
            hessian, thermo, critical
        store/              where results live and how to find them again
            basin_store, artifacts, record, report, worklist
        data/               locating input structures
            curated_qm9, qm9_uncharacterized
        qm_interfaces/      reference-level quantum chemistry  (ALF's own name)
            orca
        extensions/         optional cross-checks; no production number depends on one

The subpackages are named for what they DO, not for which branch of this project's plan
they belong to. `branch_a` and `branch_b` are internal plan structure, and a reader of the
published package should not have to learn the letters to find the conformer search.

`config` and `capabilities` stay at the top level on purpose: the first is what every
other module reads, and the second introspects the whole package, so neither belongs
inside one part of it.

THE PUBLIC API DID NOT CHANGE
-----------------------------
`from openqha import qha` still works, and so does every other name that used to be a
top-level module. 57 files outside this package import from it, and breaking 57 call
sites to tidy a directory would not be a tidy-up.

That compatibility is a **lazy** re-export -- PEP 562 `__getattr__` below -- and the
laziness is the load-bearing half. This package deliberately imports no torch, no ase, no
rdkit and no openmm when you `import openqha`; that is the whole reason
`openqha.capabilities` can report on what is installed without needing it installed. An
eager `from .quasi_harmonic import qha` here would throw that away to save a dictionary
lookup.

New code should prefer the explicit path -- `from openqha.quasi_harmonic import qha` --
because it says which part of the pipeline the caller is reaching into. Both work.

WHAT THIS PACKAGE IS
--------------------
**Since 2026-08-28 openQHA is an independent framework**: it imports nothing from, and
reads no file of, any other stage. The edge set, species table, symmetry numbers,
parameters and data paths are all written out in `configs/openqha.yaml`, and anything
copied from elsewhere records its source, the date it was copied, and a checksum, so that
drift is discovered rather than inherited (`D0-41`).
"""
__version__ = "0.3.0"

import importlib
import sys
from pathlib import Path

#: This repository's root. Every openQHA path is relative to it, and **there is no
#: constant pointing at another stage**.
S0_ROOT = Path(__file__).resolve().parents[1]

#: old top-level module name -> the subpackage it now lives in.
#:
#: This table is the compatibility layer, and it is also the answer to "where did X go".
#: A name that is not here has never been a module of this package.
_MOVED = {
    # conformational search (branch A)
    "crest": "conformer_search", "conformers": "conformer_search",
    "crest_census": "conformer_search", "refine_analysis": "conformer_search",
    "filters": "conformer_search", "symmetry": "conformer_search",
    # quasi-harmonic analysis (branch B)
    "qha": "quasi_harmonic", "openmm_mace": "quasi_harmonic", "vdos": "quasi_harmonic",
    "torsion_cv": "quasi_harmonic", "perturb": "quasi_harmonic",
    "mdtraj_io": "quasi_harmonic", "gmx_io": "quasi_harmonic",
    # the potential
    "engine": "potentials", "mace_patch": "potentials", "mace_server": "potentials",
    # thermochemistry
    "hessian": "thermochem", "thermo": "thermochem", "critical": "thermochem",
    # results
    "basin_store": "store", "artifacts": "store", "record": "store",
    "report": "store", "worklist": "store",
    # inputs
    "curated_qm9": "data", "qm9_uncharacterized": "data",
    # reference-level quantum chemistry
    "orca": "qm_interfaces",
}

#: Subpackages, for `dir()` and for anyone reading this file to find the map.
SUBPACKAGES = ("conformer_search", "quasi_harmonic", "potentials", "thermochem",
               "store", "data", "qm_interfaces", "extensions")

__all__ = ["S0_ROOT", "__version__", "SUBPACKAGES", "config", "capabilities"] + \
    sorted(_MOVED)


def __getattr__(name):
    """Resolve a pre-2026-09-07 module name to its new home, on first use.

    PEP 562. Three things happen here and all three matter:

      * the module is imported only when it is actually asked for, so `import openqha`
        still costs nothing and pulls in no heavy dependency;
      * it is cached as an attribute of this package, so the second access is a normal
        attribute lookup and not another import;
      * it is ALSO registered in `sys.modules` under the old dotted name, so
        `from openqha.thermo import KB_KCAL` and `import openqha.qha` keep working --
        `__getattr__` alone does not cover those, because they go through the import
        system rather than through attribute access.

    An unknown name raises AttributeError naming the subpackages, rather than the bare
    "module 'openqha' has no attribute 'x'" that says nothing about where to look.
    """
    pkg = _MOVED.get(name)
    if pkg is None:
        raise AttributeError(
            "openqha has no attribute {!r}.\n"
            "Modules now live in subpackages: {}.\n"
            "Every pre-2026-09-07 top-level name still resolves; this one was never "
            "one of them.".format(name, ", ".join(SUBPACKAGES)))
    module = importlib.import_module("{}.{}.{}".format(__name__, pkg, name))
    sys.modules.setdefault("{}.{}".format(__name__, name), module)
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(list(globals()) + __all__))


def where(name):
    """Which subpackage a module lives in. For an error message or a `--help`.

        >>> where("qha")
        'openqha.quasi_harmonic.qha'
    """
    if name in _MOVED:
        return "{}.{}.{}".format(__name__, _MOVED[name], name)
    if name in ("config", "capabilities"):
        return "{}.{}".format(__name__, name)
    raise KeyError("no module {!r} in openqha".format(name))
