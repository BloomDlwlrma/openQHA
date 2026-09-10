"""The potential: a named registry and a provenance record.

WHY A REGISTRY AND NOT ONE HARD-CODED PATH
------------------------------------------
Until 2026-09-03 this module hard-coded MACE-OFF23-SC. It now holds a registry, for
two reasons that arrived together:

  * S0-A-15 (user ruling 2026-09-03): production moves to the standard MACE-OFF
    family (23 / 24) rather than the SC variant.
  * S0-C-10: branch C needs a COMMITTEE -- several models evaluated on one structure --
    so "the engine" can no longer be a single global path.

HOW A POTENTIAL IS LOADED
-------------------------
**The registry gives a filename; that filename is looked for in one flat directory; that
file is used.** Three steps, no fourth. Nothing reads the file's bytes, computes a
digest, compares it against anything, or can refuse it.

A MISSING weight file raises and never falls back to another potential -- that is the one
rule, and it is about the file being absent, not about what is inside it.

The identity of a run's potential is therefore **the engine name and the path**, both of
which go into every product's provenance record. Keeping a level consistent across
products (D0-4) is the operator's job: point the runs at the same engine and the same
directory.

WHAT CHANGING THE DEFAULT COSTS -- read before relying on any older number
--------------------------------------------------------------------------
Every measurement below was made on MACE-OFF23-SC and is NOT transferable to another
surface. Re-measure before citing any of them against the new default:

  * force RMSD 0.1435 eV/A against composite QZ* (D0-P2-11)
  * isomerisation dE_el MAD 1.87, max 3.89 kcal/mol (D0-P2-12)
  * edge-level error bar 0.50 / 0.66 kcal/mol (D0-P2-14)
  * every basin in package 1 -- those are MACE-OFF23-SC minima, and on a different
    surface they are not stationary points; they need re-tightening (not necessarily
    re-searching, since CREST's workhorse is GFN2-xTB and only `refine=opt` touched MACE)
  * the composite notation itself, which becomes `RI-MP2/cc-pVTZ // <new engine>`

D0-23 chose MACE-OFF23-SC for two reasons. The first (cost: 25.8 ms per force call
against Egret-1's 104.4) still holds for the whole family. The second -- that it is the
SAME surface as stage 2, so the two routes differ only in method -- is LOST by this
change, and that was a load-bearing part of stage 0's rationale. It has to be either
accepted explicitly or restored by pointing stage 2 at the same new engine.
"""
import os
import sys
from pathlib import Path

from .. import S0_ROOT

#: ONE DIRECTORY, FLAT. `<root>/<filename>` and nothing else.
#:
#: The weight files are NOT in this repository -- they are large binaries that belong
#: beside the run, not in version control. Put them in the directory below. The
#: registry holds their FILENAMES and nothing else.
#:
#: -------------------------------------------------------------------------------------
#: WHY ONE FLAT DIRECTORY (user ruling 2026-09-09)
#: -------------------------------------------------------------------------------------
#: This used to be a SEARCH over four candidate roots x seven relative layouts, with a
#: per-family subdirectory (`mace_off23/`, `mace_off24/`) on top. It was written that way
#: because a literal path had already broken twice here -- `scripts/` moved, then the
#: reference tree moved out to `../source-code/`. The cure turned out worse than the
#: disease, in a way worth stating because it is a general trap:
#:
#:   **A search with many candidates does not have one answer you can check. It has a
#:   list of places it might have looked, and when it fails nobody can tell whether the
#:   file is missing or merely somewhere the list does not cover.**
#:
#: Two concrete failures came out of that, both on TianheXY-AI:
#:   * weights copied FLAT into `data/potentials/` -- what everyone does when moving them
#:     to a cluster -- were invisible, because a directory only counted if it held a
#:     literal `mace_off23` subdirectory;
#:   * `install_dependency.sh` told people to put them in `data/potentials/`, which was
#:     not in the search list at all. It appeared to work only because the installer also
#:     writes `env_openqha.sh`, which exports S0_MACE_ROOT -- and a cluster job never
#:     sources that file. The documented location failed on exactly the path the
#:     documentation was written for.
#:
#: So: ONE root, flat, and the root is a single value you can print. If it is wrong you
#: can see that it is wrong. The old fragility is answered by `S0_ROOT` being computed
#: rather than written down, not by searching harder.
#:
#: -------------------------------------------------------------------------------------
#: WHERE TO PUT WEIGHTS
#: -------------------------------------------------------------------------------------
#:     <repo>/data/potentials/MACE-OFF23_medium.model
#:     <repo>/data/potentials/MACE-OFF23-SC_swa.model
#:     ...
#: No subdirectories. `S0_MACE_ROOT` points the whole directory somewhere else;
#: `S0_MACE_MODEL` overrides one individual file. To change potentials, use `S0_ENGINE`.
#: **The filename is the whole identity check.** Nothing reads, hashes or verifies the
#: contents; the file under that name is the model. See `provenance`.
#:
#: -------------------------------------------------------------------------------------
#: ADDING A NEW POTENTIAL (any MLIP, not just MACE)
#: -------------------------------------------------------------------------------------
#:   1. Drop the file in the directory above -- flat, keeping its own filename.
#:   2. Add an entry to ENGINES: `filename`, `source`, `note`. Three fields, no more.
#:   3. Select it with `S0_ENGINE=<name>`; make it the default by changing
#:      DEFAULT_ENGINE, and then also `configs/openqha.yaml` -> `engine:` -- see the
#:      note there about why those two disagreeing is not a cosmetic problem.
#: Nothing else in this module needs touching; there is no per-family special case left.
_MODEL_ROOT_ENV = "S0_MACE_ROOT"

#: The one place. Relative to the repository, so it moves with the repository.
DEFAULT_MODEL_ROOT = S0_ROOT / "data" / "potentials"


def model_root():
    """The single directory holding every weight file.

    Read at CALL time, not at import: a job that exports S0_MACE_ROOT after this module
    was first imported still gets the directory it asked for, and tests can point it
    somewhere without reloading the module.
    """
    env = os.environ.get(_MODEL_ROOT_ENV)
    return Path(env) if env else DEFAULT_MODEL_ROOT

#: Registry of selectable potentials.
#:
#: **A name and a filename. That is the whole entry.**
#: `source` is the paper to cite; `note` is why the entry exists. Nothing here inspects,
#: verifies or gates the file -- put the file in the directory under the right name and it
#: is used.
ENGINES = {
    "MACE-OFF24_medium": dict(
        filename="MACE-OFF24_medium.model",
        source="https://github.com/ACEsuit/mace-off",
        note="Committee member. Was briefly the default on 2026-09-03; superseded the same day by S0-A-16.",
    ),
    "MACE-OFF23_medium": dict(
        filename="MACE-OFF23_medium.model",
        source="https://arxiv.org/abs/2312.15211",
        note="PRODUCTION DEFAULT since 2026-09-03 (S0-A-16). Most widely used member of the family; closest lineage to stage 2 surface.",
    ),
    "MACE-OFF23_small": dict(
        filename="MACE-OFF23_small.model",
        source="https://arxiv.org/abs/2312.15211",
        note="Committee member (S0-C-10). Too small to be a production engine.",
    ),
    "MACE-OFF23_large": dict(
        filename="MACE-OFF23_large.model",
        source="https://arxiv.org/abs/2312.15211",
        note="Committee member.",
    ),
    "MACE-OFF23b_medium": dict(
        filename="MACE-OFF23b_medium.model",
        source="https://github.com/ACEsuit/mace-off",
        note="Committee member.",
    ),
    "MACE-OFF23-SC": dict(
        filename="MACE-OFF23-SC_swa.model",
        source="https://arxiv.org/abs/2405.18171",
        note=("The engine from D0-23, superseded as the default on 2026-09-03 but KEPT "
              "SELECTABLE: every package-1 basin and every number in D0-P2-11/12/14 "
              "lives on this surface and cannot be reproduced without it."),
    ),
}

#: Production default. Override per run with the S0_ENGINE environment variable.
DEFAULT_ENGINE = "MACE-OFF23_medium"

#: Numerical precision. Geometry optimisation and finite-difference Hessians both need
#: float64. float32 only takes a force call from 25.8 ms to about 18 ms while putting
#: the 1e-4 eV/A convergence criterion out of reach -- a numerical tolerance, not a
#: cost that can be saved.
DTYPE = "float64"

_CACHE = {}


def engine_name():
    """Selected engine name. S0_ENGINE overrides the default; unknown names raise."""
    name = os.environ.get("S0_ENGINE", DEFAULT_ENGINE)
    if name not in ENGINES:
        raise KeyError(
            "unknown engine {!r}. stage 0 does not accept an unregistered potential.\n"
            "registered: {}".format(name, ", ".join(sorted(ENGINES))))
    return name


def model_path(name=None):
    """Path to the weights. S0_MACE_MODEL overrides it -- for moving machines, not
    for changing potentials.

    Resolved HERE, not at import: `model_root()/<filename>`, read fresh each call, so a
    job that exports S0_MACE_ROOT late still gets the directory it asked for.
    """
    name = name or engine_name()

    override = os.environ.get("S0_MACE_MODEL")
    if override:
        p = Path(override)
        if not p.is_file():
            raise FileNotFoundError(
                "S0_MACE_MODEL is set to {} and that is not a file.\n"
                "Unset it to use {}/<filename>, or point it at one .model file."
                .format(p, model_root()))
        return p

    entry = ENGINES[name]
    root = model_root()
    p = root / entry["filename"]
    if p.is_file():
        return p

    # ONE expected path in the message, because there is now only one. The previous
    # version listed eight directories it had searched, which sounds more helpful and is
    # not: it left the reader to guess which of the eight was the intended one. Say where
    # the file goes, then say what is actually in that directory -- that one line
    # distinguishes "wrong directory" from "right directory, wrong filename", which were
    # indistinguishable before and are the two things that actually go wrong.
    if root.is_dir():
        present = sorted(q.name for q in root.glob("*.model"))
        found = ("directory exists and holds: " + (", ".join(present) if present
                                                   else "no .model files at all"))
    else:
        found = "that directory does not exist"
    raise FileNotFoundError(
        "weights for {} not found.\n"
        "  expected: {}\n"
        "  {}\n"
        "The weight files are not kept in this repository. Copy the file into that "
        "directory, keeping its name, with no subdirectory.\n"
        "  {}=<dir>   move the whole directory\n"
        "  S0_MACE_MODEL=<file>   override this one file\n"
        "  S0_ENGINE=<name>       use a different registered potential: {}".format(
            name, p, found, _MODEL_ROOT_ENV, ", ".join(sorted(ENGINES))))


def provenance(name=None):
    """Provenance record for the potential. Every product must carry it.

    **Loading a potential is: the registry gives a filename, `model_path` finds that
    filename in the one flat directory, and that file is used.** Nothing inspects it,
    nothing verifies it, nothing can refuse it.

    What this record therefore is: the engine name, where the model came from so it can be
    cited, the path actually loaded, and the parts of the software stack that decide the
    numbers -- the MACE version and module path, the dtype, and the neighbour-list patch
    state. That is what makes a product reproducible by someone holding the same weights.

    The engine name and the file path are the identity. If two runs name the same engine
    and the same path, they are the same potential as far as this repository is concerned.
    """
    name = name or engine_name()
    entry = ENGINES[name]
    p = model_path(name)

    import mace
    import torch
    return dict(
        engine=name,
        source=entry["source"],
        note=entry["note"],
        weights_path=str(p),
        bytes=p.stat().st_size,
        interface="mace.calculators.MACECalculator",
        mace_torch_version=mace.__version__,
        # A version string is not an implementation. Two MACE trees in this workspace both
        # answer 0.3.x and only one of them has a translation-invariant neighbour list;
        # recording only the version is what allowed a defect in the tree that does NOT
        # run to be attributed, in writing, to the tree that does. The path and the
        # neighbour-list fingerprint in `neighbour_list_patch["installed"]` are the part
        # that can be checked.
        mace_module_path=str(getattr(mace, "__file__", "")),
        torch_version=torch.__version__,
        dtype=DTYPE,
        # The interface is not the stock one and a product must not imply that it is.
        # openQHA rebinds MACE's neighbour-list construction to restore translation
        # invariance; without it a wandering molecule silently loses neighbours. See
        # openqha/mace_patch.py for the measurement that made this necessary.
        neighbour_list_patch=_patch_state(),
    )


def _patch_state():
    from . import mace_patch
    return mace_patch.state()


def calculator(device="cpu", name=None):
    """Return (calculator, engine_name, provenance). Reused within one process."""
    name = name or engine_name()
    key = (str(model_path(name)), device, DTYPE)
    if key not in _CACHE:
        # Before the model is built, not after: the wrapper has to be in place before any
        # neighbour list is constructed. It restores translation invariance, which the
        # vendored MACE does not have -- a molecule that wanders more than about 6 A
        # negative of the origin silently loses neighbours and its energy jumps by
        # hundreds of kcal/mol. See openqha/mace_patch.py. `strict=True`: a run that
        # cannot be patched must stop, because the alternative is one that looks healthy
        # and is wrong.
        from . import mace_patch
        mace_patch.apply(strict=True)
        from mace.calculators import MACECalculator
        _CACHE[key] = MACECalculator(model_paths=str(model_path(name)),
                                     device=device, default_dtype=DTYPE)
        check_weights_are_physical(_CACHE[key], name)
    return _CACHE[key], name, provenance(name)


#: A trained interatomic-potential weight is O(1). MACE-OFF23_medium's largest actual
#: weight is 37.07, and the largest tensor of any kind is the atomic reference energies
#: at 7.005e4 eV. This ceiling is five orders above that: it cannot fire on a real model
#: and it fires immediately on a corrupted one.
WEIGHT_CEILING = 1.0e6

#: Set to skip the check. It exists for a model whose parameters really are enormous --
#: none is known -- and so that the check can never be the thing standing between an
#: operator and a run they have decided to make.
_SKIP_CHECK_ENV = "S0_SKIP_WEIGHT_CHECK"


def check_weights_are_physical(calc, name, ceiling=None):
    """Refuse a weight file whose parameters are not physically possible.

    **THIS IS NOT AN IDENTITY CHECK AND NOT A HASH.** It does not care which model this
    is, where it came from, or whether it matches anything. It asks one question about
    the numbers themselves: are they the size a trained network's weights can be.

    It exists because of a measured, expensive failure. On 2026-09-10 a Tianhe branch A
    run returned a non-finite energy on **1719 of 1719** calls. The file was the right
    size (18 350 596 bytes), held the right count of tensors (77) and parameters
    (2 265 399), and **every one of them was finite** -- so every check that existed
    passed it. `interactions.0.linear.weight` contained **2.084e+306**. Multiplied by an
    ordinary O(1) feature two layers later that overflows float64, and the whole run
    produced NaN. Three days of debugging went to a corrupt copy that announced itself
    only as `nan`.

    Finiteness is not enough, and a file's length is not enough. Magnitude is the cheapest
    property that a corrupted transfer cannot fake, and it costs one pass over the
    parameters (about 2.3 M numbers, a fraction of a second, once per process).
    """
    import os
    if os.environ.get(_SKIP_CHECK_ENV):
        return None
    import torch
    ceiling = WEIGHT_CEILING if ceiling is None else ceiling
    worst_mag, worst_name, n_bad = 0.0, None, 0
    for model in getattr(calc, "models", []):
        for pname, v in list(model.named_parameters()) + list(model.named_buffers()):
            if not (torch.is_tensor(v) and v.is_floating_point() and v.numel()):
                continue
            if not bool(torch.isfinite(v).all()):
                n_bad += 1
                worst_mag, worst_name = float("inf"), pname
                continue
            mag = float(v.abs().max())
            if mag > worst_mag:
                worst_mag, worst_name = mag, pname
    if n_bad or worst_mag > ceiling:
        raise ValueError(
            "the weight file for {} is not usable.\n"
            "  path            {}\n"
            "  largest |value| {:.4g}   in {}\n"
            "  ceiling         {:.4g}\n"
            "{}"
            "A trained potential's weights are O(1) -- this model's largest real weight "
            "is 37.07. A value this size is a corrupted copy, and it produces NaN "
            "energies rather than wrong ones, silently, on every call.\n"
            "  Re-copy the file and check the size at both ends.\n"
            "  {}=1 skips this check.".format(
                name, model_path(name), worst_mag, worst_name, ceiling,
                "  {} tensor(s) are not even finite\n".format(n_bad) if n_bad else "",
                _SKIP_CHECK_ENV))
    return worst_mag


#: How far from the origin the vendored MACE stays translation invariant, in angstrom.
#: MEASURED, not assumed -- see `translation_invariance`. The neighbour list is binned in
#: a box anchored at the ORIGIN, so a molecule that wanders far enough in the NEGATIVE
#: direction silently loses neighbours. Acetone (3.0 A extent) is exact to -6.58 A and
#: wrong by +148.8 kcal/mol at -7.08 A.
SAFE_ORIGIN_RADIUS_A = 5.0


def translation_invariance(atoms, shifts_A=(0.0, 4.0, 8.0, 16.0, -4.0, -8.0, -16.0),
                           axis=0, name=None, device="cpu"):
    """Rigidly translate a molecule and watch the energy that must not change.

    A translation is an exact symmetry of any interatomic potential, so this returns zero
    for a correct implementation and is otherwise a bug detector with no false positives.

    It exists because a MACE tree in this workspace fails it -- but NOT the one that
    runs here, and an earlier revision of this docstring said otherwise. The correction
    matters more than the defect, so it is written out.

    WHICH TREE FAILS. The MACE **develop** tree vendored under
    `stage2-.../refs/source-code/mace-develop` sizes its neighbour-search box from the
    atoms' EXTENT (`extent + 2*cutoff + 1`) and anchors it at the coordinate ORIGIN, so a
    molecule far from the origin is binned wrongly and loses pairs. Reproduced 2026-09-04
    by loading that file by path over the installed one, acetone, MACE-OFF23_medium:

        box side                14.003 A     safe band  shift -5.5 .. +19.0 A
        shift   -6.0 A          dE = +133.34 kcal/mol
        shift   -8.0 A          dE = +550.21    max|F| = 5.95 eV/A
        shift  +50 / -50 A      dE = +978.52831    every neighbour lost

    The safe region is a BAND around the origin, not a half-space: moving the molecule far
    into the positive quadrant does not help, it lands in the dead zone at once.

    WHICH TREE RUNS. `mace_torch 0.3.16` from site-packages, whose sizing is
    `(max(abs(positions)) + 1) * 5 * cutoff` -- it grows with distance from the origin and
    cannot fail this way. Measured UNPATCHED on the same molecule and model: dE = 0.000000
    kcal/mol at -8, +50, +1000 and +100000 A, along x and along (1,1,1), with an identical
    90-edge neighbour list throughout.

    So this function returns zero here for two independent reasons, and that is exactly
    why it is kept: a pass is not evidence that the patch works. `mace_patch.state()`
    carries `installed`, which says which variant is present, fingerprinted AND probed.

    WHAT THIS DOES NOT EXPLAIN. The branch B trajectory blow-ups. They were blamed first
    on a short-range hole in the potential (refuted by measurement) and then on this
    neighbour list (refuted here, since the code that runs does not have the defect). The
    cause is OPEN. Nothing may cite either explanation until something is measured.
    """
    import numpy as np
    calc, engine_name, prov = calculator(device=device, name=name)
    a = atoms.copy()
    a.calc = calc
    p0 = a.get_positions().copy()
    e0 = float(a.get_potential_energy())
    rows = []
    for s in shifts_A:
        d = np.zeros(3)
        d[axis] = float(s)
        a.set_positions(p0 + d[None, :])
        e = float(a.get_potential_energy())
        f = float(np.abs(a.get_forces()).max())
        rows.append(dict(shift_A=float(s),
                         delta_energy_kcal=(e - e0) * 23.060547830618307,
                         max_force_eV_per_A=f))
    a.set_positions(p0)
    worst = max(abs(r["delta_energy_kcal"]) for r in rows)
    return dict(engine=engine_name, axis=int(axis),
                extent_A=float((p0[:, axis].max() - p0[:, axis].min())),
                points=rows,
                max_abs_delta_energy_kcal=worst,
                translation_invariant=bool(worst < 1e-6),
                safe_origin_radius_A=SAFE_ORIGIN_RADIUS_A)


def committee(names=None, device="cpu"):
    """Load several potentials at once, for ensemble uncertainty (branch C).

    Returns a list of (name, calculator, provenance). Committee disagreement is only
    meaningful if every member is present -- a missing member raises rather than
    quietly shrinking the committee and therefore its apparent uncertainty.
    """
    names = names or ["MACE-OFF23_small", "MACE-OFF23_medium", "MACE-OFF23_large",
                      "MACE-OFF23b_medium", "MACE-OFF24_medium"]
    out = []
    for n in names:
        calc, nm, prov = calculator(device=device, name=n)
        out.append((nm, calc, prov))
    return out


def composite_notation(reference="RI-MP2/cc-pVTZ", name=None):
    """The `high//low` string that must appear in every product (D0-4)."""
    return "{} // {}".format(reference, name or engine_name())
