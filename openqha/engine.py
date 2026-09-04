"""The potential: a named registry, provenance, and the refusal to swap silently.

WHY A REGISTRY AND NOT ONE HARD-CODED PATH
------------------------------------------
Until 2026-09-03 this module hard-coded MACE-OFF23-SC. It now holds a registry, for
two reasons that arrived together:

  * S0-A-15 (user ruling 2026-09-03): production moves to the standard MACE-OFF
    family (23 / 24) rather than the SC variant.
  * S0-C-10: branch C needs a COMMITTEE -- several models evaluated on one structure --
    so "the engine" can no longer be a single global path.

THE RULE THAT DOES NOT CHANGE
-----------------------------
A missing or altered weight file raises. It never falls back to another potential.
Silently swapping the potential strips every downstream number of the level it claims
to be at, and level consistency is the only condition under which the composite
decomposition holds (D0-4, skills section 2.1).

Each registry entry carries an expected SHA-256. `provenance()` recomputes it and
raises on mismatch, so a truncated download or a swapped file is caught at load time
rather than showing up as a puzzling number three weeks later.

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
import hashlib
import os
from pathlib import Path

from . import S0_ROOT

#: Where the weights might live, in priority order. They are NOT in this repository and
#: must never be: the MACE-OFF weights are under the Academic Software Licence, which
#: forbids redistribution, and openQHA is meant to be publishable. Only paths and SHA-256
#: digests belong here.
#:
#: This is a SEARCH rather than a literal path because a literal one has already broken
#: twice in this repository: `scripts/` moved and every `parents[1]` broke; then the
#: reference tree moved from `openQHA/ref-papers/` out to `../source-code/ref-papers/`
#: (the correct move -- see above) and every engine path broke. A name written into a
#: string does not move with the thing it names.
#:
#: `S0_MACE_ROOT` overrides the search, and `S0_MACE_MODEL` still overrides an individual
#: file. Neither is a way to change potentials: the SHA-256 check in `provenance()` is what
#: makes a swap impossible, and it runs whichever path was used.
_MODEL_ROOT_ENV = "S0_MACE_ROOT"
_CANDIDATE_ROOTS = (
    S0_ROOT,
    S0_ROOT.parent,
    S0_ROOT.parent / "source-code",
    S0_ROOT  / "openqha" /  "source-code",
)

#: Relative locations of the two families, under whichever root holds them.
#:
#: SEVERAL layouts, in preference order, because there are two legitimate ones. The
#: developer tree keeps the weights inside a clone of the upstream `mace-off` repository;
#: `install_dependency.sh` puts them in `data/potentials/mace_off23/`, i.e. directly under
#: whatever `S0_MACE_ROOT` names. The empty relative `Path(".")` covers the second case
#: and also lets anyone point S0_MACE_ROOT straight at a directory of their own.
_OFF_RELATIVES = (
    Path("ref-papers") / "mace-training" / "mace-off-main",
    Path("mace-off-main"),
    Path("mace-off"),
    Path("."),
)
_SC_RELATIVES = (Path("MACE-OFF23-SC-main"), Path("."))

#: What has to be present for a directory to count as holding that family. Without a
#: marker the empty relative would match EVERY candidate root -- `Path(".")` is always a
#: directory -- and the first root would win whether or not it held any weights.
_OFF_MARKER = "mace_off23"
_SC_MARKER = "MACE-OFF23-SC_swa.model"


def _find_root(relatives, marker, extra=None):
    """First `root/relative` that actually contains `marker`.

    Falls back to the first root and the first relative so that the error message from
    `model_path` names a concrete path rather than nothing.
    """
    roots = []
    env = os.environ.get(_MODEL_ROOT_ENV)
    if env:
        roots.append(Path(env))
    roots.extend(extra or ())
    roots.extend(_CANDIDATE_ROOTS)
    for root in roots:
        for relative in relatives:
            if (root / relative / marker).exists():
                return root / relative
    return roots[0] / relatives[0]


_OFF_ROOT = _find_root(_OFF_RELATIVES, _OFF_MARKER)
_SC_ROOT = _find_root(_SC_RELATIVES, _SC_MARKER,
                      extra=(S0_ROOT.parent / "source-code", S0_ROOT / "openQHA"))

#: Registry of selectable potentials.
#:
#: `sha256` is the expected digest; None means "not yet pinned" and provenance() will
#: report the computed value and flag it, rather than silently accepting anything.
#: Pin it the first time a model is used for anything that gets recorded.
ENGINES = {
    "MACE-OFF24_medium": dict(
        path=_OFF_ROOT / "mace_off24" / "MACE-OFF24_medium.model",
        sha256="e5ccf5837f685899811a68754e7c994393bfd1a81720393b03c643b46c70bc69",
        source="https://github.com/ACEsuit/mace-off",
        licence="Academic Software Licence (ASL) -- academic non-commercial",
        note="Committee member. Was briefly the default on 2026-09-03; superseded the same day by S0-A-16.",
    ),
    "MACE-OFF23_medium": dict(
        path=_OFF_ROOT / "mace_off23" / "MACE-OFF23_medium.model",
        sha256="4842c52ad210d6e1f84d6cf1ffa70fae25a7e0d755ed55cf223f43913f587db7",
        source="https://arxiv.org/abs/2312.15211",
        licence="Academic Software Licence (ASL) -- academic non-commercial",
        note="PRODUCTION DEFAULT since 2026-09-03 (S0-A-16). Most widely used member of the family; closest lineage to stage 2 surface.",
    ),
    "MACE-OFF23_small": dict(
        path=_OFF_ROOT / "mace_off23" / "MACE-OFF23_small.model",
        sha256="165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f",
        source="https://arxiv.org/abs/2312.15211",
        licence="Academic Software Licence (ASL) -- academic non-commercial",
        note="Committee member (S0-C-10). Too small to be a production engine.",
    ),
    "MACE-OFF23_large": dict(
        path=_OFF_ROOT / "mace_off23" / "MACE-OFF23_large.model",
        sha256="a29e397dbf3e7a24ac50a9b0dfc919bd5a62efa346f5895a6237b0950c1d76f4",
        source="https://arxiv.org/abs/2312.15211",
        licence="Academic Software Licence (ASL) -- academic non-commercial",
        note="Committee member.",
    ),
    "MACE-OFF23b_medium": dict(
        path=_OFF_ROOT / "mace_off23" / "MACE-OFF23b_medium.model",
        sha256="871653738a4fbc8124dba1ff5bc595bd0abf7b849a8538c0825dc29ebadf1680",
        source="https://github.com/ACEsuit/mace-off",
        licence="Academic Software Licence (ASL) -- academic non-commercial",
        note="Committee member.",
    ),
    "MACE-OFF23-SC": dict(
        path=_SC_ROOT / "MACE-OFF23-SC_swa.model",
        sha256="32c9fb51704f96da855c67e0cdc9894f3e41694e98b7a0ed8813388b9f21db33",
        source="https://arxiv.org/abs/2405.18171",
        licence="Academic Software Licence (ASL) -- academic non-commercial",
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
    for changing potentials."""
    name = name or engine_name()
    override = os.environ.get("S0_MACE_MODEL")
    p = Path(override) if override else Path(ENGINES[name]["path"])
    if not p.exists():
        raise FileNotFoundError(
            "weights for {} not found: {}\n"
            "The weights are NOT part of this repository -- they are under the Academic "
            "Software Licence and cannot be redistributed. Searched under:\n  {}\n"
            "Set {} to the directory holding ref-papers/ and MACE-OFF23-SC-main/, or "
            "{} to one file.\n"
            "stage 0 does not accept a silent potential swap.".format(
                name, p,
                "\n  ".join(str(r) for r in _CANDIDATE_ROOTS),
                _MODEL_ROOT_ENV, "S0_MACE_MODEL"))
    return p


def provenance(name=None):
    """Provenance record for the potential. Every product must carry it.

    Recomputes the SHA-256 and raises if it disagrees with the pinned value.
    """
    name = name or engine_name()
    entry = ENGINES[name]
    p = model_path(name)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()

    expected = entry.get("sha256")
    if expected and digest != expected:
        raise ValueError(
            "weight file for {} does not match its pinned SHA-256.\n"
            "  expected {}\n  computed {}\n  path     {}\n"
            "Refusing to run: a changed potential invalidates the level every "
            "downstream number claims (D0-4).".format(name, expected, digest, p))

    import mace
    import torch
    return dict(
        engine=name,
        source=entry["source"],
        licence=entry["licence"],
        note=entry["note"],
        weights_path=str(p),
        sha256=digest,
        sha256_pinned=bool(expected),
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
    return _CACHE[key], name, provenance(name)


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
