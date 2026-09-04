"""Restore translation invariance to MACE's neighbour list, and say which MACE needs it.

READ THE SECOND SECTION FIRST. The defect below is real and reproducible, and it is NOT
in the MACE that openQHA imports. An earlier revision of this file said it was. That claim
was wrong, it was published, and the correction is the reason this module now MEASURES
which variant is installed instead of asserting it.

The defect, in the tree that has it
-----------------------------------
The MACE **develop** tree vendored at
`stage2-lambda-delta0-mace-learning/refs/source-code/mace-develop` sizes its
neighbour-search box from the atoms' EXTENT (`extent + 2 * cutoff + 1`) but anchors that
box at the coordinate ORIGIN. matscipy bins atoms in that box, so a molecule far enough
from the origin is binned wrongly and loses pairs, and the energy changes under a RIGID
TRANSLATION -- an exact symmetry of any interatomic potential.

Reproduced 2026-09-04 by loading that file by path and rebinding it over the installed
one, acetone, MACE-OFF23_medium, shift along x:

    box side (extent 3.003 + 2*5 + 1)      14.003 A
    safe band for the shift                -5.5 .. +19.0 A   (width 24.5 A)
    shift  -6 A                            dE =  +133.34 kcal/mol   max|F| 3.78 eV/A
    shift  -7 A                            dE =  +293.51
    shift  -8 A                            dE =  +550.21           max|F| 5.95 eV/A
    shift +20 A                            dE =  +235.87
    shift +50 / +1000 / -50 A              dE =  +978.52831        every neighbour lost

The safe region is a BAND around the origin, not a half-space: moving a molecule "far into
the positive quadrant" to escape does not buy time, it lands in the dead zone at once.

The tree that does NOT have it -- which is the one openQHA runs
---------------------------------------------------------------
`mace_torch 0.3.16` from site-packages sizes the box as
`(max(|positions|) + 1) * 5 * cutoff`. That grows with distance from the origin, so it
cannot fail this way. Measured 2026-09-04 with the patch NOT applied, same molecule, same
model, shifts along x and along (1,1,1):

    shift -8 / +50 / +1000 / +100000 A     dE = 0.000000 kcal/mol, max|F| unchanged
    neighbour list                         90 edges, identical at every shift

So on the current install this module is INSURANCE, not a repair. It becomes a repair the
moment openQHA is pointed at the develop tree -- which stage 2 uses -- and that is why it
is kept rather than deleted. `installed_variant()` reports which of the two is present, by
source fingerprint AND by measurement, and `state()` puts that in the provenance record so
no future reader has to reconstruct it the way this one had to.

What this correction does NOT explain
-------------------------------------
The branch B trajectory blow-ups. They were first blamed on a short-range hole in the
potential; that was refuted by measurement. They were then blamed on this neighbour list;
that is refuted here, because the code that runs does not have the defect. **The cause is
open again.** Nothing in branch B may cite either explanation until something is measured.

The fix
-------
For a fully aperiodic system the neighbour list depends only on RELATIVE positions, so
translating the atoms to their own centroid before the search is exact -- it cannot change
which pairs are within the cutoff, and it cannot change the returned cell, which is built
from the extent and is translation independent. This wrapper subtracts the centroid, calls
the original, and returns its result unchanged. On a variant that is already invariant it
is a no-op that costs one centroid subtraction per force call.

Periodic systems are passed through untouched: there the absolute position carries meaning
and matscipy wraps into the real cell.

Why it is done here and not in the MACE tree
--------------------------------------------
That tree is shared with stage 2. Patching it would change stage 2's behaviour without
stage 2 asking, and openQHA is supposed to be an independent repository. This module is
openQHA's own property: it is applied when openQHA builds a calculator, it says so in the
provenance record every product carries, and it leaves every vendored source untouched.

Three call sites, not one
-------------------------
`mace/data/atomic_data.py` does `from .neighborhood import get_neighborhood`, so it holds
its own reference. Patching only `mace.data.neighborhood.get_neighborhood` would leave the
one that actually runs untouched -- and would look like it had worked. All three names are
rebound and `verify()` proves it by measurement rather than by inspection.
"""
import hashlib

import numpy as np

#: Set by `apply()`. A record, not a flag: products should carry what was done.
_STATE = dict(applied=False, sites=[], original=None, error=None)

#: Names that must be rebound. The third is the one that actually runs.
_SITES = (("mace.data.neighborhood", "get_neighborhood"),
          ("mace.data", "get_neighborhood"),
          ("mace.data.atomic_data", "get_neighborhood"))

#: Source markers that tell the two sizings apart. Fingerprinting is a hint, not the
#: verdict -- `installed_variant()` also calls the function and looks at what it returns.
_MARKER_ORIGIN_ANCHORED = "extent + 2 * cutoff + 1"
_MARKER_RADIAL = "max_positions * 5 * cutoff"


def _wrap(original):
    """Centre the atoms before the neighbour search. Exact for aperiodic systems."""

    def get_neighborhood(positions, cutoff, pbc=None, cell=None,
                         true_self_interaction=False):
        if pbc is None or not any(pbc):
            positions = np.asarray(positions, dtype=float)
            positions = positions - positions.mean(axis=0, keepdims=True)
        return original(positions=positions, cutoff=cutoff, pbc=pbc, cell=cell,
                        true_self_interaction=true_self_interaction)

    get_neighborhood.__doc__ = (
        "openQHA wrapper restoring translation invariance; see openqha/mace_patch.py.\n\n"
        + (original.__doc__ or ""))
    get_neighborhood._openqha_patched = True
    get_neighborhood._openqha_original = original
    return get_neighborhood


def is_applied():
    return bool(_STATE["applied"])


def _edge_list(fn, positions, cutoff, shift_A=0.0, axis=0):
    shifted = np.array(positions, dtype=float, copy=True)
    shifted[:, axis] += float(shift_A)
    edge_index, _shifts, unit_shifts, _cell = fn(
        positions=shifted, cutoff=cutoff, pbc=(False, False, False), cell=None)
    edge_index = np.asarray(edge_index)
    return sorted(zip(edge_index[0].tolist(), edge_index[1].tolist(),
                      [tuple(u) for u in np.asarray(unit_shifts).tolist()]))


#: A deterministic probe that does not need a molecule file or a model: four atoms
#: spanning about 2 A, which gives the origin-anchored variant a box of roughly 13 A and
#: therefore fails at a shift of -8 A. Small enough to run in microseconds.
_PROBE = np.array([[0.0, 0.0, 0.0],
                   [1.5, 0.0, 0.0],
                   [0.0, 1.5, 0.0],
                   [0.0, 0.0, 1.5]], dtype=float)


def installed_variant(positions=None, cutoff=5.0, probe_shift_A=-8.0):
    """Which `get_neighborhood` is installed -- fingerprinted AND measured.

    This exists because the earlier revision of this module named a defect that the
    installed MACE does not have, and nothing in the provenance record could have caught
    that: it stored `mace_torch_version`, and both trees answer 0.3.x. A version string is
    not an implementation.

    `defect_present` is a MEASUREMENT: the original function is called on a probe at the
    origin and at `probe_shift_A`, and the two edge lists are compared. `sizing` is the
    source fingerprint, and the two are reported side by side so that a disagreement
    between them is visible rather than resolved silently.
    """
    import importlib
    import inspect
    record = dict(module_path=None, sha256=None, sizing="unknown",
                  defect_present=None, n_edges=None, error=None)
    try:
        module = importlib.import_module("mace.data.neighborhood")
        original = _STATE.get("original") or module.get_neighborhood
        original = getattr(original, "_openqha_original", original)
        record["module_path"] = getattr(module, "__file__", None)
        try:
            source = inspect.getsource(module)
            record["sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
            if _MARKER_ORIGIN_ANCHORED in source:
                record["sizing"] = "origin-anchored extent (defective)"
            elif _MARKER_RADIAL in source:
                record["sizing"] = "radial max|positions| (translation invariant)"
        except OSError:
            pass
        probe = _PROBE if positions is None else np.asarray(positions, dtype=float)
        at_origin = _edge_list(original, probe, cutoff, 0.0)
        shifted = _edge_list(original, probe, cutoff, probe_shift_A)
        record["n_edges"] = len(at_origin)
        record["defect_present"] = bool(at_origin != shifted)
        record["probe_shift_A"] = float(probe_shift_A)
    except Exception as exc:                        # noqa: BLE001 -- reported, not hidden
        record["error"] = "{}: {}".format(type(exc).__name__, exc)
    return record


def state():
    """What was patched, and what it was patched over. For the provenance record."""
    return dict(applied=bool(_STATE["applied"]), sites=list(_STATE["sites"]),
                error=_STATE["error"],
                what="centre atoms before the neighbour search (aperiodic systems only)",
                why="an origin-anchored search box loses neighbours far from the origin; "
                    "see `installed` for whether the MACE actually loaded has that defect",
                installed=installed_variant())


def apply(strict=True):
    """Rebind every call site. Idempotent. Returns the record.

    `strict=True` raises when the patch cannot be applied. It is kept strict even though
    the currently installed MACE does not need the patch: "the fix is optional here" is a
    property of an install that can change under the repository's feet, and a run that
    silently drops the guard is exactly how the defect would come back unnoticed.
    """
    import importlib
    if _STATE["applied"]:
        return state()
    try:
        base = importlib.import_module("mace.data.neighborhood")
        original = base.get_neighborhood
        if getattr(original, "_openqha_patched", False):
            _STATE.update(applied=True, sites=["already patched"])
            return state()
        wrapped = _wrap(original)
        sites = []
        for module_name, attr in _SITES:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            if hasattr(module, attr):
                setattr(module, attr, wrapped)
                sites.append("{}.{}".format(module_name, attr))
        if not sites:
            raise RuntimeError("no call site of get_neighborhood was found to patch")
        _STATE.update(applied=True, sites=sites, original=original, error=None)
    except Exception as exc:                       # noqa: BLE001 -- reported, not hidden
        _STATE.update(applied=False, error="{}: {}".format(type(exc).__name__, exc))
        if strict:
            raise
    return state()


def verify(positions, cutoff=5.0, shifts_A=(0.0, -8.0, 50.0, -1000.0)):
    """Prove the patch by measurement: the edge list must not change under translation.

    Checks the neighbour list itself rather than the energy, so it needs no model and runs
    in milliseconds. `apply()` having returned successfully is not evidence that the right
    function was rebound -- this is.

    On an install whose neighbour list is already invariant this passes without the patch
    doing anything, so a pass here is NOT evidence that the patch works. `installed_variant`
    is what separates the two cases, and the regression test reads both.
    """
    from mace.data import atomic_data          # the site that actually runs
    fn = atomic_data.get_neighborhood
    positions = np.asarray(positions, dtype=float)
    reference = None
    rows = []
    for s in shifts_A:
        # (edge_index, shifts, unit_shifts, cell). edge_index is 2 x n_edges -- naming
        # the first return value `sender` counted 2 "edges" for every molecule, which is
        # the sort of label that later gets believed.
        edges = _edge_list(fn, positions, cutoff, s)
        if reference is None:
            reference = edges
        rows.append(dict(shift_A=float(s), n_edges=len(edges),
                         identical_to_reference=bool(edges == reference)))
    return dict(patched=bool(getattr(fn, "_openqha_patched", False)),
                points=rows,
                invariant=bool(all(r["identical_to_reference"] for r in rows)))
