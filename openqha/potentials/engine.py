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
A MISSING weight file raises, and it never falls back to another potential. An ALTERED
one no longer does -- see below. Silently swapping the potential strips every downstream
number of the level it claims to be at, and level consistency is the only condition under
which the composite decomposition holds (D0-4, skills section 2.1); after 2026-09-09
keeping that condition is the operator's job, not the code's.

Each registry entry carries the SHA-256 of the file this engine was developed against.
**It is RECORDED in every product and NOT enforced** (user ruling 2026-09-09): loading a
potential is "find this filename in the one flat directory and use it", nothing more.
The gate was removed after it refused a working file on Tianhe twice -- a torch
re-serialisation changes the digest and the file size while the weights stay identical,
so the digest could not tell a repackaged model from a different one. The accepted cost
is that a genuinely swapped file now changes every downstream number silently; the
digest in the record makes it visible afterwards, and `scripts/tooling/s0_check_weights.py`
compares parameters rather than bytes for anyone who wants to ask.

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
import sys
from pathlib import Path

from .. import S0_ROOT

#: ONE DIRECTORY, FLAT. `<root>/<filename>` and nothing else.
#:
#: The weights are NOT in this repository and must never be: MACE-OFF is under the
#: Academic Software Licence, which forbids redistribution, and openQHA is meant to be
#: publishable. Only filenames and SHA-256 digests belong here.
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
#: **Nothing verifies the CONTENTS of the file any more** (user ruling 2026-09-09) -- the
#: filename is the whole identity check. See `provenance`.
#:
#: -------------------------------------------------------------------------------------
#: ADDING A NEW POTENTIAL (any MLIP, not just MACE)
#: -------------------------------------------------------------------------------------
#:   1. Drop the file in the directory above -- flat, keeping its own filename.
#:   2. Add an entry to ENGINES: `filename`, `sha256`, `source`, `licence`, `note`.
#:      `sha256=None` is allowed while you are still deciding; provenance() will report
#:      the computed digest and flag it as unpinned rather than accept anything silently.
#:      Pin it the moment the model is used for a number that gets recorded.
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
#: `sha256` is the expected digest; None means "not yet pinned" and provenance() will
#: report the computed value and flag it, rather than silently accepting anything.
#: Pin it the first time a model is used for anything that gets recorded.
ENGINES = {
    "MACE-OFF24_medium": dict(
        filename="MACE-OFF24_medium.model",
        sha256="e5ccf5837f685899811a68754e7c994393bfd1a81720393b03c643b46c70bc69",
        source="https://github.com/ACEsuit/mace-off",
        licence="Academic Software Licence (ASL) -- academic non-commercial",
        note="Committee member. Was briefly the default on 2026-09-03; superseded the same day by S0-A-16.",
    ),
    "MACE-OFF23_medium": dict(
        filename="MACE-OFF23_medium.model",
        sha256="4842c52ad210d6e1f84d6cf1ffa70fae25a7e0d755ed55cf223f43913f587db7",
        # The bytes and the numbers are two different identities, and only the second one
        # is what "the same potential" means. Measured 2026-09-09: a torch save/load
        # round trip of THIS EXACT model changed the file SHA-256 and the file size
        # (18 350 596 -> 18 367 938 bytes) while every tensor stayed bit-identical.
        # A file hash therefore cannot tell a re-serialised copy from a different model,
        # and on Tianhe it did not. See `parameter_fingerprint`.
        params_sha256="8dca373ad57c67faf89f41b0c1a58caf9b64df39016449e74238428a268f2ac2",
        n_tensors=79,
        size_bytes=18350596,
        source="https://arxiv.org/abs/2312.15211",
        licence="Academic Software Licence (ASL) -- academic non-commercial",
        note="PRODUCTION DEFAULT since 2026-09-03 (S0-A-16). Most widely used member of the family; closest lineage to stage 2 surface.",
    ),
    "MACE-OFF23_small": dict(
        filename="MACE-OFF23_small.model",
        sha256="165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f",
        source="https://arxiv.org/abs/2312.15211",
        licence="Academic Software Licence (ASL) -- academic non-commercial",
        note="Committee member (S0-C-10). Too small to be a production engine.",
    ),
    "MACE-OFF23_large": dict(
        filename="MACE-OFF23_large.model",
        sha256="a29e397dbf3e7a24ac50a9b0dfc919bd5a62efa346f5895a6237b0950c1d76f4",
        source="https://arxiv.org/abs/2312.15211",
        licence="Academic Software Licence (ASL) -- academic non-commercial",
        note="Committee member.",
    ),
    "MACE-OFF23b_medium": dict(
        filename="MACE-OFF23b_medium.model",
        sha256="871653738a4fbc8124dba1ff5bc595bd0abf7b849a8538c0825dc29ebadf1680",
        source="https://github.com/ACEsuit/mace-off",
        licence="Academic Software Licence (ASL) -- academic non-commercial",
        note="Committee member.",
    ),
    "MACE-OFF23-SC": dict(
        filename="MACE-OFF23-SC_swa.model",
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
        "The weights are NOT part of this repository -- they are under the Academic "
        "Software Licence and cannot be redistributed. Copy the file there, keeping its "
        "name, with no subdirectory.\n"
        "  {}=<dir>   move the whole directory\n"
        "  S0_MACE_MODEL=<file>   override this one file\n"
        "  S0_ENGINE=<name>       use a different registered potential: {}\n"
        "stage 0 does not accept a silent potential swap.".format(
            name, p, found, _MODEL_ROOT_ENV, ", ".join(sorted(ENGINES))))


def parameter_fingerprint(path=None, name=None):
    """SHA-256 over the model's tensors, in canonical key order.

    **The bytes and the numbers are two different identities**, and only the second one
    is what "the same potential" means. Measured 2026-09-09 on this repository's own
    `MACE-OFF23_medium.model`: a `torch.save` / `torch.load` round trip of that exact
    model changed the file SHA-256 **and the file size** -- 18 350 596 -> 18 367 938
    bytes -- while every one of the 79 tensors stayed bit-identical.

    So a file hash cannot tell a re-serialised copy from a different model. It says
    "different" to both, and the operator is then left to argue about whether the check
    is worth having. This function is the other half: it is invariant to the container
    and sensitive to the weights, which is the discrimination the check was always for.

    Returns `(hexdigest, n_tensors)`. Costs about 3 s -- it has to load the model -- so
    it is computed only when the file hash has already disagreed.
    """
    import torch
    p = Path(path) if path else model_path(name)
    model = torch.load(str(p), map_location="cpu", weights_only=False)
    state = model.state_dict() if hasattr(model, "state_dict") else model
    h = hashlib.sha256()
    for k in sorted(state):
        v = state[k]
        h.update(k.encode("utf-8"))
        h.update(str(v.dtype).encode("utf-8"))
        h.update(str(tuple(v.shape)).encode("utf-8"))
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest(), len(state)


def _classify_weight_mismatch(name, entry, p, digest):
    """Say WHICH kind of mismatch this is, instead of only that there is one.

    Four situations produce a different file hash and they do not deserve the same
    answer:

      1. truncated or corrupted transfer          -> refuse, and say so; the size says it
      2. same parameters, different container     -> proceed; record both hashes
      3. genuinely different weights              -> refuse; this is the one the pin is for
      4. no parameter pin recorded for this model -> refuse, but say the check was blind

    Returns `(ok, message, extra)`.
    """
    size = p.stat().st_size
    pinned_size = entry.get("size_bytes")
    pinned_params = entry.get("params_sha256")

    head = ("weight file for {} does not match its pinned SHA-256.\n"
            "  expected {}\n  computed {}\n  path     {}\n"
            "  size     {} bytes{}\n".format(
                name, entry.get("sha256"), digest, p, size,
                "" if not pinned_size else
                "  (pinned file was {})".format(pinned_size)))

    if pinned_size and size < pinned_size * 0.9:
        return False, head + (
            "This file is {:.0%} of the pinned size -- it looks TRUNCATED, which is what "
            "an interrupted copy leaves behind. Re-copy it and check the size first:\n"
            "    rsync -a --partial --progress data/potentials/ "
            "<host>:<repo>/data/potentials/".format(size / pinned_size)), {}

    if not pinned_params:
        return False, head + (
            "No parameter fingerprint is pinned for this engine, so the only identity "
            "available is the file hash and it disagrees. Refusing.\n"
            "Record one with:  python scripts/tooling/s0_check_weights.py --pin"), {}

    try:
        params, n = parameter_fingerprint(p, name)
    except Exception as exc:                                          # noqa: BLE001
        return False, head + (
            "The file could not be loaded to compare its PARAMETERS ({}: {}), so it is "
            "not merely a different container -- it is not a usable model file."
            .format(type(exc).__name__, exc)), {}

    extra = dict(params_sha256=params, params_sha256_pinned=pinned_params,
                 n_tensors=n, size_bytes=size)
    if params == pinned_params:
        return True, (
            "openQHA: the weight FILE differs from its pin but the PARAMETERS are "
            "identical.\n"
            "  file sha256    {} (pinned {})\n"
            "  params sha256  {}  <- matches, {} tensors\n"
            "This is a re-serialised copy of the same model -- a different torch version "
            "writing the same numbers. Proceeding; both hashes go into every product.\n"
            "If you want the file hash to agree too, re-pin it with\n"
            "    python scripts/tooling/s0_check_weights.py --pin".format(
                digest[:16], str(entry.get("sha256"))[:16], params[:16], n), extra)

    return False, head + (
        "  params sha256  {}\n  expected       {}\n"
        "**The PARAMETERS differ**, not just the container. This is a different model, "
        "and running it would change the level every downstream number claims (D0-4) "
        "without saying so. Refusing.\n"
        "Get the same weights, or register the new model as its own engine with its own "
        "name.".format(params, pinned_params)), extra


def provenance(name=None, strict=False):
    """Provenance record for the potential. Every product must carry it.

    **THE HASH IS RECORDED, NOT ENFORCED** (user ruling 2026-09-09).
    -------------------------------------------------------------
    Loading a potential is now: the registry gives a filename, `model_path` finds that
    filename in the one flat directory, and that file is used. Nothing else gates it.

    The gate was removed after it refused a working weight file on Tianhe twice. It was
    not a wrong idea, but it was the wrong instrument: measured 2026-09-09 on this
    repository's own `MACE-OFF23_medium.model`, a `torch.save`/`torch.load` round trip
    changes the file SHA-256 **and the file size** while every tensor stays bit-identical.
    So the digest was reporting "different model" for a re-serialised copy of the same
    one, and an operator who has seen that twice is right not to trust it.

    What replaces it is nothing automatic, and that is a real, accepted cost: a genuinely
    swapped weight file now changes every downstream number silently, and D0-4's level
    consistency rests on the operator rather than on the code. The digest and the file
    size still go into every product, so the swap is *visible afterwards* in the record
    even though it is no longer *prevented*. `scripts/tooling/s0_check_weights.py`
    remains, and compares parameters rather than bytes, for anyone who wants to ask.

    `strict=True` restores the old refusal for a caller that wants it. Nothing in the
    repository passes it.
    """
    name = name or engine_name()
    entry = ENGINES[name]
    p = model_path(name)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()

    mismatch = None
    expected = entry.get("sha256")
    if expected and digest != expected:
        mismatch = dict(expected_sha256=expected, computed_sha256=digest,
                        size_bytes=p.stat().st_size,
                        enforced=bool(strict),
                        note=("the file differs from the digest this engine was "
                              "developed against. Recorded, not enforced. Compare the "
                              "PARAMETERS with scripts/tooling/s0_check_weights.py if "
                              "you want to know whether it is the same model."))
        if strict:
            ok, message, extra = _classify_weight_mismatch(name, entry, p, digest)
            if not ok:
                raise ValueError(message)
            mismatch.update(extra)

    import mace
    import torch
    return dict(
        weight_file_mismatch=mismatch,
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
