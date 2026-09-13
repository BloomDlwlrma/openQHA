#!/usr/bin/env python
"""Can branch B run on this card? Six checks, in the order they fail, on acetone.

TOOLING. Read-only, no products written, and it needs a card -- so it runs on a compute
node, not a login one (`bash hpc/tools/gpu_shell.sh` gets you one interactively).

    python scripts/tooling/s0_probe_openmm_cuda.py                 # CUDA
    python scripts/tooling/s0_probe_openmm_cuda.py --platform CPU  # the control

Run it on a compute node with a card (`bash hpc/tools/gpu_shell.sh` gets you one). It is
the breakpoint for the 2026-09-12 failure, where all twelve branch B trajectories died at
`openmm.Context(...)` with CUDA_ERROR_UNSUPPORTED_PTX_VERSION after three minutes each of
torch import and model tracing -- and the job log said only `ht_numel, "Invalid weight
shape"`, a fragment of an unrelated warning.

The order is the point. Each section is cheaper than the one after it and rules out the
cheap causes first, so a failure names itself:

    1  versions        nvrtc vs the driver -- the PTX rule, milliseconds, no GPU needed
    2  cards           what Slurm gave this process, and what nvidia-smi agrees to
    2b stack           numpy/torch/openmm/mace: versions, paths, and whether numpy and
                       torch can speak to each other at all
    3  platforms       does this OpenMM even have a CUDA platform compiled in
    4  empty context   a 1-particle System on that platform: THE PTX JIT, alone,
                       without torch, without MACE, without the model
    5  MACE system     the real TorchForce, traced, built
    6  10 steps        integrate, and read back an energy that must be finite

Section 4 is the one that matters: it is the first thing that JITs a kernel, it takes
about a second, and if it fails the fault is the toolkit/driver pair and nothing else.
"""
import argparse
import os
import sys
import time
from pathlib import Path


def _repo_root():
    """Walk up to the checkout, rather than counting directories.

    `parents[2]` breaks the moment a script moves one level, and this repository's
    taxonomy test refuses it for that reason.
    """
    for p in Path(__file__).resolve().parents:
        if (p / "openqha" / "__init__.py").is_file():
            return p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))

#: Acetone, the geometry this project uses everywhere as its fixed reference.
NUMBERS = [6, 6, 8, 6, 1, 1, 1, 1, 1, 1]
MASSES = [12.011, 12.011, 15.999, 12.011, 1.008, 1.008, 1.008, 1.008, 1.008, 1.008]
POSITIONS_A = [
    [-1.2809934968, 0.6994441688, -0.0288203015],
    [0.0000153572, -0.0996419871, 0.0010395785],
    [-0.0001858261, -1.3066590418, 0.0032700762],
    [1.2812912419, 0.6991249934, 0.0279540551],
    [-1.3321808748, 1.3482968846, 0.8457929031],
    [-1.2933583680, 1.3450133645, -0.9072646153],
    [-2.1369652508, 0.0332433243, -0.0465246622],
    [2.1370415306, 0.0327106863, 0.0481295399],
    [1.3327048086, 1.3447118089, -0.8490598226],
    [1.2938784159, 1.3479347152, 0.9040066524],
]


def detail(exc):
    """The traceback, not just the exception's `str`.

    **An ImportError from a C extension says nothing useful without its stack.** On
    2026-09-12 section 5 printed `ImportError: numpy._core.multiarray failed to import`
    and that was the whole report: it names no package, and `numpy._core` is the NumPy 2
    layout while the installed numpy was 1.26.4 -- so the message points at numpy and the
    culprit is whatever was built against the wrong one. The frames name it.
    """
    import traceback
    for line in traceback.format_exc().rstrip().splitlines():
        print("    | " + line)


def rule(title):
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


#: Filled in as the sections run, so --quiet can print a line without re-deriving it.
RESULT = {}


class _SkipSection(Exception):
    """Not an error: --no-accuracy leaves section 5b out of a throughput sweep."""


def platform_properties(platform_name, precision, dtype_name="float64"):
    """Context properties: an explicit --precision, else the one matching the dtype.

    Defaulting to "whatever OpenMM does" was how 74.2 ms/step got measured and written
    into four documents as "the A800 is slower than the CPU". It is not: that was
    float64 under the platform's default `single`. Matched, the same card does 39.2.
    """
    if platform_name != "CUDA":
        return {}
    if precision is not None:
        return {"Precision": precision}
    import torch
    from openqha.quasi_harmonic import openmm_mace as _om
    return _om.platform_properties_for(platform_name, getattr(torch, dtype_name))


def describe_context(ctx):
    """What precision did this Context actually end up with?"""
    plat = ctx.getPlatform()
    bits = [plat.getName()]
    for name in plat.getPropertyNames():
        if name in ("Precision", "Threads", "DeterministicForces"):
            try:
                bits.append("{}={}".format(name, plat.getPropertyValue(ctx, name)))
            except Exception:                                           # noqa: BLE001
                pass
    return "  ".join(bits)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", default="CUDA")
    ap.add_argument("--steps", type=int, default=10)
    # **Two independent precisions, and they are often confused for one.**
    #
    #   --dtype     the dtype MACE itself evaluates in. This is the one that decides
    #               how the energy and its gradient are computed, because the force
    #               comes from TorchForce, not from an OpenMM kernel.
    #   --precision the OpenMM CUDA platform's `Precision` property, which governs
    #               OpenMM's own arrays and integration. **Its default is `single`,
    #               not `mixed`** -- read off the installed build, 2026-09-12:
    #                   CUDA properties: DeviceIndex, DeviceName, UseBlockingSync,
    #                   Precision, UseCpuPme, CudaCompiler, TempDirectory,
    #                   CudaHostCompiler, DisablePmeStream, DeterministicForces
    #                   Precision = 'single'
    #               The CPU platform has NO precision property at all (only Threads
    #               and DeterministicForces), and Reference has none either, so
    #               --precision is accepted and ignored off CUDA.
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"],
                    help="dtype MACE evaluates in (default float64)")
    ap.add_argument("--precision", default=None,
                    choices=["single", "mixed", "double"],
                    help="OpenMM CUDA Precision property; default matches --dtype")
    # A concurrency sweep runs this N times at once. N x 200 lines of TracerWarning is
    # not a measurement, it is a haystack -- so --quiet swallows every section and
    # prints one line of numbers. A FAILING quiet run prints everything it swallowed,
    # because the whole point of the sections is to say where it broke.
    ap.add_argument("--quiet", action="store_true",
                    help="one line of results; full output only if something fails")
    ap.add_argument("--tag", default="",
                    help="string echoed in the --quiet line, to label a worker")
    ap.add_argument("--no-accuracy", action="store_true",
                    help="skip section 5b; it builds a second float64 CPU system, which "
                         "doubles the cost and measures nothing about throughput")
    args = ap.parse_args()
    platform_name = args.platform.upper()
    fail = []

    # ---- 1. versions ---------------------------------------------------------------
    rule("1. versions -- nvrtc vs the driver (the PTX rule)")
    from openqha import gpu_preflight
    d = gpu_preflight.describe()
    print("  nvrtc                 {}".format(d["nvrtc"] or "not loadable here"))
    print("  nvrtc comes from      {}".format(d.get("nvrtc_path") or "(path unknown)"))
    print("  libcuda comes from    {}".format(d.get("libcuda_path") or "(not mapped)"))
    print("  driver supports CUDA  {}".format(d["driver_cuda"] or "not readable here"))
    if d.get("ld_stubs"):
        print("  LD_LIBRARY_PATH has   {}".format("  ".join(d["ld_stubs"])))
    print("  verdict               {}".format(d["reason"]))
    if d.get("stub_loaded") or (d.get("ld_stubs") and not d["nvrtc"]):
        print()
        print("  **STOP HERE.** A `stubs/` directory is on the library search path, so the")
        print("  libcuda / libnvrtc this process gets are NVIDIA's link-time stubs: every")
        print("  entry point returns an error, nvrtc prints 'You are running using the stub")
        print("  version of nvrtc', and openmm.Context() dies with CUDA error 34")
        print("  (CUDA_ERROR_STUB_LIBRARY). No version comparison means anything until the")
        print("  stub is gone. Find who put it there and take it off the path:")
        print("      module list;  module show CUDA/<ver> | grep -n stubs")
        print("      echo $LD_LIBRARY_PATH | tr : '\\n' | grep -n stubs")
        print("      export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr : '\\n' | grep -v stubs | paste -sd:)")
        print("  then re-run this probe. (On an104 2026-09-13 the stub arrived with a CUDA")
        print("  module; the environment's own nvrtc was too NEW for the driver, which is")
        print("  a different fault with a different fix -- a module no newer than 12.4.)")
        fail.append("CUDA stub library on the load path")
    if d["nvrtc"] and d["driver_cuda"] and not d["ok"]:
        print()
        if platform_name == "CUDA":
            print("  **STOP HERE.** OpenMM JITs every kernel from PTX, and this driver")
            print("  cannot read PTX from that toolkit. Sections 4-6 cannot pass.")
            print()
            # The nvrtc that gets loaded is whichever libnvrtc.so is first on
            # LD_LIBRARY_PATH. A site CUDA module puts ITS lib dir ahead of conda's, so
            # loading a module no newer than the driver fixes this with no install and
            # no network -- which is what an45 needed (module CUDA/12.2 over a conda
            # 12.3) and what a compute node, with no route out, can actually do.
            # examples/chain_body.sh already loads CUDA/12.2 for every job; this probe
            # run did not, which is why it sees the environment's own toolkit.
            drv = "{}.{}".format(*d["driver_cuda"])
            print("  Fix, in order:")
            print("    1. a site CUDA module no newer than the driver, then re-run:")
            print("         module avail CUDA          # what is there")
            print("         module load CUDA/{}       # or the highest <= {}".format(drv, drv))
            print("       (needs nothing installed; examples/chain_body.sh does this itself)")
            print("    2. only if no such module exists, pin the environment's toolkit,")
            print("       ON A LOGIN NODE -- a compute node has no outbound network:")
            print("         mamba install -n openqha-gpu cuda-version={}".format(drv))
            fail.append("nvrtc newer than the driver")
        else:
            # The CPU platform compiles nothing, so this mismatch cannot affect the run
            # being probed. Reported, not counted -- a control that fails for a reason
            # unrelated to what it controls is not a control.
            print("  (not a failure for --platform {}: nothing here JITs PTX. It would be"
                  .format(platform_name))
            print("   one for CUDA.)")

    # ---- 2. cards ------------------------------------------------------------------
    rule("2. cards -- what this process was given")
    for v in ("SLURM_JOB_ID", "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS", "SLURM_CPUS_PER_TASK",
              "CUDA_VISIBLE_DEVICES", "S0_CARD", "S0_SCRATCH", "S0_SOCKET_DIR"):
        print("  {:22s} {}".format(v, os.environ.get(v, "(unset)")))
    import subprocess
    try:
        smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60)
        for line in (smi.stdout or "").splitlines():
            print("  nvidia-smi             {}".format(line))
        if smi.returncode != 0:
            print("  nvidia-smi             failed: {}".format((smi.stderr or "").strip()[:120]))
    except Exception as exc:                                            # noqa: BLE001
        print("  nvidia-smi             {}: {}".format(type(exc).__name__, exc))

    # ---- 2b. the stack -------------------------------------------------------------
    # Section 5 used to fail here with a bare `ImportError: numpy._core.multiarray failed
    # to import` after 38 seconds of loading, which names neither package nor version.
    # Ask the cheap question first: are these two builds compatible at all.
    rule("2b. the stack -- versions, paths, and the numpy/torch ABI")
    prefix = sys.prefix
    print("  active prefix         {}".format(prefix))
    stack, outside = {}, []
    for mod in ("numpy", "torch", "openmm", "openmmtorch", "e3nn", "mace", "scipy", "ase"):
        try:
            m = __import__(mod)
            stack[mod] = getattr(m, "__version__", "?")
            where = getattr(m, "__file__", "") or ""
            # **Is it from THIS environment?** A package under ~/.local (the per-user
            # site, PEP 370) is on sys.path AHEAD of the environment's site-packages, so
            # it wins silently -- and it was installed against whatever numpy happened to
            # be around at the time. Measured 2026-09-12: scipy 1.16.1 in ~/.local, built
            # against numpy 2, against the environment's numpy 1.26.4.
            mark = "" if where.startswith(prefix) else "   <== NOT FROM THIS ENVIRONMENT"
            if mark:
                outside.append((mod, where))
            print("  {:12s} {:24s} {}{}".format(mod, str(stack[mod]), where, mark))
        except Exception as exc:                                        # noqa: BLE001
            stack[mod] = None
            print("  {:12s} {:24s} {}: {}".format(mod, "IMPORT FAILED",
                                                  type(exc).__name__, exc))
    if outside:
        print()
        print("  **{} PACKAGE(S) COME FROM OUTSIDE THE ACTIVE ENVIRONMENT.**".format(
            len(outside)))
        print("  Python searches the per-user site (~/.local/lib/pythonX.Y/site-packages)")
        print("  BEFORE an environment's own site-packages, so these shadow the versions")
        print("  this environment was solved with, and they were built against whatever")
        print("  was installed when someone ran `pip install --user`.")
        print("  Prove it in one command, no install needed:")
        print("      PYTHONNOUSERSITE=1 python {}".format(
            "scripts/tooling/s0_probe_openmm_cuda.py"))
        print("  If that passes, the fix is to remove them:")
        for mod, where in outside:
            print("      python -m pip uninstall -y {}       # {}".format(mod, where))
        print("  hpc/env/common.sh sets PYTHONNOUSERSITE=1 for every job for this reason.")
    # **WHAT NUMPY DOES THE INSTALLED TORCH ACTUALLY REQUIRE?** conda records every
    # package's declared dependencies in <prefix>/conda-meta/<pkg>.json, offline, so this
    # needs no network and no guessing. It is the question that should have been asked
    # first on 2026-09-12: `environment-tianhe-gpu.yml` recorded numpy 1.26.4 beside
    # pytorch 2.5.1 build 303 from a DRY RUN that was never executed, and the pairing was
    # assumed to hold. If torch declares numpy >=2 and numpy is 1.x, the pin in that file
    # is backwards and no amount of reinstalling numpy 1.26.4 will help.
    metas = sorted((Path(prefix) / "conda-meta").glob("*.json")) \
        if (Path(prefix) / "conda-meta").is_dir() else []
    numpy_needs = []
    for meta in metas:
        stem = meta.name.rsplit("-", 2)[0]
        if stem not in ("pytorch", "libtorch", "openmm-torch", "scipy", "mace-torch"):
            continue
        try:
            import json as _json
            rec = _json.loads(meta.read_text(encoding="utf-8"))
        except Exception:                                               # noqa: BLE001
            continue
        np_dep = [d for d in (rec.get("depends") or []) if d.split()[0] == "numpy"]
        if np_dep:
            numpy_needs.append((stem, rec.get("version", "?"), "; ".join(np_dep)))
            print("  {:12s} {:12s} requires  {}".format(stem, rec.get("version", "?"),
                                                        "; ".join(np_dep)))
    if not metas:
        print("  conda-meta            not readable (not a conda prefix?)")

    # **IS THE NUMPY ON DISK THE NUMPY CONDA INSTALLED?** Asked on 2026-09-12 only after
    # the versions had been checked and agreed: pytorch 2.5.1 declares `numpy >=1.19,<3`
    # and scipy 1.13.1 declares `numpy <2.3`, both satisfied by the installed 1.26.4 --
    # so no version is wrong and the failure is still there. What torch reported was
    #
    #     module 'numpy._globals' has no attribute '_signature_descriptor'
    #
    # an AttributeError raised INSIDE numpy's own import. No released numpy 1.26.4 file
    # refers to that name, so a file under numpy/ is not from 1.26.4 even though
    # `numpy.__version__` says it is. That is what a pip overwrite, or a second numpy
    # removed by file list, leaves behind: the right version string, a mixed tree.
    #
    # conda-meta/<pkg>.json records every file the package owns and its size, offline, so
    # the tree can be compared against what conda installed without a network or a
    # reinstall. Missing files mean something deleted them; a size mismatch means
    # something rewrote them; an undeclared file under numpy/ is a leftover.
    npy_meta = [m for m in metas if m.name.rsplit("-", 2)[0] in ("numpy", "numpy-base")]
    if metas and not npy_meta:
        # Measured an104 2026-09-13: numpy 2.4.6 present, `numpy tree` lines absent, and
        # the absence went unremarked -- the check had simply nothing to compare against.
        # No conda record means conda did not put this numpy here: pip did, over whatever
        # the solver had chosen, and every conda package that declares a numpy bound was
        # solved against a numpy that is no longer there.
        print("  numpy tree            **NO CONDA RECORD FOR numpy** -- it was not installed")
        print("                        by conda/mamba, so pip put it here, over the solved")
        print("                        environment. What is on disk:")
        print("                          ls -d $CONDA_PREFIX/lib/python*/site-packages/numpy*")
        print("                          python -m pip show numpy | head -3")
        print("                        Two dist-infos there = the same overwrite an45 had:")
        print("                        pip uninstall until none is left, then")
        print("                          mamba install -n <env> --offline numpy=1.26.4")
        print("                        (login node without --offline if the cache lacks it)")
    if npy_meta:
        import json as _json
        declared, sizes = set(), {}
        for meta in npy_meta:
            try:
                rec = _json.loads(meta.read_text(encoding="utf-8"))
            except Exception:                                           # noqa: BLE001
                continue
            for rel in rec.get("files") or []:
                declared.add(rel)
            for ent in (rec.get("paths_data") or {}).get("paths") or []:
                # Skip the files conda rewrites as it installs them: anything carrying a
                # prefix placeholder (a shebang, a build path in __config__.py) is edited
                # on the way in, so its size differs from the package record by design.
                # Measured 2026-09-12 on a healthy environment: bin/f2py and
                # numpy/__config__.py both flagged this way. Comparing them would make
                # every environment look damaged, which is worse than not comparing.
                if ent.get("_path") and ent.get("size_in_bytes") is not None                         and not ent.get("prefix_placeholder"):
                    sizes[ent["_path"]] = ent["size_in_bytes"]
        missing, resized = [], []
        for rel in sorted(declared):
            f = Path(prefix) / rel
            if not f.exists():
                missing.append(rel)
            elif rel in sizes and f.is_file() and not rel.endswith(".pyc")                     and not rel.startswith(("bin/", "Scripts/"))                     and Path(rel).name != "__config__.py"                     and f.stat().st_size != sizes[rel]:
                resized.append((rel, sizes[rel], f.stat().st_size))
        # Anything under numpy/ that conda never installed.
        try:
            import numpy as _npt
            pkg = Path(_npt.__file__).resolve().parent
        except Exception:                                               # noqa: BLE001
            pkg = None
        strays = []
        if pkg is not None and pkg.name == "numpy" and pkg.is_dir():
            for f in pkg.rglob("*"):
                if not f.is_file() or f.suffix in (".pyc",):
                    continue
                try:
                    rel = str(f.relative_to(prefix)).replace("\\", "/")
                except ValueError:
                    continue
                if rel not in declared:
                    strays.append(rel)
        print("  numpy tree            {} file(s) declared by conda".format(len(declared)))
        for label, items in (("MISSING (deleted by something else)", missing),
                             ("UNDECLARED (left by something else)", strays)):
            if items:
                print("  **{}**: {}".format(label, len(items)))
                for rel in items[:12]:
                    print("      {}".format(rel))
                if len(items) > 12:
                    print("      ... and {} more".format(len(items) - 12))
        if resized:
            print("  **REWRITTEN (size differs from the installed package)**: {}".format(
                len(resized)))
            for rel, recorded, got in resized[:12]:
                print("      {}  conda {} bytes, on disk {}".format(rel, recorded, got))
            if len(resized) > 12:
                print("      ... and {} more".format(len(resized) - 12))
        if not (missing or strays or resized):
            print("  numpy tree            matches the installed package exactly")
        else:
            print("  The numpy in this environment is NOT the package conda installed.")
            # Which numpy did the leftovers come from? These basenames exist only in
            # numpy 2.x, so their presence in a 1.26.4 tree dates the overwrite without
            # any guessing. Measured on an45 2026-09-12: _expired_attrs_2_0.py,
            # _array_api_info.py, _configtool.py and lib/_type_check_impl.py all present
            # beside a numpy reporting 1.26.4.
            v2 = sorted({Path(rel).name for rel in strays} & {
                "_expired_attrs_2_0.py", "_array_api_info.py", "_configtool.py",
                "_type_check_impl.py", "_utils_impl.py", "_core"})
            if v2:
                print("  The leftovers are from **numpy 2.x** -- {} exist(s) in no "
                      "1.x release.".format(", ".join(v2)))
            if strays:
                # **A REINSTALL CANNOT FIX THIS.** conda rewrites the files it owns and
                # has no record of the rest, so `--force-reinstall` would restore the
                # declared tree and leave every leftover in place, importable, exactly as
                # before. The directory has to go first. Saying `--force-reinstall` alone
                # here (as this probe did on 2026-09-12) sends the reader round the loop
                # again with nothing changed.
                print()
                print("  **A REINSTALL ALONE WILL NOT FIX THIS.** conda only rewrites the")
                print("  {} files it owns; the {} leftover(s) are not in its records and".format(
                    len(declared), len(strays)))
                print("  would survive --force-reinstall untouched. Remove the tree first.")
                print()
                print("  First, see whether pip recorded the overwrite (then it can undo it):")
                print("      ls -d $CONDA_PREFIX/lib/python*/site-packages/numpy*")
                print("  A `numpy-2.*.dist-info` there means:")
                print("      python -m pip uninstall -y numpy     # removes what pip wrote")
                print("  No dist-info means nothing tracks those files, so remove by hand:")
                print("      rm -rf $CONDA_PREFIX/lib/python*/site-packages/numpy")
                print("  Either way the declared files go too, so restore them afterwards:")
            else:
                print("  Repair it from the package cache:")
            print("      mamba install -n <env> --offline --force-reinstall numpy=={}".format(
                stack.get("numpy") or "1.26.4"))
            print("  (drop --offline on a login node if the cache no longer has it)")
            print("  Then re-run this probe: the tree must report no leftovers before")
            print("  anything else in the stack is worth testing.")

    # Is numpy INTACT, not merely present? A version number says nothing about whether
    # the C extension underneath it loads, and an interrupted install (this site's proxy
    # times out often) leaves exactly that: the right version, a broken import.
    try:
        import numpy.core.multiarray as _ma
        print("  numpy.core.multiarray OK   {}".format(getattr(_ma, "__file__", "?")))
    except Exception as exc:                                            # noqa: BLE001
        print("  numpy.core.multiarray **BROKEN**: {}: {}".format(type(exc).__name__, exc))
        detail(exc)
    # numpy 1.26 ships `numpy/_core/` only as a shim for unpickling numpy 2 arrays; the
    # real module is `numpy.core`. torch's complaint names `numpy._core.multiarray`, so
    # import it here: on a clean 1.26.4 this resolves through the shim, and when it does
    # not, the traceback names the file that raises rather than torch's one-line summary.
    try:
        import importlib as _il
        _c2 = _il.import_module("numpy._core.multiarray")
        print("  numpy._core.multiarray OK  {}".format(getattr(_c2, "__file__", "?")))
    except Exception as exc:                                            # noqa: BLE001
        print("  numpy._core.multiarray **BROKEN**: {}: {}".format(
            type(exc).__name__, exc))
        detail(exc)
    try:
        import glob as _glob
        import numpy as _npx
        sp = str(Path(_npx.__file__).resolve().parent.parent)
        odd = sorted(n.split("/")[-1] for n in _glob.glob(sp + "/*umpy*")
                     if n.split("/")[-1].startswith("~"))
        print("  site-packages         {}".format(sp))
        if odd:
            print("  **LEFTOVERS FROM AN INTERRUPTED INSTALL**: {}".format(", ".join(odd)))
            print("  A `~umpy` directory shadows the real package and produces exactly")
            print("  this class of AttributeError. Remove it and reinstall numpy.")
    except Exception:                                                   # noqa: BLE001
        pass

    # A declared bound that the installed numpy violates is a finding on its own. On
    # an104 (2026-09-13) torch's round trip WORKED with numpy 2.4.6 while scipy 1.13.1
    # declared `numpy <2.3` and warned at import -- and this probe said nothing, because
    # the check below used to live inside the round-trip failure branch only.
    try:
        import numpy as _npc
        _ver = str(_npc.__version__)
        _maj = int(_ver.split(".")[0])
        _min = int(_ver.split(".")[1]) if "." in _ver else 0

        def _violates(dep):
            # numpy <2.3 / numpy >=2 / numpy <2 / numpy >=1.19,<3 -- the forms conda-meta uses.
            # conda-meta gives one bound per string ("numpy >=1.19,<3"); this probe joins
            # a package's several strings with "; ". Split on both, or the second string
            # rides along inside the first clause and int() raises -- which the try
            # around this block would then swallow, silently. Caught by a test.
            flat = dep.replace("numpy", "").replace(" ", "").replace(";", ",")
            for clause in flat.split(","):
                if clause.startswith("<") and not clause.startswith("<="):
                    hi = clause[1:].split(".")
                    if (_maj, _min) >= (int(hi[0]), int(hi[1]) if len(hi) > 1 else 0):
                        return True
                if clause.startswith(">=") and _maj < int(clause[2:].split(".")[0]):
                    return True
            return False

        _bad = [(n, v, d) for (n, v, d) in numpy_needs if _violates(d)]
        if _bad:
            print()
            print("  **numpy {} VIOLATES A DECLARED BOUND:**".format(_ver))
            for n, v, d in _bad:
                print("      {} {} requires  {}".format(n, v, d))
            print("  The package was built and solved against a numpy in that range; outside")
            print("  it the import may warn (scipy does) or misbehave without warning. This")
            print("  is a finding whether or not the torch round trip below succeeds.")
    except Exception:                                                   # noqa: BLE001
        pass

    # The one question that matters: can torch actually use numpy? torch prints its
    # complaint as a UserWarning at import and then fails much later, so provoke it here.
    try:
        import numpy as _np
        import torch as _t
        _t.from_numpy(_np.zeros(3)).sum().item()
        print("  torch <-> numpy       OK (a round trip worked)")
    except Exception as exc:                                            # noqa: BLE001
        print("  torch <-> numpy       **BROKEN**: {}: {}".format(type(exc).__name__, exc))
        print()
        detail(exc)
        print()
        print("  numpy and this torch build cannot share an array. Nothing below can pass,")
        print("  and no GPU is involved. The version number alone does NOT settle this:")
        print("  on 2026-09-12 numpy was 1.26.4 -- the version this file pins -- and the")
        print("  round trip still failed, so the fault is a binary built against a")
        print("  different numpy, or an install left half-written. The traceback above")
        print("  names the module; that is the thing to reinstall.")
        if outside:
            print()
            print("  **START WITH THE SHADOWED PACKAGE(S) LISTED ABOVE.** That is the")
            print("  likeliest cause here and it costs one environment variable to test.")
        # The declared requirement beats every inference about which numpy is "right".
        import numpy as _npv
        major = int(str(_npv.__version__).split(".")[0])
        conflict = [(n, v, d) for (n, v, d) in numpy_needs
                    if (">=2" in d.replace(" ", "") and major < 2)
                    or ("<2" in d.replace(" ", "") and major >= 2)]
        if conflict:
            print()
            print("  **THE INSTALLED PACKAGES ASK FOR A DIFFERENT NUMPY THAN IS HERE.**")
            for n, v, d in conflict:
                print("      {} {} requires numpy {}, and numpy is {}".format(
                    n, v, d, _npv.__version__))
            print("  That is a declared dependency, not an inference. Satisfy it -- on a")
            print("  login node -- rather than reinstalling the numpy already present.")
        print()
        if conflict:
            print("  Fix: satisfy the DECLARED requirement printed above -- that is a fact")
            print("  about the installed package, not a guess about which numpy is right.")
        else:
            print("  No declared conflict was found, so the numpy version is NOT known to")
            print("  be the fault. Do not reinstall numpy on a hunch; find what the")
            print("  traceback names first.")
        print("  Whatever the change, run it ON A LOGIN NODE -- a compute node has no")
        print("  outbound network and mamba fails with 'Failed to connect to <proxy>'.")
        print("  Then re-run this probe before submitting anything.")
        fail.append("numpy/torch ABI")
        return report(fail)

    # ---- 3. platforms --------------------------------------------------------------
    rule("3. platforms this OpenMM was built with")
    import openmm
    print("  openmm {}".format(openmm.version.version))
    names = [openmm.Platform.getPlatform(i).getName()
             for i in range(openmm.Platform.getNumPlatforms())]
    print("  platforms  {}".format(", ".join(names)))
    if platform_name not in [n.upper() for n in names]:
        print("  **{} IS NOT COMPILED INTO THIS OPENMM.** A cpu-only build was installed;"
              .format(platform_name))
        print("  environment-tianhe-gpu.yml asks for a *cuda* build to make that a solver")
        print("  error rather than a silent CPU run.")
        fail.append("no {} platform".format(platform_name))
        return report(fail)

    # ---- 4. the empty context: the PTX JIT, alone ----------------------------------
    rule("4. a 1-particle context on {} -- this is the PTX JIT and nothing else".format(platform_name))
    t0 = time.time()
    try:
        sysm = openmm.System()
        sysm.addParticle(1.0)
        integ = openmm.VerletIntegrator(0.001)
        ctx = openmm.Context(sysm, integ,
                             openmm.Platform.getPlatformByName(platform_name),
                             platform_properties(platform_name, args.precision,
                                                 args.dtype))
        print("  built in {:.2f} s   {}".format(time.time() - t0, describe_context(ctx)))
        del ctx
    except Exception as exc:                                            # noqa: BLE001
        print("  FAILED after {:.2f} s".format(time.time() - t0))
        print("  {}: {}".format(type(exc).__name__, exc))
        detail(exc)
        if "PTX" in str(exc):
            print()
            print("  This is the toolkit/driver mismatch, isolated: no torch, no MACE, no")
            print("  model file involved. Section 1's numbers are the whole diagnosis.")
        elif "CUDA error (34)" in str(exc) or "STUB" in str(exc).upper():
            print()
            print("  CUDA error 34 is CUDA_ERROR_STUB_LIBRARY: the libcuda.so this process")
            print("  loaded is the link-time stub, not the driver. Section 1 shows where it")
            print("  came from and which LD_LIBRARY_PATH entry to remove.")
        fail.append("context on {}".format(platform_name))
        return report(fail)

    # ---- 5. the real system --------------------------------------------------------
    rule("5. the MACE TorchForce system")
    import numpy as np
    # The same import forms the production driver uses, so this probe cannot pass
    # while s0_B_qha_trajectory_openmm.py fails to import.
    from openqha import engine, openmm_mace
    t0 = time.time()
    try:
        name = engine.engine_name()
        model_path = str(engine.model_path(name))
        print("  engine     {}".format(name))
        print("  weights    {}".format(model_path))
        import torch as _torch
        want_dtype = getattr(_torch, args.dtype)
        system, force_record = openmm_mace.build_system(
            NUMBERS, MASSES, model_path,
            example_positions_nm=np.array(POSITIONS_A) / openmm_mace.NM_TO_A,
            platform=platform_name, dtype=want_dtype)
        RESULT["trace_s"] = time.time() - t0
        print("  traced and built in {:.1f} s   dtype {}   traced on {}".format(
            time.time() - t0, force_record.get("dtype"),
            force_record.get("trace_device")))
        # Section 6 below used to be where a device mismatch surfaced, 61 s after the
        # trace and with a TorchScript traceback that names no file of ours. Print what
        # the graph carries, so the mismatch is visible here instead.
        print("  device constants in the graph  {}   (.to() mismatches: {})".format(
            ", ".join(force_record.get("device_constants") or []) or "(none)",
            ", ".join(force_record.get("device_mismatches") or []) or "none"))
    except Exception as exc:                                            # noqa: BLE001
        print("  FAILED after {:.1f} s".format(time.time() - t0))
        print("  {}: {}".format(type(exc).__name__, exc))
        detail(exc)
        fail.append("build_system")
        return report(fail)

    # ---- 5b. what does this precision cost? ----------------------------------------
    # A ms/step with no error beside it cannot settle anything: the question is never
    # "is float32 faster", it is "is float32 close enough for what this number feeds".
    # Branch B feeds a covariance matrix of positions, so the force matters more than
    # the energy, and both are printed.
    #
    # The baseline is float64 on the CPU platform, which is the configuration every
    # number in this repository was measured with. When that IS the configuration being
    # probed, there is nothing to compare and the section says so.
    rule("5b. E and F against float64 on the CPU, at the same geometry")
    from openmm import unit as _unit

    # **Two geometries, and the second is the one that means anything.** At the
    # reference geometry the forces are ~0 by construction (it is essentially a
    # minimum), so a relative force error there divides by nothing and reads as a huge
    # percentage of a vanishing quantity. openqha/thermochem/hessian.py already carries
    # this warning for the analytic-vs-finite-difference comparison; it applies here
    # verbatim. The displaced geometry is a fixed pseudo-random 0.1 A kick -- the scale
    # of thermal motion at 298 K -- where forces are hundreds of kJ/mol/nm and the
    # comparison is against something.
    rng = np.random.default_rng(20260912)
    GEOMETRIES = [
        ("at the reference geometry", np.array(POSITIONS_A)),
        ("displaced 0.1 A (forces are real here)",
         np.array(POSITIONS_A) + rng.normal(0.0, 0.1, (len(POSITIONS_A), 3))),
    ]

    def single_point(sys_obj, plat, props, positions_A):
        integ = openmm.VerletIntegrator(0.001 * _unit.picoseconds)
        ctx = openmm.Context(sys_obj, integ,
                             openmm.Platform.getPlatformByName(plat), props)
        ctx.setPositions((np.array(positions_A) / openmm_mace.NM_TO_A) * _unit.nanometer)
        st = ctx.getState(getEnergy=True, getForces=True)
        e = st.getPotentialEnergy().value_in_unit(_unit.kilojoule_per_mole)
        f = st.getForces(asNumpy=True).value_in_unit(
            _unit.kilojoule_per_mole / _unit.nanometer)
        del ctx
        return float(e), np.array(f)

    is_baseline = (platform_name == "CPU" and args.dtype == "float64")
    props_here = platform_properties(platform_name, args.precision, args.dtype)
    if args.no_accuracy:
        print("  skipped (--no-accuracy)")
    try:
        if args.no_accuracy:
            raise _SkipSection()
        if is_baseline:
            for label, geom in GEOMETRIES:
                e_here, f_here = single_point(system, platform_name, props_here, geom)
                print("  {:40s} E {:.6f}   max|F| {:.4f}".format(
                    label, e_here, np.abs(f_here).max()))
            print("  this IS the float64/CPU baseline, so there is nothing to compare it")
            print("  against here. Run --dtype float32 or --platform CUDA for a delta.")
        else:
            base_system, _base_record = openmm_mace.build_system(
                NUMBERS, MASSES, model_path,
                example_positions_nm=np.array(POSITIONS_A) / openmm_mace.NM_TO_A,
                platform="CPU", dtype=_torch.float64)
            for label, geom in GEOMETRIES:
                e_here, f_here = single_point(system, platform_name, props_here, geom)
                e_ref, f_ref = single_point(base_system, "CPU", {}, geom)
                d_f = np.abs(f_here - f_ref)
                scale = max(np.abs(f_ref).max(), 1e-30)
                if "displaced" in label:        # the non-degenerate one
                    RESULT.update(maxF=float(np.abs(f_here).max()),
                                  dE_ref=float(e_here - e_ref),
                                  maxdF=float(d_f.max()),
                                  rel_dF=float(d_f.max() / scale))
                print("  {}".format(label))
                print("      this build    E {:.6f} kJ/mol   max|F| {:.4f} kJ/mol/nm"
                      .format(e_here, np.abs(f_here).max()))
                print("      float64/CPU   E {:.6f} kJ/mol   max|F| {:.4f} kJ/mol/nm"
                      .format(e_ref, np.abs(f_ref).max()))
                print("      difference    dE {:+.4e} kJ/mol   max|dF| {:.4e} kJ/mol/nm"
                      "  ({:.2e} of max|F|)".format(
                          e_here - e_ref, d_f.max(), d_f.max() / scale))
            # Numbers to hold those against, rather than eyeballing a magnitude.
            print("  for scale     k_B T = 2.479 kJ/mol at 298.15 K; branch B's T*S")
            print("                budget is 0.084 kJ/mol (0.02 kcal/mol). An absolute")
            print("                energy OFFSET cancels in every difference branch B")
            print("                takes -- the force error is what reaches a covariance.")
    except _SkipSection:
        pass
    except Exception as exc:                                            # noqa: BLE001
        print("  FAILED: {}: {}".format(type(exc).__name__, exc))
        detail(exc)
        fail.append("precision comparison")

    # ---- 6. integrate --------------------------------------------------------------
    rule("6. {} steps, and an energy that must be finite".format(args.steps))
    from openmm import unit
    t0 = time.time()
    try:
        integ = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        ctx = openmm.Context(system, integ,
                             openmm.Platform.getPlatformByName(platform_name),
                             platform_properties(platform_name, args.precision,
                                                 args.dtype))
        print("  running on      {}   MACE dtype {}".format(
            describe_context(ctx), args.dtype))
        ctx.setPositions((np.array(POSITIONS_A) / openmm_mace.NM_TO_A) * unit.nanometer)
        # **Velocities, or this proves nothing.** From a minimum at rest the atoms do not
        # move in 10 fs and the energy is unchanged to every digit printed -- which reads
        # as "it ran" whether or not a single force was ever evaluated. At 298.15 K with a
        # fixed seed the energy MUST move, and section 6 asserts that it did.
        ctx.setVelocitiesToTemperature(298.15 * unit.kelvin, 20260912)
        e0 = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)
        print("  E(0)       {:.6f} kJ/mol   (velocities at 298.15 K, seed 20260912)".format(e0))
        integ.step(args.steps)
        st = ctx.getState(getEnergy=True, getPositions=True)
        e1 = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        dt = time.time() - t0
        RESULT.update(e0=float(e0), e1=float(e1), dE=float(e1 - e0),
                      ms_per_step=1000.0 * dt / max(1, args.steps),
                      steps=int(args.steps), wall_s=float(dt))
        print("  E({})      {:.6f} kJ/mol".format(args.steps, e1))
        print("  {} steps in {:.2f} s  =  {:.1f} ms/step  ->  {:.0f} s/ps at 1 fs".format(
            args.steps, dt, 1000.0 * dt / max(1, args.steps),
            1000.0 * dt / max(1, args.steps)))
        import math
        if not (math.isfinite(e0) and math.isfinite(e1)):
            print("  **NON-FINITE ENERGY.** The card ran, and the numbers are not usable.")
            fail.append("non-finite energy")
        elif e1 == e0:
            print("  **THE ENERGY DID NOT MOVE** in {} steps from a thermalised start.".format(
                args.steps))
            print("  Either the integrator did nothing or the force is identically zero;")
            print("  both would let a trajectory 'complete' and mean nothing.")
            fail.append("energy unchanged")
        else:
            print("  dE         {:+.6f} kJ/mol   (it moved, so the force was evaluated)"
                  .format(e1 - e0))
    except Exception as exc:                                            # noqa: BLE001
        print("  FAILED after {:.1f} s".format(time.time() - t0))
        print("  {}: {}".format(type(exc).__name__, exc))
        detail(exc)
        fail.append("integration")

    return report(fail)


def report(fail):
    rule("verdict")
    if not fail:
        print("  branch B can run here. The ms/step above is the number to size a")
        print("  campaign with. Measured on an45, 2026-09-12, MACE-OFF23_medium,")
        print("  10 atoms, 200 steps, one trajectory at a time:")
        print("      CUDA  float32  Precision=single    25.9 ms/step")
        print("      CPU   float32  (112 threads)       34.9")
        print("      CUDA  float64  Precision=double    39.2")
        print("      CPU   float64  (112 threads)       61.7")
        print("      CUDA  float64  Precision=single    74.2  <- MISMATCHED, the default")
        print("  **Match the platform precision to the MACE dtype.** The last row is")
        print("  1.9x the fourth for the same answer, and it is what you get by leaving")
        print("  the properties out. float32 costs 1e-06 relative force error; double")
        print("  on CUDA reproduces float64 on the CPU to 3.9e-14.")
        return 0
    for f in fail:
        print("  FAILED  {}".format(f))
    return 1


def quiet_main():
    """Run every section with the output captured; print one line of numbers.

    Anything that fails still prints everything -- a silent failure in a sweep is the
    worst of both worlds, a missing row with no reason attached.
    """
    import contextlib
    import io
    import warnings

    warnings.filterwarnings("ignore")
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = main()
    except BaseException:                                               # noqa: BLE001
        sys.stdout.write(buf.getvalue())
        raise
    if code != 0:
        sys.stdout.write(buf.getvalue())
        return code
    tag = ""
    for i, a in enumerate(sys.argv):
        if a == "--tag" and i + 1 < len(sys.argv):
            tag = sys.argv[i + 1]
    r = RESULT
    bits = ["{:>10s}".format(tag or "-")]
    bits.append("{:>7.2f} ms/step".format(r.get("ms_per_step", float("nan"))))
    bits.append("wall {:>6.2f}s".format(r.get("wall_s", float("nan"))))
    bits.append("trace {:>5.1f}s".format(r.get("trace_s", float("nan"))))
    bits.append("E0 {:.6f}".format(r.get("e0", float("nan"))))
    bits.append("dE {:+.6f}".format(r.get("dE", float("nan"))))
    if "maxF" in r:
        bits.append("max|F| {:.4f}".format(r["maxF"]))
        bits.append("dE_ref {:+.2e}".format(r.get("dE_ref", float("nan"))))
        bits.append("max|dF| {:.2e} ({:.1e} rel)".format(
            r.get("maxdF", float("nan")), r.get("rel_dF", float("nan"))))
    print("  ".join(bits), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(quiet_main() if "--quiet" in sys.argv else main())
