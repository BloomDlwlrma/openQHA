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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", default="CUDA")
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()
    want = args.platform.upper()
    fail = []

    # ---- 1. versions ---------------------------------------------------------------
    rule("1. versions -- nvrtc vs the driver (the PTX rule)")
    from openqha import gpu_preflight
    d = gpu_preflight.describe()
    print("  nvrtc                 {}".format(d["nvrtc"] or "not loadable here"))
    print("  nvrtc comes from      {}".format(d.get("nvrtc_path") or "(path unknown)"))
    print("  driver supports CUDA  {}".format(d["driver_cuda"] or "not readable here"))
    print("  verdict               {}".format(d["reason"]))
    if d["nvrtc"] and d["driver_cuda"] and not d["ok"]:
        print()
        if want == "CUDA":
            print("  **STOP HERE.** OpenMM JITs every kernel from PTX, and this driver")
            print("  cannot read PTX from that toolkit. Sections 4-6 cannot pass. Fix:")
            print("      mamba install -n openqha-gpu cuda-version={}.{}".format(*d["driver_cuda"]))
            fail.append("nvrtc newer than the driver")
        else:
            # The CPU platform compiles nothing, so this mismatch cannot affect the run
            # being probed. Reported, not counted -- a control that fails for a reason
            # unrelated to what it controls is not a control.
            print("  (not a failure for --platform {}: nothing here JITs PTX. It would be"
                  .format(want))
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
    stack = {}
    for mod in ("numpy", "torch", "openmm", "openmmtorch", "e3nn", "mace", "scipy", "ase"):
        try:
            m = __import__(mod)
            stack[mod] = getattr(m, "__version__", "?")
            print("  {:12s} {:24s} {}".format(mod, str(stack[mod]),
                                              getattr(m, "__file__", "") or ""))
        except Exception as exc:                                        # noqa: BLE001
            stack[mod] = None
            print("  {:12s} {:24s} {}: {}".format(mod, "IMPORT FAILED",
                                                  type(exc).__name__, exc))
    # Is numpy INTACT, not merely present? A version number says nothing about whether
    # the C extension underneath it loads, and an interrupted install (this site's proxy
    # times out often) leaves exactly that: the right version, a broken import.
    try:
        import numpy.core.multiarray as _ma
        print("  numpy.core.multiarray OK   {}".format(getattr(_ma, "__file__", "?")))
    except Exception as exc:                                            # noqa: BLE001
        print("  numpy.core.multiarray **BROKEN**: {}: {}".format(type(exc).__name__, exc))
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
        print("  Fix:  mamba install -n openqha-gpu numpy=1.26.4")
        print("        ON A LOGIN NODE -- a compute node has no outbound network, and")
        print("        mamba fails there with 'Failed to connect to <proxy> port 3138'.")
        print("        Already cached?  mamba install -n openqha-gpu --offline numpy=1.26.4")
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
    if want not in [n.upper() for n in names]:
        print("  **{} IS NOT COMPILED INTO THIS OPENMM.** A cpu-only build was installed;"
              .format(want))
        print("  environment-tianhe-gpu.yml asks for a *cuda* build to make that a solver")
        print("  error rather than a silent CPU run.")
        fail.append("no {} platform".format(want))
        return report(fail)

    # ---- 4. the empty context: the PTX JIT, alone ----------------------------------
    rule("4. a 1-particle context on {} -- this is the PTX JIT and nothing else".format(want))
    t0 = time.time()
    try:
        sysm = openmm.System()
        sysm.addParticle(1.0)
        integ = openmm.VerletIntegrator(0.001)
        ctx = openmm.Context(sysm, integ, openmm.Platform.getPlatformByName(want))
        print("  built in {:.2f} s   platform {}".format(
            time.time() - t0, ctx.getPlatform().getName()))
        del ctx
    except Exception as exc:                                            # noqa: BLE001
        print("  FAILED after {:.2f} s".format(time.time() - t0))
        print("  {}: {}".format(type(exc).__name__, exc))
        detail(exc)
        if "PTX" in str(exc):
            print()
            print("  This is the toolkit/driver mismatch, isolated: no torch, no MACE, no")
            print("  model file involved. Section 1's numbers are the whole diagnosis.")
        fail.append("context on {}".format(want))
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
        system, force_record = openmm_mace.build_system(
            NUMBERS, MASSES, model_path,
            example_positions_nm=np.array(POSITIONS_A) / openmm_mace.NM_TO_A)
        print("  traced and built in {:.1f} s   dtype {}".format(
            time.time() - t0, force_record.get("dtype")))
    except Exception as exc:                                            # noqa: BLE001
        print("  FAILED after {:.1f} s".format(time.time() - t0))
        print("  {}: {}".format(type(exc).__name__, exc))
        detail(exc)
        fail.append("build_system")
        return report(fail)

    # ---- 6. integrate --------------------------------------------------------------
    rule("6. {} steps, and an energy that must be finite".format(args.steps))
    from openmm import unit
    t0 = time.time()
    try:
        integ = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName(want))
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
        print("  branch B can run here. The ms/step above is the number to size a campaign")
        print("  with; on a T400 workstation it is ~69 ms/step (2026-09-12), and this")
        print("  repository has no A800 measurement at all until you run this there.")
        return 0
    for f in fail:
        print("  FAILED  {}".format(f))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
