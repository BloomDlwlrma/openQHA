"""Per-step force timing for Egret-1 and MACE-OFF23-SC, and the effect of thread count.

CALIBRATION. Per-step force timing and thread scaling; its numbers size the
molecular-dynamics budget.

The molecular-dynamics budget from the lecture notes: (3 ps equilibration x 3 rounds
+ 6 ps production) / 0.5 fs = 30000 steps per species, 60000 steps for the two species. The measured per-step cost below converts straight into a total wall time.
"""
import os
import time

import numpy as np
import torch
from ase import Atoms

print("nproc        =", os.cpu_count())
print("torch threads=", torch.get_num_threads())
print("interop      =", torch.get_num_interop_threads())

# PATHS COME FROM THE REGISTRY, NOT FROM STRINGS (fixed 2026-09-09).
#
# This file used to open with three absolute paths hard-coded to one person's WSL mount,
# rooted at `.../stage0-tradition-free-energy-calc` -- a directory that has not existed
# since the 2026-09-03 rename to `openQHA`. So the script could not run anywhere,
# including on the machine it was written on. It is the same defect as the two found on
# Tianhe: a location written into a string does not move with the thing it names.
#
# The MACE weights now come from `engine.model_path()`, which resolves
# `model_root()/<filename>` -- one flat directory, `S0_MACE_ROOT` to move it.
#
# Egret is NOT in the registry: it is a comparison engine for this benchmark only and no
# production number depends on it. It stays a path, but a path you can set rather than
# one baked in, and the script says what to do when it is absent instead of dying on a
# stack trace. Register it if it ever becomes more than a benchmark.
import sys
from pathlib import Path


def _repo_root():
    """Walk up until the openqha package is actually there.

    NOT `parents[2]`. Counting directory levels is the same defect as writing the path
    into a string: it is right until a file moves, and then it silently points at the
    wrong tree. `tests/unit/t_repo_bootstrap.py` fails the build over it -- and caught
    exactly this in the 2026-09-09 rewrite of this file's header, where the fix for one
    hard-coded path introduced a level count instead.
    """
    for p in Path(__file__).resolve().parents:
        if (p / "openqha" / "__init__.py").is_file():
            return p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))
from openqha.potentials import engine  # noqa: E402

#: Which registered potential to time. Any name in `engine.ENGINES`.
MACE_ENGINE = os.environ.get("S0_ENGINE", "MACE-OFF23-SC")
MACE = str(engine.model_path(MACE_ENGINE))

#: Egret's compiled models. Set S0_EGRET_ROOT to the directory holding them.
_EGRET_ROOT = Path(os.environ.get(
    "S0_EGRET_ROOT",
    engine.model_root().parent / "egret" / "compiled_models"))
EGRET = str(_EGRET_ROOT / "EGRET_1.model")
EGRET_M = str(_EGRET_ROOT / "EGRET_1M.model")
EGRET_S = str(_EGRET_ROOT / "EGRET_1S.model")
if not Path(EGRET).is_file():
    print("Egret models not found under {}.\n"
          "  Set S0_EGRET_ROOT to the directory holding EGRET_1*.model, or run only the\n"
          "  MACE rows. Egret is a comparison engine here; no product depends on it."
          .format(_EGRET_ROOT))


def geometry():
    p = str(_repo_root() / "docs" / "orca_inputs" / "C2H5O1N1_19_36_acetamide.inp")
    lines = open(p, encoding="utf-8").read().split("\n")
    i = next(i for i, l in enumerate(lines) if l.startswith("* xyz"))
    sym, xyz = [], []
    for l in lines[i + 1:]:
        if l.strip() == "*":
            break
        f = l.split()
        sym.append(f[0]); xyz.append([float(v) for v in f[1:4]])
    return Atoms(symbols=sym, positions=np.array(xyz))


atoms = geometry()
STEPS_TOTAL = 2 * (3 * 3.0 + 6.0) * 1000 / 0.5     # total steps for the two species


def bench(calc, label, n=40):
    a = atoms.copy(); a.calc = calc
    a.get_forces()                                   # warm-up
    t0 = time.time()
    for _ in range(n):
        a.calc.results.clear()
        a.get_forces()
    dt = (time.time() - t0) / n
    print("   {:22s} {:8.1f} ms/step -> all {:.0f} steps of the lecture notes, about {:6.1f} min".format(
        label, dt * 1e3, STEPS_TOTAL, dt * STEPS_TOTAL / 60))
    return dt


for nthreads in (torch.get_num_threads(), max(1, (os.cpu_count() or 4) - 2)):
    torch.set_num_threads(nthreads)
    print()
    print("=== torch threads = {} ===".format(nthreads))
    from mace.calculators import mace_off, MACECalculator
    bench(MACECalculator(model_paths=MACE, device="cpu", default_dtype="float64"),
          MACE_ENGINE)
    bench(mace_off(model=EGRET, default_dtype="float64", device="cpu"), "Egret-1")
    bench(mace_off(model=EGRET_M, default_dtype="float64", device="cpu"), "Egret-1M")
    bench(mace_off(model=EGRET_S, default_dtype="float64", device="cpu"), "Egret-1S")

print()
print("float32 control (speed only; not for production -- optimisation and frequencies need float64):")
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
bench(mace_off(model=EGRET, default_dtype="float32", device="cpu"), "Egret-1 float32")
