#!/usr/bin/env python
"""Why does this potential return a non-finite energy? Answer it in one command.

TOOLING. Read-only, runs in seconds, safe on a login node. It exists because on
2026-09-09 and again on 2026-09-10 a Tianhe branch A run produced a non-finite energy on
**every** call (1628 then 1719 of them, 100%), and the same code with the same geometry
on the workstation returned -5259.489825324396 eV. Two things can produce that and they
need completely different fixes:

  * **the file** -- what is on disk is not a usable set of weights;
  * **the stack** -- the weights are fine and torch / e3nn / mace on this machine turn
    them into NaN.

The discriminator is cheap and it is step 3 below: **load the checkpoint and look at the
parameters themselves.** A tensor full of NaN in the file is the first case. Parameters
that are all finite, followed by a forward pass that is not, is the second.

Run it on BOTH machines and put the two outputs side by side::

    python scripts/tooling/s0_probe_potential.py

Exit status is 0 only if the potential returns a finite energy and finite forces.
"""
import sys
from pathlib import Path


def _repo_root():
    for p in Path(__file__).resolve().parents:
        if (p / "openqha" / "__init__.py").is_file():
            return p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))

#: One fixed molecule so the two machines are compared on the same input. This is the
#: exact acetone CREST handed the model on Tianhe, copied from the calcspace of job
#: 7347197 -- an ordinary geometry, C-C 1.52 A, C=O 1.21 A, nothing unusual about it.
SYMBOLS = ["C", "C", "C", "O", "H", "H", "H", "H", "H", "H"]
POSITIONS = [
    [-1.2681248863, 0.7031358649, 0.0064151880],
    [0.0000958229, -0.1143359431, -0.0055449952],
    [1.2673727530, 0.7047445624, -0.0021823544],
    [0.0007795035, -1.3178067377, -0.0165894488],
    [-1.2059059001, 1.4955296623, 0.7477351398],
    [-1.3982716110, 1.1640095224, -0.9713847976],
    [-2.1252228643, 0.0673254127, 0.2109085715],
    [2.1254595878, 0.0736896073, -0.2171899694],
    [1.2046689800, 1.5102483048, -0.7292046014],
    [1.3963396460, 1.1483492205, 0.9837167461],
]

#: Measured on this project's workstation, 2026-09-10, MACE-OFF23_medium, float64,
#: numpy 1.26.4 / torch 2.12.1 / e3nn 0.4.4 / mace 0.3.16. A machine that reproduces
#: these has a working potential; one that does not is what this script is for.
REFERENCE = dict(energy_eV=-5259.489825324396, max_force_eV_A=0.3852005576922286,
                 n_tensors=77, n_parameters=2265399, bytes=18350596)


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main():
    verdicts = []

    # ---- 1. the stack ------------------------------------------------------------------
    rule("1. WHAT IS INSTALLED HERE")
    import numpy
    import torch
    mods = [numpy, torch]
    for name in ("e3nn", "mace"):
        try:
            mods.append(__import__(name))
        except ImportError as exc:
            print("  {:8s} NOT IMPORTABLE: {}".format(name, exc))
    for m in mods:
        print("  {:8s} {:12s} {}".format(
            m.__name__, str(getattr(m, "__version__", "?")),
            str(getattr(m, "__file__", ""))))
    print("  python   {}".format(sys.executable))
    # openQHA rebinds MACE's neighbour-list construction (openqha/potentials/mace_patch.py).
    # Whether that rebinding takes hold depends on the MACE tree that is installed, so it
    # is part of "the stack" and a candidate for a machine-specific NaN. Print it here
    # rather than leave it to be guessed at.
    try:
        from openqha.potentials import mace_patch
        mace_patch.apply(strict=False)
        st = mace_patch.state()
        print("  patch    applied={}  sites={}  error={}".format(
            st.get("applied"), st.get("sites"), st.get("error")))
    except Exception as exc:                                              # noqa: BLE001
        print("  patch    could not be inspected: {}: {}".format(type(exc).__name__, exc))

    # ---- 2. the file -------------------------------------------------------------------
    rule("2. WHICH FILE")
    from openqha.potentials import engine
    name = engine.engine_name()
    path = engine.model_path(name)
    size = path.stat().st_size
    print("  engine        {}".format(name))
    print("  path          {}".format(path))
    print("  bytes         {}   (workstation: {})".format(size, REFERENCE["bytes"]))
    if size != REFERENCE["bytes"]:
        print("  NOTE: a different size does NOT by itself mean different weights -- a")
        print("        torch re-serialisation changes the size while every tensor stays")
        print("        identical. Step 3 is the one that decides.")

    # ---- 3. THE DISCRIMINATOR ----------------------------------------------------------
    rule("3. ARE THE PARAMETERS IN THE FILE FINITE?")
    print("  (this is the step that separates 'bad file' from 'bad stack')")
    bad = []
    n_tensors = 0
    n_elements = 0
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
        state = obj.state_dict() if hasattr(obj, "state_dict") else obj
        for key in sorted(state):
            v = state[key]
            if not hasattr(v, "dtype") or not v.is_floating_point():
                continue
            n_tensors += 1
            n_elements += v.numel()
            if not bool(torch.isfinite(v).all()):
                n_bad = int((~torch.isfinite(v)).sum())
                bad.append((key, tuple(v.shape), n_bad, v.numel()))
    except Exception as exc:                                              # noqa: BLE001
        print("  THE FILE COULD NOT BE LOADED AT ALL: {}: {}".format(
            type(exc).__name__, exc))
        print()
        print("  VERDICT: the file is not a usable checkpoint. Re-copy it.")
        return 1

    print("  floating-point tensors  {}   (workstation: {})".format(
        n_tensors, REFERENCE["n_tensors"]))
    print("  parameters              {}".format(n_elements))
    if bad:
        print("  **NON-FINITE PARAMETERS IN THE FILE ITSELF**")
        for key, shape, n_bad, total in bad[:10]:
            print("    {:50s} {} of {} bad".format(key + " " + str(shape), n_bad, total))
        print()
        print("  VERDICT: **THE FILE IS THE PROBLEM.** The weights on disk contain NaN or")
        print("           Inf before anything is computed with them. No version of torch")
        print("           can fix that. Re-copy the model file.")
        return 1
    print("  every parameter is finite")
    verdicts.append("the file's parameters are clean")

    # ---- 4. the forward pass -----------------------------------------------------------
    rule("4. WHAT COMES OUT FOR ONE FIXED ACETONE")
    from ase import Atoms
    ok = True
    for dtype in ("float64", "float32"):
        try:
            engine._CACHE.clear()
            saved, engine.DTYPE = engine.DTYPE, dtype
            calc, _, _ = engine.calculator(device="cpu")
            engine.DTYPE = saved
            atoms = Atoms(symbols=SYMBOLS, positions=POSITIONS)
            atoms.calc = calc
            e = float(atoms.get_potential_energy())
            f = atoms.get_forces()
            fin = (e == e and abs(e) != float("inf") and bool(numpy.isfinite(f).all()))
            print("  {:8s} energy {!r:>24}   forces finite {}".format(dtype, e, fin))
            if dtype == "float64":
                if fin:
                    d = e - REFERENCE["energy_eV"]
                    print("           workstation {!r}   difference {:.3e} eV".format(
                        REFERENCE["energy_eV"], d))
                else:
                    ok = False
        except Exception as exc:                                          # noqa: BLE001
            print("  {:8s} RAISED {}: {}".format(dtype, type(exc).__name__, exc))
            ok = False

    # ---- 5. the verdict ----------------------------------------------------------------
    rule("5. VERDICT")
    if ok:
        print("  The potential works on this machine. If a run still produces non-finite")
        print("  energies, the difference is not the model -- look at what is being sent")
        print("  to it (S0_MACE_TRACE records every call).")
        return 0
    print("  **THE STACK IS THE PROBLEM, NOT THE FILE.**")
    print("  Every parameter in the checkpoint is finite, and the forward pass still is")
    print("  not. That is torch / e3nn / mace on this machine, not the weights.")
    print()
    print("  Compare section 1 against the workstation:")
    print("    numpy 1.26.4   torch 2.12.1   e3nn 0.4.4   mace 0.3.16")
    print("  and rebuild the environment to match:  bash install_dependency.sh --tianhe")
    return 1


if __name__ == "__main__":
    sys.exit(main())
