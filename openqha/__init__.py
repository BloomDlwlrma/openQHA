"""stage 0 -- the comparison-sample pipeline, with no alchemy in it.

The module split matches the package 0 deliverable in plan
`.mem/plan/plan_stage0-vdos-baseline.md` section 4:

    config.py      this repository's own configuration loading (configs/openqha.yaml)
    filters.py     species selection gates F0-F6 (neutral, no unpaired electrons, no formal charge)
    engine.py      the single potential (MACE-OFF23-SC) and the record of where it came from
    conformers.py  package 1: ETKDG embedding -> optimisation on the potential -> energy
                   ordering -> best-root-mean-square-deviation deduplication
    hessian.py     package 2: finite-difference Hessian, Eckart projection, normal frequencies
    thermo.py      rigid-rotor harmonic-oscillator thermodynamics, and the conversion from a
                   frequency error to a free-energy error

**Since 2026-08-28 stage 0 is an independent open framework**, to be pushed to GitHub
on its own. It therefore **imports nothing from, and reads no file of, stage 1 or
stage 2** -- the edge set, species table, symmetry numbers, parameters and data paths
are all written out again in `configs/openqha.yaml`, and anything copied
from elsewhere records its source, the date it was copied, and a checksum. See
decision `D0-41`.
"""
__version__ = "0.2.0"

from pathlib import Path

#: This repository's root. Every stage 0 path is relative to it, and **there is no
#: longer any constant pointing at another stage**.
S0_ROOT = Path(__file__).resolve().parents[1]

__all__ = ["S0_ROOT", "__version__"]
