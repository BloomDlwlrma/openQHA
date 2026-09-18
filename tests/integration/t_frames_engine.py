"""Integration (needs the MACE weights and a molecule directory with basins): the Frame
set built by `frames.generate` reuses the basin's stored Hessian to 0, and a displaced
frame's file Hessian agrees with a fresh engine Hessian at the file's (8-decimal)
positions to < 1e-4 eV/A^2 (2.5e-6 measured on propanal 2026-09-18: the rounding of the
positions, not of the matrix).

    python tests/integration/t_frames_engine.py <molecule dir>
"""
import sys
from pathlib import Path

import numpy as np


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from openqha.data import frames                              # noqa: E402
from openqha.potentials import engine                        # noqa: E402
from openqha.store import layout                             # noqa: E402
from openqha.thermochem import hessian as hm                 # noqa: E402


def main(molecule):
    out = frames.generate(molecule, n_displaced=1)
    lvl = out["info"]["LEVEL"]
    fb = frames.read_frames(layout.frames_file(molecule, "basin", lvl))
    h0 = np.load(Path(molecule) / "mace" / "basin00" / "hessian.npy")
    d_basin = float(np.abs(fb[0].info["hessian"] - h0).max())
    calc, name, prov = engine.calculator()
    a = frames.read_frames(layout.frames_file(molecule, "displaced", lvl))[0]
    h, _ = hm.hessian(a, calc, mode="analytic")
    d_disp = float(np.abs(a.info["hessian"] - h).max())
    ok = d_basin == 0.0 and d_disp < 1e-4 and a.info["engine_params_sha256"] == prov["params_sha256"]
    print("basin frame vs hessian.npy max|dH| = {}   displaced frame file vs engine max|dH| = {:.2e}   fingerprint in frame {}".format(
        d_basin, d_disp, "matches provenance" if a.info["engine_params_sha256"] == prov["params_sha256"] else "DIFFERS"))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1])
