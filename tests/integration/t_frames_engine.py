"""Integration (needs the MACE weights and a molecule directory with basins): the Frame
set built by `frames.generate` reuses the basin's stored Hessian to 0, and a displaced
frame carries energy and forces only -- no engine Hessian, `has_hessian = False`
(S0-C-66, ticket 29) -- with the engine's parameter fingerprint beside it.

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


def main(molecule):
    out = frames.generate(molecule, n_displaced=1)
    lvl = out["info"]["LEVEL"]
    fb = frames.read_frames(layout.frames_file(molecule, "basin", lvl))
    h0 = np.load(Path(molecule) / "mace" / "basin00" / "hessian.npy")
    d_basin = float(np.abs(fb[0].info["hessian"] - h0).max())
    calc, name, prov = engine.calculator()
    a = frames.read_frames(layout.frames_file(molecule, "displaced", lvl))[0]
    # S0-C-66 (ticket 29): a displaced frame is energy + forces only -- the file has no
    # `hessian` key and `has_hessian = False`; the engine fingerprint is still recorded.
    no_disp_hessian = a.info.get("has_hessian") is False and "hessian" not in a.info
    ok = d_basin == 0.0 and no_disp_hessian and a.info["engine_params_sha256"] == prov["params_sha256"]
    print("basin frame vs hessian.npy max|dH| = {}   displaced frame carries no engine Hessian (S0-C-66)   fingerprint in frame {}".format(
        d_basin, "matches provenance" if a.info["engine_params_sha256"] == prov["params_sha256"] else "DIFFERS"))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    molecule = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "tests" / "data" / "propanal_molecule")
    main(molecule)
