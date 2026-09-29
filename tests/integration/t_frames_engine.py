"""Integration (needs the MACE weights and a molecule directory with basins): the Frame
set built by `frames.generate` reuses the basin's stored Hessian to 0, and a displaced
frame carries energy and forces only -- no engine Hessian, `has_hessian = False`
(S0-C-66, ticket 29) -- with the registered engine's NAME beside it: the identity since
2026-09-27 is the engine name + the resolved weight file (decision 04; no fingerprint,
no pin).

    python tests/integration/t_frames_engine.py <molecule dir>

The molecule directory is copied to a temporary directory first and the check runs
there: the source -- the tests/data fixture by default -- is never written to (this test
used to rewrite the fixture in place; fixed 2026-09-29).
"""
import shutil
import sys
import tempfile
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
from openqha.store import layout, property as prop           # noqa: E402


def main(molecule):
    with tempfile.TemporaryDirectory(prefix="frames_engine_") as tmp:
        work = Path(tmp) / Path(molecule).name
        shutil.copytree(molecule, work)
        _check(work)


def _check(molecule):
    out = frames.generate(molecule, n_displaced=1)
    lvl = out["info"]["LEVEL"]
    fb = frames.read_frames(layout.frames_file(molecule, "basin", lvl))
    h0 = np.load(Path(molecule) / "mace" / "basin00" / "hessian.npy")
    d_basin = float(np.abs(fb[0].info["hessian"] - h0).max())
    calc, name, prov = engine.calculator()
    a = frames.read_frames(layout.frames_file(molecule, "displaced", lvl))[0]
    # S0-C-66 (ticket 29): a displaced frame is energy + forces only -- the file has no
    # `hessian` key and `has_hessian = False`; the registered engine's name is recorded
    # per frame, and the Record names the resolved weight file (2026-09-27, decision 04).
    no_disp_hessian = a.info.get("has_hessian") is False and "hessian" not in a.info
    rec_info = prop.load(layout.frames_dir(molecule) / "frames.toml")["Calculation_Info"]
    ok = (d_basin == 0.0 and no_disp_hessian and a.info["engine"] == name
          and rec_info["WEIGHTS_FILE"] == prov["weights_path"])
    print("basin frame vs hessian.npy max|dH| = {}   displaced frame carries no engine Hessian (S0-C-66)   "
          "engine in frame {}   weights in Record {}".format(
              d_basin, name if a.info["engine"] == name else "DIFFERS", rec_info["WEIGHTS_FILE"]))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    molecule = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "tests" / "data" / "propanal_molecule")
    main(molecule)
