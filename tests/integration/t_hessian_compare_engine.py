"""The engine seam: the MACE Hessian at a reference geometry today equals
the stored fixture.

INTEGRATION. Loads the real MACE-OFF23_medium calculator, evaluates the analytic Hessian
and the forces at the reference (wB97M-D3BJ/def2-TZVPPD) geometry of propanal basin 0
(`$atoms` of tests/data/propanal_molecule/orca/.../basin00/job.hess), and compares with
`mace/basin00/{hessian,forces}_at_wb97m-d3bj_def2-tzvppd.npy` written 2026-09-17. A drift
here means the engine changed, not the comparison. SKIPs without the model.
"""
import sys
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
SRC = ROOT / "tests" / "data" / "propanal_molecule"
LEVEL = "wb97m-d3bj_def2-tzvppd"


def main():
    import numpy as np
    from ase import Atoms
    from openqha.potentials import engine
    from openqha.qm_interfaces import orca
    from openqha.store import layout
    from openqha.thermochem import hessian as hessian_mod
    try:
        calc, name, _prov = engine.calculator(device="cpu")
    except Exception as exc:                                   # noqa: BLE001
        print("SKIP: {}".format(exc))
        return 0
    parsed = orca.parse_hess(layout.orca_level_file(SRC, LEVEL, 0, ".hess"))
    atoms = Atoms(symbols=parsed["symbols"],
                  positions=np.asarray(parsed["positions_bohr"]) / orca.BOHR_PER_ANGSTROM)
    h, asym = hessian_mod.hessian(atoms, calc, mode="analytic")
    atoms.calc = calc
    f = atoms.get_forces()
    h0 = np.load(SRC / "mace" / "basin00" / "hessian_at_{}.npy".format(LEVEL))
    f0 = np.load(SRC / "mace" / "basin00" / "forces_at_{}.npy".format(LEVEL))
    dh, df = float(np.abs(h - h0).max()), float(np.abs(f - f0).max())
    print("  {}: max|dH| = {:.2e} eV/A^2, max|dF| = {:.2e} eV/A, asymmetry {:.1e}".format(name, dh, df, asym))
    ok = dh < 1e-6 and df < 1e-6
    print("  {:70s} {}".format("MACE Hessian and forces at the reference geometry match the fixture to 1e-6",
                               "ok" if ok else "FAIL"))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
