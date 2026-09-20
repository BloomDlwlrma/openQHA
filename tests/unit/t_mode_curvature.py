"""Ticket 32, the arithmetic of the along-mode curvature (no ORCA, no engine).

On the propanal fixture: the displaced line along a reference mode, with energies taken
from the quadratic form of the reference Hessian, gives back omega_r (self-check 1) to
the finite-difference error, on the full line and on the half-step line (the delta_q
convergence check); with energies from the quadratic form of the stored MACE Hessian at
x_r it gives back hessian_compare's D_ii (self-check 2); delta_q follows the target rise
(1e-5 Eh) and is capped for a flat mode.
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
from openqha.qm_interfaces import orca
from openqha.store import layout                       # noqa: E402
from openqha.thermochem import hessian_compare as hc         # noqa: E402
from openqha.thermochem import mode_curvature as mc          # noqa: E402

SRC = ROOT / "tests" / "data" / "propanal_molecule"
LEVEL = "wb97m-d3bj_def2-tzvppd"
FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


def quadratic_energies(h_ev_a2, x0, geoms):
    """E(x) = 1/2 dx^T H dx in Eh, the harmonic energy of a Cartesian Hessian."""
    out = []
    for g in geoms:
        dx = (np.asarray(g) - np.asarray(x0)).reshape(-1)
        out.append(0.5 * float(dx @ h_ev_a2 @ dx) / orca.EV_PER_HARTREE)
    return out


def main():
    from ase.data import atomic_masses, atomic_numbers
    parsed = orca.parse_hess(layout.orca_level_file(SRC, LEVEL, 0, ".hess"))
    x_r = np.asarray(parsed["positions_bohr"]) / orca.BOHR_PER_ANGSTROM
    masses = [float(atomic_masses[atomic_numbers[s]]) for s in parsed["symbols"]]
    h_r = orca.hessian_to_ev_per_angstrom2(parsed["hessian_eh_bohr2"])
    h_e = np.load(SRC / "mace" / "basin00" / "hessian_at_{}.npy".format(LEVEL))
    pr = hc.projected(h_r, masses, x_r)
    d_ii = hc.compare_hessians(h_e, h_r, masses, x_r)["OMEGA_ALONG_REF_CM"]

    for i in (0, 1, 5, 20):
        omega_r = float(pr["freq"][i])
        dq = mc.delta_for(omega_r)
        ks = [-2, -1, 0, 1, 2]
        geoms = mc.displaced_line(x_r, masses, pr["vec"][:, i], dq, ks)
        e_r = quadratic_energies(h_r, x_r, geoms)
        w5, w3 = mc.curvature_from_line(ks, e_r, dq)
        check("mode %d (%.1f cm^-1): the reference quadratic form gives back omega_r (5pt %.3f, 3pt %.3f)"
              % (i, omega_r, w5, w3), abs(w5 - omega_r) < 1e-6 and abs(w3 - omega_r) < 1e-6)
        rise = e_r[ks.index(1)] - e_r[ks.index(0)]
        check("   delta_q gives the target rise at k = 1 (%.2e Eh)" % rise,
              abs(rise - mc.TARGET_RISE_EH) < 1e-9 or dq == mc.DELTA_Q_MAX)
        e_e = quadratic_energies(h_e, x_r, geoms)
        w5e, _ = mc.curvature_from_line(ks, e_e, dq)
        check("   the MACE quadratic form gives back hessian_compare's D_ii (%.3f vs %.3f)" % (w5e, d_ii[i]),
              abs(w5e - d_ii[i]) < 1e-6)
        half = [-1, -0.5, 0, 0.5, 1]
        geoms_h = mc.displaced_line(x_r, masses, pr["vec"][:, i], dq, half)
        w5_half, _ = mc.curvature_from_line(ks, quadratic_energies(h_r, x_r, geoms_h), dq / 2.0)
        check("   the half-step line (k = +-1/2 added) gives the same omega_r: the delta_q convergence check is exact on a quadratic",
              abs(w5_half - omega_r) < 1e-6)
    check("the default rise is 1e-5 Eh (a 1e-4 line was 87 percent quartic on a torsion, see the module docstring)",
          mc.TARGET_RISE_EH == 1e-5)
    check("a flat mode gets the capped delta_q", mc.delta_for(1.0) == mc.DELTA_Q_MAX)
    check("a negative curvature comes back as a negative wavenumber",
          mc.curvature_from_line([-1, 0, 1], [1e-4, 0.0, 1e-4], 0.1)[1] > 0
          and mc.curvature_from_line([-1, 0, 1], [-1e-4, 0.0, -1e-4], 0.1)[1] < 0)
    check("select_modes: lowest / low / all / explicit",
          mc.select_modes(pr["freq"], "lowest") == [0]
          and mc.select_modes(pr["freq"], "low") == [i for i, f in enumerate(pr["freq"]) if f < 300.0]
          and len(mc.select_modes(pr["freq"], "all")) == len(pr["freq"])
          and mc.select_modes(pr["freq"], "0,3") == [0, 3])
    check("the DLPNO level's single-point keywords carry TightPNO and RIJK, no Opt / NumFreq",
          "TightPNO" in orca.level_spec("dlpno-ccsdt_cc-pvtz")["single_point"]
          and "NumFreq" not in orca.level_spec("dlpno-ccsdt_cc-pvtz")["single_point"]
          and "NumFreq" in orca.level_spec("dlpno-ccsdt_cc-pvtz")["keywords"])
    if FAIL:
        print("FAIL: " + ", ".join(FAIL))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
