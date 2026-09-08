"""Unit test: the mode pairing, and the two claims example 02d rests on.

No potential is loaded. The Hessian is synthetic, built from a random orthogonal basis
with a chosen spectrum, so every expected answer is known in closed form -- including a
deliberate four-fold degenerate block, which is what the per-mode overlap gate fails on.

The two claims under test:

  * `mode_match.projected_modes` and `hessian.project_and_diagonalise` are the same
    computation. They have to be, because 02d compares eigenvectors from the first
    against frequencies the rest of the package gets from the second.
  * inside a degenerate block, per-mode overlap collapses while BLOCK overlap does not.
    That is the whole reason the gate is on the block.
"""
import sys
from pathlib import Path

import numpy as np


def _repo_root():
    for p in Path(__file__).resolve().parents:
        if (p / "openqha" / "__init__.py").is_file():
            return p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))

from openqha.quasi_harmonic import mode_match                          # noqa: E402
from openqha.thermochem import hessian as hess_mod                     # noqa: E402
from openqha.thermochem import thermo                                  # noqa: E402

N_ATOMS = 6
SEED = 20260908


def synthetic():
    """A mass-weighted Hessian with a known spectrum, including a degenerate block.

    Built as `V diag(w^2) V^T` in mass-weighted coordinates and then un-mass-weighted,
    with the rigid subspace given eigenvalue zero so the projection has something real
    to remove.
    """
    rng = np.random.RandomState(SEED)
    masses = np.array([12.011, 12.011, 15.999, 1.008, 1.008, 1.008])
    pos = rng.normal(scale=1.2, size=(N_ATOMS, 3))
    m3 = np.repeat(masses, 3)

    v_rigid, _s, rank = hess_mod.rigid_body_vectors(masses, pos)
    p = np.eye(3 * N_ATOMS) - v_rigid @ v_rigid.T
    # An orthonormal basis of the vibrational subspace.
    q, _r = np.linalg.qr(p @ rng.normal(size=(3 * N_ATOMS, 3 * N_ATOMS)))
    keep = np.linalg.norm(v_rigid.T @ q, axis=0) ** 2 <= 0.5
    basis = q[:, keep][:, :3 * N_ATOMS - rank]

    # cm^-1: a spread of well separated modes plus a FOUR-FOLD near-degenerate block.
    freqs = np.array([120.0, 340.0, 610.0, 880.0, 1150.0, 1420.0,
                      1500.0, 1502.0, 1504.0, 1506.0,          # the degenerate block
                      1900.0, 3010.0], dtype=float)
    assert basis.shape[1] == len(freqs), (basis.shape, len(freqs))
    lam = (freqs / hess_mod.CM_INV_PER_SQRT_EV_A2_AMU) ** 2
    hm = basis @ np.diag(lam) @ basis.T
    h = hm * np.sqrt(np.outer(m3, m3))
    return masses, pos, 0.5 * (h + h.T), freqs


def main():
    masses, pos, h, freqs_in = synthetic()

    omega, vec = mode_match.projected_modes(h, masses, pos, "hessian")
    ref = hess_mod.project_and_diagonalise(h, masses, pos)
    agree = float(np.abs(np.sort(omega)
                         - np.sort(ref["frequencies_cm_inv"])).max())
    recovered = float(np.abs(np.sort(omega) - np.sort(freqs_in)).max())

    # A second mode set: the same modes, but with the degenerate block deliberately
    # re-mixed by an arbitrary rotation -- exactly what a covariance eigensolver does
    # to a degenerate subspace, and what the per-mode gate cannot survive.
    blocks = mode_match.degenerate_blocks(omega, 20.0)
    big = max(blocks, key=len)
    rng = np.random.RandomState(SEED + 1)
    rot, _ = np.linalg.qr(rng.normal(size=(len(big), len(big))))
    mixed = vec.copy()
    mixed[:, big] = vec[:, big] @ rot

    m = mode_match.match(omega, mixed, omega, vec)
    in_block = [i for i in big]
    per_mode_in_block = min(m["max_overlap"][i] for i in in_block)
    block_in_block = min(m["block_overlap"][i] for i in in_block)
    per_mode_outside = min(m["max_overlap"][i] for i in range(len(omega))
                           if i not in in_block)

    # The hybrid: with nu == omega, any cut must leave G untouched, and the count of
    # modes taken from QHA must equal the count below the cut.
    spec, rec = mode_match.hybrid_spectrum(omega, omega, m, 1000.0)
    n_below = int((np.asarray(omega) < 1000.0).sum())
    hybrid_identical = float(np.abs(spec - omega).max())

    # `thermodynamic_terms` against `thermo`, directly.
    t = mode_match.thermodynamic_terms(omega)
    zpe_direct = 0.5 * thermo.HC_KCAL * np.asarray(omega).sum()
    ts_direct = sum(thermo.s_mode_kcal_per_K(float(x)) for x in omega) * thermo.T_REF

    checks = [
        ("mode_match.projected_modes and hessian.project_and_diagonalise are the same "
         "computation",
         "max difference {:.3e} cm^-1".format(agree), agree < 1e-9),
        ("the projection recovers the spectrum that was put in",
         "max difference {:.3e} cm^-1".format(recovered), recovered < 1e-6),
        ("the degenerate block was found",
         "largest block has {} members at {}".format(
             len(big), np.round(np.asarray(omega)[big], 1).tolist()), len(big) == 4),
        ("inside a re-mixed degenerate block, PER-MODE overlap collapses",
         "min per-mode overlap in block {:.3f}".format(per_mode_in_block),
         per_mode_in_block < 0.7),
        ("...while BLOCK overlap does not -- this is why the gate is on the block",
         "min block overlap in block {:.3f}".format(block_in_block),
         block_in_block > 0.99),
        ("outside the block the per-mode pairing is unaffected",
         "min per-mode overlap outside {:.3f}".format(per_mode_outside),
         per_mode_outside > 0.99),
        ("hybrid_spectrum takes exactly the modes below the cut from QHA",
         "{} taken, {} are below 1000 cm^-1".format(rec["n_from_qha"], n_below),
         rec["n_from_qha"] == n_below),
        ("with nu == omega the hybrid spectrum is the Hessian spectrum",
         "max difference {:.3e} cm^-1".format(hybrid_identical),
         hybrid_identical < 1e-12),
        ("thermodynamic_terms reproduces thermo's ZPE and T*S",
         "dZPE {:.3e}, dT*S {:.3e} kcal/mol".format(
             t["ZPE_kcal"] - zpe_direct, t["TS_kcal"] - ts_direct),
         abs(t["ZPE_kcal"] - zpe_direct) < 1e-9
         and abs(t["TS_kcal"] - ts_direct) < 1e-9),
        ("overlap_matrix rows sum to 1 for two complete bases",
         "max deviation {:.3e}".format(m["max_row_sum_deviation"]),
         m["max_row_sum_deviation"] < 1e-9),
    ]

    bad = 0
    for text, measured, ok in checks:
        print("[{}] {}".format("PASS" if ok else "FAIL", text))
        print("       measured: {}".format(measured))
        bad += 0 if ok else 1
    print()
    print("{} of {} checks failed".format(bad, len(checks)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
