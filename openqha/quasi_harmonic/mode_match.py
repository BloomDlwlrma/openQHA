"""Pairing quasi-harmonic modes with Hessian modes, and the hybrid spectrum built on it.

THE QUESTION THIS MODULE ANSWERS
--------------------------------
Quasi-harmonic analysis reads a frequency off a variance, `nu = sqrt(k_B T / lambda)`;
a Hessian reads one off a curvature, `omega = sqrt(k / mu)`. In a harmonic well the two
are the same number. The whole of branch B rests on that, and so does any scheme that
uses `nu` for some modes and `omega` for others:

    S_i^final = S_i^QHA(nu_pi(i))   if nu_pi(i) < nu_cut and the mode is well paired
                S_i^Hessian(omega_i) otherwise

`pi` is a pairing between two eigenbases that are NOT the same basis, so it has to be
computed, and how well it worked has to be reported rather than assumed.

WHY THE PER-MODE OVERLAP GATE IS THE WRONG GATE (MEASURED 2026-09-08)
---------------------------------------------------------------------
The obvious gate is `max_j O_ij >= 0.7`. On acetone's own MACE Hessian, with a synthetic
harmonic trajectory of 50 000 frames -- a surface that IS harmonic, where the identity
holds exactly -- it rejects modes that are reproduced to a fraction of a wavenumber:

    mode   nu        omega     max_j O_ij   block O
      13   1455.60   1455.49     0.774       0.996
      14   1460.09   1459.89     0.427       0.996
      15   1468.79   1459.89     0.499       0.993

Acetone's two methyl groups make a four-fold near-degenerate block at 1455-1481 cm^-1.
**Inside a degenerate block the eigenvectors are an arbitrary rotation**, so the pairing
between individual members is meaningless and its overlap is small for reasons that have
nothing to do with the quasi-harmonic approximation. The quantity that is invariant
under that rotation is the overlap with the whole BLOCK, and it stays above 0.88 for
every mode at every block threshold from 5 to 50 cm^-1.

So the gate here is the block overlap. The per-mode value is still reported, because it
is what a reader expects to see and because its collapse is itself the diagnostic that
says "these modes are degenerate", but it is not what decides anything.

UNITS
-----
Frequencies cm^-1, masses amu, positions angstrom. Eigenvectors are columns in
MASS-WEIGHTED cartesian coordinates -- which is what makes the two sets comparable at
all, since both are orthonormal bases of the same 3N space after the same Eckart
projection.
"""
import numpy as np

from ..thermochem import hessian as _hessian
from ..thermochem import thermo

#: Two modes closer than this are treated as degenerate for the purposes of pairing.
#: Measured above: the verdict does not move between 5 and 50 cm^-1, so the value is
#: not delicate -- which is the only reason a default is acceptable here.
DEFAULT_DEGENERACY_GAP_CM = 20.0

#: Below this block overlap a pairing is refused, and the mode falls back to its Hessian
#: value. A convention, not a measurement, and the record always carries the number so a
#: different choice can be re-decided from the product.
DEFAULT_MIN_BLOCK_OVERLAP = 0.7


def projected_modes(matrix, masses, positions, kind, temperature_K=thermo.T_REF):
    """Eckart-project, diagonalise, and return `(frequencies_cm_inv, eigenvectors)`.

    `kind` is `"hessian"` (matrix in eV/A^2, mass-weighted here) or `"covariance"`
    (matrix already mass-weighted, amu*A^2, as `qha.mass_weighted_covariance` returns).

    The rigid subspace comes from `hessian.rigid_body_vectors`, the same function
    `hessian.project_and_diagonalise` and `qha.spectrum` use, so the three cannot
    disagree about what "rigid" means. A rigid mode is identified by its overlap with
    that subspace, never by "take the six smallest".

    The frequencies this returns are asserted against `project_and_diagonalise` by
    `tests/unit/t_mode_match.py`; the eigenvectors are the only thing that is new here.
    """
    if kind not in ("hessian", "covariance"):
        raise ValueError("kind must be 'hessian' or 'covariance', received {!r}"
                         .format(kind))
    m = np.asarray(masses, dtype=float)
    m3 = np.repeat(m, 3)
    a = np.asarray(matrix, dtype=float)
    if kind == "hessian":
        a = a / np.sqrt(np.outer(m3, m3))
    v, _sing, _rank = _hessian.rigid_body_vectors(m, positions)
    p = np.eye(len(m3)) - v @ v.T
    ap = p @ a @ p
    ap = 0.5 * (ap + ap.T)
    lam, vec = np.linalg.eigh(ap)

    keep = np.linalg.norm(v.T @ vec, axis=0) ** 2 <= 0.5
    lam, vec = lam[keep], vec[:, keep]
    if kind == "hessian":
        freq = _hessian.eigenvalues_to_cm_inv(lam)
    else:
        from . import qha                                    # local: avoids a cycle
        freq = qha.frequencies_cm_inv(lam, temperature_K)
    order = np.argsort(np.where(np.isfinite(freq), freq, np.inf))
    return np.asarray(freq)[order], vec[:, order]


def overlap_matrix(vectors_a, vectors_b):
    """`O[i, j] = (a_i . b_j)^2`, the squared overlap of two orthonormal mode sets.

    Squared, so each row and column sums to 1 when both sets span the same space --
    which makes a row that does not sum to 1 a visible sign that they do not.
    """
    a = np.asarray(vectors_a, dtype=float)
    b = np.asarray(vectors_b, dtype=float)
    if a.shape[0] != b.shape[0]:
        raise ValueError("mode sets live in different spaces: {} vs {} coordinates"
                         .format(a.shape[0], b.shape[0]))
    return (a.T @ b) ** 2


def degenerate_blocks(frequencies, gap_cm=DEFAULT_DEGENERACY_GAP_CM):
    """Split an ascending frequency list into runs separated by more than `gap_cm`."""
    f = np.asarray(frequencies, dtype=float)
    if np.any(np.diff(f) < -1e-9):
        raise ValueError("frequencies must be ascending; got a decrease of {:.4f}"
                         .format(float(np.diff(f).min())))
    blocks, current = [], [0]
    for i in range(1, len(f)):
        if f[i] - f[i - 1] <= gap_cm:
            current.append(i)
        else:
            blocks.append(current)
            current = [i]
    blocks.append(current)
    return blocks


def match(nu, vectors_nu, omega, vectors_omega,
          gap_cm=DEFAULT_DEGENERACY_GAP_CM):
    """Pair each quasi-harmonic mode with a Hessian mode, and say how well it worked.

    Returns a record holding, per quasi-harmonic mode `i`:

        pairing[i]            index of the Hessian mode of largest overlap
        max_overlap[i]        that overlap -- REPORTED, never used as a gate
        block_overlap[i]      overlap with the whole near-degenerate block around it,
                              which IS the gate (see the module docstring)
        delta_cm[i]           nu_i - omega_pairing[i]

    Duplicate pairings are counted rather than resolved. A greedy or Hungarian
    assignment would produce a permutation whatever the data looked like, and the
    number of collisions is the honest signal that the two bases have stopped
    corresponding.
    """
    nu = np.asarray(nu, dtype=float)
    om = np.asarray(omega, dtype=float)
    o = overlap_matrix(vectors_nu, vectors_omega)
    pairing = o.argmax(axis=1)
    max_ov = o.max(axis=1)

    blocks = degenerate_blocks(om, gap_cm)
    block_of = {}
    for k, blk in enumerate(blocks):
        for j in blk:
            block_of[j] = k
    block_ov = np.array([o[i, blocks[block_of[int(pairing[i])]]].sum()
                         for i in range(len(nu))])

    delta = np.array([nu[i] - om[int(pairing[i])] for i in range(len(nu))])
    n_dup = int(len(pairing) - len(set(pairing.tolist())))
    row_sums = o.sum(axis=1)

    bands = {}
    for lo, hi, name in ((0.0, 500.0, "below_500"), (500.0, 1500.0, "500_to_1500"),
                         (1500.0, np.inf, "above_1500")):
        sel = [i for i in range(len(nu)) if lo <= om[int(pairing[i])] < hi]
        if sel:
            d = delta[sel]
            bands[name] = dict(n=len(sel), mean_cm=float(d.mean()),
                               rms_cm=float(np.sqrt((d ** 2).mean())),
                               max_abs_cm=float(np.abs(d).max()))
    return dict(
        n_modes=int(len(nu)),
        degeneracy_gap_cm=float(gap_cm),
        n_blocks=int(len(blocks)),
        block_sizes=[len(b) for b in blocks],
        pairing=[int(x) for x in pairing],
        max_overlap=[float(x) for x in max_ov],
        block_overlap=[float(x) for x in block_ov],
        delta_cm=[float(x) for x in delta],
        min_max_overlap=float(max_ov.min()),
        min_block_overlap=float(block_ov.min()),
        n_duplicate_pairings=n_dup,
        max_row_sum_deviation=float(np.abs(row_sums - 1.0).max()),
        signed_deviation_by_band=bands,
        mean_signed_deviation_cm=float(delta.mean()),
        rms_deviation_cm=float(np.sqrt((delta ** 2).mean())),
    )


def hybrid_spectrum(nu, omega, match_record, nu_cut_cm,
                    min_block_overlap=DEFAULT_MIN_BLOCK_OVERLAP):
    """The spectrum used for the entropy, one frequency per Hessian mode.

    Implements

        S_i^final = S_i^QHA(nu_pi(i))    if nu_pi(i) < nu_cut and block overlap is sound
                    S_i^Hessian(omega_i) otherwise

    as a frequency substitution rather than as two entropy formulas, so that ZPE,
    enthalpy and entropy can all be evaluated from one array and cannot silently come
    from different spectra.

    Returned indexed by HESSIAN mode, because that is what the Hessian-side terms are
    indexed by. A Hessian mode that no quasi-harmonic mode paired to keeps `omega`, and
    is counted in `n_unclaimed`.
    """
    nu = np.asarray(nu, dtype=float)
    om = np.asarray(omega, dtype=float)
    pairing = np.asarray(match_record["pairing"], dtype=int)
    block_ov = np.asarray(match_record["block_overlap"], dtype=float)

    out = om.copy()
    source = ["hessian"] * len(om)
    claimed = np.zeros(len(om), dtype=bool)
    n_rejected_overlap = 0
    for i in range(len(nu)):
        j = int(pairing[i])
        if claimed[j]:
            continue                       # a duplicate pairing: first one wins, and
                                           # `n_duplicate_pairings` already reports it
        if not np.isfinite(nu[i]) or nu[i] <= 0:
            continue
        if nu[i] >= nu_cut_cm:
            continue
        if block_ov[i] < min_block_overlap:
            n_rejected_overlap += 1
            continue
        out[j] = nu[i]
        source[j] = "qha"
        claimed[j] = True
    return out, dict(
        nu_cut_cm=float(nu_cut_cm),
        min_block_overlap=float(min_block_overlap),
        n_from_qha=int(sum(1 for s in source if s == "qha")),
        n_from_hessian=int(sum(1 for s in source if s == "hessian")),
        n_rejected_by_overlap=int(n_rejected_overlap),
        n_unclaimed=int((~claimed).sum()),
        source=source,
    )


def thermodynamic_terms(frequencies_cm, temperature_K=thermo.T_REF):
    """ZPE, vibrational internal energy, T*S and A_vib from one spectrum, kcal/mol.

    One function, one spectrum, four numbers -- so that "which frequencies went into
    the ZPE" can never differ from "which went into the entropy".

    A non-positive or non-finite frequency is dropped and counted, never made positive:
    an imaginary mode is information.
    """
    f = np.asarray(frequencies_cm, dtype=float)
    ok = np.isfinite(f) & (f > 0.0)
    good = f[ok]
    return dict(
        n_modes=int(ok.sum()),
        n_dropped=int((~ok).sum()),
        ZPE_kcal=float(0.5 * thermo.HC_KCAL * good.sum()),
        E_vib_kcal=float(sum(thermo.e_mode_kcal(float(x), temperature_K)
                             for x in good)),
        TS_kcal=float(sum(thermo.s_mode_kcal_per_K(float(x), temperature_K)
                          for x in good) * temperature_K),
        A_vib_kcal=float(sum(thermo.a_mode_kcal(float(x), temperature_K)
                             for x in good)),
        lowest_cm=float(good.min()) if good.size else float("nan"),
        highest_cm=float(good.max()) if good.size else float("nan"),
    )
