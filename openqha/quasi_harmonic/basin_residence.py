"""Did the trajectory stay in the basin it was started in?

WHY THIS EXISTS, AND WHY IT WAS MISSING
---------------------------------------
Quasi-harmonic analysis assumes the sampled configurations are **one multivariate
Gaussian**. Branch A's job is to enumerate the basins; branch B's job is to measure the
entropy *inside* one of them. A trajectory that crosses a torsional barrier samples two
basins, and the covariance then contains the displacement BETWEEN the minima as well as
the fluctuation within them. That inflates every soft eigenvalue and inflates T*S -- and
nothing else in the chain notices.

This is the reason a longer trajectory is not automatically a better one here, and it is
the correction to a mistake made on 2026-09-07: the published protein protocol (1.5 ns)
was adopted wholesale for 10-19 atom molecules. A solvated protein at 300 K barely
explores its basin in 1.5 ns. Acetone crosses its methyl barrier -- about 0.8 kcal/mol
against kT = 0.59 -- constantly.

**The consequence for acceptance criterion 1.** That criterion fails when T*S is still
rising over the last doubling of trajectory length. There are two reasons it can rise:

    the estimator has not converged        -> run LONGER
    the trajectory has started hopping     -> run SHORTER, or restrain, or drop the rotor

They demand opposite actions and look identical in the saturation curve alone. So the
transition count is reported beside it, and `interpret_saturation()` refuses to read one
without the other.

TWO KINDS OF CROSSING, AND ONLY ONE IS A DIFFERENT BASIN
--------------------------------------------------------
A methyl rotor has order n = 3: turning it by 120 degrees gives an **indistinguishable**
structure. That is not a new conformer, and the conformer count and the symmetry number
already account for it.

**But it still breaks the Gaussian assumption**, because the three hydrogens physically
move to new coordinates and the Cartesian covariance sees them sweep a circle. The
symmetry number corrects the ROTATIONAL partition function and the counting of distinct
conformers; it does nothing to the quasi-harmonic covariance. So both kinds are counted,
and they are counted separately:

    symmetry_equivalent_crossings   the rotor turned into a copy of itself.
                                    NOT a new basin. STILL contaminates the covariance.
    distinct_basin_crossings        the wrapped dihedral changed well.
                                    A different conformer. Fatal for this trajectory.

A distinct crossing means the result is not an intra-basin entropy at all. A
symmetry-equivalent crossing means the all-atom number is contaminated -- and is the main
reason a heavy-atom analysis is better conditioned for these molecules, because the atoms
doing the sweeping are exactly the ones it drops.

HOW A WELL IS ASSIGNED
----------------------
For a rotor of order `n` the surface has period 2*pi/n. The dihedral is wrapped into one
period, which makes symmetry-equivalent minima the same point, and the well index is then
which of the `n` sectors of the FULL circle the raw angle sits in.

Crossings are counted with **hysteresis**: a crossing is recorded only once the angle has
moved past the new well centre by `commit_fraction` of the half-width. Without it, an
angle sitting on a barrier is counted as hopping every few frames and the count says more
about the frame spacing than about the dynamics.
"""
import numpy as np

from . import torsion_cv

#: How far past the boundary an angle must go before a crossing is committed, as a
#: fraction of the half-width of a well. 0.25 is a convention: large enough to ignore
#: barrier-top rattling, small enough not to miss a genuine hop.
COMMIT_FRACTION = 0.25

#: Elements treated as heavy. The source paper superimposed and analysed non-hydrogen
#: atoms; `heavy_atom_mask` is how that choice is expressed here.
HEAVY_ELEMENTS = ("C", "N", "O", "F", "S", "P", "Cl", "Br")


def heavy_atom_mask(symbols):
    """Boolean mask selecting the non-hydrogen atoms, in input order."""
    return np.array([str(s) not in ("H", "D") for s in symbols], dtype=bool)


def dihedral_series(frames, atoms):
    """The dihedral of one CV over every frame, in radians, wrapped to (-pi, pi]."""
    x = np.asarray(frames, dtype=float)
    return np.array([torsion_cv.dihedral(f, atoms) for f in x], dtype=float)


def _unwrap_to_period(angles, order):
    """Fold an angle series into ONE period of a rotor of the given order.

    For order n the physical surface repeats every 2*pi/n, so this is the coordinate in
    which two symmetry-equivalent minima are the same point. A change of well HERE is a
    change of conformer; a change of sector in the raw angle need not be.
    """
    n = max(1, int(order))
    period = 2.0 * np.pi / n
    return np.mod(np.asarray(angles, dtype=float), period)


def _count_crossings(angles, period, commit_fraction=COMMIT_FRACTION):
    """Committed well-to-well crossings of a periodic coordinate.

    Returns (n_crossings, occupancy) where occupancy counts frames per well index.
    """
    a = np.asarray(angles, dtype=float)
    if len(a) == 0:
        return 0, {}
    half = period / 2.0
    well = np.floor(a / period + 0.5).astype(int)
    current = int(well[0])
    crossings = 0
    occupancy = {}
    for k, ang in zip(well, a):
        occupancy[int(k)] = occupancy.get(int(k), 0) + 1
        if int(k) == current:
            continue
        # Committed only once the angle is well inside the new well, not merely across
        # the boundary. An angle rattling on a barrier top otherwise counts as a hop on
        # every frame, and the number then describes the frame spacing, not the dynamics.
        centre = int(k) * period
        if abs(ang - centre) < (1.0 - commit_fraction) * half:
            crossings += 1
            current = int(k)
    return crossings, occupancy


def basin_residence(frames, symbols, commit_fraction=COMMIT_FRACTION):
    """Whether this trajectory stayed in one basin, and what moved if it did not.

    Returns a record fit to be stored beside the entropy, because an intra-basin entropy
    computed on a trajectory that left the basin is not a wrong number -- it is a number
    for a different quantity, and only this record says which one you have.
    """
    x = np.asarray(frames, dtype=float)
    symbols = list(symbols)
    if x.ndim != 3:
        raise ValueError("frames must be (n_frames, n_atoms, 3); got {}".format(x.shape))

    # Returns (cvs, diagnostics) -- the second half says which bonds were skipped and
    # why, which is worth keeping: "no torsions found" and "every torsion was a ring
    # bond" are different situations and only the diagnostics separates them.
    cvs, cv_diag = torsion_cv.torsion_collective_variables(symbols, x[0])
    rotors, distinct, equivalent = [], 0, 0
    for cv in cvs:
        atoms = cv["atoms"]
        # `periodicity`, not `order`: the CV record carries the rotor order of each end
        # separately and `periodicity` is max(oi, oj, 1), which is the period of the
        # surface along this axis.
        order = int(cv.get("periodicity", 1))
        raw = dihedral_series(x, atoms)

        # Distinct conformers: in the folded coordinate the symmetry copies coincide.
        folded = _unwrap_to_period(raw, order)
        n_distinct, occ_distinct = _count_crossings(
            folded, 2.0 * np.pi / max(1, order), commit_fraction)
        # Every crossing of the FULL circle, including turning a rotor into its own copy.
        n_all, occ_all = _count_crossings(
            np.mod(raw, 2.0 * np.pi), 2.0 * np.pi / max(1, order), commit_fraction)

        n_equivalent = max(0, n_all - n_distinct)
        distinct += n_distinct
        equivalent += n_equivalent
        rotors.append(dict(
            atoms=[int(i) for i in atoms], periodicity=order,
            kind=cv.get("kind"), elements=cv.get("elements"),
            distinct_basin_crossings=int(n_distinct),
            symmetry_equivalent_crossings=int(n_equivalent),
            wells_visited=len(occ_all),
            angle_std_deg=float(np.degrees(np.std(folded))),
            angle_range_deg=float(np.degrees(folded.max() - folded.min())),
        ))

    return dict(
        n_frames=int(len(x)),
        n_torsional_cvs=len(cvs),
        cv_diagnostics=cv_diag,
        distinct_basin_crossings=int(distinct),
        symmetry_equivalent_crossings=int(equivalent),
        stayed_in_one_basin=bool(distinct == 0),
        covariance_is_single_gaussian=bool(distinct == 0 and equivalent == 0),
        rotors=rotors,
        commit_fraction=float(commit_fraction),
        note=(
            "distinct_basin_crossings > 0: this is NOT an intra-basin entropy. The "
            "covariance contains the displacement between two minima and T*S is "
            "inflated. | symmetry_equivalent_crossings > 0: the end state is "
            "indistinguishable and the conformer count is unaffected, but the atoms "
            "moved, so the CARTESIAN covariance is still inflated. A heavy-atom "
            "analysis drops exactly the atoms that do this."),
    )


def interpret_saturation(saturation, residence, budget_kcal=0.3):
    """Read acceptance criterion 1 together with the transition count.

    The criterion fails when T*S is still rising over the last doubling. On its own that
    is ambiguous, and the two readings ask for OPPOSITE actions:

        rising, no crossings      the estimator has not converged   -> run LONGER
        rising, crossings         the trajectory is leaving a basin -> run SHORTER

    Returning "fail" without saying which of those it is invites the wrong fix, and the
    wrong fix here makes the number worse while making the criterion look satisfied.
    """
    rise = float(saturation.get("increment_over_last_doubling_kcal", float("nan")))
    passed = bool(abs(rise) <= budget_kcal)
    distinct = int(residence.get("distinct_basin_crossings", 0))
    equivalent = int(residence.get("symmetry_equivalent_crossings", 0))

    if passed and distinct == 0:
        verdict, action = "converged intra-basin", "none"
    elif passed and distinct > 0:
        verdict = "flat BUT the trajectory changed basin"
        action = ("Do not report this. A flat saturation curve on a multi-basin "
                  "trajectory means the inflated covariance has itself converged; the "
                  "quantity is not an intra-basin entropy. Shorten, or restrain the "
                  "rotor that moved.")
    elif not passed and distinct > 0:
        verdict = "rising because the trajectory is leaving the basin"
        action = "Run SHORTER, or restrain the rotor. Running longer makes it worse."
    else:
        verdict = "rising, and no basin change -- genuinely unconverged"
        action = "Run LONGER."

    return dict(
        increment_over_last_doubling_kcal=rise,
        budget_kcal=float(budget_kcal),
        criterion_1_passed=passed,
        distinct_basin_crossings=distinct,
        symmetry_equivalent_crossings=equivalent,
        verdict=verdict,
        action=action,
        cartesian_covariance_contaminated=bool(distinct or equivalent),
    )
