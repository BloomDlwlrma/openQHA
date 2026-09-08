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
The raw dihedral is divided into the `n` wells the rotor has -- `n` from its periodicity
for a symmetric rotor, three for a general torsion (a rotatable single bond has three
staggered minima; `rotor_order` describes the symmetry of the ends, not the number of
minima). Crossings are counted once, on that coordinate, and are then CLASSIFIED by what
the rotor is, using `torsion_cv`'s own `kind`.

An earlier version counted a second time on an angle folded into a single period. That
was wrong in the worst way: folding leaves a coordinate spanning exactly one period, and
the well rule then split that single well in half at an arbitrary midpoint, so every
thermal wiggle across the midpoint counted as a conformer change. It reported 33
"distinct" crossings in 40 frames of acetone -- whose true count is zero, since a methyl
turn maps the molecule onto itself -- and because the equivalent count was derived by
subtraction, the spurious number also erased the real one.

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


#: Wells assumed for a general torsion whose reported periodicity is 1.
#:
#: `rotor_order` describes the SYMMETRY of each end of the bond, not the number of minima
#: on the potential. A rotatable single bond has three staggered minima whatever the end
#: groups look like, so a periodicity of 1 means "no symmetry", not "one well". The
#: number is a convention and is carried in the record so that a molecule where it is
#: wrong can be seen rather than assumed.
DEFAULT_WELLS_GENERAL_TORSION = 3


def wells_for(cv):
    """How many minima this torsion has, and whether they are copies of one another.

    Returns (n_wells, symmetric). `symmetric` is read from `torsion_cv`'s own `kind`
    rather than re-derived: a methyl-type rotor maps the molecule onto itself every
    2*pi/n, so a crossing changes no conformer.
    """
    periodicity = int(cv.get("periodicity", 1) or 1)
    symmetric = "symmetric rotor" in str(cv.get("kind", ""))
    if symmetric and periodicity > 1:
        return periodicity, True
    return max(periodicity, DEFAULT_WELLS_GENERAL_TORSION), False


def _count_crossings(angles, period, commit_fraction=COMMIT_FRACTION, n_wells=None):
    """Committed well-to-well crossings of a periodic coordinate.

    Returns (n_crossings, occupancy) where occupancy counts frames per well index.

    The index is taken MODULO the well count. Without that, `floor(a/period + 0.5)` on an
    angle wrapped to [0, 2*pi) returns 0..n rather than 0..n-1 -- the top half of the last
    well rounds up to index n, which is index 0 again. A 3-fold rotor then reports four
    wells, and every pass through the wrap point counts as a crossing although nothing
    moved.
    """
    a = np.asarray(angles, dtype=float)
    if len(a) == 0:
        return 0, {}
    half = period / 2.0
    n_wells = int(n_wells or max(1, round(2.0 * np.pi / period)))
    well = np.floor(a / period + 0.5).astype(int) % n_wells
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
        # On the circle: the distance from an angle to a well centre must be measured
        # the short way round, or well 0 looks 2*pi away from an angle just below 2*pi.
        span = 2.0 * np.pi if abs(period * n_wells - 2.0 * np.pi) < 1e-9 else None
        d = abs(ang - centre)
        if span is not None:
            d = min(d, span - d)
        if d < (1.0 - commit_fraction) * half:
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
        n_wells, symmetric = wells_for(cv)
        raw = dihedral_series(x, atoms)

        # ONE count, on the raw angle, over the wells the rotor actually has. The earlier
        # version counted a second time on an angle folded into a single period, where
        # `floor(a/period + 0.5)` splits that one well in half at an arbitrary midpoint --
        # so every thermal wiggle across the midpoint was counted as a conformer change.
        # It reported 33 "distinct" crossings in 40 frames for acetone, whose true count
        # is zero.
        n_crossings, occupancy = _count_crossings(
            np.mod(raw, 2.0 * np.pi), 2.0 * np.pi / n_wells, commit_fraction,
            n_wells=n_wells)

        # WHAT the crossing was is a property of the rotor, not of the trajectory.
        if symmetric:
            equivalent += n_crossings
            n_distinct, n_equivalent = 0, n_crossings
        else:
            distinct += n_crossings
            n_distinct, n_equivalent = n_crossings, 0

        rotors.append(dict(
            atoms=[int(i) for i in atoms], periodicity=int(cv.get("periodicity", 1) or 1),
            n_wells=int(n_wells), symmetric=bool(symmetric),
            kind=cv.get("kind"), elements=cv.get("elements"),
            distinct_basin_crossings=int(n_distinct),
            symmetry_equivalent_crossings=int(n_equivalent),
            wells_visited=len(occupancy),
            angle_std_deg=float(np.degrees(np.std(raw))),
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
