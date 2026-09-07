"""External rotational symmetry number sigma, derived from the 3D GEOMETRY.

Branch A, plan_A section 2.5. This module unblocks every free-energy label in
branches A and C: sigma enters G_rot as +RT ln(sigma), and D0-9 refuses to run
without it, so the 69 package-1 molecules that never had a declared sigma could
not be labelled at all.

--------------------------------------------------------------------------------
What D0-9 forbids, and why this module does not do it
--------------------------------------------------------------------------------
D0-9 forbids taking sigma from the MOLECULAR GRAPH:

    sigma = len(mol.GetSubstructMatches(mol, uniquify=False))    # FORBIDDEN

because graph automorphism counts internal-rotor and H-permutation symmetry,
which are not external rotational symmetry. Ethane's graph automorphism count is
3 x 3 x 2 = 18 while its external sigma is 6 -- a factor 3, worth
kT ln 3 = 0.651 kcal/mol, the same order as the 1.0 kcal/mol target, and it does
not cancel between isomers.

This module uses graph automorphisms **only as candidate atom permutations** and
then asks a question the graph cannot answer:

    does this permutation correspond to a RIGID-BODY motion of this geometry?

A permutation is accepted only if the best rigid superposition of the geometry
onto its permuted copy has RMSD below a tolerance in Angstrom. An internal rotor
fails that test -- rotating one methyl of ethane while holding the other still
moves hydrogens by ~1.8 A, and no rigid rotation puts them back. The whole-
molecule C3 passes. So the count that comes out is the order of the geometry's
proper rotation group, which is exactly sigma.

    sigma = number of accepted permutations whose optimal orthogonal
            transform has det = +1   (identity included)

Improper operations (det = -1: mirrors, S_n) are counted separately. They belong
to the point group but NOT to sigma.

--------------------------------------------------------------------------------
Why not just call pymsym
--------------------------------------------------------------------------------
pymsym (libmsym binding, MIT) detects the point group from coordinates and passed
our seven-species back-test 7/7 on 2026-09-03, including acetone, which ORCA gets
wrong (D0-P2-15). But it only works on IDEAL geometries: measured the same day,
a random displacement of **0.005 A** drops acetone and oxetane from C2v to C1.
Our basins are numerically optimised structures whose deviation from ideal
symmetry is exactly that size, so a bare pymsym call would report sigma = 1 for
all of them -- low by a factor 2, i.e. G low by RT ln 2 = 0.411 kcal/mol, the
same error and the same sign as ORCA's.

2026-09-03, this session, correcting plan_A section 2.5.6: the claim "pymsym 0.3.5
exposes no tolerance parameter" is true of the high-level `get_symmetry_number`
but NOT of the library. `pymsym.Context.set_thresholds()` exposes all seven
libmsym thresholds (zero, geometry, angle, equivalence, eigfact, permutation,
orthogonalization; defaults 0.001/0.001/0.001/0.0005/0.001/0.005/0.01).

**They were measured and they do not help.** On acetone displaced by 0.02 A,
every one of the seven, swept over 0.005 - 0.3 individually and all together,
still leaves libmsym raising "Error determining point group". `symmetrize_elements`
cannot rescue it either: it needs a point group first, and reports
"Point group has no primary axis for reorientation". So the plan's conclusion
stands, with a better reason: the knob exists, and turning it changes nothing.

pymsym is therefore kept as an INDEPENDENT CROSS-CHECK on ideal geometries and
as the source of the point-group LABEL, never as the production path.

--------------------------------------------------------------------------------
Parameter classification
--------------------------------------------------------------------------------
`TOLERANCE_A`        modifiable_convention -- swept by `analyse`, and the sweep
                     is part of every product record.
`TOLERANCE_SCAN_A`   modifiable_convention.
"""
import itertools

import numpy as np

from . import conformers

#: Superposition RMSD (Angstrom) below which a candidate permutation counts as a
#: real symmetry operation of this geometry. modifiable_convention.
#:
#: This is the knob libmsym would not give us, and it is in a unit we can reason
#: about: it is "how far this structure may sit from ideal symmetry". Optimised
#: basins deviate by 1e-3 to 1e-2 A, while the nearest wrong answer -- accepting a
#: permutation that is not a symmetry -- needs a whole atom to move. The default
#: sits in that gap. It is NOT a fitted number; every record carries the sweep it
#: was chosen from, plus the margin between the worst operation kept and the best
#: one rejected.
TOLERANCE_A = 0.10

#: Tolerances reported in every product record, so the plateau is visible and a
#: reader can see how far this geometry sits from a flip. modifiable_convention.
TOLERANCE_SCAN_A = (0.01, 0.02, 0.05, 0.10, 0.20, 0.40)

#: Above this many candidate permutations, refuse rather than grind. A molecule
#: this symmetric is not a QM9 organic and deserves to be looked at by hand.
MAX_CANDIDATES = 200000


# ======================================================================================
# 1. Candidate permutations: automorphisms of the connectivity graph
# ======================================================================================
def _refine_colours(numbers, adjacency):
    """1-D Weisfeiler-Lehman refinement. Returns an integer colour per atom.

    Purely a SPEED device: it shrinks the backtracking search by never trying to
    map atoms whose local environments differ. It cannot change the answer,
    because two atoms with different refined colours cannot be exchanged by any
    automorphism.
    """
    colours = np.asarray(numbers, dtype=int).copy()
    for _ in range(len(numbers)):
        sigs = []
        for i in range(len(numbers)):
            nb = sorted(int(colours[j]) for j in np.nonzero(adjacency[i])[0])
            sigs.append((int(colours[i]), tuple(nb)))
        uniq = {s: k for k, s in enumerate(sorted(set(sigs)))}
        new = np.array([uniq[s] for s in sigs], dtype=int)
        if np.array_equal(new, colours):
            break
        colours = new
    return colours


def graph_automorphisms(numbers, adjacency, limit=MAX_CANDIDATES):
    """All automorphisms of the element-labelled connectivity graph.

    Returned as a list of permutation arrays `p` with the meaning
    "atom i takes the place of atom p[i]".

    **This list is NOT sigma.** It is the candidate pool that the geometric test
    in `symmetry_operations` then filters. Reporting its length as a symmetry
    number is precisely what D0-9 forbids; `analyse` reports it under the name
    `n_graph_automorphisms` next to sigma so the gap between the two is visible
    in every product.
    """
    n = len(numbers)
    colours = _refine_colours(numbers, adjacency)
    by_colour = {}
    for i, c in enumerate(colours):
        by_colour.setdefault(int(c), []).append(i)

    # Search the most constrained atom first: fewest candidate images, then
    # highest degree. Only affects speed.
    order = sorted(range(n), key=lambda i: (len(by_colour[int(colours[i])]),
                                            -int(np.asarray(adjacency[i]).sum())))
    adj = np.asarray(adjacency, dtype=bool)

    out = []
    mapping = [-1] * n
    used = [False] * n

    def backtrack(k):
        if len(out) > limit:
            raise RuntimeError(
                "more than {} graph automorphisms -- refusing to enumerate. "
                "Look at this molecule by hand.".format(limit))
        if k == len(order):
            out.append(np.array(mapping, dtype=int))
            return
        i = order[k]
        for j in by_colour[int(colours[i])]:
            if used[j]:
                continue
            ok = True
            for m in range(k):
                a = order[m]
                if adj[i, a] != adj[j, mapping[a]]:
                    ok = False
                    break
            if not ok:
                continue
            mapping[i] = j
            used[j] = True
            backtrack(k + 1)
            used[j] = False
            mapping[i] = -1

    backtrack(0)
    return out


# ======================================================================================
# 2. The geometric test: is this permutation a rigid-body motion?
# ======================================================================================
def _best_fits(a, b, masses):
    """Best PROPER and best IMPROPER superposition of `a` onto `b`.

    Both mass-centred; mass weighting so the answer is about the rigid body, not
    about where the hydrogens went. Returns (rmsd_proper, rmsd_improper).

    Why BOTH determinants are solved for separately, rather than taking the
    unconstrained optimum and reading off its sign
    -----------------------------------------------------------------------
    The unconstrained optimum is not unique when the geometry is planar or
    linear: the third singular value of the correlation matrix is zero, so the
    sign of the determinant is arbitrary and numpy's SVD picks one. Measured
    2026-09-03, first version of this module: water came out sigma = 1 instead
    of 2, benzene 2 instead of 12, methane 4 instead of 12 -- every planar or
    highly symmetric case, and only those. The seven production species passed
    anyway, so the seven species alone would NOT have caught this. That is the
    whole reason plan_A section 2.5.3 demands cases the criterion should fail on.

    The physical statement is also cleaner this way. A permutation of a planar
    molecule is realised BOTH by an in-plane C2 and by the perpendicular mirror;
    those are two different operations of the point group that move atoms the
    same way. sigma counts proper rotations, so the question to ask of each
    permutation is "does a PROPER rotation realise it", not "which fit happened
    to win".
    """
    w = np.asarray(masses, dtype=float)
    m = (w[:, None] * a).T @ b                     # sum_i w_i a_i b_i^T
    u, _s, vt = np.linalg.svd(m)
    base = vt.T @ u.T                              # unconstrained argmax
    sign = 1.0 if np.linalg.det(base) > 0 else -1.0
    out = []
    for want in (+1.0, -1.0):
        d = np.diag([1.0, 1.0, sign * want])
        r = vt.T @ d @ u.T
        resid = b - a @ r.T
        out.append(float(np.sqrt((w[:, None] * resid ** 2).sum() / w.sum())))
    return out[0], out[1]


def _mass_centre(positions, masses):
    m = np.asarray(masses, dtype=float)
    return (m[:, None] * np.asarray(positions, float)).sum(0) / m.sum()


def default_masses(numbers):
    from ase.data import atomic_masses
    return np.array([atomic_masses[int(z)] for z in numbers], dtype=float)


def symmetry_operations(numbers, positions, masses=None, adjacency=None,
                        tolerance_A=TOLERANCE_A, candidates=None):
    """Accepted symmetry operations of THIS geometry, split proper / improper.

    Returns a dict. `sigma` is the count of accepted PROPER operations, which is
    the external rotational symmetry number.
    """
    positions = np.asarray(positions, dtype=float)
    numbers = np.asarray(numbers, dtype=int)
    if masses is None:
        masses = default_masses(numbers)
    masses = np.asarray(masses, dtype=float)

    if adjacency is None:
        adjacency = conformers.connectivity(numbers, positions)
    if candidates is None:
        candidates = graph_automorphisms(numbers, adjacency)

    x = positions - _mass_centre(positions, masses)

    proper, improper, rejected = [], [], []
    for p in candidates:
        # `p[i] = j` means atom i takes atom j's place, so the permuted geometry
        # is x[p]. Both copies share a mass centre by construction: a graph
        # automorphism preserves elements and hence masses.
        y = x[p]
        rmsd_p, rmsd_i = _best_fits(x, y, masses)
        rec = dict(permutation=[int(v) for v in p],
                   rmsd_proper_A=rmsd_p, rmsd_improper_A=rmsd_i,
                   rmsd_A=min(rmsd_p, rmsd_i))
        if rmsd_p <= tolerance_A:
            # A proper rotation realises this permutation. This is what sigma
            # counts, whether or not an improper operation realises it too.
            proper.append(rec)
        elif rmsd_i <= tolerance_A:
            # Only a mirror or S_n realises it: part of the point group, not of
            # the rotational subgroup, so it must NOT enter sigma.
            improper.append(rec)
        else:
            rejected.append(rec)

    rejected.sort(key=lambda d: d["rmsd_A"])
    accepted_rmsd = ([r["rmsd_proper_A"] for r in proper]
                     + [r["rmsd_improper_A"] for r in improper])
    return dict(
        sigma=len(proper),
        n_proper=len(proper),
        n_improper=len(improper),
        n_graph_automorphisms=len(candidates),
        tolerance_A=float(tolerance_A),
        max_accepted_rmsd_A=max(accepted_rmsd) if accepted_rmsd else 0.0,
        min_rejected_rmsd_A=(rejected[0]["rmsd_A"] if rejected else None),
        # The MARGIN is the whole story: how much room there is between the worst
        # operation kept and the best one thrown away. A margin near zero means
        # the tolerance is doing the deciding, and the sweep must be read.
        margin_A=((rejected[0]["rmsd_A"] - max(accepted_rmsd))
                  if rejected and accepted_rmsd else None),
        proper_operations=proper,
        improper_operations=improper,
        closest_rejected=rejected[:3],
    )


def group_closure_defect(operations, n_atoms):
    """How far the accepted PROPER operations are from being a group.

    A set of rotations that is a genuine symmetry group is closed under
    composition. If it is not, the count is not a group order and sigma is
    meaningless -- so this is checked and reported rather than assumed.

    Returns the number of products p*q that fall outside the accepted set.
    """
    perms = {tuple(op["permutation"]) for op in operations}
    if not perms:
        return None
    defect = 0
    for a, b in itertools.product(perms, repeat=2):
        comp = tuple(a[b[i]] for i in range(n_atoms))
        if comp not in perms:
            defect += 1
    return int(defect)


# ======================================================================================
# 3. pymsym cross-check (point-group LABEL only)
# ======================================================================================
def pymsym_point_group(numbers, positions):
    """Point-group label from pymsym, or None if pymsym is absent.

    Used for the label and as an independent check on ideal geometries. Never as
    the production source of sigma -- see the module docstring for the measured
    reason.
    """
    try:
        import pymsym
    except ImportError:
        return None
    try:
        return pymsym.get_point_group([int(z) for z in numbers],
                                      [[float(c) for c in p] for p in positions])
    except Exception:
        return None
    # NOTE: pymsym returns the literal string "C1" both when the molecule really
    # is C1 and when libmsym raised inside and it swallowed the exception. Those
    # two are not the same statement and must not be recorded as if they were.


def pymsym_sigma(point_group):
    """Symmetry number implied by a point-group label, or None."""
    if point_group is None:
        return None
    try:
        from pymsym.high_level import SYMMNO_BY_POINT_GROUP
    except ImportError:
        return None
    return SYMMNO_BY_POINT_GROUP.get(point_group)


# ======================================================================================
# 4. The production entry point
# ======================================================================================
def analyse(numbers, positions, masses=None, tolerance_A=TOLERANCE_A,
            scan=TOLERANCE_SCAN_A, declared=None):
    """Full sigma record for ONE geometry. This is what goes into a product row.

    `declared` is the value from configs/openqha.yaml, if the species
    has one. The declaration WINS (plan_A section 2.5.4) -- but the detected value
    is recorded next to it either way, so a disagreement is visible instead of
    being silently overridden.
    """
    numbers = np.asarray(numbers, dtype=int)
    positions = np.asarray(positions, dtype=float)
    adjacency = conformers.connectivity(numbers, positions)
    candidates = graph_automorphisms(numbers, adjacency)

    ops = symmetry_operations(numbers, positions, masses=masses,
                              adjacency=adjacency, tolerance_A=tolerance_A,
                              candidates=candidates)
    sweep = {}
    for t in sorted(set(list(scan) + [tolerance_A])):
        o = symmetry_operations(numbers, positions, masses=masses,
                                adjacency=adjacency, tolerance_A=t,
                                candidates=candidates)
        sweep["{:g}".format(t)] = dict(sigma=o["sigma"], n_improper=o["n_improper"])

    pg = pymsym_point_group(numbers, positions)
    pg_sigma = pymsym_sigma(pg)

    # Stability is asked about the tolerances AT OR BELOW the working one, not about
    # the whole sweep. The loose end of the sweep is there to show WHERE the method
    # breaks, and it does break there: measured on acetone 2026-09-03, sigma goes
    # 2 -> 14 at 0.40 A, because at that tolerance the methyl-rotation permutations
    # start being accepted. That is the D0-9 failure mode appearing exactly where it
    # should, and it is evidence the method works -- demanding a flat sweep would
    # have thrown that evidence away as a failure.
    #
    # What matters is that the working tolerance sits on a plateau with room under
    # it, and the MARGIN says how much room: on acetone the accepted operations sit
    # at 8e-6 A and the nearest rejected one at 0.327 A, a factor of 4e4.
    below = [float(k) for k in sweep if float(k) <= tolerance_A]
    sigma_below = {sweep["{:g}".format(t)]["sigma"] for t in below}
    flip = None
    for k in sorted(sweep, key=float):
        if sweep[k]["sigma"] != ops["sigma"]:
            flip = float(k)
            break
    values = sorted({v["sigma"] for v in sweep.values()})
    rec = dict(
        sigma=int(ops["sigma"]),
        sigma_source="geometry_rigid_superposition",
        tolerance_A=float(tolerance_A),
        n_proper=ops["n_proper"],
        n_improper=ops["n_improper"],
        n_graph_automorphisms=ops["n_graph_automorphisms"],
        graph_automorphism_note=(
            "Candidate pool only. Reporting this as sigma is what D0-9 forbids: "
            "for ethane it is 18 against a true sigma of 6."),
        max_accepted_rmsd_A=ops["max_accepted_rmsd_A"],
        min_rejected_rmsd_A=ops["min_rejected_rmsd_A"],
        margin_A=ops["margin_A"],
        tolerance_sweep=sweep,
        sigma_stable_below_tolerance=bool(len(sigma_below) == 1),
        sigma_flip_tolerance_A=flip,
        sigma_stable_over_whole_sweep=bool(len(values) == 1),
        sigma_values_over_sweep=values,
        group_closure_defect=group_closure_defect(ops["proper_operations"],
                                                  len(numbers)),
        pymsym_point_group=pg,
        pymsym_sigma=pg_sigma,
        pymsym_note=(
            "pymsym returns 'C1' both for a genuine C1 and for an internal "
            "libmsym failure; on a geometry displaced from ideal by 0.005 A it "
            "already returns C1 for acetone. Label only, never the production "
            "source of sigma."),
    )

    if declared is not None:
        rec["declared_sigma"] = int(declared)
        rec["agrees_with_declaration"] = bool(int(declared) == rec["sigma"])
        rec["detected_sigma"] = int(ops["sigma"])
        rec["sigma"] = int(declared)
        rec["sigma_source"] = "declared_in_config"
        if not rec["agrees_with_declaration"]:
            rec["disagreement_warning"] = (
                "declared sigma = {} but the geometry gives {}. The declaration "
                "is used; this molecule must be looked at by hand.".format(
                    declared, ops["sigma"]))

    if rec["group_closure_defect"]:
        rec["closure_warning"] = (
            "the accepted proper operations are not closed under composition "
            "({} products fall outside the set) -- the count is not a group "
            "order and sigma cannot be trusted".format(rec["group_closure_defect"]))
    if not rec["sigma_stable_below_tolerance"]:
        rec["sweep_warning"] = (
            "sigma is not constant at or below the working tolerance {} A "
            "(values {}) -- this geometry sits near a flip and the answer depends "
            "on where the tolerance was put".format(tolerance_A, values))
    elif flip is not None and flip <= 2.0 * tolerance_A:
        # Stable, but only just. Worth saying so rather than reporting a clean pass.
        rec["margin_warning"] = (
            "sigma is stable up to the working tolerance {} A but flips at {} A, "
            "less than a factor 2 above it".format(tolerance_A, flip))
    return rec


def analyse_atoms(atoms, **kwargs):
    """`analyse` for an ase.Atoms."""
    return analyse(atoms.get_atomic_numbers(), atoms.get_positions(),
                   masses=atoms.get_masses(), **kwargs)
