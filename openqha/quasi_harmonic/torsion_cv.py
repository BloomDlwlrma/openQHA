"""Derive torsional collective variables automatically from a three-dimensional
structure -- strategy one of branch 1 (conformational isomers).

**Why RDKit's default definition of a rotatable bond is not used**: its standard SMARTS is

requires heavy-atom degree `> 1` at both ends, which **excludes terminal rotors
entirely**. Measured 2026-09-01: `Lipinski.NumRotatableBonds(ethanol)` returns **0**,
while the reference paper (J. Chem. Theory Comput. 2020, 16, 196-210) uses **precisely
ethanol's hydroxyl and methyl terminal rotors**. **The default definition would miss the
reference paper's own collective variables.**

**This module's rule, written as a sentence**: *a collective variable is taken for every
bond that is acyclic, single, and heavy-atom at both ends; each end must also have at
least one other neighbour, hydrogen included.* -- the only difference from RDKit's
default is one thing: **heavy-atom degree greater than one is not required**, so methyl,
hydroxyl and amino terminal rotors are kept.

**Handling redundancy** (the user's key caution of 2026-09-01): for a symmetric rotor
such as a methyl group, **define one dihedral, not three**. The way to do that is to take
one **determinate** reference atom at each end (heavy atom first, then lowest index) and
to record the rotor's **order of symmetry** `n` alongside -- `n` sets the period of the
free-energy surface along that axis and is the basis of the symmetry acceptance in
section 7.

Units: angles are always in radians; atom indices always start **from 0**, matching the
line order of an xyz file.
"""
import numpy as np

# Covalent radii in A, Cordero et al. 2008; only the elements this project meets
COVALENT_RADII = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57}
BOND_TOLERANCE = 1.25          # a convention that may be changed: a bond exists when the
                               # length is < 1.25 x the sum of the radii
HEAVY = tuple(e for e in COVALENT_RADII if e != "H")


def connectivity(symbols, positions, tolerance=BOND_TOLERANCE):
    """Decide bonds from covalent radii. Returns an adjacency table `{i: set(j)}`.

    **This is a convention, not a quantum-chemical bond order** -- a tolerance of 1.25 is
    enough for this project's equilibrium structures, but it goes wrong on stretched
    transition-state structures. A caller using non-equilibrium geometry must check for
    itself.
    """
    n = len(symbols)
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            ri = COVALENT_RADII.get(symbols[i])
            rj = COVALENT_RADII.get(symbols[j])
            if ri is None or rj is None:
                raise ValueError("no covalent radius for {} or {}".format(
                    symbols[i], symbols[j]))
            d = float(np.linalg.norm(positions[i] - positions[j]))
            if d < tolerance * (ri + rj):
                adj[i].add(j)
                adj[j].add(i)
    return adj


# Bond order: this module does **no quantum-chemical bond-order analysis**; it uses the
# bond-length ratio as a heuristic criterion only.
# A defect found by self-review on 2026-09-01: the module's rule says "single bond", while
# the first version of the code **never checked bond order at all** -- this project's
# seven species have no carbon-carbon double bond so it never showed, but on a molecule
# containing C=C it would treat a double bond as rotatable.
MULTIPLE_BOND_RATIO = 0.93     # a convention that may be changed: length/sum-of-radii
                               # < 0.93 counts as a multiple bond


def bond_order_hint(symbols, positions, i, j, ratio=MULTIPLE_BOND_RATIO):
    """Returns ("single" / "multiple", the bond-length ratio). **A heuristic, not a bond
    order.**"""
    d = float(np.linalg.norm(positions[i] - positions[j]))
    s = COVALENT_RADII[symbols[i]] + COVALENT_RADII[symbols[j]]
    r = d / s
    return ("multiple" if r < ratio else "single"), r


def ring_bonds(adj):
    """Every bond that lies on a ring. The method: if the two ends are still connected
    after deleting the edge, it is on a ring."""
    out = set()
    for i in adj:
        for j in adj[i]:
            if i >= j:
                continue
            seen, stack = {i}, [i]
            while stack:
                a = stack.pop()
                for b in adj[a]:
                    if (a, b) in ((i, j), (j, i)):
                        continue
                    if b not in seen:
                        seen.add(b)
                        stack.append(b)
            if j in seen:
                out.add((i, j))
    return out


def rotor_order(symbols, adj, centre, exclude):
    """Order of symmetry of the rotor at the `centre` end: three equivalent hydrogens = 3,
    two = 2, anything else = 1.

    It counts hydrogens only and does **no group theory** -- a methyl `-CH3` gives 3, a
    methylene end gives 2, everything else gives 1. This is a **convention**, good enough
    for this project's methyl and hydroxyl groups, and to be checked for more complicated
    rotors.
    """
    others = [k for k in adj[centre] if k != exclude]
    if not others:
        return 0
    n_h = sum(1 for k in others if symbols[k] == "H")
    return 3 if (len(others) == 3 and n_h == 3) else (1 if n_h != len(others)
                                                     else len(others))


def _reference_neighbour(symbols, adj, centre, exclude):
    """Take one **determinate** reference atom at each end: heavy atom first, then lowest
    index.

    "Determinate" matters -- change the reference atom and the dihedral shifts bodily, and
    then the free-energy surfaces no longer line up. **The rule has to be deterministic
    and must not vary with the structure.**
    """
    others = [k for k in adj[centre] if k != exclude]
    if not others:
        return None
    heavy = sorted(k for k in others if symbols[k] != "H")
    return heavy[0] if heavy else sorted(others)[0]


def torsion_collective_variables(symbols, positions, tolerance=BOND_TOLERANCE):
    """Derive the torsional collective variables. Returns (a list of them, a diagnostics
    dict).

    Each collective variable is a dict: `atoms` (four 0-based indices), `bond`, `order`
    (the order of symmetry along that axis; how it is taken from the two ends is below),
    and `kind`.
    """
    symbols = list(symbols)
    positions = np.asarray(positions, dtype=float)
    adj = connectivity(symbols, positions, tolerance)
    rings = ring_bonds(adj)
    cvs, skipped = [], []
    for i in sorted(adj):
        for j in sorted(adj[i]):
            if i >= j:
                continue
            if symbols[i] == "H" or symbols[j] == "H":
                continue                                  # both ends must be heavy atoms
            if (i, j) in rings:
                skipped.append(dict(bond=[i, j], reason="ring bond, not rotatable"))
                continue
            order, ratio = bond_order_hint(symbols, positions, i, j)
            if order == "multiple":
                skipped.append(dict(bond=[i, j],
                                    reason="bond-length ratio {:.3f} < {} -- judged a "
                                           "multiple bond, not rotatable (**a heuristic, "
                                           "not bond-order analysis**)".format(
                                               ratio, MULTIPLE_BOND_RATIO)))
                continue
            a = _reference_neighbour(symbols, adj, i, j)
            d = _reference_neighbour(symbols, adj, j, i)
            if a is None or d is None:
                skipped.append(dict(bond=[i, j],
                                    reason="one end has no other neighbour, so the "
                                           "dihedral is undefined"))
                continue
            oi = rotor_order(symbols, adj, i, j)
            oj = rotor_order(symbols, adj, j, i)
            cvs.append(dict(atoms=[int(a), int(i), int(j), int(d)],
                            bond_length_ratio=float(ratio),
                            bond=[int(i), int(j)],
                            elements=[symbols[a], symbols[i], symbols[j], symbols[d]],
                            rotor_order_i=int(oi), rotor_order_j=int(oj),
                            periodicity=int(max(oi, oj, 1)),
                            kind=("methyl-type symmetric rotor" if 3 in (oi, oj)
                                  else "general torsion")))
    diag = dict(n_atoms=len(symbols), n_bonds=sum(len(v) for v in adj.values()) // 2,
                n_ring_bonds=len(rings), skipped=skipped,
                tolerance=tolerance,
                rule=("acyclic, not a multiple bond (by the bond-length heuristic), heavy "
                      "atoms at both ends, another neighbour at each end; **heavy-atom "
                      "degree greater than one is NOT required**, so terminal rotors are "
                      "kept"))
    return cvs, diag


def dihedral(positions, atoms):
    """The dihedral defined by four atoms, in radians on `(-pi, pi]`. The standard IUPAC
    sign convention."""
    p = np.asarray(positions, dtype=float)
    b0 = p[atoms[0]] - p[atoms[1]]
    b1 = p[atoms[2]] - p[atoms[1]]
    b2 = p[atoms[3]] - p[atoms[2]]
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def dihedral_gradient(positions, atoms):
    """Analytic gradient of the dihedral with respect to the Cartesian coordinates of the
    four atoms, in 1/A. The biasing force of metadynamics needs it.

    Self-checked against central differences: see `tests`.
    **This is not numerical differentiation, it is the analytic expression.**
    """
    p = np.asarray(positions, dtype=float)
    i, j, k, l = atoms
    f = p[i] - p[j]
    g = p[j] - p[k]
    h = p[l] - p[k]
    a = np.cross(f, g)
    b = np.cross(h, g)
    ga = np.dot(a, a)
    gb = np.dot(b, b)
    gn = np.linalg.norm(g)
    grad = np.zeros_like(p)
    grad[i] = -gn / ga * a
    grad[l] = gn / gb * b
    t1 = np.dot(f, g) / (ga * gn) * a
    t2 = np.dot(h, g) / (gb * gn) * b
    grad[j] = -grad[i] + t1 - t2
    grad[k] = -grad[l] - t1 + t2
    return grad
