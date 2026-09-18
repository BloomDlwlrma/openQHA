"""Enantiomer degeneracy g' of every basin: a port of CREST's `intraconfRMSD`.

WHAT g' IS (CONTEXT.md, "Enantiomer degeneracy")
------------------------------------------------
The Gibbs-Shannon conformational entropy counts *states*. Two geometries are two states
only if they are distinguishable and not related by a permutation of identical nuclei.
CREST's conformer sort has already put every structure of equal energy and equal
rotational constants into one conformer with several rotamers; this module looks inside
each rotamer group and counts how many genuinely different *core structures* it holds.
That count is g' (the paper's g'_i, Pracht & Grimme, Chem. Sci. 2021, 12, 6551, eq. 10):
1 for most conformers, 2 for a geometric enantiomer pair such as gauche-propanal.

A methyl rotated by 120 degrees is the same quantum state (three identical protons
permuted), so the rotor atoms are EXCLUDED from the comparison and never counted. A
mirror image cannot be superimposed by any proper rotation, so it forms a core of its
own: that is how g' = 2 arises. No rotamer factor (g_rot) exists here at all.

THE ALGORITHM (crest-master `src/entropy/entropic.f90`, 2026-09-16)
-------------------------------------------------------------------
1. Equivalence classes of nuclei (CREST: `anmr_nucinfo`, derived from the ensemble;
   here: the automorphism classes of the connectivity graph, which coincide for the
   rotor groups this rule needs).
2. Rotor groups (`distsubgr`): two equivalent nuclei bonded to one common neighbour
   are a rotatable group; every such atom is excluded from the RMSD (methyl and CH2
   hydrogens, NH2, ...). CREST's ring and eta5-Cp special cases are not ported: no
   molecule of this project has a freely rotating ring.
3. Pairwise RMSD (`intraconfRMSD`) between all rotamers of one conformer on the
   remaining "core" atoms, after optimal PROPER rotation (quaternion / Kabsch with the
   determinant sign fixed), threshold `RTHR = 0.125 A`.
4. Greedy clustering (`uniqueCore`): the number of clusters is the number of core
   structures = g' (CREST: `corefac`, the factor that enters S'_conf).
5. Mirror check: for a pair above RTHR, negate x of one partner and re-superimpose;
   below `0.75 * RTHR` the conformer is flagged as having an enantiomer (CREST:
   `enantiofac`, diagnostic).

Both mirror images must have been SAMPLED by CREST for step 4 to see them; a rotamer
group with one member is reported as `unsampled`, g' = 1.

Inputs are CREST's own engine files in the crest engine folder: `crest_rotamers.xyz`
and `cre_members` (one line per conformer: n_rotamers, first, last, 1-based). Output is
the `degeneracy` Calculation's Record in the level folder.
"""
import math
import re
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers

from ..store import layout, property as prop, report
from . import crest

#: CREST `intraconfRMSD`: `rthr` -- core RMSD below which two rotamers are one core.
RTHR_A = 0.125
#: CREST `intraconfRMSD`: the mirrored partner must fall below `0.75 * rthr`.
MIRROR_FACTOR = 0.75

STEP = "degeneracy"
PROGNAME = "openQHA degeneracy"

SCHEMA = {
    "Calculation_Info": {
        "MOLECULE_DIR": ("String", None, "the molecule directory"),
        "LEVEL": ("String", None, "the level the basins belong to"),
        "CREST_DIR": ("String", None, "the crest engine folder the rotamer file was read from"),
        "N_CONFORMERS": ("Integer", None, "conformers in cre_members"),
        "N_ROTAMERS": ("Integer", None, "structures in crest_rotamers.xyz"),
        "RTHR": ("Double", "A", "core RMSD below which two rotamers are one core"),
        "MIRROR_FACTOR": ("Double", None, "mirror match threshold as a fraction of RTHR"),
        "N_EXCLUDED_ATOMS": ("Integer", None, "rotor-group atoms left out of the RMSD"),
        "CHIRALITY_THRESHOLD": ("Double", "A", "self-mirror RMSD below which a conformer is achiral (MIRROR_FACTOR * RTHR)"),
        "PERMUTATION_CAP": ("Integer", None, "equivalence-class permutations tried before the chirality is unresolved"),
        "PROCRUSTES_VERSION": ("String", None, "qc-procrustes version behind the rotational / orthogonal RMSDs"),
    },
    "Basin": {
        "INDEX": ("Integer", None, "basin index"),
        "CONFORMER": ("Integer", None, "the CREST conformer the basin came from (0-based)"),
        "G_PRIME": ("Integer", None, "enantiomer degeneracy g'"),
        "G_PRIME_SOURCE": ("String", None, "mirror_pair, class_inherited, achiral, no_mirror_sampled, label_fallback, pooled_frame or mirror_is_basin"),
        "CHIRALITY": ("String", None, "achiral, chiral or unresolved, from MIRROR_SELF_RMSD against CHIRALITY_THRESHOLD"),
        "MIRROR_SELF_RMSD": ("Double", "A", "rotational-Procrustes RMSD of the core to its own reflection, minimised over equivalence-class permutations"),
        "SYMMETRY_LABEL": ("String", None, "point-group class label (diagnostic only; decides nothing)"),
        "MIRROR_RMSD_ROT": ("Double", "A", "rotational-Procrustes RMSD of the closest mirrored rotamer pair"),
        "MIRROR_RMSD_ORTHO": ("Double", "A", "orthogonal-Procrustes RMSD of that pair (reflection allowed)"),
        "N_ROTAMERS": ("Integer", None, "rotamers of that conformer"),
        "N_CORES": ("Integer", None, "RMSD-distinct core structures among them"),
        "MIRROR_FLAG": ("Boolean", None, "a mirrored partner was found below the threshold"),
        "MAX_CORE_RMSD": ("Double", "A", "largest core RMSD inside the rotamer group"),
        "MIRROR_PARTNER": ("Integer", None, "the basin that is this one's mirror image, or -1"),
        "DUPLICATE_OF": ("Integer", None, "a basin with the same core (should not happen), or -1"),
    },
}


# ====================================================================== geometry
def kabsch_rmsd(a, b):
    """RMSD after optimal PROPER rotation (reflections excluded) and translation."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean(0)
    b = b - b.mean(0)
    h = a.T @ b
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    if d == 0.0:
        d = 1.0
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return float(np.sqrt(((a @ r.T - b) ** 2).sum(1).mean()))


#: Above this many equivalence-class permutations the self-mirror RMSD is not searched;
#: the conformer's chirality is `unresolved` and g' falls back to the point-group label.
PERMUTATION_CAP = 10000
#: Our Kabsch and the library's rotational Procrustes are the same SVD; they are asserted
#: equal on every pair to this tolerance (a disagreement is a bug, not a convention).
KABSCH_LIBRARY_TOL_A = 1e-6


def procrustes_version():
    from importlib import metadata
    try:
        return metadata.version("qc-procrustes")
    except metadata.PackageNotFoundError:
        return "unknown"


def procrustes_rmsds(a, b):
    """(RMSD_rot, RMSD_ortho): the rotational-Procrustes RMSD (proper rotations only) and
    the orthogonal-Procrustes RMSD (reflection allowed) between two point sets after
    translation, from the `procrustes` library (Meng et al., CPC 2022, 276, 108334).
    `error` there is the squared Frobenius norm; RMSD = sqrt(error / N)."""
    from procrustes import orthogonal, rotational
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.shape[0]
    r_rot = math.sqrt(max(rotational(a, b, translate=True, scale=False).error, 0.0) / n)
    r_ort = math.sqrt(max(orthogonal(a, b, translate=True, scale=False).error, 0.0) / n)
    return float(r_rot), float(r_ort)


def checked_rmsd(a, b):
    """`kabsch_rmsd`, asserted equal to the library's rotational RMSD on this pair."""
    r = kabsch_rmsd(a, b)
    r_lib, _ = procrustes_rmsds(a, b)
    if abs(r - r_lib) > KABSCH_LIBRARY_TOL_A:
        raise AssertionError("kabsch_rmsd {:.8f} != procrustes rotational RMSD {:.8f} A"
                             .format(r, r_lib))
    return r


def _class_permutations(classes, cap=PERMUTATION_CAP):
    """Every permutation of the atom indices that maps each equivalence class onto
    itself, or None when there are more than `cap` of them."""
    import itertools
    groups = {}
    for i, c in enumerate(classes):
        groups.setdefault(c, []).append(i)
    members = [g for g in groups.values() if len(g) > 1]
    count = 1
    for g in members:
        count *= math.factorial(len(g))
        if count > cap:
            return None
    n = len(classes)
    perms = []
    for combo in itertools.product(*[itertools.permutations(g) for g in members]):
        perm = list(range(n))
        for g, pg in zip(members, combo):
            for src, dst in zip(g, pg):
                perm[src] = dst
        perms.append(perm)
    return perms


def self_mirror_rmsd(positions, classes, cap=PERMUTATION_CAP, perms=None):
    """Distance of a structure from achirality: the smallest rotational-Procrustes RMSD
    between the structure and its own reflection (x -> -x), over every permutation of
    chemically equivalent atoms (the library has no permutation search; `classes` are
    RDKit's canonical ranks). None when the permutation count exceeds `cap`.

    The search runs on `kabsch_rmsd` (the same SVD, 16x cheaper than the library call)
    and the winning permutation is then evaluated by the library and asserted equal, so
    the reported number is the library's. `perms` may be passed in when the same
    molecule is scored many times (one enumeration per molecule, not per conformer)."""
    xyz = np.asarray(positions, dtype=float)
    if perms is None:
        perms = _class_permutations(classes, cap)
    if perms is None:
        return None
    mirrored = xyz.copy()
    mirrored[:, 0] *= -1.0
    best, best_perm = float("inf"), None
    for perm in perms:
        r = kabsch_rmsd(xyz, mirrored[perm])
        if r < best:
            best, best_perm = r, perm
    return checked_rmsd(xyz, mirrored[best_perm])


def equivalence_classes(symbols, positions):
    """Class label per atom: automorphism classes of the connectivity graph (RDKit's
    canonical ranks without tie breaking). Two atoms with the same label are chemically
    equivalent nuclei in CREST's sense."""
    from ase.data import atomic_numbers
    from rdkit import Chem
    from . import conformers
    numbers = np.array([atomic_numbers[s] for s in symbols])
    adj = conformers.connectivity(numbers, np.asarray(positions, dtype=float))
    rw = Chem.RWMol()
    for z in numbers:
        rw.AddAtom(Chem.Atom(int(z)))
    n = len(numbers)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                rw.AddBond(i, j, Chem.BondType.SINGLE)
    m = rw.GetMol()
    for a in m.GetAtoms():
        a.SetNoImplicit(True)
    m.UpdatePropertyCache(strict=False)
    ranks = list(Chem.CanonicalRankAtoms(m, breakTies=False))
    return ranks, adj


def rotor_group_atoms(classes, adjacency):
    """Atoms to exclude from the core RMSD: members of one equivalence class that are
    bonded to one common neighbour (CREST `distsubgr`, the 'simple rotameric group'
    rule, also applied to two-member groups such as CH2)."""
    adj = np.asarray(adjacency)
    n = len(classes)
    excluded = set()
    by_class = {}
    for i, c in enumerate(classes):
        by_class.setdefault(c, []).append(i)
    for members in by_class.values():
        if len(members) < 2:
            continue
        for k, a in enumerate(members):
            for b in members[k + 1:]:
                common = [c for c in range(n) if adj[a, c] and adj[b, c]]
                if len(common) == 1:
                    excluded.add(a)
                    excluded.add(b)
    return sorted(excluded)


def symmetry_class(numbers, positions):
    """CREST propagates the enantiomer flag inside a point-group class; here the class
    is `achiral` (an improper operation exists), or `chiral_s<sigma>` (none exists, sigma
    proper operations). C1 -> chiral_s1, C2 -> chiral_s2, Cs / Ci / C2v / C2h -> achiral."""
    from . import symmetry
    ops = symmetry.symmetry_operations(np.asarray(numbers, dtype=int),
                                       np.asarray(positions, dtype=float))
    if ops["n_improper"] > 0:
        return "achiral"
    return "chiral_s{}".format(int(ops["sigma"]))


def rotor_factor(classes, adjacency):
    """CREST's g_rot: the product of the sizes of the freely rotatable groups (a methyl
    contributes 3). A group is freely rotatable when its common neighbour has at most one
    other neighbour (ESI rule); a CH2 in a chain is excluded from the RMSD but is not a
    rotor. Constant for every conformer of a molecule; cancels in every population."""
    adj = np.asarray(adjacency)
    n = len(classes)
    by_class = {}
    for i, c in enumerate(classes):
        by_class.setdefault(c, []).append(i)
    factor = 1
    for members in by_class.values():
        if len(members) < 2:
            continue
        a, b = members[0], members[1]
        common = [c for c in range(n) if adj[a, c] and adj[b, c]]
        if len(common) != 1:
            continue
        centre = common[0]
        if all(adj[centre, m] for m in members):
            others = [c for c in range(n) if adj[centre, c] and c not in members]
            if len(others) <= 1:
                factor *= len(members)
    return int(factor)


# ====================================================================== CREST files
def read_cre_members(path):
    """`cre_members`: first line the conformer count, then one line per conformer
    `n_rotamers first last` (1-based, inclusive)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("cre_members not found: {} -- the crest engine folder has no "
                                "rotamer bookkeeping, g' cannot be derived".format(path))
    lines = [l.split() for l in path.read_text().splitlines() if l.strip()]
    n = int(lines[0][0])
    rows = [(int(a), int(b), int(c)) for a, b, c in (l[:3] for l in lines[1:1 + n])]
    if len(rows) != n:
        raise ValueError("cre_members announces {} conformers but lists {}".format(n, len(rows)))
    return rows


def unique_cores(rmat, thr):
    """CREST `uniqueCore`: greedy grouping over the RMSD matrix; returns the group count."""
    n = rmat.shape[0]
    mp = [0] * n
    gr = 0
    for i in range(n):
        if mp[i] == 0:
            gr += 1
            mp[i] = gr
        for j in range(n):
            if rmat[i, j] < thr:
                mp[j] = gr
    return gr


def conformer_degeneracies(crest_dir, rthr=RTHR_A, mirror_factor=MIRROR_FACTOR,
                           cap=PERMUTATION_CAP):
    """g' for every conformer of a CREST run, from `crest_rotamers.xyz` and `cre_members`.

    Returns one dict per conformer (in cre_members order): n_rotamers, n_cores, g_prime,
    g_prime_source, mirror_flag, max_core_rmsd, excluded_atoms (0-based), and the
    chirality (`mirror_self_rmsd`, `chirality`, the point-group `symmetry_class` label).
    """
    crest_dir = Path(crest_dir)
    rot_path = crest_dir / "crest_rotamers.xyz"
    if not rot_path.is_file():
        raise FileNotFoundError("crest_rotamers.xyz not found: {} -- g' needs CREST's rotamer "
                                "file, not only the conformers".format(rot_path))
    members = read_cre_members(crest_dir / "cre_members")
    frames = crest.read_ensemble(rot_path)
    if not frames:
        raise ValueError("crest_rotamers.xyz is empty: {}".format(rot_path))
    symbols = [a[0] for a in frames[0][1]]
    xyz = np.array([[[a[1], a[2], a[3]] for a in f[1]] for f in frames], dtype=float)
    classes, adj = equivalence_classes(symbols, xyz[0])
    excluded = rotor_group_atoms(classes, adj)
    core = [i for i in range(len(symbols)) if i not in set(excluded)]
    if len(core) < 2:
        core_ok = False
    else:
        core_ok = True
    numbers = [atomic_numbers[s] for s in symbols]
    core_classes = [classes[i] for i in core]
    core_perms = _class_permutations(core_classes, cap) if core_ok else None
    g_rot = rotor_factor(classes, adj)
    thr = mirror_factor * rthr
    out = []
    for n_rot, lo, hi in members:
        idx = list(range(lo - 1, hi))
        if len(idx) != n_rot or hi > len(frames):
            raise ValueError("cre_members row ({}, {}, {}) does not match {} structures in "
                             "crest_rotamers.xyz".format(n_rot, lo, hi, len(frames)))
        # the continuous chirality of the conformer: its core against its own reflection
        # (ticket 29); the point-group label is kept beside it as a diagnostic
        self_rmsd = (self_mirror_rmsd(xyz[idx[0]][core], core_classes, cap, perms=core_perms)
                     if core_ok and core_perms is not None else None)
        if self_rmsd is None:
            chirality = "unresolved"
        else:
            chirality = "achiral" if self_rmsd < thr else "chiral"
        rec = dict(n_rotamers=n_rot, excluded_atoms=excluded, core_atoms=core,
                   symmetry_class=symmetry_class(numbers, xyz[idx[0]]), g_rot=g_rot,
                   mirror_self_rmsd=self_rmsd, chirality=chirality,
                   mirror_rmsd_rot=None, mirror_rmsd_ortho=None)
        if n_rot < 2 or not core_ok:
            rec.update(n_cores=1, mirror_flag=False, max_core_rmsd=0.0,
                       cores_source="unsampled" if n_rot < 2 else "core_too_small")
            out.append(rec)
            continue
        m = len(idx)
        rmat = np.zeros((m, m))
        mirror = False
        best_pair = None
        for i in range(m):
            ci = xyz[idx[i]][core]
            for j in range(i):
                cj = xyz[idx[j]][core]
                rmat[i, j] = rmat[j, i] = checked_rmsd(ci, cj)
                if rmat[i, j] > rthr:
                    # CREST's mirror test as the library phrases it: a mirror pair when the
                    # orthogonal (reflection allowed) fit is below the threshold and the
                    # rotational (proper) one is not
                    r_rot, r_ort = procrustes_rmsds(ci, cj)
                    if r_ort < thr <= r_rot:
                        mirror = True
                        if best_pair is None or r_ort < best_pair[1]:
                            best_pair = (r_rot, r_ort)
        n_cores = unique_cores(rmat, rthr)
        rec.update(n_cores=n_cores, mirror_flag=bool(mirror),
                   max_core_rmsd=float(rmat.max()), cores_source="core_count")
        if best_pair is not None:
            rec.update(mirror_rmsd_rot=float(best_pair[0]), mirror_rmsd_ortho=float(best_pair[1]))
        out.append(rec)
    # CREST's g' (`enantiofac`, entropic.f90 lines 1044-1061): 2 for every conformer whose
    # class contains a conformer with a mirror match, 1 otherwise. The class is propagated
    # on purpose: a chiral conformer whose mirror image was not sampled still has one. The
    # class here is the continuous chirality (achiral / chiral), not CREST's point-group
    # label -- the label flipped c1 / cs between two runs on propanal and moved S_conf by
    # R ln 2. A conformer whose chirality is unresolved (permutation cap) falls back to
    # the label rule and says so. The core count is kept as the diagnostic it is in CREST
    # (`corefac`, the factor in cre_degen2 = g_rot * n_cores / g_sym).
    any_chiral_mirror = any(r["mirror_flag"] and r["chirality"] == "chiral" for r in out)
    flagged_labels = {r["symmetry_class"] for r in out if r["mirror_flag"]}
    for r in out:
        if r["mirror_flag"]:
            r.update(g_prime=2, g_prime_source="mirror_pair")
        elif r["chirality"] == "chiral" and any_chiral_mirror:
            r.update(g_prime=2, g_prime_source="class_inherited")
        elif r["chirality"] == "achiral":
            r.update(g_prime=1, g_prime_source="achiral")
        elif r["chirality"] == "unresolved":
            r.update(g_prime=2 if r["symmetry_class"] in flagged_labels else 1,
                     g_prime_source="label_fallback")
        else:
            r.update(g_prime=1, g_prime_source="no_mirror_sampled")
        r["cre_degen2_equivalent"] = int(g_rot * r["n_cores"])
    return out


# ====================================================================== the Calculation
_CONF_RE = re.compile(r"\bconformer=(\d+)")


def basin_conformers(molecule):
    """(basin index, conformer index) for every basin, read from the `conformer=` key
    branch A writes into each `basin.extxyz` header."""
    from ..store import basins as basins_mod
    out = []
    for path in basins_mod.basin_files(molecule):
        header = path.read_text(encoding="utf-8").splitlines()[1]
        m = _CONF_RE.search(header)
        if not m:
            raise ValueError("{} carries no conformer= key in its header; cannot map the "
                             "basin to a CREST conformer".format(path))
        out.append((int(path.parent.name[len("basin"):]), int(m.group(1))))
    return out


def n_frames_from_crest(molecule):
    """How many of branch A's input frames came from CREST's conformer file: `[Census]
    N_FROM_CREST` of `branchA.toml`. The frames after them are pooled reference
    geometries that have no CREST rotamer group. None when the record is absent."""
    p = layout.records_dir(molecule) / "branchA.toml"
    if not p.is_file():
        return None
    doc = prop.load(p)
    return int((doc.get("Census") or {}).get("N_FROM_CREST"))


def basin_mirror_pairs(molecule, excluded, rthr=RTHR_A, mirror_factor=MIRROR_FACTOR):
    """Pairs of basins that are mirror images of each other on the core atoms (branch A's
    deduplication superimposes by proper rotation only, so it keeps both enantiomers when
    both were among its input frames). Returns ({basin: partner}, {basin: duplicate}).

    This is the "count once" rule (plan_AB section 4.1) applied at the basin level: a
    basin whose mirror image is itself a basin contributes with g' = 1, whatever CREST's
    rotamer group said, because the partner already stands in the sum.
    """
    from ase.io import read
    from ..store import basins as basins_mod
    atoms = [(int(p.parent.name[len("basin"):]), read(str(p), format="extxyz"))
             for p in basins_mod.basin_files(molecule)]
    if not atoms:
        return {}, {}
    n = len(atoms[0][1])
    core = [i for i in range(n) if i not in set(excluded)]
    partner, duplicate = {}, {}
    for k, (bi, ai) in enumerate(atoms):
        for bj, aj in atoms[:k]:
            ci = ai.get_positions()[core]
            cj = aj.get_positions()[core]
            r = checked_rmsd(ci, cj)
            if r < rthr:
                duplicate[bi] = bj
                duplicate[bj] = bi
                continue
            r_rot, r_ort = procrustes_rmsds(ci, cj)
            if r_ort < mirror_factor * rthr <= r_rot:
                partner[bi] = bj
                partner[bj] = bi
    return partner, duplicate


def run_calculation(molecule, level, crest_folder=None, rthr=RTHR_A,
                    mirror_factor=MIRROR_FACTOR, cap=PERMUTATION_CAP):
    """The `degeneracy` Calculation: g' per basin, written to the level folder.

    Three sources of g', in this order of precedence:
      mirror_is_basin  the basin's mirror image is itself a basin: g' = 1 for both
      core_count       CREST's rotamer group of the basin's conformer (g' = n_cores)
      unsampled        that group has one member: g' = 1, a floor not a finding
      pooled_frame     the basin came from a pooled reference geometry, not from CREST:
                       no rotamer group exists, g' = 1
    Returns the record (also what the Property file holds, plus the per-conformer detail
    the Report prints).
    """
    molecule = Path(molecule)
    crest_dir = Path(crest_folder) if crest_folder else layout.crest_dir(molecule)
    confs = conformer_degeneracies(crest_dir, rthr=rthr, mirror_factor=mirror_factor, cap=cap)
    pairs = basin_conformers(molecule)
    if not pairs:
        raise FileNotFoundError("no basin.extxyz under {}".format(molecule / "mace"))
    from ..store import basins as basins_mod
    files = {int(p.parent.name[len("basin"):]): p for p in basins_mod.basin_files(molecule)}
    n_crest = n_frames_from_crest(molecule)
    if n_crest is None:
        n_crest = len(confs)
    excluded = confs[0]["excluded_atoms"] if confs else []
    partner, duplicate = basin_mirror_pairs(molecule, excluded, rthr, mirror_factor)
    rows = []
    for b, c in pairs:
        if c < n_crest:
            if c >= len(confs):
                raise ValueError("basin {} points at CREST conformer {} but cre_members "
                                 "lists {}".format(b, c, len(confs)))
            r = confs[c]
            row = dict(index=b, conformer=c, g_prime=r["g_prime"],
                       g_prime_source=r["g_prime_source"], n_rotamers=r["n_rotamers"],
                       n_cores=r["n_cores"], mirror_flag=r["mirror_flag"],
                       max_core_rmsd=r["max_core_rmsd"], symmetry_class=r["symmetry_class"],
                       chirality=r["chirality"], mirror_self_rmsd=r["mirror_self_rmsd"],
                       mirror_rmsd_rot=r["mirror_rmsd_rot"], mirror_rmsd_ortho=r["mirror_rmsd_ortho"])
        else:
            # a pooled frame has no CREST rotamer group, but it has a geometry: its
            # chirality is the same number as everyone else's, from basin.extxyz
            from ase.io import read
            atoms = read(str(files[b]), format="extxyz")
            sym, pos = atoms.get_chemical_symbols(), atoms.get_positions()
            classes, _adj = equivalence_classes(sym, pos)
            core = [i for i in range(len(sym)) if i not in set(excluded)]
            self_rmsd = (self_mirror_rmsd(pos[core], [classes[i] for i in core], cap)
                         if len(core) >= 2 else None)
            chirality = ("unresolved" if self_rmsd is None
                         else "achiral" if self_rmsd < mirror_factor * rthr else "chiral")
            row = dict(index=b, conformer=c, g_prime=1, g_prime_source="pooled_frame",
                       n_rotamers=0, n_cores=1, mirror_flag=False, max_core_rmsd=0.0,
                       symmetry_class=symmetry_class([atomic_numbers[s] for s in sym], pos),
                       chirality=chirality, mirror_self_rmsd=self_rmsd,
                       mirror_rmsd_rot=None, mirror_rmsd_ortho=None)
        if b in partner:
            row.update(g_prime=1, g_prime_source="mirror_is_basin",
                       mirror_flag=True, mirror_partner=partner[b])
        else:
            row["mirror_partner"] = -1
        row["duplicate_of"] = duplicate.get(b, -1)
        rows.append(row)
    lvl = layout.level_dir(molecule, level)
    info = {"MOLECULE_DIR": str(molecule), "LEVEL": str(level), "CREST_DIR": str(crest_dir),
            "N_CONFORMERS": len(confs),
            "N_ROTAMERS": int(sum(c["n_rotamers"] for c in confs)),
            "RTHR": float(rthr), "MIRROR_FACTOR": float(mirror_factor),
            "N_EXCLUDED_ATOMS": len(excluded),
            "CHIRALITY_THRESHOLD": float(mirror_factor * rthr), "PERMUTATION_CAP": int(cap),
            "PROCRUSTES_VERSION": procrustes_version()}
    blocks = {"Calculation_Info": info,
              "Basin": [{"INDEX": r["index"], "CONFORMER": r["conformer"],
                         "G_PRIME": r["g_prime"], "G_PRIME_SOURCE": r["g_prime_source"],
                         "CHIRALITY": r["chirality"], "MIRROR_SELF_RMSD": r["mirror_self_rmsd"],
                         "SYMMETRY_LABEL": r["symmetry_class"],
                         "MIRROR_RMSD_ROT": r["mirror_rmsd_rot"],
                         "MIRROR_RMSD_ORTHO": r["mirror_rmsd_ortho"],
                         "N_ROTAMERS": r["n_rotamers"], "N_CORES": r["n_cores"],
                         "MIRROR_FLAG": r["mirror_flag"],
                         "MAX_CORE_RMSD": r["max_core_rmsd"],
                         "MIRROR_PARTNER": r["mirror_partner"],
                         "DUPLICATE_OF": r["duplicate_of"]} for r in rows]}
    missing = prop.write(lvl / "degeneracy.toml", blocks, SCHEMA, prop.NORMAL_TERMINATION,
                         PROGNAME)
    if missing:
        raise RuntimeError("degeneracy.toml keys outside the schema: {}".format(missing))

    rep = report.Report("openQHA degeneracy", "enantiomer degeneracy g' per basin, "
                        "port of CREST intraconfRMSD (entropic.f90)")
    rep.section("inputs")
    for k, v in info.items():
        rep.kv(k, v)
    rep.kv("excluded atoms (0-based)", " ".join(str(i) for i in excluded) or "none")
    rep.section("per conformer")
    rep.kv("rotor factor g_rot (cancels in every population)", confs[0]["g_rot"] if confs else 1)
    rep.table(["conformer", "rotamers", "cores", "self-mirror RMSD", "chirality", "label", "mirror", "RMSD rot", "RMSD ortho", "g'", "source", "cre_degen2 eq."],
              [[i, c["n_rotamers"], c["n_cores"],
                "%.4f" % c["mirror_self_rmsd"] if c["mirror_self_rmsd"] is not None else "unresolved",
                c["chirality"], c["symmetry_class"], c["mirror_flag"],
                "%.4f" % c["mirror_rmsd_rot"] if c["mirror_rmsd_rot"] is not None else "-",
                "%.4f" % c["mirror_rmsd_ortho"] if c["mirror_rmsd_ortho"] is not None else "-",
                c["g_prime"], c["g_prime_source"], c["cre_degen2_equivalent"]] for i, c in enumerate(confs)],
              title="chirality threshold {:.4f} A (MIRROR_FACTOR * RTHR); the point-group label decides nothing".format(
                  mirror_factor * rthr))
    rep.section("per basin")
    rep.table(["basin", "conformer", "g'", "mirror partner", "source"],
              [[r["index"], r["conformer"], r["g_prime"],
                r["mirror_partner"] if r["mirror_partner"] >= 0 else "-",
                r["g_prime_source"]] for r in rows])
    if duplicate:
        rep.warn("basins with the same core structure were kept apart by branch A: {}; "
                 "they would be counted twice".format(sorted(duplicate.items())))
    rep.note("g' follows CREST's enantiofac: 2 for every conformer whose class holds a "
             "conformer with a sampled mirror partner (a chiral conformer has a mirror image "
             "whether or not it was sampled), 1 otherwise. The class is the continuous "
             "chirality -- the rotational-Procrustes RMSD of the core to its own reflection, "
             "minimised over permutations of equivalent atoms, against the threshold -- and "
             "not CREST's point-group label, which flipped c1/cs between two runs on propanal. "
             "Mirror pairs: orthogonal Procrustes below the threshold, rotational above "
             "(Meng et al., Comput. Phys. Commun. 2022, 276, 108334, key meng2022procrustes); "
             "our Kabsch is asserted equal to the library's rotational RMSD on every pair. The "
             "core count is the diagnostic CREST writes into cre_degen2 (= g_rot * cores). A "
             "basin whose mirror image is itself a basin is counted once: g' = 1 for both.")
    rep.write(lvl / "degeneracy.out", step=STEP)
    return dict(level=str(level), crest_dir=str(crest_dir), conformers=confs, basins=rows,
                excluded_atoms=excluded, record=lvl / "degeneracy.toml")
