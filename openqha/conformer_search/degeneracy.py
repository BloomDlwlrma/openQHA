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
import re
from pathlib import Path

import numpy as np

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
    },
    "Basin": {
        "INDEX": ("Integer", None, "basin index"),
        "CONFORMER": ("Integer", None, "the CREST conformer the basin came from (0-based)"),
        "G_PRIME": ("Integer", None, "enantiomer degeneracy g'"),
        "G_PRIME_SOURCE": ("String", None, "core_count, or unsampled (one rotamer)"),
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


def conformer_degeneracies(crest_dir, rthr=RTHR_A, mirror_factor=MIRROR_FACTOR):
    """g' for every conformer of a CREST run, from `crest_rotamers.xyz` and `cre_members`.

    Returns one dict per conformer (in cre_members order): n_rotamers, n_cores, g_prime,
    g_prime_source, mirror_flag, max_core_rmsd, excluded_atoms (0-based).
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
    out = []
    for n_rot, lo, hi in members:
        idx = list(range(lo - 1, hi))
        if len(idx) != n_rot or hi > len(frames):
            raise ValueError("cre_members row ({}, {}, {}) does not match {} structures in "
                             "crest_rotamers.xyz".format(n_rot, lo, hi, len(frames)))
        rec = dict(n_rotamers=n_rot, excluded_atoms=excluded, core_atoms=core)
        if n_rot < 2 or not core_ok:
            rec.update(n_cores=1, g_prime=1, mirror_flag=False, max_core_rmsd=0.0,
                       g_prime_source="unsampled" if n_rot < 2 else "core_too_small")
            out.append(rec)
            continue
        m = len(idx)
        rmat = np.zeros((m, m))
        mirror = False
        for i in range(m):
            ci = xyz[idx[i]][core]
            for j in range(i):
                cj = xyz[idx[j]][core]
                rmat[i, j] = rmat[j, i] = kabsch_rmsd(ci, cj)
                if rmat[i, j] > rthr:
                    mirrored = cj.copy()
                    mirrored[:, 0] *= -1.0
                    if kabsch_rmsd(ci, mirrored) < mirror_factor * rthr:
                        mirror = True
        n_cores = unique_cores(rmat, rthr)
        rec.update(n_cores=n_cores, g_prime=n_cores, mirror_flag=bool(mirror),
                   max_core_rmsd=float(rmat.max()), g_prime_source="core_count")
        out.append(rec)
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
            r = kabsch_rmsd(ci, cj)
            if r < rthr:
                duplicate[bi] = bj
                duplicate[bj] = bi
                continue
            m = cj.copy()
            m[:, 0] *= -1.0
            if kabsch_rmsd(ci, m) < mirror_factor * rthr:
                partner[bi] = bj
                partner[bj] = bi
    return partner, duplicate


def run_calculation(molecule, level, crest_folder=None, rthr=RTHR_A,
                    mirror_factor=MIRROR_FACTOR):
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
    confs = conformer_degeneracies(crest_dir, rthr=rthr, mirror_factor=mirror_factor)
    pairs = basin_conformers(molecule)
    if not pairs:
        raise FileNotFoundError("no basin.extxyz under {}".format(molecule / "mace"))
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
                       max_core_rmsd=r["max_core_rmsd"])
        else:
            row = dict(index=b, conformer=c, g_prime=1, g_prime_source="pooled_frame",
                       n_rotamers=0, n_cores=1, mirror_flag=False, max_core_rmsd=0.0)
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
            "N_EXCLUDED_ATOMS": len(excluded)}
    blocks = {"Calculation_Info": info,
              "Basin": [{"INDEX": r["index"], "CONFORMER": r["conformer"],
                         "G_PRIME": r["g_prime"], "G_PRIME_SOURCE": r["g_prime_source"],
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
    rep.table(["conformer", "rotamers", "cores", "g'", "mirror", "max core RMSD", "source"],
              [[i, c["n_rotamers"], c["n_cores"], c["g_prime"], c["mirror_flag"],
                "%.4f" % c["max_core_rmsd"], c["g_prime_source"]] for i, c in enumerate(confs)])
    rep.section("per basin")
    rep.table(["basin", "conformer", "g'", "mirror partner", "source"],
              [[r["index"], r["conformer"], r["g_prime"],
                r["mirror_partner"] if r["mirror_partner"] >= 0 else "-",
                r["g_prime_source"]] for r in rows])
    if duplicate:
        rep.warn("basins with the same core structure were kept apart by branch A: {}; "
                 "they would be counted twice".format(sorted(duplicate.items())))
    rep.note("g' counts RMSD-distinct core structures inside one conformer's rotamer group "
             "with the rotor-group atoms excluded; a methyl rotation is never a state, a "
             "mirror image always is. Both mirror images must have been sampled by CREST; "
             "'unsampled' means the group has one member and g' = 1 is a floor, not a finding.")
    rep.write(lvl / "degeneracy.out", step=STEP)
    return dict(level=str(level), crest_dir=str(crest_dir), conformers=confs, basins=rows,
                excluded_atoms=excluded, record=lvl / "degeneracy.toml")
