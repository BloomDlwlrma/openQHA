"""Turn a **CREST ensemble** into **a basin list under this repository's criteria**.

--------------------------------------------------------------------------------------
Why this layer has to exist
--------------------------------------------------------------------------------------
The "number of conformers" CREST reports is **not** the "number of basins" in this
repository's sense. Measured 2026-08-30 (defect 54):

    acetone: CREST reports 2 conformers, 0.8118 kcal/mol apart;
    after each is tightened by this repository's criterion (`fmax = 1e-4 eV/A`) the two
    energies are **identical to the last digit** (-5259.524313 eV), which is exactly
    package 2's lowest ETKDG basin.
    That 0.8118 kcal/mol is **CREST's convergence residual**, not an energy difference.

There are two reasons, neither of them a fault of CREST; **the criteria simply differ**:

1. **Different convergence criterion.** CREST's `optlev = "tight"` is looser than package
   2's `fmax = 1e-4 eV/A`. A structure not converged all the way is still a few tenths of
   a kcal/mol above the minimum in energy.
2. **Different deduplication criterion.** CREST's `cregen` uses a triple criterion of
   energy window, rotational constants and root-mean-square deviation; this repository
   uses the **all-atom best root-mean-square deviation, minimised over the
   automorphisms**. The two are not equivalent.

There is a third reason, and it is **physical**: CREST's optimisation only guarantees a
small gradient, **not that the point is a minimum** -- a saddle point has zero gradient
too. Package 2 has already caught one case of this (a cyclopropanol "basin" with one
imaginary frequency at -195.79 cm^-1).

**So the rule is**: any ensemble arriving from an external program must be **tightened by
this repository's own `fmax` and then put through a Hessian** before its energies are
used. This module is the implementation of that rule.

--------------------------------------------------------------------------------------
The division of labour between this module and `openqha/conformers.py`
--------------------------------------------------------------------------------------
`conformers.py` takes the ETKDG route (embed -> force-field pruning -> optimise ->
deduplicate). This module takes the CREST route, **starting from "a pile of geometries
already exists"**, but **everything downstream calls the very same functions in
`conformers.py`** (`optimise`, `dedup`, `all_atom_best_rms`, `basin_populations`), so the
basin lists of the two routes are **measured with the same ruler** and can be compared
directly.

--------------------------------------------------------------------------------------
Symmetry number and electronic degeneracy
--------------------------------------------------------------------------------------
A free energy needs the symmetry number sigma and the electronic degeneracy g0, and this
repository requires that they be **declared explicitly and never derived automatically**.
So: **only species declared in the configuration get a free energy**; every other molecule
stops at "basin list plus Boltzmann weights" (which is precisely package 1's product
shape). The Hessian's **minimum-or-not verdict needs no sigma**, so it is done for every
molecule.
"""
import json
import tempfile
from pathlib import Path

import numpy as np

from . import conformers
from ..thermochem import hessian

#: Default Hessian instrument. "analytic" since 2026-09-03 -- defect 64 / D0-P1-47:
#: finite differences at delta = 0.01 A resolve only to +-9 to 20 cm^-1, while the
#: modes being condemned as imaginary had absolute values of 9 to 24 cm^-1. The
#: criterion was being applied below the resolution of the instrument. The analytic
#: path is also 1.8x cheaper, so nothing is traded away. See openqha/hessian.py.
HESSIAN_MODE_DEFAULT = "analytic"

#: This module's own defaults all come from configs/openqha.yaml and are not
#: repeated here. These two names exist only so a caller can read off "which configuration
#: entry is used when nothing is passed".
FMAX_KEY = ("package2", "fmax_hessian_eV_A")
DEDUP_KEY = ("package1", "dedup_rmsd_A")


# ======================================================================================
# 1. Read the ensemble into an RDKit molecule whose **atom order matches the template**
# ======================================================================================
def read_ensemble_atoms(path):
    """Read a CREST multi-frame xyz, returning (a list of ase.Atoms, a list of comment
    lines).

    CREST puts the energy (in Hartree) in the comment line; it is kept verbatim and not
    parsed into a number here -- parsing is the caller's job and is **for diagnostics
    only**. Every conclusion in this module rests on **recomputed** energies.
    """
    from ase import Atoms
    frames, comments = [], []
    for comment, rows in _read_raw(path):
        frames.append(Atoms(symbols=[r[0] for r in rows],
                            positions=[[r[1], r[2], r[3]] for r in rows]))
        comments.append(comment)
    return frames, comments


def _read_raw(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    out, i = [], 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i].split()[0])
        rows = []
        for k in range(n):
            f = lines[i + 2 + k].split()
            rows.append((f[0], float(f[1]), float(f[2]), float(f[3])))
        out.append((lines[i + 1], rows))
        i += n + 2
    return out


def frames_into_mol(smiles, frames):
    """Build a template molecule, add every frame to it as a conformer, **and check the
    atom order**.

    CREST does not reorder atoms, so the order **should** match the input -- but "should"
    is not a criterion. The correspondence is measured once on frame 0 by **graph
    isomorphism** (`conformers.match_order_by_graph`, which does not look at coordinates);
    if it is not the identity permutation, the permutation is applied to every frame.

    **Why geometry cannot be trusted here**: this project has been caught twice by
    geometric assignment (defects 26 and 30), the worst of which swapped acetamide's oxygen
    and nitrogen and reported 0.0024 A as 1.0693 A.

    Returns (mol, list of conformer ids, a record of the check).
    """
    from rdkit import Chem

    mol, repair = conformers.mol_from_smiles(smiles)
    mol = Chem.AddHs(mol)
    n_template = mol.GetNumAtoms()
    if not frames:
        raise ValueError("the ensemble is empty")
    if len(frames[0]) != n_template:
        raise ValueError("the ensemble has {} atoms per frame and the SMILES template has "
                         "{} -- not the same molecule".format(
                             len(frames[0]), n_template))

    syms0 = frames[0].get_chemical_symbols()
    for k, fr in enumerate(frames[1:], start=1):
        if fr.get_chemical_symbols() != syms0:
            raise ValueError("the element order of frame {} differs from frame 0 -- the "
                             "ensemble is not self-consistent".format(k))

    with tempfile.TemporaryDirectory() as td:
        probe = Path(td) / "frame0.xyz"
        conformers.write_xyz(frames[0], probe, "frame 0 for atom-order check")
        _, match = conformers.match_order_by_graph(mol, probe)

    identity = list(match) == list(range(n_template))
    perm = list(match)

    cids = []
    for fr in frames:
        cid = _add_conformer(mol, np.asarray(fr.get_positions())[perm])
        cids.append(cid)

    info = dict(n_frames=len(frames), n_atoms=n_template,
                valence_repair=repair,
                atom_order_is_identity=bool(identity),
                atom_order_match=perm,
                how="graph isomorphism (rdDetermineBonds.DetermineConnectivity plus "
                    "substructure matching), which **does not look at coordinates** -- "
                    "geometric assignment has been wrong twice in this project "
                    "(defects 26 and 30)")
    return mol, cids, info


def _add_conformer(mol, positions):
    from rdkit.Chem import rdchem
    from rdkit.Geometry import Point3D
    conf = rdchem.Conformer(mol.GetNumAtoms())
    for i, p in enumerate(positions):
        conf.SetAtomPosition(i, Point3D(float(p[0]), float(p[1]), float(p[2])))
    return int(mol.AddConformer(conf, assignId=True))


# ======================================================================================
# 2. Tighten, deduplicate, Hessian -- the basin list under this repository's criteria
# ======================================================================================
def census_from_frames(smiles, frames, calc, name="", fmax=1e-4, threshold_A=0.30,
                       temperature_K=298.15, do_hessian=True,
                       reject_imaginary=True, species=None, comments=None,
                       max_opt_steps=2000, progress=None,
                       hessian_mode=HESSIAN_MODE_DEFAULT,
                       ethr_kcal=conformers.CREGEN_ETHR_KCAL,
                       bthr_rel=conformers.CREGEN_BTHR_REL,
                       molecule_dir=None):
    """From a pile of geometries, produce a basin list **measured with the same ruler as
    the ETKDG route**.

    `molecule_dir` (ADR 0001, 2026-09-14): when given, MACE's engine files go under its
    `mace/` folder and nothing else is written there --

        mace/confNN/   opt.traj  opt.log  conf.extxyz     every tightened conformer
        mace/basinNN/  basin.extxyz  hessian.npy          every surviving basin

    `conf.extxyz` carries energy and forces as ASE writes them; `basin.extxyz` names the
    conformer it came from (`conformer`, `crest_comment` in its header); `hessian.npy` is
    the raw analytic Hessian (3N x 3N, eV/A^2, float64), neither mass-weighted nor
    projected, so the projection can be redone from the matrix.

    The steps, each leaving a number that can be checked:

    1. **Tighten** to `fmax` (package 2's 1e-4 eV/A by default, tighter than CREST's
       `tight`); the number of optimisation steps is recorded at the same time -- **the
       step count is itself a measure of how far the geometry CREST handed over was from
       the minimum**.
    2. **Compare the connectivity matrix before and after** -- a structure that reacted
       during the optimisation must be seen, and may not enter the sum silently.
    3. **Deduplicate**: all-atom best root-mean-square deviation (minimised over the
       automorphisms), at package 1's threshold of 0.30 A. It calls **the very same
       function** `conformers.dedup`, so the two routes stay comparable.
    4. **Hessian**: one finite-difference Hessian plus Eckart projection per surviving
       basin, counting imaginary frequencies. With `reject_imaginary=True` a saddle point
       is **thrown out** of the basin list (package 2's practice) and recorded separately
       -- thrown out, not accommodated by relaxing the criterion.
    5. If `species` supplies sigma and g0, the four-term free energy is computed as well;
       **without them nothing is computed**, and nothing is guessed.

    Returns (a record dict, a list of ase.Atoms for the surviving basins, mol).
    """
    mol, cids, order_info = frames_into_mol(smiles, frames)

    from ..store import layout

    energies, fmaxes, converged, steps, graph_changed = [], [], [], [], []
    for j, cid in enumerate(cids):
        atoms = conformers._mol_to_atoms(mol, cid)
        a0 = conformers.connectivity(atoms.numbers, atoms.positions)
        traj = log = None
        if molecule_dir is not None:
            d = layout.mace_conformer_dir(molecule_dir, j)
            d.mkdir(parents=True, exist_ok=True)
            traj, log = str(d / "opt.traj"), str(d / "opt.log")
        e, fm, ok, ns = conformers.optimise(atoms, calc, fmax=fmax, steps=max_opt_steps,
                                            logfile=log, trajectory=traj)
        if molecule_dir is not None:
            _write_extxyz(layout.mace_conformer_dir(molecule_dir, j) / "conf.extxyz", atoms,
                          conformer=j, crest_comment=(comments[j] if comments else ""))
        a1 = conformers.connectivity(atoms.numbers, atoms.positions)
        conformers._set_conf(mol, cid, atoms.positions)
        energies.append(e); fmaxes.append(fm); converged.append(ok); steps.append(ns)
        graph_changed.append(bool(not np.array_equal(a0, a1)))
        if progress is not None:
            progress("tighten", j + 1, len(cids))

    # ---- the energies CREST reports itself, for diagnostics only --------------------
    crest_rel = _crest_relative_kcal(comments) if comments else None

    blocked = {}
    kept, mapping, merge_warn = conformers.dedup(
        mol, cids, energies, threshold_A=threshold_A,
        ethr_kcal=ethr_kcal, bthr_rel=bthr_rel, blocked=blocked)
    e = np.asarray(energies)
    kept = sorted(kept, key=lambda c: e[cids.index(c)])

    # ---- Hessian: once per basin, to decide whether it really is a minimum ----------
    hess = {}
    raw_h = {}
    saddles = []
    if do_hessian:
        for j, cid in enumerate(list(kept)):
            atoms = conformers._mol_to_atoms(mol, cid)
            h, asym = hessian.hessian(atoms, calc, mode=hessian_mode)
            raw_h[int(cid)] = np.asarray(h, dtype=float)
            hr = hessian.project_and_diagonalise(h, atoms.get_masses(),
                                                 atoms.get_positions())
            nu = np.asarray(hr["frequencies_cm_inv"], dtype=float)
            rec_h = dict(n_imaginary=int(hr["n_imaginary"]),
                         hessian_mode=hessian_mode,
                         n_rigid_modes_removed=int(hr["n_rigid_modes_removed"]),
                         separation_gap_ratio=hr.get("separation_gap_ratio"),
                         hessian_asymmetry_eV_A2=float(asym),
                         lowest_frequency_cm_inv=float(nu.min()),
                         frequencies_cm_inv=[float(x) for x in nu])
            hess[int(cid)] = rec_h
            if hr["n_imaginary"] > 0 and reject_imaginary:
                saddles.append(dict(conformer_id=int(cid),
                                    n_imaginary=int(hr["n_imaginary"]),
                                    lowest_frequency_cm_inv=float(nu.min()),
                                    energy_eV=float(e[cids.index(cid)])))
                kept.remove(cid)
            if progress is not None:
                progress("hessian", j + 1, len(hess) + len(saddles))

    if not kept:
        raise RuntimeError("no basin survives the tightening and the imaginary-frequency "
                           "filter -- refusing to report an empty basin list")

    if molecule_dir is not None:
        # Basin i is kept[i]: the survivors in ascending energy, saddles removed. The
        # folder index is the basin index every later step (branch B, 02c, 02d) uses.
        for i, cid in enumerate(kept):
            d = layout.mace_basin_dir(molecule_dir, i)
            d.mkdir(parents=True, exist_ok=True)
            j = cids.index(cid)
            atoms = conformers._mol_to_atoms(mol, cid)
            atoms.calc = calc
            _write_extxyz(d / "basin.extxyz", atoms, conformer=j,
                          crest_comment=(comments[j] if comments else ""),
                          basin=i, energy_eV=float(e[j]))
            if int(cid) in raw_h:
                tmp = d / "hessian.part.npy"
                with open(tmp, "wb") as fh:
                    np.save(fh, raw_h[int(cid)])
                tmp.replace(d / "hessian.npy")

    e_kept = np.asarray([e[cids.index(c)] for c in kept])
    rel = (e_kept - e_kept.min()) * conformers.EV_TO_KCAL

    rec = dict(
        name=name, smiles=smiles, source="CREST ensemble",
        atom_order_check=order_info,
        n_frames_in=len(cids),
        n_not_converged=int(sum(1 for c in converged if not c)),
        n_graph_changed=int(sum(graph_changed)),
        max_residual_force_eV_A=float(max(fmaxes)),
        fmax_criterion_eV_A=float(fmax),
        opt_steps_per_frame=[int(s) for s in steps],
        opt_steps_total=int(sum(steps)),
        tightened_energies_eV=[float(x) for x in e],
        crest_relative_kcal=crest_rel,
        dedup_threshold_A=float(threshold_A),
        dedup_criterion=("cregen_three_fold" if (ethr_kcal is not None
                                                 and bthr_rel is not None)
                         else "rmsd_only"),
        dedup_ethr_kcal=ethr_kcal,
        dedup_bthr_relative=bthr_rel,
        n_merges_blocked_by_energy=int(blocked.get("energy", 0)),
        n_merges_blocked_by_rotational=int(blocked.get("rotational", 0)),
        dedup_blocked_note=("How many merges each extra CREGEN condition prevented. "
                            "Both were zero on OCCC(=O)CO and OCCO at 0.125 A -- the "
                            "first non-zero value is a molecule to look at, not a log "
                            "line."),
        dedup_metric="all-atom best root-mean-square deviation (minimised over the "
                     "molecular automorphisms) -- the same function as the ETKDG route",
        duplicate_map={int(k): int(v) for k, v in mapping.items()},
        merge_energy_warnings=merge_warn,
        n_saddles_rejected=len(saddles), saddles=saddles,
        n_basins=len(kept),
        basin_conformer_ids=[int(c) for c in kept],
        basin_energies_eV=[float(x) for x in e_kept],
        basin_relative_kcal=[float(x) for x in rel],
        populations=conformers.basin_populations(rel, temperature_K),
        hessian={str(k): v for k, v in hess.items()},
        temperature_K=float(temperature_K))

    # ---- conformer count CREST reports vs basin count under our criteria ------------
    rec["crest_vs_repo"] = dict(
        n_conformers_reported_by_crest=len(cids),
        n_basins_by_repo_criteria=len(kept),
        n_collapsed=len(cids) - len(kept),
        max_tightening_spread_kcal=float(
            (e.max() - e.min()) * conformers.EV_TO_KCAL) if len(e) > 1 else 0.0,
        max_spread_among_survivors_kcal=float(rel.max()) if len(rel) else 0.0,
        note="**the conformer count CREST reports is not a basin count** -- both the "
             "convergence criterion and the deduplication criterion differ. Defect 54: on "
             "acetone CREST reported 2, and after tightening the energies were identical "
             "to the last digit.")

    if rec["n_graph_changed"]:
        rec["graph_change_warning"] = (
            "{} frame(s) had their connectivity matrix change during tightening -- they "
            "are not conformers of the same molecule and must be inspected one by "
            "one".format(rec["n_graph_changed"]))

    # ---- free energy: only for species whose sigma and g0 are explicitly declared ----
    if species and do_hessian:
        from ..thermochem import thermo
        gs = []
        for c in kept:
            atoms = conformers._mol_to_atoms(mol, c)
            nu = np.asarray(hess[int(c)]["frequencies_cm_inv"], dtype=float)
            g = thermo.g_minus_eel(atoms.get_masses(), atoms.get_positions(), nu,
                                   symmetry_number=int(species["symmetry_number"]),
                                   degeneracy=int(species["electronic_degeneracy"]),
                                   temperature_K=temperature_K)
            gs.append(float(g["G_minus_Eel_kcal"]))
        g_tot = e_kept * conformers.EV_TO_KCAL + np.asarray(gs)
        rec["G_minus_Eel_kcal_per_basin"] = gs
        rec["G_relative_kcal"] = [float(x) for x in (g_tot - g_tot.min())]
        rec["symmetry_number"] = int(species["symmetry_number"])
        rec["electronic_degeneracy"] = int(species["electronic_degeneracy"])
        rec["symmetry_note"] = ("sigma and g0 are taken from the explicit declaration in "
                                "the configuration; **this repository never derives a "
                                "symmetry number automatically**")
    elif species is None:
        rec["free_energy_note"] = (
            "no free energy was computed -- this molecule's symmetry number sigma and "
            "electronic degeneracy g0 are not explicitly declared in the configuration. "
            "This repository requires that they never be derived automatically, so only "
            "the basin list and the Boltzmann weights are given here.")

    basins = [conformers._mol_to_atoms(mol, c) for c in kept]
    return rec, basins, mol


def _write_extxyz(path, atoms, **info):
    """One extxyz frame with energy and forces (ASE's own writer) and `info` in its header.

    `atoms.calc` must be the calculator that relaxed it, so the energy and forces ASE
    writes are the ones at this geometry; both are evaluated here if the calculator has
    not already done so at exactly these positions.
    """
    from ase.io import write
    a = atoms.copy()
    a.calc = atoms.calc
    a.get_potential_energy()
    a.get_forces()
    a.info.update({k: (str(v) if isinstance(v, str) else v) for k, v in info.items()})
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    write(str(tmp), a, format="extxyz")
    tmp.replace(path)


def _crest_relative_kcal(comments):
    """The energies in CREST's comment lines (Hartree) -> relative kcal/mol.
    **Diagnostics only.**"""
    vals = []
    for c in comments:
        try:
            vals.append(float(c.split()[0]))
        except (ValueError, IndexError):
            return None
    if not vals:
        return None
    v = np.asarray(vals)
    return [float(x) for x in (v - v.min()) * 627.5094740631]


# ======================================================================================
# 3. Comparing the basins of the two routes -- CREST against ETKDG
# ======================================================================================
def compare_basin_sets(smiles, frames_a, frames_b, label_a="A", label_b="B",
                       threshold_A=0.30, energies_a=None, energies_b=None):
    """Put two sets of **already tightened** basins into the same molecule and pair them
    up one by one.

    The criterion is exactly the one used for deduplication (all-atom best
    root-mean-square deviation below the threshold means the same basin), so "A has one
    that B does not" is a checkable statement rather than an impression formed by looking
    at energies.

    Returns a record: the pairing table, A-only, B-only, and the size of the union.
    """
    mol_a, cids_a, info_a = frames_into_mol(smiles, frames_a)
    perm = info_a["atom_order_match"]
    # the second set uses the same template, added frame by frame (its order is checked by
    # the same graph isomorphism, inside frames_into_mol)
    mol_b, cids_b, info_b = frames_into_mol(smiles, frames_b)
    if info_b["atom_order_match"] != perm:
        # the two ensembles have different atom orders -- allowed, but it must be recorded,
        # because it changes what a cid means
        pass
    merged = mol_a
    map_b = {}
    for cid in cids_b:
        pos = np.array(mol_b.GetConformer(cid).GetPositions())
        map_b[cid] = _add_conformer(merged, pos)

    rms = np.zeros((len(cids_a), len(cids_b)))
    for i, ca in enumerate(cids_a):
        for j, cb in enumerate(cids_b):
            rms[i, j] = conformers.all_atom_best_rms(merged, ca, map_b[cb])

    pairs, used_b = [], set()
    for i in range(len(cids_a)):
        j = int(np.argmin(rms[i]))
        if rms[i, j] < threshold_A and j not in used_b:
            used_b.add(j)
            pairs.append(dict(index_a=i, index_b=j, rms_A=float(rms[i, j])))
    only_a = [i for i in range(len(cids_a))
              if i not in {p["index_a"] for p in pairs}]
    only_b = [j for j in range(len(cids_b)) if j not in used_b]

    out = dict(label_a=label_a, label_b=label_b,
               n_a=len(cids_a), n_b=len(cids_b),
               threshold_A=float(threshold_A),
               rms_matrix_A=[[float(x) for x in row] for row in rms],
               matched=pairs, only_in_a=only_a, only_in_b=only_b,
               n_union=len(cids_a) + len(only_b),
               criterion="all-atom best root-mean-square deviation (minimised over the "
                         "automorphisms) below {} A means the same basin; it is the same "
                         "function used for deduplication".format(threshold_A))
    if energies_a is not None and energies_b is not None:
        ea, eb = np.asarray(energies_a), np.asarray(energies_b)
        for p in pairs:
            p["energy_difference_kcal"] = float(
                (ea[p["index_a"]] - eb[p["index_b"]]) * conformers.EV_TO_KCAL)
        out["max_paired_energy_difference_kcal"] = float(
            max((abs(p["energy_difference_kcal"]) for p in pairs), default=0.0))
        out["lowest_a_minus_lowest_b_kcal"] = float(
            (ea.min() - eb.min()) * conformers.EV_TO_KCAL)
    return out


def conformational_correction_kcal(rel_kcal, temperature_K=298.15):
    """The multi-conformer correction relative to "the lowest basin only":
    -kT ln sum exp(-dG_i/kT).

    The input is energies **relative to the lowest basin** (kcal/mol). The return value is
    always <= 0.
    """
    kt = conformers.KB_KCAL * temperature_K
    rel = np.asarray(rel_kcal, dtype=float)
    return float(-kt * np.log(np.exp(-rel / kt).sum()))


def dump_json(obj, path):
    return conformers.dump_json(obj, Path(path))


# ======================================================================================
# 4. Merging the basins of several repeats onto one **global basin table**
# ======================================================================================
def global_basins(smiles, frame_groups, energy_groups, threshold_A=0.30):
    """Put K groups of (already tightened) basins into the same molecule, deduplicate
    **once globally**, and obtain a global basin table.

    That turns "which basins did repeat k find" into a set of integer indices, so unions,
    intersections and the saturation curve can all be computed exactly rather than judged
    by eye from energies.

    `frame_groups[k]` is group k's list of ase.Atoms and `energy_groups[k]` the
    corresponding energies (eV). Returns (a record dict, a list of ase.Atoms for the global
    basins).
    """
    flat, flat_e, owner = [], [], []
    for k, (fr, en) in enumerate(zip(frame_groups, energy_groups)):
        for a, e in zip(fr, en):
            flat.append(a)
            flat_e.append(float(e))
            owner.append(k)
    mol, cids, info = frames_into_mol(smiles, flat)
    kept, mapping, warn = conformers.dedup(mol, cids, flat_e, threshold_A=threshold_A)
    e = np.asarray(flat_e)
    kept = sorted(kept, key=lambda c: e[cids.index(c)])
    gid = {int(c): i for i, c in enumerate(kept)}
    # which global basin each frame belongs to
    frame_to_global = {}
    for c in cids:
        root = c
        while root in mapping:
            root = mapping[root]
        frame_to_global[int(c)] = gid[int(root)]

    membership = [sorted({frame_to_global[int(c)]
                          for c, o in zip(cids, owner) if o == k})
                  for k in range(len(frame_groups))]
    e_kept = np.asarray([e[cids.index(c)] for c in kept])
    rel = (e_kept - e_kept.min()) * conformers.EV_TO_KCAL
    rec = dict(n_global_basins=len(kept),
               global_energies_eV=[float(x) for x in e_kept],
               global_relative_kcal=[float(x) for x in rel],
               membership_per_group=membership,
               n_frames_pooled=len(flat),
               threshold_A=float(threshold_A),
               merge_energy_warnings=warn,
               atom_order_check=info,
               criterion="global deduplication uses the same function conformers.dedup as "
                         "single-run deduplication, so basin identity is comparable across "
                         "repeats")
    basins = [conformers._mol_to_atoms(mol, c) for c in kept]
    return rec, basins


def saturation_curve(membership, n_global):
    """The saturation curve: how many global basins k repeats cover on average (averaged
    over all orderings of the repeats).

    This answers the production question: **how many runs are enough.**
    """
    import itertools
    k_max = len(membership)
    sets = [set(m) for m in membership]
    curve = []
    perms = list(itertools.permutations(range(k_max)))
    for k in range(1, k_max + 1):
        counts = []
        for p in perms:
            u = set()
            for i in p[:k]:
                u |= sets[i]
            counts.append(len(u))
        counts = np.asarray(counts, dtype=float)
        curve.append(dict(n_repeats=k,
                          mean_basins=float(counts.mean()),
                          min_basins=int(counts.min()),
                          max_basins=int(counts.max()),
                          fraction_of_union=float(counts.mean() / max(n_global, 1)),
                          probability_complete=float((counts == n_global).mean())))
    return curve


# ======================================================================================
# 5. Two pieces the batch workflow needs: tightening alone, and the imaginary-frequency
#    filter alone
# ======================================================================================
# `census_from_frames` binds "tighten -> deduplicate -> Hessian" into one step, which is
# enough for a single species.
# **The batch workflow has to take them apart**: the geometries of the two routes (ETKDG
# and CREST) are tightened separately, merged, deduplicated **once** globally, and only
# then is a Hessian taken of the **deduplicated** basins -- otherwise the same basin has
# its Hessian computed twice, and the Hessian is the only expensive thing at this layer.
def tighten_frames(smiles, frames, calc, fmax=1e-4, max_opt_steps=2000, progress=None):
    """Tighten a pile of geometries by this repository's criterion. Returns (a record dict,
    the tightened ase.Atoms, the energies).

    **The number of optimisation steps is itself a conclusion**: it measures how far the
    supplied geometry was from the minimum of this potential. Structures handed over by
    CREST's `optlev = "tight"` measured 16-103 steps (acetone, defect 54).
    """
    mol, cids, order_info = frames_into_mol(smiles, frames)
    energies, fmaxes, converged, steps, graph_changed = [], [], [], [], []
    for j, cid in enumerate(cids):
        atoms = conformers._mol_to_atoms(mol, cid)
        a0 = conformers.connectivity(atoms.numbers, atoms.positions)
        e, fm, ok, ns = conformers.optimise(atoms, calc, fmax=fmax, steps=max_opt_steps)
        a1 = conformers.connectivity(atoms.numbers, atoms.positions)
        conformers._set_conf(mol, cid, atoms.positions)
        energies.append(float(e)); fmaxes.append(float(fm))
        converged.append(bool(ok)); steps.append(int(ns))
        graph_changed.append(bool(not np.array_equal(a0, a1)))
        if progress is not None:
            progress(j + 1, len(cids))
    rec = dict(n_frames=len(cids), fmax_criterion_eV_A=float(fmax),
               n_not_converged=int(sum(1 for c in converged if not c)),
               n_graph_changed=int(sum(graph_changed)),
               graph_changed_frames=[i for i, g in enumerate(graph_changed) if g],
               max_residual_force_eV_A=float(max(fmaxes)) if fmaxes else None,
               opt_steps_per_frame=steps, opt_steps_total=int(sum(steps)),
               energies_eV=energies, atom_order_check=order_info)
    return rec, [conformers._mol_to_atoms(mol, c) for c in cids], energies


def hessian_screen(basins, calc, reject_imaginary=True, progress=None,
                   mode=HESSIAN_MODE_DEFAULT):
    """One Hessian plus Eckart projection per basin, counting imaginary frequencies.

    Returns (a record per basin, the surviving indices, records of the saddle points thrown
    out).
    **A small gradient is not a minimum** -- a saddle point has zero gradient too. Package 2
    caught a -195.79 cm^-1 cyclopropanol saddle point sitting in a basin list because of
    this.

    `mode` defaults to "analytic" (defect 64 / D0-P1-47). The finite-difference
    path is still reachable, and whichever was used is written into every record
    under `hessian_mode`.
    """
    recs, keep, saddles = [], [], []
    for i, atoms in enumerate(basins):
        h, asym = hessian.hessian(atoms, calc, mode=mode)
        hr = hessian.project_and_diagonalise(h, atoms.get_masses(), atoms.get_positions())
        nu = np.asarray(hr["frequencies_cm_inv"], dtype=float)
        r = dict(index=i, n_imaginary=int(hr["n_imaginary"]),
                 hessian_mode=mode,
                 n_rigid_modes_removed=int(hr["n_rigid_modes_removed"]),
                 separation_gap_ratio=hr.get("separation_gap_ratio"),
                 hessian_asymmetry_eV_A2=float(asym),
                 lowest_frequency_cm_inv=float(nu.min()),
                 frequencies_cm_inv=[float(x) for x in nu])
        recs.append(r)
        if hr["n_imaginary"] == 0 or not reject_imaginary:
            keep.append(i)
        else:
            saddles.append(dict(index=i, n_imaginary=int(hr["n_imaginary"]),
                                lowest_frequency_cm_inv=float(nu.min())))
        if progress is not None:
            progress(i + 1, len(basins))
    return recs, keep, saddles
