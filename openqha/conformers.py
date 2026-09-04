"""Package 1 -- conformer generation and deduplication.

The chain (plan section 4, package 1):

    SMILES --ETKDGv3--> N embeddings --MMFF94, for deduplication ONLY, never for
    ordering--> survivors --MACE-OFF23-SC optimisation--> ordered by **the potential's
    energy** --heavy-atom root-mean-square-deviation deduplication--> basin list

**Why MMFF deduplicates but never orders.** The plan requires that "the same potential
picks the basins and computes the energies". If MMFF energies were used to discard
conformers, the discarded one might be the lowest basin on the MACE surface -- and then
"the same potential" would be a false statement. So MMFF does exactly one thing here: it
merges ETKDG embeddings that are **almost coincident geometrically**, with a very tight
threshold (0.125 A over all atoms), conservative enough that it cannot throw away a real
basin. Ordering and the final deduplication happen on the MACE surface throughout.

**The atom-correspondence problem.** Any measure of "how far apart two conformers are"
must solve the atom correspondence first, or the number is meaningless (this project has
already been caught by it once: greedy nearest-neighbour matching reported a
root-mean-square deviation of 1.6447 A while the model energies differed by only
0.17 kcal/mol, a self-contradiction). What is used here is RDKit's `GetBestRMS`, which
enumerates the molecular automorphisms (permutations of symmetry-equivalent atoms) and
takes the minimum -- the correspondence problem is solved by that.
"""
import json
import math
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ parameter classification
# Conventions that may be changed (changing one changes the number of basins, and the
# consequence must be reported):
N_EMBED_DEFAULT = 50            # number of embedded conformers. Too few and basins are missed
PRUNE_RMS_EMBED = 0.125         # A, **all-atom** geometric deduplication after the force
                                # field, only to save MACE force calls. NOTE (2026-09-04):
                                # this was described here as "deliberately tight". It is
                                # not tight -- 0.125 A is CREST's PRODUCTION threshold
                                # (RTHR). Only the value's description was wrong; the
                                # value stands, and this path is retired anyway.

# ---- the deduplication criterion: CREST's CREGEN, all three conditions ---------------
#
# Two structures are the same basin only if ALL of:
#
#     all-atom best RMSD  <  DEDUP_RMSD_A     0.125 A   (CREST RTHR)
#     |dE|                <  CREGEN_ETHR_KCAL 0.05 kcal/mol (CREST ETHR)
#     rotational constants agree to CREGEN_BTHR_REL  1%  (CREST BTHR)
#
# 2026-09-04: DEDUP_RMSD_A was 0.30 A, with **no source anywhere in the repository** --
# the config claimed "this package carries its own threshold sweep" and the sweep had
# never been run. It has now been run, on two molecules (branch A checkpoint 5):
#
#     threshold / A     OCCC(=O)CO basins     OCCO basins
#     0.05 - 0.25             23                  10       <- flat plateau
#     0.125 (CREST RTHR)      23                  10       <- mid-plateau
#     0.30  (the old value)   22                   9       <- first value off the edge
#     0.40                    17                   7
#
# **0.30 A sat on a slope, not a plateau.** The two merges it bought were
# 0.2786 A / 0.3769 kcal/mol and 0.2937 A / 2.1986 kcal/mol -- the second merged two
# structures 2.2 kcal/mol apart into one basin.
#
# The energy and rotational conditions cost NOTHING on the data measured so far: at
# 0.125 A the three-fold criterion returns exactly the plateau count on both molecules.
# They are free guards against the failure mode above on molecules not yet run. Because
# they have never yet changed an outcome, `dedup` counts how often each of them blocks
# a merge, and **the first non-zero count is a molecule to look at, not a log line.**
DEDUP_RMSD_A = 0.125            # A, CREST RTHR
CREGEN_ETHR_KCAL = 0.05         # kcal/mol, CREST ETHR
CREGEN_BTHR_REL = 0.01          # relative, CREST BTHR lower bound (upstream adjusts it
                                # dynamically up to 2.5%; we take the lower bound, which
                                # is the stricter -- i.e. merges less -- choice)
ENERGY_WINDOW_KT = 5.0          # both the k_BT and the 5 k_BT window are reported
# Merge-consistency criterion: if two conformers judged to be the same basin differ in
# energy by more than this, the deduplication has merged two different minima -> warn.
# This is not an adjustable knob; it only diagnoses and changes no merge result.
MERGE_ENERGY_WARN_KCAL = 0.05

# Numerical tolerances:
FMAX_CENSUS_EV_A = 1.0e-3       # for the census: only has to land in the right basin
FMAX_HESSIAN_EV_A = 1.0e-4      # for the finite-difference Hessian: a residual force
                                # contaminates the curvature
MAX_OPT_STEPS = 500

# Physical constants
KB_KCAL = 1.987204259e-3        # kcal/(mol*K)
EV_TO_KCAL = 23.060547830618307
T_REF = 298.15


def _rdkit():
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolAlign
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    return Chem, AllChem, rdMolAlign


def embed(smiles, n_embed=N_EMBED_DEFAULT, seed=20260827,
          prune_rms=PRUNE_RMS_EMBED, mmff_prune=True):
    """SMILES -> a batch of candidate conformers that do not coincide geometrically
    (an RDKit Mol carrying several conformers).

    **The ETKDG stage deliberately does no deduplication at all** (`pruneRmsThresh = -1`).
    RDKit's deduplication during embedding uses the **heavy-atom** root-mean-square
    deviation, so every rotor that "only moves hydrogens" -- hydroxyl, amino, methyl --
    looks like a duplicate conformer to it and is deleted on the spot. The measured
    consequence in this project: cyclopropanol, acetamide and acetone asked for 50
    embeddings and got **1** back, while the cis/trans hydroxyl is precisely the softest
    degree of freedom in those molecules and the one carrying the most free-energy weight.

    Deduplication is therefore deferred until after the force field and switched to the
    **all-atom** best root-mean-square deviation (with the equivalent permutations of
    methyl hydrogens solved by the molecular automorphisms), so an equivalent methyl
    rotation is still merged away while a cis/trans hydroxyl is not.
    """
    Chem, AllChem, _ = _rdkit()
    mol, rec_repair = mol_from_smiles(smiles)
    if Chem.GetFormalCharge(mol) != 0 or any(a.GetNumRadicalElectrons()
                                             for a in mol.GetAtoms()):
        # MACE-OFF23-SC has no charge input and has never seen an open shell -- it cannot
        # represent either kind of species. Normally F3/F4 in openqha.filters have already
        # kept them out; this is the last line of defence.
        raise ValueError(
            "{} carries a net charge or unpaired electrons -- this potential handles only "
            "neutral closed-shell species, refusing to run. It should have been removed "
            "earlier by the F3/F4 gates of openqha.filters".format(smiles))
    mol = Chem.AddHs(mol)
    cids, recipe, embed_fallbacks = _embed_with_fallback(mol, n_embed, seed, AllChem)
    rec = dict(smiles=smiles, n_embed_requested=int(n_embed),
               n_embed_returned=len(cids), embed_seed=int(seed),
               embed_recipe=recipe, embed_fallbacks=embed_fallbacks,
               prune_rms_embed_A=float(prune_rms),
               embed_stage_pruning="disabled (pruneRmsThresh = -1)",
               valence_repair=rec_repair,
               forcefield=None, forcefield_converged=None)
    if not cids:
        raise RuntimeError("ETKDG embedded no conformer at all: {}".format(smiles))
    if mmff_prune:
        # The force field is only here to save MACE force evaluations. Its failure **must
        # not** fail the whole molecule -- measured: on cyclopentane (`C1CCCC1`) RDKit's
        # BFGS raises `Invariant Violation: bad direction in linearSearch`. Optimisation on
        # the potential converges just as well starting from an embedding the force field
        # never touched, only more slowly. Degrade and record; do not lose the molecule.
        res = []
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
                rec["forcefield"] = "MMFF94"
            elif AllChem.UFFHasAllMoleculeParams(mol):
                res = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000)
                rec["forcefield"] = "UFF"
            else:
                rec["forcefield"] = "none (no parameters, sent straight to MACE)"
        except RuntimeError as exc:
            res = []
            rec["forcefield"] = "failed (sent straight to MACE)"
            rec["forcefield_error"] = "{}: {}".format(
                type(exc).__name__, " ".join(str(exc).split()))
        rec["forcefield_converged"] = (sum(1 for c, _ in res if c == 0) if res else None)
        rec["forcefield_note"] = ("the force field is used only to merge embeddings that "
                                  "are almost coincident geometrically, and **takes no "
                                  "part in any ordering or selection**")
        keep = _geometric_prune(mol, cids, prune_rms)
        rec["n_after_forcefield_prune"] = len(keep)
        for c in cids:
            if c not in keep:
                mol.RemoveConformer(c)
    return mol, rec


def mol_from_smiles(smiles):
    """SMILES -> an RDKit molecule. Raises if it will not parse, and **repairs no
    valences**.

    Returns (mol, None). The second return value is a position left over from history; it
    is always None and is kept only so that callers need not change.

    > **There used to be a block of code here that "assigned formal charges uniquely from
    > valence"** (`D0-38`): a group of zwitterionic SMILES in QM9 carry no charges (e.g.
    > `[NH3]CCC(=O)[O]`), RDKit refuses to parse them, and I repaired them to
    > `[NH3+]CCC(=O)[O-]` and thereby rescued them.
    > **That was wrong.** What the repair produced was exactly a **zwitterion**, and this
    > project's species scope (the user's ruling of 2026-08-28, with the same meaning as
    > stage 1's gate F5) **explicitly excludes zwitterions**. The repair let in a group of
    > molecules that should have been rejected.
    > Such molecules are now removed by F0/F5 in `openqha/filters.py` **before entering any
    > calculation**, with an attribution recorded for each. See `D0-41`.
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(
            "SMILES will not parse: {} -- this package repairs no valences. "
            "Molecules like this should have been removed at the F0/F5 gates of "
            "openqha.filters; reaching here means the filtering is not enabled or was "
            "never called.".format(smiles))
    return mol, None


def _embed_with_fallback(mol, n_embed, seed, AllChem):
    """ETKDG embedding, with two levels of fallback.

    **Measured**: on cyclopentane (`C1CCCC1`) `EmbedMultipleConfs` raises
    `RuntimeError: Invariant Violation / bad direction in linearSearch` outright (RDKit's
    distance-geometry stage uses BFGS, whose line search fails on this molecule). That is
    not a problem with our input, it is a numerical failure upstream -- changing how the
    initial coordinates are produced steps around it. The fallbacks are ordered from the
    smallest change to the largest, and whichever was used is always recorded in the
    product.
    """
    recipes = [
        ("ETKDGv3 + useSmallRingTorsions", True, False),
        ("ETKDGv3 + useRandomCoords", True, True),
        ("ETKDGv3 without useSmallRingTorsions + useRandomCoords", False, True),
    ]
    fallbacks = []
    for label, small_ring, random_coords in recipes:
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        params.pruneRmsThresh = -1.0        # never deduplicate during embedding
        params.useSmallRingTorsions = bool(small_ring)
        params.useRandomCoords = bool(random_coords)
        try:
            cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(n_embed),
                                                   params=params))
        except RuntimeError as exc:
            fallbacks.append(dict(recipe=label, error="{}: {}".format(
                type(exc).__name__, " ".join(str(exc).split())[:200])))
            continue
        if cids:
            return cids, label, fallbacks
        fallbacks.append(dict(recipe=label, error="returned 0 conformers"))
    raise RuntimeError("all three embedding recipes failed: {}".format(fallbacks))


def _geometric_prune(mol, cids, threshold_A):
    """All-atom geometric deduplication after the force field. Purely to save the later
    MACE force evaluations, with a deliberately tight threshold."""
    keep = []
    for c in cids:
        if all(all_atom_best_rms(mol, c, k) >= threshold_A for k in keep):
            keep.append(c)
    return keep


def connectivity(numbers, positions, scale=1.3):
    """Connectivity matrix from a covalent-radius x scale distance criterion. Used to
    detect "it reacted during the optimisation"."""
    from ase.data import covalent_radii
    r = np.asarray([covalent_radii[z] for z in numbers])
    d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    cut = scale * (r[:, None] + r[None, :])
    a = (d < cut).astype(np.int8)
    np.fill_diagonal(a, 0)
    return a


def optimise(atoms, calc, fmax=FMAX_CENSUS_EV_A, steps=MAX_OPT_STEPS, logfile=None):
    """Relax one conformer to a minimum on the potential surface. Returns
    (energy in eV, maximum force in eV/A, converged, number of steps)."""
    from ase.optimize import LBFGS
    atoms.calc = calc
    opt = LBFGS(atoms, logfile=logfile)
    opt.run(fmax=float(fmax), steps=int(steps))
    f = atoms.get_forces()
    fmax_final = float(np.abs(f).max())
    return (float(atoms.get_potential_energy()), fmax_final,
            bool(fmax_final <= fmax), int(opt.get_number_of_steps()))


def _mol_to_atoms(mol, conf_id):
    from ase import Atoms
    conf = mol.GetConformer(conf_id)
    pos = np.array(conf.GetPositions(), dtype=float)
    z = [a.GetAtomicNum() for a in mol.GetAtoms()]
    return Atoms(numbers=z, positions=pos)


def _set_conf(mol, conf_id, positions):
    from rdkit.Geometry import Point3D
    conf = mol.GetConformer(conf_id)
    for i, p in enumerate(positions):
        conf.SetAtomPosition(i, Point3D(float(p[0]), float(p[1]), float(p[2])))


def heavy_atom_best_rms(mol, cid_a, cid_b):
    """Heavy-atom best root-mean-square deviation (A). RDKit enumerates the molecular
    automorphisms and takes the minimum -- that is what solves the correspondence."""
    Chem, _, rdMolAlign = _rdkit()
    probe = Chem.RemoveHs(Chem.Mol(mol))
    return float(rdMolAlign.GetBestRMS(probe, probe, prbId=cid_a, refId=cid_b))


def all_atom_best_rms(mol, cid_a, cid_b):
    """**All-atom** best root-mean-square deviation (A), hydrogens included.

    This is the measure by which this package decides "the same basin". Plan section 4,
    package 1 says "heavy-atom root-mean-square deviation" in its own words; switching to
    all atoms is **a deviation with measured grounds**: the heavy-atom measure is blind to
    rotors that move only hydrogens (cis/trans hydroxyl, amino inversion) and merges two
    minima of clearly different energy into one. Automorphism enumeration still cancels the
    equivalent permutations of methyl hydrogens, so a methyl rotation is not mistaken for a
    new basin.
    """
    _, _, rdMolAlign = _rdkit()
    return float(rdMolAlign.GetBestRMS(mol, mol, prbId=cid_a, refId=cid_b))


def rotational_constants(masses, positions):
    """Rotational constants, up to a common factor. Sorted, so axis labels do not matter.

    B_i is proportional to 1 / I_i and the constant cancels in the ratio test used by
    the deduplication, so no unit conversion is needed here.
    """
    from . import thermo
    moments = thermo.principal_moments(np.asarray(masses, dtype=float),
                                       np.asarray(positions, dtype=float))
    moments = np.asarray(sorted(float(m) for m in moments), dtype=float)
    with np.errstate(divide="ignore"):
        return 1.0 / np.where(moments > 1e-12, moments, np.inf)


def dedup(mol, cids, energies_eV, threshold_A=DEDUP_RMSD_A,
          ethr_kcal=CREGEN_ETHR_KCAL, bthr_rel=CREGEN_BTHR_REL, blocked=None):
    """Greedy deduplication in ascending energy, using CREST's CREGEN criterion.

    Two conformers are the same basin only if ALL of the enabled conditions hold:

        all-atom best RMSD  <  threshold_A     (CREST RTHR, 0.125 A)
        |dE|                <  ethr_kcal       (CREST ETHR, 0.05 kcal/mol)
        rotational constants agree to bthr_rel (CREST BTHR, 1%)

    Passing `ethr_kcal=None` and `bthr_rel=None` restores the RMSD-only rule this repo
    used before 2026-09-04, so the old behaviour stays reproducible rather than being
    written out of existence.

    `blocked`, if a dict is passed, receives per-condition counts of how many merges each
    extra condition prevented. Those counts belong in the product: the energy and
    rotational tests have never yet changed an outcome on the molecules measured, so the
    first time one of them fires is a molecule that needs looking at.

    Returns (kept conformer ids, assignment table, consistency warnings). A warning is
    emitted when a merge that DID happen crosses `MERGE_ENERGY_WARN_KCAL`; with the
    energy condition enabled that should now be impossible, and if it ever appears the
    two thresholds have been set inconsistently.
    """
    order = list(np.argsort(np.asarray(energies_eV)))
    e = np.asarray(energies_eV)
    kept, mapping, warn = [], {}, []
    counts = dict(energy=0, rotational=0) if blocked is None else blocked
    counts.setdefault("energy", 0)
    counts.setdefault("rotational", 0)

    for k in order:
        cid = cids[k]
        dup_of = None
        for kcid in kept:
            if all_atom_best_rms(mol, cid, kcid) >= threshold_A:
                continue
            kk = cids.index(kcid)
            de = abs(e[k] - e[kk]) * EV_TO_KCAL
            if ethr_kcal is not None and de >= ethr_kcal:
                counts["energy"] += 1
                continue
            if bthr_rel is not None:
                b1 = rotational_constants(_mol_to_atoms(mol, cid).get_masses(),
                                          _mol_to_atoms(mol, cid).get_positions())
                b2 = rotational_constants(_mol_to_atoms(mol, kcid).get_masses(),
                                          _mol_to_atoms(mol, kcid).get_positions())
                rel = np.abs(b1 - b2) / np.maximum(np.abs(b2), 1e-30)
                if float(rel.max()) >= bthr_rel:
                    counts["rotational"] += 1
                    continue
            dup_of = kcid
            break
        if dup_of is None:
            kept.append(cid)
        else:
            mapping[cid] = dup_of
            de = abs(e[k] - e[cids.index(dup_of)]) * EV_TO_KCAL
            if de > MERGE_ENERGY_WARN_KCAL:
                warn.append(dict(merged=int(cid), into=int(dup_of),
                                 energy_difference_kcal=float(de),
                                 all_atom_rms_A=float(all_atom_best_rms(mol, cid, dup_of)),
                                 note=("this merge crossed the energy warning threshold "
                                       "even though the energy CONDITION was enabled -- "
                                       "the two are set inconsistently")))
    return kept, mapping, warn


def dedup_threshold_scan(mol, cids, energies_eV,
                         thresholds=(0.10, 0.15, 0.20, 0.25, 0.30, 0.40,
                                     0.50, 0.60, 0.80, 1.00)):
    """Acceptance 2: sweep the threshold from 0.1 to 1.0 A and report how the basin count
    and the relative energies change.

    **Both measures are swept** -- all-atom (what this package actually uses) and
    heavy-atom (what the plan says in its own words). The difference between the two curves
    IS how many basins the "rotors that move only hydrogens" are worth, measured directly.
    The two paired tables are computed in advance, so the sweep itself costs nothing extra.
    """
    n = len(cids)
    rms_all = np.zeros((n, n))
    rms_heavy = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rms_all[i, j] = rms_all[j, i] = all_atom_best_rms(mol, cids[i], cids[j])
            rms_heavy[i, j] = rms_heavy[j, i] = heavy_atom_best_rms(mol, cids[i], cids[j])
    e = np.asarray(energies_eV)
    order = list(np.argsort(e))

    def count(rms, t):
        kept = []
        for k in order:
            if all(rms[k, m] >= t for m in kept):
                kept.append(k)
        return kept

    out = []
    for t in thresholds:
        ka, kh = count(rms_all, t), count(rms_heavy, t)
        rel = (e[ka] - e[ka].min()) * EV_TO_KCAL
        out.append(dict(threshold_A=float(t),
                        n_basins=len(ka), n_basins_heavy_atom_metric=len(kh),
                        relative_energies_kcal=[float(x) for x in np.sort(rel)]))
    return out, rms_all


def basin_populations(rel_kcal, temperature_K=T_REF):
    """The number of basins within the k_BT and 5 k_BT windows, and the Boltzmann
    weights."""
    kt = KB_KCAL * temperature_K
    rel = np.asarray(rel_kcal, dtype=float)
    w = np.exp(-rel / kt)
    w = w / w.sum()
    return dict(kT_kcal=float(kt),
                n_within_1kT=int((rel <= kt).sum()),
                n_within_5kT=int((rel <= ENERGY_WINDOW_KT * kt).sum()),
                boltzmann_weights=[float(x) for x in w],
                weight_of_lowest=float(w[int(np.argmin(rel))]))


def census(smiles, calc, name="", n_embed=N_EMBED_DEFAULT, seed=20260827,
           fmax=FMAX_CENSUS_EV_A, threshold_A=DEDUP_RMSD_A,
           scan_thresholds=True, reference_xyz=None, temperature_K=T_REF,
           include_reference_as_seed=True):
    """A complete basin census for one species. Returns (a record dict, a list of
    ase.Atoms for the kept basins, the optimised mol).

    When `reference_xyz` is given (the native QM9 B3LYP geometry, say), it plays two roles
    **at once**:

    * **an extra starting point** (`include_reference_as_seed`) -- once relaxed it joins
      the embedded conformers in the deduplication;
    * **a report** -- which basin it lands in (the number acceptance 3 of plan section 4,
      package 1 asks for).

    The first role was forced by measurement: N-methylformamide asked for 50 embeddings and
    missed no basin -- but the native QM9 geometry relaxed into **a different real minimum**
    (0 imaginary frequencies), 0.0429 kcal/mol higher in electronic energy and
    0.1639 kcal/mol lower in the thermodynamic terms, so **lower in free energy overall**.
    ETKDG's torsion library gives only one phase for the nearly free N-methyl rotor, so no
    embedding in the whole batch can reach that basin. Not using a free starting point is
    a waste.
    """
    try:
        mol, rec = embed(smiles, n_embed=n_embed, seed=seed)
    except RuntimeError as exc:
        # All three embedding recipes failed -- measured cases cluster on strained caged
        # fused rings (bicyclobutanes, spiro compounds, fused three-membered rings). But we
        # still hold **a free real geometry**: the native QM9 structure. Using it as the
        # only starting point gives **a degraded census** (no conformer search, one basin
        # only), which is far better than losing the whole molecule. The degradation itself
        # goes into the record, so downstream knows this basin list is incomplete.
        if reference_xyz is None:
            raise
        mol, rec = _census_from_reference_only(smiles, reference_xyz, exc)
        include_reference_as_seed = False
    cids = [c.GetId() for c in mol.GetConformers()]

    energies, fmaxes, converged, steps, graph_changed = [], [], [], [], []
    for cid in cids:
        atoms = _mol_to_atoms(mol, cid)
        a0 = connectivity(atoms.numbers, atoms.positions)
        e, fm, ok, ns = optimise(atoms, calc, fmax=fmax)
        a1 = connectivity(atoms.numbers, atoms.positions)
        _set_conf(mol, cid, atoms.positions)
        energies.append(e); fmaxes.append(fm); converged.append(ok); steps.append(ns)
        graph_changed.append(bool(not np.array_equal(a0, a1)))

    ref_cid = None
    if reference_xyz is not None and include_reference_as_seed:
        from ase import Atoms
        try:
            pos, match = match_order_by_graph(mol, reference_xyz)
        except ValueError as exc:
            # The connectivity graph of the reference geometry is not isomorphic to the
            # SMILES template -- a few QM9 entries have SMILES with unconventional valence
            # (carbenes, nitrenes) whose perceived graph does not line up. **The census
            # itself is still valid**; all that is lost is the "which basin does the
            # reference geometry land in" report. Degrade and record; do not fail entirely.
            reference_xyz = None
            rec["reference_unavailable"] = "{}: {}".format(type(exc).__name__, exc)
        if reference_xyz is None:
            pos = None
    if reference_xyz is not None and include_reference_as_seed:
        ref = Atoms(numbers=[a.GetAtomicNum() for a in mol.GetAtoms()], positions=pos)
        rec["reference_single_point_eV"] = float(_single_point(ref, calc))
        e, fm, ok, ns = optimise(ref, calc, fmax=fmax)
        ref_cid = mol.AddConformer(mol.GetConformer(cids[0]), assignId=True)
        _set_conf(mol, ref_cid, ref.get_positions())
        cids.append(ref_cid)
        energies.append(e); fmaxes.append(fm); converged.append(ok); steps.append(ns)
        graph_changed.append(False)
        rec["reference_seeded"] = True
        rec["reference_atom_order_match"] = list(match)

    kept, mapping, merge_warn = dedup(mol, cids, energies, threshold_A=threshold_A)
    e = np.asarray(energies)
    e_kept = e[[cids.index(c) for c in kept]]
    order = np.argsort(e_kept)
    kept = [kept[i] for i in order]
    e_kept = e_kept[order]
    rel = (e_kept - e_kept.min()) * EV_TO_KCAL

    rec.update(
        name=name,
        n_conformers_optimised=len(cids),
        n_not_converged=int(sum(1 for c in converged if not c)),
        n_graph_changed=int(sum(graph_changed)),
        max_residual_force_eV_A=float(max(fmaxes)),
        opt_steps_total=int(sum(steps)),
        fmax_criterion_eV_A=float(fmax),
        dedup_threshold_A=float(threshold_A),
        n_basins=len(kept),
        basin_energies_eV=[float(x) for x in e_kept],
        basin_relative_kcal=[float(x) for x in rel],
        basin_relative_kT=[float(x / (KB_KCAL * temperature_K)) for x in rel],
        populations=basin_populations(rel, temperature_K),
        duplicate_map={int(k): int(v) for k, v in mapping.items()},
        merge_energy_warnings=merge_warn,
        dedup_metric="all-atom best root-mean-square deviation (minimised over the "
                     "molecular automorphisms)")

    if rec["n_graph_changed"]:
        rec["graph_change_warning"] = (
            "{} conformer(s) had their connectivity matrix change during optimisation -- "
            "they are not conformers of the same molecule and must be inspected one by "
            "one; they may not go straight into the sum".format(rec["n_graph_changed"]))

    if scan_thresholds:
        scan, _ = dedup_threshold_scan(mol, cids, energies)
        rec["dedup_threshold_scan"] = scan

    if reference_xyz is not None:
        if ref_cid is not None:
            rms_all = {int(c): all_atom_best_rms(mol, ref_cid, c) for c in kept}
            rms_h = {int(c): heavy_atom_best_rms(mol, ref_cid, c) for c in kept}
            nearest = min(rms_all, key=rms_all.get)
            e_ref = energies[cids.index(ref_cid)]
            rec["reference_geometry"] = dict(
                path=str(reference_xyz), seeded_into_census=True,
                single_point_on_raw_geometry_eV=rec.get("reference_single_point_eV"),
                relaxed_energy_eV=float(e_ref),
                relaxation_drop_kcal=float(
                    (rec["reference_single_point_eV"] - e_ref) * EV_TO_KCAL),
                rms_to_each_basin_A={str(k): float(v) for k, v in rms_all.items()},
                rms_heavy_to_each_basin_A={str(k): float(v) for k, v in rms_h.items()},
                nearest_basin_conformer_id=int(nearest),
                nearest_basin_rank_zero_based=int(kept.index(nearest)),
                is_own_basin=bool(nearest == ref_cid),
                energy_gap_to_lowest_kcal=float((e_ref - min(energies)) * EV_TO_KCAL),
                note="the reference geometry took part in the deduplication as an extra "
                     "starting point; is_own_basin true means it landed in a basin that no "
                     "ETKDG embedding reached")
        else:
            rec["reference_geometry"] = _locate_reference(
                mol, cids, kept, energies, reference_xyz, calc, fmax, threshold_A)

    basins = [_mol_to_atoms(mol, c) for c in kept]
    return rec, basins, mol


def _census_from_reference_only(smiles, reference_xyz, embed_exc):
    """The degraded path for when ETKDG cannot embed at all: build one conformer using the
    reference geometry as the only starting point."""
    from rdkit import Chem
    from rdkit.Chem import rdchem
    mol, repair = mol_from_smiles(smiles)
    mol = Chem.AddHs(mol)
    pos, match = match_order_by_graph(mol, reference_xyz)
    conf = rdchem.Conformer(mol.GetNumAtoms())
    for i, p in enumerate(pos):
        conf.SetAtomPosition(i, (float(p[0]), float(p[1]), float(p[2])))
    mol.AddConformer(conf, assignId=True)
    rec = dict(smiles=smiles, n_embed_requested=int(0), n_embed_returned=1,
               embed_seed=None, embed_recipe="**degraded**: all ETKDG recipes failed, "
                                             "reference geometry only",
               embed_fallbacks=[], valence_repair=repair,
               embed_stage_pruning="not applicable",
               forcefield=None, forcefield_converged=None,
               n_after_forcefield_prune=1,
               embed_failed="{}: {}".format(type(embed_exc).__name__,
                                            " ".join(str(embed_exc).split())[:300]),
               census_degraded=("no conformer search -- the basin list holds only the basin "
                                "the reference geometry is in; it is **incomplete** and "
                                "must not be used as a complete census"),
               reference_geometry_path=str(reference_xyz),
               reference_atom_order_match=list(match))
    return mol, rec


def _single_point(atoms, calc):
    atoms = atoms.copy()
    atoms.calc = calc
    return atoms.get_potential_energy()


def _locate_reference(mol, cids, kept, energies, reference_xyz, calc, fmax, threshold_A):
    """Relax an outside geometry (the native QM9 B3LYP one) on the potential surface and
    report which basin it lands in.

    The order matters: **align the atom order by graph isomorphism FIRST**, then build the
    ase object and relax, then compare. Relaxing before aligning mixes "different order"
    into "different structure" -- which is exactly how the old implementation got it wrong.
    """
    from ase import Atoms
    pos, match = match_order_by_graph(mol, reference_xyz)
    z = [a.GetAtomicNum() for a in mol.GetAtoms()]
    ref = Atoms(numbers=z, positions=pos)
    ref.calc = calc
    e_ref_raw = float(ref.get_potential_energy())
    e_ref, fm, ok, ns = optimise(ref, calc, fmax=fmax)

    tmp_id = mol.AddConformer(mol.GetConformer(cids[0]), assignId=True)
    _set_conf(mol, tmp_id, ref.get_positions())
    rms_all = {int(c): all_atom_best_rms(mol, tmp_id, c) for c in kept}
    rms_heavy = {int(c): heavy_atom_best_rms(mol, tmp_id, c) for c in kept}
    nearest = min(rms_all, key=rms_all.get)
    rank = kept.index(nearest)
    e_kept = {int(c): energies[cids.index(c)] for c in kept}
    return dict(path=str(reference_xyz),
                atom_order_match=list(match),
                single_point_on_raw_geometry_eV=e_ref_raw,
                relaxed_energy_eV=e_ref, relaxed_fmax_eV_A=fm,
                relaxed_converged=ok, relax_steps=ns,
                relaxation_drop_kcal=float((e_ref_raw - e_ref) * EV_TO_KCAL),
                rms_to_each_basin_A={str(k): float(v) for k, v in rms_all.items()},
                rms_heavy_to_each_basin_A={str(k): float(v) for k, v in rms_heavy.items()},
                nearest_basin_conformer_id=int(nearest),
                nearest_basin_rank_zero_based=int(rank),
                energy_gap_to_nearest_kcal=float(
                    (e_ref - e_kept[nearest]) * EV_TO_KCAL),
                same_basin=bool(rms_all[nearest] < threshold_A),
                note="the outside geometry has its atom order aligned by graph isomorphism "
                     "first, is then relaxed on the same potential, and is finally assigned "
                     "by all-atom best root-mean-square deviation")


def _connectivity_only(m):
    """Reduce an RDKit molecule to "the connectivity graph only": every bond order set to
    single, aromaticity and charges cleared.

    This is what makes it possible to substructure-match the bond-order-free molecule that
    `DetermineConnectivity` produces against the template.
    """
    from rdkit import Chem
    rw = Chem.RWMol(m)
    for b in rw.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in rw.GetAtoms():
        a.SetIsAromatic(False)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
        a.SetFormalCharge(0)
    out = rw.GetMol()
    Chem.SanitizeMol(out, Chem.SanitizeFlags.SANITIZE_SYMMRINGS
                     | Chem.SanitizeFlags.SANITIZE_ADJUSTHS, catchErrors=True)
    return out


def match_order_by_graph(mol, xyz_path):
    """Reorder the coordinates of an xyz file into the atom order of an RDKit mol --
    **by graph isomorphism, not by geometry**.

    Geometric methods (nearest neighbour, or a Hungarian assignment within each element)
    give the wrong correspondence when the two structures have not been aligned first, and
    this project has been caught by that twice: once when greedy nearest neighbour reported
    a root-mean-square deviation of 1.6447 A, and once when the old implementation of this
    function swapped acetamide's oxygen and nitrogen and reported 0.02 A as 1.07 A. Graph
    isomorphism does not look at coordinates and so has no such failure mode; when there is
    no solution it raises, and never returns an approximate correspondence.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    raw = Chem.MolFromXYZFile(str(xyz_path))
    if raw is None:
        raise ValueError("cannot read xyz: {}".format(xyz_path))
    rdDetermineBonds.DetermineConnectivity(raw, charge=0)
    match = _connectivity_only(raw).GetSubstructMatch(_connectivity_only(mol))
    if len(match) != mol.GetNumAtoms():
        raise ValueError(
            "the connectivity graph of {} is not isomorphic to the template (matched "
            "{}/{} atoms) -- refusing to substitute a geometric approximation".format(
                Path(xyz_path).name, len(match), mol.GetNumAtoms()))
    pos = np.array(raw.GetConformer().GetPositions())
    return pos[list(match)], list(match)


def write_xyz(atoms, path, comment=""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["{}".format(len(atoms)), comment]
    for s, p in zip(atoms.get_chemical_symbols(), atoms.positions):
        lines.append("{:2s} {:18.10f} {:18.10f} {:18.10f}".format(s, *p))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def dump_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
