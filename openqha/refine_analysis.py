# -*- coding: utf-8 -*-
"""Why `refine="opt"` returns fewer basins than `refine="sp"`.

THE MEASUREMENT THIS EXPLAINS (acceptance criterion 3, 2026-09-04, OCCC(=O)CO, gfn2):

    refine   CREST wall   conformers   basins   correction        error vs union
    opt         5980.0 s      24          23    -0.0488 kcal/mol   0.4023 kcal/mol
    sp           776.6 s      82          37    -0.4486            0.0025
    union of both, pooled and re-deduplicated: 45 basins

`opt` costs 7.7x and returns 14 fewer basins. The config had predicted the opposite --
that the gfnff-era gap would close under gfn2, because gfn2 geometries start much nearer
the MACE minimum.

THE HYPOTHESIS THIS MODULE TESTS
--------------------------------
`refine="opt"` makes CREST relax the ensemble on MACE and then hand it to CREGEN.
CREGEN's structural condition is an RMSD threshold (0.125 A). Two distinct minima that
are still closer than that AT CREST'S CONVERGENCE get merged -- and a merge is not
recoverable: the survivor is kept and the other structure is discarded before we ever see
it. `refine="sp"` merges nothing on the MACE surface; it hands over the raw ensemble, and
this repository tightens to fmax = 1e-4 eV/A FIRST and deduplicates SECOND, so the same
two minima have separated by the time the threshold is applied.

If that is the mechanism, then the number of basins recovered from ONE fixed ensemble
must depend on how tightly it was converged before deduplication -- loose convergence
giving few basins, tight convergence giving many. If instead `opt` simply sampled
differently, the count will be flat in convergence and the explanation is elsewhere.

That is what `basins_vs_convergence` measures, and it is a criterion that can fail: a
flat curve refutes the hypothesis.

WHY IT IS AFFORDABLE
--------------------
The ladder is walked along ONE optimisation trajectory per structure, snapshotting each
time the force drops through the next threshold. The whole ladder therefore costs about
one full relaxation of the ensemble, not one per rung.
"""
from __future__ import annotations

import numpy as np

from . import conformers, crest_census

#: eV -> kcal/mol.
EV_TO_KCAL = 23.060548

#: Force thresholds in eV/A, loosest first. The tight end is package 2's own
#: `FMAX_HESSIAN_EV_A`; the loose end is deliberately looser than any converged geometry
#: so that the first rung shows the ensemble while it is still collapsed together.
FMAX_LADDER_EV_A = (3.0e-1, 1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4)


def _fmax(atoms):
    return float(np.abs(atoms.get_forces()).max())


def relax_with_snapshots(atoms, calc, ladder=FMAX_LADDER_EV_A, max_steps=2000):
    """Relax one structure, keeping a copy each time |F|max drops through a rung.

    Returns (snapshots, energies) as dicts keyed by the rung. A rung already satisfied at
    the start records the starting structure, which is the honest answer: at that
    tolerance, this is what the geometry was.
    """
    from ase.optimize import BFGS

    atoms = atoms.copy()
    atoms.calc = calc
    snaps, energies = {}, {}
    remaining = sorted(ladder, reverse=True)

    def capture():
        # `while`, not `if`: one BFGS step can cross several rungs at once, and skipping
        # the ones it jumped would leave holes that look like missing data.
        f = _fmax(atoms)
        while remaining and f <= remaining[0]:
            rung = remaining.pop(0)
            snaps[rung] = atoms.copy()
            energies[rung] = float(atoms.get_potential_energy())

    capture()
    if remaining:
        opt = BFGS(atoms, logfile=None)
        for _ in opt.irun(fmax=min(ladder), steps=max_steps):
            capture()
            if not remaining:
                break
    # Anything never reached (hit the step limit) is left absent rather than filled in
    # with the last geometry -- an unconverged structure is not a datum at that rung.
    return snaps, energies


def basins_vs_convergence(frames, calc, smiles, ladder=FMAX_LADDER_EV_A,
                          threshold_A=conformers.DEDUP_RMSD_A,
                          ethr_kcal=conformers.CREGEN_ETHR_KCAL,
                          bthr_rel=conformers.CREGEN_BTHR_REL,
                          temperature_K=298.15, max_steps=2000, progress=None):
    """How many basins ONE ensemble yields as a function of how tightly it was converged
    before the CREGEN criterion was applied.

    The deduplication criterion is held fixed at every rung -- the same three conditions,
    the same thresholds, the same code path (`conformers.dedup`). The ONLY thing that
    varies is convergence. That is what makes the result attributable.
    """
    # The pipeline's own entry point: it adds hydrogens, and it verifies the atom order
    # by graph isomorphism rather than trusting it. Skipping that check has cost this
    # repository two defects (26, 30).
    mol, cids, order_info = crest_census.frames_into_mol(smiles, list(frames))

    per_frame = []
    for i, atoms in enumerate(frames):
        if progress:
            progress("relaxing {}/{}".format(i + 1, len(frames)))
        per_frame.append(relax_with_snapshots(atoms, calc, ladder, max_steps))

    rows = []
    for rung in sorted(ladder, reverse=True):
        have = [(s[rung], e[rung]) for s, e in per_frame if rung in s]
        if not have:
            rows.append(dict(fmax_eV_A=rung, n_input=0, n_basins=None,
                             note="no structure reached this tolerance"))
            continue
        # Load this rung's geometries onto the SAME conformers and deduplicate with the
        # SAME function the pipeline uses. Only the geometries change between rungs.
        idx = [i for i, (s, _e) in enumerate(per_frame) if rung in s]
        rung_cids = [cids[i] for i in idx]
        energies = [per_frame[i][1][rung] for i in idx]
        for i, cid in zip(idx, rung_cids):
            conformers._set_conf(mol, cid, per_frame[i][0][rung].get_positions())

        blocked = {}
        kept, _mapping, _warn = conformers.dedup(
            mol, rung_cids, energies, threshold_A=threshold_A,
            ethr_kcal=ethr_kcal, bthr_rel=bthr_rel, blocked=blocked)
        pos_of = {c: k for k, c in enumerate(rung_cids)}
        keep = [pos_of[c] for c in kept]
        lowest = min(energies[k] for k in keep)
        rel = [(energies[k] - lowest) * EV_TO_KCAL for k in keep]
        rows.append(dict(
            fmax_eV_A=rung,
            n_input=len(have),
            n_basins=len(keep),
            merges=len(have) - len(keep),
            blocked=dict(blocked),
            atom_order_checked=bool(order_info),
            correction_kcal=_correction(rel, temperature_K),
            spread_kcal=(max(rel) - min(rel)) if rel else 0.0))
    return rows


def _correction(rel_kcal, temperature_K):
    """-RT ln sum(exp(-dE/RT)) -- the conformational free-energy correction."""
    if not rel_kcal:
        return 0.0
    rt = 1.987204259e-3 * temperature_K
    return float(-rt * np.log(np.sum(np.exp(-np.asarray(rel_kcal) / rt))))
