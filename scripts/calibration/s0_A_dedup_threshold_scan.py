"""Sweep the deduplication threshold, and compare our criterion with CREST's.

CALIBRATION. Branch A. It produces the numbers used to decide what the deduplication
criterion should be; it produces nothing that enters a deliverable.

Why this exists
---------------
On `OCCC(=O)CO` the pipeline merged two structures **0.279 Å apart** whose energies
differ by **0.377 kcal/mol** (branch A checkpoint 4). The declared threshold is 0.30 Å,
so the merge followed the rule — but whether the rule is right was never measured, and
RMSD alone cannot answer it.

Then a second problem surfaced: **0.30 Å has no source.** It is marked
`modifiable_convention` in the config with the note "this package carries its own
threshold sweep", and the sweep was never run. Meanwhile CREST's own documented default
is `RTHR = 0.125 Å` — 2.4× tighter — and CREST does not use RMSD alone.

What CREST actually does, and what we do
----------------------------------------
CREGEN merges two structures only if **all three** hold:

    RMSD           < RTHR    default 0.125 Å
    |ΔE|           < ETHR    default 0.05 kcal/mol
    rot. constants within BTHR, default 1% (adjusted dynamically up to 2.5%)

We merge on RMSD alone, at 0.30 Å, and merely *warn* when |ΔE| > 0.05 kcal/mol — which
is CREST's ETHR used as a diagnostic rather than as a criterion.

So the 0.279 Å / 0.377 kcal/mol pair fails CREST's test **twice over**: the RMSD is 2.2×
RTHR and the energy gap is 7.5× ETHR. It is not a marginal call under CREST's rule; it
is a clear reject.

This script measures what each rule does to the basin count, on real ensembles, so the
choice can be made from numbers instead of from which default was inherited.

A note on permutation invariance
--------------------------------
CREST 3.1 adds `--irmsd`, a permutation-invariant RMSD (canonical atom identities,
rotational-axis alignment, Hungarian assignment) that fixes the case where "chemically
identical structures are written down differently". **The installed CREST 3.0.2 does not
have it** -- measured, not assumed: `--irmsd` aborts in `parseflags_`
(`src/confparse.f90:788`), and the strings `irmsd`, `isort` and `hungarian` do not occur
in the binary, while `rthr`, `bthr`, `ethr` and `cregen` all do.

We do not need it for correctness: `conformers.all_atom_best_rms` already minimises over
the molecular automorphisms, which is the same invariance obtained a different way. What
CREST's version would add is a second, independent implementation to check ours against
-- worth having, not blocking.

Usage
-----
    python scripts/calibration/s0_A_dedup_threshold_scan.py \\
        --run ~/runs/openQHA/branchA/multibasin/dihydroxybutanone \\
        --smiles "OCCC(=O)CO" --label dihydroxybutanone
"""
import argparse
import json
import sys
from pathlib import Path


def _repo_root():
    for p in Path(__file__).resolve().parents:
        if (p / "openqha" / "__init__.py").is_file():
            return p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))

import numpy as np  # noqa: E402

from openqha import S0_ROOT, config, conformers, crest_census, engine, thermo  # noqa: E402

EV_TO_KCAL = conformers.EV_TO_KCAL

#: Thresholds swept. 0.125 is CREST's documented RTHR default; 0.30 is this repo's
#: current value; the rest bracket both so a plateau (or its absence) is visible.
#: modifiable_convention.
THRESHOLDS_A = (0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)

#: CREGEN defaults, from the CREST documentation (keywords page). Quoted, not guessed.
CREST_RTHR_A = 0.125          # "Set RMSD threshold to <REAL> Ångström", default 0.125
CREST_ETHR_KCAL = 0.05        # "energy threshold between conformer pairs", default 0.05
CREST_BTHR_REL = 0.01         # rotational constant threshold, default 1%, up to 2.5%


def rotational_constants(atoms):
    """Rotational constants, in the same units for every structure.

    B_i is proportional to 1 / I_i, so the ratio test below is unit-free and the
    proportionality constant cancels. Returned sorted, because CREGEN compares
    rotational constants and not axis labels.
    """
    moments = thermo.principal_moments(atoms.get_masses(), atoms.get_positions())
    moments = np.asarray(sorted(float(m) for m in moments), dtype=float)
    with np.errstate(divide="ignore"):
        return 1.0 / np.where(moments > 1e-12, moments, np.inf)


def dedup_generic(mol, cids, energies_eV, atoms_list, threshold_A,
                  ethr_kcal=None, bthr_rel=None):
    """Greedy deduplication in ascending energy, with optional extra CREGEN criteria.

    `ethr_kcal=None` and `bthr_rel=None` reproduce this repo's current rule (RMSD
    alone). Supplying them adds CREST's other two conditions, so that "merge" requires
    all of the supplied tests to pass -- which is what CREGEN does.
    """
    order = list(np.argsort(np.asarray(energies_eV)))
    e = np.asarray(energies_eV, dtype=float)
    kept, mapping, merges = [], {}, []

    for k in order:
        cid = cids[k]
        dup_of = None
        for kcid in kept:
            rms = conformers.all_atom_best_rms(mol, cid, kcid)
            if rms >= threshold_A:
                continue
            kk = cids.index(kcid)
            de = abs(e[k] - e[kk]) * EV_TO_KCAL
            if ethr_kcal is not None and de >= ethr_kcal:
                continue
            if bthr_rel is not None:
                b1 = rotational_constants(atoms_list[k])
                b2 = rotational_constants(atoms_list[kk])
                rel = np.abs(b1 - b2) / np.maximum(np.abs(b2), 1e-30)
                if float(rel.max()) >= bthr_rel:
                    continue
            dup_of = kcid
            merges.append(dict(merged=int(cid), into=int(kcid),
                               rmsd_A=float(rms), energy_difference_kcal=float(de)))
            break
        if dup_of is None:
            kept.append(cid)
        else:
            mapping[cid] = dup_of
    return kept, mapping, merges


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True,
                    help="a CREST work directory containing crest_conformers.xyz")
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = config.load()
    fmax = float(cfg["package2"]["fmax_hessian_eV_A"])
    current = float(cfg["package1"]["dedup_rmsd_A"])

    ens = Path(args.run).expanduser() / "crest_conformers.xyz"
    frames, comments = crest_census.read_ensemble_atoms(ens)
    calc, name, prov = engine.calculator(device=args.device)

    print("=" * 92)
    print("Branch A -- deduplication threshold sweep: {}  ({})".format(
        args.label, args.smiles))
    print("=" * 92)
    print("ensemble      {}  ({} frames)".format(ens, len(frames)))
    print("engine        {}".format(name))
    print("current rule  RMSD < {:.3f} Å, energy only WARNS above {:.2f} kcal/mol".format(
        current, conformers.MERGE_ENERGY_WARN_KCAL))
    print("CREST rule    RMSD < {:.3f} Å AND |ΔE| < {:.2f} kcal/mol AND ΔB < {:.1%}".format(
        CREST_RTHR_A, CREST_ETHR_KCAL, CREST_BTHR_REL))
    print()

    # ---- tighten ONCE; every threshold below then sees identical geometries --------
    # Sweeping on differently-converged structures would confound the threshold with
    # the convergence, and the whole point is to isolate the threshold.
    print("tightening {} frames to fmax = {:g} eV/Å ...".format(len(frames), fmax))
    trec, tight, energies = crest_census.tighten_frames(
        args.smiles, frames, calc, fmax=fmax,
        progress=lambda i, n: print("\r  %d/%d" % (i, n), end="", flush=True))
    print()
    print("  max residual force {:.3e} eV/Å | not converged {} | graph changed {}".format(
        trec["max_residual_force_eV_A"], trec["n_not_converged"],
        trec["n_graph_changed"]))
    print()

    mol, cids, _order = crest_census.frames_into_mol(args.smiles, tight)
    for cid, a in zip(cids, tight):
        conformers._set_conf(mol, cid, a.get_positions())

    # ---- the sweep -----------------------------------------------------------------
    rows = []
    print("%-9s %8s %8s %14s %16s" % ("thr / Å", "basins", "merges",
                                      "max ΔE merged", "max RMSD merged"))
    for thr in THRESHOLDS_A:
        kept, _m, merges = dedup_generic(mol, cids, energies, tight, thr)
        de = max([x["energy_difference_kcal"] for x in merges], default=0.0)
        rm = max([x["rmsd_A"] for x in merges], default=0.0)
        mark = "  <- current" if abs(thr - current) < 1e-9 else (
            "  <- CREST RTHR" if abs(thr - CREST_RTHR_A) < 1e-9 else "")
        print("%-9.3f %8d %8d %14.4f %16.4f%s"
              % (thr, len(kept), len(merges), de, rm, mark))
        rows.append(dict(threshold_A=thr, n_basins=len(kept), n_merges=len(merges),
                         max_energy_difference_merged_kcal=de,
                         max_rmsd_merged_A=rm, merges=merges))

    # ---- CREST's full three-fold criterion -----------------------------------------
    kept_c, _mc, merges_c = dedup_generic(
        mol, cids, energies, tight, CREST_RTHR_A,
        ethr_kcal=CREST_ETHR_KCAL, bthr_rel=CREST_BTHR_REL)
    print()
    print("CREST's full CREGEN criterion (RMSD AND energy AND rotational constants):")
    print("  basins %d, merges %d" % (len(kept_c), len(merges_c)))
    for m in merges_c:
        print("    %d -> %d   RMSD %.4f Å   ΔE %.4f kcal/mol"
              % (m["merged"], m["into"], m["rmsd_A"], m["energy_difference_kcal"]))

    # ---- the specific pair that started this --------------------------------------
    kept_now, _mn, merges_now = dedup_generic(mol, cids, energies, tight, current)
    print()
    print("under the current rule (%.2f Å, RMSD only): %d basins, %d merges"
          % (current, len(kept_now), len(merges_now)))
    for m in merges_now:
        verdicts = []
        verdicts.append("RMSD %s RTHR" % ("<" if m["rmsd_A"] < CREST_RTHR_A else ">="))
        verdicts.append("ΔE %s ETHR" % ("<" if m["energy_difference_kcal"]
                                        < CREST_ETHR_KCAL else ">="))
        print("    %d -> %d   RMSD %.4f Å   ΔE %.4f kcal/mol   [%s]"
              % (m["merged"], m["into"], m["rmsd_A"], m["energy_difference_kcal"],
                 ", ".join(verdicts)))

    counts = [r["n_basins"] for r in rows]
    plateau = len(set(counts)) == 1
    summary = dict(
        generated_by="scripts/calibration/s0_A_dedup_threshold_scan.py",
        label=args.label, smiles=args.smiles, ensemble=str(ens),
        engine=prov, n_frames=len(frames),
        tighten=trec,
        current_threshold_A=current,
        current_rule="all-atom best RMSD alone; energy difference only warns",
        crest_defaults=dict(rthr_A=CREST_RTHR_A, ethr_kcal=CREST_ETHR_KCAL,
                            bthr_relative=CREST_BTHR_REL,
                            source="CREST documentation, keywords page"),
        sweep=rows,
        basin_count_is_flat_over_sweep=plateau,
        crest_criterion=dict(n_basins=len(kept_c), n_merges=len(merges_c),
                             merges=merges_c),
        current_criterion=dict(n_basins=len(kept_now), n_merges=len(merges_now),
                               merges=merges_now),
        irmsd=dict(
            available_in_installed_crest=False,
            evidence=("crest 3.0.2 aborts in parseflags_ (src/confparse.f90:788) on "
                      "--irmsd; the strings irmsd/isort/hungarian do not occur in the "
                      "binary while rthr/bthr/ethr/cregen do"),
            note=("our all_atom_best_rms already minimises over the molecular "
                  "automorphisms, so permutation invariance is not missing; CREST's "
                  "version would be a second independent implementation to check "
                  "against")),
    )

    out = Path(args.out) if args.out else (
        S0_ROOT / "analysis" / "branchA" / "dedup_threshold_scan_{}.json".format(args.label))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("basin count flat across the whole sweep: %s" % plateau)
    print("written %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
