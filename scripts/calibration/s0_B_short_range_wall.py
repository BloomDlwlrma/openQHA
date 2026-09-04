"""Does the potential have a repulsive wall at short range? Scan it and find out.

CALIBRATION. It measures a property of each MODEL, not of a molecule, and it decides
which engine branch B can run molecular dynamics on. It produces no thermodynamic number.

Why
---
A branch B trajectory on MACE-OFF23_medium dissociated at 1.656 ps with two geminal
hydrogens at 0.315 A, and 3 of 8 seeds did the same. The obvious suspect was a hole in the
model's short-range repulsion, which is the classic failure of a machine-learned potential
where its training set has no data -- and no electronic-structure dataset contains a 0.3 A
H-H contact.

**This script was written to test that suspicion, and it refuted it.** Models whose wall is
intact (MACE-OFF23-SC, MACE-OFF23b_medium) blew up at the same rate as the holed default,
so the wall does not predict stability. **The cause is still unknown.** A second
explanation -- a neighbour list that is not translation invariant -- was written down here
and has also been retracted: that defect is real but lives in the MACE develop tree
vendored under stage 2, not in the `mace_torch 0.3.16` that openQHA imports, which is
exactly invariant with the patch off. See openqha/mace_patch.py and the docstring of
scripts/calibration/s0_B_md_stability.py.

The script is kept, and still run, because what it measures is real and still matters: the
holes exist, they are simply not reachable in a correct simulation at 298 K. It is the
check to run before trusting any model at elevated temperature or in a biased sampling
scheme, where the short-range region does become accessible.

It asks one question of each model, and the answer is a fact about the model:
**as two atoms are pushed together, does the energy rise all the way in?**

What it reports, and why these numbers
--------------------------------------
  * `first_attractive_distance_A` -- the largest separation at which pushing the pair
    CLOSER lowers the energy. That is the mouth of the hole; above it the model behaves,
    below it the model pulls. `None` means no hole was found on the grid, which is the
    answer a usable model gives.
  * `min_energy_below_equilibrium_kcal` -- how deep the hole is. A hole a few kcal/mol
    deep is a wrinkle; one thousands of kcal/mol deep is a bomb, because that energy
    turns into kinetic energy in a few femtoseconds.
  * `energy_at_*` -- the raw curve, so a reader can see the shape rather than trust the
    summary.

The rest of the molecule is held fixed and the pair is moved symmetrically about its
midpoint. That is not the dynamical path, and the barrier along the true path is lower --
which makes this scan a LOWER bound on the problem, not an upper one.

    python scripts/calibration/s0_B_short_range_wall.py
    python scripts/calibration/s0_B_short_range_wall.py --engines MACE-OFF23-SC
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))

from openqha import config, engine, report  # noqa: E402

EV_TO_KCAL = 23.060547830618307

#: Distances to sample, angstrom. Dense where the wall should be and where the hole was
#: found; the grid runs well past both so the shape is visible rather than inferred.
GRID_A = (2.00, 1.80, 1.60, 1.40, 1.20, 1.00, 0.90, 0.80, 0.70, 0.60, 0.50,
          0.45, 0.40, 0.35, 0.315, 0.30, 0.25, 0.20, 0.15, 0.10)

#: Pairs to probe on acetone, by atom index into the QM9 geometry
#: ['C','C','C','O','H','H','H','H','H','H'].
#: H8-H9 is the pair that actually broke a trajectory; the others are there so that a
#: model which is safe for H-H but not for C-H does not pass by luck.
PAIRS = (("H-H (geminal, same methyl)", 8, 9),
         ("C-H (same methyl)", 2, 8),
         ("C-C (carbonyl to methyl)", 1, 2),
         ("C-O (carbonyl)", 1, 3))


def scan_pair(atoms, calc, i, j, grid=GRID_A):
    """Push atoms i and j together symmetrically about their midpoint, everything else fixed."""
    p0 = atoms.get_positions()
    e0 = atoms.get_potential_energy()
    u = p0[j] - p0[i]
    d0 = float(np.linalg.norm(u))
    u = u / d0
    mid = 0.5 * (p0[i] + p0[j])
    out = []
    for d in grid:
        q = p0.copy()
        q[i] = mid - 0.5 * d * u
        q[j] = mid + 0.5 * d * u
        atoms.set_positions(q)
        out.append((float(d), float((atoms.get_potential_energy() - e0) * EV_TO_KCAL)))
    atoms.set_positions(p0)
    return d0, out


def summarise(curve, equilibrium_A):
    """Turn the curve into the numbers that decide whether a model is usable.

    Two different flags, because they answer different questions:

      * `monotonic_wall` is the strict one -- the energy never falls at all as the pair
        closes. It is the ideal, and a model can fail it on a wiggle high up the wall
        that nothing will ever reach.
      * `wall_holds` is the one that decides MD stability: the curve never drops BELOW
        the starting energy, so there is no downhill path inward and nothing to release.
        `max_inward_drop_kcal` says how much energy the largest downhill run would give
        up, which is the quantity that turns into the kinetic energy that breaks a
        molecule apart.
    """
    d_all = np.array([c[0] for c in curve])
    e_all = np.array([c[1] for c in curve])
    # ONLY the compressed side counts. Approaching a bond from 2.0 A down to its 1.2 A
    # minimum is downhill and physical; counting that as "attractive" would flag every
    # model on every pair, which the first version of this script duly did. The question
    # is what happens INSIDE the minimum.
    inside = d_all < equilibrium_A
    d = d_all[inside]
    e = e_all[inside]
    # The grid descends, so index k+1 is CLOSER than index k. The pair is attractive
    # wherever moving closer lowers the energy.
    attractive = [k for k in range(len(d) - 1) if e[k + 1] < e[k]]
    first = float(d[attractive[0]]) if attractive else None
    drop = 0.0
    for k in range(len(e)):
        run = float(e[k] - e[k:].min())
        drop = max(drop, run)
    return dict(
        equilibrium_energy_kcal=0.0,
        first_attractive_distance_A=first,
        monotonic_wall=bool(not attractive),
        wall_holds=bool(e.min() >= 0.0),
        max_inward_drop_kcal=drop,
        min_energy_below_equilibrium_kcal=float(e.min()) if e.min() < 0 else None,
        distance_of_minimum_A=(float(d[int(np.argmin(e))]) if e.min() < 0 else None),
        energy_at_1p0_A=float(np.interp(1.0, d_all[::-1], e_all[::-1])),
        energy_at_0p5_A=float(np.interp(0.5, d_all[::-1], e_all[::-1])),
        energy_at_0p3_A=float(np.interp(0.3, d_all[::-1], e_all[::-1])),
        energy_at_0p1_A=float(np.interp(0.1, d_all[::-1], e_all[::-1])),
        curve=[dict(distance_A=x, energy_kcal=y) for x, y in curve],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default="dsgdb9nsd_000018")
    ap.add_argument("--engines", nargs="*", default=None,
                    help="engine names from openqha/engine.py; default is all of them")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ase.io import read
    from ase.optimize import BFGS

    cfg = config.load()
    names = args.engines or list(engine.ENGINES)
    # MACE-OFF23-SC first: the solvation paper says it was trained with short-range
    # repulsion, so it is the model most likely to answer the question in the affirmative.
    names = sorted(names, key=lambda n: (0 if n == "MACE-OFF23-SC" else 1, n))

    print("=" * 96)
    print("Branch B calibration -- is there a repulsive wall at short range?")
    print("=" * 96)
    print("species {}   pairs probed: {}".format(
        args.species, ", ".join(p[0] for p in PAIRS)))
    print()

    rows = []
    for name in names:
        try:
            calc, engine_name, prov = engine.calculator(device=args.device, name=name)
        except Exception as exc:                       # a missing weight file is not fatal
            print("{:<22} SKIP: {}: {}".format(name, type(exc).__name__, exc))
            continue
        atoms = read(str(config.qm9_xyz(args.species, cfg)))
        atoms.calc = calc
        BFGS(atoms, logfile=None).run(fmax=0.005, steps=300)
        print("{}   (relaxed on its own surface)".format(name))
        for label, i, j in PAIRS:
            d0, curve = scan_pair(atoms, calc, i, j)
            s = summarise(curve, d0)
            s.update(engine=name, pair=label, atom_i=i, atom_j=j,
                     equilibrium_distance_A=d0)
            rows.append(s)
            print("   {:<28} d_eq {:5.3f}   wall {:<5}  inward drop {:>10.1f} kcal/mol  "
                  "{}".format(
                      label, d0, "HOLDS" if s["wall_holds"] else "HOLE",
                      s["max_inward_drop_kcal"],
                      "" if s["first_attractive_distance_A"] is None else
                      "first attractive at {:.3f} A{}".format(
                          s["first_attractive_distance_A"],
                          "" if s["min_energy_below_equilibrium_kcal"] is None else
                          ", bottom {:.0f} kcal/mol BELOW equilibrium".format(
                              s["min_energy_below_equilibrium_kcal"]))))
        print()

    engines = sorted({r["engine"] for r in rows})
    usable = [n for n in engines
              if all(x["wall_holds"] for x in rows if x["engine"] == n)]
    holed = [n for n in engines
             if any(not x["wall_holds"] for x in rows if x["engine"] == n)]
    print("wall HOLDS on every pair (no downhill path inward): {}".format(
        ", ".join(usable) or "NONE"))
    print("wall has a HOLE on at least one pair              : {}".format(
        ", ".join(holed) or "none"))
    print()
    print("{:<22} {:>16}  {}".format("engine", "worst drop", "worst pair"))
    for n in engines:
        mine = [r for r in rows if r["engine"] == n]
        worst = max(mine, key=lambda r: r["max_inward_drop_kcal"])
        print("{:<22} {:>16.1f}  {}".format(n, worst["max_inward_drop_kcal"],
                                            worst["pair"]))

    out_stem = Path(args.out) if args.out else (
        _repo_root() / "analysis" / "qha" / "calibration_short_range_wall")
    rp = report.Report(
        "Branch B calibration -- short-range repulsion of each MACE model",
        subtitle="{}   {} pairs   {} models".format(
            args.species, len(PAIRS), len({r["engine"] for r in rows})))
    rp.section("The question")
    rp.note("As two atoms are pushed together with the rest of the molecule fixed, does "
            "the energy rise all the way in? A model that says no has a hole, and an "
            "unbiased trajectory will find it: on MACE-OFF23_medium the measured rate "
            "was 0.154 events per ps, a mean survival near 6.5 ps.")
    rp.note("The pair is moved symmetrically about its midpoint with everything else "
            "frozen. That is not the dynamical path and the true barrier is lower, so "
            "this scan is a LOWER bound on the problem.")
    rp.section("Result")
    rp.table(["engine", "pair", "d_eq", "wall", "inward drop", "hole opens", "bottom",
              "E(1.0)", "E(0.5)", "E(0.3)"],
             [[r["engine"], r["pair"], round(r["equilibrium_distance_A"], 3),
               "HOLDS" if r["wall_holds"] else "HOLE",
               round(r["max_inward_drop_kcal"], 1),
               "-" if r["first_attractive_distance_A"] is None
               else round(r["first_attractive_distance_A"], 3),
               "-" if r["min_energy_below_equilibrium_kcal"] is None
               else round(r["min_energy_below_equilibrium_kcal"], 1),
               round(r["energy_at_1p0_A"], 1), round(r["energy_at_0p5_A"], 1),
               round(r["energy_at_0p3_A"], 1)] for r in rows],
             units=[None, None, "angstrom", None, "kcal/mol", "angstrom", "kcal/mol",
                    "kcal/mol", "kcal/mol", "kcal/mol"])
    rp.kv("models_whose_wall_holds_on_every_pair", ", ".join(usable) or "NONE")
    rp.kv("models_with_a_hole", ", ".join(holed) or "none")
    rp.note("`wall holds` means the curve never drops below the starting energy, so "
            "there is no downhill path inward. `inward drop` is the largest energy a "
            "downhill run would release, which is what turns into the kinetic energy "
            "that breaks a molecule apart.")
    rp.json_dump(dict(species=args.species, grid_A=list(GRID_A),
                      pairs=[dict(label=p[0], i=p[1], j=p[2]) for p in PAIRS],
                      scans=rows))
    log = rp.write(str(out_stem) + ".log")
    flat = [{k: v for k, v in r.items() if k != "curve"} for r in rows]
    written = report.write_parquet(dict(scans=flat), out_stem)
    print("\nwritten:\n  {}".format(log))
    for p, n, _c in written:
        print("  {}  ({} rows)".format(p, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
