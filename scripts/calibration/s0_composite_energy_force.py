"""Benchmark the **energies and forces** of MACE-OFF23-SC against a composite reference; **the frequency capability exists but is off by default**.

CALIBRATION. Benchmarks energies and forces against the composite reference.

The scope was ruled on twice, the second revising the first. Both are recorded here; history is not overwritten:

* **2026-08-29 (morning)**: what we care about is the accuracy of **energies and forces**;
    **validating frequencies costs too much** (validating a composite frequency means really
    running 61 `CCSD(T)/cc-pVTZ` gradients, about 5.1 days), and the 0.073 eV/Angstrom in the original paper is a **force** error -- **force accuracy is not automatically frequency accuracy**.
* **2026-08-29 (afternoon)**: the user asked for it to "be able to produce them". So `--hessian`
    restores the capability: central differences of the **composite forces** give a composite Hessian, which then goes through the same Eckart projection and diagonalisation as package 2.
    **Off by default**; when switched on it first shows you the cost of `6N x (pool size)`
    single points, and without `--hessian-confirm-cost` it reports that cost and runs nothing.
    The "too expensive" argument was about **treating frequencies as a production reference quantity**, not about forbidding the capability.

Two measures, on different terms, which must be kept apart:

* **Forces** -- compared component by component, in eV/Angstrom, on the same terms as Allen et al.
    (their 0.073 eV/Angstrom is "the root-mean-square error of each force component").
    **They can only be read on structures away from a minimum** -- at a MACE minimum the MACE force
    is about zero by construction, the "deviation" is identically the reference force, and what is measured is the geometric difference of two minima, not force accuracy.
* **Energies** -- **only relative quantities are comparable**. The absolute energy of MACE and of
    coupled cluster have different zeros, so subtracting them is meaningless. What is comparable:
    **energy differences between conformers of one species**, **the energy of a thermally displaced structure relative to its own parent minimum**, and the reaction energy between two species of the same formula.

**The term pool**: along a basis ladder (DZ -> TZ* -> QZ*) the single points of the lower rungs are
a subset of the higher one. This script solves over the **deduplicated pool**, so measuring the whole
ladder costs what measuring the top rung costs and the middle rungs are **free** -- turning "has the ladder converged" from an assumption into a number you can read.

Usage::

    # smoke test: the single term MP2/cc-pVDZ
    python scripts/calibration/s0_composite_energy_force.py --recipe smoke_mp2_dz --limit 3 --displace 3

    # E2 (the accuracy ruled by the user on 2026-08-29, "MP2 DZ* -> QZ*"): 44 structures, one ladder
    python scripts/calibration/s0_composite_energy_force.py --ladder mp2_dz_to_qz --displace 3 \
        --nprocs 12 --tag E2 --resume

    # the frequency capability (off by default; look at the cost first)
    python scripts/calibration/s0_composite_energy_force.py --recipe dz_base --limit 1 --hessian

Products::

    analysis/composite_energy_force_<tag>.json           the final summary
    analysis/composite_energy_force_<tag>.partial.jsonl  one line per finished structure (resumable)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

def _repo_root():
    """Directory holding the openQHA package, found by walking up from this file.

    Depth-independent on purpose: this file keeps working wherever under the
    repository it is moved to. The earlier move into `scripts/_superseded/` broke
    every `parents[1]` in the moved files silently, which is what this replaces.
    """
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))

from openqha import S0_ROOT, config, conformers, engine, hessian, orca

OUT = S0_ROOT / "analysis"


def load_structures(cfg, limit=None):
    """Take every basin of every species already written by package 2 (true minima)."""
    pkg2 = OUT / "package2"
    if not pkg2.exists():
        raise FileNotFoundError(
            "{} does not exist -- run scripts/calibration/s0_package2_hessian_benchmark.py first".format(pkg2))
    from ase.io import read
    items = []
    for qid, spec in sorted(cfg["species"].items()):
        d = pkg2 / spec["name"]
        for xyz in sorted(d.glob("basin*.xyz")):
            items.append(dict(qm9_index=qid, name=spec["name"],
                              basin=int(xyz.stem.replace("basin", "")),
                              path=str(xyz), atoms=read(str(xyz))))
    if not items:
        raise FileNotFoundError("no basin*.xyz found under {}".format(pkg2))
    return items[:limit] if limit else items


def resolve_recipes(p2, args):
    """Turn `--recipe` / `--recipes` / `--ladder` into [(name, terms), ...]."""
    recipes = p2["composite_recipes"]
    if args.ladder:
        ladders = p2.get("recipe_ladders", {})
        if args.ladder not in ladders:
            raise KeyError("no ladder {!r} in the configuration; available: {}".format(
                args.ladder, sorted(ladders)))
        names = list(ladders[args.ladder]["recipes"])
    elif args.recipes:
        names = [x.strip() for x in args.recipes.split(",") if x.strip()]
    else:
        names = [args.recipe or p2["reference_level"]]
    out = []
    for n in names:
        if n not in recipes:
            raise KeyError("no recipe {!r} in the configuration; available: {}".format(n, sorted(recipes)))
        out.append((n, [[t[0], t[1], t[2]] for t in recipes[n]["terms"]]))
    return out


def key_of(it):
    return "{}|{}|{}".format(it["name"], it["basin"],
                             "min" if it.get("displaced") is None
                             else "d{}".format(it["displaced"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default=None,
                    help="a single recipe name; defaults to reference_level from the configuration")
    ap.add_argument("--recipes", default=None, help="several comma-separated recipes sharing one term pool")
    ap.add_argument("--ladder", default=None, help="the name of a recipe_ladders entry in the configuration")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N minima (for a smoke test)")
    ap.add_argument("--nprocs", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=None, help="time limit in seconds for one ORCA run")
    ap.add_argument("--displace", type=int, default=0,
                    help="sample N extra thermally displaced structures per minimum. "
                         "**The force benchmark requires this** -- comparing forces at a MACE minimum is degenerate")
    ap.add_argument("--temperature", type=float, default=298.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="suffix of the product filename; defaults to the recipe/ladder name")
    ap.add_argument("--resume", action="store_true",
                    help="read an existing .partial.jsonl and skip the structures already done")
    ap.add_argument("--hessian", action="store_true",
                    help="**a capability that is off by default**: central differences of the composite forces give a composite Hessian and frequencies. "
                         "Minima only. Without --hessian-confirm-cost it reports the cost and runs nothing")
    ap.add_argument("--hessian-confirm-cost", action="store_true",
                    help="having seen the cost, actually run the composite Hessian")
    ap.add_argument("--hessian-delta", type=float, default=hessian.DELTA_A,
                    help="central-difference displacement for the composite Hessian, in Angstrom")
    args = ap.parse_args()

    cfg = config.load()
    p2 = config.package(2, cfg)
    chosen = resolve_recipes(p2, args)
    pool_terms = orca.unique_terms([t for _n, t in chosen])
    tag = args.tag or (args.ladder or "_".join(n for n, _t in chosen))

    calc, engine_name, prov = engine.calculator()
    items = load_structures(cfg, args.limit)

    # ---- thermal displacement: the force benchmark has to leave the minimum -------------
    displaced_rec = None
    if args.displace:
        print("sampling {} thermally displaced structures at {:.2f} K per minimum ...".format(
            args.displace, args.temperature))
        extra, displaced_rec = [], []
        for it in items:
            s_list, rec = hessian.thermal_displacements(
                it["atoms"], calc, temperature_K=args.temperature,
                n_samples=args.displace, seed=args.seed)
            rec["parent"] = "{} basin {}".format(it["name"], it["basin"])
            displaced_rec.append(rec)
            for j, st in enumerate(s_list):
                extra.append(dict(qm9_index=it["qm9_index"], name=it["name"],
                                  basin=it["basin"],
                                  path=it["path"] + "#displaced{}".format(j),
                                  displaced=j, atoms=st))
            print("   {:20s} basin {}  rms displacement {} A  {} rejection(s)".format(
                it["name"], it["basin"],
                [round(x, 3) for x in rec["rms_displacement_A"]],
                rec["n_draws_rejected"]))
        items = items + extra
        print()

    print("=" * 100)
    print("energy and force benchmark   {}  vs  {}".format(engine_name, tag))
    print("=" * 100)
    for name, terms in chosen:
        print("recipe {}".format(name))
        for c, m, b in terms:
            print("      {:+d}  {:8s} {}".format(int(c), m, b))
    print("term pool (deduplicated; only these run per geometry): {}".format(
        ", ".join("{}/{}".format(m, b) for m, b in pool_terms)))
    print("spin {}   ORCA {}".format(p2.get("reference_spin", "?"), orca.orca_binary()))
    print("{} structure(s) ({} true minima + {} thermally displaced)".format(
        len(items), sum(1 for i in items if i.get("displaced") is None),
        sum(1 for i in items if i.get("displaced") is not None)))
    print()

    # ---- resume from a checkpoint -------------------------------------------------------
    part = OUT / "composite_energy_force_{}.partial.jsonl".format(tag)
    part.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if args.resume and part.exists():
        for line in part.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r
        print("resuming: results already exist for {} structure(s); skipping them.".format(len(done)))
        print()

    results, t0 = [], time.time()
    for k, it in enumerate(items):
        kk = key_of(it)
        if kk in done:
            results.append(done[kk])
            continue
        a = it["atoms"]
        a.calc = calc
        e_mace = float(a.get_potential_energy())
        f_mace = a.get_forces()

        print("   {:>2}/{:<2} {:20s} basin {} {}".format(
            k + 1, len(items), it["name"], it["basin"],
            "" if it.get("displaced") is None
            else "displaced {}".format(it["displaced"])), flush=True)

        def prog(i, n, m, b, s):
            print("      [{}/{}] {:8s} {:9s} {:7.1f} s".format(i, n, m, b, s),
                  flush=True)

        pool = orca.term_pool(a.get_chemical_symbols(), a.get_positions(),
                              pool_terms, nprocs=args.nprocs,
                              timeout_s=args.timeout, progress=prog)

        per_recipe = {}
        for name, terms in chosen:
            ref = orca.combine(pool, terms)
            fm = orca.force_metrics(ref["forces_eV_A"], f_mace)
            per_recipe[name] = dict(
                energy_ref_eV=ref["energy_eV"],
                forces_ref_eV_A=ref["forces_eV_A"].tolist(),
                force_metrics=fm, orca_version=ref["orca_version"],
                per_term=ref["per_term"])
            print("      {:24s} forces: mean absolute deviation {:.4f}  rms deviation {:.4f}  "
                  "maximum {:.4f}   (rms of the reference force itself {:.4f}) eV/A".format(
                      name, fm["mae_eV_A"], fm["rmse_eV_A"],
                      fm["max_abs_eV_A"], fm["ref_rms_eV_A"]), flush=True)

        rec = dict(key=kk, qm9_index=it["qm9_index"], name=it["name"],
                   basin=it["basin"], displaced=it.get("displaced"),
                   path=it["path"], n_atoms=len(a),
                   energy_mace_eV=e_mace, forces_mace_eV_A=f_mace.tolist(),
                   pool_seconds=float(sum(r["seconds"] for r in pool.values())),
                   per_recipe=per_recipe)
        results.append(rec)
        # **Write after every finished structure** -- an 8-hour run must not write only at the end
        with open(str(part), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- summary ------------------------------------------------------------------------
    at_min = [r for r in results if r.get("displaced") is None]
    off_min = [r for r in results if r.get("displaced") is not None]

    def pool_forces(subset, name):
        if not subset:
            return None
        aa = np.concatenate([np.array(r["per_recipe"][name]["forces_ref_eV_A"]).ravel()
                             for r in subset])
        bb = np.concatenate([np.array(r["forces_mace_eV_A"]).ravel() for r in subset])
        return orca.force_metrics(aa, bb)

    def relative_energies(name):
        """(a) conformer relative energies: minima only; (b) displacement energies: relative to each parent minimum."""
        conf, by_species = [], {}
        for r in at_min:
            by_species.setdefault(r["qm9_index"], []).append(r)
        for qid, rs in by_species.items():
            if len(rs) < 2:
                continue
            e0m = min(x["energy_mace_eV"] for x in rs)
            e0r = min(x["per_recipe"][name]["energy_ref_eV"] for x in rs)
            for x in rs:
                dm = (x["energy_mace_eV"] - e0m) * conformers.EV_TO_KCAL
                dr = ((x["per_recipe"][name]["energy_ref_eV"] - e0r)
                      * conformers.EV_TO_KCAL)
                conf.append(dict(qm9_index=qid, name=x["name"], basin=x["basin"],
                                 rel_mace_kcal=dm, rel_ref_kcal=dr,
                                 diff_kcal=dm - dr))
        disp, parent = [], {(r["qm9_index"], r["basin"]): r for r in at_min}
        for r in off_min:
            pa = parent.get((r["qm9_index"], r["basin"]))
            if pa is None:
                continue
            dm = (r["energy_mace_eV"] - pa["energy_mace_eV"]) * conformers.EV_TO_KCAL
            dr = ((r["per_recipe"][name]["energy_ref_eV"]
                   - pa["per_recipe"][name]["energy_ref_eV"]) * conformers.EV_TO_KCAL)
            disp.append(dict(qm9_index=r["qm9_index"], name=r["name"],
                             basin=r["basin"], displaced=r["displaced"],
                             rel_mace_kcal=dm, rel_ref_kcal=dr, diff_kcal=dm - dr))
        return conf, disp

    def stat(rows):
        if not rows:
            return None
        d = np.array([x["diff_kcal"] for x in rows])
        return dict(n=len(rows), mae_kcal=float(np.abs(d).mean()),
                    rmse_kcal=float(np.sqrt((d ** 2).mean())),
                    max_abs_kcal=float(np.abs(d).max()),
                    signed_mean_kcal=float(d.mean()))

    summary = {}
    print()
    print("=" * 100)
    print("summary")
    print("=" * 100)
    for name, _terms in chosen:
        fa, fo = pool_forces(at_min, name), pool_forces(off_min, name)
        conf, disp = relative_energies(name)
        summary[name] = dict(force_at_minima=fa, force_displaced=fo,
                             relative_energy_conformers=conf,
                             relative_energy_displacements=disp,
                             energy_stat_conformers=stat(conf),
                             energy_stat_displacements=stat(disp))
        print()
        print("### recipe {}".format(name))
        for tagline, pm, degenerate in (
                ("at the minima (**degenerate; not force accuracy**)", fa, True),
                ("on thermal displacements (**this is force accuracy**)", fo, False)):
            if pm is None:
                continue
            print("  forces -- {}".format(tagline))
            print("     {} component(s): mean absolute deviation {:.4f}  **rms deviation {:.4f}**  "
                  "maximum {:.4f} eV/A".format(
                      pm["n_components"], pm["mae_eV_A"], pm["rmse_eV_A"],
                      pm["max_abs_eV_A"]))
            print("     signed mean {:+.4f}; rms of the reference force itself {:.4f} eV/A".format(
                pm["signed_mean_eV_A"], pm["ref_rms_eV_A"]))
            if degenerate:
                print("     ** here the MACE force is about zero by construction, so the deviation is identically the reference force.")
        for lab, st in (("conformer relative energies (minima only)",
                         summary[name]["energy_stat_conformers"]),
                        ("thermal displacement energies (relative to each parent minimum)",
                         summary[name]["energy_stat_displacements"])):
            if st is None:
                print("  {}: nothing to compare against; skipped.".format(lab))
                continue
            print("  {} ({}): mean absolute deviation {:.4f}  rms deviation {:.4f}  "
                  "maximum {:.4f} kcal/mol".format(
                      lab, st["n"], st["mae_kcal"], st["rmse_kcal"],
                      st["max_abs_kcal"]))
    print()
    print("   for reference: in Allen et al. the composite has an rms force error of 0.073 eV/A against true CCSD(T)/QZ")

    # ---- the ladder: the difference between adjacent rungs -------------------------------
    ladder = None
    if len(chosen) > 1:
        ladder = []
        names = [n for n, _t in chosen]
        for lo, hi in zip(names[:-1], names[1:]):
            de, df = [], []
            for r in results:
                de.append((r["per_recipe"][hi]["energy_ref_eV"]
                           - r["per_recipe"][lo]["energy_ref_eV"])
                          * conformers.EV_TO_KCAL)
                df.append(np.array(r["per_recipe"][hi]["forces_ref_eV_A"])
                          - np.array(r["per_recipe"][lo]["forces_ref_eV_A"]))
            dfa = np.concatenate([x.ravel() for x in df])
            ladder.append(dict(
                lower=lo, upper=hi,
                energy_shift_mean_kcal=float(np.mean(de)),
                energy_shift_spread_kcal=float(np.std(de)),
                force_rms_change_eV_A=float(np.sqrt((dfa ** 2).mean())),
                force_max_change_eV_A=float(np.abs(dfa).max())))
        print()
        print("ladder: the difference between adjacent rungs **of the reference itself** (not against MACE; this is how far the reference has converged)")
        for r in ladder:
            print("   {:22s} -> {:22s}  rms change in force {:.4f} eV/A (maximum {:.4f}); "
                  "energy shift {:+.3f} +/- {:.3f} kcal/mol".format(
                      r["lower"], r["upper"], r["force_rms_change_eV_A"],
                      r["force_max_change_eV_A"], r["energy_shift_mean_kcal"],
                      r["energy_shift_spread_kcal"]))
        print("   the **mean** energy shift is meaningless (the absolute zero moved); what matters is its **spread**:")
        print("   a small spread = this basis correction is nearly the same constant for every structure = the relative quantities have converged.")

    # ---- the composite Hessian (a capability that is off by default) ---------------------
    hess = None
    if args.hessian:
        hess = run_hessian(args, chosen, at_min, calc)

    payload = dict(
        generated_by="scripts/calibration/s0_composite_energy_force.py",
        scope=("energies and forces primarily (ruled 2026-08-29 morning); the frequency capability exists but is off by default "
               "(the user asked on the afternoon of 2026-08-29 for it to be able to produce them)"),
        engine=prov, tag=tag,
        recipes={n: t for n, t in chosen},
        pool_terms=[list(x) for x in pool_terms],
        reference_spin=p2.get("reference_spin"),
        orca_binary=orca.orca_binary(),
        n_structures=len(results),
        summary_per_recipe=summary,
        ladder_between_levels=ladder,
        displacement_sampling=displaced_rec,
        composite_hessian=hess,
        note_forces=("comparing forces at a MACE minimum is **degenerate**: the MACE force is about zero, "
                     "and the deviation is identically the reference force. Force accuracy can be read only on **thermally displaced structures**."),
        note_energy=("absolute energies are not comparable (MACE and coupled cluster have different zeros); "
                     "what is compared here are relative quantities within one species."),
        literature_reference=("Allen et al., Reactive Chemistry at Unrestricted Coupled "
                              "Cluster Level: the composite has an rms force error against true CCSD(T)/QZ of "
                              "0.073 eV/Å"),
        per_structure=results, total_seconds=time.time() - t0)
    out = OUT / "composite_energy_force_{}.json".format(tag)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("written:", out)
    print("per-structure resume file:", part)
    print("total time {:.0f} s".format(time.time() - t0))


def run_hessian(args, chosen, at_min, calc):
    """Composite Hessian -- report the cost first, run only once confirmed. Minima only (a displaced structure is not a minimum)."""
    from ase.io import read
    name, terms = chosen[-1]        # the top rung of the ladder
    print()
    print("=" * 100)
    print("composite Hessian (**a capability off by default**, switched on here by --hessian)  recipe {}".format(name))
    print("=" * 100)
    # Estimate the cost from the single-point times **measured in this run**, not from an estimate in the configuration
    spt = {}
    for r in at_min:
        for t in r["per_recipe"][name]["per_term"]:
            spt.setdefault((t["method"], t["basis"]), []).append(t["seconds"])
    spt = {k: float(np.mean(v)) for k, v in spt.items()}
    total_s = 0.0
    for r in at_min:
        c = orca.hessian_cost(r["n_atoms"], terms, spt)
        total_s += c["estimated_seconds"]
        print("   {:20s} basin {}  {} atoms -> {} displaced geometries x {} terms = {} single points, "
              "about {:.1f} hours".format(
                  r["name"], r["basin"], c["n_atoms"], c["n_displaced_geometries"],
                  c["n_terms_per_geometry"], c["n_single_points"],
                  c["estimated_hours"]))
    print("   -- about {:.1f} hours in total ({:.2f} days), from the single-point times **measured in this run**".format(
        total_s / 3600.0, total_s / 86400.0))
    if not args.hessian_confirm_cost:
        print("   **--hessian-confirm-cost was not given: the cost is reported and nothing is run.**")
        return dict(mode="cost_only", recipe=name,
                    estimated_seconds=total_s,
                    estimated_hours=total_s / 3600.0,
                    seconds_per_term_measured={"{}/{}".format(*k): v
                                               for k, v in spt.items()},
                    note="the capability exists, is off by default, and reports its cost first. It runs only with --hessian-confirm-cost.")
    out = []
    for r in at_min:
        a = read(r["path"])
        print("   running {} basin {} ...".format(r["name"], r["basin"]), flush=True)
        h, asym, logs = orca.composite_hessian(
            a.get_chemical_symbols(), a.get_positions(), terms,
            delta_A=args.hessian_delta, nprocs=args.nprocs,
            timeout_s=args.timeout,
            progress=lambda i, n: print("      coordinate {}/{}".format(i, n), flush=True))
        rec = hessian.project_and_diagonalise(h, a.get_masses(), a.get_positions())
        rec.update(name=r["name"], basin=r["basin"], recipe=name,
                   hessian_asymmetry_eV_A2=asym, delta_A=args.hessian_delta,
                   n_single_points=len(logs) * len(orca.unique_terms([terms])))
        # MACE frequencies on the same geometry, for a direct comparison
        a2 = a.copy()
        a2.calc = calc
        hm, _asym_m = hessian.finite_difference_hessian(a2, calc,
                                                        delta=args.hessian_delta)
        rec["mace_frequencies_cm_inv"] = hessian.project_and_diagonalise(
            hm, a.get_masses(), a.get_positions())["frequencies_cm_inv"]
        d = (np.array(rec["frequencies_cm_inv"])
             - np.array(rec["mace_frequencies_cm_inv"]))
        rec["frequency_deviation"] = dict(
            mae_cm_inv=float(np.abs(d).mean()),
            rmse_cm_inv=float(np.sqrt((d ** 2).mean())),
            max_abs_cm_inv=float(np.abs(d).max()),
            signed_mean_cm_inv=float(d.mean()))
        print("      frequency deviation: mean absolute {:.2f}  rms {:.2f}  maximum {:.2f} cm^-1".format(
            rec["frequency_deviation"]["mae_cm_inv"],
            rec["frequency_deviation"]["rmse_cm_inv"],
            rec["frequency_deviation"]["max_abs_cm_inv"]))
        out.append(rec)
    return dict(mode="computed", recipe=name, per_structure=out)


if __name__ == "__main__":
    main()
