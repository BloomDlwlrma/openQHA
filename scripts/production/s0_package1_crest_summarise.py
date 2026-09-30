"""Package 1 - the CREST branch summary -- **runnable at any moment, including while the job is still running**.

PRODUCTION. Summary of the CREST branch; safe to run while the jobs are still going.

--------------------------------------------------------------------------------------
Why this summary can be run at any time
--------------------------------------------------------------------------------------
The batch driver processes molecules in a **fixed-seed random permutation** (configuration `package1.crest_batch.order_seed`).
So **whatever has finished is an unbiased random sample of the 3819 molecules** --
there is no need to wait for the full run (measured at about 20.8 days) before quoting a statistic.

This script **states the current sample size and the width of its 95% interval**, so that
a mean over n = 12 is not quoted as if it were a mean over n = 3819.

--------------------------------------------------------------------------------------
The three questions it answers
--------------------------------------------------------------------------------------
1. **CREST's reported conformer count vs the basin count under this repository's criteria** -- how far apart they are.
2. **Which route misses basins, and what a missed basin is worth in kcal/mol** --
   the only meaningful question once the two routes are pooled.
3. **Cost** -- the distribution of measured per-molecule time, and the extrapolation for what is left.

Usage::

    python scripts/production/s0_package1_crest_summarise.py
Products::
    analysis/package1_crest_summary_<lo>_<hi>.json
    analysis/package1_crest_summary_<lo>_<hi>.png
"""
import argparse
import json
import sys
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
from openqha import S0_ROOT, config, crest_census

CFG = config.load()
P1 = config.package(1, CFG)
T_REF = config.temperature(CFG)
TARGET = float(CFG["thermodynamics"]["target_accuracy_kcal"])


def ci95(x):
    """Half-width of the 95% interval of the mean (normal approximation). **When the sample is small this is the point.**"""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan")
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=1)
    ap.add_argument("--hi", type=int, default=4000)
    args = ap.parse_args()

    root = (S0_ROOT / "analysis" / "package1" / "crest"
            / "{}_{}".format(*tuple(P1["qm9_range"]))
            / "{}_{}".format(args.lo, args.hi))
    mol_dir = root / "mol"
    if not mol_dir.exists():
        raise FileNotFoundError("no products yet: {}".format(mol_dir))

    ok, failed = [], []
    for p in sorted(mol_dir.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        (failed if p.name.endswith(".FAILED.json") else ok).append(d)

    man = root / "manifest.json"
    n_total = None
    if man.exists():
        n_total = len(json.loads(man.read_text(encoding="utf-8"))
                      .get("processing_order", []))

    print("=" * 104)
    print("package 1 - CREST branch   summary   indices {}-{}".format(args.lo, args.hi))
    print("=" * 104)
    print("{} done / {} planned ({:.2%}); {} failed".format(
        len(ok), n_total, len(ok) / max(n_total or 1, 1), len(failed)))
    print("**The processing order is a random permutation with fixed seed {}, so these {} are an unbiased random sample**".format(
        P1["crest_batch"]["order_seed"], len(ok)))
    if not ok:
        print("no molecule succeeded.")
        return

    nc = np.array([r["n_conformers_reported_by_crest"] for r in ok])
    nb = np.array([r["n_basins"] for r in ok])
    co = np.array([r["n_basins_crest_only"] for r in ok])
    eo = np.array([r["n_basins_etkdg_only"] for r in ok])
    bo = np.array([r["n_basins_both"] for r in ok])
    corr = np.array([r["conformational_correction_kcal"] for r in ok])
    w0 = np.array([r["populations"]["weight_of_lowest"] for r in ok])
    sec = np.array([r.get("total_seconds") or np.nan for r in ok], dtype=float)
    sad = np.array([r["n_saddles_rejected"] for r in ok])
    # Basins admitted with an inversion window: the lowest mode lies in
    # [ithr, 0) and the thermochemistry inverts it -- production's new class. Absent on
    # records written before 2026-09-25, which admitted no window basins.
    nwin = np.array([sum(int(h.get("n_inversion_window") or 0)
                         for h in (r.get("basin_hessian") or [])) for r in ok])
    gch = np.array([r["tighten_crest"]["n_graph_changed"] for r in ok])
    steps_c = np.concatenate([r["tighten_crest"]["opt_steps_per_frame"] for r in ok])
    _se = [r["tighten_etkdg"]["opt_steps_per_frame"] for r in ok
           if r.get("tighten_etkdg")]
    steps_e = np.concatenate(_se) if _se else np.array([np.nan])

    # ---- how much a single route misses ------------------------------------------------
    # From the pooled basin list, select the "CREST only" and "ETKDG only" subsets by
    # source, recompute the correction for each and compare against the pooled correction.
    # **That is the correct way to price a missed basin** -- not comparing basin counts.
    only_crest_err, only_etkdg_err = [], []
    for r in ok:
        rel = np.asarray(r["basin_relative_kcal"])
        prov = r["basin_provenance"]
        full = crest_census.conformational_correction_kcal(rel, T_REF)
        for labs, bag in (("crest", only_crest_err), ("etkdg", only_etkdg_err)):
            sel = [i for i, p in enumerate(prov) if labs in p]
            if not sel:
                bag.append(float("nan"))
                continue
            sub = rel[sel] - rel[sel].min()
            bag.append(crest_census.conformational_correction_kcal(sub, T_REF) - full)
    only_crest_err = np.asarray(only_crest_err)
    only_etkdg_err = np.asarray(only_etkdg_err)

    print()
    print("[1] CREST reported conformer count vs the basin count under this repository criteria")
    print("  CREST conformers  mean {:.2f} +/- {:.2f}(95%)  median {:.0f}  maximum {}".format(
        nc.mean(), ci95(nc), np.median(nc), nc.max()))
    print("  pooled basins     mean {:.2f} +/- {:.2f}(95%)  median {:.0f}  maximum {}".format(
        nb.mean(), ci95(nb), np.median(nb), nb.max()))
    print("  tightening steps  CREST side mean {:.1f} max {}; ETKDG side mean {:.1f} max {}".format(
        steps_c.mean(), steps_c.max(), np.nanmean(steps_e), np.nanmax(steps_e)))
    print("                    **the step count measures how far the supplied geometry is from a minimum of this potential**")

    print()
    print("[2] which route misses basins, and what that is worth")
    print("  basin source      CREST only {:.2f} +/- {:.2f}; ETKDG only {:.2f} +/- {:.2f}; "
          "both {:.2f}".format(co.mean(), ci95(co), eo.mean(), ci95(eo), bo.mean()))
    print("  molecules with a CREST-only basin {:.1%}; with an ETKDG-only basin {:.1%}".format(
        (co > 0).mean(), (eo > 0).mean()))
    print("  conformational-correction error using CREST only   mean {:+.4f}  max {:+.4f} kcal/mol".format(
        np.nanmean(only_crest_err), np.nanmax(np.abs(only_crest_err))))
    print("  conformational-correction error using ETKDG only   mean {:+.4f}  max {:+.4f} kcal/mol".format(
        np.nanmean(only_etkdg_err), np.nanmax(np.abs(only_etkdg_err))))
    print("  molecules past the {} kcal/mol criterion scale: CREST only {}; ETKDG only {}".format(
        TARGET, int((np.abs(only_crest_err) > TARGET).sum()),
        int((np.abs(only_etkdg_err) > TARGET).sum())))

    print()
    print("[3] properties of the basin list")
    print("  lowest-basin weight  mean {:.3f}; fraction with weight < 0.9 is {:.1%}".format(
        w0.mean(), (w0 < 0.9).mean()))
    print("  conformational correction  mean {:+.4f} +/- {:.4f}  median {:+.4f}  most negative {:+.4f}".format(
        corr.mean(), ci95(corr), np.median(corr), corr.min()))
    print("  saddle points rejected (below the frequency floor ithr)  {} molecule(s) ({:.1%}), {} in total".format(
        int((sad > 0).sum()), (sad > 0).mean(), int(sad.sum())))
    print("  basins admitted with an inversion window  {} molecule(s) ({:.1%}), {} window mode(s) in total".format(
        int((nwin > 0).sum()), (nwin > 0).mean(), int(nwin.sum())))
    print("  connectivity matrix changed  {} molecule(s) ({:.1%}) -- each must be inspected".format(
        int((gch > 0).sum()), (gch > 0).mean()))

    per_job = float(np.nanmean(sec))
    remain = (n_total or len(ok)) - len(ok)
    par = 4
    print()
    print("[4] cost")
    print("  measured per job  mean {:.0f} s  median {:.0f} s  maximum {:.0f} s".format(
        per_job, np.nanmedian(sec), np.nanmax(sec)))
    print("  {} left; extrapolated over {} parallel slots: {:.1f} hours = {:.1f} days".format(
        remain, par, per_job * remain / par / 3600, per_job * remain / par / 86400))

    if failed:
        print()
        print("[5] {} failure(s):".format(len(failed)))
        for f in failed[:20]:
            print("   {} {:24s} {}: {}".format(f["qm9_index"], f["smiles"],
                                               f["error_type"], f["error"][:70]))

    payload = dict(
        qm9_range=[args.lo, args.hi], n_done=len(ok), n_planned=n_total,
        n_failed=len(failed), temperature_K=T_REF,
        sample_is_unbiased=True,
        order_seed=int(P1["crest_batch"]["order_seed"]),
        sample_note="the processing order is a random permutation with a fixed seed, so the finished batch is an unbiased random sample; "
                    "every mean carries the half-width of its 95% interval, and with a small sample the interval is what counts",
        crest_conformers=dict(mean=float(nc.mean()), ci95=ci95(nc),
                              median=float(np.median(nc)), max=int(nc.max())),
        pooled_basins=dict(mean=float(nb.mean()), ci95=ci95(nb),
                           median=float(np.median(nb)), max=int(nb.max())),
        tighten_steps=dict(crest_mean=float(steps_c.mean()),
                           crest_max=int(steps_c.max()),
                           etkdg_mean=float(np.nanmean(steps_e)),
                           etkdg_max=float(np.nanmax(steps_e))),
        provenance=dict(crest_only_mean=float(co.mean()), crest_only_ci95=ci95(co),
                        etkdg_only_mean=float(eo.mean()), etkdg_only_ci95=ci95(eo),
                        both_mean=float(bo.mean()),
                        frac_with_crest_only=float((co > 0).mean()),
                        frac_with_etkdg_only=float((eo > 0).mean())),
        correction_error_if_single_route=dict(
            only_crest_mean=float(np.nanmean(only_crest_err)),
            only_crest_max_abs=float(np.nanmax(np.abs(only_crest_err))),
            only_etkdg_mean=float(np.nanmean(only_etkdg_err)),
            only_etkdg_max_abs=float(np.nanmax(np.abs(only_etkdg_err))),
            n_above_target_only_crest=int((np.abs(only_crest_err) > TARGET).sum()),
            n_above_target_only_etkdg=int((np.abs(only_etkdg_err) > TARGET).sum()),
            target_accuracy_kcal=TARGET,
            method="select subsets by source from the pooled basin list, recompute the correction and subtract it from the pooled one -- "
                   "**not** a comparison of basin counts"),
        populations=dict(weight_of_lowest_mean=float(w0.mean()),
                         frac_below_0p9=float((w0 < 0.9).mean())),
        correction=dict(mean=float(corr.mean()), ci95=ci95(corr),
                        median=float(np.median(corr)), min=float(corr.min())),
        saddles=dict(n_molecules=int((sad > 0).sum()), n_total=int(sad.sum())),
        inversion_windows=dict(n_molecules=int((nwin > 0).sum()),
                               n_modes_total=int(nwin.sum())),
        graph_changed=dict(n_molecules=int((gch > 0).sum())),
        cost=dict(seconds_per_job_mean=per_job,
                  seconds_per_job_median=float(np.nanmedian(sec)),
                  seconds_per_job_max=float(np.nanmax(sec)),
                  n_remaining=int(remain), parallel_slots=par,
                  projected_days=float(per_job * remain / par / 86400)),
        failures=[dict(qm9_index=f["qm9_index"], smiles=f["smiles"],
                       error_type=f["error_type"], error=f["error"])
                  for f in failed])
    out = S0_ROOT / "analysis" / "package1_crest_summary_{}_{}.json".format(
        args.lo, args.hi)
    crest_census.dump_json(payload, out)
    print()
    print("written: {}".format(out))

    _plot(nc, nb, co, eo, only_crest_err, only_etkdg_err, args)


def _plot(nc, nb, co, eo, ec, ee, args):
    # **Do not call matplotlib.use("Agg")** -- figures were silently lost that way twice.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    m = max(nc.max(), nb.max())
    ax[0].scatter(nc, nb, s=28, alpha=0.7)
    ax[0].plot([0, m], [0, m], "k--", lw=1)
    ax[0].set_xlabel("CREST reported conformers")
    ax[0].set_ylabel("basins under repo criteria (pooled)")
    ax[0].set_title("CREST count is not a basin count")
    bins = np.arange(-0.5, max(co.max(), eo.max()) + 1.5)
    ax[1].hist([co, eo], bins=bins, label=["CREST only", "ETKDG only"])
    ax[1].set_xlabel("basins unique to one route")
    ax[1].set_ylabel("molecules")
    ax[1].legend()
    ax[1].set_title("the two routes are complementary")
    ax[2].hist([ec[~np.isnan(ec)], ee[~np.isnan(ee)]], bins=20,
               label=["only CREST", "only ETKDG"])
    ax[2].axvline(TARGET, color="r", ls="--", lw=1)
    ax[2].set_xlabel("error in conformational correction / kcal/mol")
    ax[2].set_ylabel("molecules")
    ax[2].legend()
    ax[2].set_title("what one route alone costs")
    fig.tight_layout()
    png = S0_ROOT / "analysis" / "package1_crest_summary_{}_{}.png".format(
        args.lo, args.hi)
    fig.savefig(png, dpi=140)
    print("written: {}".format(png))


if __name__ == "__main__":
    main()
