# -*- coding: utf-8 -*-
"""Cost model for branch 2 -- **extrapolated from measured probes to all 436 conformers**.

CALIBRATION. Extrapolates measured probes to all conformers to decide what branch 2
can afford. Its product is a budget, not a result.

**Why a model instead of just running it**: the 436 conformers package 1 produced span 8 to 19
atoms, and a `RI-MP2/cc-pVTZ` numerical Hessian needs `6N` displaced gradients, each of which
grows close to the fourth power of the number of basis functions. **The two ends differ by one
to two orders of magnitude**, so starting without measuring that curve means not knowing how long the run takes.

**Model form** (three parameters: two fixed by the probes, one by the structure):

    wall clock (one conformer) = t_opt + 6 * N_atoms * t_gradient(N_basis)
    t_gradient(N) = t0 * (N / N0) ** p

`p` is **measured** from two probes of different size. **Two points fixing an exponent is thin**,
so this script reports `p` and the total extrapolated from it **together**, plus the sensitivity
at `p = 3.0 / 3.5 / 4.0`, so a reader can judge how far to trust the extrapolation.

**Basis functions are counted as spherical-harmonic cc-pVTZ**: H = 14, C/N/O/F = 30.
Checked on 2026-09-02 with the probe molecule `OC(C=O)C=O` (C3H4O3): `6*30 + 4*14 = 236`,
**exactly matching** the 236 in the ORCA output.

Usage::

    python scripts/calibration/s0_branch2_cost_model.py --probes analysis/branch2_probe_10.jsonl,analysis/branch2_probe_15.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

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

from openqha import S0_ROOT

# cc-pVTZ spherical-harmonic basis-function counts
CC_PVTZ = {"H": 14, "C": 30, "N": 30, "O": 30, "F": 30}
BASINS = S0_ROOT / "analysis" / "package1_crest_1_4000__basins.parquet"
MOLECULES = S0_ROOT / "analysis" / "package1_crest_1_4000__molecules.parquet"


def n_basis_from_smiles(smiles):
    from rdkit import Chem
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    n = 0
    for a in m.GetAtoms():
        s = a.GetSymbol()
        if s not in CC_PVTZ:
            raise ValueError("no cc-pVTZ count for {}".format(s))
        n += CC_PVTZ[s]
    return n, m.GetNumAtoms()


def load_probes(paths):
    out = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            o = r.get("orca", {})
            if not o.get("terminated_normally"):
                out.append(dict(qm9_index=r.get("qm9_index"),
                                n_atoms=r.get("n_atoms"), failed=True,
                                error=o.get("error")))
                continue
            si = r.get("simple_input", "")
            # **The level must travel with every point.** Defect 52 (2026-09-02): the first version
            # did not record the level, so it fitted 10-atom RIJK together with 15-atom RIJCOSX,
            # and RIJCOSX is about 3 times faster -- the scaling exponent was flattened to 1.06 and
            # the total cost underestimated more than twofold.
            # **An exponent fitted across mixed levels is meaningless; the points must be grouped.**
            method = ("RIJCOSX" if "RIJCOSX" in si else
                      ("RIJK" if "RIJK" in si else "unknown"))
            out.append(dict(qm9_index=r["qm9_index"], n_atoms=r["n_atoms"],
                            smiles=r.get("smiles"), method=method,
                            n_basis=o.get("n_basis"),
                            n_basis_lines=o.get("n_basis_lines_reported"),
                            wall=r.get("total_wall_seconds"),
                            failed=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes",
                    default="analysis/branch2_probe_10.jsonl,"
                            "analysis/branch2_probe_15_cosx.jsonl,"
                            "analysis/branch2_probe_15_rijk.jsonl,"
                            "analysis/branch2_optfreq_shard0.jsonl,"
                            "analysis/branch2_optfreq_shard1.jsonl,"
                            "analysis/branch2_optfreq_shard2.jsonl")
    ap.add_argument("--jobs", type=int, default=4,
                    help="number of concurrent jobs (4 cores each; 16 cores on this machine)")
    ap.add_argument("--method", default=None,
                    help="fit using points of this level only: RIJK / RIJCOSX. Default: whichever has the most points")
    ap.add_argument("--out", default="analysis/branch2_cost_model.json")
    args = ap.parse_args()

    import pandas as pd

    probes = load_probes(args.probes.split(","))
    good = [p for p in probes if not p["failed"] and p.get("wall")]
    print("=" * 96)
    print("branch 2 cost model")
    print("=" * 96)
    for p in probes:
        if p["failed"]:
            print("probe {:2d} atoms  **failed**  {}".format(
                p.get("n_atoms") or 0, str(p.get("error"))[:70]))
        else:
            print("{:2d} atoms  {}  {:8s}  basis {}  wall {:.0f} s".format(
                p["n_atoms"], p["qm9_index"], p["method"], p["n_basis"],
                p["wall"]))

    # **Fit only within one level.** A mixed-level exponent is meaningless (defect 52).
    by_method = {}
    for r in good:
        by_method.setdefault(r["method"], []).append(r)
    print()
    print("grouped by level: {}".format(
        ", ".join("{} {} point(s)".format(k, len(v)) for k, v in by_method.items())))
    method = args.method or max(by_method, key=lambda k: len(by_method[k]))
    good = by_method.get(method, [])
    print("**fitting the {} point(s) of {}**  (points of other levels take no part)".format(len(good), method))
    if len(good) < 2:
        sys.exit("\n**Fewer than two successful points at level {}; the scaling exponent cannot be fixed.**\n"
                 "  -- no invented number: wait until the second size at this level has finished.".format(method))

    good.sort(key=lambda r: r["n_basis"])
    # Work back from the total wall clock to "the equivalent cost of one displacement": `wall / (6N)`.
    # The optimisation segment is included -- it follows the same scaling and is not a large share
    # (measured on the 10-atom probe: 639/1613 = 40%).
    import numpy as np
    for r in good:
        r["t_per_disp"] = r["wall"] / (6.0 * r["n_atoms"])
    a = good[0]
    ta = a["t_per_disp"]

    if len(good) >= 3:
        # **Three points or more means least squares**; stop fixing an exponent from two points.
        x = np.log([r["n_basis"] / a["n_basis"] for r in good])
        y = np.log([r["t_per_disp"] / ta for r in good])
        p_fit, c0 = np.polyfit(x, y, 1)
        resid = y - (p_fit * x + c0)
        rms = float(np.sqrt((resid ** 2).mean()))
        fit_note = "least squares, {} point(s), root-mean-square log residual {:.3f}".format(len(good), rms)
        ta = ta * math.exp(c0)
    else:
        b = good[-1]
        p_fit = (math.log(b["t_per_disp"] / ta)
                 / math.log(b["n_basis"] / a["n_basis"]))
        fit_note = "**two points fixing an exponent, which is thin** -- three sensitivity values follow"
    print()
    for r in good:
        print("  {:>4d} basis functions  {:7.2f} s equivalent per displacement   ({} atoms, total wall {:.0f} s)".format(
            r["n_basis"], r["t_per_disp"], r["n_atoms"], r["wall"]))
    print("**measured scaling exponent p = {:.2f}**   ({})".format(p_fit, fit_note))

    basins = pd.read_parquet(BASINS)
    mols = pd.read_parquet(MOLECULES)
    nb, na = zip(*mols.smiles.map(n_basis_from_smiles))
    mols["n_basis"] = nb
    mols["n_atoms"] = na
    tbl = basins.merge(mols[["qm9_index", "smiles", "n_atoms", "n_basis"]],
                       on="qm9_index")

    rows = []
    for p in (3.0, 3.5, 4.0, p_fit):
        tot = 0.0
        for _, r in tbl.iterrows():
            t_grad = ta * (r.n_basis / a["n_basis"]) ** p
            tot += 6.0 * r.n_atoms * t_grad
        rows.append(dict(exponent=float(p), total_seconds=float(tot),
                         total_days_serial=float(tot / 86400.0),
                         total_days_parallel=float(tot / 86400.0 / args.jobs)))
    print()
    print("total cost of all {} conformers ({} concurrent jobs x 4 cores):".format(len(tbl), args.jobs))
    print("{:>10s} {:>16s} {:>14s} {:>16s}".format(
        "exponent p", "total core-hours", "serial (days)", "concurrent (days)"))
    for r in rows:
        tag = "  <- measured" if abs(r["exponent"] - p_fit) < 1e-9 else ""
        print("{:>10.2f} {:>16.1f} {:>14.1f} {:>16.1f}{}".format(
            r["exponent"], r["total_seconds"] / 3600.0 * 4,
            r["total_days_serial"], r["total_days_parallel"], tag))

    # split by molecule size -- the heavy tail has to be visible
    tbl["t_est_s"] = [6.0 * r.n_atoms * ta * (r.n_basis / a["n_basis"]) ** p_fit
                      for _, r in tbl.iterrows()]
    by = tbl.groupby("n_atoms").agg(
        n_basins=("basin", "size"), n_basis=("n_basis", "first"),
        hours_each=("t_est_s", lambda s: s.iloc[0] / 3600.0),
        hours_total=("t_est_s", lambda s: s.sum() / 3600.0))
    by["share"] = (by.hours_total / by.hours_total.sum() * 100).round(1)
    print()
    print("by molecule size (measured exponent p = {:.2f}):".format(p_fit))
    print(by.to_string())

    out = Path(args.out)
    out.write_text(json.dumps(dict(
        generated_by="scripts/calibration/s0_branch2_cost_model.py",
        probes=probes, method_fitted=method, fitted_exponent=float(p_fit),
        anchor=dict(n_basis=a["n_basis"], seconds_per_gradient=ta),
        n_basins=int(len(tbl)), jobs=args.jobs,
        projections=rows,
        by_size=json.loads(by.reset_index().to_json(orient="records")),
        note=("Model: wall = 6N * t0 * (N_basis/N0)^p. "
              "**Fitted within one level only** -- mixing RIJK and RIJCOSX flattens the exponent (defect 52). "
              "**Two points fixing an exponent is thin**, so the sensitivity at p=3.0/3.5/4.0 is given as well. "
              "Basis functions are counted as spherical-harmonic cc-pVTZ, H=14 and C/N/O/F=30, "
              "checked against the probe molecule OC(C=O)C=O as 236, matching the ORCA output.")),
        indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("written:", out)


if __name__ == "__main__":
    main()
