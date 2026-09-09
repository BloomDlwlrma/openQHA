"""Concatenate the per-molecule parquet files into the global tables -- run on the cluster after the compute job via `--dependency`.

PRODUCTION. Concatenates the per-molecule parquet files into the global basin table.

Every molecule `mol/<index>.parquet` is the **same shape, one row per basin**, so the
summary is a single `concat` and **nothing has to be recomputed**.

Products::

    <result root>/<segment>/collected/basins.parquet      one row per basin (all molecules)
    <result root>/<segment>/collected/molecules.parquet   one row per molecule (the molecule-level columns, deduplicated)
    <result root>/<segment>/collected/failures.parquet    one row per failure
    <result root>/<segment>/collected/summary.log         overview, for a person to read

Usage::

    python scripts/production/s0_package1_collect.py --root <result root>/result-s0tf_1_16000/1_4000
    python scripts/production/s0_package1_collect.py --root ... --stage stage2-mace
"""
import argparse
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
from openqha import S0_ROOT, record, report

#: The molecule-level columns (everything else is basin-level). Used with drop_duplicates to split out the molecule table.
BASIN_COLS = ("basin", "energy_eV", "relative_kcal", "provenance",
              "boltzmann_weight", "n_imaginary", "n_rigid_modes_removed",
              "lowest_frequency_cm_inv", "hessian_asymmetry_eV_A2",
              "frequencies_cm_inv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="the segment directory under the result root, for example .../result-s0tf_1_16000/1_4000")
    ap.add_argument("--stage", default="", help="subdirectory name (when staging)")
    ap.add_argument("--out", default="", help="output directory; defaults to <root>/<stage>/collected")
    args = ap.parse_args()

    import pandas as pd

    root = Path(args.root)
    if not root.is_absolute():
        root = S0_ROOT / root
    if args.stage:
        root = root / args.stage
    mol_dir = root / "mol"
    if not mol_dir.exists():
        raise SystemExit("no mol/ directory: {}".format(mol_dir))
    out = Path(args.out) if args.out else (root / "collected")
    out.mkdir(parents=True, exist_ok=True)

    good = sorted(p for p in mol_dir.glob("*.parquet") if ".FAILED" not in p.name)
    bad = sorted(mol_dir.glob("*.FAILED.parquet"))

    print("=" * 96)
    print("collecting the per-molecule parquet files -> global tables")
    print("=" * 96)
    print("source    {}".format(mol_dir))
    print("{} succeeded, {} failed".format(len(good), len(bad)))
    if not good:
        raise SystemExit("there is nothing to collect.")

    # ---- concatenate, in chunks so that memory is not eaten all at once ----------------
    chunks, step = [], 2000
    for i in range(0, len(good), step):
        chunks.append(pd.concat([pd.read_parquet(p) for p in good[i:i + step]],
                                ignore_index=True))
        print("   read {}/{}".format(min(i + step, len(good)), len(good)), flush=True)
    basins = pd.concat(chunks, ignore_index=True)
    del chunks

    mols = basins.drop(columns=[c for c in BASIN_COLS if c in basins.columns]) \
                 .drop_duplicates(subset="qm9_index").reset_index(drop=True)

    basins.to_parquet(out / "basins.parquet", index=False)
    mols.to_parquet(out / "molecules.parquet", index=False)
    print("   wrote basins.parquet    {} row(s) x {} column(s)".format(*basins.shape))
    print("   wrote molecules.parquet {} row(s) x {} column(s)".format(*mols.shape))

    fails = None
    if bad:
        fails = pd.concat([pd.read_parquet(p) for p in bad], ignore_index=True)
        fails.to_parquet(out / "failures.parquet", index=False)
        print("   wrote failures.parquet {} row(s)".format(len(fails)))

    # ---- the overview .log ---------------------------------------------------------------
    r = report.Report("package 1 - CREST branch   global summary",
                      subtitle=str(root))
    r.section("scale")
    r.kv("molecules", len(mols))
    r.kv("basins", len(basins))
    r.kv("failures", len(fails) if fails is not None else 0)
    r.section("tables")
    rows = [["basins.parquet", len(basins), "one row per basin (with every frequency of that basin)"],
            ["molecules.parquet", len(mols), "one row per molecule"]]
    if fails is not None:
        rows.append(["failures.parquet", len(fails), "one row per failed molecule"])
    r.table(["file", "rows", "content"], rows)

    r.section("distributions")
    stats = []
    for k, unit in (("n_basins", ""), ("n_conformers_crest", ""),
                    ("n_basins_crest_only", ""), ("n_basins_etkdg_only", ""),
                    ("n_saddles_rejected", ""),
                    ("conformational_correction_kcal", "kcal/mol"),
                    ("weight_of_lowest", ""), ("crest_seconds", "s"),
                    ("analysis_seconds", "s"), ("total_seconds", "s"),
                    ("tighten_steps_max", "")):
        if k not in mols.columns:
            continue
        v = pd.to_numeric(mols[k], errors="coerce").dropna().to_numpy()
        if not len(v):
            continue
        stats.append([k, unit or "-", len(v), "{:.4f}".format(v.mean()),
                      "{:.4f}".format(np.median(v)), "{:.4f}".format(v.min()),
                      "{:.4f}".format(v.max())])
    if stats:
        r.table(["quantity", "unit", "n", "mean", "median", "minimum", "maximum"], stats)

    if "crest_terminated_early" in mols.columns:
        e = pd.to_numeric(mols["crest_terminated_early"], errors="coerce").fillna(0)
        r.section("criteria")
        r.verdict("every molecule CREST run must have terminated EARLY exactly 0 times",
                  "{} molecule(s) are non-zero".format(int((e > 0).sum())), bool((e > 0).sum() == 0))
    if "n_imaginary" in basins.columns:
        im = pd.to_numeric(basins["n_imaginary"], errors="coerce").fillna(0)
        r.verdict("a structure entering the basin list must have no imaginary frequency",
                  "{} basin(s) have a non-zero imaginary count".format(int((im > 0).sum())),
                  bool((im > 0).sum() == 0))
    if "shake_fallback_used" in mols.columns:
        n_fb = int(mols["shake_fallback_used"].notna().sum())
        r.section("SHAKE fallback")
        r.kv("molecules_using_fallback", n_fb,
             note="these molecules hit terminated EARLY under the default SHAKE and succeeded on a retry with shake=1")

    r.section("how to read this")
    r.note("import pandas as pd; df = pd.read_parquet('collected/basins.parquet')")
    r.note("**The precision lives in the parquet** (raw float64); the numbers in this .log are "
           "formatted for reading, so do not use them for digit-for-digit comparison.")
    p = r.write(out / "summary.log")
    print()
    print("written: {}".format(out))
    print("      {}".format(p.name))


if __name__ == "__main__":
    main()
