"""Turn the JSON under `analysis/` into **something readable** -- `.log` text plus parquet tables.

TOOLING. Converts JSON under analysis/ into readable text plus parquet tables.
Produces no science.

User, 2026-08-31: "`analysis/` is all JSON, I cannot read it".

**The approach: three layers, each with one job**:

| layer | product | who reads it |
|---|---|---|
| archive | `xxx.json` | machines; the complete record the checkpoint standard requires. **Never deleted, never edited** |
| report | `xxx.log` | **people**. ORCA/CREST style: banner, sections, dotted leaders, aligned tables, units in the header |
| table | `xxx__<table>.parquet` | pandas. **Generated only where there are many numbers of the same shape** |

**Both the `.log` and the parquet are generated from the JSON**, so the three cannot drift;
when the JSON changes, regenerate. **The JSON remains the single source of truth.**

The two per-molecule directories (4000 JSON files each) are **not converted one by one**
-- 4000 text files are just as unreadable. They become **parquet tables**:
one row per molecule, one row per basin, one row per frequency, queried in one line of pandas.

Usage::

    python scripts/tooling/s0_analysis_to_readable.py            # everything
    python scripts/tooling/s0_analysis_to_readable.py --only-logs
    python scripts/tooling/s0_analysis_to_readable.py --only-parquet
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
from openqha import S0_ROOT, report

ANALYSIS = S0_ROOT / "analysis"

#: The two per-molecule JSON collections: **not converted one by one**, converted to parquet
PER_MOLECULE = (
    ("etkdg", ANALYSIS / "package1" / "1_16000" / "1_4000" / "mol"),
    ("crest", ANALYSIS / "package1" / "crest" / "1_16000" / "1_4000" / "mol"),
)


# ======================================================================================
# 1. general: one .log per JSON
# ======================================================================================
def convert_logs():
    skip_dirs = {p.resolve() for _, p in PER_MOLECULE if p.exists()}
    made = []
    for p in sorted(ANALYSIS.rglob("*.json")):
        if any(d in p.resolve().parents for d in skip_dirs) or p.resolve().parent in skip_dirs:
            continue
        if p.name.endswith(".partial.jsonl"):
            continue
        try:
            out = report.json_to_log(p)
            made.append((out, out.stat().st_size))
        except Exception as exc:                     # record it all; swallow nothing
            print("   conversion failed {} -- {}: {}".format(p.name, type(exc).__name__, exc))
    return made


# ======================================================================================
# 2. the per-molecule collections -> parquet
# ======================================================================================
def _crest_rows(d):
    """one CREST-branch record -> (molecule row, basin rows, frequency rows, tightening rows)."""
    q = d["qm9_index"]
    run = d.get("crest_run") or {}
    tc = d.get("tighten_crest") or {}
    te = d.get("tighten_etkdg") or {}
    mol = dict(
        qm9_index=q, smiles=d.get("smiles"),
        n_conformers_crest=d.get("n_conformers_reported_by_crest"),
        n_etkdg_basins_in=d.get("n_etkdg_basins_in"),
        n_pooled_before_hessian=(d.get("pooling") or {}).get("n_global_before_hessian"),
        n_basins=d.get("n_basins"),
        n_basins_crest_only=d.get("n_basins_crest_only"),
        n_basins_etkdg_only=d.get("n_basins_etkdg_only"),
        n_basins_both=d.get("n_basins_both"),
        n_saddles_rejected=d.get("n_saddles_rejected"),
        conformational_correction_kcal=d.get("conformational_correction_kcal"),
        weight_of_lowest=(d.get("populations") or {}).get("weight_of_lowest"),
        n_within_1kT=(d.get("populations") or {}).get("n_within_1kT"),
        crest_seconds=run.get("seconds"),
        total_seconds=d.get("total_seconds"),
        total_engrad_calls=run.get("total_engrad_calls"),
        crest_terminated_normally=run.get("terminated_normally"),
        crest_terminated_early=run.get("n_terminated_early"),
        tighten_steps_crest_max=max(tc.get("opt_steps_per_frame") or [0]),
        tighten_steps_etkdg_max=max(te.get("opt_steps_per_frame") or [0]),
        n_graph_changed=tc.get("n_graph_changed"),
        start_geometry_source=(d.get("start_geometry") or {}).get("source"),
        temperature_K=d.get("temperature_K"))
    basins, freqs = [], []
    prov = d.get("basin_provenance") or []
    hs = d.get("basin_hessian") or []
    for i, (e, rel) in enumerate(zip(d.get("basin_energies_eV", []),
                                     d.get("basin_relative_kcal", []))):
        h = hs[i] if i < len(hs) else {}
        basins.append(dict(qm9_index=q, basin=i, energy_eV=e, relative_kcal=rel,
                           provenance="+".join(prov[i]) if i < len(prov) else None,
                           n_imaginary=h.get("n_imaginary"),
                           lowest_frequency_cm_inv=h.get("lowest_frequency_cm_inv"),
                           n_rigid_modes_removed=h.get("n_rigid_modes_removed"),
                           boltzmann_weight=((d.get("populations") or {})
                                             .get("boltzmann_weights") or [None] * 99)[i]))
        for k, nu in enumerate(h.get("frequencies_cm_inv") or []):
            freqs.append(dict(qm9_index=q, basin=i, mode=k, frequency_cm_inv=nu))
    return mol, basins, freqs


def _etkdg_rows(d):
    q = d.get("qm9_index")
    pop = d.get("populations") or {}
    mol = dict(
        qm9_index=q, smiles=d.get("smiles"),
        n_embed_returned=d.get("n_embed_returned"),
        n_after_forcefield_prune=d.get("n_after_forcefield_prune"),
        n_conformers_optimised=d.get("n_conformers_optimised"),
        n_basins=d.get("n_basins"),
        n_not_converged=d.get("n_not_converged"),
        n_graph_changed=d.get("n_graph_changed"),
        n_merge_warnings=len(d.get("merge_energy_warnings") or []),
        weight_of_lowest=pop.get("weight_of_lowest"),
        n_within_1kT=pop.get("n_within_1kT"), n_within_5kT=pop.get("n_within_5kT"),
        max_residual_force_eV_A=d.get("max_residual_force_eV_A"),
        seconds=d.get("seconds"), n_atoms=d.get("n_atoms"),
        n_heavy_atoms=d.get("n_heavy_atoms"),
        reference_nearest_basin_rank=(d.get("reference_geometry") or {})
        .get("nearest_basin_rank_zero_based"),
        reference_is_own_basin=(d.get("reference_geometry") or {}).get("is_own_basin"))
    basins = []
    w = pop.get("boltzmann_weights") or []
    for i, (e, rel) in enumerate(zip(d.get("basin_energies_eV", []),
                                     d.get("basin_relative_kcal", []))):
        basins.append(dict(qm9_index=q, basin=i, energy_eV=e, relative_kcal=rel,
                           boltzmann_weight=w[i] if i < len(w) else None))
    return mol, basins


def convert_parquet():
    made = []
    for tag, d in PER_MOLECULE:
        if not d.exists():
            continue
        mols, basins, freqs, fails = [], [], [], []
        for p in sorted(d.glob("*.json")):
            obj = json.loads(p.read_text(encoding="utf-8"))
            if p.name.endswith(".FAILED.json"):
                fails.append(dict(qm9_index=obj.get("qm9_index"),
                                  smiles=obj.get("smiles"),
                                  error_type=obj.get("error_type"),
                                  error=obj.get("error")))
                continue
            if tag == "crest":
                m, b, f = _crest_rows(obj)
                freqs += f
            else:
                m, b = _etkdg_rows(obj)
            mols.append(m)
            basins += b
        tables = dict(molecules=mols, basins=basins, failures=fails)
        if freqs:
            tables["frequencies"] = freqs
        stem = ANALYSIS / "package1_{}_1_4000".format(tag)
        made += report.write_parquet(tables, stem)
        _collection_log(tag, stem, mols, basins, freqs, fails)
    return made


def _collection_log(tag, stem, mols, basins, freqs, fails):
    """Write one **overview** `.log` per per-molecule collection -- the entry point to 4000 JSON files."""
    import numpy as np
    name = {"etkdg": "package 1 - ETKDG census", "crest": "package 1 - CREST branch"}[tag]
    r = report.Report(name + "   indices 1-4000",
                      subtitle="overview of the per-molecule records; the detail is in the parquet table of the same name")
    r.section("what this file is")
    r.note("The complete per-molecule record is {} JSON files, and converting them one by one to text "
           "is just as unreadable. So the detail goes to parquet and this file is the overview and entry point.".format(len(mols)))
    r.note("How to read it: import pandas as pd; "
           "df = pd.read_parquet('analysis/{}__molecules.parquet')".format(stem.name))
    r.section("scale")
    r.kv("molecules_done", len(mols))
    r.kv("basins_total", len(basins))
    if freqs:
        r.kv("frequencies_total", len(freqs))
    r.kv("failures", len(fails))
    r.section("tables")
    rows = [("{}__molecules.parquet".format(stem.name), len(mols),
             "one row per molecule"),
            ("{}__basins.parquet".format(stem.name), len(basins),
             "one row per basin (energy, relative energy, Boltzmann weight, source, imaginary count)")]
    if freqs:
        rows.append(("{}__frequencies.parquet".format(stem.name), len(freqs),
                     "one row per normal mode"))
    if fails:
        rows.append(("{}__failures.parquet".format(stem.name), len(fails),
                     "one row per failed molecule (exception type and message)"))
    r.table(["file", "rows", "content"], rows)

    if mols:
        r.section("distributions (current sample)")
        def col(k):
            v = [m[k] for m in mols if m.get(k) is not None]
            return np.asarray(v, dtype=float) if v else None
        stats = []
        for k in ("n_basins", "conformational_correction_kcal", "weight_of_lowest",
                  "n_saddles_rejected", "total_seconds", "seconds",
                  "n_basins_crest_only", "n_basins_etkdg_only"):
            a = col(k)
            if a is None or not len(a):
                continue
            stats.append([report.strip_unit(k), report.unit_of(k) or "-", len(a),
                          "{:.4f}".format(a.mean()), "{:.4f}".format(np.median(a)),
                          "{:.4f}".format(a.min()), "{:.4f}".format(a.max())])
        if stats:
            r.table(["quantity", "unit", "n", "mean", "median", "minimum", "maximum"], stats)
    if fails:
        r.section("failures")
        r.table(["index", "SMILES", "exception", "message"],
                [[f["qm9_index"], f["smiles"], f["error_type"],
                  (f["error"] or "")[:48]] for f in fails[:30]])
    p = r.write(stem.with_suffix(".log"))
    print("   overview: {}".format(p.name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-logs", action="store_true")
    ap.add_argument("--only-parquet", action="store_true")
    args = ap.parse_args()

    print("=" * 96)
    print("converting the JSON under analysis/ into a readable form")
    print("=" * 96)
    print("The JSON remains the single source of truth; the .log and the parquet are generated from it, so they cannot drift.")
    print()

    if not args.only_parquet:
        print("[1] one .log per JSON")
        made = convert_logs()
        for p, n in made:
            print("   {:52s} {:8.1f} KB".format(p.name, n / 1024))
        print("   {} in total".format(len(made)))

    if not args.only_logs:
        print()
        print("[2] per-molecule collections -> parquet (4000 JSON files are not converted one by one)")
        made = convert_parquet()
        for p, n, cols in made:
            print("   {:52s} {:6d} row(s)  {} column(s)".format(p.name, n, len(cols)))

    print()
    print("how to read them:")
    print("   .log      open it directly; ORCA/CREST style")
    print("   .parquet  import pandas as pd; pd.read_parquet(path)")


if __name__ == "__main__":
    main()
