"""Filing of per-molecule products -- **`.log` plus `.parquet`, JSON no longer written**.

--------------------------------------------------------------------------------------
The user's ruling of 2026-08-31 and what follows from it
--------------------------------------------------------------------------------------
"we do not want openQHA to save json format files, save them in the log and parquet file
formats I asked for earlier".

So four things are written per molecule:

    out/crest_<index>.out       **CREST's native output, kept verbatim, only renamed**
    log/<index>.log             the full human-readable record (ORCA/CREST style)
    mol/<index>.parquet         the machine-readable table: **one row per basin**, with
                                molecule-level fields repeated
    xyz/<index>_basins.xyz      the basin geometries

**Which one is "the one that carries the precision"**: the `.parquet`. It stores the raw
float64 values. The `.log` is rendered text with numbers laid out to 6 decimal places --
enough to read, but **do not use the numbers in a `.log` for a digit-by-digit
comparison** (several conclusions in this repository rest on energies being identical to
the last digit, acetone's -5259.524313 eV for one). **To compare digits, read the
parquet.**

--------------------------------------------------------------------------------------
Why the parquet is "one row per basin" rather than "one row per molecule with nested columns"
--------------------------------------------------------------------------------------
One row per basin, with molecule-level fields repeated, is the classic denormalised table:

* **concatenation costs nothing** -- `pd.concat` over 122k files IS the global basin table;
* **nothing needs nesting** (except each basin's frequencies, which use `list<double>`);
* molecule-level statistics come out of `drop_duplicates("qm9_index")`.

Parquet does support a nested struct, but every consumer downstream would then have to
unpack it every time, which is not worth it.
"""
import gzip
import shutil
from pathlib import Path

import numpy as np

from . import report


# ======================================================================================
# 1. parquet: one row per basin
# ======================================================================================
def record_to_rows(rec):
    """One per-molecule record -> several rows, one per basin. Molecule-level fields are
    repeated on every row."""
    run = rec.get("crest_run") or {}
    tc = rec.get("tighten_crest") or {}
    te = rec.get("tighten_etkdg") or {}
    pool = rec.get("pooling") or {}
    pop = rec.get("populations") or {}
    start = rec.get("start_geometry") or {}
    worker = rec.get("worker") or {}
    steps = tc.get("opt_steps_per_frame") or []

    mol_level = dict(
        qm9_index=rec.get("qm9_index"), smiles=rec.get("smiles"),
        temperature_K=rec.get("temperature_K"),
        n_conformers_crest=rec.get("n_conformers_reported_by_crest"),
        n_etkdg_basins_in=rec.get("n_etkdg_basins_in"),
        n_pooled_frames=pool.get("n_pooled_frames"),
        n_global_before_hessian=pool.get("n_global_before_hessian"),
        n_basins=rec.get("n_basins"),
        n_basins_crest_only=rec.get("n_basins_crest_only"),
        n_basins_etkdg_only=rec.get("n_basins_etkdg_only"),
        n_basins_both=rec.get("n_basins_both"),
        n_saddles_rejected=rec.get("n_saddles_rejected"),
        conformational_correction_kcal=rec.get("conformational_correction_kcal"),
        # the floor the Hessian screen applied (ticket 41): the counters n_below_ithr /
        # n_inversion_window in the basin rows belong to this number, so it travels
        # with them into the machine-readable table
        ithr_cm=rec.get("ithr_cm"),
        weight_of_lowest=pop.get("weight_of_lowest"),
        n_within_1kT=pop.get("n_within_1kT"),
        n_within_5kT=pop.get("n_within_5kT"),
        # --- run and diagnostics ---
        crest_seconds=run.get("seconds"),
        crest_terminated_normally=run.get("terminated_normally"),
        crest_terminated_early=run.get("n_terminated_early"),
        crest_engrad_calls=run.get("total_engrad_calls"),
        crest_reused_scratch=run.get("reused_scratch"),
        shake_fallback_used=run.get("shake_fallback_used"),
        crest_version=(run.get("crest") or {}).get("version"),
        crest_commit=(run.get("crest") or {}).get("commit"),
        analysis_seconds=rec.get("analysis_seconds"),
        total_seconds=rec.get("total_seconds"),
        tighten_steps_total=tc.get("opt_steps_total"),
        tighten_steps_max=int(max(steps)) if steps else None,
        tighten_steps_mean=float(np.mean(steps)) if steps else None,
        tighten_steps_etkdg_total=te.get("opt_steps_total"),
        n_graph_changed=tc.get("n_graph_changed"),
        n_not_converged=tc.get("n_not_converged"),
        max_residual_force_eV_A=tc.get("max_residual_force_eV_A"),
        start_geometry_source=start.get("source"),
        dedup_threshold_A=rec.get("dedup_threshold_A"),
        tighten_fmax_eV_A=rec.get("tighten_fmax_eV_A"),
        worker_pid=worker.get("pid"), worker_device=worker.get("device"),
        stage=rec.get("stage"))

    prov = rec.get("basin_provenance") or []
    hs = rec.get("basin_hessian") or []
    w = pop.get("boltzmann_weights") or []
    rows = []
    for i, (e, rel) in enumerate(zip(rec.get("basin_energies_eV", []),
                                     rec.get("basin_relative_kcal", []))):
        h = hs[i] if i < len(hs) else {}
        r = dict(mol_level)
        r.update(basin=i, energy_eV=float(e), relative_kcal=float(rel),
                 provenance="+".join(prov[i]) if i < len(prov) else None,
                 boltzmann_weight=(float(w[i]) if i < len(w) else None),
                 n_imaginary=h.get("n_imaginary"),
                 n_rigid_modes_removed=h.get("n_rigid_modes_removed"),
                 lowest_frequency_cm_inv=h.get("lowest_frequency_cm_inv"),
                 n_below_ithr=h.get("n_below_ithr"),
                 n_inversion_window=h.get("n_inversion_window"),
                 hessian_asymmetry_eV_A2=h.get("hessian_asymmetry_eV_A2"),
                 frequencies_cm_inv=[float(x) for x in
                                     (h.get("frequencies_cm_inv") or [])])
        rows.append(r)
    if not rows:                       # keep a row even with no basins, or this molecule
                                       # disappears from the table
        r = dict(mol_level)
        r.update(basin=None, energy_eV=None, relative_kcal=None, provenance=None,
                 boltzmann_weight=None, n_imaginary=None,
                 n_rigid_modes_removed=None, lowest_frequency_cm_inv=None,
                 n_below_ithr=None, n_inversion_window=None,
                 hessian_asymmetry_eV_A2=None, frequencies_cm_inv=[])
        rows.append(r)
    return rows


def write_parquet_rows(rows, path):
    import pandas as pd
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


# ======================================================================================
# 2. the .log: the full human-readable record
# ======================================================================================
def write_log(rec, path):
    """The readable per-molecule report. **Complete** -- including diagnostics that do not
    go into the parquet."""
    qid = rec.get("qm9_index", "?")
    r = report.Report("{}   {}".format(qid, rec.get("smiles", "")),
                      subtitle="package 1 - CREST branch   per-molecule record")

    r.section("criteria and settings")
    r.kv("tighten_fmax_eV_A", rec.get("tighten_fmax_eV_A"))
    r.kv("dedup_threshold_A", rec.get("dedup_threshold_A"))
    r.kv("ithr_cm", rec.get("ithr_cm"),
         note="the frequency floor of the Hessian screen: a lowest mode in "
              "[ithr, 0) is admitted as an inversion window (the thermochemistry "
              "inverts it); only a lowest mode below ithr is a saddle, thrown out")
    r.kv("temperature_K", rec.get("temperature_K"))
    run = rec.get("crest_run") or {}
    st = run.get("settings") or {}
    r.kv("crest_version", (run.get("crest") or {}).get("version"))
    r.kv("crest_commit", (run.get("crest") or {}).get("commit"))
    for k in ("runtype", "optlev", "refine", "backend", "shake", "threads"):
        if k in st:
            r.kv("crest_" + k, st[k])
    if run.get("shake_fallback_used") is not None:
        r.warn("this molecule used the SHAKE fallback (shake = {}) and retried; "
               "before the fallback it terminated EARLY {} times. The root cause is in "
               "the shake_fallback comment in the configuration."
               .format(run.get("shake_fallback_used"),
                       run.get("n_terminated_early_before_fallback")))

    r.section("CREST sampling")
    r.kv("terminated_normally", run.get("terminated_normally"))
    r.kv("n_terminated_early", run.get("n_terminated_early"),
         note="criterion: must be 0")
    r.kv("total_engrad_calls", run.get("total_engrad_calls"))
    r.kv("crest_seconds", run.get("seconds"))
    r.kv("n_conformers_reported_by_crest", rec.get("n_conformers_reported_by_crest"),
         note="**this is not the number of basins** -- CREST's convergence and "
              "deduplication criteria both differ from this repository's (defect 54)")

    tc = rec.get("tighten_crest") or {}
    te = rec.get("tighten_etkdg") or {}
    r.section("tightened by this repository's criteria")
    for lab, t in (("CREST side", tc), ("ETKDG side", te)):
        if not t:
            continue
        r.text("  [{}]".format(lab))
        r.kv("n_frames", t.get("n_frames"))
        r.kv("opt_steps_total", t.get("opt_steps_total"))
        steps = t.get("opt_steps_per_frame") or []
        if steps:
            r.kv("opt_steps_max", int(max(steps)),
                 note="the step count says how far the supplied geometry was from the "
                      "minimum of this potential")
        r.kv("n_not_converged", t.get("n_not_converged"))
        r.kv("n_graph_changed", t.get("n_graph_changed"))
        r.kv("max_residual_force_eV_A", t.get("max_residual_force_eV_A"))

    pool = rec.get("pooling") or {}
    r.section("pooling and deduplication")
    r.kv("labels", "+".join(pool.get("labels") or []))
    r.kv("n_pooled_frames", pool.get("n_pooled_frames"))
    r.kv("n_global_before_hessian", pool.get("n_global_before_hessian"))
    r.kv("n_saddles_rejected", rec.get("n_saddles_rejected"),
         note="candidates whose lowest mode lies below the frequency floor ithr are "
              "thrown out and listed with their below-floor count and lowest "
              "frequency (the criterion is not relaxed)")
    for s in rec.get("saddles") or []:
        if s.get("n_below_ithr") is None:
            # a record written before the floor rule (the migration path re-renders
            # those): its rejection counter is the imaginary count, and its lowest
            # mode need not lie below ithr -- the 2026-09-25 rule is what made that
            # distinction
            r.warn("saddle rejected: index {} has {} imaginary mode(s), lowest "
                   "{:.2f} cm^-1".format(
                s.get("index"), s.get("n_imaginary"),
                s.get("lowest_frequency_cm_inv") or float("nan")))
        else:
            r.warn("saddle rejected: index {} has {} mode(s) below ithr, lowest "
                   "{:.2f} cm^-1".format(
                s.get("index"), s.get("n_below_ithr"),
                s.get("lowest_frequency_cm_inv") or float("nan")))
    for wmsg in pool.get("merge_energy_warnings") or []:
        r.warn("merge consistency warning: {}".format(wmsg))

    r.section("basin list")
    prov = rec.get("basin_provenance") or []
    hs = rec.get("basin_hessian") or []
    pop = rec.get("populations") or {}
    w = pop.get("boltzmann_weights") or []
    rows = []
    for i, (e, rel) in enumerate(zip(rec.get("basin_energies_eV", []),
                                     rec.get("basin_relative_kcal", []))):
        h = hs[i] if i < len(hs) else {}
        rows.append([i, "{:.8f}".format(e), "{:.4f}".format(rel),
                     "{:.4f}".format(w[i]) if i < len(w) else "-",
                     "+".join(prov[i]) if i < len(prov) else "-",
                     h.get("n_imaginary"),
                     h.get("n_inversion_window"),
                     "{:.2f}".format(h["lowest_frequency_cm_inv"])
                     if h.get("lowest_frequency_cm_inv") is not None else "-"])
    if rows:
        r.table(["basin", "energy", "relative", "weight", "origin", "imaginary",
                 "window", "lowest"], rows,
                units=["", "eV", "kcal/mol", "", "", "", "", "cm^-1"])
    r.kv("conformational_correction_kcal", rec.get("conformational_correction_kcal"))
    r.kv("weight_of_lowest", pop.get("weight_of_lowest"))
    if rec.get("free_energy_note"):
        r.note(rec["free_energy_note"])

    r.section("timings")
    r.kv("crest_seconds", run.get("seconds"))
    r.kv("analysis_seconds", rec.get("analysis_seconds"))
    r.kv("total_seconds", rec.get("total_seconds"))

    r.section("note on precision")
    r.note("the numbers in this file are laid out to 6 decimal places; **do not use it "
           "for a digit-by-digit comparison**. The copy carrying full float64 precision "
           "is the .parquet of the same name.")
    return r.write(path)


# ======================================================================================
# 3. everything written for one molecule
# ======================================================================================
def write_molecule(rec, dirs, qid, basins=None, crest_out=None,
                   keep_crest_out=True, gzip_crest_out=False):
    """Write out all the products of one molecule. Returns a dict of the paths written.

    `dirs` must contain the four directories `log` / `mol` / `xyz` / `out`.
    """
    out = {}
    out["log"] = write_log(rec, Path(dirs["log"]) / (qid + ".log"))
    out["parquet"] = write_parquet_rows(record_to_rows(rec),
                                        Path(dirs["mol"]) / (qid + ".parquet"))
    if basins is not None:
        out["xyz"] = _write_basins_xyz(
            Path(dirs["xyz"]) / (qid + "_basins.xyz"), basins,
            rec.get("basin_energies_eV") or [],
            rec.get("basin_provenance") or [], qid)
    # **CREST's native output, kept verbatim, only renamed** -- the naming the user
    # specified on 2026-08-31
    if keep_crest_out and crest_out and Path(crest_out).exists():
        dst = Path(dirs["out"]) / "crest_{}.out{}".format(
            qid, ".gz" if gzip_crest_out else "")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if gzip_crest_out:
            with open(crest_out, "rb") as fi, gzip.open(dst, "wb") as fo:
                shutil.copyfileobj(fi, fo)
        else:
            shutil.copy2(crest_out, dst)
        out["crest_out"] = dst
    return out


def _write_basins_xyz(path, basins, energies, prov, qid):
    lines = []
    for k, (a, e) in enumerate(zip(basins, energies)):
        lines.append(str(len(a)))
        lines.append("{} basin {} E = {:.8f} eV  from={}  (MACE-OFF23-SC)".format(
            qid, k, e, "+".join(prov[k]) if k < len(prov) else "?"))
        for s, p in zip(a.get_chemical_symbols(), a.positions):
            lines.append("{:2s} {:18.10f} {:18.10f} {:18.10f}".format(s, *p))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_failure(fr, dirs, qid):
    """A failure is filed too, and **still without JSON**."""
    r = report.Report("{}   {}   **FAILED**".format(qid, fr.get("smiles", "")),
                      subtitle="package 1 - CREST branch   failure record")
    r.section("exception")
    r.kv("error_type", fr.get("error_type"))
    r.kv("error", fr.get("error"))
    r.kv("seconds", fr.get("seconds"))
    r.section("full traceback")
    r.text(fr.get("traceback") or "(none)")
    p = Path(dirs["log"]) / (qid + ".FAILED.log")
    r.write(p)
    write_parquet_rows([dict(qm9_index=qid, smiles=fr.get("smiles"),
                             error_type=fr.get("error_type"),
                             error=str(fr.get("error"))[:2000],
                             seconds=fr.get("seconds"))],
                       Path(dirs["mol"]) / (qid + ".FAILED.parquet"))
    return p
