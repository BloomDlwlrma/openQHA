"""Scan analysis/, assign every artifact a category and a status, write the INDEX.

TOOLING. Produces no science.

Why an index rather than a tidy-up
----------------------------------
analysis/ grew to 77 top-level entries / 12 388 files / 188 MB with no convention.
The repo's own contrast shows what fixes that and what does not: record.py has a
four-directory convention (out/log/mol/xyz) and has never drifted, while analysis/
had none. So the durable fix is `openqha/artifacts.py::write_artifact`, which makes
category and status REQUIRED at write time.

This script is the other half: it reads what is already on disk, states an identity
for each entry, and refuses to invent one where none can be established.

THE RULE ABOUT NOT GUESSING
---------------------------
Anything whose identity cannot be established from the repo record gets
status="unknown" -- never a plausible-looking guess. A source that cannot be traced is
written down as untraced rather than dressed up.

Statuses
--------
    production   feeds a deliverable
    calibration  produced a number used to make a decision
    diagnostics  written to locate one numbered defect
    raw          third-party output kept verbatim
    superseded   belongs to a route that has been ruled out; kept, never cited as live
    unknown      identity not established -- counted and listed, never silently filed

Usage
-----
    python scripts/tooling/openqha_index_artifacts.py                 # scan + write INDEX
    python scripts/tooling/openqha_index_artifacts.py --check         # verify, exit 1 on drift
"""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
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
from openqha import S0_ROOT  # noqa: E402

ANALYSIS = S0_ROOT / "analysis"
SHA_MAX_BYTES = 8 * 1024 * 1024      # only checksum small artifacts; big trees by size

#: Explicit identity table. Keys are top-level names under analysis/.
#:
#: Every entry states (category, status, decision, one_line). `one_line` is the thing
#: that was missing: what this artifact IS, in words, for someone who was not here.
#: Where the repo record does not settle it, status is "unknown" and one_line says so.
IDENTITY = {
    # ---- superseded routes -------------------------------------------------------
    "package1": ("production", "superseded", "S0-D-7",
                 "Branch A conformer census (ETKDG + CREST). Its basins are minima on "
                 "MACE-OFF23-SC; under the current engine they are no longer "
                 "stationary points. Kept for the measurements built on it; "
                 "regenerate under MACE-OFF23_medium."),
    "package1_etkdg_1_4000.log": ("production", "superseded", "S0-A-3", "ETKDG census log; ETKDG retired."),
    "package1_etkdg_1_4000__basins.parquet": ("production", "superseded", "S0-A-3", "ETKDG basins; ETKDG retired."),
    "package1_etkdg_1_4000__failures.parquet": ("production", "superseded", "S0-A-3", "ETKDG failures; ETKDG retired."),
    "package1_etkdg_1_4000__molecules.parquet": ("production", "superseded", "S0-A-3", "ETKDG molecules; ETKDG retired."),
    "package1_summary_1_4000.json": ("production", "superseded", "S0-A-3", "ETKDG census summary."),
    "package1_summary_1_4000.log": ("production", "superseded", "S0-A-3", "ETKDG census summary, human-readable."),
    "package1_dedup_threshold_1_4000.png": ("calibration", "superseded", "S0-A-3",
                                            "Dedup-threshold sweep from the ETKDG census."),
    "qm9_critical": ("production", "superseded", "D0-P3-19",
                     "Critical constants for all of QM9, for the near-critical gas-box "
                     "route. That route was withdrawn in full."),
    "qm9_critical_table.csv": ("production", "superseded", "D0-P3-19", "Same route, full table."),
    "qm9_critical_table_smoke.csv": ("calibration", "superseded", "D0-P3-19", "Same route, smoke subset."),
    "species_critical_constants.json": ("production", "superseded", "D0-P3-19",
                                        "Critical constants for the 7 species; same withdrawn route."),
    "torsion_collective_variables.json": ("production", "superseded", "D0-C-38",
                                          "Torsional collective variables for branch-1 metadynamics; archived."),
    "option_b_force_perturbation_NULL.json": ("calibration", "superseded", "D0-P2-13",
                                              "Null control that killed the VDOS route: potential fixed, seed varied, "
                                              "noise floor 10.04 / 16.74 kcal/mol."),
    "option_b_force_perturbation_NULL.log": ("calibration", "superseded", "D0-P2-13", "Same, human-readable."),
    "option_b_force_perturbation_B3.json": ("calibration", "superseded", "D0-P2-13",
                                            "Perturbed arm; signal below the null noise floor."),
    "option_b_force_perturbation_B3.log": ("calibration", "superseded", "D0-P2-13", "Same, human-readable."),
    "option_b_force_perturbation_HARM.json": ("calibration", "superseded", "D0-P2-14",
                                              "Harmonic control that supplied the 0.50 / 0.66 kcal/mol error bar."),
    "option_b_force_perturbation_HARM.log": ("calibration", "superseded", "D0-P2-14", "Same, human-readable."),
    "option_b_force_perturbation_B0-INVALID-temperature.json": (
        "calibration", "superseded", "D0-P2-8",
        "First option-B run, VOIDED: trajectories ran at 177-1209 K, not 298. Kept under "
        "its INVALID name deliberately so the numbers are never re-cited."),
    "option_b_force_perturbation_B0-INVALID-temperature.log": (
        "calibration", "superseded", "D0-P2-8", "Same, human-readable."),

    # ---- live calibration --------------------------------------------------------
    "committee_calibration.json": ("calibration", "calibration", "S0-C-17",
                                   "Committee gate: 46 structures, 5 published MACE-OFF models. "
                                   "Failed on all three channels (rho -0.057 / +0.061 / +0.397)."),
    "symmetry_backtest.json": ("calibration", "calibration", "S0-A-13",
                               "pymsym against the 7 hand-declared symmetry numbers: 7/7, but sigma=2 "
                               "species flip to 1 at 0.005 A displacement."),
    "branchA_workhorse": ("calibration", "calibration", "S0-A-7",
                          "Four-arm attribution of the gfn2 terminated-EARLY failure "
                          "(shake x timestep x workhorse)."),
    "engine_benchmark.json": ("calibration", "calibration", "D0-22", "Per-force-call cost across engines."),
    "engine_benchmark.log": ("calibration", "calibration", "D0-22", "Same, human-readable."),
    "engine_provenance.json": ("calibration", "calibration", "D0-19", "Engine weights provenance record."),
    "engine_provenance.log": ("calibration", "calibration", "D0-19", "Same, human-readable."),
    "branch2_cost_model.json": ("calibration", "calibration", "S0-C-5",
                                "RI-MP2 OPT NumFreq cost model over the 436 basins."),
    "package1_cost_model_full_qm9.json": ("calibration", "superseded", "D0-C-23",
                                          "Full-QM9 cost extrapolation, measured with the GFN-FF workhorse; "
                                          "must be re-measured under gfn2."),
    "package1_cost_model_full_qm9.log": ("calibration", "superseded", "D0-C-23", "Same, human-readable."),
    "package1_flexibility_predictor.json": ("calibration", "calibration", "D0-P1-31",
                                            "Rotatable-bond / basin-count rank sum as a cost predictor."),
    "package1_flexibility_predictor.log": ("calibration", "calibration", "D0-P1-31", "Same, human-readable."),
    "rimp2_gradient_cost.json": ("calibration", "calibration", "D0-P3-23",
                                 "RI-MP2 gradient cost ladder; 44.26 s/structure at cc-pVTZ+RIJK."),
    "lowfreq_and_separable_terms.json": ("calibration", "calibration", "D0-16",
                                         "Exact cancellation of translational and electronic terms; "
                                         "hindered-rotor comparison."),
    "lowfreq_and_separable_terms.log": ("calibration", "calibration", "D0-16", "Same, human-readable."),
    "lowfreq_hindered_rotor.png": ("calibration", "calibration", "D0-17", "Hindered-rotor error figure."),
    "composite_routes_comparison.json": ("calibration", "calibration", "D0-73",
                                         "Composite QZ*/TZ* route cost comparison."),
    "composite_routes_comparison.log": ("calibration", "calibration", "D0-73", "Same, human-readable."),
    "composite_energy_force_E2.json": ("calibration", "calibration", "D0-P2-11",
                                       "E2: MACE vs composite QZ*, 44 structures, 8.24 h. "
                                       "Force RMSD 0.1435 eV/A."),
    "composite_energy_force_E2.log": ("calibration", "calibration", "D0-P2-11", "Same, human-readable."),
    "composite_energy_force_E2.partial.jsonl": ("calibration", "calibration", "D0-P2-11", "E2 incremental records."),
    "frequency_benchmark.json": ("calibration", "calibration", "D0-35",
                                 "Package-2 frequency benchmark vs QM9 native B3LYP; 3428 numeric fields, "
                                 "used as the refactor reproduction criterion."),
    "frequency_benchmark.log": ("calibration", "calibration", "D0-35", "Same, human-readable."),
    "hessian_rrho_free_energy.json": ("calibration", "calibration", "D0-28", "RRHO free energies from the Hessian bypass."),
    "hessian_rrho_free_energy.log": ("calibration", "calibration", "D0-28", "Same, human-readable."),
    "qm9_native_reference.json": ("calibration", "calibration", "D0-32", "QM9 native geometries as an independent sample."),
    "qm9_native_reference.log": ("calibration", "calibration", "D0-32", "Same, human-readable."),
    "package2": ("calibration", "calibration", "D0-28", "Package-2 Hessian bypass working directory."),
    "s0-1_edge_result.json": ("production", "superseded", "D0-25",
                              "First assembled edge, +17.5088 kcal/mol. Not a result: 6 ps trajectory, "
                              "no error bar, and its electronic term is marked SUPERSEDED."),
    "s0-1_edge_result.log": ("production", "superseded", "D0-25", "Same, human-readable."),
    "package1_crest_1_4000.log": ("production", "superseded", "S0-A-16",
                                  "CREST census log; basins live on MACE-OFF23-SC."),
    "package1_crest_summary_1_4000.json": ("production", "superseded", "S0-A-16", "CREST census summary."),
    "package1_crest_summary_1_4000.log": ("production", "superseded", "S0-A-16", "Same, human-readable."),
    "package1_crest_summary_1_4000.png": ("production", "superseded", "S0-A-16", "CREST census figure."),
    "package1_crest_1_4000__basins.parquet": ("production", "superseded", "S0-A-16", "CREST basins table."),
    "package1_crest_1_4000__molecules.parquet": ("production", "superseded", "S0-A-16", "CREST molecules table."),
    "package1_crest_1_4000__failures.parquet": ("production", "superseded", "S0-A-16", "CREST failures table."),
    "package1_crest_1_4000__frequencies.parquet": ("production", "superseded", "S0-A-16", "CREST frequencies table."),

    # ---- raw third-party output --------------------------------------------------
    "logs": ("raw", "raw", "skills-1.3",
             "Driver logs and full-text extractions of reference papers, kept verbatim."),

    # ---- branch 2 production (RI-MP2 labels; still live as branch-C input) --------
    "branch2_optfreq_shard0.jsonl": ("production", "production", "S0-C-5",
                                     "RI-MP2 OPT NumFreq records, shard 0. provenance.status=test "
                                     "(local ORCA 6.0.1; production numbers come only from deimos 6.1.1)."),
    "branch2_optfreq_shard1.jsonl": ("production", "production", "S0-C-5", "Same, shard 1."),
    "branch2_optfreq_shard2.jsonl": ("production", "production", "S0-C-5", "Same, shard 2."),
    "symmetry_number.json": ("calibration", "calibration", "S0-A-13",
                             "Validation of openqha/symmetry.py: the 7 declared species reproduced, "
                             "PLUS must-fail cases (ethane graph automorphism 72 against correct "
                             "sigma 6; methane 24 against 12). Verdict PASS, 0 failures, 0 cases "
                             "printing the forbidden answer. Its detector holds sigma=2 out to "
                             "0.05 A where raw pymsym flips at 0.005 A."),
    "isomerisation_electronic_energy.json": ("calibration", "calibration", "D0-P2-12",
                                             "MACE isomerisation dE_el vs composite QZ*: MAD 1.87 kcal/mol."),
}

#: Entries whose identity the repo record does not settle:
#: these get status="unknown" rather than a guess. Each line says what IS known.
UNKNOWN = {
    "composite_energy_force_SMOKE2.json":
        "Smoke run of the composite energy/force driver. What distinguishes SMOKE2 from "
        "SMOKE3 is not recorded anywhere in the repository record; both predate the E2 production run.",
    "composite_energy_force_SMOKE2.log": "See composite_energy_force_SMOKE2.json.",
    "composite_energy_force_SMOKE2.partial.jsonl": "See composite_energy_force_SMOKE2.json.",
    "composite_energy_force_SMOKE3.json":
        "Second smoke run of the composite driver; its difference from SMOKE2 is unrecorded.",
    "composite_energy_force_SMOKE3.log": "See composite_energy_force_SMOKE3.json.",
    "composite_energy_force_SMOKE3.partial.jsonl": "See composite_energy_force_SMOKE3.json.",
    "composite_energy_force_smoke_mp2_dz.json":
        "MP2/cc-pVDZ smoke run (12 structures, 59 s). "
        "Whether this file is that exact run is not established.",
    "composite_energy_force_smoke_mp2_dz.log": "See composite_energy_force_smoke_mp2_dz.json.",
    "branch2_probe_10.jsonl":
        "Branch-2 cost probe, 10-atom molecule. Individual files are not tied to "
        "specific runs.",
    "branch2_probe_15.jsonl": "Branch-2 cost probe, 15-atom molecule. Same caveat.",
    "branch2_probe_15_cosx.jsonl":
        "Branch-2 cost probe, 15 atoms, apparently the RIJCOSX arm. The suffix is not "
        "defined anywhere.",
    "branch2_probe_15_rijk.jsonl":
        "Branch-2 cost probe, 15 atoms, apparently the RIJK arm. Same caveat.",
    "_def46_tmp.md":
        "Scratch note, apparently about s0_mace_engrad.py's shebang. "
        "Neither its author nor its status is recorded.",
    "crest_diag2":
        "EMPTY DIRECTORY. A CREST diagnostic working directory whose contents are gone; "
        "nothing references it.",
}


def sha256_of(path):
    if path.is_dir() or path.stat().st_size > SHA_MAX_BYTES:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_stats(path):
    if path.is_file():
        return 1, path.stat().st_size
    n = size = 0
    for p in path.rglob("*"):
        if p.is_file():
            n += 1
            size += p.stat().st_size
    return n, size


def scan(root):
    rows = []
    for entry in sorted(root.iterdir()):
        name = entry.name
        if name.startswith("INDEX") or name == "MIGRATION_MANIFEST.csv":
            continue
        n_files, size = tree_stats(entry)
        if name in IDENTITY:
            category, status, decision, one_line = IDENTITY[name]
        elif name in UNKNOWN:
            category, status, decision, one_line = "unknown", "unknown", None, UNKNOWN[name]
        else:
            category, status, decision = "unknown", "unknown", None
            one_line = ("Not listed in the identity table. Identity not established -- "
                        "do not cite until it is.")
        rows.append({
            "path": str(entry.relative_to(S0_ROOT)),
            "name": name,
            "is_dir": entry.is_dir(),
            "category": category,
            "status": status,
            "decision": decision or "",
            "one_line": one_line,
            "n_files": n_files,
            "bytes": size,
            "sha256": sha256_of(entry) or "",
            "mtime": datetime.fromtimestamp(entry.stat().st_mtime, timezone.utc).isoformat(),
        })
    return rows


def write_index(rows, root):
    by_status = {}
    for r in rows:
        s = by_status.setdefault(r["status"], {"n": 0, "files": 0, "bytes": 0})
        s["n"] += 1
        s["files"] += r["n_files"]
        s["bytes"] += r["bytes"]

    (root / "INDEX.csv").write_text(
        "".join([",".join(rows[0].keys()) + "\n"] +
                [",".join('"{}"'.format(str(v).replace('"', "'")) for v in r.values()) + "\n"
                 for r in rows]),
        encoding="utf-8")

    lines = [
        "# `analysis/` INDEX",
        "",
        "> **Generated by `scripts/tooling/openqha_index_artifacts.py` -- do not hand-edit.**",
        "> Every artifact carries a category, a status, and one line saying what it is.",
        "> `status=\"unknown\"` means the identity could not be established from the repo",
        "> record. It is never a guess.",
        "",
        "## Summary by status",
        "",
        "| status | entries | files | size |",
        "|---|---:|---:|---:|",
    ]
    for s in ("production", "calibration", "diagnostics", "raw", "superseded", "unknown"):
        if s in by_status:
            v = by_status[s]
            lines.append("| **{}** | {} | {} | {:.1f} MB |".format(
                s, v["n"], v["files"], v["bytes"] / 1e6))
    lines += ["", "## Entries", "",
              "| path | category | status | decision | files | size | what it is |",
              "|---|---|---|---|---:|---:|---|"]
    for r in sorted(rows, key=lambda r: (r["status"], r["name"])):
        lines.append("| `{}` | {} | **{}** | {} | {} | {:.2f} MB | {} |".format(
            r["name"], r["category"], r["status"], r["decision"] or "--",
            r["n_files"], r["bytes"] / 1e6, r["one_line"]))
    (root / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return by_status


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ANALYSIS))
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any entry has status=unknown that is not in the UNKNOWN table")
    args = ap.parse_args()

    root = Path(args.root)
    rows = scan(root)
    by_status = write_index(rows, root)

    print("=" * 96)
    print("analysis/ INDEX")
    print("=" * 96)
    for s in ("production", "calibration", "diagnostics", "raw", "superseded", "unknown"):
        if s in by_status:
            v = by_status[s]
            print("  {:12s} {:3d} entries  {:6d} files  {:8.1f} MB".format(
                s, v["n"], v["files"], v["bytes"] / 1e6))
    print()
    unlisted = [r["name"] for r in rows
                if r["status"] == "unknown" and r["name"] not in UNKNOWN]
    if unlisted:
        print("  NOT IN ANY IDENTITY TABLE ({}):".format(len(unlisted)))
        for n in unlisted:
            print("     ", n)
    print()
    print("written {} and {}".format(root / "INDEX.md", root / "INDEX.csv"))
    if args.check and unlisted:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
