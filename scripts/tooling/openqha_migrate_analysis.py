"""Move superseded artifacts under analysis/_superseded/. Move-only, and reversible.

TOOLING. Produces no science.

What it moves, and what it deliberately does not
------------------------------------------------
It moves ONLY entries the INDEX marks `status="superseded"`. Everything else --
production, calibration, raw, unknown -- stays where it is.

That restraint is the point. Filing the live artifacts into production/ and
calibration/ subdirectories would break the scripts that write them, and those
scripts would simply re-scatter new files across the top level on their next run.
Tidying without the write-time rule recreates the mess; the write-time rule is
`openqha/artifacts.py::write_artifact` and it is not written yet.
So: move what is finished, leave what is live, and say so.

Reversibility
-------------
Every move is appended to `analysis/MIGRATION_MANIFEST.csv` (old path, new path,
timestamp). `--undo` replays that file backwards. Nothing is ever deleted; this
repository is not under git, so move-and-manifest is the only rollback there is.

Usage
-----
    python scripts/tooling/openqha_migrate_analysis.py --dry-run
    python scripts/tooling/openqha_migrate_analysis.py
    python scripts/tooling/openqha_migrate_analysis.py --undo
"""
import argparse
import csv
import shutil
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import openqha_index_artifacts as idx  # noqa: E402

ANALYSIS = S0_ROOT / "analysis"
MANIFEST = ANALYSIS / "MIGRATION_MANIFEST.csv"

#: Where each superseded route lands. Grouping by ROUTE, not by file type, so that
#: "everything from the near-critical gas box" is one directory rather than a filter.
ROUTE_OF = {
    "package1": "engine_MACE-OFF23-SC_basins",
    "package1_crest": "engine_MACE-OFF23-SC_basins",
    "package1_etkdg": "etkdg_census",
    "package1_summary": "etkdg_census",
    "package1_dedup": "etkdg_census",
    "package1_cost_model": "etkdg_census",
    "qm9_critical": "near_critical_gas_box",
    "species_critical": "near_critical_gas_box",
    "option_b": "option_b_dos_perturbation",
    "torsion_collective": "branch1_metadynamics",
    "s0-1_edge": "first_edge_assembly",
}


def route_for(name):
    for prefix, route in sorted(ROUTE_OF.items(), key=lambda kv: -len(kv[0])):
        if name.startswith(prefix):
            return route
    return "other"


def load_manifest():
    if not MANIFEST.exists():
        return []
    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def append_manifest(rows):
    exists = MANIFEST.exists()
    with open(MANIFEST, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["old_path", "new_path", "moved_at", "route"])
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def do_undo(dry):
    rows = load_manifest()
    if not rows:
        print("nothing to undo -- no manifest")
        return 0
    n = 0
    for r in reversed(rows):
        src, dst = S0_ROOT / r["new_path"], S0_ROOT / r["old_path"]
        if not src.exists():
            print("  MISSING (already moved back?) {}".format(r["new_path"]))
            continue
        print("  {} <- {}".format(r["old_path"], r["new_path"]))
        if not dry:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        n += 1
    if not dry and n:
        MANIFEST.unlink()
    print("\n{} entries {}".format(n, "would move back" if dry else "moved back"))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--undo", action="store_true")
    args = ap.parse_args()

    if args.undo:
        return do_undo(args.dry_run)

    rows = idx.scan(ANALYSIS)
    targets = [r for r in rows if r["status"] == "superseded"]
    if not targets:
        print("nothing marked superseded -- already migrated?")
        return 0

    dest_root = ANALYSIS / "_superseded"
    moved, total_files, total_bytes = [], 0, 0
    print("=" * 96)
    print("{} {} superseded entries -> analysis/_superseded/".format(
        "WOULD MOVE" if args.dry_run else "MOVING", len(targets)))
    print("=" * 96)
    for r in targets:
        src = S0_ROOT / r["path"]
        route = route_for(r["name"])
        dst = dest_root / route / r["name"]
        print("  {:52s} -> _superseded/{}/  ({} files, {:.1f} MB)".format(
            r["name"][:52], route, r["n_files"], r["bytes"] / 1e6))
        total_files += r["n_files"]
        total_bytes += r["bytes"]
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                print("     SKIPPED: destination already exists")
                continue
            shutil.move(str(src), str(dst))
            moved.append({
                "old_path": r["path"],
                "new_path": str(dst.relative_to(S0_ROOT)),
                "moved_at": datetime.now(timezone.utc).isoformat(),
                "route": route,
            })

    print()
    print("  {} entries, {} files, {:.1f} MB".format(len(targets), total_files, total_bytes / 1e6))
    if args.dry_run:
        print("\n  dry run -- nothing moved")
        return 0

    append_manifest(moved)
    print("\n  manifest: {}  ({} rows)".format(MANIFEST, len(moved)))
    print("  reverse with: python scripts/tooling/openqha_migrate_analysis.py --undo")

    # A README in the destination, so the directory explains itself without the plan.
    (dest_root / "README.md").write_text(
        "# `analysis/_superseded/`\n\n"
        "Artifacts of routes that have been ruled out. **Kept, never deleted** --\n"
        "A refuted result is written down so it is not\n"
        "re-cited as live. Nothing here may be quoted as a current number.\n\n"
        "| route | why it was ruled out |\n|---|---|\n"
        "| `engine_MACE-OFF23-SC_basins` | Basins are minima on MACE-OFF23-SC; the engine "
        "changed to MACE-OFF23_medium, so they are no longer stationary points. |\n"
        "| `etkdg_census` | ETKDG conformer generation retired. |\n"
        "| `near_critical_gas_box` | Multi-molecule near-critical route withdrawn in full: "
        "two amides required 642-674 K and mode-resolved temperatures were "
        "587-849 K against a 450 K target. |\n"
        "| `option_b_dos_perturbation` | VDOS route refuted by its own null control "
        "the noise floor 10.04 / 16.74 kcal/mol, larger than the signal. |\n"
        "| `branch1_metadynamics` | Branch 1 archived. |\n"
        "| `first_edge_assembly` | First assembled edge; its electronic term is "
        "SUPERSEDED. |\n\n"
        "Moves are recorded in `../MIGRATION_MANIFEST.csv` and reversible with\n"
        "`python scripts/tooling/openqha_migrate_analysis.py --undo`.\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
