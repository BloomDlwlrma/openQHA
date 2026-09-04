"""Measure the CREST workhorse + MACE-refinement cost on ONE edge.

CALIBRATION. Branch A, step 1: the cost of one edge under a given workhorse. It
produces a number used to choose settings, and nothing that enters a deliverable.

Branch A, step 1. User ruling 2026-09-03: "start measure GFN2-XTB and mace opt in
one edge".

What this measures, and what it deliberately does NOT
-----------------------------------------------------
It runs CREST `imtd-gc` on both species of a single edge, under a chosen workhorse
(`--workhorse`) and refinement level (`--refine`), and records:

  * wall-clock seconds per species,
  * number of energy+gradient calls, split into total and MACE-only,
  * CREST's own conformer count,
  * `terminated EARLY` / `completed successfully` counters,

and nothing else. It does not run the downstream tighten/dedup/Hessian stage --
that is `openqha/crest_census.py` and it is a separate, separately-measured cost
(D0-P1-24: after the switch to refine=sp the analysis stage became the bottleneck,
78% of one molecule's total).

Cost numbers are bound to their measurement conditions, per D0-P1-12 and defect 34:
the record carries the machine load, the thread count and whether any scratch
directory was reused. Do NOT divide a serial number by a process count.

Default edge is `C2H5O1N1_19_36` (acetamide <-> N-methylformamide): the only edge
carrying all four elements, and stage 2's experiment of record.

Usage
-----
    python -m openqha.mace_server --socket /tmp/s0_mace_engrad.sock &
    S0_MACE_SOCKET=/tmp/s0_mace_engrad.sock \
        python scripts/calibration/s0_A_workhorse_edge_measure.py --workhorse gfn2 --refine opt

    # the comparison arm (same edge, previous production settings):
    ... --workhorse gfnff --refine sp --tag gfnff_sp
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
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
from openqha import S0_ROOT, config, crest  # noqa: E402

HARTREE_KCAL = 627.5094740631

#: Species making up each edge, read from the config's `species` block.
#: An edge id `C2H5O1N1_19_36` names QM9 indices 19 and 36.
def edge_species(edge_id):
    """Return the two `dsgdb9nsd_XXXXXX` ids named by an edge id."""
    tail = edge_id.rsplit("_", 2)
    if len(tail) != 3:
        raise ValueError("edge id not in the expected FORMULA_a_b shape: {!r}".format(edge_id))
    return tuple("dsgdb9nsd_{:06d}".format(int(x)) for x in tail[1:])


def load_start_geometry(qm9_index, dest):
    """Copy the reference geometry for one species into `dest` as `<index>.xyz`.

    The repo ships the 7 target species' reference geometries (D0-41), so this
    needs no external data.
    """
    src = S0_ROOT / "data" / "reference-geometries" / "{}.xyz".format(qm9_index)
    if not src.exists():
        raise FileNotFoundError(
            "reference geometry not found: {}\n"
            "The repo ships only the 7 stage-0 species (D0-41); for anything else "
            "run scripts/tooling/s0_prepare_data.py first.".format(src))
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def machine_load():
    """1-minute load average and core count -- the measurement conditions.

    A cost number without these is not reportable (D0-P3-6: two earlier benchmarks
    differing by 4.5x were both contaminated by an unrecorded competing job).
    """
    try:
        one, five, fifteen = os.getloadavg()
    except (OSError, AttributeError):
        one = five = fifteen = None
    return {
        "loadavg_1min": one,
        "loadavg_5min": five,
        "loadavg_15min": fifteen,
        "cpu_count": os.cpu_count(),
        "node": platform.node(),
    }


def relative_energies_kcal(ensemble_path):
    """Relative energies of an ensemble, kcal/mol, lowest first."""
    if not Path(ensemble_path).exists():
        return []
    energies = []
    for comment, _ in crest.read_ensemble(ensemble_path):
        try:
            energies.append(float(comment.split()[0]))
        except (ValueError, IndexError):
            pass
    if not energies:
        return []
    lowest = min(energies)
    return sorted((e - lowest) * HARTREE_KCAL for e in energies)


def run_one(qm9_index, outdir, cfg, args):
    """Run CREST once for one species and return the full record."""
    workdir = Path(outdir) / qm9_index
    xyz = load_start_geometry(qm9_index, workdir / "{}.xyz".format(qm9_index))

    before = machine_load()
    started = time.time()
    record = crest.run(
        workdir, xyz,
        runtype=cfg["runtype"],
        threads=int(args.threads),
        optlev=cfg["optlev"],
        refine=args.refine,
        backend=cfg["backend"],
        engine_client=S0_ROOT / cfg["engine_client"],
        workhorse=args.workhorse,
        shake=args.shake if args.shake is not None else cfg.get("shake"),
        tstep_fs=args.tstep_fs,
        timeout_s=int(args.timeout_s),
    )
    record["qm9_index"] = qm9_index
    record["wall_seconds"] = time.time() - started
    record["load_before"] = before
    record["load_after"] = machine_load()
    record["settings"] = {
        "workhorse": args.workhorse,
        "refine": args.refine,
        "runtype": cfg["runtype"],
        "optlev": cfg["optlev"],
        "threads": int(args.threads),
        "shake": args.shake if args.shake is not None else cfg.get("shake"),
        "tstep_fs": args.tstep_fs,   # None = CREST's own default, which it prints as 5.0
        "backend": cfg["backend"],
    }
    record["relative_energies_kcal"] = relative_energies_kcal(
        workdir / "crest_conformers.xyz")
    record["n_conformers_crest_reports"] = len(record["relative_energies_kcal"])
    # D0-P1-1 / D0-95: CREST's conformer count is NOT the basin count. What
    # survives cregen includes unconverged duplicates. The basin count only exists
    # after openqha/crest_census.py tightens to fmax=1e-4 and dedups. Naming the field
    # this way is the whole point.
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edge", default="C2H5O1N1_19_36",
                    help="edge id from configs/openqha.yaml (default: the four-element edge)")
    ap.add_argument("--workhorse", default="gfn2", choices=list(crest.WORKHORSES),
                    help="CREST sampling-level method (default: gfn2, the published iMTD-GC setting)")
    ap.add_argument("--refine", default="opt",
                    help="quality-layer refinement: sp | add | opt | none (default: opt)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tstep-fs", type=float, default=None,
                    help="metadynamics timestep in fs; omit to leave CREST's own default (5.0 fs)")
    ap.add_argument("--shake", type=int, default=None, choices=(0, 1, 2),
                    help="SHAKE level override; omit to take the value from the config (1)")
    ap.add_argument("--only", default=None,
                    help="run just this qm9 index instead of both species of the edge")
    ap.add_argument("--timeout-s", type=int, default=14400,
                    help="per-species wall clock ceiling; gfn2 cost is unmeasured, so this is generous")
    ap.add_argument("--tag", default=None,
                    help="output tag; defaults to <workhorse>_<refine>")
    args = ap.parse_args()

    cfg = config.load()["crest"]
    tag = args.tag or "{}_{}".format(args.workhorse, args.refine)
    outdir = S0_ROOT / "analysis" / "branchA_workhorse" / args.edge / tag
    outdir.mkdir(parents=True, exist_ok=True)

    species = edge_species(args.edge)
    if args.only:
        species = tuple(s for s in species if s == args.only) or (args.only,)
    version = crest.crest_version()

    print("=" * 92)
    print("Branch A step 1 -- workhorse cost measurement on one edge")
    print("=" * 92)
    print("edge        {}  ->  {}".format(args.edge, " , ".join(species)))
    print("CREST       {} (commit {})".format(version["version"], version["commit"]))
    print("workhorse   {}      refine   {}".format(args.workhorse, args.refine))
    print("runtype     {}   optlev  {}   threads {}   shake {}".format(
        cfg["runtype"], cfg["optlev"], args.threads, cfg.get("shake")))
    print("socket      {}".format(os.environ.get("S0_MACE_SOCKET", cfg["socket"])))
    print("output      {}".format(outdir))
    print()

    records = []
    for qm9_index in species:
        print("-" * 92)
        print("running {} ...".format(qm9_index))
        record = run_one(qm9_index, outdir, cfg, args)
        records.append(record)
        print("  wall            {:.1f} s = {:.2f} h".format(
            record["wall_seconds"], record["wall_seconds"] / 3600.0))
        print("  terminated normally      {}".format(record["terminated_normally"]))
        print("  terminated EARLY         {}   <- criterion: must be 0".format(
            record["n_terminated_early"]))
        print("  completed successfully   {}".format(record["n_completed_successfully"]))
        print("  energy+gradient calls    {}".format(record["total_engrad_calls"]))
        print("  CREST reports            {} conformers  (NOT basins)".format(
            record["n_conformers_crest_reports"]))
        rel = record["relative_energies_kcal"]
        if rel:
            print("  relative energies kcal/mol  {}".format(
                " ".join("{:.4f}".format(x) for x in rel[:12])))
        print("  criteria ok     {}".format(record["ok"]))

    summary = {
        "generated_by": "scripts/calibration/s0_A_workhorse_edge_measure.py",
        "edge": args.edge,
        "species": list(species),
        "tag": tag,
        "crest_version": version,
        "settings": records[0]["settings"] if records else None,
        "records": records,
        "totals": {
            "wall_seconds": sum(r["wall_seconds"] for r in records),
            "engrad_calls": sum(r["total_engrad_calls"] or 0 for r in records),
            "all_ok": all(r["ok"] for r in records),
        },
        "identity": (
            "Cost measurement only. The downstream tighten/dedup/Hessian stage "
            "(openqha/crest_census.py) is NOT included and is a separate cost. "
            "CREST's conformer count is not the basin count (D0-P1-1, D0-95)."
        ),
    }
    out = outdir / "measurement.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("=" * 92)
    print("total wall  {:.1f} s = {:.2f} h   all criteria ok: {}".format(
        summary["totals"]["wall_seconds"],
        summary["totals"]["wall_seconds"] / 3600.0,
        summary["totals"]["all_ok"]))
    print("written     {}".format(out))


if __name__ == "__main__":
    main()
