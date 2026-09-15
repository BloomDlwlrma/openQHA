"""The molecule's answer: F_conf over every basin, from branch A + branch B products.

PRODUCTION. It assembles, it does not compute: every number it uses was produced by
branch A (`s0_A_pipeline.py`) or branch B (`s0_B_qha_trajectory*.py`), and the assembly
itself lives in `openqha.quasi_harmonic.ensemble` so that this file and the parameter
scans cannot drift apart.

WHAT IT PUTS TOGETHER
---------------------
    dG_i = dE_el,i  -  T*S_i           per basin, relative to the lowest
    F_conf = -kT ln sum_i exp(-dG_i / kT)

    dE_el,i   branch A, from the basin store
    T*S_i     branch B, from collect's per-trajectory table (mean over seeds)

WHERE T*S IS READ FROM, AND WHY NOT THE FRAMES
----------------------------------------------
`s0_B_qha_analyse.py` (the collect step) writes `<molecule>/_records/md_openmm/<setting>/
collect__trajectories.parquet` (ADR 0001; before 2026-09-14 `analysis/qha/<tag>/<species>__
trajectories.parquet`), one row per (basin, seed) with the entropy it judged against the criteria.
This step reads THAT. Until 2026-09-13 it re-ran `qha.analyse` on the raw frames -- a
second analysis that could differ from the judged one (temperature, settings) and that
skipped, without a word, any trajectory shorter than 3N frames. The first chain to reach
this step (an113, six 5-frame test trajectories) got "entropy? MISSING" for every basin
ten seconds after collect had printed their entropies. A basin with no row in that table
is reported as missing; nothing is recomputed to fill it.

The `--atoms heavy` set is the one exception: collect produces only the all-atom analysis,
so the heavy-atom entropy is computed here, from the frames, and says so.

WHY THIS IS A SEPARATE STEP AND NOT PART OF THE ANALYSIS
--------------------------------------------------------
`s0_B_qha_analyse.py` works on ONE species' trajectories and answers "is this basin's
entropy converged". It is per-trajectory and per-basin by design. The ensemble is the
level above: it needs branch A's relative energies, which that script never reads, and it
is where the two branches finally meet.

Keeping it separate also keeps the failure separable. "Basin 3's trajectory did not
converge" and "the ensemble is dominated by one basin" are different problems with
different fixes, and a single script reporting one number would hide whichever it was not
looking for.

    python scripts/production/s0_B_report_ensemble.py --species dsgdb9nsd_000035 --tag prod
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))

from openqha import config                                          # noqa: E402
from openqha.quasi_harmonic import basin_residence as br            # noqa: E402
from openqha.quasi_harmonic import ensemble, qha                    # noqa: E402
from openqha.store import basins as basin_reader                    # noqa: E402


def basin_electronic(rec):
    """Relative electronic energy per basin, kcal/mol, from branch A's record.

    The key is `relative_kcal`. It is looked up by name and a basin without it RAISES --
    it does not fall back to 0.0. An earlier version guessed three other names and, when
    all three missed, gave every basin zero: propanal's three basins (0.0000 / 0.8360 /
    0.8360) came out degenerate and F_conf read -0.6509 instead of -0.2354. There was no
    error, because a default of 0.0 is a perfectly valid relative energy.
    """
    out = []
    for i, b in enumerate(rec.get("basins", [])):
        if "relative_kcal" not in b:
            raise KeyError(
                "basin {} of this branch A record has no `relative_kcal`. Its keys are "
                "{}. Refusing to assume zero: a missing relative energy would make the "
                "basins look degenerate and F_conf would be wrong without saying so."
                .format(i, sorted(b)))
        out.append(float(b["relative_kcal"]))
    return out


def collect_stem(species, tag, setting="default", root=None, route="auto"):
    """Where s0_B_qha_analyse.py put this molecule's tables (its own default --out):
    <molecule>/_records/<route>/<setting>/collect (ADR 0001). `tag` is the basin tag;
    route `auto` is the one the trajectories are found under (openmm before ase)."""
    from openqha.store import layout
    from openqha.quasi_harmonic import trajectory_reader
    mol = basin_reader.molecule_for(species, tag, root=root)
    r = route if route in ("openmm", "ase") else (trajectory_reader.route_found(mol, setting) or "openmm")
    return layout.records_for(mol, r, setting) / "collect"


def collect_tables(species, tag, setting="default", root=None, route="auto"):
    """(trajectories, criteria) as lists of row dicts, from collect's parquet tables.

    Refuses when the trajectories table is absent: this step sums what collect judged
    and has nothing to sum before collect has run. The criteria table is optional in
    the reading (older products have none) and its absence is recorded, not ignored.
    """
    import pandas as pd
    stem = collect_stem(species, tag, setting, root, route)
    traj = stem.parent / (stem.name + "__trajectories.parquet")
    if not traj.is_file():
        raise SystemExit(
            "s0_B_report_ensemble: no collect product for {} under tag {!r}:\n    {}\n"
            "  This step sums the entropies collect judged; it does not analyse frames. "
            "Run collect first:\n"
            "    python scripts/production/s0_B_qha_analyse.py --species {} --tag {}"
            .format(species, tag, traj, species, tag))
    rows = pd.read_parquet(traj).to_dict("records")
    crit_path = stem.parent / (stem.name + "__criteria.parquet")
    criteria = (pd.read_parquet(crit_path).to_dict("records")
                if crit_path.is_file() else None)
    return rows, criteria


def crossings_per_basin(species, tag, cfg, setting="default", root=None, route="auto"):
    """Basin residence from the frames -- the one thing collect's table does not carry.

    The frames are read from the engine folder (traj.dcd, ADR 0001); returns
    (per-basin dict, the openmm/<setting> folder it read)."""
    from openqha.quasi_harmonic import trajectory_reader
    molecule = basin_reader.molecule_for(species, tag, cfg, root=root)
    dirs = trajectory_reader.trajectory_dirs(molecule, setting, route)
    from openqha.store import layout as _layout
    root = molecule / _layout.md_folder(trajectory_reader.route_of(dirs[0][1]) if dirs else "openmm")
    out = {}
    for b, eng, rec in dirs:
        res_all = []
        tr = trajectory_reader.read_trajectory(eng, records_dir=rec, setting=setting)
        res_all.append(br.basin_residence(tr["positions_A"], tr["symbols"]))
        out[b] = dict(
            distinct_crossings=int(sum(r["distinct_basin_crossings"] for r in res_all)),
            symmetry_crossings=int(sum(r["symmetry_equivalent_crossings"] for r in res_all)),
            n_seeds_with_frames=len(res_all))
    return out, root


def entropy_per_basin(species, tag, cfg, atoms_set="all", setting="default", root=None,
                      route="auto"):
    """Mean T*S over the seeds of each basin, plus what the trajectories did.

    `all`: from collect's trajectories table -- the judged numbers, nothing recomputed.
    `heavy`: computed here from the frames (collect has no heavy-atom product); a
    trajectory below 3N frames is skipped AND SAID, because a silent skip is how every
    basin of the first run to reach this step came out "MISSING".

    A basin with no trajectory returns None rather than 0. `ensemble` refuses to treat
    the two the same, because "we did not run it" and "its entropy is zero" are different
    statements and only one of them is ever true.
    """
    xing, traj_root = crossings_per_basin(species, tag, cfg, setting, root, route)
    per_basin = {}
    if atoms_set == "all":
        rows, _ = collect_tables(species, tag, setting, root, route)
        by_basin = {}
        for r in rows:
            by_basin.setdefault(basin_index(r["basin"]), []).append(r)
        for b in sorted(set(by_basin) | set(xing)):
            vals = [float(r["TS_QH_kcal"]) for r in by_basin.get(b, [])]
            if not vals:
                per_basin[b] = None
                continue
            per_basin[b] = dict(
                TS_kcal=float(np.mean(vals)), spread_kcal=float(np.std(vals)),
                n_seeds=len(vals),
                n_frames=int(max(r["n_frames"] for r in by_basin[b])),
                source="collect table",
                **{k: v for k, v in (xing.get(b) or {}).items()
                   if k in ("distinct_crossings", "symmetry_crossings")})
        return per_basin, traj_root

    from openqha.quasi_harmonic import trajectory_reader
    molecule = basin_reader.molecule_for(species, tag, cfg, root=root)
    for b, eng, rec in trajectory_reader.trajectory_dirs(molecule, setting, route):
        vals, frames_seen = [], 0
        tr = trajectory_reader.read_trajectory(eng, records_dir=rec, setting=setting)
        frames = tr["positions_A"]
        syms = tr["symbols"]
        masses = np.asarray((tr["meta"] or {}).get("masses_amu") or tr["masses_amu"],
                            dtype=float)
        if len(frames) < 3 * len(syms):
            print("  heavy: {} has {} frames for {} coordinates; the covariance "
                  "cannot be formed, trajectory skipped".format(
                      eng.relative_to(traj_root), len(frames), 3 * len(syms)))
        else:
            mask = br.heavy_atom_mask(syms)
            ana = qha.analyse(frames[:, mask, :], masses[mask])
            vals.append(ana["entropy"]["TS_QH_kcal"])
            frames_seen = len(frames)
        per_basin[b] = (dict(
            TS_kcal=float(np.mean(vals)), spread_kcal=float(np.std(vals)),
            n_seeds=len(vals), n_frames=frames_seen, source="recomputed here (heavy)",
            **{k: v for k, v in (xing.get(b) or {}).items()
               if k in ("distinct_crossings", "symmetry_crossings")})
            if vals else None)
    return per_basin, traj_root


def basin_index(label):
    """0 from 'basin00' -- collect's tables carry the trajectory DIRECTORY names
    (`basin00`, `seed00`, from s0_B_qha_analyse.discover), not integers. The first
    read of that table (an113, 2026-09-13) did int('basin00'). An int is accepted too."""
    text = str(label).strip()
    if text.startswith("basin"):
        text = text[len("basin"):]
    return int(text)


def criteria_verdict(criteria):
    """(passed, total) from collect's criteria table; (None, None) if it wrote none."""
    if not criteria:
        return None, None
    return sum(1 for c in criteria if bool(c.get("passed"))), len(criteria)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", required=True)
    ap.add_argument("--tag", default="prod")
    # A short test run must not write into the production tag -- the trajectory driver
    # RESUMES from whatever frames it finds, so 5 ps of test frames would become the
    # first 5 ps of production. So a test writes under its own --tag and reads branch A
    # from --basin-tag, exactly as the 02d identity chain already does.
    ap.add_argument("--basin-tag", default=None,
                    help="tag whose branch A basins to sum over (default: --tag)")
    ap.add_argument("--setting", default="default",
                    help="which setting's trajectories to sum over")
    ap.add_argument("--route", default="auto", choices=("auto", "openmm", "ase"),
                    help="which engine's trajectories to read: md_openmm/basinNN (traj.dcd) "
                         "or md_ase/basinNN (md.traj); auto takes openmm when present, else ase")

    ap.add_argument("--atoms", default="all", choices=("all", "heavy", "both"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = config.load()
    temperature = config.temperature(cfg)
    basin_tag = args.basin_tag or args.tag
    rec_a = basin_reader.read_record(args.species, tag=basin_tag)
    if rec_a is None:
        raise SystemExit(
            basin_reader.missing_message(args.species, basin_tag)
            + "\n  The ensemble is a sum over the basins branch A found; without it there "
            "is nothing to sum.\n"
            "  python scripts/production/s0_A_pipeline.py --species {} --tag {}"
            .format(args.species, basin_tag))

    e_rel = basin_electronic(rec_a)
    sets = ("all", "heavy") if args.atoms == "both" else (args.atoms,)

    print("=" * 88)
    print("ENSEMBLE FREE ENERGY -- {}  tag {}  T = {} K".format(
        args.species, args.tag, temperature))
    print("{} basin(s) from branch A".format(len(e_rel)))
    print("=" * 88)

    # collect's verdict rides along. A sum over trajectories that failed their gates is
    # written (the file is the record of what was summed) and then refused at exit,
    # the way collect refuses -- except under OPENQHA_SMOKE=1, where the test chain has
    # already said every number under this tag is plumbing.
    _, criteria = collect_tables(args.species, basin_tag, args.setting, route=args.route)
    n_pass, n_crit = criteria_verdict(criteria)
    _stem = collect_stem(args.species, basin_tag, args.setting, route=args.route)
    print("collect   {} of {} criteria passed  (_records/{}/{}/collect__criteria.parquet; tag {})".format(
        n_pass, n_crit, _stem.parent.parent.name, args.setting, args.tag) if n_crit
        else "collect   wrote no criteria table for this molecule")

    report = dict(species=args.species, tag=args.tag, basin_tag=basin_tag,
                  temperature_K=temperature, n_basins_branch_a=len(e_rel),
                  collect_criteria=dict(passed=n_pass, total=n_crit), results={})
    for atoms_set in sets:
        per_basin, root = entropy_per_basin(args.species, basin_tag, cfg, atoms_set,
                                            args.setting, route=args.route)
        basins = []
        for i, de in enumerate(e_rel):
            got = per_basin.get(i)
            basins.append(dict(rel_electronic_kcal=de,
                               TS_kcal=(got or {}).get("TS_kcal")))
        text, rec = ensemble.describe(basins, temperature)
        print()
        print("-- atoms: {} ".format(atoms_set) + "-" * (72 - len(atoms_set)))
        print(text)

        xing = sum((per_basin.get(i) or {}).get("distinct_crossings", 0)
                   for i in range(len(e_rel)))
        sym = sum((per_basin.get(i) or {}).get("symmetry_crossings", 0)
                  for i in range(len(e_rel)))
        print()
        print("basin crossings across all trajectories: {} distinct, {} "
              "symmetry-equivalent".format(xing, sym))
        if xing:
            print("  ** {} DISTINCT crossings: those trajectories left their basin and "
                  "their T*S is inflated. F_conf above is not an intra-basin "
                  "answer.".format(xing))
        elif sym:
            print("  symmetry-equivalent only: no conformer changed, but the atoms moved,")
            print("  so the all-atom covariance carries them. Compare --atoms heavy.")
        rec["per_basin_detail"] = {str(k): v for k, v in per_basin.items()}
        rec["distinct_crossings"] = xing
        rec["symmetry_crossings"] = sym
        report["results"][atoms_set] = rec
        report["trajectory_root"] = str(root)

    from openqha.store import layout as _layout
    out = Path(args.out) if args.out else (
        collect_stem(args.species, basin_tag, args.setting, route=args.route).parent / "ensemble.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print()
    print("written {}".format(out))

    # A missing basin means the sum is over fewer terms than the molecule has, and the
    # answer is biased toward whatever was run. That is a failure, not a note -- in a
    # smoke run too, where a basin without a row means the plumbing lost it.
    for atoms_set, rec in report["results"].items():
        if rec.get("n_electronic_only"):
            return 1
    if n_crit and n_pass != n_crit:
        if os.environ.get("OPENQHA_SMOKE") == "1":
            print()
            print("**SMOKE RUN (OPENQHA_SMOKE=1): collect passed {} of {} criteria; F_conf "
                  "above is a test of the plumbing, not a result.**".format(n_pass, n_crit))
            return 0
        print()
        print("F_conf above sums entropies that FAILED {} of {} of collect's criteria; "
              "it is written as the record of what was summed and refused as an answer."
              .format(n_crit - n_pass, n_crit))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
