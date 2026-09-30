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
    T*S_i     branch B, from the trajectories section of collect's Table (mean over seeds)

WHERE T*S IS READ FROM, AND WHY NOT THE FRAMES
----------------------------------------------
`s0_B_qha_analyse.py` (the collect step) writes `<molecule>/_records/md_<route>/
collect[_<setting>].dat`, section `[trajectories]` (one Table since 2026-09-16; four `.dat` from 2026-09-15; before 2026-09-14
`analysis/qha/<tag>/<species>__trajectories.parquet`), one row per (basin, seed) with the entropy it judged against the criteria.
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
from openqha.store import branch_a_property                         # noqa: E402


def basin_electronic(rec):
    """Relative electronic energy per basin, kcal/mol, from branch A's Property file:
    RELATIVE of each [[Basin]] block, looked up by name. A basin without it RAISES (in
    `branch_a_property.relative_kcal`) rather than defaulting to 0.0: an earlier version
    guessed three names and, when all missed, gave every basin zero, so propanal's three
    basins came out degenerate and F_conf read -0.6509 instead of -0.2354 with no error.
    """
    return branch_a_property.relative_kcal(rec)


def collect_stem(species, tag, setting="default", root=None, route="auto"):
    """Where s0_B_qha_analyse.py put this molecule's tables (its own default --out):
    <molecule>/_records/md_<route>/collect[_<setting>] (no setting level, records redesign
    2026-09-15). `tag` is the basin tag; route `auto` is the one the trajectories are
    found under (openmm before ase)."""
    from openqha.quasi_harmonic import chain_records, trajectory_reader
    mol = basin_reader.molecule_for(species, tag, root=root)
    r = route if route in ("openmm", "ase") else (trajectory_reader.route_found(mol, setting) or "openmm")
    return chain_records.stem(mol, r, setting, chain_records.COLLECT)


def collect_trajectories(species, tag, setting="default", root=None, route="auto"):
    """The `trajectories` section of collect's Table, `collect[_<setting>].dat`, as row
    dicts. Refuses when the Table is absent: this step sums what collect judged and has
    nothing to sum before collect has run."""
    from openqha.quasi_harmonic import chain_records
    stem = collect_stem(species, tag, setting, root, route)
    table = chain_records.collect_paths(stem)["dat"]
    if not table.is_file():
        raise SystemExit(
            "s0_B_report_ensemble: no collect product for {} under tag {!r}:\n    {}\n"
            "  This step sums the entropies collect judged; it does not analyse frames. "
            "Run collect first:\n"
            "    python scripts/production/s0_B_qha_analyse.py --species {} --tag {}"
            .format(species, tag, table, species, tag))
    tables = chain_records.read_collect_table(stem)
    if "trajectories" not in tables:
        raise SystemExit("s0_B_report_ensemble: {} has no [trajectories] section (it has {}); "
                         "nothing to sum".format(table, sorted(k for k in tables if k) or "no sections"))
    return tables["trajectories"]


def collect_verdict(species, tag, setting="default", root=None, route="auto"):
    """(passed, total) from `[Criteria]` in collect's Property file, `collect.toml` --
    the file a later step reads back. (None, None) when there is no
    Property file or no `[Criteria]` block: "no verdict", never zero."""
    from openqha.quasi_harmonic import chain_records
    from openqha.store import property as prop
    stem = collect_stem(species, tag, setting, root, route)
    toml = chain_records.collect_paths(stem)["toml"]
    if not toml.is_file():
        return None, None
    crit = (prop.load(toml) or {}).get("Criteria") or {}
    if crit.get("N_TOTAL") is None:
        return None, None
    return int(crit.get("N_PASSED") or 0), int(crit["N_TOTAL"])


def crossings_per_basin(species, tag, cfg, setting="default", root=None, route="auto"):
    """Basin residence from the frames -- the one thing collect's table does not carry.

    The frames are read from the engine folder (traj.dcd); returns
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
        rows = collect_trajectories(species, tag, setting, root, route)
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


def _msrrho_comparison(molecule, rec_a, report, temperature):
    """Lines comparing the msRRHO Calculation's per-basin T*S_vib and S_abs with the
    trajectory route's T*S, or [] when no thermo_msrrho record exists at the potential's
    level. Reads `msrrho/thermo/<level>.thermo_msrrho.toml`."""
    from openqha.potentials import engine
    from openqha.store import layout, property as prop
    engine_name = (rec_a.get("Calculation_Info") or {}).get("ENGINE")
    if not engine_name:
        return []
    try:
        level = engine.level_name(engine_name)
    except KeyError:
        return []
    path = layout.level_file(molecule, level, "thermo_msrrho.toml")
    if not path.is_file():
        return []
    doc = prop.load(path)
    res = doc.get("Result") or {}
    rows = {int(r["INDEX"]): r for r in doc.get("Basin") or []}
    lines = ["  msRRHO at {}: S_abs = {:.3f} cal/mol/K, G_total = {:.4f} kcal/mol  ({})"
             .format(level, res.get("S_ABS", float("nan")), res.get("G_TOTAL", float("nan")),
                     path)]
    if "S_EXPERIMENT" in res:
        lines.append("  experiment {:.2f} cal/mol/K [{}]: S_abs - experiment = {:+.3f}".format(
            res["S_EXPERIMENT"], res.get("S_EXPERIMENT_SOURCE"), res.get("S_ABS_MINUS_EXPERIMENT")))
    for atoms_set, rec in (report.get("results") or {}).items():
        pb = rec.get("per_basin_detail") or {}
        for b, d in sorted(pb.items()):
            r = rows.get(int(b))
            ts_h = (r.get("S_VIB") or 0.0) * temperature / 1000.0 if r and not r.get("EXCLUDED") else None
            lines.append("  basin {} ({} atoms): T*S trajectory {} kcal/mol   T*S_vib msRRHO {}".format(
                b, atoms_set, "{:.4f}".format(d["TS_kcal"]) if d else "-",
                "{:.4f}".format(ts_h) if ts_h is not None else "-"))
    return lines


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
    n_pass, n_crit = collect_verdict(args.species, basin_tag, args.setting, route=args.route)
    _stem = collect_stem(args.species, basin_tag, args.setting, route=args.route)
    print("collect   {} of {} criteria passed  ({}.toml [Criteria]; tag {})".format(
        n_pass, n_crit, _stem, args.tag) if n_crit
        else "collect   left no verdict for this molecule (no collect.toml, or no [Criteria])")

    report = dict(species=args.species, tag=args.tag, basin_tag=basin_tag, setting=args.setting,
                  route=(args.route if args.route in ("openmm", "ase") else _stem.parent.name[3:]),
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

    # The Record (records redesign, 2026-09-15): ensemble.out is the Report, written first;
    # ensemble.toml the Property file, written last (STATUS is the marker); the setting in
    # the stem, beside collect's files in _records/md_<route>/.
    from openqha.store import report as _rep
    from openqha.quasi_harmonic import chain_records
    from openqha.store import layout as _layout
    stem = (Path(args.out).with_suffix("") if args.out
            else _stem.parent / _layout.record_file_name(chain_records.ENSEMBLE, args.setting))
    stem.parent.mkdir(parents=True, exist_ok=True)
    rr = _rep.Report("openQHA -- F_conf over the ensemble",
                     subtitle="{}  tag {}  basin tag {}  setting {}".format(
                         args.species, args.tag, basin_tag, args.setting))
    for atoms_set, rec in (report.get("results") or {}).items():
        rr.section("atoms: {}".format(atoms_set))
        for k, v in rec.items():
            if not isinstance(v, (dict, list)):
                rr.kv(k, v)
        pb = rec.get("per_basin_detail") or {}
        if pb:
            rr.table(["basin", "T*S / kcal", "n_frames", "source", "distinct", "symmetry"],
                     [[b, "{:.4f}".format(d["TS_kcal"]) if d else "-", (d or {}).get("n_frames", "-"),
                       (d or {}).get("source", "-"), (d or {}).get("distinct_crossings", "-"),
                       (d or {}).get("symmetry_crossings", "-")] for b, d in sorted(pb.items())])
    # The Hessian route beside the trajectory route: when the thermo_msrrho
    # Calculation has run for this molecule at the potential's level, its per-basin
    # msRRHO T*S_vib and its S_abs are printed next to the trajectory T*S -- one line of
    # comparison, never merged into F_conf.
    msrrho_line = _msrrho_comparison(basin_reader.molecule_for(args.species, basin_tag),
                                     rec_a, report, temperature)
    if msrrho_line:
        rr.section("msRRHO (Hessian route) beside the trajectory route")
        for line in msrrho_line:
            rr.text(line)
    rr.json_dump(report, title="complete record (expanded)")
    out = rr.write(stem.with_suffix(".out"), step="ensemble")
    unknown = chain_records.write_ensemble(stem, report)
    if unknown:
        raise RuntimeError("ensemble.toml: keys outside the schema {}".format(unknown))
    print()
    print("written {} and {}".format(out, stem.with_suffix(".toml")))

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
