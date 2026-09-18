"""msRRHO free energy of one molecule: the MACE level, the reference level, the comparison.

PRODUCTION. One Calculation per invocation, on one molecule directory (ADR 0001) found
from --species and --tag; records go to the molecule's level folder `levels/<level>/`
(ADR 0004). Three steps, each a separate --step so a Batch can run them on different
machines (the reference level needs ORCA; the other two need nothing but the files):

    mace             levels/mace-off23_medium/degeneracy.*  thermo_msrrho.*      (ticket 24)
    reference        orca/<level>/basinNN/job.*  levels/<level>/thermo_msrrho.*  merge_map.dat
                                                                                (ticket 26)
    hessian_compare  mace/basinNN/{hessian,forces}_at_<level>.npy  levels/hessian_compare.*
                     (the MACE Hessian at the reference geometry, ticket 27; needs MACE)
    compare          levels/level_compare.*                                      (ticket 26/27)
    mode_curvature   orca/<level>/basinNN/modeII_k*/  levels/<level>/mode_curvature_dryrun.*
                     (the higher level's curvature along the wB97M modes from energies,
                     the dry run before a numerical Hessian; ticket 32)
    write_jobs       <jobs-dir>/<species>_<level>/{*.inp,worker.sh,run.sbatch,README.md}
                     (the hkuhpc bundle of a numerical reference level; not submitted here)

The reference level is `wb97m-d3bj_def2-tzvppd` (`! wB97M-D3BJ def2-TZVPPD TightOpt Freq
TightSCF`, the analytic Hessian; ORCA 6.0.1 measured 226 s for propanal basin 0 on 8
cores). Production reference calculations run on deimos with ORCA 6.1.1 (D0-75); a
local run is a dry run and says so through the ORCA version in the record. A finished
basin is never recomputed, so a killed Batch resumes where it stopped.

Usage:
    python scripts/production/s0_thermo_msrrho.py --species dsgdb9nsd_000035 --tag propanal --step mace
    python scripts/production/s0_thermo_msrrho.py --species dsgdb9nsd_000035 --tag propanal --step reference --nprocs 8
    python scripts/production/s0_thermo_msrrho.py --species dsgdb9nsd_000035 --tag propanal --step compare
"""
import argparse
import sys
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))
from openqha.potentials import engine                       # noqa: E402
from openqha.store import basins as basin_reader            # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", required=True)
    ap.add_argument("--tag", default="prod")
    ap.add_argument("--step", required=True,
                    choices=("mace", "reference", "hessian_compare", "compare", "mode_curvature", "write_jobs"))
    ap.add_argument("--level", default=None,
                    help="level name (CONTEXT.md spelling). For reference / hessian_compare / compare: the reference "
                         "level (default {}); for mode_curvature / write_jobs: the HIGHER level probed at the "
                         "reference geometry (default dlpno-ccsdt_cc-pvtz)".format(engine.REFERENCE_LEVEL))
    ap.add_argument("--keywords", default=None,
                    help="ORCA keyword line for the reference level (default: orca.LEVELS[--level])")
    ap.add_argument("--start-from", default=None,
                    help="reference: start the optimisation from this level's relaxed geometries "
                         "(e.g. wb97m-d3bj_def2-tzvppd for a numerical CCSD(T) optimisation)")
    ap.add_argument("--modes", default="lowest",
                    help="mode_curvature: lowest, low, all, or comma-separated mode indices")
    ap.add_argument("--jobs-dir", default=None, help="write_jobs: where the hkuhpc bundle goes")
    ap.add_argument("--nprocs", type=int, default=8)
    ap.add_argument("--maxcore", type=int, default=3000)
    ap.add_argument("--basins", default=None,
                    help="comma-separated MACE basin indices to run at the reference level "
                         "(default: all)")
    ap.add_argument("--preset", default="crest", choices=("crest", "xtb", "grimme2012"))
    ap.add_argument("--policy", default="refuse", choices=("refuse", "invert_below", "crest_native"),
                    help="imaginary-mode policy of the [Result] block at the MACE level; every "
                         "record carries [Imaginary_Spread] with all three (ticket 28)")
    args = ap.parse_args()
    if args.level is None:
        # the two families of steps mean different things by --level: probing the reference
        # against itself (LEVEL_ERROR_CM identically 0) was the failure a shared default gave
        args.level = "dlpno-ccsdt_cc-pvtz" if args.step in ("mode_curvature", "write_jobs") else engine.REFERENCE_LEVEL

    molecule = basin_reader.molecule_for(args.species, args.tag)
    rec = basin_reader.read_record(args.species, tag=args.tag)
    if rec is None:
        raise SystemExit(basin_reader.missing_message(args.species, args.tag))
    print("molecule directory  {}".format(molecule))

    if args.step == "mace":
        from openqha.thermochem import msrrho_ensemble as me
        level = engine.level_name(rec["Calculation_Info"]["ENGINE"])
        out = me.run_calculation(molecule, level=level, qm9_index=args.species, preset=args.preset,
                                 imaginary_policy=args.policy)
        print("level {}: S_abs = {:.3f} cal/mol/K  G_total = {:.4f} kcal/mol  basins {} (excluded {})"
              .format(level, out["S_abs_cal_per_K"], out["G_total_kcal"], out["n_basins"], out["n_excluded"]))
        if out["experimental"]:
            print("experiment {:.2f} [{}]: S_abs - experiment = {:+.3f}".format(
                out["experimental"][0], out["experimental"][1],
                out["S_abs_cal_per_K"] - out["experimental"][0]))
        for r in out["imaginary_spread"]:
            if r.get("AVAILABLE") is False:            # a preset without an ithr: the policy has no answer
                print("  policy {:13s} not available for this preset".format(r["POLICY"]))
                continue
            print("  policy {:13s} S_abs {}  included {} excluded {} inverted {} kept negative {}".format(
                r["POLICY"], "{:.4f}".format(r["S_ABS"]) if r.get("S_ABS") is not None else "-",
                r.get("N_INCLUDED", "-"), r.get("N_EXCLUDED", "-"), r.get("N_INVERTED", "-"), r.get("N_KEPT_NEGATIVE", "-")))
        print("record  {}".format(out["record"]))
    elif args.step == "reference":
        from openqha.qm_interfaces import orca
        from openqha.thermochem import reference_level as rl
        basins = [int(x) for x in args.basins.split(",")] if args.basins else None
        out = rl.run_calculation(molecule, level=args.level, keywords=args.keywords,
                                 nprocs=args.nprocs, maxcore=args.maxcore, basins=basins,
                                 qm9_index=args.species, preset=args.preset, start_from=args.start_from)
        print("level {}: S_abs = {:.3f} cal/mol/K  G_total = {:.4f} kcal/mol  reference basins {} "
              "(excluded {})  Hessian route {}".format(
                  args.level, out["S_abs_cal_per_K"], out["G_total_kcal"], out["n_basins"],
                  out["n_excluded"], ",".join(out["hessian_routes"])))
        for r in out["merge_map"]:
            print("  MACE basin {} -> {} {}  shift {:.3f} A  imaginary {}  {:.0f} s".format(
                r["mace_basin"], r["status"], r["reference_basin"], r["rmsd_displacement_A"],
                r["n_imaginary"], r["seconds"] or 0))
        print("record  {}".format(out["record"]))
    elif args.step == "mode_curvature":
        from openqha.thermochem import mode_curvature as mc
        basins = [int(x) for x in args.basins.split(",")] if args.basins else None
        out = mc.run_calculation(molecule, level=args.level, qm9_index=args.species, modes=args.modes,
                                 basins=basins, nprocs=args.nprocs, maxcore=args.maxcore, preset=args.preset)
        for r in out["modes"]:
            if r["COMPLETE"]:
                print("  basin {} mode {} ({:.1f} cm^-1): {} {:.2f} (dq err {:.2f}, anharm {:.2f})  wB97M fd {:.2f}  "
                      "MACE fd {:.2f}  model {:+.2f}  level {:+.2f}  {:.0f} s/pt".format(
                          r["BASIN"], r["MODE"], r["OMEGA_R"], args.level, r["OMEGA_HIGHER_5PT"],
                          r["DELTA_CONVERGENCE_CM"], r["FD_ERROR_CM"], r["OMEGA_REF_FD"], r["OMEGA_ENGINE_FD"],
                          r["MODEL_ERROR_CM"], r["LEVEL_ERROR_CM"], r.get("SECONDS_PER_POINT") or 0))
                if "WELL_DEPTH_HIGHER_CM" in r:
                    print("    wide profile well depth (cm^-1): {} {:+.1f}  wB97M {:+.1f}  MACE {:+.1f}".format(
                        args.level, r["WELL_DEPTH_HIGHER_CM"], r["WELL_DEPTH_REF_CM"], r["WELL_DEPTH_ENGINE_CM"]))
            else:
                print("  basin {} mode {}: INCOMPLETE".format(r["BASIN"], r["MODE"]))
        print("record  {}".format(out["record"]))
    elif args.step == "write_jobs":
        from openqha.qm_interfaces import orca_jobs
        from openqha.qm_interfaces import orca as _orca
        from openqha.store import dat as _dat, layout as _layout
        from ase.io import read as _read
        import numpy as _np
        start = args.start_from or engine.REFERENCE_LEVEL
        jobs = []
        rows = _dat.read_table(_layout.level_dir(molecule, start) / "merge_map.dat")
        for r in rows:
            if r["status"] != "kept":
                continue
            b = int(r["mace_basin"])
            parsed = _orca.parse_hess(_layout.orca_level_dir(molecule, start, b) / "job.hess")
            jobs.append(dict(name="{}_basin{:02d}".format(args.species, b), symbols=list(parsed["symbols"]),
                             positions=(_np.asarray(parsed["positions_bohr"]) / _orca.BOHR_PER_ANGSTROM).tolist()))
        out_dir = Path(args.jobs_dir or (Path(molecule) / "jobs")) / "{}_{}".format(args.species, args.level)
        out = orca_jobs.write_bundle(out_dir, jobs, args.level, nprocs=args.nprocs)
        print("bundle  {}  ({} jobs, {} ranks each, {} concurrent, maxcore {} MB)".format(
            out["dir"], len(jobs), args.nprocs, out["concurrent"], out["maxcore"]))
        for name, n, e in out["estimates"]:
            print("  {}: {} atoms  ~{} opt + {} NumFreq = {} single points".format(name, n, e["opt"], e["numfreq"], e["total"]))
    elif args.step == "hessian_compare":
        from openqha.thermochem import hessian_compare as hc
        basins = [int(x) for x in args.basins.split(",")] if args.basins else None
        out = hc.run_calculation(molecule, reference_level=args.level, qm9_index=args.species,
                                 preset=args.preset, basins=basins)
        print("hessian_compare: {} basins ({} engine Hessians computed now)".format(
            len(out["basins"]), out["n_computed_now"]))
        for r in out["basins"]:
            print("  MACE basin {} -> ref {}: H MAE {:.4f} eV/A^2  freq MAE {:.2f} (low {:.2f}) cm^-1  "
                  "slope {:.4f}  cos v1 {:.4f}  mixing {:.4f}  T*S low delta {:+.5f} kcal/mol".format(
                      r["MACE_BASIN"], r["REF_BASIN"], r["HESSIAN_MAE"], r["FREQ_MAE_CM"], r["FREQ_MAE_LOW_CM"],
                      r["SOFTENING_SLOPE"], r["EIGVEC1_COS_ECKART"], r["MIXING"], r["TS_LOW_DELTA"]))
        for k, v in out["ensemble"].items():
            print("  {:22s} {}".format(k, "{:+.5f}".format(v) if isinstance(v, float) else v))
        print("record  {}".format(out["record"]))
    else:
        from openqha.thermochem import reference_level as rl
        out = rl.level_compare(molecule, qm9_index=args.species, reference_level=args.level)
        for r in out["levels"]:
            print("  {:28s} present {}  S_abs {}".format(
                r["LEVEL"], r["PRESENT"], "{:.3f}".format(r["S_ABS"]) if r["PRESENT"] else "-"))
        for k, v in out["tiers"].items():
            print("  {:22s} {}".format(k, "{:+.4f}".format(v) if isinstance(v, float) else v))
        print("record  {}".format(out["record"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
