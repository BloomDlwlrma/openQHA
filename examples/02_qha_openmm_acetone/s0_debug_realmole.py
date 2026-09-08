"""The whole chain on one REAL molecule, over a grid of settings. All basins, real F_conf.

CALIBRATION. It measures how the branch B settings move the DELIVERABLE, and produces no
number that anything downstream consumes.

WHAT MAKES THIS DIFFERENT FROM `s0_qha_parameter_scan.py --stage estimator`
---------------------------------------------------------------------------
That one draws from the closed-form canonical distribution: bias is known exactly, and it
settles the estimator question in seconds. It is silent on the two things that actually
decide these settings, because neither has a closed form:

    * does the real trajectory STAY IN ITS BASIN at that length? -- the reason the 1.5 ns
      protein protocol was withdrawn on 2026-09-07;
    * do the BASINS REWEIGHT when T*S moves? The deliverable is not one basin's entropy.
      It is  F_conf = -kT ln sum_i exp(-dG_i / kT)  over every basin branch A found, and
      a setting that shifts two basins by the same amount changes no answer at all.

The second is the one that a single-basin scan cannot even see, and it is the one that
matters: a 0.3 kcal/mol wobble in T*S is alarming in a single basin and irrelevant in a
correction if it is common to all of them.

SO THIS RUNS THE WHOLE CHAIN
----------------------------
    branch A  ->  every basin of acetone
    branch B  ->  a trajectory per (basin, seed, thermostat)
    analysis  ->  per-basin T*S, then F_conf over the ensemble, per grid cell

CONFIGURED IN THE `.conf` CONVENTION, NOT ON THE COMMAND LINE
-------------------------------------------------------------
Taken from `00_QM9_reaction_eng/hkuhpc/REPT-dNN/search/*.conf`: `<NAME>_LIST=(...)` is a
swept axis, `export NAME="${NAME:-default}"` is a fixed setting, and every cell appends
one row to `$RESULT_LOG`. A grid that lives in a shell history is a grid nobody can
reproduce; this one is a file that can be diffed.

    python examples/02_qha_openmm_acetone/s0_debug_realmole.py \\
        --conf examples/02_qha_openmm_acetone/debug_realmole_smoke.conf

ONE TRAJECTORY PER (BASIN, SEED, THERMOSTAT). Length is a truncation, interval is a
stride, atom set is a mask -- all three are read off the SAME trajectory. The 150-cell
grid therefore costs (n_basins x seeds x n_thermostats) trajectories, not 150.
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import time
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
from openqha.conformer_search import crest_census                    # noqa: E402
from openqha.quasi_harmonic import basin_residence as br             # noqa: E402
from openqha.quasi_harmonic import qha                               # noqa: E402
from openqha.store import basin_store                                # noqa: E402

#: name -> the trajectory driver's thermostat arguments. See the .conf header on why
#: `nhc_*` (a coupling TIME in fs) and `langevin_*` (a friction in ps^-1) are not
#: interchangeable even when the number is the same.
THERMOSTATS = {
    "nhc_20": ["--thermostat", "nose-hoover", "--tdamp-fs", "20"],
    "nhc_100": ["--thermostat", "nose-hoover", "--tdamp-fs", "100"],
    "langevin_1": ["--thermostat", "langevin", "--friction", "1.0"],
    "langevin_5": ["--thermostat", "langevin", "--friction", "5.0"],
    "langevin_50": ["--thermostat", "langevin", "--friction", "50.0"],
}


# ======================================================================================
# The .conf format
# ======================================================================================
# A TRAILING COMMENT IS ALLOWED, and getting that wrong is not cosmetic: the first
# version required end-of-line right after the value, so every setting written as
#     export SEEDS="${SEEDS:-3}"    # 3 is the minimum that gives a blank control
# was silently dropped and took its default from THIS FILE instead of from the
# .conf. A config parser that ignores half the config is worse than no parser,
# because the run still starts. `--print-conf` is how that stays visible.
_LIST = re.compile(
    r'^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)_LIST=\(([^)]*)\)\s*(?:#.*)?$')
_VAR = re.compile(r'^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)='
                  r'(?:"\$\{\1:-(.*?)\}"|"([^"]*)"|([^\s#]+))\s*(?:#.*)?$')


def read_conf(path):
    """Parse a REPT-dNN-style `.conf` without invoking a shell.

    Parsed rather than sourced on purpose: sourcing an arbitrary file to read four
    settings executes it, and this one is read on a login node. The two forms actually
    used are the two matched above; anything else in the file is ignored, and
    `--print-conf` shows what was understood so a typo is visible rather than silently
    dropped.
    """
    lists, fixed = {}, {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = _LIST.match(line)
        if m:
            lists[m.group(1)] = [v.strip().strip('"\'')
                                 for v in m.group(2).split() if v.strip()]
            continue
        m = _VAR.match(line)
        if m:
            val = next((g for g in m.groups()[1:] if g is not None), "")
            fixed[m.group(1)] = val
    return lists, fixed


# ======================================================================================
def branch_a_basins(species, tag, threads=4, force=False):
    """Every basin branch A finds. Runs it if the product is not already there.

    NOT `--basin 0`. The deliverable is an ensemble over basins, and a grid evaluated on
    one basin cannot show the thing this file exists to show -- that a setting which
    moves T*S may move every basin together and change F_conf by nothing.
    """
    j, x = basin_store.paths_for(species, tag=tag)[:2]
    if force or not (j.exists() and x.exists()):
        print("-- branch A: {} (tag {})".format(species, tag))
        cmd = [sys.executable, "-u",
               str(ROOT / "scripts" / "production" / "s0_A_pipeline.py"),
               "--species", species, "--tag", tag, "--threads", str(threads),
               "--hessian-mode", "analytic"]
        proc = subprocess.run(cmd, cwd=str(ROOT), text=True)
        if proc.returncode != 0:
            raise SystemExit("branch A failed for {}; see analysis/branchA/{}/{}/"
                             .format(species, tag, species))
    rec = basin_store.read(species, tag=tag)
    if rec is None:
        raise SystemExit("branch A wrote no basin record at {}".format(j))
    return rec, x


def basin_energies_and_sigma(rec, species, cfg):
    """(relative electronic energy kcal, sigma, degeneracy) per basin, from branch A.

    A basin with no symmetry number is not silently given one: `config.species` raises
    if the molecule has none declared, and that refusal is deliberate (`D0-9`).
    """
    spec = config.species(species, cfg)
    out = []
    for i, b in enumerate(rec.get("basins", [])):
        e = b.get("relative_energy_kcal")
        if e is None:
            e = b.get("energy_kcal_relative", b.get("rel_kcal"))
        sigma = (b.get("symmetry") or {}).get("sigma", spec["symmetry_number"])
        out.append(dict(index=i, rel_kcal=float(e if e is not None else 0.0),
                        sigma=int(sigma),
                        degeneracy=int(spec["electronic_degeneracy"])))
    return out


# ======================================================================================
def run_trajectory(species, tag, basins_file, basin, seed_index, thermostat,
                   prod_ps, equil_ps, interval_ps, route, platform):
    """One trajectory, through the PRODUCTION driver as a subprocess.

    The real driver rather than MD written here, so this grid cannot accidentally
    measure a different protocol from the one production runs -- which is exactly the
    defect it was built to correct.
    """
    steps = int(round(interval_ps * 1000.0))            # at 1 fs
    script = ("s0_B_qha_trajectory_openmm.py" if route == "openmm"
              else "s0_B_qha_trajectory.py")
    cmd = [sys.executable, "-u", str(ROOT / "scripts" / "production" / script),
           "--species", species, "--tag", tag, "--basins", str(basins_file),
           "--basin", str(basin), "--seed-index", str(seed_index),
           "--prod-ps", str(prod_ps), "--equil-ps", str(equil_ps),
           "--sample-every", str(steps)]
    cmd += THERMOSTATS[thermostat] if route == "ase" else ["--platform", platform]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return proc, time.time() - t0, cmd


def frames_path(tag, species, basin, seed):
    return (Path(config.runs_dir("qha")) / tag / species
            / "basin{:02d}".format(basin) / "seed{:02d}".format(seed) / "frames.npy")


# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", required=True)
    ap.add_argument("--print-conf", action="store_true",
                    help="show what was parsed out of the .conf and stop")
    ap.add_argument("--force-branch-a", action="store_true")
    args = ap.parse_args()

    lists, fixed = read_conf(args.conf)
    if args.print_conf:
        print(json.dumps(dict(swept=lists, fixed=fixed), indent=2))
        return 0

    species = fixed.get("SPECIES", "dsgdb9nsd_000018")
    tag = fixed.get("TAG", "debug_realmole")
    seeds = int(fixed.get("SEEDS", 3))
    equil = float(fixed.get("EQUIL_PS", 50))
    route = fixed.get("ROUTE", "ase")
    platform = fixed.get("PLATFORM", "CPU")
    temperature = float(fixed.get("TEMPERATURE_K", 298.15))
    lengths = sorted(float(v) for v in lists.get("PROD_PS", ["500"]))
    intervals = sorted(float(v) for v in lists.get("INTERVAL_PS", ["1.0"]))
    thermostats = lists.get("THERMOSTAT", ["nhc_20"])
    atom_sets = lists.get("ANALYSIS_ATOMS", ["all"])

    cfg = config.load()
    rec_a, basins_xyz = branch_a_basins(
        species, fixed.get("BRANCH_A_TAG", tag), force=args.force_branch_a)
    basin_info = basin_energies_and_sigma(rec_a, species, cfg)
    n_basins = len(basin_info)

    from ase.io import read
    ref = read(str(basins_xyz), index="0")
    syms, masses = ref.get_chemical_symbols(), ref.get_masses()

    print("=" * 96)
    print("DEBUG REAL MOLECULE -- {}   {} atoms   {} basins from branch A".format(
        species, len(syms), n_basins))
    if n_basins < 2:
        print("NOTE: this molecule has ONE basin, so F_conf = -kT ln sum exp(-dG/kT) is")
        print("      identically 0 and the column below means nothing. The basin-")
        print("      RESIDENCE question is still answered (acetone's two methyl rotors")
        print("      are exactly the case it exists for), but the REWEIGHTING question")
        print("      needs a molecule with several conformers -- set SPECIES in the")
        print("      .conf to one, e.g. a species whose branch A record has n_basins > 1.")
    print("grid: {} lengths x {} intervals x {} thermostats x {} atom sets = {} cells"
          .format(len(lengths), len(intervals), len(thermostats), len(atom_sets),
                  len(lengths) * len(intervals) * len(thermostats) * len(atom_sets)))
    print("cost: {} basins x {} seeds x {} thermostats = {} trajectories".format(
        n_basins, seeds, len(thermostats), n_basins * seeds * len(thermostats)))
    print("=" * 96)

    longest, finest = max(lengths), min(intervals)
    traj = {}
    for thermostat in thermostats:
        for b in range(n_basins):
            for s in range(seeds):
                t_tag = "{}_{}".format(tag, thermostat)
                proc, wall, cmd = run_trajectory(
                    species, t_tag, basins_xyz, b, s, thermostat,
                    longest, equil, finest, route, platform)
                p = frames_path(t_tag, species, b, s)
                if proc.returncode != 0 or not p.exists():
                    print("   FAILED basin {} seed {} {}: {}".format(
                        b, s, thermostat, (proc.stderr or "")[-200:]))
                    continue
                traj[(thermostat, b, s)] = np.load(p)
                print("   {} basin {} seed {}: {} frames, {:.0f} s".format(
                    thermostat, b, s, len(traj[(thermostat, b, s)]), wall))

    if not traj:
        raise SystemExit("no trajectory succeeded -- nothing to analyse")

    # ---- the grid, all of it post hoc on the trajectories above -----------------------
    out_csv = ROOT / fixed.get("RESULT_LOG", "analysis/qha/debug_realmole/results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = fixed.get("RESULT_COLUMNS", "").split(",")
    rows, skipped = [], []

    print("\n{:>8} {:>9} {:>12} {:>6} {:>8} {:>11} {:>11} {:>10} {:>11} {:>6} {:>7}  {}"
          .format("len/ps", "step/ps", "thermostat", "atoms", "frames", "frames/DOF",
                  "mean T*S", "spread", "F_conf", "xing", "symxing", "crit 1"))
    for length in lengths:
        for interval in intervals:
            stride = int(round(interval / finest))
            keep = int(round(length / finest))
            for thermostat in thermostats:
                for which in atom_sets:
                    per_basin, ts_all, xing_d, xing_s, crit_ok, verdicts = {}, [], 0, 0, True, []
                    n_frames = frames_per_dof = 0
                    for b in range(n_basins):
                        vals = []
                        for s in range(seeds):
                            f = traj.get((thermostat, b, s))
                            if f is None:
                                continue
                            sub = f[:keep:stride]
                            if len(sub) < 3 * len(syms):
                                # NOT silent. A covariance from T frames has at most T
                                # non-zero eigenvalues, so below 3N the cell cannot be
                                # computed at all -- and a scan that drops every cell
                                # without saying so prints an empty table and exits 0.
                                skipped.append((length, interval, which, len(sub),
                                                3 * len(syms)))
                                continue
                            mask = (br.heavy_atom_mask(syms) if which == "heavy"
                                    else np.ones(len(syms), bool))
                            fs, ms = sub[:, mask, :], np.asarray(masses)[mask]
                            dof = 3 * int(mask.sum()) - 6
                            rec = qha.analyse(fs, ms)
                            vals.append(rec["entropy"]["TS_QH_kcal"])
                            res = br.basin_residence(sub, syms)
                            xing_d += res["distinct_basin_crossings"]
                            xing_s += res["symmetry_equivalent_crossings"]
                            v = br.interpret_saturation(qha.saturation_curve(fs, ms), res)
                            crit_ok = crit_ok and v["criterion_1_passed"]
                            verdicts.append(v["verdict"])
                            n_frames, frames_per_dof = len(sub), len(sub) / dof
                        if vals:
                            per_basin[b] = float(np.mean(vals))
                            ts_all.extend(vals)
                    if not per_basin:
                        continue

                    # THE DELIVERABLE: the ensemble, not one basin. dG_i is the basin's
                    # electronic energy plus -T*S_i, both relative to the lowest basin.
                    g = np.array([basin_info[b]["rel_kcal"] - per_basin[b]
                                  for b in sorted(per_basin)])
                    g = g - g.min()
                    f_conf = crest_census.conformational_correction_kcal(g, temperature)

                    row = dict(
                        length_ps=length, interval_ps=interval, thermostat=thermostat,
                        atoms=which, n_frames=n_frames,
                        frames_per_dof=round(frames_per_dof, 1),
                        mean_TS_kcal=round(float(np.mean(ts_all)), 4),
                        spread_TS_kcal=round(float(np.std(ts_all)), 4),
                        n_basins=len(per_basin), F_conf_kcal=round(f_conf, 4),
                        distinct_crossings=xing_d, symmetry_crossings=xing_s,
                        criterion_1="pass" if crit_ok else "FAIL",
                        verdict=max(set(verdicts), key=verdicts.count) if verdicts else "")
                    row["per_basin_TS_kcal"] = {str(k): round(v, 4)
                                                for k, v in per_basin.items()}
                    rows.append(row)
                    print("{:>8.0f} {:>9} {:>12} {:>6} {:>8} {:>11.1f} {:>11.4f} "
                          "{:>10.4f} {:>11.4f} {:>6} {:>7}  {}".format(
                              length, interval, thermostat, which, n_frames,
                              frames_per_dof, row["mean_TS_kcal"], row["spread_TS_kcal"],
                              row["F_conf_kcal"], xing_d, xing_s, row["criterion_1"]))

    if skipped:
        need = skipped[0][4]
        print()
        print("{} cell(s) SKIPPED: fewer than {} frames, which is 3N and the point below"
              .format(len(skipped), need))
        print("which the covariance cannot have enough non-zero eigenvalues to analyse.")
        for length, interval, which, got, want in skipped[:8]:
            print("   {:>6.0f} ps @ {} ps, {:>5} -> {} frames, need {}".format(
                length, interval, which, got, want))
        if len(skipped) > 8:
            print("   ... and {} more".format(len(skipped) - 8))
        print("Raise the lengths in the .conf, or shorten the interval.")
    if not rows:
        print()
        print("NO ROWS WERE PRODUCED. The table above is empty and the CSV has only a")
        print("header. This is not a passing run -- see the skip reasons above.")
        return 1

    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[c for c in cols if c] or list(rows[0]),
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out_csv.with_suffix(".json")).write_text(
        json.dumps(dict(species=species, n_basins=n_basins, basins=basin_info,
                        swept=lists, fixed=fixed, rows=rows), indent=2, default=str),
        encoding="utf-8")

    print("\nwritten {}".format(out_csv))
    print("        {}".format(out_csv.with_suffix(".json")))
    print("\nREAD `distinct_crossings` BEFORE `mean_TS_kcal`. A non-zero count means the")
    print("trajectory left its basin and T*S is inflated -- however converged it looks.")
    if fixed.get("SMOKE"):
        print("\nThis was a SMOKE conf. Not a result: the lengths are below the floor at")
        print("which this branch will read acceptance criteria 1 and 5.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
