"""02c -- is MACE-OFF's curvature good enough to do thermochemistry with?

EXAMPLE. Three levels on the same molecules, all the way from energies and forces to
`G - E_el`:

    MACE-OFF23_medium      the production potential
    GFN2-xTB               branch A's workhorse -- the level that PICKS the geometries
    RI-MP2/RIJK/cc-pVTZ    the reference (plan_C section 5.3, user-specified)

WHAT IT PRODUCES, AND WHY IT IS NEEDED BEFORE ANYTHING ELSE
-----------------------------------------------------------
A number for how far MACE's thermochemistry is from the reference. Every other question
in this project -- whether a quasi-harmonic `nu` may stand in for a Hessian `omega`,
whether a 0.4 kcal/mol symmetry term matters, whether a correction is worth its cost --
is a comparison against that number. Without it there is no scale, and 0.3 kcal/mol is
either negligible or fatal depending on an assumption nobody wrote down.

THREE THINGS THAT WOULD MAKE THIS MEASUREMENT MEAN LESS THAN IT LOOKS
---------------------------------------------------------------------
1. **MACE-OFF23's own reference level is wB97M-D3(BJ)/def2-TZVPPD** (SPICE; see the
   MACE-OFF23 data release). It was never trained towards RI-MP2/cc-pVTZ. So
   `MACE - RI-MP2` is a model error PLUS a level difference, and this script cannot
   separate them. It reports the total and says so; separating them needs a fourth
   column at MACE's own level, which is not run here.
2. **A Hessian is only a Hessian at a stationary point of the same surface.** Each
   level is therefore relaxed to its OWN minimum before its Hessian is taken, and the
   displacement from MACE's minimum is reported. Evaluating GFN2's second derivatives
   at MACE's geometry would fold the difference between two minima into what looks
   like a curvature error.
3. **Forces are compared at ONE common geometry** -- MACE's minimum -- because a force
   comparison between two different geometries measures the geometries. At a MACE
   minimum MACE's own forces are ~0 by construction, so what the other two levels show
   there is how far their gradient is from zero at MACE's answer, which is the
   quantity that actually matters downstream.

    python examples/02c_hessian_benchmark_levels/s0_level_benchmark.py \
        --species dsgdb9nsd_000018 --tag 02c_prod --levels mace,gfn2
"""
import argparse
import json
import re
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

from openqha import config                                             # noqa: E402
from openqha.potentials import engine                                  # noqa: E402
from openqha.qm_interfaces import orca, xtb                            # noqa: E402
from openqha.quasi_harmonic import mode_match                          # noqa: E402
from openqha.store import basins as basin_reader                       # noqa: E402
from openqha.store import branch_a_property                          # noqa: E402
from openqha.thermochem import hessian as hess_mod                     # noqa: E402
from openqha.thermochem import thermo                                  # noqa: E402

#: The reference route. Held identical to the production driver by
#: `assert_route_matches_production` below rather than by anyone remembering.
ORCA_OPTFREQ_ROUTE = "! RI-MP2 cc-pVTZ cc-pVTZ/C cc-pVTZ/JK RIJK TightOpt NumFreq TightSCF"
#: Single-point gradients at the common geometry use the same correlation level without
#: the optimiser.
ORCA_ENGRAD_METHOD = "RI-MP2"
ORCA_ENGRAD_BASIS = "cc-pVTZ cc-pVTZ/C cc-pVTZ/JK RIJK TightSCF"

#: Bands for the signed frequency deviation. plan_C section 8.5's convention: the sign
#: is what carries the systematic-softening prediction, so it is never made absolute.
BANDS = ((0.0, 500.0, "below_500"), (500.0, 1500.0, "500_to_1500"),
         (1500.0, float("inf"), "above_1500"))


def assert_route_matches_production():
    """The reference level here must be the one the production driver uses.

    Two copies of a route line is one copy too many, and the failure mode of a drifted
    copy is a benchmark that silently measures a different level from the production
    numbers it is meant to calibrate.
    """
    src = (ROOT / "scripts" / "production" / "s0_branch2_opt_freq.py").read_text(
        encoding="utf-8")
    m = re.search(r'^SIMPLE\s*=\s*"([^"]+)"', src, re.M)
    if not m:
        raise RuntimeError("no SIMPLE route line found in s0_branch2_opt_freq.py")
    if m.group(1).strip() != ORCA_OPTFREQ_ROUTE.strip():
        raise ValueError(
            "this example's reference route has drifted from the production driver's.\n"
            "  production: {}\n  here      : {}".format(m.group(1), ORCA_OPTFREQ_ROUTE))
    return m.group(1)


# ---------------------------------------------------------------------- the three levels
def level_mace(symbols, positions, calc):
    """MACE at branch A's own minimum -- no re-relaxation, it is already the minimum."""
    from ase import Atoms
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.calc = calc
    t0 = time.time()
    h, asym = hess_mod.analytic_hessian(atoms, calc)
    seconds = time.time() - t0
    masses = atoms.get_masses()
    rec = hess_mod.project_and_diagonalise(h, masses, positions)
    return dict(level="MACE-OFF23_medium",
                frequencies_cm_inv=[float(x) for x in rec["frequencies_cm_inv"]],
                n_imaginary=int(rec["n_imaginary"]),
                positions_A=np.asarray(positions).tolist(),
                max_displacement_A=0.0,
                displacement_note="branch A's basin IS a MACE minimum; not re-relaxed",
                energy_eV=float(atoms.get_potential_energy()),
                max_force_eV_A=float(np.abs(atoms.get_forces()).max()),
                hessian_asymmetry_eV_A2=float(asym),
                seconds=float(seconds),
                masses_amu=[float(x) for x in masses])


def level_gfn2(symbols, positions, masses, workdir=None):
    """GFN2-xTB: relax to its own minimum, then its analytic Hessian."""
    rec = xtb.optimise_and_hessian(symbols, positions, gfn=2, workdir=workdir,
                                   keep=workdir is not None)
    check = xtb.verify_frequencies(rec, masses)
    freqs = mode_match.projected_modes(rec["hessian_eV_A2"], masses,
                                       rec["positions_A"], "hessian")[0]
    return dict(level=rec["method"], version=rec["version"],
                frequencies_cm_inv=[float(x) for x in freqs],
                n_imaginary=int((np.asarray(freqs) < 0).sum()),
                positions_A=rec["positions_A"],
                max_displacement_A=rec["max_displacement_A"],
                parse_check=check,
                seconds=rec["wall_seconds"], command=rec["command"])


def _read_rimp2(stem, masses, positions, seconds=None, reused=False):
    """Turn a finished ORCA `.hess` into this example's level record.

    Shared by the fresh run and the resume so the two cannot report different things
    about the same file.
    """
    parsed = orca.parse_hess(str(stem) + ".hess")
    check = orca.verify_hess_frequencies(parsed)
    pos_A = parsed["positions_bohr"] / orca.BOHR_PER_ANGSTROM
    h_ev = orca.hessian_to_ev_per_angstrom2(parsed["hessian_eh_bohr2"])
    freqs = mode_match.projected_modes(h_ev, masses, pos_A, "hessian")[0]
    return dict(level="RI-MP2/RIJK/cc-pVTZ", route=ORCA_OPTFREQ_ROUTE,
                frequencies_cm_inv=[float(x) for x in freqs],
                orca_frequencies_cm_inv=[float(x) for x in parsed["frequencies_cm_inv"]],
                n_imaginary=int((np.asarray(freqs) < 0).sum()),
                positions_A=pos_A.tolist(),
                max_displacement_A=float(
                    np.abs(pos_A - np.asarray(positions, float)).max()),
                parse_check=check, seconds=seconds, reused_existing_hess=bool(reused),
                workdir=str(Path(stem).parent))


def level_rimp2(symbols, positions, masses, workdir, nprocs, maxcore, timeout_s,
                refresh=False):
    """RI-MP2/RIJK/cc-pVTZ `TightOpt NumFreq`, then our own projection of its Hessian.

    ORCA's own `THERMOCHEMISTRY` block is read for nothing: it guesses the symmetry
    number, and this repository has already measured it guessing acetone wrong (C1
    instead of C2v, worth RT ln 2 = 0.41 kcal/mol). Frequencies come out; free energies
    are assembled by `thermo` from a declared sigma.
    """
    import subprocess
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    stem = workdir / "optfreq"

    # Reuse a finished job. Half an hour of MP2 per structure is worth resuming, and the
    # reuse is safe because the .hess is only accepted if `verify_hess_frequencies`
    # still passes on it -- a truncated or mismatched file fails there rather than being
    # trusted for existing. `--refresh` forces the run.
    if not refresh and Path(str(stem) + ".hess").exists():
        try:
            return _read_rimp2(stem, masses, positions, reused=True)
        except Exception as exc:                                        # noqa: BLE001
            print("   existing {}.hess rejected ({}); rerunning".format(stem, exc))

    lines = [ORCA_OPTFREQ_ROUTE,
             "%maxcore {}".format(int(maxcore)),
             "%pal nprocs {} end".format(int(nprocs)),
             "* xyz 0 1"]
    for s, r in zip(symbols, np.asarray(positions, dtype=float)):
        lines.append("{:2s} {:18.10f} {:18.10f} {:18.10f}".format(s, *r))
    lines.append("*")
    Path(str(stem) + ".inp").write_text("\n".join(lines) + "\n", encoding="utf-8")

    t0 = time.time()
    with open(str(stem) + ".out", "w") as fh:
        subprocess.call([orca.orca_binary(), str(stem) + ".inp"], stdout=fh,
                        stderr=subprocess.STDOUT, cwd=str(workdir), timeout=timeout_s)
    seconds = time.time() - t0
    out = Path(str(stem) + ".out").read_text(encoding="utf-8", errors="replace")
    if "****ORCA TERMINATED NORMALLY****" not in out:
        raise RuntimeError("ORCA did not finish normally in {}. Tail:\n{}".format(
            workdir, "\n".join(out.splitlines()[-25:])))

    return _read_rimp2(stem, masses, positions, seconds=float(seconds))


# ------------------------------------------------------------------------ comparisons
def frequency_deviation(freqs_test, freqs_ref):
    """Signed `test - ref`, paired by ascending index, split by band.

    Index pairing, not eigenvector pairing: the two spectra come from DIFFERENT
    geometries, so their eigenvectors are not expressed in a common frame and an
    overlap between them would be measuring the displacement. Eigenvector pairing is
    example 02d's business, where both mode sets come from the same geometry.
    """
    a = np.sort(np.asarray(freqs_test, dtype=float))
    b = np.sort(np.asarray(freqs_ref, dtype=float))
    if a.shape != b.shape:
        raise ValueError("different mode counts: {} vs {}".format(a.shape, b.shape))
    d = a - b
    out = dict(n=int(len(d)), mean_signed_cm=float(d.mean()),
               rms_cm=float(np.sqrt((d ** 2).mean())),
               max_abs_cm=float(np.abs(d).max()),
               mean_relative=float((d / b).mean()))
    for lo, hi, name in BANDS:
        sel = (b >= lo) & (b < hi)
        if sel.any():
            out[name] = dict(n=int(sel.sum()), mean_signed_cm=float(d[sel].mean()),
                             rms_cm=float(np.sqrt((d[sel] ** 2).mean())),
                             mean_relative=float((d[sel] / b[sel]).mean()))
    return out


def assemble(level_rec, masses, sigma, degeneracy, temperature_K):
    """`G - E_el` and its parts from one level's spectrum, with a DECLARED sigma."""
    f = np.asarray(level_rec["frequencies_cm_inv"], dtype=float)
    if (f <= 0).any():
        return dict(refused="{} imaginary/zero mode(s); thermodynamics is not defined "
                            "at a non-minimum".format(int((f <= 0).sum())))
    g = thermo.g_minus_eel(masses, level_rec["positions_A"], f, sigma, degeneracy,
                           temperature_K=temperature_K)
    t = mode_match.thermodynamic_terms(f, temperature_K)
    t["G_minus_Eel_kcal"] = float(g["G_minus_Eel_kcal"])
    t["A_rot_kcal"] = float(g["rotational"]["A_rot_kcal"])
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default="dsgdb9nsd_000018")
    # The default is what examples/02c_hessian_benchmark_levels/branchA.conf writes. Until
    # 2026-09-11 it was "prod", which no conf in this repository sets.
    ap.add_argument("--tag", default="02c_prod",
                    help="branch A tag to read the basins from; = branchA.conf's TAG")
    ap.add_argument("--levels", default="mace,gfn2",
                    help="comma separated subset of mace,gfn2,rimp2")
    ap.add_argument("--basin", type=int, default=None)
    ap.add_argument("--nprocs", type=int, default=4)
    ap.add_argument("--maxcore", type=int, default=3500)
    ap.add_argument("--timeout-s", type=int, default=36000)
    ap.add_argument("--refresh", action="store_true",
                    help="rerun ORCA even when a usable .hess is already on disk")
    ap.add_argument("--forces", action="store_true",
                    help="also compare forces at the common geometry (extra ORCA run)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    cfg = config.load()
    temperature = config.temperature(cfg)
    if "rimp2" in levels:
        print("reference route (checked against the production driver):")
        print("  " + assert_route_matches_production())

    rec_a = basin_reader.read_record(args.species, tag=args.tag)
    if rec_a is None:
        raise SystemExit(basin_reader.missing_message(args.species, args.tag))
    # The geometries are the engine files mace/basinNN/basin.extxyz (ADR 0001), located
    # by the layout, never by a path inside the record (the record crosses clusters,
    # a path in it does not).
    geoms = [(list(a.get_chemical_symbols()), np.asarray(a.get_positions(), dtype=float))
             for a in basin_reader.read_basins(args.species, tag=args.tag)]
    if not geoms:
        raise SystemExit(basin_reader.missing_message(args.species, args.tag))

    calc, engine_name, prov = engine.calculator(device="cpu")
    runs = Path(config.runs_dir("level_benchmark", cfg)) / args.tag / args.species

    print("=" * 96)
    print("02c  levels on the same molecule -- {}  tag {}  T = {} K".format(
        args.species, args.tag, temperature))
    print("levels: {}   basins: {}".format(", ".join(levels), len(geoms)))
    print("=" * 96)

    report = dict(example="02c", species=args.species, tag=args.tag,
                  temperature_K=temperature, engine=engine_name, provenance=prov,
                  levels=levels, basins={})
    for b, (symbols, positions) in enumerate(geoms):
        if args.basin is not None and b != args.basin:
            continue
        br_ = branch_a_property.basin_rows(rec_a)[b]       # the [[Basin]] block
        sigma = int(br_["SIGMA"])
        degen = int(br_.get("G0", 1))
        print()
        print("-" * 96)
        print("basin {}  sigma {}  rel {:.4f} kcal/mol".format(
            b, sigma, branch_a_property.relative_kcal(rec_a)[b]))
        print("-" * 96)

        got = {}
        m = level_mace(symbols, positions, calc)
        masses = np.asarray(m["masses_amu"], dtype=float)
        got["mace"] = m
        if "gfn2" in levels:
            got["gfn2"] = level_gfn2(symbols, positions, masses,
                                     workdir=runs / "basin{:02d}".format(b) / "gfn2")
        if "rimp2" in levels:
            got["rimp2"] = level_rimp2(symbols, positions, masses,
                                       runs / "basin{:02d}".format(b) / "rimp2",
                                       args.nprocs, args.maxcore, args.timeout_s,
                                       refresh=args.refresh)

        hdr = "{:>22} {:>7} {:>9} {:>9} {:>9} {:>10} {:>10} {:>9}"
        print(hdr.format("level", "n_imag", "nu_min", "nu_max", "dx_max/A",
                         "ZPE", "T*S", "G-Eel"))
        terms = {}
        for k in ("mace", "gfn2", "rimp2"):
            if k not in got:
                continue
            r = got[k]
            f = np.asarray(r["frequencies_cm_inv"], dtype=float)
            t = assemble(r, masses, sigma, degen, temperature)
            terms[k] = t
            print(hdr.format(
                r["level"][:22], r["n_imaginary"], "%.2f" % f.min(), "%.2f" % f.max(),
                "%.4f" % r["max_displacement_A"],
                "refused" if "refused" in t else "%.4f" % t["ZPE_kcal"],
                "refused" if "refused" in t else "%.4f" % t["TS_kcal"],
                "refused" if "refused" in t else "%.4f" % t["G_minus_Eel_kcal"]))

        dev = {}
        ref = "rimp2" if "rimp2" in got else "gfn2"
        for k in got:
            if k == ref:
                continue
            dev[k + "_vs_" + ref] = frequency_deviation(
                got[k]["frequencies_cm_inv"], got[ref]["frequencies_cm_inv"])
        if dev:
            print()
            print("signed frequency deviation against {}, by band (cm^-1)".format(ref))
            h2 = "{:>18} {:>16} {:>16} {:>16} {:>10}"
            print(h2.format("pair", "below 500", "500-1500", "above 1500", "rms all"))
            for name, d in dev.items():
                cells = []
                for _lo, _hi, bn in BANDS:
                    cells.append("{:+.1f} (n={})".format(d[bn]["mean_signed_cm"],
                                                         d[bn]["n"])
                                 if bn in d else "-")
                print(h2.format(name[:18], cells[0], cells[1], cells[2],
                                "%.1f" % d["rms_cm"]))
            print()
            print("thermochemistry difference against {} (kcal/mol)".format(ref))
            h3 = "{:>18} {:>10} {:>10} {:>10} {:>12}"
            print(h3.format("pair", "dZPE", "dE_vib", "dT*S", "d(G-Eel)"))
            for k in got:
                if k == ref or "refused" in terms.get(k, {}) or \
                        "refused" in terms.get(ref, {}):
                    continue
                a, c = terms[k], terms[ref]
                print(h3.format(
                    k + "-" + ref, "%+.4f" % (a["ZPE_kcal"] - c["ZPE_kcal"]),
                    "%+.4f" % (a["E_vib_kcal"] - c["E_vib_kcal"]),
                    "%+.4f" % (a["TS_kcal"] - c["TS_kcal"]),
                    "%+.4f" % (a["G_minus_Eel_kcal"] - c["G_minus_Eel_kcal"])))

        report["basins"][str(b)] = dict(basin_index=b, sigma=sigma,
                                        electronic_degeneracy=degen,
                                        levels=got, terms=terms, deviation=dev,
                                        reference_level=ref)

    out = Path(args.out) if args.out else (
        ROOT / "analysis" / "levels" / args.tag /
        "{}_02c_level_benchmark.json".format(args.species))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print()
    print("written {}".format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
