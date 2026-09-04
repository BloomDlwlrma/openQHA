"""What does changing the engine cost, in the quantity branch B actually delivers?

CALIBRATION. It measures the difference between candidate potentials in units of the
deliverable -- kcal/mol on T*S_vib and on (G - E_el) -- so that a decision about which
model to run molecular dynamics on can be made against the 1.0 kcal/mol accuracy target
instead of against adjectives.

Why it is needed
----------------
The production default is MACE-OFF23_medium, and it has a hole in its short-range
repulsion that destroys a trajectory every few picoseconds. Three models hold their wall:
MACE-OFF23-SC, MACE-OFF23_large and MACE-OFF23b_medium. Switching to one of them is a
repo-wide change -- the engine sets branch A's basins as well, and the composite notation
`RI-MP2/cc-pVTZ // <engine>` has to name one surface (D0-4).

So the question "which engine" is not settled by the wall alone. This script supplies the
other half: **how far apart are these models on the number branch B exists to produce?**

Method, and its limit
---------------------
Each model relaxes acetone on ITS OWN surface, then gets an analytic Hessian there, then
the harmonic entropy from `openqha/thermo.py`. That is the rigid-rotor-harmonic-oscillator
vibrational term, not the quasi-harmonic one -- deliberately, because it needs no
trajectory and therefore no potential needs to survive one. Two models that agree here
could still disagree on anharmonicity; what this measures is the harmonic core, which is
the dominant part and the part that can be measured today.
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))

from openqha import config, engine, hessian, report, thermo  # noqa: E402


def measure(name, species, cfg, device="cpu"):
    from ase.io import read
    from ase.optimize import BFGS
    calc, engine_name, prov = engine.calculator(device=device, name=name)
    atoms = read(str(config.qm9_xyz(species, cfg)))
    atoms.calc = calc
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=0.002, steps=500)
    h, asym = hessian.analytic_hessian(atoms, calc), None
    if isinstance(h, tuple):
        h, asym = h
    rec = hessian.project_and_diagonalise(h, atoms.get_masses(), atoms.get_positions())
    nu = np.asarray(rec["frequencies_cm_inv"], dtype=float)
    spec = config.species(species, cfg)
    out = dict(engine=engine_name, species=species,
               n_imaginary=rec["n_imaginary"],
               lowest_frequency_cm_inv=float(nu.min()),
               highest_frequency_cm_inv=float(nu.max()),
               separation_gap_ratio=rec["separation_gap_ratio"],
               fmax_eV_per_A=float(np.abs(atoms.get_forces()).max()),
               frequencies_cm_inv=[float(x) for x in nu])
    if rec["n_imaginary"] == 0:
        g = thermo.g_minus_eel(atoms.get_masses(), atoms.get_positions(), nu,
                               spec["symmetry_number"], spec["electronic_degeneracy"])
        out.update(TS_vib_kcal=float(g["vibrational"]["S_vib_kcal_per_K"]
                                     * thermo.T_REF),
                   A_vib_kcal=float(g["vibrational"]["A_vib_kcal"]),
                   zero_point_kcal=float(g["vibrational"].get("zpe_kcal", float("nan"))),
                   G_minus_Eel_kcal=float(g["G_minus_Eel_kcal"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default="dsgdb9nsd_000018")
    ap.add_argument("--engines", nargs="*", default=None)
    ap.add_argument("--reference", default=None,
                    help="engine the deltas are taken against; default is the production one")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = config.load()
    names = args.engines or list(engine.ENGINES)
    reference = args.reference or engine.engine_name()

    print("=" * 96)
    print("Branch B calibration -- what an engine change costs on the deliverable")
    print("=" * 96)
    print("species {}   reference engine {}".format(args.species, reference))
    print()

    rows = []
    for name in names:
        try:
            rows.append(measure(name, args.species, cfg, args.device))
        except Exception as exc:
            print("{:<22} SKIP: {}: {}".format(name, type(exc).__name__, exc))
    ref = next((r for r in rows if r["engine"] == reference), None)
    for r in rows:
        if ref and "TS_vib_kcal" in r and "TS_vib_kcal" in ref:
            r["TS_vib_minus_reference_kcal"] = r["TS_vib_kcal"] - ref["TS_vib_kcal"]
            r["G_minus_Eel_minus_reference_kcal"] = (r["G_minus_Eel_kcal"]
                                                     - ref["G_minus_Eel_kcal"])
            n = min(len(r["frequencies_cm_inv"]), len(ref["frequencies_cm_inv"]))
            d = (np.asarray(r["frequencies_cm_inv"][:n])
                 - np.asarray(ref["frequencies_cm_inv"][:n]))
            r["frequency_rms_deviation_cm_inv"] = float(np.sqrt((d ** 2).mean()))
            r["frequency_max_deviation_cm_inv"] = float(np.abs(d).max())

    print("{:<22} {:>6} {:>10} {:>10} {:>12} {:>12} {:>12}".format(
        "engine", "imag", "nu_min", "nu_max", "T*S_vib", "dT*S_vib", "rms d(nu)"))
    for r in rows:
        print("{:<22} {:>6} {:>10.1f} {:>10.1f} {:>12} {:>12} {:>12}".format(
            r["engine"], r["n_imaginary"], r["lowest_frequency_cm_inv"],
            r["highest_frequency_cm_inv"],
            "-" if "TS_vib_kcal" not in r else "{:.4f}".format(r["TS_vib_kcal"]),
            "-" if "TS_vib_minus_reference_kcal" not in r
            else "{:+.4f}".format(r["TS_vib_minus_reference_kcal"]),
            "-" if "frequency_rms_deviation_cm_inv" not in r
            else "{:.2f}".format(r["frequency_rms_deviation_cm_inv"])))

    out_stem = Path(args.out) if args.out else (
        _repo_root() / "analysis" / "qha" / "calibration_engine_thermo_delta")
    rp = report.Report(
        "Branch B calibration -- engine change measured on the deliverable",
        subtitle="{}   reference {}".format(args.species, reference))
    rp.section("What this is for")
    rp.note("The short-range wall decides which engines can run molecular dynamics at "
            "all. This decides what choosing among them costs, in the units branch B "
            "delivers, against the 1.0 kcal/mol accuracy target.")
    rp.note("Harmonic, not quasi-harmonic: each model is relaxed on its own surface and "
            "given an analytic Hessian there, so no trajectory has to survive. Models "
            "agreeing here could still differ in anharmonicity.")
    rp.table(["engine", "imaginary", "nu_min", "nu_max", "T*S_vib", "delta T*S_vib",
              "rms d(nu)", "max d(nu)"],
             [[r["engine"], r["n_imaginary"], round(r["lowest_frequency_cm_inv"], 1),
               round(r["highest_frequency_cm_inv"], 1),
               "-" if "TS_vib_kcal" not in r else round(r["TS_vib_kcal"], 4),
               "-" if "TS_vib_minus_reference_kcal" not in r
               else round(r["TS_vib_minus_reference_kcal"], 4),
               "-" if "frequency_rms_deviation_cm_inv" not in r
               else round(r["frequency_rms_deviation_cm_inv"], 2),
               "-" if "frequency_max_deviation_cm_inv" not in r
               else round(r["frequency_max_deviation_cm_inv"], 2)] for r in rows],
             units=[None, None, "cm^-1", "cm^-1", "kcal/mol", "kcal/mol", "cm^-1",
                    "cm^-1"])
    rp.json_dump(dict(species=args.species, reference=reference, engines=rows))
    log = rp.write(str(out_stem) + ".log")
    flat = [{k: v for k, v in r.items() if k != "frequencies_cm_inv"} for r in rows]
    written = report.write_parquet(dict(engines=flat), out_stem)
    print("\nwritten:\n  {}".format(log))
    for p, n, _c in written:
        print("  {}  ({} rows)".format(p, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
