"""Package 2 high-level frequencies -- `CCSD(T)/cc-pVTZ` numerical frequencies, the reference side of the error bar.

CALIBRATION. The reference side of the error bar.

The reference level is set by `D0-55`. **Why numerical frequencies are affordable**: ORCA 6.0.1
on this machine prints `CARTESIAN GRADIENT (ANALYTIC)` for `CCSD(T)`, so the gradient is
**analytic** and numerical frequencies need only **6N+1 gradients** (55 for 9 atoms, 61 for 10),
not the `(3N)^2`-scale single points an energy difference would need.

**Where the geometry is taken is a choice that must be declared.** Two conventions:

* `--geometry mace` (default): the high-level Hessian is computed at the **MACE-OFF23-SC minimum**.
  The two sides share a digit-for-digit identical geometry, so the difference **contains only the
  difference of curvature**, which is exactly what the error bar needs.
  The cost: that geometry is not a stationary point of the high level, so the residual gradient
  leaks into the low-frequency end through the Eckart projection -- the script reports the norm
  of the high-level residual gradient so that this contamination stays visible.
* `--geometry ccsdt`: optimise at the high level first, then compute frequencies. The most orthodox
  convention, but it costs tens of extra gradients -- **more than double**.

Usage:
    python scripts/calibration/s0_package2_highlevel_freq.py --estimate        # report the cost only, compute nothing
    python scripts/calibration/s0_package2_highlevel_freq.py --only acetamide  # a single species
    python scripts/calibration/s0_package2_highlevel_freq.py                   # all seven species
Products:
    analysis/package2/highlevel/<species>/{freq.inp,freq.out,freq.hess}
    analysis/package2/highlevel/highlevel_frequencies.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

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
from openqha import S0_ROOT, config, thermo

CFG = config.load()
T_REF = config.temperature(CFG)
OUT = S0_ROOT / "analysis" / "package2" / "highlevel"
BENCH = S0_ROOT / "analysis" / "frequency_benchmark.json"

ORCA = os.environ.get("S0_ORCA_BIN", "/home/ubuntu/packages/orca_6_0_1/orca")
# resource_budget: 8 physical cores on this machine; Open MPI counts slots by physical core.
NPROCS = int(os.environ.get("S0_ORCA_NPROCS", "8"))
MAXCORE_MB = int(os.environ.get("S0_ORCA_MAXCORE", "4000"))

TEMPLATE = """! CCSD(T) cc-pVTZ {task} TightSCF
# stage 0 package 2 high-level reference frequencies. The level is set by D0-55.
# Geometry convention: {geometry_note}
%maxcore {maxcore}
%pal nprocs {nprocs} end
{extra}* xyz 0 1
{coords}*
"""


def geometry_of(name, basin, cfg=CFG):
    p = S0_ROOT / "analysis" / "package2" / name / "basin{}.xyz".format(basin)
    if not p.exists():
        raise FileNotFoundError("package 2 basin geometry not found: {} -- run "
                                "scripts/calibration/s0_package2_hessian_benchmark.py".format(p))
    lines = p.read_text(encoding="utf-8").strip().split("\n")
    return "".join("{:2s} {:18.10f} {:18.10f} {:18.10f}\n".format(
        f[0], float(f[1]), float(f[2]), float(f[3]))
        for f in (l.split() for l in lines[2:]) if len(f) == 4)


def load_bench():
    if not BENCH.exists():
        raise FileNotFoundError("{} is absent -- run the MACE side of package 2 first".format(BENCH))
    return json.loads(BENCH.read_text(encoding="utf-8"))


def run_orca(workdir, inp_name, log):
    """Compute in a temporary directory, then move the finished product back atomically (the same convention as on the cluster)."""
    tmp = Path(tempfile.mkdtemp(prefix="s0_orca_"))
    try:
        for f in workdir.iterdir():
            if f.is_file():
                shutil.copy2(f, tmp)
        t0 = time.time()
        with (tmp / (inp_name.replace(".inp", ".out"))).open("w") as fh:
            proc = subprocess.run([ORCA, inp_name], cwd=tmp, stdout=fh,
                                  stderr=subprocess.STDOUT)
        seconds = time.time() - t0
        for pat in ("*.out", "*.hess", "*.engrad", "*.xyz"):
            for f in tmp.glob(pat):
                shutil.copy2(f, workdir)
        return proc.returncode, seconds
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def parse_hess(path):
    """Read the frequencies (cm^-1) and the atom count out of an ORCA .hess."""
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    if "$vibrational_frequencies" not in txt:
        raise RuntimeError("no $vibrational_frequencies in {}".format(path))
    block = txt.split("$vibrational_frequencies", 1)[1].split("$", 1)[0].strip().split("\n")
    n = int(block[0].split()[0])
    freqs = [float(block[1 + i].split()[1]) for i in range(n)]
    return np.asarray(freqs)


def terminated_normally(out_path):
    """Judge success **from the output file itself**, not from the exit code (skills section 2.4(c))."""
    txt = Path(out_path).read_text(encoding="utf-8", errors="replace")
    return "****ORCA TERMINATED NORMALLY****" in txt


def residual_gradient(out_path):
    """Norm of the high-level residual gradient at that geometry (Eh/bohr). It measures how far the geometry is from a high-level stationary point."""
    txt = Path(out_path).read_text(encoding="utf-8", errors="replace")
    key = "CARTESIAN GRADIENT"
    if key not in txt:
        return None
    tail = txt.rsplit(key, 1)[1].split("\n")
    vals = []
    for line in tail[2:]:
        f = line.split()
        if len(f) >= 6 and f[1] == ":":
            vals.extend(float(x) for x in f[-3:])
        elif vals:
            break
    return float(np.linalg.norm(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run a single species (English name)")
    ap.add_argument("--geometry", choices=("mace", "ccsdt"), default="mace")
    ap.add_argument("--estimate", action="store_true", help="report the cost only; submit nothing")
    ap.add_argument("--seconds-per-gradient", type=float, default=None,
                    help="measured time of one gradient, used to extrapolate the cost")
    args = ap.parse_args()

    bench = load_bench()
    species = bench["species"]
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("package 2 high-level frequencies   CCSD(T)/cc-pVTZ numerical   T = {} K".format(T_REF))
    print("=" * 100)
    print("ORCA        {}".format(ORCA))
    print("parallel    {} process(es) x {} MB".format(NPROCS, MAXCORE_MB))
    print("geometry    {}".format(
        "the MACE-OFF23-SC minimum (same geometry on both sides; the difference contains only the difference of curvature)"
        if args.geometry == "mace" else "optimise at CCSD(T)/cc-pVTZ first, then compute frequencies"))
    print()

    rows = []
    for qid, v in species.items():
        if args.only and v["name"] != args.only:
            continue
        n_at = int(v["n_atoms"])
        rows.append((qid, v, n_at, 6 * n_at + 1))

    total_grad = sum(r[3] for r in rows)
    print("{:22s} {:>8s} {:>10s} {:>12s}".format("species", "atoms", "gradients", "basin used"))
    print("-" * 100)
    for qid, v, n_at, ng in rows:
        print("{:22s} {:8d} {:10d} {:12d}".format(
            v["name"], n_at, ng, v["benchmark_vs_qm9"]["conformer_used"]))
    print("-" * 100)
    print("{:22s} {:8s} {:10d}".format("total", "", total_grad))

    spg = args.seconds_per_gradient
    if spg:
        print()
        print("extrapolated at the measured {:.0f} s/gradient: {:.1f} hours = {:.2f} days (serial, {} processes)".format(
            spg, total_grad * spg / 3600.0, total_grad * spg / 86400.0, NPROCS))
        if args.geometry == "ccsdt":
            print("  **plus the high-level geometry optimisation**, at 20-40 gradients per species, a further {:.1f}-{:.1f} hours".format(
                20 * len(rows) * spg / 3600.0, 40 * len(rows) * spg / 3600.0))
        print("  **This is a resource budget, not a physical parameter** -- it can be shortened only by")
        print("  changing machine or level, never by using fewer displacements (that changes the accuracy of the numerical frequencies).")

    if args.estimate:
        print()
        print("(`--estimate` mode; nothing was submitted)")
        return

    results = {}
    for qid, v, n_at, ng in rows:
        d = OUT / v["name"]
        d.mkdir(parents=True, exist_ok=True)
        out_file = d / "freq.out"
        hess_file = d / "freq.hess"
        if hess_file.exists() and out_file.exists() and terminated_normally(out_file):
            print("[skip] {} already has a product".format(v["name"]))
        else:
            coords = geometry_of(v["name"], v["benchmark_vs_qm9"]["conformer_used"])
            task = "NumFreq" if args.geometry == "mace" else "Opt NumFreq"
            (d / "freq.inp").write_text(TEMPLATE.format(
                task=task, maxcore=MAXCORE_MB, nprocs=NPROCS, extra="",
                geometry_note=args.geometry, coords=coords), encoding="utf-8")
            print("[run]  {}  {} gradient(s) ...".format(v["name"], ng), flush=True)
            code, seconds = run_orca(d, "freq.inp", None)
            if not out_file.exists() or not terminated_normally(out_file):
                print("  ** did not finish normally (exit code {}, {:.0f} s) -- recorded, continuing".format(
                    code, seconds))
                results[qid] = dict(name=v["name"], status="failed",
                                    returncode=code, seconds=seconds)
                continue
            print("  done in {:.0f} s = {:.2f} hours ({:.0f} s/gradient)".format(
                seconds, seconds / 3600.0, seconds / ng))

        nu_hi = parse_hess(hess_file)
        nu_hi = np.asarray(sorted(x for x in nu_hi if abs(x) > 1e-6))
        nu_lo = np.asarray(v["hessian"]["frequencies_cm_inv"])
        n_imag_hi = int((nu_hi < 0).sum())
        rec = dict(name=v["name"], status="ok", n_atoms=n_at, n_gradients=ng,
                   geometry_convention=args.geometry,
                   residual_gradient_Eh_bohr=residual_gradient(out_file),
                   n_imaginary_highlevel=n_imag_hi,
                   frequencies_highlevel_cm_inv=[float(x) for x in nu_hi],
                   frequencies_mace_cm_inv=[float(x) for x in nu_lo])
        if nu_hi.shape == nu_lo.shape and n_imag_hi == 0:
            rec["comparison"] = thermo.direct_shift(nu_lo, nu_hi, T_REF)
            c = rec["comparison"]
            print("  vs CCSD(T): mean absolute deviation {:.2f}, root-mean-square {:.2f}, maximum {:.2f} cm^-1; "
                  "signed mean {:+.2f}".format(
                      c["mean_absolute_deviation_cm_inv"],
                      c["root_mean_square_deviation_cm_inv"],
                      c["max_absolute_deviation_cm_inv"],
                      c["signed_mean_deviation_cm_inv"]))
            print("  -> difference in A_vib (measured, assuming no correlation) = {:+.4f} kcal/mol".format(
                c["delta_A_vib_kcal"]))
        else:
            rec["comparison"] = None
            print("  ** {} modes vs {}, {} imaginary at the high level -- they cannot be compared mode by mode".format(
                nu_hi.shape, nu_lo.shape, n_imag_hi))
        results[qid] = rec

    payload = dict(generated_by="scripts/calibration/s0_package2_highlevel_freq.py",
                   reference_level="CCSD(T)/cc-pVTZ numerical frequencies (D0-55)",
                   geometry_convention=args.geometry,
                   orca_binary=ORCA, nprocs=NPROCS, temperature_K=T_REF,
                   species=results)
    (OUT / "highlevel_frequencies.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("written: {}".format(OUT / "highlevel_frequencies.json"))


if __name__ == "__main__":
    main()
