"""The third-party seam: our `crest` preset against the real `crest --thermo`.

INTEGRATION. Needs the crest executable (environment `s0crest`, or S0_CREST_BIN); a few
seconds. Skips when crest is absent.

Why this is the check that counts: the unit test's expected numbers were produced by a
re-implementation of CREST's `thermodyn`; a shared misreading of the Fortran would pass
there and fail here. The same 24 acetone frequencies go into a Turbomole `vibspectrum`
(six zero rows first: CREST drops |omega| < 1 cm^-1 as rigid-body modes), `crest
acetone.xyz --thermo vibspectrum --sthr 25 --ithr -50 --fscal 1.0` is run, and the VIB
row at 298.15 K is compared term by term: entropy (cal/mol/K), thermal enthalpy
(cal/mol) and heat capacity (cal/mol/K). The ROT and TR rows are not compared: CREST's
translational term is at 1 atm with historical constants and this repository's is at
1 bar, a 0.04 cal/mol/K convention difference that is not this seam's.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "unit"))
from openqha.thermochem import thermo                       # noqa: E402
from t_msrrho_presets import (ACETONE_MASSES, ACETONE_OMEGA_CM,   # noqa: E402
                              ACETONE_POSITIONS, ACETONE_SYMBOLS)

TOL_S_CAL = 0.0034            # 0.001 kcal/mol on T*S at 298.15 K
TOL_H_CAL = 1.0               # CREST prints the enthalpy to 0.001 kcal/mol
TOL_CP_CAL = 0.001
FAIL = []


def check(label, ok, detail=""):
    print("  {:62s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


def write_inputs(d):
    with open(d / "acetone.xyz", "w") as f:
        f.write("{}\nacetone basin 0, MACE-OFF23_medium\n".format(len(ACETONE_SYMBOLS)))
        for s, (x, y, z) in zip(ACETONE_SYMBOLS, ACETONE_POSITIONS):
            f.write("{:2s} {:16.10f} {:16.10f} {:16.10f}\n".format(s, x, y, z))
    with open(d / "vibspectrum", "w") as f:
        f.write("$vibrational spectrum\n#  mode     symmetry     wave number   IR intensity"
                "    selection rules\n#                         cm**(-1)        km/mol"
                "         IR     RAMAN\n")
        rows = [0.0] * 6 + list(ACETONE_OMEGA_CM)
        for i, w in enumerate(rows, 1):
            if w == 0.0:
                f.write("%6d %8s %12.2f %14.5f %6s %6s\n" % (i, "", 0.0, 0.0, "-", "-"))
            else:
                f.write("%6d %8s %12.2f %14.5f %6s %6s\n" % (i, "a", w, 0.0, "YES", "YES"))
        f.write("$end\n")


def parse_vib_row(text):
    """The `298.15  VIB  q  h  cp  s` line of CREST's thermo printout."""
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[1] == "VIB" and abs(float(parts[0]) - 298.15) < 1e-6:
            return dict(h_cal=float(parts[3]), cp_cal=float(parts[4]), s_cal=float(parts[5]))
    raise ValueError("no VIB row at 298.15 K in crest output")


def main():
    binary = os.environ.get("S0_CREST_BIN") or shutil.which("crest")
    if not binary:
        print("SKIP: no crest executable (activate s0crest or set S0_CREST_BIN)")
        return
    with tempfile.TemporaryDirectory(prefix="msrrho_crest_thermo_") as tmp:
        d = Path(tmp)
        write_inputs(d)
        out = subprocess.run([binary, "acetone.xyz", "--thermo", "vibspectrum",
                              "--sthr", "25.0", "--ithr", "-50.0", "--fscal", "1.0"],
                             cwd=str(d), capture_output=True, text=True)
        check("crest --thermo exits 0", out.returncode == 0, out.stdout[-300:])
        check("crest read 24 frequencies and found c2v / sigma 2",
              "# frequencies                          24" in out.stdout
              and "rotational number                       2" in out.stdout)
        vib = parse_vib_row(out.stdout)
    ours = thermo.msrrho(ACETONE_OMEGA_CM, ACETONE_MASSES, ACETONE_POSITIONS,
                         preset="crest", temperature_K=298.15)
    s_ours = ours["S_vib_kcal_per_K"] * 1000.0
    h_ours = ours["H_thermal_kcal"] * 1000.0
    cp_ours = ours["Cp_vib_kcal_per_K"] * 1000.0
    check("S_vib: ours %.4f vs crest %.3f cal/mol/K" % (s_ours, vib["s_cal"]),
          abs(s_ours - vib["s_cal"]) < TOL_S_CAL)
    check("H(T)-H(0): ours %.2f vs crest %.3f cal/mol" % (h_ours, vib["h_cal"]),
          abs(h_ours - vib["h_cal"]) < TOL_H_CAL)
    check("Cp_vib: ours %.4f vs crest %.3f cal/mol/K" % (cp_ours, vib["cp_cal"]),
          abs(cp_ours - vib["cp_cal"]) < TOL_CP_CAL)
    ts_ours = ours["TS_vib_kcal"]
    ts_crest = vib["s_cal"] * 298.15 / 1000.0
    check("T*S_vib agrees to 0.001 kcal/mol (%.4f vs %.4f)" % (ts_ours, ts_crest),
          abs(ts_ours - ts_crest) < 0.001)
    if FAIL:
        print("FAIL: " + ", ".join(FAIL))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
