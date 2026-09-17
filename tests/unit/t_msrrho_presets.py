"""Ticket 22: msRRHO presets and the per-basin term; level names and composite_notation.

UNIT. Pure Python, no engine, under a second.

The numbers below are the acetone basin of examples/02d (MACE-OFF23_medium analytic
Hessian, 24 projected frequencies) and were first computed on 2026-09-15 with a
standalone re-implementation of CREST's `thermo.f90::thermodyn`; the independent check
against the real `crest --thermo` is tests/integration/t_msrrho_crest_thermo.py.
"""
import sys
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))
from openqha.potentials import engine                       # noqa: E402
from openqha.thermochem import thermo                       # noqa: E402

ACETONE_SYMBOLS = list("CCOCHHHHHH")
ACETONE_MASSES = [12.011, 12.011, 15.999, 12.011, 1.008, 1.008, 1.008, 1.008, 1.008, 1.008]
ACETONE_POSITIONS = [
    [1.2427253969, -0.1558021620, 0.3132645858],
    [-0.0094217187, -0.4181877846, -0.4886516094],
    [-0.0271062658, -1.2031304402, -1.4058562414],
    [-1.2381879362, 0.3572017568, -0.0779292560],
    [2.0610761927, -0.7604539721, -0.0625966511],
    [1.0634154821, -0.3843875617, 1.3640228750],
    [1.5040140102, 0.9010251752, 0.2554666407],
    [-2.0760681643, 0.0950294877, -0.7149503953],
    [-1.0399240397, 1.4270616965, -0.1456600166],
    [-1.4805229571, 0.1416438044, 0.9628900681],
]
ACETONE_OMEGA_CM = [
    79.7153, 136.3472, 374.5261, 486.9488, 537.6047, 804.9732, 884.1629, 895.6035,
    1077.5456, 1116.7463, 1241.1356, 1384.9331, 1397.0063, 1455.4897, 1459.8896,
    1464.5768, 1480.9083, 1818.7887, 3062.8335, 3065.9766, 3116.4022, 3121.4043,
    3180.4881, 3182.5959,
]

#: T*S_vib at 298.15 K, kcal/mol, per preset (2026-09-15 measurement).
EXPECTED_TS = dict(HO=2.9297, crest=2.9272, xtb=2.8953, grimme2012=2.7367)
TOL_TS = 0.0006          # the literals are rounded to 4 decimals


def check(cond, msg):
    if not cond:
        print("FAIL: " + msg)
        sys.exit(1)
    print("ok    " + msg)


def main():
    kw = dict(masses=ACETONE_MASSES, positions=ACETONE_POSITIONS, temperature_K=298.15)

    # --- the three presets against the recorded numbers ------------------------------
    ho = thermo.vibrational(ACETONE_OMEGA_CM, 298.15)["S_vib_kcal_per_K"] * 298.15
    check(abs(ho - EXPECTED_TS["HO"]) < TOL_TS, "HO T*S_vib = %.4f" % ho)
    out = {}
    for name in ("crest", "xtb", "grimme2012"):
        r = thermo.msrrho(ACETONE_OMEGA_CM, preset=name, **kw)
        out[name] = r
        check(abs(r["TS_vib_kcal"] - EXPECTED_TS[name]) < TOL_TS,
              "%s preset T*S_vib = %.4f (expected %.4f)" % (name, r["TS_vib_kcal"],
                                                            EXPECTED_TS[name]))
        check(r["preset"] == name and r["fscal"] == 1.0 and r["imaginary_policy"] == "refuse",
              "%s record names its preset, fscal 1.0 and the refuse policy" % name)
    check(out["crest"]["tau_cm"] == 25.0 and out["xtb"]["tau_cm"] == 50.0
          and out["grimme2012"]["tau_cm"] == 100.0, "tau per preset is 25 / 50 / 100")
    check(abs(out["grimme2012"]["rotor_cap_kg_m2"] - 1.0e-44) < 1e-60,
          "grimme2012 caps the rotor moment at the fixed 1e-44 kg m^2")
    check(out["crest"]["rotor_cap_kg_m2"] != out["grimme2012"]["rotor_cap_kg_m2"],
          "crest caps the rotor moment at the molecule's own mean principal moment")

    # --- what is never interpolated -------------------------------------------------
    zpe = {k: v["ZPE_kcal"] for k, v in out.items()}
    hth = {k: v["H_thermal_kcal"] for k, v in out.items()}
    check(max(zpe.values()) - min(zpe.values()) < 1e-12, "ZPE identical across presets")
    check(max(hth.values()) - min(hth.values()) < 1e-12,
          "H(T)-H(0) identical across presets")
    check("Cp_vib_kcal_per_K" in out["crest"] and out["crest"]["Cp_vib_kcal_per_K"]
          != out["grimme2012"]["Cp_vib_kcal_per_K"],
          "Cp is interpolated by crest and not by grimme2012")

    # --- the per-mode table and the spread ------------------------------------------
    modes = out["crest"]["modes"]
    check(len(modes) == 24 and modes[0]["omega_cm"] < modes[1]["omega_cm"],
          "per-mode table has 24 rows in ascending omega")
    check(abs(modes[0]["w_HO"] - 0.9904) < 5e-4 and modes[0]["TS_FR_kcal"] < modes[0]["TS_HO_kcal"],
          "lowest mode: w_HO 0.990 at tau 25 and the free-rotor entropy is below HO")
    sp = thermo.preset_spread(ACETONE_OMEGA_CM, **kw)
    check(abs(sp["max_minus_min_kcal"] - (EXPECTED_TS["HO"] - EXPECTED_TS["grimme2012"]))
          < 2 * TOL_TS, "preset spread = %.4f kcal/mol on T*S" % sp["max_minus_min_kcal"])

    # --- hard errors ----------------------------------------------------------------
    try:
        thermo.msrrho([0.4] + ACETONE_OMEGA_CM[1:], preset="crest", **kw)
        check(False, "a projected mode below 1 cm^-1 must raise")
    except ValueError as exc:
        check("rigid-body" in str(exc), "mode below 1 cm^-1 raises: " + str(exc)[:60])
    try:
        thermo.msrrho([-35.74] + ACETONE_OMEGA_CM[1:], preset="crest", **kw)
        check(False, "an imaginary mode under refuse must raise")
    except ValueError as exc:
        check("-35.74" in str(exc), "imaginary mode refused with its frequency named")
    inv = thermo.msrrho([-35.74] + ACETONE_OMEGA_CM[1:], preset="crest",
                        imaginary_policy="invert_below", **kw)
    check(inv["n_inverted"] == 1 and inv["imaginary_policy"] == "invert_below",
          "invert_below inverts a -35.74 mode under the crest ithr of -50")
    try:
        thermo.msrrho([-61.68] + ACETONE_OMEGA_CM[1:], preset="crest",
                      imaginary_policy="invert_below", **kw)
        check(False, "a mode below ithr must still raise under invert_below")
    except ValueError as exc:
        check("-61.68" in str(exc), "invert_below refuses a mode below ithr")

    # --- crest_native (ticket 28): CREST 3.0.2's third regime ------------------------
    # thermocalc.f90:209 inverts only modes in (ithr, 0); a mode below ithr stays
    # negative, gets S = 0 (thermo.f90:135-138) and still enters ZPE, H(T)-H(0), Cp.
    base = thermo.msrrho(ACETONE_OMEGA_CM, preset="crest", **kw)
    nat = thermo.msrrho([-61.68] + ACETONE_OMEGA_CM[1:], preset="crest",
                        imaginary_policy="crest_native", **kw)
    check(nat["n_kept_negative"] == 1 and nat["n_inverted"] == 0,
          "crest_native keeps a -61.68 mode (below ithr -50) instead of raising")
    kept = [m for m in nat["modes"] if m["kept_negative"]]
    check(len(kept) == 1 and kept[0]["omega_cm"] == -61.68 and kept[0]["TS_kcal"] == 0.0
          and kept[0]["TS_HO_kcal"] == 0.0 and kept[0]["TS_FR_kcal"] == 0.0,
          "the kept mode carries zero entropy (HO and free rotor both 0)")
    rest = thermo.msrrho(ACETONE_OMEGA_CM[1:], preset="crest", **kw)
    check(abs(nat["ZPE_kcal"] - (rest["ZPE_kcal"] - 0.5 * thermo.HC_KCAL * 61.68)) < 1e-12,
          "ZPE is lowered by |nu|/2 = 30.84 cm^-1 by the kept negative mode")
    check(abs(nat["S_vib_kcal_per_K"] - rest["S_vib_kcal_per_K"]) < 1e-12,
          "S_vib equals the spectrum without that mode")
    check(nat["H_thermal_kcal"] > rest["H_thermal_kcal"]
          and nat["Cp_vib_kcal_per_K"] > rest["Cp_vib_kcal_per_K"],
          "H(T)-H(0) and Cp still take the negative frequency (exp(-beta omega) > 1)")
    nat2 = thermo.msrrho([-35.74] + ACETONE_OMEGA_CM[1:], preset="crest",
                         imaginary_policy="crest_native", **kw)
    check(nat2["n_inverted"] == 1 and nat2["n_kept_negative"] == 0
          and abs(nat2["S_vib_kcal_per_K"] - inv["S_vib_kcal_per_K"]) < 1e-12,
          "a -35.74 mode is inverted under crest_native exactly as under invert_below")
    check(abs(base["S_vib_kcal_per_K"]
              - thermo.msrrho(ACETONE_OMEGA_CM, preset="crest", imaginary_policy="crest_native",
                              **kw)["S_vib_kcal_per_K"]) < 1e-12,
          "with no imaginary mode the three policies are the same number")

    # a sub-ithr spectrum under crest_native: grimme2012 (no ithr) is absent from the
    # spread and says why; the spread itself is still a number (code-review finding)
    sp2 = thermo.preset_spread([-61.68] + ACETONE_OMEGA_CM[1:], imaginary_policy="crest_native", **kw)
    check(set(sp2["TS_vib_kcal"]) == {"crest", "xtb", "HO"} and "grimme2012" in sp2["presets_absent"]
          and "ithr" in sp2["presets_absent"]["grimme2012"],
          "preset_spread under crest_native: grimme2012 absent with its reason, crest/xtb present")
    try:
        thermo.preset_spread([-61.68] + ACETONE_OMEGA_CM[1:], imaginary_policy="refuse", **kw)
        check(False, "preset_spread under refuse must raise on an imaginary spectrum")
    except ValueError as exc:
        check("no preset" in str(exc), "preset_spread under refuse raises when every preset refuses")

    # --- level names and the composite notation -------------------------------------
    check(engine.level_name("MACE-OFF23_medium") == "mace-off23_medium",
          "engine name maps to the CONTEXT.md level spelling")
    check(engine.REFERENCE_LEVEL == "wb97m-d3bj_def2-tzvppd", "reference level is wB97M")
    check(engine.composite_notation(name="MACE-OFF23_medium")
          == "wb97m-d3bj_def2-tzvppd // mace-off23_medium",
          "composite_notation is derived from the level names")
    check(engine.composite_notation(reference="dlpno-ccsdt_cc-pvtz", name="MACE-OFF23-SC")
          == "dlpno-ccsdt_cc-pvtz // mace-off23-sc", "a different reference level is spelled the same way")
    print("PASS")


if __name__ == "__main__":
    main()
