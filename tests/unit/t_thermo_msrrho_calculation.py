"""The thermo_msrrho Calculation at MACE, its records, the experimental comparison.

UNIT. NumPy + RDKit, no engine; a few seconds. Runs on the real branch A product of
propanal (tests/data/propanal_molecule: branchA.toml, three basins with their MACE
Hessians, CREST's rotamer file), so the numbers are the chain's own.

Seams: the Property file the Calculation leaves; the three
algebraic guards (one-basin molecule -> zero ensemble terms; G_total from the partition
function equals the Gibbs-Shannon route; an enantiomer pair as two basins with g' = 1
equals one basin with g' = 2); the frequency floor (a window mode inverted, a below-floor
mode excluded with its reason, a sub-1 cm^-1 mode dropped and counted, the sub-floor
ensemble refusing); the configuration refusing an experimental value without a citation.
"""
import copy
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from ase.io import read


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from openqha import config                                   # noqa: E402
from openqha.thermochem import hessian as hessian_mod     # noqa: E402
from openqha.thermochem import msrrho_ensemble as me         # noqa: E402
from openqha.store import layout, property as prop, report   # noqa: E402

SRC = ROOT / "tests" / "data" / "propanal_molecule"
LEVEL = "mace-off23_medium"
FAIL = []


def check(label, ok, detail=""):
    print("  {:70s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


def _push_lowest_mode(root, target_cm):
    """Rank-one surgery on every basin's mass-weighted projected Hessian: push the lowest
    non-rigid mode to `-target_cm`, then un-weight back into `hessian.npy`. Builds a
    molecule whose every basin carries one imaginary mode (in the window, or below the
    floor), whichever basin ends up the reference basin included."""
    for b in (0, 1, 2):
        hb = np.load(root / "mace" / "basin{:02d}".format(b) / "hessian.npy")
        ab = read(str(root / "mace" / "basin{:02d}".format(b) / "basin.extxyz"), format="extxyz")
        m3 = np.repeat(ab.get_masses(), 3)
        hm = hb / np.sqrt(np.outer(m3, m3))
        v, _s, _r = hessian_mod.rigid_body_vectors(ab.get_masses(), ab.get_positions())
        pmat = np.eye(len(m3)) - v @ v.T
        kmat = pmat @ hm @ pmat
        lam, vec = np.linalg.eigh(0.5 * (kmat + kmat.T))
        keep = np.linalg.norm(v.T @ vec, axis=0) ** 2 <= 0.5
        i0 = np.where(keep)[0][np.argmin(lam[keep])]
        target = -(target_cm / hessian_mod.CM_INV_PER_SQRT_EV_A2_AMU) ** 2
        hm2 = hm + (target - lam[i0]) * np.outer(vec[:, i0], vec[:, i0])
        np.save(root / "mace" / "basin{:02d}".format(b) / "hessian.npy", hm2 * np.sqrt(np.outer(m3, m3)))


def main():
    with tempfile.TemporaryDirectory(prefix="thermo_msrrho_") as tmp:
        mol = Path(tmp) / "dsgdb9nsd_000035"
        shutil.copytree(SRC, mol)
        out = me.run_calculation(mol, level=LEVEL, qm9_index="dsgdb9nsd_000035")
        doc = prop.load(layout.level_file(mol, LEVEL, "thermo_msrrho.toml"))

        # ---------------------------------------------------------------- records
        check("thermo_msrrho.toml starts with [Calculation_Status] NORMAL TERMINATION",
              next(iter(doc)) == "Calculation_Status"
              and doc["Calculation_Status"]["STATUS"] == "NORMAL TERMINATION")
        check("the Report ends with the terminal line",
              report.terminated_normally(layout.level_file(mol, LEVEL, "thermo_msrrho.out"), "thermo_msrrho"))
        check("only the spec's blocks are in the file (no [Imaginary_Spread])",
              set(doc) == {"Calculation_Status", "Calculation_Info", "Basin", "Ensemble", "Result"}, set(doc))
        check("nothing was written under _records/ by this Calculation",
              set(p.name for p in (mol / "_records").iterdir()) == {"branchA.toml"})
        check("degeneracy.toml was produced beside it (run when absent)",
              (layout.level_file(mol, LEVEL, "degeneracy.toml")).is_file())
        info = doc["Calculation_Info"]
        check("conventions are named: preset crest, tau 25, invert_below, fscal 1.0",
              info["PRESET"] == "crest" and info["TAU"] == 25.0
              and info["ITHR_POLICY"] == "invert_below" and info["FSCAL"] == 1.0)

        # ---------------------------------------------------------------- physics
        rows = doc["Basin"]
        res = doc["Result"]
        ens = doc["Ensemble"]
        check("three basins, none excluded, populations sum to 1",
              len(rows) == 3 and res["N_EXCLUDED"] == 0
              and abs(sum(r["POPULATION"] for r in rows) - 1.0) < 1e-12)
        check("g' = (1, 1, 1): the gauche pair is two basins, counted once",
              [r["G_PRIME"] for r in rows] == [1, 1, 1])
        check("reference basin is the lowest G_i",
              info["REFERENCE_BASIN"] == int(np.argmin([r["G_I"] for r in rows])))
        s_abs = res["S_ABS"]
        check("S_abs is an absolute entropy of the right size (60-80 cal/mol/K): %.3f" % s_abs,
              60.0 < s_abs < 80.0)
        check("S_abs = S_ref + S'_conf + dS_bar to 1e-9",
              abs(s_abs - (out["S_ref_cal_per_K"] + ens["S_CONF_PRIME"] + ens["DS_BAR"])) < 1e-9)
        # G_total from the partition function vs the Gibbs-Shannon route:
        #   G_total = G_ref + <H>_conf - T S'_conf ... written out by the module as a check
        check("G_total (partition function) equals the Gibbs-Shannon route to 1e-9 kcal/mol",
              abs(out["G_total_kcal"] - out["G_total_gibbs_shannon_kcal"]) < 1e-9,
              (out["G_total_kcal"], out["G_total_gibbs_shannon_kcal"]))
        check("N_BASINS_90 lists the basins that carry 90 % of the population",
              isinstance(res["N_BASINS_90"], list) and 1 <= len(res["N_BASINS_90"]) <= 3)
        check("experimental value 72.75 (li2016lbh) is in [Result] with the difference",
              res.get("S_EXPERIMENT") == 72.75 and res.get("S_EXPERIMENT_SOURCE") == "li2016lbh"
              and abs(res["S_ABS_MINUS_EXPERIMENT"] - (s_abs - 72.75)) < 1e-9)
        print("      propanal at MACE: S_abs = %.3f cal/mol/K, experiment 72.75, diff %+.3f"
              % (s_abs, s_abs - 72.75))

        # ---------------------------------------------------------------- one basin
        one = Path(tmp) / "one"
        shutil.copytree(SRC, one)
        shutil.rmtree(one / "mace" / "basin01")
        shutil.rmtree(one / "mace" / "basin02")
        text = (one / "_records" / "branchA.toml").read_text()
        head, _sep, _rest = text.partition("[[Basin]]\nINDEX        = 1")
        (one / "_records" / "branchA.toml").write_text(head)
        o1 = me.run_calculation(one, level=LEVEL, qm9_index="dsgdb9nsd_000018")
        d1 = prop.load(layout.level_file(one, LEVEL, "thermo_msrrho.toml"))
        check("one-basin molecule: S'_conf = 0 and dS_bar = 0 exactly",
              d1["Ensemble"]["S_CONF_PRIME"] == 0.0 and d1["Ensemble"]["DS_BAR"] == 0.0)
        check("one-basin molecule: S_abs equals that basin's S_msRRHO",
              abs(d1["Result"]["S_ABS"] - o1["S_ref_cal_per_K"]) < 1e-12)
        check("a species without a declared experimental value -> no S_EXPERIMENT key",
              "S_EXPERIMENT" not in d1["Result"])

        # ---------------------------------------------------------------- invariance
        basins = out["basins"]
        # the algebraic guard: the same basin twice with g' = 1 must equal it once with g' = 2
        twin = copy.deepcopy(basins[2]); twin["index"] = 9; twin["g_prime"] = 1
        two = [copy.deepcopy(basins[0]), copy.deepcopy(basins[2]), twin]
        two[1]["g_prime"] = 1
        pair_as_two = me.assemble(two, temperature_K=298.15)
        merged = [copy.deepcopy(basins[0]), copy.deepcopy(basins[2])]
        merged[1]["g_prime"] = 2
        pair_as_one = me.assemble(merged, temperature_K=298.15)
        # and the real pair (basins 1 and 2 are numerically distinct mirror images): how far
        # the two MACE Hessians of one enantiomer pair disagree, as a number
        real_two = me.assemble(copy.deepcopy(basins), temperature_K=298.15)
        print("      real mirror pair vs one basin g'=2: dS'_conf = %.2e cal/mol/K, dG_total = %.2e kcal/mol"
              % (real_two["S_conf_prime_cal_per_K"] - pair_as_one["S_conf_prime_cal_per_K"],
                 real_two["G_total_kcal"] - pair_as_one["G_total_kcal"]))
        check("the real mirror pair agrees with g'=2 to 1e-4 kcal/mol in G_total",
              abs(real_two["G_total_kcal"] - pair_as_one["G_total_kcal"]) < 1e-4)
        check("enantiomer pair as two basins (g'=1) == one basin (g'=2): S'_conf",
              abs(pair_as_two["S_conf_prime_cal_per_K"] - pair_as_one["S_conf_prime_cal_per_K"]) < 1e-9)
        check("... and S_abs, G_total agree to 1e-9",
              abs(pair_as_two["S_abs_cal_per_K"] - pair_as_one["S_abs_cal_per_K"]) < 1e-9
              and abs(pair_as_two["G_total_kcal"] - pair_as_one["G_total_kcal"]) < 1e-9)
        no_g = [copy.deepcopy(basins[0]), copy.deepcopy(basins[2])]
        wrong = me.assemble(no_g, temperature_K=298.15)
        check("dropping the partner without g'=2 changes S'_conf (the R ln 2 is real)",
              abs(wrong["S_conf_prime_cal_per_K"] - pair_as_one["S_conf_prime_cal_per_K"]) > 0.1)

        # ---------------------------------------------------------------- the floor
        win = copy.deepcopy(basins)
        win[1]["frequencies_cm"] = [-35.74] + list(win[1]["frequencies_cm"][1:])
        win[1]["n_imaginary"] = 1
        win[1] = me.basin_thermochemistry_from_frequencies(win[1], temperature_K=298.15)
        asm = me.assemble(win, temperature_K=298.15)
        check("a window mode is INVERTED, not excluded: the basin stays in the ensemble",
              asm["n_excluded"] == 0 and win[1]["excluded"] is False
              and win[1]["n_inverted"] == 1 and win[1]["inverted_frequencies_cm"] == [-35.74])
        bad = copy.deepcopy(basins)
        bad[1]["frequencies_cm"] = [-61.68] + list(bad[1]["frequencies_cm"][1:])
        bad[1]["n_imaginary"] = 1
        bad[1] = me.basin_thermochemistry_from_frequencies(bad[1], temperature_K=298.15)
        asm2 = me.assemble(bad, temperature_K=298.15)
        check("a sub-floor mode EXCLUDES the basin, counted, with its frequency in the reason",
              asm2["n_excluded"] == 1 and bad[1]["excluded"] is True
              and "-61.68" in bad[1]["excluded_reason"])
        sub = copy.deepcopy(basins)
        sub[1]["frequencies_cm"] = [-0.5] + list(sub[1]["frequencies_cm"][1:])
        sub[1] = me.basin_thermochemistry_from_frequencies(sub[1], temperature_K=298.15)
        check("a sub-1 cm^-1 mode is dropped ORCA-style and recorded, the basin kept",
              sub[1]["excluded"] is False and sub[1]["n_below_floor"] == 1
              and sub[1]["dropped_frequencies_cm"] == [-0.5])
        check("every [[Basin]] row carries N_IMAGINARY / N_INVERTED / N_KEPT_NEGATIVE / "
              "N_BELOW_FLOOR = 0 and SOFT_SADDLE = false on a clean molecule",
              all(r["N_IMAGINARY"] == 0 and r["N_INVERTED"] == 0 and r["N_KEPT_NEGATIVE"] == 0
                  and r["N_BELOW_FLOOR"] == 0 and r["SOFT_SADDLE"] is False for r in doc["Basin"]))
        check("... and no clean row carries INVERTED_CM / DROPPED_CM (empty lists are absent)",
              all("INVERTED_CM" not in r and "DROPPED_CM" not in r for r in doc["Basin"]))
        # a record written under the removed policy name: data, not code
        rec_path = layout.level_file(mol, LEVEL, "thermo_msrrho.toml")
        text = rec_path.read_text(encoding="utf-8")
        rec_path.write_text(text.replace('"invert_below"', '"refuse"'), encoding="utf-8")
        check("a historical record naming 'refuse' still loads (its recomputation is out of scope)",
              prop.load(rec_path)["Calculation_Info"]["ITHR_POLICY"] == "refuse")
        rec_path.write_text(text, encoding="utf-8")

        # ---------------------------------------------------------------- crest_native, the seam's policy
        deep = copy.deepcopy(basins)
        deep[1]["frequencies_cm"] = [-61.68] + list(deep[1]["frequencies_cm"][1:])
        deep[1]["n_imaginary"] = 1
        deep[1] = me.basin_thermochemistry_from_frequencies(deep[1], temperature_K=298.15,
                                                            imaginary_policy="crest_native")
        check("crest_native: a -61.68 basin is kept negative (1), not inverted, not excluded",
              deep[1]["excluded"] is False and deep[1]["n_kept_negative"] == 1
              and deep[1]["n_inverted"] == 0)
        # grimme2012 defines no ithr. It runs on a clean molecule (the old upfront guard
        # refused the whole calculation for a condition that did not apply), and an
        # imaginary basin is excluded saying exactly what is missing.
        gmol = Path(tmp) / "grimme2012"
        shutil.copytree(SRC, gmol)
        o_g = me.run_calculation(gmol, level=LEVEL, qm9_index="dsgdb9nsd_000035", preset="grimme2012")
        check("grimme2012 on a clean molecule runs and its record names the preset",
              o_g["info"]["PRESET"] == "grimme2012" and o_g["n_excluded"] == 0)
        g = me.basin_thermochemistry_from_frequencies(
            dict(win[1], frequencies_cm=[-30.0] + list(basins[1]["frequencies_cm"][1:])),
            temperature_K=298.15, preset="grimme2012")
        check("grimme2012 with a window mode: the basin is excluded, naming the missing ithr",
              g["excluded"] is True and "no ithr" in g["excluded_reason"])
        # every basin pushed BELOW the floor: no basin survives the production policy
        native = Path(tmp) / "native"
        shutil.copytree(SRC, native)
        _push_lowest_mode(native, 61.68)
        try:
            me.run_calculation(native, level=LEVEL, qm9_index="dsgdb9nsd_000035")
            check("a molecule whose every basin is below the floor is refused, and says so", False)
        except ValueError as exc:
            check("a molecule whose every basin is below the floor is refused, and says so",
                  "no basin survived" in str(exc))
        # the same surgery into the WINDOW instead: the production policy admits every
        # basin and records the mode it inverted
        window = Path(tmp) / "window"
        shutil.copytree(SRC, window)
        _push_lowest_mode(window, 35.74)
        o_win = me.run_calculation(window, level=LEVEL, qm9_index="dsgdb9nsd_000035")
        d_win = prop.load(layout.level_file(window, LEVEL, "thermo_msrrho.toml"))
        check("a molecule whose every basin carries one window mode finishes: nothing excluded, N_INVERTED = 1 each",
              d_win["Result"]["N_EXCLUDED"] == 0
              and all(r["N_INVERTED"] == 1 for r in d_win["Basin"])
              and d_win["Calculation_Info"]["ITHR_POLICY"] == "invert_below")
        check("... and INVERTED_CM stores the mode that was inverted (-35.74 cm^-1)",
              all(len(r["INVERTED_CM"]) == 1 and abs(r["INVERTED_CM"][0] + 35.74) < 0.5
                  for r in d_win["Basin"]), [r.get("INVERTED_CM") for r in d_win["Basin"]])
        # the crest_native record on the sub-floor molecule is still written, and the
        # preset spread says grimme2012 (no ithr) is absent instead of failing the record
        o_nat = me.run_calculation(native, level=LEVEL, qm9_index="dsgdb9nsd_000035",
                                   imaginary_policy="crest_native")
        d_nat = prop.load(layout.level_file(native, LEVEL, "thermo_msrrho.toml"))
        ref_row = next(r for r in d_nat["Basin"] if r["INDEX"] == d_nat["Calculation_Info"]["REFERENCE_BASIN"])
        check("crest_native with a sub-ithr mode on every basin: the record is written, the reference basin kept",
              d_nat["Calculation_Info"]["ITHR_POLICY"] == "crest_native" and ref_row["N_KEPT_NEGATIVE"] == 1
              and d_nat["Result"]["N_EXCLUDED"] == 0)
        check("... and the preset spread names grimme2012 (no ithr) as absent instead of failing the record",
              set(o_nat["preset_spread"]["presets_absent"]) == {"grimme2012"}
              and set(o_nat["preset_spread"]["TS_vib_kcal"]) == {"crest", "xtb", "HO"})

        # ---------------------------------------------------------------- config guard
        cfg = copy.deepcopy(config.load())
        cfg["species"]["dsgdb9nsd_000035"]["experimental_entropy_source"] = "no_such_key"
        try:
            config.experimental_entropy("dsgdb9nsd_000035", cfg)
            check("an experimental value with an unknown citation key is refused", False)
        except KeyError as exc:
            check("an experimental value with an unknown citation key is refused", "no_such_key" in str(exc))
        del cfg["species"]["dsgdb9nsd_000035"]["experimental_entropy_source"]
        try:
            config.experimental_entropy("dsgdb9nsd_000035", cfg)
            check("an experimental value without a source is refused", False)
        except KeyError:
            check("an experimental value without a source is refused", True)

    if FAIL:
        print("FAIL: " + ", ".join(FAIL))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
