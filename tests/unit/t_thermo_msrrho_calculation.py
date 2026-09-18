"""Ticket 24: the thermo_msrrho Calculation at MACE, its records, the experimental comparison.

UNIT. NumPy + RDKit, no engine; a few seconds. Runs on the real branch A product of
propanal (tests/data/propanal_molecule: branchA.toml, three basins with their MACE
Hessians, CREST's rotamer file), so the numbers are the chain's own.

Seams (spec, Testing Decisions): the Property file the Calculation leaves; the three
algebraic guards (one-basin molecule -> zero ensemble terms; G_total from the partition
function equals the Gibbs-Shannon route; an enantiomer pair as two basins with g' = 1
equals one basin with g' = 2); the refuse policy; the configuration refusing an
experimental value without a citation.
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


def main():
    with tempfile.TemporaryDirectory(prefix="thermo_msrrho_") as tmp:
        mol = Path(tmp) / "dsgdb9nsd_000035"
        shutil.copytree(SRC, mol)
        out = me.run_calculation(mol, level=LEVEL, qm9_index="dsgdb9nsd_000035")
        lvl = layout.level_dir(mol, LEVEL)
        doc = prop.load(lvl / "thermo_msrrho.toml")

        # ---------------------------------------------------------------- records
        check("thermo_msrrho.toml starts with [Calculation_Status] NORMAL TERMINATION",
              next(iter(doc)) == "Calculation_Status"
              and doc["Calculation_Status"]["STATUS"] == "NORMAL TERMINATION")
        check("the Report ends with the terminal line",
              report.terminated_normally(lvl / "thermo_msrrho.out", "thermo_msrrho"))
        check("only the spec's blocks are in the file",
              set(doc) == {"Calculation_Status", "Calculation_Info", "Basin", "Ensemble", "Result", "Imaginary_Spread"}, set(doc))
        check("nothing was written under _records/ by this Calculation",
              set(p.name for p in (mol / "_records").iterdir()) == {"branchA.toml"})
        check("degeneracy.toml was produced beside it (run when absent)",
              (lvl / "degeneracy.toml").is_file())
        info = doc["Calculation_Info"]
        check("conventions are named: preset crest, tau 25, refuse, fscal 1.0",
              info["PRESET"] == "crest" and info["TAU"] == 25.0
              and info["ITHR_POLICY"] == "refuse" and info["FSCAL"] == 1.0)

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
        d1 = prop.load(layout.level_dir(one, LEVEL) / "thermo_msrrho.toml")
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

        # ---------------------------------------------------------------- refuse
        bad = copy.deepcopy(basins)
        bad[1]["frequencies_cm"] = [-35.74] + list(bad[1]["frequencies_cm"][1:])
        bad[1] = me.basin_thermochemistry_from_frequencies(bad[1], temperature_K=298.15)
        asm = me.assemble(bad, temperature_K=298.15)
        check("a basin with an imaginary mode is EXCLUDED and counted",
              asm["n_excluded"] == 1 and bad[1]["excluded"] is True
              and "-35.74" in bad[1]["excluded_reason"])

        # ---------------------------------------------------------------- ticket 28
        sp = {r["POLICY"]: r for r in doc["Imaginary_Spread"]}
        check("[Imaginary_Spread] has one row per policy and names the [Result] policy",
              set(sp) == {"refuse", "invert_below", "crest_native"}
              and doc["Calculation_Info"]["ITHR_POLICY"] == "refuse")
        check("0 imaginary modes at the MACE level: the three policies give one S_ABS to 1e-9",
              max(abs(sp[k]["S_ABS"] - sp["refuse"]["S_ABS"]) for k in sp) < 1e-9
              and all(sp[k]["N_EXCLUDED"] == 0 and sp[k]["N_KEPT_NEGATIVE"] == 0 for k in sp))
        check("every [[Basin]] row carries N_IMAGINARY / N_INVERTED / N_KEPT_NEGATIVE = 0",
              all(r["N_IMAGINARY"] == 0 and r["N_INVERTED"] == 0 and r["N_KEPT_NEGATIVE"] == 0
                  for r in doc["Basin"]))
        # a synthetic sub-ithr basin: refuse and invert_below exclude it, crest_native keeps it
        deep = copy.deepcopy(basins)
        deep[1]["frequencies_cm"] = [-61.68] + list(deep[1]["frequencies_cm"][1:])
        deep[1]["n_imaginary"] = 1
        rows = {r["POLICY"]: r for r in me.imaginary_spread(deep, temperature_K=298.15)}
        check("a -61.68 basin: excluded under refuse and invert_below, kept negative under crest_native",
              rows["refuse"]["N_EXCLUDED"] == 1 and rows["invert_below"]["N_EXCLUDED"] == 1
              and rows["crest_native"]["N_EXCLUDED"] == 0 and rows["crest_native"]["N_KEPT_NEGATIVE"] == 1)
        check("crest_native S_ABS differs from refuse (the kept basin enters with S = 0 for that mode)",
              abs(rows["crest_native"]["S_ABS"] - rows["refuse"]["S_ABS"]) > 1e-3)
        shallow = copy.deepcopy(basins)
        shallow[1]["frequencies_cm"] = [-35.74] + list(shallow[1]["frequencies_cm"][1:])
        shallow[1]["n_imaginary"] = 1
        rows2 = {r["POLICY"]: r for r in me.imaginary_spread(shallow, temperature_K=298.15)}
        check("a -35.74 basin: invert_below and crest_native agree to 1e-9, refuse excludes it",
              rows2["refuse"]["N_EXCLUDED"] == 1 and rows2["invert_below"]["N_INVERTED"] == 1
              and abs(rows2["invert_below"]["S_ABS"] - rows2["crest_native"]["S_ABS"]) < 1e-9)
        # under grimme2012 (no ithr) the two non-refuse policies cannot be applied to an
        # imaginary spectrum: the rows say AVAILABLE = false instead of repeating refuse
        rows3 = {r["POLICY"]: r for r in me.imaginary_spread(shallow, temperature_K=298.15, preset="grimme2012")}
        check("grimme2012 with an imaginary basin: invert_below / crest_native rows are AVAILABLE = false",
              rows3["refuse"]["AVAILABLE"] is True and rows3["invert_below"]["AVAILABLE"] is False
              and rows3["crest_native"]["AVAILABLE"] is False and "S_ABS" not in rows3["crest_native"])
        rows4 = {r["POLICY"]: r for r in me.imaginary_spread(basins, temperature_K=298.15, preset="grimme2012")}
        check("grimme2012 without an imaginary basin: all three rows available and equal",
              all(rows4[k]["AVAILABLE"] for k in rows4)
              and max(abs(rows4[k]["S_ABS"] - rows4["refuse"]["S_ABS"]) for k in rows4) < 1e-9)
        try:
            me.run_calculation(mol, level=LEVEL, qm9_index="dsgdb9nsd_000035", preset="grimme2012",
                               imaginary_policy="crest_native")
            check("run_calculation refuses a non-refuse policy with a preset that has no ithr", False)
        except ValueError as exc:
            check("run_calculation refuses a non-refuse policy with a preset that has no ithr", "no ithr" in str(exc))
        # the record is written under a non-refuse policy even when the reference basin
        # (lowest G_i) carries a sub-ithr mode (code-review finding: preset_spread)
        native = Path(tmp) / "native"
        shutil.copytree(SRC, native)
        # push the lowest mode of EVERY basin to -61.68 cm^-1 by rank-one surgery on the
        # mass-weighted projected matrix (so whichever basin ends up the reference basin
        # carries a sub-ithr mode), then un-weight it back into hessian.npy
        for b in (0, 1, 2):
            hb = np.load(native / "mace" / "basin{:02d}".format(b) / "hessian.npy")
            ab = read(str(native / "mace" / "basin{:02d}".format(b) / "basin.extxyz"), format="extxyz")
            m3 = np.repeat(ab.get_masses(), 3)
            hm = hb / np.sqrt(np.outer(m3, m3))
            v, _s, _r = hessian_mod.rigid_body_vectors(ab.get_masses(), ab.get_positions())
            pmat = np.eye(len(m3)) - v @ v.T
            kmat = pmat @ hm @ pmat
            lam, vec = np.linalg.eigh(0.5 * (kmat + kmat.T))
            keep = np.linalg.norm(v.T @ vec, axis=0) ** 2 <= 0.5
            i0 = np.where(keep)[0][np.argmin(lam[keep])]
            target = -(61.68 / hessian_mod.CM_INV_PER_SQRT_EV_A2_AMU) ** 2
            hm2 = hm + (target - lam[i0]) * np.outer(vec[:, i0], vec[:, i0])
            np.save(native / "mace" / "basin{:02d}".format(b) / "hessian.npy", hm2 * np.sqrt(np.outer(m3, m3)))
        o_nat = me.run_calculation(native, level=LEVEL, qm9_index="dsgdb9nsd_000035",
                                   imaginary_policy="crest_native")
        d_nat = prop.load(layout.level_dir(native, LEVEL) / "thermo_msrrho.toml")
        ref_row = next(r for r in d_nat["Basin"] if r["INDEX"] == d_nat["Calculation_Info"]["REFERENCE_BASIN"])
        check("crest_native with a sub-ithr mode on every basin: the record is written, the reference basin kept",
              d_nat["Calculation_Info"]["ITHR_POLICY"] == "crest_native" and ref_row["N_KEPT_NEGATIVE"] == 1
              and d_nat["Result"]["N_EXCLUDED"] == 0)
        check("... and the preset spread names grimme2012 (no ithr) as absent instead of failing the record",
              set(o_nat["preset_spread"]["presets_absent"]) == {"grimme2012"}
              and set(o_nat["preset_spread"]["TS_vib_kcal"]) == {"crest", "xtb", "HO"})
        try:
            me.run_calculation(native, level=LEVEL, qm9_index="dsgdb9nsd_000035", imaginary_policy="refuse")
            check("under refuse the same molecule has no basin left and says so", False)
        except ValueError as exc:
            check("under refuse the same molecule has no basin left and says so", "no basin survived" in str(exc))

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
