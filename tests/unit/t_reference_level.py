"""The reference level (wb97m-d3bj_def2-tzvppd) and level_compare.

UNIT. No engine: tests/data/propanal_molecule carries the ORCA 6.0.1 engine files of the
propanal dry run (`! wB97M-D3BJ def2-TZVPPD TightOpt Freq TightSCF`, 8 cores, basin 0 in
226 s with the ANALYTIC Hessian, 2026-09-16) under orca/wb97m-d3bj_def2-tzvppd/basinNN/,
so `optimise_and_hessian` finds finished jobs and runs nothing.

Asserted: the .hess passes the frequency round-trip; the merge map lists every requested
MACE basin exactly once with its status and displacement; the Hessian route is recorded
(analytic here); the level's thermo_msrrho record lands in msrrho/thermo/<level>.* with the
thermo_msrrho blocks; level_compare states an absent level as PRESENT = false and computes the
tiers only where both sides exist; on the basins the fixture holds, S_abs at the reference
level is compared with the declared experiment.

Further: `relaxation_verdict` on the shared floor classifier (minimum / below-floor
saddle / window soft saddle, the floor line itself in the window); `relax_with_retry`'s
branches with a fake `run` (one job on a minimum and on a below-floor saddle; one retry
from the ORCA-relaxed geometry with `rerun=True` into a minimum or another saddle; a
stay-saddle retry reported `soft_saddle = true` with the final lowest; no second retry of
a job already on disk); the merge map's `soft_saddle` / `lowest_frequency_cm` columns and
the record's SOFT_SADDLE key (false on the clean fixture). The wired ORCA path is
`tests/integration/t_reference_level_retry.py` (a fake binary, the propanal fixture).
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from openqha.thermochem import msrrho_ensemble as me         # noqa: E402
from openqha.thermochem import reference_level as rl         # noqa: E402
from openqha.thermochem import thermo                        # noqa: E402
from openqha.store import dat, layout, property as prop, report   # noqa: E402

SRC = ROOT / "tests" / "data" / "propanal_molecule"
LEVEL = "wb97m-d3bj_def2-tzvppd"
ITHIR = thermo.MSRRHO_PRESETS["crest"]["ithr_cm"]
FAIL = []


def check(label, ok, detail=""):
    print("  {:74s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


def _relax(records):
    """A `run(positions, rerun)` for `relax_with_retry`: returns `records` in order."""
    calls = []

    def run(positions, rerun):
        rec = records[len(calls)]
        calls.append((np.asarray(positions, dtype=float), bool(rerun), rec))
        return (dict(seconds=rec["seconds"], positions_A=rec["positions_A"]),
                dict(lowest_cm_inv=rec["lowest"]))

    return run, calls


def _relax_case(records, start=((1.0, 2.0, 3.0),)):
    run, calls = _relax(records)
    return rl.relax_with_retry(run, np.asarray(start, dtype=float), ITHIR), calls


def main():
    available = sorted(int(p.name[len("orca." + LEVEL + ".basin"):-len(".hess")])
                       for p in layout.msrrho_dir(SRC).glob("orca.{}.basin*.hess".format(LEVEL)))
    print("  fixture holds reference-level jobs for basins", available)
    with tempfile.TemporaryDirectory(prefix="reference_level_") as tmp:
        mol = Path(tmp) / "dsgdb9nsd_000035"
        shutil.copytree(SRC, mol)

        # level_compare before anything: the reference level is absent, stated as such
        cmp0 = rl.level_compare(mol)
        by = {r["LEVEL"]: r for r in cmp0["levels"]}
        check("level_compare with no reference record: PRESENT = false, no model-error tier",
              by[LEVEL]["PRESENT"] is False and "MODEL_ERROR_S" not in cmp0["tiers"])

        out = rl.run_calculation(mol, level=LEVEL, basins=available)
        check("ORCA was not run (finished engine files reused); routes recorded: %s" % out["hessian_routes"],
              out["hessian_routes"] == ["analytic"])
        rows = dat.read_table(layout.level_file(mol, LEVEL, "merge_map.dat"))
        check("merge map lists every requested MACE basin exactly once",
              sorted(int(r["mace_basin"]) for r in rows) == available)
        check("every row has a status in {kept, merged, saddle} and a displacement RMSD",
              all(r["status"] in ("kept", "merged", "saddle") and r["rmsd_displacement_A"] >= 0 for r in rows))
        check("merge map rows carry soft_saddle (false here) and the lowest frequency",
              all(r["soft_saddle"] is False and r["lowest_frequency_cm"] > 0 for r in rows),
              [(r["mace_basin"], r["soft_saddle"], r["lowest_frequency_cm"]) for r in rows])
        check("the frequency round-trip on every .hess is below 0.5 cm^-1",
              all(r["roundtrip_cm"] < 0.5 for r in rows), [r["roundtrip_cm"] for r in rows])
        doc = prop.load(layout.level_file(mol, LEVEL, "thermo_msrrho.toml"))
        check("msrrho/thermo/<level>.thermo_msrrho.toml has the thermo_msrrho blocks (no [Imaginary_Spread]) and LEVEL = " + LEVEL,
              set(doc) == {"Calculation_Status", "Calculation_Info", "Basin", "Ensemble", "Result"}
              and doc["Calculation_Info"]["LEVEL"] == LEVEL)
        check("[Calculation_Info].ITHR_POLICY = invert_below at the reference level",
              doc["Calculation_Info"]["ITHR_POLICY"] == "invert_below")
        check("every [[Basin]] row carries SOFT_SADDLE = false on a clean molecule",
              all(r["SOFT_SADDLE"] is False for r in doc["Basin"]))
        check("the Report ends with the terminal line",
              report.terminated_normally(layout.level_file(mol, LEVEL, "thermo_msrrho.out"), "thermo_msrrho"))
        res = doc["Result"]
        print("      reference level on basins %s: S_abs = %.3f cal/mol/K (experiment 72.75, diff %+.3f)"
              % (available, res["S_ABS"], res["S_ABS"] - 72.75))
        check("S_abs is an absolute entropy (60-80 cal/mol/K) and the experiment is attached",
              60 < res["S_ABS"] < 80 and res.get("S_EXPERIMENT") == 72.75)
        e_el = [r["E_EL"] for r in doc["Basin"]]
        check("E_el at the reference level is read from the .out (not the .hess placeholder 0)",
              all(abs(e) > 1e5 for e in e_el), e_el)
        if len(available) == 3:
            check("all three basins: the mirror pair stays two reference basins (dedup by proper rotation)",
                  res["N_BASINS"] >= 2)
            check("all three basins: S_abs within 1.0 cal/mol/K of LBH (72.75)",
                  abs(res["S_ABS"] - 72.75) < 1.0, res["S_ABS"])

        # the MACE level beside it, then the comparison
        me.run_calculation(mol, level="mace-off23_medium")
        cmp1 = rl.level_compare(mol)
        by = {r["LEVEL"]: r for r in cmp1["levels"]}
        check("level_compare: both levels present", by[LEVEL]["PRESENT"] and by["mace-off23_medium"]["PRESENT"])
        t = cmp1["tiers"]
        check("three tiers computed with the experiment attached",
              all(k in t for k in ("MODEL_ERROR_S", "LEVEL_ERROR_S", "TOTAL_ERROR_S", "S_EXPERIMENT")))
        check("tiers are consistent: total = model + level (to 1e-9)",
              abs(t["TOTAL_ERROR_S"] - (t["MODEL_ERROR_S"] + t["LEVEL_ERROR_S"])) < 1e-9)
        print("      tiers: model %+.3f  level %+.3f  total %+.3f cal/mol/K"
              % (t["MODEL_ERROR_S"], t["LEVEL_ERROR_S"], t["TOTAL_ERROR_S"]))
        # every ensemble term per level, and a MODEL_ERROR_ line for each
        for need in ("S_REF", "S_CONF_PRIME", "DS_BAR", "H_CONF", "CP_CONF", "G_TOTAL", "N_BASINS", "N_BASINS_90"):
            check("[[Level]] carries %s for every present level" % need,
                  all(need in r for r in cmp1["levels"] if r["PRESENT"]))
        check("[Tiers] carries MODEL_ERROR_ for S_REF, S_CONF_PRIME, DS_BAR, H_CONF, CP_CONF, N_BASINS_90",
              all("MODEL_ERROR_" + k in t for k in ("S_REF", "S_CONF_PRIME", "DS_BAR", "H_CONF", "CP_CONF", "N_BASINS_90")))
        check("MODEL_ERROR_S = MODEL_ERROR_S_REF + MODEL_ERROR_S_CONF_PRIME + MODEL_ERROR_DS_BAR (to 1e-9)",
              abs(t["MODEL_ERROR_S"] - (t["MODEL_ERROR_S_REF"] + t["MODEL_ERROR_S_CONF_PRIME"] + t["MODEL_ERROR_DS_BAR"])) < 1e-9)
        print("      model error split: S_ref %+.3f  S'_conf %+.3f  dS_bar %+.3f cal/mol/K"
              % (t["MODEL_ERROR_S_REF"], t["MODEL_ERROR_S_CONF_PRIME"], t["MODEL_ERROR_DS_BAR"]))
        rtext = (layout.thermo_file(mol, "level_compare.out")).read_text(encoding="utf-8")
        check("the Report prints the per-basin table (each level at its own geometry)",
              "per basin, each level at its own geometry" in rtext)
        cdoc = prop.load(layout.thermo_file(mol, "level_compare.toml"))
        check("level_compare.toml starts with [Calculation_Status]; report ends with its terminal line",
              next(iter(cdoc)) == "Calculation_Status"
              and report.terminated_normally(layout.thermo_file(mol, "level_compare.out"), "level_compare"))

        # ---- the soft-saddle retry ---------------------------------------
        print("      the retry is decided on the shared floor classifier")
        for lowest, want in ((131.14, "minimum"), (0.0, "minimum"), (-0.5, "soft_saddle"),
                             (-6.84, "soft_saddle"), (ITHIR, "soft_saddle"),
                             (ITHIR - 1e-9, "saddle"), (-195.79, "saddle")):
            check("relaxation_verdict(%+.2f, ithr) = %s" % (lowest, want),
                  rl.relaxation_verdict(lowest, ITHIR) == want, rl.relaxation_verdict(lowest, ITHIR))
        check("a preset with no floor has no window: a negative mode is a saddle, never a retry",
              rl.relaxation_verdict(-6.84, None) == "saddle"
              and rl.relaxation_verdict(12.0, None) == "minimum")

        clean = [dict(lowest=131.14, seconds=100.0, positions_A=[[1.0, 2.0, 3.0]])]
        step, calls = _relax_case(clean)
        check("a minimum: one job, no retry, verdict minimum, no soft flag",
              len(calls) == 1 and calls[0][1] is False and step["verdict"] == "minimum"
              and step["retried"] is False and step["soft_saddle"] is False)
        below = [dict(lowest=-195.79, seconds=100.0, positions_A=[[1.0, 0.0, 0.0]])]
        step, calls = _relax_case(below)
        check("a saddle below the floor: excluded as today, no retry, no soft flag",
              len(calls) == 1 and step["verdict"] == "saddle" and step["retried"] is False
              and step["soft_saddle"] is False)
        to_min = [dict(lowest=-6.84, seconds=100.0, positions_A=[[1.0, 2.0, 3.0]]),
                  dict(lowest=128.5, seconds=105.0, positions_A=[[4.0, 5.0, 6.0]])]
        step, calls = _relax_case(to_min)
        check("a window saddle: the retry starts from the ORCA-relaxed geometry with rerun=True",
              len(calls) == 2 and calls[1][1] is True
              and np.allclose(calls[1][0], [[1.0, 2.0, 3.0]], atol=1e-12), calls[1][0].tolist())
        check("... and a retry into a minimum enters normally (no soft flag)",
              step["verdict"] == "minimum" and step["retried"] is True
              and step["soft_saddle"] is False and step["record"]["seconds"] == 105.0)
        still = [dict(lowest=-6.84, seconds=100.0, positions_A=[[1.0, 2.0, 3.0]]),
                 dict(lowest=-5.2, seconds=105.0, positions_A=[[1.0, 2.0, 3.0]])]
        step, calls = _relax_case(still)
        check("a retry that stays a soft saddle: soft_saddle = true and the final lowest",
              len(calls) == 2 and step["verdict"] == "soft_saddle" and step["soft_saddle"] is True
              and step["record"]["seconds"] == 105.0
              and step["roundtrip"]["lowest_cm_inv"] == -5.2)
        drops = [dict(lowest=-6.84, seconds=100.0, positions_A=[[1.0, 2.0, 3.0]]),
                 dict(lowest=-61.68, seconds=105.0, positions_A=[[1.0, 2.0, 3.0]])]
        step, calls = _relax_case(drops)
        check("a retry that ends below the floor is a plain saddle (soft false), still retried",
              len(calls) == 2 and step["verdict"] == "saddle" and step["retried"] is True
              and step["soft_saddle"] is False)
        reused = [dict(lowest=-6.84, seconds=None, positions_A=[[1.0, 2.0, 3.0]])]
        step, calls = _relax_case(reused)
        check("a job already on disk (seconds None, a Batch resume) is not retried again",
              len(calls) == 1 and step["retried"] is False and step["soft_saddle"] is True)
    if FAIL:
        print("FAIL: " + ", ".join(FAIL))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
