"""Ticket 26: the reference level (wb97m-d3bj_def2-tzvppd) and level_compare.

UNIT. No engine: tests/data/propanal_molecule carries the ORCA 6.0.1 engine files of the
propanal dry run (`! wB97M-D3BJ def2-TZVPPD TightOpt Freq TightSCF`, 8 cores, basin 0 in
226 s with the ANALYTIC Hessian, 2026-09-16) under orca/wb97m-d3bj_def2-tzvppd/basinNN/,
so `optimise_and_hessian` finds finished jobs and runs nothing.

Asserted: the .hess passes the frequency round-trip; the merge map lists every requested
MACE basin exactly once with its status and displacement; the Hessian route is recorded
(analytic here); the level's thermo_msrrho record lands in levels/<level>/ with the
ticket-24 blocks; level_compare states an absent level as PRESENT = false and computes the
tiers only where both sides exist; on the basins the fixture holds, S_abs at the reference
level is compared with the declared experiment.
"""
import shutil
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
from openqha.thermochem import msrrho_ensemble as me         # noqa: E402
from openqha.thermochem import reference_level as rl         # noqa: E402
from openqha.store import dat, layout, property as prop, report   # noqa: E402

SRC = ROOT / "tests" / "data" / "propanal_molecule"
LEVEL = "wb97m-d3bj_def2-tzvppd"
FAIL = []


def check(label, ok, detail=""):
    print("  {:74s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


def main():
    available = sorted(int(p.name[len("basin"):]) for p in (SRC / "orca" / LEVEL).iterdir()
                       if (p / "job.hess").is_file())
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
        lvl = layout.level_dir(mol, LEVEL)
        check("ORCA was not run (finished engine files reused); routes recorded: %s" % out["hessian_routes"],
              out["hessian_routes"] == ["analytic"])
        rows = dat.read_table(lvl / "merge_map.dat")
        check("merge map lists every requested MACE basin exactly once",
              sorted(int(r["mace_basin"]) for r in rows) == available)
        check("every row has a status in {kept, merged, saddle} and a displacement RMSD",
              all(r["status"] in ("kept", "merged", "saddle") and r["rmsd_displacement_A"] >= 0 for r in rows))
        check("the frequency round-trip on every .hess is below 0.5 cm^-1",
              all(r["roundtrip_cm"] < 0.5 for r in rows), [r["roundtrip_cm"] for r in rows])
        doc = prop.load(lvl / "thermo_msrrho.toml")
        check("levels/<level>/thermo_msrrho.toml has the ticket-24 blocks and LEVEL = " + LEVEL,
              set(doc) == {"Calculation_Status", "Calculation_Info", "Basin", "Ensemble", "Result"}
              and doc["Calculation_Info"]["LEVEL"] == LEVEL)
        check("the Report ends with the terminal line",
              report.terminated_normally(lvl / "thermo_msrrho.out", "thermo_msrrho"))
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
        cdoc = prop.load(mol / "levels" / "level_compare.toml")
        check("level_compare.toml starts with [Calculation_Status]; report ends with its terminal line",
              next(iter(cdoc)) == "Calculation_Status"
              and report.terminated_normally(mol / "levels" / "level_compare.out", "level_compare"))
    if FAIL:
        print("FAIL: " + ", ".join(FAIL))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
