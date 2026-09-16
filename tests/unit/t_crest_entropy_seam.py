"""Ticket 25: the GFN2 seam -- CREST --entropy's own inputs through our assembly.

UNIT. No engine: tests/data/propanal_crest_entropy holds two real `crest --entropy --gfn2`
runs on propanal (2026-09-16, 40 s each, CREST 3.x in s0crest) as engine files, and the
xtb --hess engine files at CREST's conformers and at the reference structure.

What is asserted (spec, Testing Decisions; the ticket's criteria):
  * S'_conf, H_conf, Cp_conf recomputed from CREST's conformer energies and CREST's own
    degeneracies equal CREST's printout to 1e-4 cal/mol/K -- both runs, even though the
    runs disagree with each other (g' = (2,1,2) vs (2,1,1): a point-group label flip).
  * our cre_degen2 equivalents equal CREST's cre_degen2 (g_rot * cores).
  * S_ref from xtb's analytic Hessian at CREST's reference geometry is within 0.05 of
    CREST's numerical one (measured +0.011).
  * two runs are recorded with their spread; a single run is refused.
  * the records live in levels/gfn2/ and nowhere else.
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
from openqha.thermochem import crest_entropy as ce          # noqa: E402
from openqha.store import layout, property as prop, report   # noqa: E402

SRC = ROOT / "tests" / "data" / "propanal_crest_entropy"
FAIL = []


def check(label, ok, detail=""):
    print("  {:72s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


def main():
    with tempfile.TemporaryDirectory(prefix="crest_entropy_seam_") as tmp:
        mol = Path(tmp) / "dsgdb9nsd_000035"
        shutil.copytree(SRC, mol)
        out = ce.run_calculation(mol, run_crest=False, runs=(1, 2))
        lvl = layout.level_dir(mol, "gfn2")
        doc = prop.load(lvl / "thermo_msrrho.toml")

        check("records in levels/gfn2/ only",
              (lvl / "thermo_msrrho.toml").is_file() and (lvl / "thermo_msrrho.out").is_file()
              and not (mol / "_records").exists())
        check("Property file starts with [Calculation_Status] and the Report ends with the terminal line",
              next(iter(doc)) == "Calculation_Status"
              and report.terminated_normally(lvl / "thermo_msrrho.out", "thermo_msrrho"))
        check("[Crest] has two runs with sthr 25, ithr -50, fscal 1",
              len(doc["Crest"]) == 2 and all(r["STHR"] == 25.0 and r["ITHR"] == -50.0 and r["FSCAL"] == 1.0
                                             for r in doc["Crest"]))
        for e in out["runs"]:
            s = e["seam"]
            check("run {}: S'_conf, H_conf, Cp_conf match CREST to 1e-4 (dS' {:+.1e}, dH {:+.1e}, dCp {:+.1e})"
                  .format(e["run"], s["S_CONF_DELTA"], s["H_CONF_DELTA"], s["CP_CONF_DELTA"]),
                  s["ALGEBRAIC_TERMS_MATCH"])
            check("run {}: our cre_degen2 equivalents equal CREST's {}".format(e["run"], s["CRE_DEGEN2"]),
                  s["CRE_DEGEN2_OURS"] == s["CRE_DEGEN2"])
            check("run {}: S_ref within 0.05 of CREST's ({:+.4f})".format(e["run"], s.get("S_REF_DELTA", 9)),
                  abs(s.get("S_REF_DELTA", 9)) < ce.TOL_S_REF_CAL)
            check("run {}: dS_bar delta is reported ({:+.4f}, Hessian implementation)".format(
                e["run"], s.get("DS_BAR_DELTA", float("nan"))), "DS_BAR_DELTA" in s)
        g1 = out["runs"][0]["seam"]["G_PRIME_CREST"]
        g2 = out["runs"][1]["seam"]["G_PRIME_CREST"]
        check("CREST's own g' differs between the runs ([2,1,2] vs [2,1,1]): the label flip is reproduced",
              g1 == [2, 1, 2] and g2 == [2, 1, 1])
        check("the S_conf spread over runs is the R ln 2 of that flip (0.43 cal/mol/K)",
              abs(out["S_conf_spread"] - 0.4296) < 0.01, out["S_conf_spread"])
        check("our own g' is the same in both runs (geometric chirality class)",
              out["runs"][0]["seam"]["G_PRIME_OURS"] == out["runs"][1]["seam"]["G_PRIME_OURS"])
        check("the third conformer is excluded by xtb --hess (imaginary mode), counted",
              all(e["seam"]["N_CONFORMERS_EXCLUDED"] == 1 for e in out["runs"]))
        try:
            ce.run_calculation(mol, run_crest=False, runs=(1,))
            check("a single run is refused", False)
        except ValueError as exc:
            check("a single run is refused", "two" in str(exc))
    if FAIL:
        print("FAIL: " + ", ".join(FAIL))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
