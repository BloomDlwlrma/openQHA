"""The package-1 batch screen uses the frequency floor (ticket 41).

INTEGRATION. Runs the batch screen's own path -- `s0_package1_crest_census.analyse_one`
-- on the propanal CREST fixture with the real MACE calculator (no CREST binary
involved); tens of seconds. Skips with a message when MACE is absent.

What is asserted here: the record carries the floor the screen applied, and that floor
is the configured one (`package2.ithr_cm`, the value the `crest` preset carries -- the
single source); every pooled candidate's record carries the verdict and the
window/below-floor counters; every saddle is listed with its below-floor count and
lowest frequency; the basin list is exactly the kept subset of the verdicts; and the
fixture's own basins come through as basins against the real spectra.

The window and below-floor decisions themselves are pinned on synthetic spectra by the
unit test `t_hessian_screen_floor.py`, and the rule's single definition by
`census_verdict` (ticket 37).
"""
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
sys.path.insert(0, str(ROOT / "scripts" / "production"))
ENS = ROOT / "tests" / "data" / "propanal_crest" / "crest_conformers.xyz"
QID = "dsgdb9nsd_000035"
FAIL = []


def check(label, ok, detail=""):
    print("  {:64s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:300]))
    if not ok:
        FAIL.append(label)


def main():
    try:
        from openqha import config, engine
        from openqha.thermochem import thermo
        import s0_package1_crest_census as batch
    except Exception as exc:                          # noqa: BLE001
        print("SKIP: {}".format(exc))
        return 0

    cfg = config.load()
    ithr_cfg = float(config.package(2, cfg)["ithr_cm"])

    try:
        calc, _name, _prov = engine.calculator(device="cpu")
    except Exception as exc:                          # noqa: BLE001
        print("SKIP: {}".format(exc))
        return 0

    with tempfile.TemporaryDirectory(prefix="openqha_t41_") as td:
        rec, basins = batch.analyse_one(
            QID, "CCC=O", Path(td), calc,
            start_info=dict(source="fixture: tests/data/propanal_crest"),
            run_rec={}, do_hessian=True, ensemble=str(ENS))

        print("A. the floor reached the screen from the configuration")
        check("the record names the floor the screen applied",
              rec.get("ithr_cm") == ithr_cfg, rec.get("ithr_cm"))
        check("that floor is the crest preset's own value -- one number, two seams",
              ithr_cfg == thermo.MSRRHO_PRESETS["crest"]["ithr_cm"] == -50.0, ithr_cfg)

        print("B. every pooled candidate carries the verdict and the counters")
        hs = rec["hessian_all_pooled"]
        check("{} pooled candidate(s), every record with verdict / n_below_ithr / "
              "n_inversion_window".format(len(hs)),
              len(hs) >= 1
              and all(h.get("verdict") in ("basin", "saddle") for h in hs)
              and all("n_below_ithr" in h and "n_inversion_window" in h
                      and "lowest_frequency_cm_inv" in h for h in hs), hs)
        check("the verdict is exactly the floor rule: saddle iff a mode is below ithr "
              "(n_below_ithr > 0)",
              all((h["verdict"] == "saddle") == (h["n_below_ithr"] > 0) for h in hs), hs)
        check("every saddle is listed with its below-floor count and lowest frequency",
              len(rec["saddles"]) == rec["n_saddles_rejected"]
              and all("n_below_ithr" in s and "lowest_frequency_cm_inv" in s
                      and s["n_below_ithr"] >= 1 for s in rec["saddles"]),
              rec["saddles"])

        print("C. the basin list is exactly the kept subset of the verdicts")
        kept = sorted(h["index"] for h in rec["basin_hessian"])
        by_verdict = sorted(i for i, h in enumerate(hs) if h["verdict"] == "basin")
        check("basin_hessian holds one record per kept candidate, in energy order",
              rec["n_basins"] == len(rec["basin_hessian"]) == len(basins)
              and kept == by_verdict, (rec["n_basins"], kept, by_verdict))
        check("every kept basin carries its verdict, its lowest mode and its window "
              "count; the fixture's spectra are clean minima",
              all(h["verdict"] == "basin" and h["n_below_ithr"] == 0
                  and h["lowest_frequency_cm_inv"] == min(h["frequencies_cm_inv"])
                  and h["n_inversion_window"] == 0 for h in rec["basin_hessian"]),
              [h["lowest_frequency_cm_inv"] for h in rec["basin_hessian"]])

    print("\n{}".format("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL)))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
