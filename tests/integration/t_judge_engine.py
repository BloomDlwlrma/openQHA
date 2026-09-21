"""Ticket 14 of the Hessian-learning set: the judge on the REAL engine, with its two
calibrations.

INTEGRATION (engine, CPU). Builds a one-molecule Dataset from the 2-methyloxirane frame
fixture (basin frame with a reference Hessian) and judges MACE-OFF23_medium against
itself:

  * must-PASS: the base model's numbers reproduce `hessian_compare`'s known ones and
    `||A||_F^2/n_vib` = 2.8166e-2 (the value ticket 12 measured on the same frame); the
    engine columns equal the base columns (it IS the base); against `H_r := H_base` the
    loss is ~0; no degradation line FAILs.
  * must-FAIL: `ScaledCalculator(0.9)` -- every frequency 0.9x -- fails the low-mode line.
  * and the base model ITSELF fails the low-mode line on this molecule (12.7 cm^-1
    against the 8.5 threshold): 2-methyloxirane is an out-of-distribution ring, which is
    where MACE-OFF23's curvature error lives (S0-C-41). A judge that passed it here would
    not be measuring what the fine-tune is for.

SKIPs without the model.
"""
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
FIX = ROOT / "tests" / "data" / "methyloxirane_frames"
LEVEL = "wb97m-d3bj_def2-tzvppd"
MACE_LEVEL = "mace-off23_medium"
FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


def main():
    from openqha.data import dataset, frames
    from openqha.potentials import engine
    from openqha.training import judge, phl
    try:
        calc, name, prov = engine.calculator(device="cpu")
    except Exception as exc:                                     # noqa: BLE001
        print("SKIP: {}".format(exc))
        return 0
    print("  engine {}  params {}...".format(name, prov["params_sha256"][:12]))

    ref = frames.read_frames(FIX / "basin.{}.extxyz".format(LEVEL))
    mace_basin = frames.read_frames(FIX / "basin.{}.extxyz".format(MACE_LEVEL))[0]
    with tempfile.TemporaryDirectory() as tmp:
        from openqha.data import dataset as dataset_mod
        d = Path(dataset_mod.datasets_dir(Path(tmp), "smoke", "smoke"))
        d.mkdir(parents=True)
        dataset._write_split(d / "test.{}.extxyz".format(LEVEL), [(a, "test") for a in ref], reference=True)

        rows, anh = judge.frame_rows(d, "smoke", LEVEL, calc, base_calc=calc, splits=("test",),
                                     index=[dict(qm9_index="dsgdb9nsd_000044", classes="epoxide;small_ring")])
        r = rows[0]
        exact = phl.projected_loss_full(mace_basin.info["hessian"], ref[0].info["hessian"],
                                        ref[0].get_masses(), ref[0].positions)
        check("the judge's ||A||_F^2/n_vib on the basin frame = the fixture's 2.8166e-2 (1e-6 relative)",
              abs(r["loss_exact"] / exact - 1) < 1e-6 and abs(r["loss_exact"] - 2.8166e-2) < 1e-5,
              (r["loss_exact"], exact))
        check("the engine columns equal the base columns (it IS the base model)",
              abs(r["freq_mae_cm"] - r["base_freq_mae_cm"]) < 1e-9
              and abs(r["loss_exact"] - r["base_loss_exact"]) < 1e-15,
              (r["freq_mae_cm"], r["base_freq_mae_cm"], r["loss_exact"], r["base_loss_exact"]))
        check("the frame carries its classes, its distribution and the Label's noise floor",
              r["classes"] == "epoxide;small_ring" and r["distribution"] == "out_of_molecule"
              and r["noise_floor_cm"] > 0, r)
        print("    2-methyloxirane basin: low-mode MAE {:.2f}, full MAE {:.2f} cm^-1, "
              "||A||^2/n {:.4e}, noise floor {:.1f} cm^-1".format(
                  r["freq_mae_low_cm"], r["freq_mae_cm"], r["loss_exact"], r["noise_floor_cm"]))

        # must-pass: the base model against a Label that IS its own Hessian
        self_label = [a.copy() for a in ref]
        for a, src in zip(self_label, ref):
            a.calc = src.calc
            a.info = dict(src.info)
            a.info["hessian"] = np.asarray(mace_basin.info["hessian"])
        dd = Path(dataset_mod.datasets_dir(Path(tmp), "selfsmoke", "selfsmoke"))
        dd.mkdir(parents=True)
        dataset._write_split(dd / "test.{}.extxyz".format(LEVEL), [(a, "test") for a in self_label], reference=True)
        srows, _ = judge.frame_rows(dd, "selfsmoke", LEVEL, calc, splits=("test",), index=[])
        # not exactly zero: the Label is the engine's Hessian at the FIXTURE's positions,
        # while the engine recomputes at the extxyz's 8-decimal ones (5e-9 A) -- 3e-5 cm^-1
        check("A5 through the judge: H_r := H_base gives ~0 loss and ~0 frequency error",
              srows[0]["loss_exact"] < 1e-12 and srows[0]["freq_mae_cm"] < 1e-3, srows[0])

        out = judge.run(Path(tmp), "smoke", "smoke", LEVEL, calc, name, base_calc=calc, base_engine=name,
                        run_name="base", splits=("test",), engine_params_sha256=prov["params_sha256"],
                        base_params_sha256=prov["params_sha256"], write=True)
        lines = {l["LINE"]: l for l in out["verdict"]}
        # The must-pass of the ticket is the NO-DEGRADATION line (base against base) and
        # the self-label zero above -- NOT the low-mode line. On this molecule the base
        # model FAILS the low-mode line at 12.7 cm^-1 against the 8.5 threshold, and that
        # is the judge working: 2-methyloxirane is an out-of-distribution ring, exactly
        # where MACE-OFF23's curvature error lives (S0-C-41: -57 cm^-1 on the lowest mode
        # of one of the three rings). A judge that passed the base model here would be
        # measuring nothing the fine-tune is for.
        check("must-pass: judged against ITSELF no line FAILs (there is no degradation)",
              all(l["RESULT"] != "FAIL" for l in out["verdict"] if l["LINE"] != "held_out_low_mode_mae_cm"),
              [(l["LINE"], l["RESULT"]) for l in out["verdict"]])
        check("the low-mode line is measured and the base model FAILS it on this ring "
              "(the error the fine-tune exists to fix)",
              lines["held_out_low_mode_mae_cm"]["RESULT"] == "FAIL"
              and lines["held_out_low_mode_mae_cm"]["VALUE"] > 8.5, lines["held_out_low_mode_mae_cm"])
        check("... and the line names the Label's own grid noise, so nobody reads a 12.7 "
              "against a 24 cm^-1 floor as settled",
              "grid noise" in lines["held_out_low_mode_mae_cm"]["NOTE"])
        check("... the Record is on disk (judge.out / .toml / .dat)",
              all((out["run_dir"] / ("judge" + e)).is_file() for e in (".out", ".toml", ".dat")))
        print("    verdict lines: " + ", ".join("{} {}".format(l["LINE"], l["RESULT"]) for l in out["verdict"]))

        # must-fail: every frequency x0.9
        bad = judge.run(Path(tmp), "smoke", "smoke", LEVEL, judge.ScaledCalculator(calc, 0.9), name,
                        base_calc=calc, base_engine=name, run_name="base_x0.9", splits=("test",),
                        scale=0.9, write=True)
        blines = {l["LINE"]: l for l in bad["verdict"]}
        check("must-fail: the 0.9x-scaled potential FAILs the low-mode line",
              blines["held_out_low_mode_mae_cm"]["RESULT"] == "FAIL" and bad["info"]["VERDICT"] == "FAIL",
              blines["held_out_low_mode_mae_cm"])
        print("    scaled 0.9: low-mode MAE {:.2f} cm^-1 against the base model's {:.2f}".format(
            blines["held_out_low_mode_mae_cm"]["VALUE"], lines["held_out_low_mode_mae_cm"]["VALUE"]))
        check("... and its ||A||^2/n is far worse than the base model's",
              bad["distributions"][0]["LOSS_EXACT"] > 10 * out["distributions"][0]["LOSS_EXACT"],
              (bad["distributions"][0]["LOSS_EXACT"], out["distributions"][0]["LOSS_EXACT"]))

        try:
            judge.run(Path(tmp), "nowhere", "nowhere", LEVEL, calc, name, write=False)
            check("a judge run with no labelled frame REFUSES (it must not answer PASS)", False)
        except ValueError as exc:
            check("a judge run with no labelled frame REFUSES (it must not answer PASS)",
                  "nothing to judge" in str(exc), str(exc))

    print("\n{} checks, {} failed".format(11, len(FAIL)))
    print("PASS" if not FAIL else "FAIL")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
