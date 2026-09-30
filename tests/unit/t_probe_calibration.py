"""`scripts/tooling/s0_probe_calibration.py`,
the fixed-probe calibration, on the stored fixture pair (propanal: the ORCA Hessian and the
MACE-OFF23_medium Hessian at the same basin) and on the 2-methyloxirane frames.

Asserted, all on the two stored matrices (no model call, so this test runs anywhere):
`frame_statistics` reproduces the exact target to 1e-12 and is unbiased -- the mean of the
seed sets sits within a few predicted standard deviations of the exact value, and the
deterministic limit holds (K = 3N unit probes would be exact; the Rademacher spread falls as
1/sqrt(K), checked as a ratio between K = 1 and K = 16); the predicted per-frame sd (eq. 2.3)
agrees with the measured spread of many sets within 15 %; `summarise` computes OFFSET and
SEED_SPREAD in units of the exact mean and draws no verdict (the `ENOUGH` column and the
`--project-n` extrapolation are gone: whether K is enough is measured on a real
run, not extrapolated here); the mean's spread
falls as 1/sqrt(n_frames) when the same frame is repeated; the reader takes a glob of labelled
extxyz and skips frames without a Hessian; a molecule whose branch A finished but whose Frame set
is missing is skipped rather than raised on (the tianhe crash of 2026-09-23) and the funnel says so;
the tool runs end to end on the fixture glob with a stub engine and writes a Record whose [[K]] rows
carry every schema key.
"""
import math
import subprocess
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
sys.path.insert(0, str(ROOT / "scripts" / "tooling"))
from openqha.qm_interfaces import orca                            # noqa: E402
from openqha.store import property as prop                        # noqa: E402
from openqha_hessian import phl, phl_loss                         # noqa: E402
import s0_probe_calibration as tool                               # noqa: E402

LEVEL = "wb97m-d3bj_def2-tzvppd"
FIXP = ROOT / "tests" / "data" / "propanal_molecule"
FIXM = ROOT / "tests" / "data" / "methyloxirane_frames"
FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:300]))
    if not ok:
        FAIL.append(label)


def fixture_pair():
    parsed = orca.parse_hess(FIXP / "msrrho" / "orca.{}.basin00.hess".format(LEVEL))
    h_r = orca.hessian_to_ev_per_angstrom2(parsed["hessian_eh_bohr2"])
    h_t = np.load(FIXP / "mace" / "basin00" / "hessian_at_{}.npy".format(LEVEL))
    return np.asarray(h_t, dtype=float), np.asarray(h_r, dtype=float)


def main():
    h_t, h_r = fixture_pair()
    n3 = h_t.shape[0]
    exact = float(np.sum((h_t - h_r) ** 2)) / (n3 * n3)
    ks = [1, 2, 4, 16]
    n_at = n3 // 3
    prod_v, sets_v = tool.frame_probes("dsgdb9nsd_000035", ("basin", 0, 0), n_at, max(ks), seed_sets=200)
    check("the production set is [k_max, 3N] ~ N(0, I) (PHL's Algorithm 1), drawn from the "
          "frame's IDENTITY and nothing else: the same call twice gives the same set, another "
          "frame's differs, and a rewritten Label does not enter it",
          prod_v.shape == (max(ks), n3) and abs(float(np.std(prod_v)) - 1.0) < 0.15
          and np.array_equal(prod_v, tool.frame_probes("dsgdb9nsd_000035", ("basin", 0, 0), n_at, max(ks), 1)[0])
          and not np.array_equal(prod_v, tool.frame_probes("dsgdb9nsd_000035", ("basin", 0, 1), n_at, max(ks), 1)[0]),
          prod_v.shape)
    check("a frame that carries its own stored set is read, not redrawn",
          np.array_equal(tool.frame_probes("x", ("basin", 0, 0), n_at, max(ks), 1,
                                           stored=prod_v.reshape(-1))[0], prod_v))

    st = tool.frame_statistics(h_t, h_r, ks, prod_v, sets_v)
    check("the readings are nested in K: the K = 1 reading uses the first row of the same set",
          abs(st["production"][1] - float(np.sum((prod_v[:1] @ (h_t - h_r).T) ** 2)) / (n3 * n3)) < 1e-9,
          st["production"][1])
    check("the exact target equals ||dH||^2/(9N^2) (1e-12)", abs(st["exact"] - exact) < 1e-12 * max(1.0, exact),
          (st["exact"], exact))
    check("the stable rank is finite and below 3N", 1.0 <= st["r_eff"] <= n3, st["r_eff"])

    # unbiasedness: the mean over 200 independent fixed sets is within a few s.e. of the exact value
    ok_unbiased, detail = True, []
    for k in ks:
        vals = st["sets"][k]
        se = float(np.std(vals) / np.sqrt(len(vals)))
        detail.append((k, float(np.mean(vals)), exact, se))
        if abs(float(np.mean(vals)) - exact) > 4.0 * se:
            ok_unbiased = False
    check("the mean of 200 fixed sets is within 4 s.e. of the exact value, every K", ok_unbiased, detail)

    # the spread falls as 1/sqrt(K)
    ratio = float(np.std(st["sets"][1]) / np.std(st["sets"][16]))
    check("the spread falls as 1/sqrt(K) (K = 1 vs 16: ratio 4 within 25 %)", abs(ratio - 4.0) < 1.0, ratio)

    # eq. 2.3 predicts the spread. The band is NOT a fixed percentage: a sample variance has
    # its own sampling error, sd(s^2) = s^2 sqrt((kurt - 1) / n), and with PHL's normal draw
    # X = v^T B v is a weighted sum of chi^2_1, so that error is several percent here. The
    # band is read off the draws themselves, exactly as t_phl does it.
    ok_var, detail = True, []
    for k in ks:
        vals = np.asarray(st["sets"][k], dtype=float)
        pred_var = float(st["var"][k])
        meas_var = float(vals.var())
        kurt = float(np.mean((vals - vals.mean()) ** 4) / vals.var() ** 2)
        band = 3.0 * meas_var * math.sqrt((kurt - 1) / len(vals))
        detail.append((k, "pred {:.3e}".format(pred_var), "meas {:.3e}".format(meas_var),
                       "band {:.0%}".format(band / pred_var), "kurt {:.1f}".format(kurt)))
        if abs(meas_var - pred_var) > band:
            ok_var = False
    check("the measured spread sits within 3 sd(s^2) of eq. 2.3 at every K, the band read off the "
          "draws' own kurtosis rather than asserted", ok_var, detail)
    check("phl.estimator_variance for the draw the validation uses is the source of that prediction",
          abs(st["var"][4] - phl.estimator_variance(h_t, h_r, k=4)[phl_loss.VALID_PROBE]) < 1e-18,
          (st["var"][4], phl_loss.VALID_PROBE))

    # the deterministic limit: 3N unit probes are exact
    a = h_t - h_r
    unit = float(np.sum((np.eye(n3) @ a.T) ** 2)) / (n3 * n3)
    check("3N unit probes give the exact value (1e-12)", abs(unit - exact) < 1e-12 * max(1.0, exact), (unit, exact))

    # --- summarise ------------------------------------------------------------------------------------
    exact_mean, rows = tool.summarise([st], ks)
    check("summarise's exact mean is the frame's exact value", abs(exact_mean - exact) < 1e-12 * max(1.0, exact))
    r4 = next(r for r in rows if r["K"] == 4)
    check("OFFSET is |production reading - exact| / exact",
          abs(r4["OFFSET"] - abs(st["production"][4] - exact) / exact) < 1e-12, r4["OFFSET"])
    check("SEED_SPREAD is the sd of the set means over the exact mean",
          abs(r4["SEED_SPREAD"] - float(np.std(st["sets"][4])) / exact) < 1e-12, r4["SEED_SPREAD"])
    check("COST_RATIO is K / 3N", abs(r4["COST_RATIO"] - 4.0 / n3) < 1e-12, r4["COST_RATIO"])
    check("every [[K]] row carries every schema key", all(set(r) == set(tool.K_ROW) for r in rows), set(rows[0]))
    r1 = next(r for r in rows if r["K"] == 1)
    check("SEED_SPREAD falls as 1/sqrt(K) across the rows (K = 1 vs 4: ratio 2 within 25 %)",
          abs(r1["SEED_SPREAD"] / r4["SEED_SPREAD"] - 2.0) < 0.5, (r1["SEED_SPREAD"], r4["SEED_SPREAD"]))
    check("no row claims a verdict (ENOUGH / SPREAD_AT_N are gone)",
          "ENOUGH" not in r4 and "SPREAD_AT_N" not in r4 and not hasattr(tool, "DEFAULT_TARGET"), sorted(r4))

    # the mean's spread falls as 1/sqrt(n_frames): the same frame repeated 9 times
    _m9, rows9 = tool.summarise([st] * 9, ks)
    r49 = next(r for r in rows9 if r["K"] == 4)
    check("the predicted sd of the mean falls as 1/sqrt(n_frames) (9 frames: 1/3)",
          abs(r49["SD_MEAN_PREDICTED"] / r4["SD_MEAN_PREDICTED"] - 1.0 / 3.0) < 1e-9,
          (r4["SD_MEAN_PREDICTED"], r49["SD_MEAN_PREDICTED"]))

    # --- the reader -----------------------------------------------------------------------------------
    got = tool.labelled_frames(frames_glob=str(FIXM / "basin.{}.extxyz".format(LEVEL)))
    check("the glob reader returns the labelled basin frame with its Hessian",
          len(got) == 1 and got[0][1] == "extxyz" and got[0][0].info.get("hessian") is not None, len(got))
    with tempfile.TemporaryDirectory(prefix="nohess_") as td0:
        from ase.io import write as _write
        a = got[0][0].copy(); a.info = {k: v for k, v in got[0][0].info.items() if k != "hessian"}
        _write(str(Path(td0) / "no_hessian.extxyz"), [a], format="extxyz")
        none = tool.labelled_frames(frames_glob=str(Path(td0) / "no_hessian.extxyz"))
    check("frames without a reference Hessian are skipped", none == [], len(none))

    # --- end to end with a stub engine ----------------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="probecal_") as td:
        out = Path(td) / "pc.toml"
        stub = Path(td) / "stub.py"
        stub.write_text(
            "import numpy as np\n"
            "def _hessian_at(calc, atoms):\n"
            "    n3 = 3 * len(atoms)\n"
            "    h = np.asarray(atoms.info['hessian'], float).reshape(n3, n3)\n"
            "    return 0.9 * h\n", encoding="utf-8")
        runner = Path(td) / "run.py"
        runner.write_text(
            "import sys\n"
            "sys.path.insert(0, {root!r}); sys.path.insert(0, {tooling!r}); sys.path.insert(0, {td!r})\n"
            "import stub\n"
            "import s0_probe_calibration as t\n"
            "from openqha_hessian import judge\n"
            "judge.hessian_at = stub._hessian_at\n"
            "t.judge.hessian_at = stub._hessian_at\n"
            "t.engine.calculator = lambda **kw: (None, 'stub', None)\n"
            "raise SystemExit(t.main({argv!r}))\n".format(
                root=str(ROOT), tooling=str(ROOT / "scripts" / "tooling"), td=str(td),
                argv=["--frames", str(FIXM / "basin.{}.extxyz".format(LEVEL)), "--k", "1", "4",
                      "--seed-sets", "5", "--out", str(out)]),
            encoding="utf-8")
        p = subprocess.run([sys.executable, str(runner)], capture_output=True, text=True)
        check("the tool runs end to end and exits 0", p.returncode == 0, p.stderr[-400:])
        check("the report draws no verdict", "no verdict is drawn" in p.stdout, p.stdout[-300:])
        rec = prop.load(out) if out.is_file() else {}
        info = rec.get("ProbeCalibration", {})
        check("the Record names the engine, the level and the frame count",
              info.get("ENGINE") == "stub" and info.get("N_FRAMES") == 1 and info.get("LEVEL") == LEVEL, info)
        krows = rec.get("K", [])
        check("the Record carries one [[K]] row per K", len(krows) == 2 and {int(r["K"]) for r in krows} == {1, 4}, krows)
        check("the Record carries one [[Frame]] row", len(rec.get("Frame", [])) == 1, rec.get("Frame"))
        check("PRODUCTION_K is phl_loss.VALID_N_PROBES", info.get("PRODUCTION_K") == phl_loss.VALID_N_PROBES, info)

        # an empty tree reports the funnel, not a blank line
        funnel = {}
        empty = tool.labelled_frames(tag="draw300", root=str(Path(td) / "nowhere"), level=LEVEL, funnel=funnel)
        check("an empty tree returns nothing and fills the funnel with the root it searched",
              empty == [] and funnel["n_molecules"] == 0 and funnel["tag_dir"].endswith("draw300"), funnel)
        q = subprocess.run([sys.executable, str(ROOT / "scripts" / "tooling" / "s0_probe_calibration.py"),
                            "--tag", "draw300", "--root", str(Path(td) / "nowhere")], capture_output=True, text=True)
        check("the tool exits 2 naming the root, the tag directory and the environment to source",
              q.returncode == 2 and "runs root" in q.stderr and "DOES NOT EXIST" in q.stderr
              and "S0_RUNS_ROOT" in q.stderr, q.stderr[-300:])

        # a molecule whose branch A finished but whose Frame set is missing (step 02 has not reached
        # it): skipped, not an error -- the tianhe crash of 2026-09-23
        half = Path(td) / "half" / "draw300" / "dsgdb9nsd_006415"
        (half / "_records").mkdir(parents=True)
        (half / "mace" / "basin00").mkdir(parents=True)          # branch A's product: basins.exists()
        _write(str(half / "mace" / "basin00" / "basin.extxyz"), [got[0][0]], format="extxyz")
        prop.write(half / "_records" / "branchA.toml", {"Calculation_Info": {"SMILES": "CCO"}},
                   {"Calculation_Info": {"SMILES": ("String", None, "the molecule")}},
                   prop.NORMAL_TERMINATION, "test")                # branch A's Record: NORMAL TERMINATION

        fun2 = {}
        none2 = tool.labelled_frames(tag="draw300", root=str(Path(td) / "half"), level=LEVEL, funnel=fun2)
        check("a molecule with branch A but no Frame set is skipped, not raised on, and counted in the funnel",
              none2 == [] and fun2["n_molecules"] == 1 and fun2["n_with_frameset"] == 0, fun2)
        h = subprocess.run([sys.executable, str(ROOT / "scripts" / "tooling" / "s0_probe_calibration.py"),
                            "--tag", "draw300", "--root", str(Path(td) / "half")], capture_output=True, text=True)
        check("... and the report points at 02_frames rather than at the root",
              h.returncode == 2 and "Frame set" in h.stderr and "02_frames" in h.stderr, h.stderr[-300:])

        # --device cuda on the login node: the flag is refused at parse time, not deep inside
        # torch.load (the tianhe RuntimeError of 2026-09-23)
        g = subprocess.run([sys.executable, str(ROOT / "scripts" / "tooling" / "s0_probe_calibration.py"),
                            "--tag", "draw300", "--root", str(Path(td) / "nowhere"), "--device", "cuda"],
                           capture_output=True, text=True)
        import torch                                              # the repo depends on it anyway
        if torch.cuda.is_available():
            ok = g.returncode == 2 and "runs root" in g.stderr    # a real GPU passes the guard
        else:
            ok = g.returncode == 2 and "torch.cuda.is_available() is False" in g.stderr
        check("--device cuda without a visible GPU stops at the flag, not inside torch.load",
              ok, g.stderr[-300:])

    print("\n{} checks, {} failed".format(31, len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
