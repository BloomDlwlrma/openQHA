"""Ticket 32: the hkuhpc job bundle of a numerical reference level, and the numerical-route
bookkeeping (no ORCA).

Asserted: the bundle holds one input per job with the level's keywords and blocks, %pal
and the 75 % maxcore rule, a worker that skips finished jobs and publishes only on the
terminal line, a SLURM script with the concurrency that fits the node, a README with the
(6N)^2 point estimate; `orca.LEVELS` names both reference levels with their routes;
`final_rms_gradient` / `n_single_points` / `hessian_route` read the propanal fixture.
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
from openqha.qm_interfaces import orca, orca_jobs            # noqa: E402

FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


def main():
    spec = orca.level_spec("dlpno-ccsdt_cc-pvtz")
    check("orca.LEVELS: wB97M analytic, DLPNO-CCSD(T) numerical with Opt NumGrad NumFreq and TightPNO",
          orca.level_spec("wb97m-d3bj_def2-tzvppd")["route"] == "analytic" and spec["route"] == "numerical"
          and all(k in spec["keywords"] for k in ("Opt", "NumGrad", "NumFreq", "TightPNO", "RIJK", "cc-pVTZ/C")))
    try:
        orca.level_spec("no-such-level")
        check("an unknown level raises naming the known ones", False)
    except KeyError as exc:
        check("an unknown level raises naming the known ones", "wb97m" in str(exc))
    est = orca_jobs.point_estimate(10)
    check("point estimate for 10 atoms: 600 opt + 3600 NumFreq", est["opt"] == 600 and est["numfreq"] == 3600)
    text = (ROOT / "tests/data/propanal_molecule/msrrho/orca.wb97m-d3bj_def2-tzvppd.basin00.out").read_text(encoding="utf-8", errors="replace")
    check("the propanal fixture: analytic route, final RMS gradient read, 6 single points (one per optimisation step)",
          orca.hessian_route(text) == "analytic" and 0 < orca.final_rms_gradient(text) < 1e-4
          and orca.n_single_points(text) == 6)
    check("an output without an optimisation gives no RMS gradient", orca.final_rms_gradient("no table here") is None)

    with tempfile.TemporaryDirectory(prefix="orca_jobs_") as tmp:
        jobs = [dict(name="mol_basin00", symbols=["O", "H", "H"], positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
                dict(name="mol_basin01", symbols=["O", "H", "H"], positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, -0.93, 0]])]
        out = orca_jobs.write_bundle(Path(tmp) / "bundle", jobs, "dlpno-ccsdt_cc-pvtz", nprocs=8,
                                     cpus_per_node=32, mem_gb=128)
        inp = (out["dir"] / "mol_basin00.inp").read_text()
        check("one input per job with the level's keyword line, blocks, %pal 8 and the 75 % maxcore",
              inp.startswith("! " + spec["keywords"]) and "%mdci" in inp and "TCutPairs 1e-6" in inp
              and "%pal nprocs 8 end" in inp and "%maxcore 3072" in inp and out["maxcore"] == 3072)
        check("4 concurrent 8-rank jobs on a 32-core node", out["concurrent"] == 4)
        sb = out["sbatch"].read_text()
        check("run.sbatch: one task with all cores, xargs -P 4, ORCA_BIN_DIR checked, logs/ and results/ made",
              "--cpus-per-task=32" in sb and "xargs -a jobs.txt -P 4" in sb and "ORCA not found" in sb
              and "mkdir -p" in sb)
        wk = out["worker"].read_text()
        check("worker.sh: skips a finished job, unsets PMI/SLURM, binding off, publishes only on the terminal line, keeps the full .out",
              "SKIP (finished)" in wk and "PMI|SLURM" in wk and "ORCA_SKIP_CPU_BIND=1" in wk
              and 'tail -n 100 "$name.out" | grep -q "ORCA TERMINATED NORMALLY"' in wk and "failed.out" in wk)
        rd = out["readme"].read_text()
        check("README states the route, the point estimate table (3 atoms: 180 + 324) and the copy-back path",
              "numerical" in rd and "| mol_basin00 | 3 | 180 | 324 | 504 |" in rd and "--step reference --level dlpno-ccsdt_cc-pvtz" in rd)
        check("jobs.txt lists the jobs in order", (out["dir"] / "jobs.txt").read_text().split() == ["mol_basin00", "mol_basin01"])
    if FAIL:
        print("FAIL: " + ", ".join(FAIL))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
