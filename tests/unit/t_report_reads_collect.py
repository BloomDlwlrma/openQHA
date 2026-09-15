"""report_ensemble sums the entropies collect judged; it does not analyse frames itself.

UNIT. pandas + pyarrow (the same two every chain's last step needs); under two seconds.

The defect (an113, 2026-09-13)
------------------------------
The first chain to reach `report` -- six 5-frame test trajectories, collect's table
printed ten seconds earlier -- reported

     basin     dE_el/kcal     T*S/kcal      dG/kcal   population  entropy?
         0         0.0000            -       0.0000       0.8039  MISSING
         1         0.8360            -       0.8360       0.1961  MISSING

because `entropy_per_basin` re-ran `qha.analyse` on the raw frames and skipped, without
a word, every trajectory shorter than 3N frames. Its own docstring says "It assembles, it
does not compute". Now it reads `analysis/qha/<tag>/<species>__trajectories.parquet`,
the rows collect judged, and carries collect's criteria verdict into its exit code.

What is asserted
----------------
  A. Given a trajectories table (2 basins x 3 seeds, labelled 'basin00'/'seed00' the
     way collect labels them), the per-basin T*S is the mean over seeds of the table's
     TS_QH_kcal, with n_seeds and the source recorded; a basin with no row is None
     (MISSING), not zero and not recomputed.
  B. With no table at all it refuses, naming the collect command.
  C. criteria_verdict counts the criteria table: 5 of 10; (None, None) without one.
"""
import importlib.util
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

FAIL = []
SPECIES = "t_report_reads_collect_fake"
TAG = "t_report_reads_collect"


def _load_driver():
    path = ROOT / "scripts" / "production" / "s0_B_report_ensemble.py"
    spec = importlib.util.spec_from_file_location("s0_B_report_ensemble", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    try:
        import pandas as pd
        import pyarrow  # noqa: F401
    except ImportError as exc:
        print("skipped: {}".format(exc))
        return 0
    drv = _load_driver()
    from openqha import config
    cfg = config.load()

    with tempfile.TemporaryDirectory() as tmp:
        # Since 2026-09-14 (ADR 0001) the tables sit in the molecule directory:
        # <root>/<tag>/<range>/<chunk>/<species>/_records/md_openmm/default/collect__*.parquet
        stem = drv.collect_stem(SPECIES, TAG, root=tmp)
        stem.parent.mkdir(parents=True)

        print("B. no collect product -> refusal that names the collect command")
        try:
            drv.collect_tables(SPECIES, TAG, root=tmp)
            FAIL.append("collect_tables returned with no table present")
            print("   FAIL  returned")
        except SystemExit as exc:
            ok = "s0_B_qha_analyse.py" in str(exc) and TAG in str(exc)
            print("   {}  {}".format("ok  " if ok else "FAIL", str(exc).splitlines()[0]))
            if not ok:
                FAIL.append("the refusal does not name the collect command")

        # The labels are collect's: directory names, not integers. The first fixture
        # here used 0/1 and passed while the real table ('basin00') raised ValueError.
        rows = [dict(species=SPECIES, basin="basin{:02d}".format(b),
                     seed="seed{:02d}".format(s), n_frames=3000,
                     TS_QH_kcal=ts, TS_Schlitter_kcal=ts + 0.1)
                for b, s, ts in [(0, 0, 10.0), (0, 1, 10.2), (0, 2, 10.4),
                                 (1, 0, 12.0), (1, 1, 12.0), (1, 2, 12.0)]]
        pd.DataFrame(rows).to_parquet(
            stem.parent / (stem.name + "__trajectories.parquet"), index=False)
        crit = [dict(species=SPECIES, criterion="c{}".format(i), measured="x",
                     passed=(i < 5)) for i in range(10)]
        pd.DataFrame(crit).to_parquet(
            stem.parent / (stem.name + "__criteria.parquet"), index=False)

        print("\nA. per-basin T*S is the mean over the table's seeds")
        per_basin, _ = drv.entropy_per_basin(SPECIES, TAG, cfg, "all", root=tmp)
        want = {0: 10.2, 1: 12.0}
        for b, ts in want.items():
            got = per_basin.get(b)
            ok = (got is not None and abs(got["TS_kcal"] - ts) < 1e-12
                  and got["n_seeds"] == 3 and got.get("source") == "collect table")
            print("   {}  basin {}  T*S {}  n_seeds {}  source {}".format(
                "ok  " if ok else "FAIL", b, (got or {}).get("TS_kcal"),
                (got or {}).get("n_seeds"), (got or {}).get("source")))
            if not ok:
                FAIL.append("basin {}: {}".format(b, got))
        if per_basin.get(2) is not None:
            FAIL.append("a basin with no row came back as {}".format(per_basin.get(2)))
        print("   {}  basin 2 (no row) -> {}".format(
            "ok  " if per_basin.get(2) is None else "FAIL", per_basin.get(2)))

        print("\nC. collect's verdict is counted, not recomputed")
        _, criteria = drv.collect_tables(SPECIES, TAG, root=tmp)
        n_pass, n_crit = drv.criteria_verdict(criteria)
        ok = (n_pass, n_crit) == (5, 10) and drv.criteria_verdict(None) == (None, None)
        print("   {}  {} of {} passed; none -> {}".format(
            "ok  " if ok else "FAIL", n_pass, n_crit, drv.criteria_verdict(None)))
        if not ok:
            FAIL.append("criteria_verdict: {} {}".format(n_pass, n_crit))

    print()
    if FAIL:
        print("{} problem(s):".format(len(FAIL)))
        for f in FAIL:
            print("  - " + f)
        return 1
    print("report sums what collect judged, and says when there is nothing to sum")
    return 0


if __name__ == "__main__":
    sys.exit(main())
