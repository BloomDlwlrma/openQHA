"""report_ensemble sums the entropies collect judged; it does not analyse frames itself.

UNIT. No engine, no pandas (collect leaves one Table, collect.dat); under a second.

The defect (an113, 2026-09-13)
------------------------------
The first chain to reach `report` -- six 5-frame test trajectories, collect's table
printed ten seconds earlier -- reported

     basin     dE_el/kcal     T*S/kcal      dG/kcal   population  entropy?
         0         0.0000            -       0.0000       0.8039  MISSING
         1         0.8360            -       0.8360       0.1961  MISSING

because `entropy_per_basin` re-ran `qha.analyse` on the raw frames and skipped, without
a word, every trajectory shorter than 3N frames. Its own docstring says "It assembles, it
does not compute". It reads the `trajectories` section of `collect.dat`, the rows collect judged, and carries the verdict of `collect.toml` into its exit code.

What is asserted
----------------
  A. Given a trajectories table (2 basins x 3 seeds, labelled 'basin00'/'seed00' the
     way collect labels them), the per-basin T*S is the mean over seeds of the table's
     TS_QH_kcal, with n_seeds and the source recorded; a basin with no row is None
     (MISSING), not zero and not recomputed.
  B. With no table at all it refuses, naming the collect command.
  C. collect_verdict reads N_PASSED/N_TOTAL from [Criteria] in collect.toml: 5 of 10;
     (None, None) without a Property file or without the block.
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
    from openqha.quasi_harmonic import chain_records
    drv = _load_driver()
    from openqha import config
    cfg = config.load()

    with tempfile.TemporaryDirectory() as tmp:
        # collect's Record sits in the molecule directory:
        # <root>/<tag>/<range>/<chunk>/<species>/_records/md_openmm/collect.{out,toml,dat}
        stem = drv.collect_stem(SPECIES, TAG, root=tmp)
        stem.parent.mkdir(parents=True)

        print("B. no collect product -> refusal that names the collect command")
        try:
            drv.collect_trajectories(SPECIES, TAG, root=tmp)
            FAIL.append("collect_trajectories returned with no Table present")
            print("   FAIL  returned")
        except SystemExit as exc:
            ok = "s0_B_qha_analyse.py" in str(exc) and TAG in str(exc) and "collect.dat" in str(exc)
            print("   {}  {}".format("ok  " if ok else "FAIL", str(exc).splitlines()[0]))
            if not ok:
                FAIL.append("the refusal does not name the collect command and the Table")
        ok = drv.collect_verdict(SPECIES, TAG, root=tmp) == (None, None)
        print("   {}  no collect.toml -> verdict (None, None)".format("ok  " if ok else "FAIL"))
        if not ok:
            FAIL.append("verdict without a Property file is not (None, None)")
        # A Table that lost its [trajectories] section is refused, not read as "every basin missing".
        (chain_records.collect_paths(stem)["dat"]).write_text("\n".join(["[blank]", "# species basin", ""]), encoding="utf-8")
        try:
            drv.collect_trajectories(SPECIES, TAG, root=tmp)
            FAIL.append("a Table without [trajectories] was not refused")
            print("   FAIL  Table without [trajectories] returned")
        except SystemExit as exc:
            ok = "[trajectories]" in str(exc) and "blank" in str(exc)
            print("   {}  a Table without [trajectories] is refused, naming what it has".format("ok  " if ok else "FAIL"))
            if not ok:
                FAIL.append("the refusal for a Table without [trajectories] does not say so")

        # The labels are collect's: directory names, not integers. The first fixture
        # here used 0/1 and passed while the real table ('basin00') raised ValueError.
        rows = [dict(species=SPECIES, basin="basin{:02d}".format(b),
                     seed="seed{:02d}".format(s), n_frames=3000,
                     TS_QH_kcal=ts, TS_Schlitter_kcal=ts + 0.1)
                for b, s, ts in [(0, 0, 10.0), (0, 1, 10.2), (0, 2, 10.4),
                                 (1, 0, 12.0), (1, 1, 12.0), (1, 2, 12.0)]]
        # One Table: the trajectories section beside empty blank and assembly.
        chain_records.write_collect_table(stem, dict(trajectories=rows, blank=[], assembly=[]))
        verdicts = [("{}  criterion {}".format(i, i), "x", i < 5) for i in range(10)]
        chain_records.write_collect(stem, dict(species=SPECIES, tag=TAG, basin_tag=TAG,
                                               setting="default", route="openmm"), verdicts)

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

        print("\nC. collect's verdict is read from [Criteria] in collect.toml, not recounted")
        n_pass, n_crit = drv.collect_verdict(SPECIES, TAG, root=tmp)
        ok = (n_pass, n_crit) == (5, 10)
        print("   {}  {} of {} passed".format("ok  " if ok else "FAIL", n_pass, n_crit))
        if not ok:
            FAIL.append("collect_verdict: {} {}".format(n_pass, n_crit))
        # A Property file without the block is "no verdict", not zero.
        from openqha.store import property as prop
        prop.write(chain_records.collect_paths(stem)["toml"],
                   {prop.INFO_BLOCK: {"QM9_INDEX": SPECIES}}, chain_records.COLLECT_SCHEMA,
                   status=prop.NORMAL_TERMINATION, progname=chain_records.COLLECT_PROGNAME)
        ok = drv.collect_verdict(SPECIES, TAG, root=tmp) == (None, None)
        print("   {}  collect.toml without [Criteria] -> (None, None)".format("ok  " if ok else "FAIL"))
        if not ok:
            FAIL.append("a Property file without [Criteria] did not give (None, None)")

    # D. the electronic energies come from branchA.toml's
    #    [[Basin]] blocks, RELATIVE by name; a basin without it raises, never zero.
    doc = {"Calculation_Status": {"STATUS": "NORMAL TERMINATION"},
           "Basin": [{"INDEX": 1, "RELATIVE": 0.836}, {"INDEX": 0, "RELATIVE": 0.0}]}
    rel = drv.basin_electronic(doc)
    ok = rel == [0.0, 0.836]
    print("{}  basin_electronic reads RELATIVE from [[Basin]] in INDEX order: {}".format("ok  " if ok else "FAIL", rel))
    if not ok:
        FAIL.append("basin_electronic: {}".format(rel))
    try:
        drv.basin_electronic({"Basin": [{"INDEX": 0}]})
        FAIL.append("basin_electronic did not raise without RELATIVE")
        print("FAIL  no RELATIVE: did not raise")
    except KeyError:
        print("ok    no RELATIVE: raises, no silent zero")

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
