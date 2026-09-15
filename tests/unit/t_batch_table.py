"""The Batch table: one shape for the three parsl drivers (records redesign, ticket 20).

UNIT. No engine; under a second.

The ruling (user, 2026-09-15, Q4 (b) and Q7): a Batch writes nothing about itself; the
Slurm log is its report, and it lists every Calculation on one aligned line with the
common columns first (species basin seed rc seconds STATUS record), the driver's own
after. `openqha.store.batch_table` is that shape, so the branch A, branch B and collect
drivers print the same first seven columns.

    A. header and line align: the record column starts at the same offset in both
    B. STATUS after a run: read from the Property file; FAILED when absent and rc != 0;
       NO RECORD when absent and rc == 0; RUNNING survives
    C. the footer names the batch wall, the count and the Slurm job id (read, never chosen)
    D. a missing value prints as '-'; driver-specific columns come after the record
"""
import os
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


def check(label, ok, detail=""):
    print("  {:60s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:300]))
    if not ok:
        FAIL.append(label)


def main():
    from openqha.store import batch_table as bt, property as prop

    print("A. header and lines align:")
    extra = (("frames", 6, ">"), ("verdict", 7, "<"))
    h = bt.header(extra)
    r1 = dict(species="dsgdb9nsd_000018", basin=0, seed=0, rc=0, seconds=12.34,
              status="NORMAL TERMINATION", record="/r/t/1_16000/1_1000/dsgdb9nsd_000018/_records/md_openmm/basin00/md.toml",
              frames=77, verdict="PASS")
    r2 = dict(species="dsgdb9nsd_000035", basin=2, seed=None, rc=1, seconds=3.0, status="FAILED",
              record="/r/t/1_16000/1_1000/dsgdb9nsd_000035/_records/md_openmm/basin02/md.toml", frames=None)
    l1, l2 = bt.line(r1, extra), bt.line(r2, extra)
    check("common columns in order", h.split()[:7] == ["species", "basin", "seed", "rc", "seconds", "STATUS", "record"], h)
    check("record column at one offset", h.index("record") == l1.index("/r/") == l2.index("/r/"), (h, l1, l2))
    check("STATUS column aligned", h.index("STATUS") == l1.index("NORMAL") == l2.index("FAILED"), (l1, l2))
    check("seconds to one decimal", "12.3" in l1 and "3.0" in l2, (l1, l2))
    check("driver columns after the record", l1.index("PASS") > l1.index("md.toml"), l1)

    print("B. STATUS after a run:")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "md.toml"
        check("absent, rc 1 -> FAILED", bt.status_after(1, p) == prop.FAILED)
        check("absent, rc 0 -> NO RECORD", bt.status_after(0, p) == bt.NO_RECORD)
        prop.write(p, {"Calculation_Info": {"X": 1}}, {}, status=prop.RUNNING, progname="t")
        check("RUNNING read back whatever rc", bt.status_after(137, p) == prop.RUNNING)
        prop.write(p, {"Calculation_Info": {"X": 1}}, {}, status=prop.NORMAL_TERMINATION, progname="t")
        check("NORMAL TERMINATION read back", bt.status_after(0, p) == prop.NORMAL_TERMINATION)
        check("absolute path", bt.absolute(p) == str(p.resolve()))

    print("C. the footer:")
    os.environ["SLURM_JOB_ID"] = "7351234"
    f = bt.footer(123.4, 2, 3)
    check("wall, count, verdict text", f[0] == "batch wall 123.4 s   2/3 passed every criterion", f)
    check("the Slurm job id is read from the environment", f[1] == "slurm job 7351234", f)
    del os.environ["SLURM_JOB_ID"]
    check("...and says so when there is none", bt.footer(1.0, 0, 0)[1].startswith("slurm job (none"), bt.footer(1.0, 0, 0))

    print("D. missing values and error lines:")
    check("None prints as -", l2.split()[2] == "-" and l2.rstrip().endswith("-"), l2)
    printed = []
    bt.print_table([dict(r2, error_line="ValueError: no basins")], extra, out=printed.append)
    check("header, line, then the error line indented", len(printed) == 3 and "ValueError" in printed[2], printed)
    printed = []
    bt.print_table([r1, r2], extra, out=printed.append)
    check("print_table pads the record column so the driver columns align under their names",
          printed[0].index("frames") + len("frames") == printed[1].index("77") + 2 and printed[0].index("verdict") == printed[1].index("PASS"),
          printed)

    print()
    print("FAILED: {}".format(FAIL) if FAIL else "all checks passed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
