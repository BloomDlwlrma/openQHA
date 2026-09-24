"""The stage list's scan (`hpc/slurm/hl_list.py`): threaded, same answer as single-threaded,
and it says what it is doing while it does it.

Why this exists (Tianhe, 2026-09-24): the scan reads one record per drawn molecule -- a
stat for the failure marker, the Property file, the basin listing -- and every metadata op
costs ~8 ms on the shared pool, so 6 458 molecules took **6 min 19 s** single-threaded.
That was the silent gap between an array task's banner and its first molecule line (and 12
tasks ran it at once). Held here:

    A. `--stage branchA` on a four-molecule draw (two done -- one with a Frame set, one
       without -- one marked failed, two never run): the never-run ones are listed in the
       draw's order, the marker is counted as `failed earlier` and not listed, the summary
       line is the one the campaign page documents, and the scan announces itself.
    B. `--workers 1` and `--workers 4` write byte-identical lists.
    C. `--quiet`: no announcement, the summary line stays.
    D. `--array-n/--array-id` round-robin and `--limit` pick from that same list.
    E. `--stage frames`: the done molecule WITHOUT a Frame set is listed, the one with it is
       not; `--force` lists both.
"""
import contextlib
import io
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
sys.path.insert(0, str(ROOT / "tests" / "unit"))
sys.path.insert(0, str(ROOT / "hpc" / "slurm"))
from t_hl_campaign import place, TAG                            # noqa: E402
from t_frames import check, FAIL                                # noqa: E402

from openqha.data import dataset, structure_classes as sc       # noqa: E402
from openqha.store import basins, dat, layout                   # noqa: E402
import hl_list as hl                                            # noqa: E402


def row(qid):
    return dict(qm9_index=qid, smiles="CCC=O", n_heavy=4, classes="aldehyde",
                in_training=False, pinned=False)


def run(hl_argv):
    cap = io.StringIO()
    with contextlib.redirect_stdout(cap):
        rc = hl.main(hl_argv)
    return rc, cap.getvalue()


def main():
    with tempfile.TemporaryDirectory(prefix="hl_list_") as tmp:
        tmp = Path(tmp)
        root, _a = place(tmp, "dsgdb9nsd_000035")                   # done, with a Frame set
        _r, _b = place(tmp, "dsgdb9nsd_000036", with_frames=False)  # done, without one
        basins.write_failed(layout.records_dir(layout.molecule_dir(root, TAG, "dsgdb9nsd_000100")),
                            "branch A failed: no ensemble")          # ticket 26's marker
        d = dataset.datasets_dir(root, TAG, "p")
        d.mkdir(parents=True, exist_ok=True)
        dat.write_table(d / "draw.dat",
                        [row("dsgdb9nsd_000035"), row("dsgdb9nsd_000036"),
                         row("dsgdb9nsd_000100"), row("dsgdb9nsd_000200"), row("dsgdb9nsd_000201")],
                        list(sc.ROW_SCHEMA), sc.ROW_SCHEMA)

        old = os.environ.get("S0_RUNS_ROOT")
        os.environ["S0_RUNS_ROOT"] = str(root)
        try:
            out1, out4 = tmp / "one.txt", tmp / "four.txt"
            rc1, text1 = run(["--tag", TAG, "--name", "p", "--stage", "branchA",
                              "--out", str(out1), "--workers", "1"])
            rc4, text4 = run(["--tag", TAG, "--name", "p", "--stage", "branchA",
                              "--out", str(out4), "--workers", "4"])
            listed = out1.read_text(encoding="utf-8")
            summary = [l for l in text1.splitlines()
                       if l.startswith("branchA: ") and " drawn, " in l]
            check("A: the never-run molecules are listed in draw order, the marker is 'failed "
                  "earlier' and not listed, the summary line keeps the documented shape, the "
                  "scan announces itself",
                  rc1 == 0 and listed == "dsgdb9nsd_000200\ndsgdb9nsd_000201\n"
                  and len(summary) == 1
                  and "5 drawn, 2 pending, 2 for task 0/1" in summary[0]
                  and "1 failed earlier (_records/branchA.failed; not rerun)" in summary[0]
                  and any(l.startswith("branchA: scanning 5 drawn molecule record(s)")
                          for l in text1.splitlines()),
                  (rc1, listed, summary, text1))
            summary4 = [l for l in text4.splitlines()
                        if l.startswith("branchA: ") and " drawn, " in l]
            check("B: --workers 1 and --workers 4 write byte-identical lists (threading is an "
                  "implementation detail, not part of the answer)",
                  rc4 == 0 and out4.read_bytes() == out1.read_bytes() and len(summary4) == 1
                  and "5 drawn, 2 pending, 2 for task 0/1" in summary4[0]
                  and "1 failed earlier" in summary4[0],
                  (rc4, out4.read_text(encoding="utf-8"), summary4))

            rcq, textq = run(["--tag", TAG, "--name", "p", "--stage", "branchA",
                              "--out", str(tmp / "quiet.txt"), "--workers", "2", "--quiet"])
            check("C: --quiet drops the scan lines, the summary line stays",
                  rcq == 0 and "scanning" not in textq and "scan " not in textq
                  and any(l.startswith("branchA: ") for l in textq.splitlines()), textq)

            rc_rr, _t = run(["--tag", TAG, "--name", "p", "--stage", "branchA", "--array-n", "2",
                             "--array-id", "1", "--out", str(tmp / "rr.txt"), "--workers", "4", "--quiet"])
            rc_lim, _t2 = run(["--tag", TAG, "--name", "p", "--stage", "branchA", "--limit", "1",
                               "--out", str(tmp / "lim.txt"), "--workers", "4", "--quiet"])
            check("D: round-robin --array-n/--array-id and --limit partition that same list",
                  (tmp / "rr.txt").read_text(encoding="utf-8") == "dsgdb9nsd_000201\n"
                  and (tmp / "lim.txt").read_text(encoding="utf-8") == "dsgdb9nsd_000200\n"
                  and rc_rr == 0 and rc_lim == 0,
                  ((tmp / "rr.txt").read_text(encoding="utf-8"), (tmp / "lim.txt").read_text(encoding="utf-8")))

            _rc, _t3 = run(["--tag", TAG, "--name", "p", "--stage", "frames",
                            "--out", str(tmp / "frames.txt"), "--workers", "4", "--quiet"])
            _rc, _t4 = run(["--tag", TAG, "--name", "p", "--stage", "frames", "--force",
                            "--out", str(tmp / "frames_force.txt"), "--workers", "4", "--quiet"])
            check("E: stage frames lists the done molecule without a Frame set (not the one "
                  "with it, not the ones branch A never finished); --force lists both done ones",
                  (tmp / "frames.txt").read_text(encoding="utf-8") == "dsgdb9nsd_000036\n"
                  and (tmp / "frames_force.txt").read_text(encoding="utf-8")
                      == "dsgdb9nsd_000035\ndsgdb9nsd_000036\n",
                  ((tmp / "frames.txt").read_text(encoding="utf-8"),
                   (tmp / "frames_force.txt").read_text(encoding="utf-8")))
        finally:
            if old is None:
                os.environ.pop("S0_RUNS_ROOT", None)
            else:
                os.environ["S0_RUNS_ROOT"] = old

    print("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
