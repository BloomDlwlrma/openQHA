"""Ticket 26: the SHAKE fallback keeps the published run's ensemble when the retry crashes,
and branch A's failure marker.

UNIT. No CREST: `crest.run` is replaced by a fake that writes (or does not write) a
`crest_conformers.xyz` and returns the record shape `run` returns. Held:

    A. first attempt normal with 3 `terminated EARLY` and an ensemble, retry not normal
       without one (the 2026-09-22 dsgdb9nsd_003375 case) -> the FIRST record comes back,
       `used_shake_fallback` False, `fallback_failed` naming the retry and its last line
    B. first attempt without an ensemble, retry without one -> the retry's record as before
    C. retry normal with an ensemble -> the retry's record, `used_shake_fallback` True
    D. `basins.write_failed / failed / clear_failed`, and `failed` is False once a completed
       Property file is beside the marker
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
FAIL = []


def check(name, ok, detail=None):
    print("  {} {}".format(name, "ok" if ok else "FAIL" + (" " + repr(detail) if detail is not None else "")))
    if not ok:
        FAIL.append(name)


from openqha.conformer_search import crest          # noqa: E402
from openqha.store import basins, layout            # noqa: E402


class FakeCrest:
    """`plan`: per call, (terminated_normally, n_terminated_early, write_ensemble)."""

    def __init__(self, plan):
        self.plan, self.calls = list(plan), []

    def __call__(self, workdir, input_xyz, shake=2, **kwargs):
        normal, early, ensemble = self.plan.pop(0)
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        self.calls.append((str(workdir), shake))
        if ensemble:
            (workdir / "crest_conformers.xyz").write_text("3\nE=0\nC 0 0 0\nH 1 0 0\nH 0 1 0\n", encoding="utf-8")
        rec = dict(workdir=str(workdir), terminated_normally=normal, n_terminated_early=early, seconds=12.0,
                   settings=dict(shake=shake), ok=bool(normal and early == 0))
        if not rec["ok"]:
            rec["tail"] = "Trial MTD 6 did not converge!\nAutomatic MD restart failed 6 times!\n" if not normal else "x\nterminated EARLY\n"
        return rec


def main():
    real_run = crest.run
    try:
        with tempfile.TemporaryDirectory(prefix="fallback_") as tmp:
            xyz = Path(tmp) / "m.xyz"
            xyz.write_text("1\n\nH 0 0 0\n", encoding="utf-8")
            # A
            fake = FakeCrest([(True, 3, True), (False, 0, False)])
            crest.run = fake
            rec = crest.run_with_shake_fallback(Path(tmp) / "A" / "crest", xyz, shake=2, fallback_to=1)
            ff = rec.get("fallback_failed") or {}
            check("A: published run normal + 3 EARLY with an ensemble, retry crashed without one -> the first record, "
                  "used_shake_fallback False, fallback_failed names the retry dir, shake 1 and its last line",
                  rec["workdir"].endswith("crest") and rec["used_shake_fallback"] is False and rec["shake_used"] == 2
                  and ff.get("workdir", "").endswith("crest_shake1") and ff.get("shake") == 1
                  and ff.get("terminated_normally") is False and ff.get("last_line") == "Automatic MD restart failed 6 times!"
                  and "published run's ensemble is used" in rec["fallback_reason"] and len(fake.calls) == 2,
                  (rec.get("workdir"), rec.get("used_shake_fallback"), ff))
            # B
            fake = FakeCrest([(False, 2, False), (False, 0, False)])
            crest.run = fake
            rec_b = crest.run_with_shake_fallback(Path(tmp) / "B" / "crest", xyz, shake=2, fallback_to=1)
            check("B: neither attempt has an ensemble -> the retry's record (used_shake_fallback True), no fallback_failed",
                  rec_b["workdir"].endswith("crest_shake1") and rec_b["used_shake_fallback"] is True
                  and "fallback_failed" not in rec_b and rec_b["first_attempt"]["n_terminated_early"] == 2,
                  (rec_b.get("workdir"), rec_b.get("used_shake_fallback")))
            # C
            fake = FakeCrest([(True, 4, True), (True, 0, True)])
            crest.run = fake
            rec_c = crest.run_with_shake_fallback(Path(tmp) / "C" / "crest", xyz, shake=2, fallback_to=1)
            check("C: the retry succeeds -> its record as before (shake_used 1, used_shake_fallback True)",
                  rec_c["workdir"].endswith("crest_shake1") and rec_c["shake_used"] == 1 and rec_c["used_shake_fallback"] is True
                  and rec_c["ok"] and "fallback_failed" not in rec_c, (rec_c.get("workdir"), rec_c.get("shake_used")))
            # D
            root = Path(tmp) / "root"
            qid, tag = "dsgdb9nsd_000001", "t"
            mol = layout.molecule_dir(root, tag, qid)
            records = layout.records_dir(mol)
            before = basins.failed(qid, tag, root=root)
            basins.write_failed(records, "branch A failed for {}: CREST produced no ensemble".format(qid))
            marked = basins.failed(qid, tag, root=root)
            text = basins.failed_path(qid, tag, root=root).read_text(encoding="utf-8")
            basins.clear_failed(records)
            cleared = basins.failed(qid, tag, root=root)
            check("D: no marker -> not failed; write_failed -> failed with the text; clear_failed -> not failed; the marker is _records/branchA.failed",
                  before is False and marked is True and text.startswith("branch A failed for dsgdb9nsd_000001") and cleared is False
                  and basins.failed_path(qid, tag, root=root) == records / "branchA.failed", (before, marked, cleared))
    finally:
        crest.run = real_run
    print("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
