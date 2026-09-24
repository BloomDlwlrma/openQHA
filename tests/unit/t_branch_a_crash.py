"""A molecule that crashes leaves `_records/branchA.failed`, and the empty-basin refusal
says why (`scripts/production/s0_A_pipeline.py`, `openqha/conformer_search/crest_census.py`).

UNIT. No CREST, no engine. Held:

    A. `crash_marker_text`: the marker names the molecule, the tag, the exception (type and
       message), keeps the last lines of the traceback, and carries the hand-rerun command
       with the "not rerun by any round" sentence.
    B. `mark_crashed`: writes `_records/branchA.failed` under the run root; a marker already
       on disk (the missing-ensemble path writes its own) is NOT clobbered; `basins.failed`
       is True afterwards and False after `clear_failed` (a success).
    C. `crest_census._empty_basin_message`: the refusal lists every condemned candidate with
       its imaginary count and lowest frequency, and the tighten's convergence count -- the
       numbers the Tianhe 2026-09-24 run of dsgdb9nsd_052993 did not print.
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
sys.path.insert(0, str(ROOT / "scripts" / "production"))
FAIL = []


def check(name, ok, detail=None):
    print("  {} {}".format(name, "ok" if ok else "FAIL" + (" " + repr(detail) if detail is not None else "")))
    if not ok:
        FAIL.append(name)


from openqha import config                              # noqa: E402
from openqha.conformer_search import crest_census       # noqa: E402
from openqha.store import basins, layout                # noqa: E402
import s0_A_pipeline as A                               # noqa: E402


def main():
    with tempfile.TemporaryDirectory(prefix="branchA_crash_") as tmp:
        tmp = Path(tmp)
        old = os.environ.get("S0_RUNS_ROOT")
        os.environ["S0_RUNS_ROOT"] = str(tmp / "runs")
        try:
            cfg = config.load()
            tag, qid = "draw300", "dsgdb9nsd_052993"

            # A -- the marker text
            try:
                raise RuntimeError("no basin survives the tightening and the "
                                   "imaginary-frequency filter -- refusing to report an "
                                   "empty basin list")
            except RuntimeError as exc:
                text = A.crash_marker_text(qid, tag, exc, when="2026-09-24T18:00:00")
            check("A: the marker names molecule, tag and time; the exception's type and "
                  "message; the traceback tail; the rerun command; 'not rerun'",
                  text.startswith("branch A crashed for {} under tag {} on 2026-09-24T18:00:00".format(qid, tag))
                  and "RuntimeError: no basin survives the tightening" in text
                  and "Traceback (most recent call last)" in text
                  and "s0_A_pipeline.py --species {} --tag {}".format(qid, tag) in text
                  and "not rerun by any round" in text and text.endswith("\n"), text)

            # A2 -- which crashes park the molecule and which stay retryable
            import subprocess
            check("A2: a RuntimeError marks (one attempt, a human decides); a CREST "
                  "TimeoutExpired stays retryable -- section 2's rule is 'skipped, the next "
                  "round tries again', so it must not park the molecule",
                  A.crash_marker_applies(RuntimeError("x")) is True
                  and A.crash_marker_applies(subprocess.TimeoutExpired(["crest"], 60)) is False)

            # B -- the marker on disk, and no clobbering of a specific one
            wrote = A.mark_crashed(qid, tag, cfg, RuntimeError("boom"))
            path = basins.failed_path(qid, tag, cfg)
            marked = basins.failed(qid, tag, cfg)
            again = A.mark_crashed(qid, tag, cfg, RuntimeError("second, must not replace"))
            kept = path.read_text(encoding="utf-8")
            basins.clear_failed(layout.records_dir(basins.molecule_for(qid, tag, cfg)))
            cleared = basins.failed(qid, tag, cfg)
            check("B: mark_crashed writes _records/branchA.failed under the run root; a "
                  "second call keeps the first text; basins.failed True, False after clear",
                  wrote is True and marked is True and again is False
                  and "boom" in kept and "second, must not replace" not in kept
                  and path == basins.molecule_for(qid, tag, cfg) / "_records" / "branchA.failed"
                  and cleared is False, (wrote, again, kept[:150]))

            # C -- the refusal carries the numbers
            saddles = [dict(conformer_id=2, n_imaginary=1, lowest_frequency_cm_inv=-3.14, energy_eV=-100.0),
                       dict(conformer_id=5, n_imaginary=2, lowest_frequency_cm_inv=-201.5, energy_eV=-99.5)]
            msg = crest_census._empty_basin_message(saddles, [True, False, True], [9.9e-5, 3.3e-4, 1.1e-6])
            check("C: the refusal lists every condemned candidate (imaginary count, lowest "
                  "frequency, energy) and the tighten's convergence count -- the numbers a "
                  "reader needs to tell a real saddle from a soft mode at the noise floor",
                  "no basin survives" in msg
                  and "conformer   2: 1 imaginary, lowest -3.14 cm^-1" in msg
                  and "conformer   5: 2 imaginary, lowest -201.50 cm^-1" in msg
                  and "1 of 3 frame(s) did not reach fmax" in msg
                  and "max residual 3.30e-04" in msg, msg)
        finally:
            if old is None:
                os.environ.pop("S0_RUNS_ROOT", None)
            else:
                os.environ["S0_RUNS_ROOT"] = old
    print("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
