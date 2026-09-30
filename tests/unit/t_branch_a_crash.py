"""A molecule that crashes leaves `_records/branchA.failed`, and the empty-basin refusal
says why (`scripts/production/s0_A_pipeline.py`, `openqha/conformer_search/crest_census.py`).

UNIT. No CREST, no engine. Held:

    A. `crash_marker_text`: the marker names the molecule, the tag, the exception (type and
       message), keeps the last lines of the traceback, and carries the hand-rerun command
       with the "not rerun by any round" sentence.
    B. `mark_crashed`: writes `_records/branchA.failed` under the run root; a marker already
       on disk (the missing-ensemble path writes its own) is NOT clobbered; `basins.failed`
       is True afterwards and False after `clear_failed` (a success).
    C. `crest_census._empty_basin_message`: the refusal names the frequency floor or
       ORCA's default line, each rejection class in its own section -- every condemned
       candidate with its below-floor count and lowest frequency, every not-certified
       candidate with its residual -- and the tighten's convergence count: the numbers
       the Tianhe 2026-09-24 run of dsgdb9nsd_052993 did not print.
    D. `crest_census.census_verdict`: the pure floor screen -- a -6.84 cm^-1 candidate
       (dsgdb9nsd_052993) is a basin with one window mode, a -195.79 cm^-1 candidate is
       a saddle, and the counters are the ones the record carries.
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
                                   "frequency-floor screen -- refusing to report an "
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

            # C -- the refusal carries the numbers, in the floor vocabulary
            saddles = [dict(conformer_id=2, n_imaginary=1, n_below_ithr=1,
                            lowest_frequency_cm_inv=-64.20, energy_eV=-100.0),
                       dict(conformer_id=5, n_imaginary=2, n_below_ithr=2,
                            lowest_frequency_cm_inv=-201.5, energy_eV=-99.5)]
            not_certified = [dict(conformer_id=7, residual_eV_A=2.7e-2)]
            msg = crest_census._empty_basin_message(saddles, not_certified,
                                                    [True, False, True],
                                                    [9.9e-5, 3.3e-4, 1.1e-6], -50.0)
            check("C: the two rejection classes have separate sections -- the floor "
                  "saddles (lowest frequency, energy) and the not-certified candidates "
                  "(the final residual, against ORCA's named line) -- and the tighten's "
                  "convergence count is printed",
                  "no basin survives" in msg
                  and "below ithr = -50 cm^-1" in msg
                  and "conformer   2: 1 below ithr, lowest -64.20 cm^-1" in msg
                  and "conformer   5: 2 below ithr, lowest -201.50 cm^-1" in msg
                  and "1 candidate(s) NOT certified" in msg
                  and "TolMaxG = 3e-4 Eh/bohr = 1.543e-02 eV/A" in msg
                  and "conformer   7: residual 2.70e-02 eV/A" in msg
                  and "1 of 3 frame(s) did not reach fmax" in msg
                  and "max residual 3.30e-04" in msg, msg)

            # D -- the census verdict, pure: the frequency floor decides.
            # The two real spectra: dsgdb9nsd_052993's -6.84 cm^-1 candidate (Tianhe
            # 2026-09-24; admitted and inverted by the thermochemistry) and the -195.79
            # cm^-1 cyclopropanol saddle that motivated the screen (ejected).
            v_win = crest_census.census_verdict([-6.84, 130.0, 420.0], -50.0)
            v_sad = crest_census.census_verdict([-195.79, 130.0], -50.0)
            v_line = crest_census.census_verdict([-50.0, 120.0], -50.0)
            v_clean = crest_census.census_verdict([40.0, 120.0], -50.0)
            check("D: -6.84 cm^-1 is a basin with one window mode; -195.79 is a saddle "
                  "below the floor; the floor line itself is in the window; a clean "
                  "spectrum is a basin with nothing to invert",
                  v_win["verdict"] == "basin" and v_win["n_inversion_window"] == 1
                  and v_win["n_below_ithr"] == 0
                  and v_win["lowest_frequency_cm_inv"] == -6.84
                  and v_sad["verdict"] == "saddle" and v_sad["n_below_ithr"] == 1
                  and v_sad["n_inversion_window"] == 0
                  and v_line["verdict"] == "basin" and v_line["n_inversion_window"] == 1
                  and v_clean["verdict"] == "basin"
                  and v_clean["n_inversion_window"] == 0
                  and v_clean["n_below_ithr"] == 0,
                  (v_win, v_sad, v_line, v_clean))
        finally:
            if old is None:
                os.environ.pop("S0_RUNS_ROOT", None)
            else:
                os.environ["S0_RUNS_ROOT"] = old
    print("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
