"""Unit test: a CREST scratch directory is never reused under different settings.

UNIT. Milliseconds; writes to a temporary directory,
runs no CREST and loads no potential.

What this pins down
-------------------
Defect 57: reusing a work directory made under different settings mixes two sampling
conditions into one dataset, and **nothing in the products shows it**. The guard is
`crest.settings_match`, which compares only the keys that change the RESULT --
thread count is excluded because it changes speed alone.

`tstep` is in that list for a reason: with the
same workhorse, 5.0 fs aborted 29 metadynamics runs and 2.0 fs aborted none. A 2 fs
scratch directory silently reused as a 5 fs result is exactly defect 57 again.

The comparison also has to survive a round trip through the TOML: settings go in as
numbers and come back as strings, so `shake = 2` and `shake = "2"` must compare
equal. A test that only checked the key list would miss that and the guard would
reject every legitimate reuse instead -- failing safe, but failing.

Run::  python tests/unit/t_crest_settings_reuse.py
"""
import sys
import tempfile
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))

from openqha import crest  # noqa: E402

#: The published branch-A protocol, as the pipeline asks for it.
WANTED = dict(runtype="imtd-gc", optlev="tight", refine="opt",
              shake=2, workhorse="gfn2", tstep=5.0)

#: Each of these differs from WANTED in exactly one key that CHANGES THE RESULT.
#: Every one of them must be refused.
MUST_REFUSE = (
    ("workhorse", dict(WANTED, workhorse="gfnff")),
    ("refine", dict(WANTED, refine="sp")),
    ("shake", dict(WANTED, shake=1)),
    ("tstep", dict(WANTED, tstep=2.0)),
    ("optlev", dict(WANTED, optlev="normal")),
    ("runtype", dict(WANTED, runtype="optimize")),
)


def main():
    fails = []

    # ---- 1. what write_input actually puts on disk comes back unchanged ------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "x.xyz").write_text("1\n\nH 0.0 0.0 0.0\n", encoding="utf-8")
        client = _repo_root() / "scripts" / "production" / "s0_mace_engrad.py"
        crest.write_input(tmp, tmp / "x.xyz", runtype=WANTED["runtype"],
                          optlev=WANTED["optlev"], refine=WANTED["refine"],
                          shake=WANTED["shake"], workhorse=WANTED["workhorse"],
                          tstep_fs=WANTED["tstep"], engine_client=client)
        got = crest.read_input_settings(tmp / "input.toml")

        for key in ("runtype", "optlev", "refine", "shake", "workhorse", "tstep"):
            if got.get(key) is None:
                fails.append("read_input_settings lost {!r} -- it cannot take part "
                             "in the reuse comparison if it is not read back".format(key))

        # The round trip is the point: values go in as numbers, come back as strings.
        ok, why = crest.settings_match(got, WANTED)
        if not ok:
            fails.append("a directory written with exactly these settings was "
                         "refused: {}".format(why))
        else:
            print("PASS  round trip: written settings compare equal to what was asked")

        # ---- 2. every single-key difference is refused -----------------------------
        for key, other in MUST_REFUSE:
            ok, why = crest.settings_match(got, other)
            if ok:
                fails.append("reuse ACCEPTED across a change of {!r} -- this is "
                             "defect 57".format(key))
                print("FAIL  {:<10} accepted".format(key))
            else:
                print("PASS  {:<10} refused: {}".format(key, why))

        # ---- 3. an empty directory is refused, not treated as a match --------------
        ok, why = crest.settings_match(None, WANTED)
        if ok:
            fails.append("a directory with no input.toml was accepted for reuse")
        else:
            print("PASS  {:<10} refused: {}".format("no toml", why))

    print()
    if fails:
        print("{} failure(s):".format(len(fails)))
        for f in fails:
            print("  - " + f)
        return 1
    print("reuse guard rejects every result-changing difference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
