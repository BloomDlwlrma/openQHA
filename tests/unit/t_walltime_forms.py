"""Every walltime a resource config hands to parsl must be in the form parsl parses.

UNIT. Imports the two Tianhe resource configs and, when parsl is present, its own
walltime parser. Under a second.

The defect (an104, 2026-09-13)
------------------------------
`tianhe_ai.py` carried `QHA_WALLTIME = "3-00:00:00"` -- Slurm's day form, which Slurm
accepts and which parsl's providers do not: they split on ':' and int() each field, so
block 0 failed to start with `invalid literal for int() with base 10: '3-00'`. Six tasks,
one message each, after the whole plan had printed. It was the first time that config's
qha role had ever started a block; the value had sat there for a week.

Two syntaxes for one quantity is the kind of thing a test has to hold, because a human
reading `3-00:00:00` sees a perfectly good walltime. So: every *WALLTIME* constant in
both configs, converted through the config's own `parsl_walltime()`, must (A) match
`HH:MM:SS`, (B) be accepted by parsl's own parser when parsl is importable, and (C) the
day form must convert to the equal number of hours. (D) each config's QHA_WALL_BUDGET_S
must be 90% of its own walltime -- the pair that drifted apart once already.
"""
import re
import sys
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
import hpc.resource_configs.tianhe_a as tianhe_a       # noqa: E402
import hpc.resource_configs.tianhe_ai as tianhe_ai     # noqa: E402

HHMMSS = re.compile(r"^\d{2,}:\d{2}:\d{2}$")
FAIL = []

try:
    from parsl.utils import wtime_to_minutes          # what the providers actually call
except Exception:                                     # noqa: BLE001
    wtime_to_minutes = None


def check_config(name, mod, qha_wall_attr):
    print("{}:".format(name))
    consts = sorted(k for k in dir(mod) if k.endswith("WALLTIME") and isinstance(getattr(mod, k), str))
    for k in consts:
        raw = getattr(mod, k)
        conv = mod.parsl_walltime(raw)
        ok_form = bool(HHMMSS.match(conv))
        ok_parsl = True
        if wtime_to_minutes is not None:
            try:
                wtime_to_minutes(conv)
            except Exception as exc:                  # noqa: BLE001
                ok_parsl = False
                FAIL.append("{}.{}={!r} -> {!r}: parsl rejects it: {}".format(name, k, raw, conv, exc))
        if not ok_form:
            FAIL.append("{}.{}={!r} -> {!r} is not HH:MM:SS".format(name, k, raw, conv))
        print("  {:18s} {:>12s} -> {:>10s}  {}".format(
            k, raw, conv, "ok" if (ok_form and ok_parsl) else "FAIL"))
    # D. the budget follows its own walltime
    want = int(0.90 * mod._walltime_seconds(getattr(mod, qha_wall_attr)))
    got = mod.QHA_WALL_BUDGET_S
    print("  QHA_WALL_BUDGET_S  {} s  (90% of {} = {} s)  {}".format(
        got, qha_wall_attr, want, "ok" if got == want else "FAIL"))
    if got != want:
        FAIL.append("{}: QHA_WALL_BUDGET_S {} != 0.9 x {} ({})".format(name, got, qha_wall_attr, want))


def check_day_form():
    print("\nC. the day form converts, and to the right number of hours:")
    for raw, want in (("3-00:00:00", "72:00:00"), ("1-06:30:00", "30:30:00"),
                      ("24:00:00", "24:00:00"), ("00:30:00", "00:30:00")):
        got = tianhe_ai.parsl_walltime(raw)
        ok = got == want
        print("  {:>12s} -> {:>10s}  {}".format(raw, got, "ok" if ok else "FAIL (want {})".format(want)))
        if not ok:
            FAIL.append("parsl_walltime({!r}) = {!r}, expected {!r}".format(raw, got, want))
        if wtime_to_minutes is not None and "-" in raw:
            try:
                wtime_to_minutes(raw)
                # parsl accepting the raw day form would mean the whole premise is stale
                print("      note: parsl accepted the raw {!r} too -- has it learned the day form?".format(raw))
            except Exception:                         # noqa: BLE001
                pass


def main():
    check_config("tianhe_a", tianhe_a, "WALLTIME")
    print()
    check_config("tianhe_ai", tianhe_ai, "QHA_WALLTIME")
    check_day_form()
    print("\n(parsl {}available for the direct check)".format("" if wtime_to_minutes else "NOT "))
    if FAIL:
        print("\n{} problem(s):".format(len(FAIL)))
        for f in FAIL:
            print("  - " + f)
        return 1
    print("\nevery walltime parsl will see is one it can parse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
