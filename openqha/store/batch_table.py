"""The Batch table: what a driver over many Calculations prints, and all it leaves.

A Batch (CONTEXT.md) is one driver invocation inside one Slurm job; it owns no Record.
The Slurm log is its report, and this module makes the three parsl drivers print the
same thing: one header, one aligned line per Calculation with the common columns first,

    species  basin  seed  rc  seconds  STATUS  record  <driver-specific columns>

then a footer with the batch wall, the pass count and the Slurm job id. `rc` is the
subprocess return code; `STATUS` is read from the Calculation's Property file AFTER the
subprocess returned (`status_after`): NORMAL TERMINATION, RUNNING for a trajectory cut
during production, FAILED when the file is absent and rc is not zero, NO RECORD when it
is absent and rc is zero (a driver that exited 0 without writing is a defect to look
at); `record` is the Property file's absolute path, because the Slurm log sits in the
repository checkout and the root differs per cluster. User ruling 2026-09-15 (records
redesign, Q4 (b) and Q7).
"""
import os
from pathlib import Path

from . import property as prop

NO_RECORD = "NO RECORD"

#: (column, width, alignment). The record column is last of the common set and
#: left-aligned; anything a driver adds comes after it.
COMMON = (("species", 20, "<"), ("basin", 5, ">"), ("seed", 4, ">"), ("rc", 3, ">"),
          ("seconds", 9, ">"), ("STATUS", 18, "<"), ("record", 0, "<"))


def status_after(rc, property_path):
    """STATUS for the table, from the Property file the Calculation should have left."""
    st = prop.status_of(property_path) if property_path else None
    if st:
        return st
    return prop.FAILED if rc not in (0, None) else NO_RECORD


def _cell(v, width, align, fmt=None):
    if v is None:
        s = "-"
    elif fmt and isinstance(v, (int, float)) and not isinstance(v, bool):
        s = format(v, fmt)
    else:
        s = str(v)
    return "{:{a}{w}}".format(s, a=align, w=width) if width else s


def header(extra=(), record_width=0):
    """`extra` is a tuple of (name, width, align) for the driver's own columns;
    `record_width` pads the path column so the driver's columns align under their names."""
    cols = [_cell(name, w, a) for name, w, a in COMMON[:-1]]
    cols.append(_cell("record", record_width, "<"))
    for name, w, a in extra:
        cols.append(_cell(name, w, a))
    return "  ".join(cols).rstrip()


def line(r, extra=(), record_width=0):
    """One Calculation. `r` carries species, basin, seed, rc, seconds, status, record
    and whatever `extra` names; a missing value prints as '-'."""
    cols = [_cell(r.get("species"), 20, "<"), _cell(r.get("basin"), 5, ">"),
            _cell(r.get("seed"), 4, ">"), _cell(r.get("rc"), 3, ">"),
            _cell(r.get("seconds"), 9, ">", ".1f"), _cell(r.get("status"), 18, "<"),
            _cell(r.get("record") or "-", record_width, "<")]
    for name, w, a in extra:
        cols.append(_cell(r.get(name), w, a))
    return "  ".join(cols).rstrip()


def footer(wall_seconds, n_ok, n_total, what="passed every criterion"):
    """The closing lines: the batch wall, the count, the Slurm job id (read, never chosen)."""
    return ["batch wall {:.1f} s   {}/{} {}".format(wall_seconds, n_ok, n_total, what),
            "slurm job {}".format(os.environ.get("SLURM_JOB_ID") or "(none: not under Slurm)")]


def print_table(results, extra=(), out=print):
    width = max([len("record")] + [len(str(r.get("record") or "-")) for r in results]) if extra else 0
    out(header(extra, width))
    for r in results:
        out(line(r, extra, width))
        if r.get("error_line"):
            out("{:>20}  {}".format("", str(r["error_line"])[:160]))


def absolute(path):
    return str(Path(path).resolve()) if path else None
