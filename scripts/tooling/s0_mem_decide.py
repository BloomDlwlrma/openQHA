"""Allocate a decision identifier **per branch** and append the decision row -- the mechanical cure for colliding numbers.

TOOLING. Allocates decision identifiers per branch and appends the decision line.
Produces no science.

**The problem**: a single flat sequence `D0-1 ... D0-N` hands two sessions writing at
once the same number, and each insert in the middle forces the numbers after it to be
changed.

**Why "remember to check first next time" is not a cure**: there is a window between the
check and the write, and another session writing inside that window collides anyway. This
is a race, not carelessness -- **it has to be solved by a mechanism, not by discipline.**

**The mechanism has two layers, and neither alone is enough**:

1. **Branch by package into separate files.** Package 1 decisions go to `decisions_pkg1.md`,
   package 2 to `decisions_pkg2.md`, common constraints to `decisions_common.md`. Two sessions
   working on different packages **never touch the same file**, so a collision goes from "avoided by checking" to "physically impossible".
2. **A lock within one branch.** When two sessions really do edit one package at once, an
   `O_CREAT|O_EXCL` lock file makes "read the highest number, then append" atomic. The lock
   carries a holder and a timestamp and a stale one can be seized, so a crash leaves no permanent deadlock.

Branch prefixes::

    D0-C-n    common constraints (governance, versions, hardware, scope and standards across packages)
    D0-L-n    lecture notes s0-1
    D0-P1-n   package 1  conformer generation and deduplication
    D0-P2-n   package 2  Hessian / energies and forces / reference level
    D0-P3-n   package 3  molecular dynamics
    D0-P4-n   package 4  density of states and low frequencies (hindered rotors, MS-T)
    D0-P5-n   package 5  the electronic term

**The trunk `D0-1 ... D0-85` is frozen**: it no longer grows and is **not renumbered** --
26 files cite those numbers, renumbering would only create broken links, and **renumbering
does not reduce collisions anyway** (a collision happens only at allocation). Which branch a
historical row belongs to is recorded in the decision log.

Usage::

    python scripts/tooling/s0_mem_decide.py --branch P2 \
        --decision "package 2 E2 uses the MP2 increment ladder ..." \
        --basis "a measured pool of 4 single points ..."

    python scripts/tooling/s0_mem_decide.py --list P2        # show the numbers a branch already has
    python scripts/tooling/s0_mem_decide.py --next P2        # show the next number only; write nothing
"""
import argparse
import datetime as _dt
import errno
import os
import re
import sys
import time
from pathlib import Path

def _repo_root():
    """Directory holding the openQHA package, found by walking up from this file.

    Depth-independent on purpose: this file keeps working wherever under the
    repository it is moved to. The earlier move into `scripts/_superseded/` broke
    every `parents[1]` in the moved files silently, which is what this replaces.
    """
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))

from openqha import S0_ROOT

MEM = S0_ROOT / ".mem" / "decisions"
LOCK_STALE_SECONDS = 300        # past this the holder is assumed to have crashed and the lock may be seized

# Each entry is (identifier prefix, file stem, description).
#
# A second identifier space was introduced on 2026-09-03. The old
# `D0-*` branches are FROZEN -- they are listed here so `--all` can still report
# them, but nothing new is ever appended to them. New decisions go to the `S0-*`
# branches below.
#
# The prefix letter had to change rather than the branch letter: the old space
# already contains `D0-C-*` (common constraints), so a new branch C for the
# Hessian work would have collided and destroyed the lookup for both.
BRANCHES = {
    # --- frozen (D0- space) ----------------------------------------------------
    "C": ("D0", "common", "common constraints (governance, versions, hardware, scope, standards across packages)"),
    "L": ("D0", "lecture", "lecture notes s0-1"),
    "P1": ("D0", "pkg1", "package 1  conformer generation and deduplication"),
    "P2": ("D0", "pkg2", "package 2  Hessian / energies and forces / reference level"),
    "P3": ("D0", "pkg3", "package 3  molecular dynamics"),
    "P4": ("D0", "pkg4", "package 4  density of states and low frequencies (hindered rotors, MS-T)"),
    "P5": ("D0", "pkg5", "package 5  the electronic term"),
    # --- active (S0- space, 2026-09-03 rewrite) ---------------------------------
    "S0-G": ("S0", "G_governance", "governance, versions, hardware, scope, standards -- constraints common to every branch"),
    "S0-A": ("S0", "A_conformer", "branch A  conformer search (CREST GFN2-xTB + MACE refine=opt)"),
    "S0-B": ("S0", "B_qha", "branch B  quasi-harmonic analysis"),
    "S0-C": ("S0", "C_hessian", "branch C  supervised Hessian training and the RI-MP2/RIJK/cc-pVTZ benchmark"),
    "S0-D": ("S0", "D_repo", "branch D  repository restructuring and translating the code into English"),
}

#: Frozen branches refuse new rows. Reading and `--all` still work.
FROZEN = {"C", "L", "P1", "P2", "P3", "P4", "P5"}

ROW = re.compile(r"^\|\s*\*\*([DS]0)-([A-Z0-9]+)-(\d+)\*\*\s*\|")


def branch_prefix(branch):
    """Identifier prefix for a branch: "D0" (frozen) or "S0" (active)."""
    if branch not in BRANCHES:
        raise KeyError("no branch {!r}; available: {}".format(branch, sorted(BRANCHES)))
    return BRANCHES[branch][0]


def branch_letter(branch):
    """The letter that appears in the identifier, i.e. the part after the prefix.

    For the frozen branches the key IS the letter ("P1" -> D0-P1-n). For the new
    ones the key carries the prefix so that keys stay unique across both spaces
    ("S0-A" -> S0-A-n).
    """
    return branch.split("-", 1)[1] if branch.startswith("S0-") else branch


def branch_file(branch):
    if branch not in BRANCHES:
        raise KeyError("no branch {!r}; available: {}".format(branch, sorted(BRANCHES)))
    return MEM / "decisions_{}.md".format(BRANCHES[branch][1])


def existing_numbers(branch):
    p = branch_file(branch)
    if not p.exists():
        return []
    prefix = branch_prefix(branch)
    letter = branch_letter(branch)
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        m = ROW.match(line)
        if m and m.group(1) == prefix and m.group(2) == letter:
            out.append(int(m.group(3)))
    return sorted(out)


def next_number(branch):
    n = existing_numbers(branch)
    return (max(n) + 1) if n else 1


class Lock:
    """An `O_CREAT|O_EXCL` lock file. A stale lock may be seized, so a crash leaves no deadlock."""

    def __init__(self, branch, timeout=60.0):
        self.path = MEM / ".lock_{}".format(branch)
        self.timeout = timeout
        self.fd = None

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                self.fd = os.open(str(self.path),
                                  os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, "{} {}\n".format(os.getpid(), time.time())
                         .encode("utf-8"))
                return self
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                age = time.time() - self.path.stat().st_mtime
                if age > LOCK_STALE_SECONDS:
                    print("the lock is {:.0f} s stale (limit {}); seizing it.".format(
                        age, LOCK_STALE_SECONDS))
                    self.path.unlink(missing_ok=True)
                    continue
                if time.time() - t0 > self.timeout:
                    raise RuntimeError(
                        "timed out after {:.0f} s waiting for the lock: {} is held by another session for {:.0f} s. "
                        "It is not stale, so it is not seized -- wait for that session to finish.".format(
                            self.timeout, self.path, age))
                time.sleep(0.2)

    def __exit__(self, *a):
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)
        return False


def append(branch, decision, basis, date=None):
    """Atomically allocate the next number and append one row. Returns the identifier allocated."""
    MEM.mkdir(parents=True, exist_ok=True)
    if branch in FROZEN:
        raise ValueError(
            "branch {!r} is frozen: its identifier space no longer grows.\n"
            "Its rows live in the archive under _to_delete/stage0-discard/decisions/.\n"
            "Append to one of the active branches instead: {}".format(
                branch, sorted(b for b in BRANCHES if b not in FROZEN)))
    date = date or _dt.date.today().isoformat()
    with Lock(branch):
        n = next_number(branch)
        ident = "{}-{}-{}".format(branch_prefix(branch), branch_letter(branch), n)
        p = branch_file(branch)
        if not p.exists():
            p.write_text(header(branch), encoding="utf-8")
        row = "| **{}** | {} | {} | {} |\n".format(
            ident, date, decision.replace("|", "\\|"), basis.replace("|", "\\|"))
        with open(str(p), "a", encoding="utf-8") as fh:
            fh.write(row)
    return ident


def header(branch):
    prefix, _name, desc = BRANCHES[branch]
    ident = "{}-{}".format(prefix, branch_letter(branch))
    return (
        "# stage 0 decision log -- branch `{ident}` ({desc})\n\n"
        "> **Why branches**: a flat sequence collides when two sessions run in parallel (twice here already). "
        "Once split by branch, work on different branches **never touches the same file**; "
        "concurrency within one branch is then made atomic by the lock in `scripts/tooling/s0_mem_decide.py`.\n"
        ">\n"
        "> **Append only with `scripts/tooling/s0_mem_decide.py --branch {b}`; do not hand-write a row** -- "
        "hand-writing bypasses the lock and the collisions come back.\n"
        ">\n"
        "> The old `D0-1 ... D0-85` and `D0-C/L/P1...P5` are frozen, "
        "no longer growing and not renumbered.\n\n"
        "| identifier | date | decision | basis / where it lands |\n"
        "|---|---|---|---|\n".format(ident=ident, b=branch, desc=desc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch",
                    help="active: S0-G / S0-A / S0-B / S0-C / S0-D | "
                         "frozen (read only): C / L / P1 / P2 / P3 / P4 / P5")
    ap.add_argument("--decision", help="the decision itself, stated in one sentence")
    ap.add_argument("--basis", help="basis / where it lands")
    ap.add_argument("--date", default=None)
    ap.add_argument("--list", dest="list_branch", default=None)
    ap.add_argument("--next", dest="next_branch", default=None)
    ap.add_argument("--all", action="store_true", help="list the current state of every branch")
    args = ap.parse_args()

    if args.all:
        for b in BRANCHES:
            n = existing_numbers(b)
            state = "FROZEN" if b in FROZEN else "active"
            ident = "{}-{}".format(branch_prefix(b), branch_letter(b))
            nxt = "--" if b in FROZEN else "{}-{}".format(ident, next_number(b))
            print("{:8s}  {:14s}  {:6s}  {} row(s)  next {}".format(
                ident, BRANCHES[b][1], state, len(n), nxt))
        return
    if args.list_branch:
        b = args.list_branch
        for n in existing_numbers(b):
            print("{}-{}-{}".format(branch_prefix(b), branch_letter(b), n))
        return
    if args.next_branch:
        b = args.next_branch
        print("{}-{}-{}".format(branch_prefix(b), branch_letter(b), next_number(b)))
        return
    if not (args.branch and args.decision and args.basis):
        ap.error("give --all/--list/--next, or give --branch --decision --basis together")
    print(append(args.branch, args.decision, args.basis, args.date))


if __name__ == "__main__":
    main()
