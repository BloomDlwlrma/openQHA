#!/usr/bin/env -S python -S
"""The CREST `method = "generic"` client -- turns one xyz into one `.engrad`.

PRODUCTION. Not a driver but a runtime component: CREST starts it once per gradient
call. Remove it and no CREST run with the composite calculator can finish.

**CREST starts this script once per gradient call**, so it imports only `os/socket/sys`
and touches neither numpy nor torch. Measured start-up 13.6 ms; the real force evaluation
happens in the resident server (`python -m openqha.potentials.mace_server`).

The convention on the CREST side (`src/calculator/generic_sc.f90`):
  * CREST writes `genericinp.xyz` into the calcspace, then runs
    `cd <calcspace> && <binary> genericinp.xyz [flags] > generic.out`
  * afterwards CREST reads `genericinp.engrad` in the same directory, in the xtb/ORCA
    `.engrad` format: energy in Eh, gradient in Eh/a0, one component per line, `#` begins a comment

**Every one of those statements is an assumption about a program we do not control**, and
on 2026-09-09 a Tianhe run produced 1814 conformers whose energies were all `NaN` -- which
is what CREST does when it cannot read the gradient file. **This script's stderr goes to
`generic.out` inside the calcspace, and CREST deletes the calcspace**, so that run left no
evidence at all. Hence `S0_MACE_TRACE` below: it writes OUTSIDE the calcspace, so the
record survives whatever CREST does to its own scratch.

Usage (called by CREST; not normally run by hand)::

    s0_mace_engrad.py genericinp.xyz
"""
import os
import socket
import sys
import time

# CODATA 2018, the same set openqha.thermo uses
EV_PER_HARTREE = 27.211386245988
BOHR_PER_ANGSTROM = 1.0 / 0.529177210903

SOCKET = os.environ.get("S0_MACE_SOCKET", "/tmp/s0_mace_engrad.sock")

#: Directory for the call trace, or empty for no tracing. One line per invocation,
#: appended. **Written outside the calcspace on purpose** -- see the module docstring.
TRACE_DIR = os.environ.get("S0_MACE_TRACE", "")

#: The name the CREST documentation says CREST writes ("the script should process the
#: coordinates that crest writes into a file `genericinp.xyz`", upstream input-file
#: documentation). Used ONLY when CREST passes no argument at all -- and when it is used,
#: the trace says so, because "CREST does not pass the filename" would itself be the
#: explanation for a run full of NaN.
DOCUMENTED_INPUT = "genericinp.xyz"


def trace(status, note="", argv1="", n=0, energy=""):
    """Append one line to the trace, or do nothing. **Never raises.**

    Tracing that can break the calculation it is watching is worse than no tracing, so
    every failure here is swallowed. One short line written with O_APPEND is atomic on
    Linux below PIPE_BUF, which is what makes this safe with CREST's parallel threads.
    """
    if not TRACE_DIR:
        return
    try:
        line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
            time.strftime("%H:%M:%S"), os.getpid(), status, n, energy,
            os.getcwd(), argv1, " ".join(str(note).split())[:200])
        fd = os.open(os.path.join(TRACE_DIR, "engrad_trace.log"),
                     os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode("utf-8", "replace"))
        finally:
            os.close(fd)
    except Exception:                                                     # noqa: BLE001
        pass


def die(msg, argv1=""):
    trace("FAIL", msg, argv1)
    sys.stderr.write("s0_mace_engrad: {}\n".format(msg))
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        # **Recorded, not silently absorbed.** If CREST invokes us with no argument then
        # every call this run has already failed at this line, which alone explains a
        # NaN ensemble. Falling back lets the rest of the run be observed in the same
        # submission instead of costing another queue wait -- and the trace says
        # ARGV_MISSING, so the fallback can never be mistaken for a working contract.
        if os.path.exists(DOCUMENTED_INPUT):
            xyz = DOCUMENTED_INPUT
            trace("ARGV_MISSING", "no argv[1]; fell back to " + DOCUMENTED_INPUT, xyz)
        else:
            die("usage: s0_mace_engrad.py <xyz>  (and no {} in {})".format(
                DOCUMENTED_INPUT, os.getcwd()))
    else:
        xyz = sys.argv[1]
    try:
        with open(xyz, "r") as fh:
            lines = fh.read().split("\n")
    except OSError as exc:
        die("cannot read {}: {}".format(xyz, exc), xyz)
    try:
        n = int(lines[0].split()[0])
    except (IndexError, ValueError):
        die("the first line of {} is not an atom count".format(xyz), xyz)
    body = []
    for line in lines[2:2 + n]:
        f = line.split()
        if len(f) < 4:
            die("could not parse a coordinate line: {!r}".format(line), xyz)
        body.append("{} {} {} {}".format(f[0], f[1], f[2], f[3]))
    if len(body) != n:
        die("{} atoms declared, only {} line(s) read".format(n, len(body)), xyz)

    req = "{}\n{}\nEND\n".format(n, "\n".join(body)).encode("utf-8")
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(600.0)
        s.connect(SOCKET)
        s.sendall(req)
        buf = b""
        while b"\nEND\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        s.close()
    except OSError as exc:
        die("cannot reach the resident server {} ({}) -- start it first with "
            "`python -m openqha.potentials.mace_server --socket {}`".format(
                SOCKET, exc, SOCKET), xyz)

    out = buf.decode("utf-8").strip().split("\n")
    if not out or out[0].startswith("ERROR"):
        die("server error: {}".format(out[0] if out else "(empty reply)"), xyz)
    if len(out) < n + 2:
        die("incomplete reply: {} line(s), expected {}".format(len(out), n + 2), xyz)

    energy_eh = float(out[0]) / EV_PER_HARTREE
    # A gradient file full of NaN is the one thing CREST accepts and cannot use: it
    # poisons every energy comparison downstream and CREGEN stops discarding anything.
    # Refuse here, where the trace can say why.
    if energy_eh != energy_eh or energy_eh in (float("inf"), float("-inf")):
        die("server returned a non-finite energy: {!r}".format(out[0]), xyz)
    grad = []
    for line in out[1:1 + n]:
        fx, fy, fz = (float(v) for v in line.split())
        # force (eV/Angstrom) -> gradient (Eh/a0): negate, and convert units
        grad.append([-fx / EV_PER_HARTREE / BOHR_PER_ANGSTROM,
                     -fy / EV_PER_HARTREE / BOHR_PER_ANGSTROM,
                     -fz / EV_PER_HARTREE / BOHR_PER_ANGSTROM])

    stem = xyz[:-4] if xyz.endswith(".xyz") else xyz
    target = stem + ".engrad"
    with open(target, "w") as fh:
        fh.write("#\n# Atoms\n#\n{:5d}\n".format(n))
        fh.write("#\n# Energy ( Eh )\n#\n{:25.15f}\n".format(energy_eh))
        fh.write("#\n# Gradient ( Eh/a0 )\n#\n")
        for row in grad:
            for v in row:
                fh.write("{:25.15f}\n".format(v))
    trace("OK", target, xyz, n, "{:.6f}".format(energy_eh))


if __name__ == "__main__":
    main()
