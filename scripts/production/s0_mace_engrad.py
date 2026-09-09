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

Usage (called by CREST; not normally run by hand)::

    s0_mace_engrad.py genericinp.xyz
"""
import os
import socket
import sys

# CODATA 2018, the same set openqha.thermo uses
EV_PER_HARTREE = 27.211386245988
BOHR_PER_ANGSTROM = 1.0 / 0.529177210903

SOCKET = os.environ.get("S0_MACE_SOCKET", "/tmp/s0_mace_engrad.sock")


def die(msg):
    sys.stderr.write("s0_mace_engrad: {}\n".format(msg))
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        die("usage: s0_mace_engrad.py <xyz>")
    xyz = sys.argv[1]
    try:
        with open(xyz, "r") as fh:
            lines = fh.read().split("\n")
    except OSError as exc:
        die("cannot read {}: {}".format(xyz, exc))
    try:
        n = int(lines[0].split()[0])
    except (IndexError, ValueError):
        die("the first line of {} is not an atom count".format(xyz))
    body = []
    for line in lines[2:2 + n]:
        f = line.split()
        if len(f) < 4:
            die("could not parse a coordinate line: {!r}".format(line))
        body.append("{} {} {} {}".format(f[0], f[1], f[2], f[3]))
    if len(body) != n:
        die("{} atoms declared, only {} line(s) read".format(n, len(body)))

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
            "`python -m openqha.potentials.mace_server --socket {}`".format(SOCKET, exc, SOCKET))

    out = buf.decode("utf-8").strip().split("\n")
    if not out or out[0].startswith("ERROR"):
        die("server error: {}".format(out[0] if out else "(empty reply)"))
    if len(out) < n + 2:
        die("incomplete reply: {} line(s), expected {}".format(len(out), n + 2))

    energy_eh = float(out[0]) / EV_PER_HARTREE
    grad = []
    for line in out[1:1 + n]:
        fx, fy, fz = (float(v) for v in line.split())
        # force (eV/Angstrom) -> gradient (Eh/a0): negate, and convert units
        grad.append([-fx / EV_PER_HARTREE / BOHR_PER_ANGSTROM,
                     -fy / EV_PER_HARTREE / BOHR_PER_ANGSTROM,
                     -fz / EV_PER_HARTREE / BOHR_PER_ANGSTROM])

    stem = xyz[:-4] if xyz.endswith(".xyz") else xyz
    with open(stem + ".engrad", "w") as fh:
        fh.write("#\n# Atoms\n#\n{:5d}\n".format(n))
        fh.write("#\n# Energy ( Eh )\n#\n{:25.15f}\n".format(energy_eh))
        fh.write("#\n# Gradient ( Eh/a0 )\n#\n")
        for row in grad:
            for v in row:
                fh.write("{:25.15f}\n".format(v))


if __name__ == "__main__":
    main()
