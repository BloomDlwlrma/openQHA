"""Attach MACE-OFF23-SC to CREST -- the resident server.

**Why this has to be a resident process**: CREST's `method = "generic"` is a
**subprocess** interface, and it invokes an external script once for every gradient it
wants. Loading a MACE model takes 5-10 seconds, and one conformer search makes
10^4-10^6 gradient calls -- reloading the model each time is out of the question. So:
**the server loads the model once and stays up, and the client is a minimal script that
imports nothing but socket** (measured start-up 13.6 ms, acceptable against MACE's own
25.8 ms for a single force evaluation).

Protocol (Unix domain socket, newline-delimited, plain text, so it can be driven by
hand with `nc -U`)::

    client -> server        server -> client
    N                       energy_eV
    sym x y z   (x N, A)    fx fy fz   (x N, eV/A)
    END                     END

On an error the server replies with a single line `ERROR <message>`, on which the client
exits non-zero, so that CREST sees `iostatus /= 0` and **aborts** rather than accepting
a gradient that quietly went wrong.

Usage::

    python -m openqha.potentials.mace_server --socket /tmp/s0_mace.sock
    python -m openqha.potentials.mace_server --socket /tmp/s0_mace.sock --stop
"""
import argparse
import os
import socket
import sys
import threading
import time
from pathlib import Path

DEFAULT_SOCKET = "/tmp/s0_mace_engrad.sock"


def _read_request(conn):
    buf = b""
    while b"\nEND\n" not in buf and not buf.endswith(b"\nEND"):
        chunk = conn.recv(65536)
        if not chunk:
            break
        buf += chunk
    lines = buf.decode("utf-8").strip().split("\n")
    if not lines or lines[-1].strip() != "END":
        raise ValueError("incomplete request (no END received)")
    n = int(lines[0])
    if len(lines) < n + 2:
        raise ValueError("request declares {} atoms but only {} coordinate lines "
                         "arrived".format(n, len(lines) - 2))
    syms, pos = [], []
    for line in lines[1:1 + n]:
        f = line.split()
        syms.append(f[0])
        pos.append([float(f[1]), float(f[2]), float(f[3])])
    return syms, pos


def serve(socket_path, device="cpu", quiet=False):
    from ase import Atoms
    from . import engine

    calc, name, prov = engine.calculator(device=device)
    if not quiet:
        print("[s0-mace-server] engine {} loaded, SHA-256 {}".format(
            name, prov["sha256"][:16]), flush=True)

    if os.path.exists(socket_path):
        os.unlink(socket_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(socket_path)
    srv.listen(16)
    os.chmod(socket_path, 0o600)
    if not quiet:
        print("[s0-mace-server] listening on {}".format(socket_path), flush=True)

    n_calls, t_sum, t0 = 0, 0.0, time.time()
    n_nonfinite = 0
    #: One line per gradient. Off by default -- 10^4-10^6 calls per search -- and turned
    #: on by S0_MACE_TRACE, the same switch the client uses, so one setting instruments
    #: both ends of the socket at once.
    verbose = bool(os.environ.get("S0_MACE_TRACE"))
    lock = threading.Lock()
    try:
        while True:
            conn, _ = srv.accept()
            try:
                syms, pos = _read_request(conn)
                if syms == ["STOP"]:
                    conn.sendall(b"OK\nEND\n")
                    break
                t1 = time.time()
                with lock:                      # one model instance, calls serialised
                    atoms = Atoms(symbols=syms, positions=pos)
                    atoms.calc = calc
                    e = float(atoms.get_potential_energy())
                    f = atoms.get_forces()
                dt = time.time() - t1
                n_calls += 1
                t_sum += dt
                # **The server's own count is half the evidence.** A NaN ensemble has
                # three possible sources -- the client never arrived, the client arrived
                # but its gradient file never reached CREST, or the model itself returned
                # NaN -- and comparing "gradients served" against "energy+grad calls"
                # in crest.out separates the first from the other two. Costs one line
                # per call in a log that is already kept with the job (2026-09-09).
                if e != e or e in (float("inf"), float("-inf")):
                    n_nonfinite += 1
                    print("[s0-mace-server] call {} n={} NON-FINITE energy {!r} -- "
                          "refusing, the client will abort".format(n_calls, len(syms), e),
                          flush=True)
                    conn.sendall(b"ERROR non-finite energy from the model\nEND\n")
                    continue
                if verbose:
                    print("[s0-mace-server] call {} n={} E={:.6f} eV {:.1f} ms".format(
                        n_calls, len(syms), e, 1000 * dt), flush=True)
                out = ["{:.12f}".format(e)]
                out += ["{:.12f} {:.12f} {:.12f}".format(*row) for row in f]
                out.append("END")
                conn.sendall(("\n".join(out) + "\n").encode("utf-8"))
            except Exception as exc:            # an error must reach CREST
                try:
                    conn.sendall(("ERROR {}: {}\nEND\n".format(
                        type(exc).__name__, " ".join(str(exc).split())[:200])
                    ).encode("utf-8"))
                except OSError:
                    pass
            finally:
                conn.close()
    finally:
        srv.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        if not quiet and n_calls:
            # **Always printed, tracing or not.** This total is the number to compare
            # against `Total number of energy+grad calls` in crest.out: if CREST says
            # 94674 and the server says 0, the client never reached it and nothing about
            # the model is implicated (2026-09-09, job 7347197).
            print("[s0-mace-server] {} gradients in all, {} non-finite, mean model time "
                  "{:.1f} ms, server up for {:.0f} s".format(
                      n_calls, n_nonfinite, 1000 * t_sum / n_calls, time.time() - t0),
                  flush=True)
        elif not quiet:
            print("[s0-mace-server] **0 gradients served** in {:.0f} s -- nothing ever "
                  "connected to {}".format(time.time() - t0, socket_path), flush=True)


def stop(socket_path):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(socket_path)
    s.sendall(b"1\nSTOP 0 0 0\nEND\n")
    s.recv(4096)
    s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default=os.environ.get("S0_MACE_SOCKET", DEFAULT_SOCKET))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.stop:
        stop(a.socket)
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    serve(a.socket, device=a.device, quiet=a.quiet)


if __name__ == "__main__":
    main()

# ======================================================================================
# In-process thread version -- for the "one worker per molecule" architecture
# ======================================================================================
def serve_in_thread(socket_path, calc, backlog=8):
    """Run a server on a background thread **in the current process**, reusing an
    **already loaded** calculator.

    Returns (thread, stop_callable).

    **Why this is needed**: under the new architecture there is one worker process per
    molecule, and that worker has already loaded a MACE model for the tightening and the
    Hessian. CREST is an external process and can only reach it over a socket. Starting a
    separate server process as well would put **two** copies of the model inside the same
    worker (each measured at 821 MB). Letting the server thread reuse the same calculator
    keeps it at one model per worker.

    **Thread safety**: model calls are serialised under a lock throughout. That costs no
    parallelism -- at any moment a worker is either running CREST or doing analysis, never
    both, and parallelism ACROSS molecules comes from **multiple processes**.
    """
    import threading

    from ase import Atoms

    socket_path = str(socket_path)
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(socket_path)
    srv.listen(backlog)
    os.chmod(socket_path, 0o600)
    srv.settimeout(0.5)          # so the thread can check the stop flag periodically

    stop_evt = threading.Event()
    lock = threading.Lock()
    stats = dict(n_calls=0, seconds=0.0)

    def loop():
        while not stop_evt.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                syms, pos = _read_request(conn)
                if syms == ["STOP"]:
                    conn.sendall(b"OK\nEND\n")
                    break
                t1 = time.time()
                with lock:
                    atoms = Atoms(symbols=syms, positions=pos)
                    atoms.calc = calc
                    e = float(atoms.get_potential_energy())
                    f = atoms.get_forces()
                stats["n_calls"] += 1
                stats["seconds"] += time.time() - t1
                out = ["{:.12f}".format(e)]
                out += ["{:.12f} {:.12f} {:.12f}".format(*row) for row in f]
                out.append("END")
                conn.sendall(("\n".join(out) + "\n").encode("utf-8"))
            except Exception as exc:          # an error must reach CREST
                try:
                    conn.sendall(("ERROR {}: {}\nEND\n".format(
                        type(exc).__name__, " ".join(str(exc).split())[:200])
                    ).encode("utf-8"))
                except OSError:
                    pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
        try:
            srv.close()
        except OSError:
            pass
        if os.path.exists(socket_path):
            try:
                os.unlink(socket_path)
            except OSError:
                pass

    th = threading.Thread(target=loop, daemon=True)
    th.start()

    def stop():
        stop_evt.set()
        th.join(timeout=10)
        return dict(stats)

    return th, stop
