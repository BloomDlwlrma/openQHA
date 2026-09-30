"""Which mace is imported, and which registered weight files are present.

TOOLING. Produces no scientific number.

    python scripts/tooling/s0_check_weights.py                         # every registered engine whose file resolves
    python scripts/tooling/s0_check_weights.py MACE-OFF23_medium       # one, by name

The identity of a potential is the registered engine name and the file that name resolves
to. The file hash, the parameter fingerprint and the registry pin are retired; this tool
reads no byte of any weight file: it prints which mace is imported -- version, module
path, fork commit and dirty flag -- and, per registered engine, the path the name
resolves to and the file's size. Nothing else.
"""
import argparse
import sys
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))
from openqha.potentials import engine            # noqa: E402


def print_weights(name):
    """A registered engine: the file its name resolves to, and how big that file is."""
    p = engine.model_path(name)
    print("{}".format(name))
    print("  path            {}".format(p))
    print("  bytes           {}".format(p.stat().st_size))


def print_mace():
    """Two lines that settle which mace is imported: its version and the fork commit.
    'unknown' means a wheel, not the fork."""
    try:
        import mace
        ver = mace.__version__
    except Exception as exc:                                       # noqa: BLE001
        print("mace            not importable: {}".format(exc)); return
    info = engine.mace_fork_info()
    print("mace            {}   {}".format(ver, getattr(mace, "__file__", "")))
    print("mace fork       {}{}   (intended: {} on {})".format(
        info["mace_fork_commit"], "  DIRTY" if info["mace_fork_dirty"] else "",
        engine.MACE_FORK, engine.MACE_FORK_BASE))
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="registered engine names (default: every one whose file is present)")
    args = ap.parse_args()

    print_mace()
    names = args.names or [n for n in sorted(engine.ENGINES)
                           if (engine.model_root() / engine.ENGINES[n]["filename"]).is_file()]
    if not names:
        print("no registered weight file is present under {}".format(engine.model_root()))
        print("(copy one there, or point S0_MACE_ROOT at the directory that holds them)")
        return
    for n in names:
        print_weights(n)


if __name__ == "__main__":
    main()
