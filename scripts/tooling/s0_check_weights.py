"""Three lines that settle what a weight file is: bytes, file SHA-256, parameter
fingerprint.

TOOLING. Produces no scientific number.

    python scripts/tooling/s0_check_weights.py                         # every registered engine present
    python scripts/tooling/s0_check_weights.py MACE-OFF23_medium       # one, by name
    python scripts/tooling/s0_check_weights.py --pin path/to/new.model # the registry lines to paste
    python scripts/tooling/s0_check_weights.py --json out.json         # for a two-machine compare
    python scripts/tooling/s0_check_weights.py --compare a.json b.json

The file hash answers "the same bytes?"; the parameter fingerprint answers "the same
numbers?" (S0-G-73: a torch.save/load round trip changes the former and the byte count,
not the latter). A registry pin is compared against the fingerprint and the verdict is
one of: matches / differs / unpinned -- reported, never enforced here or in
`engine.provenance()`.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))
from openqha.potentials import engine            # noqa: E402


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def describe(path, name=None):
    p = Path(path)
    fp = engine.parameter_fingerprint(path=p)
    rec = dict(name=name, path=str(p), bytes=p.stat().st_size, file_sha256=file_sha256(p), **fp)
    if name is not None:
        pin = engine.ENGINES[name].get("params_sha256")
        rec["pin"] = pin
        rec["pin_status"] = "unpinned" if pin is None else ("matches" if pin == fp["params_sha256"] else "differs")
    return rec


def print_rec(rec):
    print("{}".format(rec["name"] or rec["path"]))
    print("  path            {}".format(rec["path"]))
    print("  bytes           {}".format(rec["bytes"]))
    print("  file sha256     {}".format(rec["file_sha256"]))
    print("  params sha256   {}   ({} tensors, {} bytes of parameters)".format(
        rec["params_sha256"], rec["n_tensors"], rec["params_bytes"]))
    if "pin_status" in rec:
        print("  registry pin    {}".format(rec["pin_status"] + ("" if rec["pin"] is None else "  " + rec["pin"][:16] + "...")))


def print_mace():
    """Two lines that settle which mace is imported: its version and the fork commit
    (ticket 10 of the Hessian-learning set). 'unknown' means a wheel, not the fork."""
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
    ap.add_argument("--pin", default=None, metavar="FILE", help="print the registry entry lines for this weight file")
    ap.add_argument("--json", default=None, metavar="OUT", help="also write the records as JSON")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"), help="diff two JSON records, by name")
    args = ap.parse_args()

    if args.compare:
        a, b = (json.load(open(f)) for f in args.compare)
        a, b = {r["name"] or r["path"]: r for r in a}, {r["name"] or r["path"]: r for r in b}
        bad = 0
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                print("{:24s} only in {}".format(k, "A" if k in a else "B")); bad += 1; continue
            same_bytes = a[k]["bytes"] == b[k]["bytes"]
            same_file = a[k]["file_sha256"] == b[k]["file_sha256"]
            same_params = a[k]["params_sha256"] == b[k]["params_sha256"]
            verdict = ("identical files" if same_file else
                       "same numbers, different container (re-serialised)" if same_params else
                       "DIFFERENT NUMBERS" + ("" if same_bytes else " (and different size)"))
            print("{:24s} {}".format(k, verdict))
            bad += not same_params
        sys.exit(1 if bad else 0)

    print_mace()
    if args.pin:
        rec = describe(args.pin)
        print_rec(rec)
        print("\nregistry entry (openqha/potentials/engine.py, ENGINES):")
        print('    "<name>": dict(')
        print('        filename="{}",'.format(Path(args.pin).name))
        print('        source="<Dataset index path> + <training config sha256>",')
        print('        note="<why this potential exists>",')
        print('        params_sha256="{}",   # {} tensors, {} bytes'.format(rec["params_sha256"], rec["n_tensors"], rec["bytes"]))
        print("    ),")
        return

    names = args.names or [n for n in sorted(engine.ENGINES) if (engine.model_root() / engine.ENGINES[n]["filename"]).is_file()]
    recs = []
    for n in names:
        rec = describe(engine.model_path(n), name=n)
        print_rec(rec)
        recs.append(rec)
    if args.json:
        json.dump(recs, open(args.json, "w"), indent=2)
        print("written {}".format(args.json))


if __name__ == "__main__":
    main()
