"""Say what a weight file actually is, and which kind of hash mismatch you have.

TOOLING. Classifies a weight file against the registry pin. Produces no science.

    python scripts/tooling/s0_check_weights.py            # this machine's file
    python scripts/tooling/s0_check_weights.py --pin      # print the registry lines

WHY THIS EXISTS
---------------
A file SHA-256 answers "are these the same bytes". The question the pin is asking is
"are these the same numbers", and the two are not the same question.

Measured 2026-09-09 on this repository's own `MACE-OFF23_medium.model`: a `torch.save` /
`torch.load` round trip of that exact model changed the file SHA-256 **and the file size**
-- 18 350 596 -> 18 367 938 bytes -- while all 79 tensors stayed bit-identical.

So a file hash cannot tell these four apart, and they do not deserve the same answer:

    truncated transfer            refuse -- and the size says so immediately
    re-serialised, same numbers   proceed -- it IS the same potential
    genuinely different weights   refuse -- this is what the pin is for
    same file, different path     fine

Run this on both machines and compare the three lines it prints. That decides it in one
command instead of an argument.
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


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))

from openqha.potentials import engine                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default=None, help="registry name; default is S0_ENGINE")
    ap.add_argument("--path", default=None, help="check this file instead of the "
                                                 "registry's")
    ap.add_argument("--pin", action="store_true",
                    help="print the registry lines to paste into openqha/potentials/"
                         "engine.py after deciding the file is right")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    name = args.engine or engine.engine_name()
    entry = engine.ENGINES[name]
    p = Path(args.path) if args.path else engine.model_path(name)
    if not p.is_file():
        raise SystemExit("not a file: {}".format(p))

    size = p.stat().st_size
    file_sha = hashlib.sha256(p.read_bytes()).hexdigest()
    params_sha, n_tensors = engine.parameter_fingerprint(p, name)

    rec = dict(engine=name, path=str(p), size_bytes=size,
               file_sha256=file_sha, params_sha256=params_sha, n_tensors=n_tensors,
               pinned_file_sha256=entry.get("sha256"),
               pinned_params_sha256=entry.get("params_sha256"),
               pinned_size_bytes=entry.get("size_bytes"))

    if args.json:
        print(json.dumps(rec, indent=2))
        return 0

    print("engine        {}".format(name))
    print("path          {}".format(p))
    print("size          {} bytes{}".format(
        size, "" if not entry.get("size_bytes")
        else "   (pinned {})".format(entry["size_bytes"])))
    print("file sha256   {}".format(file_sha))
    print("              {}".format(
        "MATCHES the pin" if file_sha == entry.get("sha256")
        else "differs from the pinned {}".format(entry.get("sha256"))))
    print("params sha256 {}   ({} tensors)".format(params_sha, n_tensors))
    pinned_p = entry.get("params_sha256")
    if pinned_p:
        print("              {}".format(
            "MATCHES the pin -- same numbers" if params_sha == pinned_p
            else "DIFFERS from the pinned {} -- different model".format(pinned_p)))
    else:
        print("              no parameter fingerprint pinned for this engine yet")

    print()
    if file_sha == entry.get("sha256"):
        print("VERDICT: identical file. Nothing to decide.")
    elif pinned_p and params_sha == pinned_p:
        print("VERDICT: **the same potential in a different container.** The numbers are")
        print("         bit-identical; only the serialisation differs, which is what a")
        print("         different torch version does. Safe to run. Re-pin the file hash")
        print("         with --pin if you want the two to agree from now on.")
    elif entry.get("size_bytes") and size < entry["size_bytes"] * 0.9:
        print("VERDICT: **truncated.** {:.0%} of the pinned size -- re-copy it:".format(
            size / entry["size_bytes"]))
        print("           rsync -a --partial --progress data/potentials/ "
              "<host>:<repo>/data/potentials/")
    elif pinned_p:
        print("VERDICT: **a different model.** The parameters themselves differ, so this")
        print("         is not a packaging difference. Running it would change the level")
        print("         every downstream number claims (D0-4) without saying so. Get the")
        print("         same weights, or register this one as its own engine name.")
    else:
        print("VERDICT: cannot decide -- no parameter fingerprint is pinned. Run this on")
        print("         a machine whose file you trust, --pin it, then compare.")

    if args.pin:
        print()
        print("registry lines for openqha/potentials/engine.py, ENGINES[{!r}]:".format(name))
        print('        sha256="{}",'.format(file_sha))
        print('        params_sha256="{}",'.format(params_sha))
        print('        n_tensors={},'.format(n_tensors))
        print('        size_bytes={},'.format(size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
