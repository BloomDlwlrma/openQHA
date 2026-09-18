"""Ticket 01 of the Hessian-learning set: the parameter fingerprint is an identity of
the NUMBERS, not of the file.

Asserted on a tiny synthetic state_dict (no MACE weights needed): the digest survives a
torch.save / torch.load round trip and a change of key order; it changes when one
element, one dtype or one shape changes; `parameter_fingerprint(path=)` reads both a
plain state_dict file and a pickled module; the registry pin of the production default
is present and 64 hex characters; `s0_check_weights.py --pin` prints a pasteable entry.
The real-weights check (79 tensors, pin matches) runs only when the file is present.
"""
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import torch


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from openqha.potentials import engine            # noqa: E402

FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 2)
        self.register_buffer("scale", torch.tensor([1.5, -2.0], dtype=torch.float64))


def main():
    torch.manual_seed(0)
    sd = OrderedDict([("b.weight", torch.randn(4, 3, dtype=torch.float64)),
                      ("a.bias", torch.randn(4, dtype=torch.float32)),
                      ("c.count", torch.tensor([7], dtype=torch.int64))])
    fp = engine.fingerprint_state_dict(sd)
    check("a fingerprint is 64 hex characters, counts 3 tensors and their bytes",
          len(fp["params_sha256"]) == 64 and fp["n_tensors"] == 3
          and fp["params_bytes"] == 4 * 3 * 8 + 4 * 4 + 8, fp)
    shuffled = OrderedDict(reversed(list(sd.items())))
    check("key order does not matter", engine.fingerprint_state_dict(shuffled)["params_sha256"] == fp["params_sha256"])
    with tempfile.TemporaryDirectory(prefix="fp_") as tmp:
        f1, f2 = Path(tmp) / "a.pt", Path(tmp) / "b.pt"
        torch.save(sd, f1)
        torch.save(torch.load(f1, weights_only=False), f2)          # a re-serialised copy
        p1 = engine.parameter_fingerprint(path=f1)["params_sha256"]
        p2 = engine.parameter_fingerprint(path=f2)["params_sha256"]
        check("save/load round trip: the same numbers give the same fingerprint through parameter_fingerprint(path=)",
              p1 == fp["params_sha256"] and p2 == fp["params_sha256"], (p1[:12], p2[:12]))
        m = Tiny()
        torch.save(m, Path(tmp) / "mod.pt")
        fm = engine.parameter_fingerprint(path=Path(tmp) / "mod.pt")
        check("a pickled module is fingerprinted through its state_dict (3 tensors: weight, bias, buffer)",
              fm["n_tensors"] == 3 and fm["params_sha256"] == engine.fingerprint_state_dict(m.state_dict())["params_sha256"], fm)
        out = subprocess.run([sys.executable, str(ROOT / "scripts/tooling/s0_check_weights.py"), "--pin", str(f1)],
                             capture_output=True, text=True)
        check("s0_check_weights.py --pin prints a pasteable registry entry with the fingerprint",
              out.returncode == 0 and 'params_sha256="{}"'.format(fp["params_sha256"]) in out.stdout
              and 'filename="a.pt"' in out.stdout, out.stdout[-300:] + out.stderr[-300:])
    changed = OrderedDict(sd); changed["b.weight"] = sd["b.weight"].clone(); changed["b.weight"][0, 0] += 1e-9
    check("one element changed by 1e-9 changes the fingerprint",
          engine.fingerprint_state_dict(changed)["params_sha256"] != fp["params_sha256"])
    dtyped = OrderedDict(sd); dtyped["a.bias"] = sd["a.bias"].to(torch.float64)
    check("a dtype change alone changes it", engine.fingerprint_state_dict(dtyped)["params_sha256"] != fp["params_sha256"])
    shaped = OrderedDict(sd); shaped["b.weight"] = sd["b.weight"].reshape(3, 4)
    check("a shape change alone (same bytes) changes it", engine.fingerprint_state_dict(shaped)["params_sha256"] != fp["params_sha256"])

    pin = engine.ENGINES[engine.DEFAULT_ENGINE].get("params_sha256")
    check("the production default carries a 64-hex registry pin", isinstance(pin, str) and len(pin) == 64, pin)
    wf = engine.model_root() / engine.ENGINES[engine.DEFAULT_ENGINE]["filename"]
    if wf.is_file():
        real = engine.parameter_fingerprint(path=wf)
        check("real weights: 79 tensors and the pin matches (integration)",
              real["n_tensors"] == 79 and real["params_sha256"] == pin, (real["n_tensors"], real["params_sha256"][:12]))
        prov = engine.provenance(engine.DEFAULT_ENGINE)
        check("provenance() carries params_sha256, n_tensors and params_pin_status = matches",
              prov["params_sha256"] == pin and prov["n_tensors"] == 79 and prov["params_pin_status"] == "matches", prov.get("params_pin_status"))
    else:
        print("  (weights absent: the real-file checks are skipped)")
    if FAIL:
        print("FAIL: " + ", ".join(FAIL))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
