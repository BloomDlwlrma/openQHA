"""The identity of a potential is the REGISTERED NAME and the RESOLVED PATH.
Nothing reads the file's bytes.

The successor of the engine-fingerprint test, at the same seam (the engine's public API),
with the opposite contract. Asserted without any real weights -- the files under the
temporary root hold plain text, and nothing ever opens them:

  * a registered name resolves to `<root>/<filename>`;
  * a registry `filename` may be a relative sub-path
    (`mace_off23_<campaign>/<run>+<YYYYMMDD-HHMMSS>.model`, the storage convention
    for self-trained revisions);
  * `S0_MACE_MODEL` still overrides one file, and a non-file there raises naming it;
  * the missing-file error names the expected path and says what the root holds --
    top-level `*.model` files AND sub-directory names, so "wrong directory", "wrong
    sub-directory" and "wrong filename" are distinguishable;
  * `provenance()` carries exactly the kept keys (engine / source / note / weights_path /
    bytes / interface / mace and fork identity / dtype / patch state) and none of the
    retired four;
  * the retired machinery is gone from the module and no registry entry carries a pin;
  * the reduced `s0_check_weights.py` runs, prints the mace identity and the presence
    listing, and refuses the deleted flags.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from openqha.potentials import engine            # noqa: E402

FAIL = []
CLI = ROOT / "scripts" / "tooling" / "s0_check_weights.py"

#: The kept provenance keys, spelled out: a new one is a deliberate act that updates this
#: list, and the retired four (below) must not creep back.
KEPT_KEYS = {"engine", "source", "note", "weights_path", "bytes", "interface",
             "mace_torch_version", "mace_module_path", "mace_fork_commit", "mace_fork_dirty",
             "mace_fork_path", "torch_version", "dtype", "neighbour_list_patch"}
RETIRED_KEYS = {"params_sha256", "n_tensors", "params_bytes", "params_pin_status"}


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:300]))
    if not ok:
        FAIL.append(label)


def main():
    check("the fingerprint machinery is gone from the module (fingerprint_state_dict / parameter_fingerprint / _FINGERPRINTS)",
          not hasattr(engine, "fingerprint_state_dict") and not hasattr(engine, "parameter_fingerprint")
          and not hasattr(engine, "_FINGERPRINTS"))
    check("no registry entry carries a pin", all("params_sha256" not in e for e in engine.ENGINES.values()),
          [n for n, e in engine.ENGINES.items() if "params_sha256" in e])

    old_root, old_model = os.environ.get("S0_MACE_ROOT"), os.environ.get("S0_MACE_MODEL")
    with tempfile.TemporaryDirectory(prefix="engine_identity_") as tmp:
        tmp = Path(tmp)
        root = tmp / "potentials"
        (root / "mace_off23_draw300").mkdir(parents=True)
        flat = root / "MACE-OFF23_medium.model"
        flat.write_bytes(b"the name and the path are the identity; nothing opens this")   # never read
        sub_rel = "mace_off23_draw300/draw300-r4+20260927-101530.model"
        (root / sub_rel).write_bytes(b"nor this")
        try:
            os.environ["S0_MACE_ROOT"] = str(root)
            os.environ.pop("S0_MACE_MODEL", None)

            check("a registered name resolves to <root>/<filename>", engine.model_path("MACE-OFF23_medium") == flat)
            engine.ENGINES["TEST_revision"] = dict(filename=sub_rel, source="test", note="temporary")
            check("a registry filename may be a relative sub-path (mace_off23_<campaign>/<run>+<stamp>.model)",
                  engine.model_path("TEST_revision") == root / sub_rel)

            prov = engine.provenance("MACE-OFF23_medium")
            check("provenance(): the engine name and the resolved path are the identity (bytes off the file, nothing read)",
                  prov["engine"] == "MACE-OFF23_medium" and Path(prov["weights_path"]) == flat
                  and prov["bytes"] == flat.stat().st_size
                  and prov["source"] == engine.ENGINES["MACE-OFF23_medium"]["source"], prov)
            check("provenance() carries exactly the kept keys -- the retired four are absent",
                  set(prov) == KEPT_KEYS and not (RETIRED_KEYS & set(prov)),
                  sorted(set(prov) ^ KEPT_KEYS) + sorted(RETIRED_KEYS & set(prov)))

            try:
                engine.model_path("MACE-OFF23_small")
                check("a missing file raises FileNotFoundError", False)
            except FileNotFoundError as exc:
                msg = str(exc)
                check("the missing-file error names the expected path first, then what the root holds: "
                      "top-level *.model files AND sub-directory names",
                      str(root / "MACE-OFF23_small.model") in msg and "MACE-OFF23_medium.model" in msg
                      and "mace_off23_draw300" in msg
                      and msg.index("expected:") < msg.index("directory exists"), msg[-400:])

            (root / sub_rel).unlink()
            try:
                engine.model_path("TEST_revision")
                check("a missing file under a sub-directory names that sub-path (wrong sub-directory is distinguishable)",
                      False)
            except FileNotFoundError as exc:
                check("a missing file under a sub-directory names that sub-path (wrong sub-directory is distinguishable)",
                      sub_rel in str(exc), str(exc)[-300:])
            (root / sub_rel).write_bytes(b"back")

            os.environ["S0_MACE_MODEL"] = str(flat)
            check("S0_MACE_MODEL still overrides one file", engine.model_path("MACE-OFF23_small") == flat)
            os.environ["S0_MACE_MODEL"] = str(tmp / "nope.model")
            try:
                engine.model_path("MACE-OFF23_medium")
                check("a S0_MACE_MODEL that is not a file raises naming it", False)
            except FileNotFoundError as exc:
                check("a S0_MACE_MODEL that is not a file raises naming it",
                      str(tmp / "nope.model") in str(exc) and "S0_MACE_MODEL" in str(exc))
            os.environ.pop("S0_MACE_MODEL")
            engine.ENGINES.pop("TEST_revision", None)

            env = dict(os.environ, S0_MACE_ROOT=str(root))
            out = subprocess.run([sys.executable, str(CLI)], capture_output=True, text=True, env=env)
            check("the reduced tool runs, prints the mace identity (version / fork) and the presence listing, exits 0",
                  out.returncode == 0 and "mace fork" in out.stdout and "MACE-OFF23_medium.model" in out.stdout
                  and "bytes" in out.stdout, out.stdout[-400:] + out.stderr[-200:])
            gone = {flag: subprocess.run([sys.executable, str(CLI), flag, "x"], capture_output=True, text=True,
                                         env=env).returncode for flag in ("--pin", "--json", "--compare")}
            check("the deleted flags are refused (--pin / --json / --compare)", all(rc != 0 for rc in gone.values()), gone)
        finally:
            engine.ENGINES.pop("TEST_revision", None)
            if old_root is None:
                os.environ.pop("S0_MACE_ROOT", None)
            else:
                os.environ["S0_MACE_ROOT"] = old_root
            if old_model is None:
                os.environ.pop("S0_MACE_MODEL", None)
            else:
                os.environ["S0_MACE_MODEL"] = old_model

    if FAIL:
        print("FAIL: " + ", ".join(FAIL))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
