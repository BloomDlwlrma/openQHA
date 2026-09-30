"""The 04_dataset.py CLI contract.

Three things the CLI must promise: a real run reaches
`build` at all -- the call passes only keywords the parser defines, so it does not
raise AttributeError before the build --, `--split-by`
defaults to `molecule`, the production granularity (the other mode is `frame`, the
smoke / fit mode), and the print step states the production Replay in its new form:
the 30,000-frame draw written twice (weights 1 and 10, one seed), with the R4 recipe
readable only as the S0 replay ladder's historical scan row.

The script is loaded as a module (`__main__` guard keeps it inert); `openqha.config` is
stubbed. The first phase's `dataset.build` records its keywords and stops `main()`
before anything prints; the second phase's `build` returns a minimal Record so `main()`
runs to the end and its stdout is captured. `--tag draw300` is the production
invocation.
"""
import contextlib
import importlib.util
import io
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
sys.path.insert(0, str(ROOT / "tests" / "unit"))
from t_frames import check, FAIL                                        # noqa: E402

SCRIPT = ROOT / "workflows" / "hessian_learning" / "04_dataset.py"


class _Stop(Exception):
    pass


def _run_cli(argv):
    """Load `04_dataset.py`, run `main()` on `argv` against stubs, return build's keywords."""
    spec = importlib.util.spec_from_file_location("wf_04_dataset_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    calls = {}

    def fake_build(root, tags, name, **kw):
        calls.update(kw)
        calls["root"], calls["tags"], calls["name"] = root, tags, name
        raise _Stop

    with tempfile.TemporaryDirectory(prefix="t_dataset_cli_") as tmp:
        real_load, real_runs = mod.config.load, mod.config.runs_root
        real_build, real_argv = mod.dataset.build, sys.argv
        try:
            mod.config.load = lambda: {}
            mod.config.runs_root = lambda cfg: Path(tmp)
            mod.dataset.build = fake_build
            sys.argv = ["04_dataset.py"] + list(argv)
            try:
                mod.main()
            except _Stop:
                pass
            else:
                raise AssertionError("main() returned without calling build")
        finally:
            mod.config.load, mod.config.runs_root = real_load, real_runs
            mod.dataset.build, sys.argv = real_build, real_argv
    return calls


#: the minimal Record the print phase runs on: every field the print step reads
_INFO = dict(NAME="t", LEVEL="wb97m-d3bj_def2-tzvppd", SPLIT_BY="molecule",
             N_MOLECULES=2, N_TEST_MOLECULES=1, N_FRAMES=30, N_TRAIN=12, N_VALID=3,
             N_TEST=3, N_POOL=12, N_HESSIAN_FRAMES=18, N_TRAIN_HESSIAN=12,
             REPLAY_R4_FRAMES=48, TRAIN_GENERATORS=["basin"], N_TRAIN_BASIN=12,
             HELD_OUT_GENERATORS=["displaced"], N_TEST_HELD_OUT=0, N_LABELLED=18,
             MERGED_FILE="-")


def _run_print(argv):
    """Load `04_dataset.py`, stub `build` to return the minimal Record, run `main()` to
    the end and capture its stdout; returns (the module, the printed text)."""
    spec = importlib.util.spec_from_file_location("wf_04_dataset_print", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = {"info": dict(_INFO), "molecules": [], "classes": [], "dir": Path(".")}
    buf = io.StringIO()
    with tempfile.TemporaryDirectory(prefix="t_dataset_cli_print_") as tmp:
        real_load, real_runs = mod.config.load, mod.config.runs_root
        real_build, real_argv = mod.dataset.build, sys.argv
        try:
            mod.config.load = lambda: {}
            mod.config.runs_root = lambda cfg: Path(tmp)
            mod.dataset.build = lambda *a, **kw: out
            sys.argv = ["04_dataset.py"] + list(argv)
            with contextlib.redirect_stdout(buf):
                mod.main()
        finally:
            mod.config.load, mod.config.runs_root = real_load, real_runs
            mod.dataset.build, sys.argv = real_build, real_argv
    return mod, buf.getvalue()


def main():
    calls = _run_cli(["--tag", "draw300"])
    check("a bare CLI run reaches the build (no AttributeError) with --split-by molecule "
          "and no resplit keyword, and the Dataset name defaults to the first tag",
          calls.get("split_by") == "molecule" and "resplit" not in calls and calls.get("name") == "draw300"
          and calls.get("tags") == ["draw300"],
          {k: v for k, v in calls.items() if k != "root"})
    calls = _run_cli(["--tag", "draw300", "--name", "draw300_r1", "--split-by", "frame"])
    check("--name overrides the tag default and an explicit --split-by frame still reaches the build",
          calls.get("split_by") == "frame" and calls.get("name") == "draw300_r1", calls)
    mod, text = _run_print(["--tag", "draw300"])
    cmd1 = ("python scripts/tooling/s0_spice_pt_draw.py --n 30000 --seed 0 --weight 1 --out "
            "<root>/spice/spice_pt_replay30k_w1.extxyz")
    cmd10 = ("python scripts/tooling/s0_spice_pt_draw.py --n 30000 --seed 0 --weight 10 --out "
             "<root>/spice/spice_pt_replay30k_w10.extxyz")
    check("the print step carries the production Replay in its new form: the 30,000-frame draw, one seed, "
          "the two commands at weights 1 and 10 writing the w1 / w10 files",
          cmd1 in text and cmd10 in text, text[-500:])
    check("the production Replay standard is defined once in the data layer "
          "(30,000 frames, seed 0, weights 1 and 10)",
          mod.dataset.REPLAY_N_FRAMES == 30000 and mod.dataset.REPLAY_SEED == 0
          and mod.dataset.REPLAY_WEIGHTS == (1.0, 10.0),
          (mod.dataset.REPLAY_N_FRAMES, mod.dataset.REPLAY_SEED, mod.dataset.REPLAY_WEIGHTS))
    check("no line reads as 'the production Replay = the R4 recipe': the R4 mention is the S0 replay ladder's "
          "scan row, marked historical, and the old spice_pt_R4 command is gone",
          "S0 replay ladder" in text and "not the production Replay" in text
          and "spice_pt_R4.extxyz" not in text, text[-500:])
    print("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
