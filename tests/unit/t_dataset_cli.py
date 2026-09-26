"""Ticket 08 of hessian-learn-framework (round-1 xyz): the 04_dataset.py CLI contract.

Two things the CLI must promise (both were broken on 2026-09-26): a real run reaches
`build` at all -- the call carried `resplit=args.resplit` while the parser never grew a
`--resplit`, so every invocation raised AttributeError before the build and
`hl_labels.slurm`'s tail call hid it (its exit code is not captured) -- and `--split-by`
defaults to `molecule`, S0-C-65's production granularity (the script said `frame`, the
smoke / fit mode, with its help text describing the two modes backwards).

The script is loaded as a module (`__main__` guard keeps it inert); `openqha.config` is
stubbed and `dataset.build` records its keywords and stops `main()` before anything
prints. `--tag draw300` is the ticket's own invocation.
"""
import importlib.util
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


def main():
    calls = _run_cli(["--tag", "draw300"])
    check("ticket 08: a bare CLI run reaches the build (no AttributeError) with --split-by molecule "
          "and no resplit keyword, and the Dataset name defaults to the first tag",
          calls.get("split_by") == "molecule" and "resplit" not in calls and calls.get("name") == "draw300"
          and calls.get("tags") == ["draw300"],
          {k: v for k, v in calls.items() if k != "root"})
    calls = _run_cli(["--tag", "draw300", "--name", "draw300_r1", "--split-by", "frame"])
    check("--name overrides the tag default and an explicit --split-by frame still reaches the build",
          calls.get("split_by") == "frame" and calls.get("name") == "draw300_r1", calls)
    print("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
