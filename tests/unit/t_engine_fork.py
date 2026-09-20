"""Ticket 10 of the Hessian-learning set: `engine.mace_fork_info` names the mace
checkout that is imported, or says "unknown".

Asserted on a temporary directory tree with an injected git runner (no real git, no
real mace): a `mace/__init__.py` whose parent holds `.git` yields the 40-hex commit
and a dirty flag from `status --porcelain`; no `.git` beside the package (a wheel in
site-packages) yields "unknown"; a git runner that fails, or answers something that is
not a commit, yields "unknown"; a `.git` two levels up is NOT found (that would be
somebody's home repository). Then the real import: `provenance()`-style keys exist
and, when the installed mace is the fork, the commit is 40 hex and the path is a
checkout holding `mace/__version__.py`.
"""
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
COMMIT = "8fac5d1" + "0" * 33


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


def fake_git(answers):
    calls = []

    def run(args):
        calls.append(list(args))
        sub = args[2]                      # ["-C", root, "<subcommand>", ...]
        a = answers[sub]
        if isinstance(a, Exception):
            raise a
        return a
    run.calls = calls
    return run


def main():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        checkout = td / "openQHA-Hessian"
        (checkout / "mace").mkdir(parents=True)
        (checkout / "mace" / "__init__.py").write_text("")
        (checkout / ".git").mkdir()
        init = str(checkout / "mace" / "__init__.py")

        run = fake_git({"rev-parse": COMMIT + "\n", "status": ""})
        info = engine.mace_fork_info(module_file=init, run=run)
        check("editable checkout: commit is the 40-hex answer", info["mace_fork_commit"] == COMMIT, info)
        check("clean tree: dirty = False", info["mace_fork_dirty"] is False, info)
        check("path is the checkout", Path(info["mace_fork_path"]) == checkout.resolve(), info)
        check("git was asked -C <checkout> rev-parse and status --porcelain --untracked-files=no",
              [c[2] for c in run.calls] == ["rev-parse", "status"] and all(c[1] == str(checkout.resolve()) for c in run.calls)
              and "--untracked-files=no" in run.calls[1], run.calls)

        run = fake_git({"rev-parse": COMMIT + "\n", "status": " M mace/tools/train.py\n"})
        info = engine.mace_fork_info(module_file=init, run=run)
        check("a modified tracked file: dirty = True", info["mace_fork_dirty"] is True, info)

        run = fake_git({"rev-parse": "fatal: not a git repository\n", "status": ""})
        info = engine.mace_fork_info(module_file=init, run=run)
        check("a non-commit answer -> unknown", info["mace_fork_commit"] == "unknown" and info["mace_fork_dirty"] is None, info)

        run = fake_git({"rev-parse": RuntimeError("git not found"), "status": ""})
        info = engine.mace_fork_info(module_file=init, run=run)
        check("a failing git -> unknown", info["mace_fork_commit"] == "unknown", info)

        site = td / "env" / "lib" / "site-packages"
        (site / "mace").mkdir(parents=True)
        (site / "mace" / "__init__.py").write_text("")
        run = fake_git({"rev-parse": COMMIT + "\n", "status": ""})
        info = engine.mace_fork_info(module_file=str(site / "mace" / "__init__.py"), run=run)
        check("a wheel in site-packages (no .git beside the package) -> unknown, git never called",
              info["mace_fork_commit"] == "unknown" and info["mace_fork_path"] is None and run.calls == [], info)

        (td / "env" / ".git").mkdir()                  # a repository two levels up must not be mistaken for mace's
        info = engine.mace_fork_info(module_file=str(site / "mace" / "__init__.py"), run=run)
        check("a .git two levels up is not mace's -> unknown", info["mace_fork_commit"] == "unknown" and run.calls == [], info)

        info = engine.mace_fork_info(module_file="", run=run)
        check("no module file -> unknown", info["mace_fork_commit"] == "unknown")

    # the real import, whatever is installed
    try:
        import mace
        info = engine.mace_fork_info()
        check("real mace: the three keys present", set(info) == {"mace_fork_commit", "mace_fork_dirty", "mace_fork_path"}, info)
        if info["mace_fork_commit"] != "unknown":
            ok = len(info["mace_fork_commit"]) == 40 and (Path(info["mace_fork_path"]) / "mace" / "__version__.py").is_file()
            check("real mace is the fork: 40-hex commit and a checkout with mace/__version__.py ({}, dirty={})".format(
                info["mace_fork_commit"][:12], info["mace_fork_dirty"]), ok, info)
        else:
            print("  (installed mace {} at {} is not a git checkout: 'unknown' -- a wheel; the fork is {})".format(
                mace.__version__, getattr(mace, "__file__", "?"), engine.MACE_FORK))
    except ImportError:
        print("  (mace not importable here; the real-import check skipped)")

    print("\n{} checks, {} failed".format(11 + (1 if "mace" in sys.modules else 0), len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
