"""`engine.mace_fork_info` names the mace
checkout that is imported, or says "unknown"; `engine.checkout_commit` is the shared
rule behind it, and the same cases run on the `openqha_hessian` package's
module file -- the commit its Records name.

Asserted on a temporary directory tree with an injected git runner (no real git, no
real mace): a `mace/__init__.py` whose parent holds `.git` yields the 40-hex commit
and a dirty flag from `status --porcelain`; no `.git` beside the package (a wheel in
site-packages) yields "unknown"; a git runner that fails, or answers something that is
not a commit, yields "unknown"; a `.git` two levels up is NOT found (that would be
somebody's home repository). Then the same on `openqha_hessian/__init__.py`: the commit
and the root, no dirty question asked, the same four "unknown"s. Then the real import:
`provenance()`-style keys exist and, when the installed mace is the fork, the commit is
40 hex and the path is a checkout holding `mace/__version__.py`; `package_identity()`
answers the distribution version and a commit-shaped field.
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
        checkout = td / "mace-checkout"
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

        # the same rules on the openqha_hessian package's module file: the commit
        # `package_identity` records; no dirty question is asked here
        pkg = checkout / "openqha_hessian"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        pkg_init = str(pkg / "__init__.py")
        run = fake_git({"rev-parse": COMMIT + "\n", "status": ""})
        commit, root = engine.checkout_commit(module_file=pkg_init, run=run)
        check("package read: an editable checkout answers the 40-hex commit and its root",
              commit == COMMIT and root == checkout.resolve(), (commit, root))
        check("package read: only rev-parse is asked (the dirty question is the fork guard's)",
              [c[2] for c in run.calls] == ["rev-parse"], run.calls)
        site2 = td / "env2" / "lib" / "site-packages"
        (site2 / "openqha_hessian").mkdir(parents=True)
        (site2 / "openqha_hessian" / "__init__.py").write_text("")
        run = fake_git({"rev-parse": COMMIT + "\n", "status": ""})
        check("package read: a wheel (no .git beside the package) answers unknown, git never called",
              engine.checkout_commit(module_file=str(site2 / "openqha_hessian" / "__init__.py"), run=run) == ("unknown", None)
              and run.calls == [], run.calls)
        run = fake_git({"rev-parse": RuntimeError("git not found"), "status": ""})
        check("package read: a failing git answers unknown",
              engine.checkout_commit(module_file=pkg_init, run=run)[0] == "unknown")
        run = fake_git({"rev-parse": "not a commit\n", "status": ""})
        check("package read: a non-commit answer answers unknown",
              engine.checkout_commit(module_file=pkg_init, run=run)[0] == "unknown")
        (td / "env2" / ".git").mkdir()
        run = fake_git({"rev-parse": COMMIT + "\n", "status": ""})
        check("package read: a .git further up (not beside the package) is not found",
              engine.checkout_commit(module_file=str(site2 / "openqha_hessian" / "__init__.py"), run=run)[0] == "unknown"
              and run.calls == [], run.calls)

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

    # the package's own read, whatever is installed: the two fields every new Record carries
    try:
        from openqha_hessian import package_identity
        ident = package_identity()
        check("real package: package_identity returns the version and a commit (40 hex or unknown)",
              isinstance(ident.get("HL_PACKAGE_VERSION"), str) and ident["HL_PACKAGE_VERSION"]
              and (ident["HL_PACKAGE_COMMIT"] == "unknown" or len(ident["HL_PACKAGE_COMMIT"]) == 40), ident)
    except ImportError:
        print("  (openqha_hessian not importable here; the package-identity check skipped)")

    print("\n{} checks, {} failed".format(
        16 + (1 if "mace" in sys.modules else 0) + (1 if "openqha_hessian" in sys.modules else 0), len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
