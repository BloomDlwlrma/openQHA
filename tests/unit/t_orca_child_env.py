"""Unit test: the ORCA child environment is Slurm-blind (orca-slurm ticket 01, ADR 0008).

UNIT. Fabricates one environment dict and statically scans the tree; no ORCA, no
scheduler, seconds.

Why the seam must be total
--------------------------
The draw300 labels round on TianheXY-C died before any chemistry because ORCA's bundled
OpenMPI 4.1 selected `ras/slurm` on `SLURM_JOBID` alone and force-terminated when the
job script's narrow unset had removed exactly the `SLURM_TASKS_PER_NODE` that component
demands. The remedy is one seam -- `openqha.qm_interfaces.orca.subprocess_env()` --
that every ORCA launch route uses, so no caller's shell can re-arm the scheduler inside
an ORCA child.

Asserted:

A. `subprocess_env()`, handed an environment full of scheduler variables, comes back
   with every name starting with `SLURM` or `PMI` deleted; everything else kept;
   `S0_ORCA_PATH` / `S0_ORCA_LIB` kept and prepended to PATH / LD_LIBRARY_PATH; and
   `OMPI_MCA_hwloc_base_binding_policy=none` set (OpenMPI's own binding off -- the
   worker's `taskset` range is placement's only owner). The caller's environment is
   not mutated: the ORCA child is stripped, the worker keeps its Slurm view.

B. No ORCA launch bypasses the seam: every spawn call in the live tree whose command
   expression mentions ORCA passes an `env=` built by `subprocess_env()`. The known
   launchers must be found (so a broken matcher cannot pass vacuously), and a new
   launcher that forgets the seam fails here. `subprocess` aliases (`import subprocess
   as _sp`, `from subprocess import run as ...`) are resolved; detection is textual on
   the command expression, so a binary resolved at runtime (`shutil.which`) cannot be
   seen -- the one such site today (`capabilities._binary_runs("orca")`, a `--version`
   probe) is listed in the ticket for disposition. (The hkuhpc bundle's generated shell
   worker is not a Python subprocess: it already carries its own full `^(PMI|SLURM)`
   unset; its undocumented `ORCA_SKIP_CPU_BIND` is a separate, planned cleanup.)

Run::  python tests/unit/t_orca_child_env.py
"""
import ast
import os
import sys
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from openqha.qm_interfaces import orca                    # noqa: E402

FAIL = []

#: The fabrication: legacy and modern Slurm names (including the daemon's), PMI and
#: PMIx. Every one must be gone from the child environment.
SCHEDULER = {
    "SLURM_JOBID": "1", "SLURM_JOB_ID": "1", "SLURM_NODELIST": "fake1",
    "SLURM_TASKS_PER_NODE": "1", "SLURM_JOB_CPUS_PER_NODE": "64",
    "SLURMD_NODENAME": "nv1", "SLURM_CPU_BIND": "none", "PMI_FD": "3",
    "PMI_RANK": "0", "PMIX_X": "y",
}

#: Directories walked by the static scan, and the directories inside them that hold no
#: live launches (superseded work never runs).
SCAN_ROOTS = ("openqha", "scripts", "examples", "hpc", "workflows")
SKIP_DIRS = {"__pycache__", "_superseded", "_backup"}

#: The subprocess entry points that spawn a process (the 3.5+ additions included).
SPAWN_METHODS = ("call", "run", "Popen", "check_call", "check_output")

#: The launch sites that route through the seam today. The scan must FIND each of
#: them: a launcher silently renamed would otherwise turn the totality check vacuous.
EXPECTED = {
    "openqha/qm_interfaces/orca.py",                    # _run_job + single_point
    "openqha/data/frame_labels.py",                     # the labels worker
    "scripts/production/s0_branch2_opt_freq.py",        # branch B opt+freq
    "scripts/calibration/s0_package2_highlevel_freq.py",  # package2 high level
    "examples/02c_hessian_benchmark_levels/s0_level_benchmark.py",
}


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


def test_child_environment():
    """Seam A: a fabricated scheduler environment comes back stripped."""
    keep = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(SCHEDULER)
        os.environ.update({"S0_ORCA_PATH": "/x/bin", "S0_ORCA_LIB": "/x/lib",
                           "PATH": "/usr/bin", "LD_LIBRARY_PATH": "/usr/lib",
                           "HOME": "/home/u",
                           "OMPI_MCA_hwloc_base_binding_policy": "core"})
        env = orca.subprocess_env()
        caller_kept = sorted(k for k in os.environ if k.startswith(("SLURM", "PMI")))
    finally:
        os.environ.clear()
        os.environ.update(keep)

    leaked = sorted(k for k in env if k.startswith(("SLURM", "PMI")))
    check("every SLURM*/PMI* variable is deleted from the ORCA child", not leaked, leaked)
    check("the caller's own environment still carries them (the worker keeps its Slurm view)",
          caller_kept == sorted(SCHEDULER), caller_kept)
    check("S0_ORCA_PATH is kept and prepended to PATH",
          env.get("S0_ORCA_PATH") == "/x/bin"
          and env.get("PATH") == "/x/bin" + os.pathsep + "/usr/bin", env.get("PATH"))
    check("S0_ORCA_LIB is kept and prepended to LD_LIBRARY_PATH",
          env.get("S0_ORCA_LIB") == "/x/lib"
          and env.get("LD_LIBRARY_PATH") == "/x/lib" + os.pathsep + "/usr/lib",
          env.get("LD_LIBRARY_PATH"))
    check("OMPI_MCA_hwloc_base_binding_policy=none is set (OpenMPI's binding off; taskset owns placement)",
          env.get("OMPI_MCA_hwloc_base_binding_policy") == "none",
          env.get("OMPI_MCA_hwloc_base_binding_policy"))
    check("OMPI_MCA_rmaps_base_oversubscribe is NOT set (documented fallback only)",
          "OMPI_MCA_rmaps_base_oversubscribe" not in env)
    check("everything else is kept", env.get("HOME") == "/home/u")


def _spawn_bindings(tree):
    """The names bound to `subprocess` itself and to its spawn functions in `tree`."""
    modules, functions = {"subprocess"}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "subprocess":
                    modules.add(a.asname or "subprocess")
        elif isinstance(node, ast.ImportFrom) and node.module == "subprocess":
            for a in node.names:
                if a.name in SPAWN_METHODS:
                    functions.add(a.asname or a.name)
    return modules, functions


def _is_spawn_call(func, modules, functions):
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id in modules and func.attr in SPAWN_METHODS
    if isinstance(func, ast.Name):
        return func.id in functions
    return False


def _orca_spawns(path):
    """Every subprocess spawn in `path` whose command expression mentions ORCA:
    (lineno, command source, env source or None)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules, functions = _spawn_bindings(tree)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_spawn_call(node.func, modules, functions):
            continue
        arg = node.args[0] if node.args else next(
            (kw.value for kw in node.keywords if kw.arg == "args"), None)
        if arg is None:
            continue
        command = ast.unparse(arg)
        if "orca" not in command.lower():
            continue
        env = next((kw for kw in node.keywords if kw.arg == "env"), None)
        out.append((node.lineno, command,
                    ast.unparse(env.value) if env is not None else None))
    return out


def test_no_launch_bypasses_the_seam():
    """Seam B: every ORCA spawn in the live tree passes subprocess_env()."""
    seen, bypass, scan_errors = {}, [], []
    for root in SCAN_ROOTS:
        for p in sorted((ROOT / root).rglob("*.py")):
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            try:
                spawns = _orca_spawns(p)
            except SyntaxError as exc:
                scan_errors.append("{}: {}".format(p.relative_to(ROOT), exc))
                continue
            for lineno, command, env in spawns:
                rel = str(p.relative_to(ROOT)).replace(os.sep, "/")
                seen.setdefault(rel, []).append(lineno)
                if env is None or "subprocess_env" not in env:
                    bypass.append("{}:{} {} (env={})".format(rel, lineno, command, env))
                else:
                    print("  {:78s} {}".format("{}:{}".format(rel, lineno), "seam ok"))
    check("no ORCA spawn bypasses subprocess_env()", not bypass, bypass)
    missing = sorted(EXPECTED - set(seen))
    check("the scan finds every known launcher (not vacuous)", not missing, missing)
    check("every scanned file parses", not scan_errors, scan_errors)


def main():
    print("A. the child environment seam")
    test_child_environment()
    print("\nB. every ORCA launch goes through the seam")
    test_no_launch_bypasses_the_seam()
    print("\nPASS" if not FAIL else "\nFAIL: " + "; ".join(FAIL))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
