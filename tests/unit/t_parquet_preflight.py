"""A missing parquet engine is refused before the work, and the refusal says what to run.

UNIT. No pandas needed; under a second.

The defect (an113, 2026-09-13)
------------------------------
The first branch B analysis to complete on a card -- six trajectories read, every
criterion judged, 10 s -- died on its final line:

    ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'.

The openqha-gpu environment there had no pyarrow, although environment-tianhe-gpu.yml
lists it. Nothing the driver had computed was written. The import that failed is the
LAST one in the chain, so the check now happens FIRST: `report.parquet_engine()` at the
top of s0_B_qha_analyse.main(), and `openqha_require_modules pandas pyarrow` at the top
of examples/chain_body.sh.

What is asserted
----------------
  A. With an engine importable, parquet_engine() returns (name, version).
  B. With both engines blocked, it raises ImportError, and the message carries the
     install command -- the reader on the compute node has no other source for it.
  C. hpc/resource_configs/tianhe_ai reads the partition from OPENQHA_PARTITION or
     SLURM_JOB_PARTITION, and ignores a value that is not one of its partitions (the
     driver on an113/a100x had recorded "h100x" and sized 14 workers on 12 CPUs).
"""
import importlib
import os
import subprocess
import sys
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from openqha.store import report                            # noqa: E402

FAIL = []


class _Block:
    """A meta-path finder that makes the named modules unimportable."""

    def __init__(self, names):
        self.names = set(names)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.names:
            raise ImportError("blocked by test: " + fullname)
        return None


def check_present():
    print("A. an engine is importable here -> (name, version)")
    try:
        name, version = report.parquet_engine()
        print("   ok    {} {}".format(name, version))
    except ImportError as exc:
        print("   skip  no engine in this environment either: {}".format(exc))


def check_absent():
    print("\nB. both engines blocked -> ImportError that says what to install")
    saved = {k: v for k, v in sys.modules.items()
             if k.split(".")[0] in ("pyarrow", "fastparquet")}
    for k in saved:
        del sys.modules[k]
    blocker = _Block(("pyarrow", "fastparquet"))
    sys.meta_path.insert(0, blocker)
    try:
        try:
            got = report.parquet_engine()
            FAIL.append("parquet_engine returned {} with both engines blocked".format(got))
            print("   FAIL  returned {}".format(got))
        except ImportError as exc:
            text = str(exc)
            ok = report.INSTALL_PARQUET in text and "pyarrow" in text
            print("   {}  ImportError: {}".format("ok  " if ok else "FAIL", text.splitlines()[0]))
            if not ok:
                FAIL.append("the refusal does not carry the install command")
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)


def check_partition():
    print("\nC. tianhe_ai takes the partition from the job's environment")
    cases = [
        ({}, "h100x", "no variable set -> default"),
        ({"OPENQHA_PARTITION": "a100x"}, "a100x", "OPENQHA_PARTITION=a100x"),
        ({"SLURM_JOB_PARTITION": "a100x"}, "a100x", "SLURM_JOB_PARTITION=a100x"),
        ({"OPENQHA_PARTITION": "ai", "SLURM_JOB_PARTITION": "a100x"}, "a100x",
         "OPENQHA_PARTITION=ai (not here) is skipped, Slurm's a100x is taken"),
        ({"OPENQHA_PARTITION": "ai"}, "h100x", "OPENQHA_PARTITION=ai alone -> default"),
    ]
    code = ("import sys; sys.path.insert(0, {!r}); "
            "from hpc.resource_configs import tianhe_ai as t; "
            "d = t.describe(); "
            "print(t.PARTITION, d['layouts']['collect']['workers_per_node'], "
            "','.join(d['roles']))").format(str(ROOT))
    for env_extra, want, label in cases:
        env = {k: v for k, v in os.environ.items()
               if k not in ("OPENQHA_PARTITION", "SLURM_JOB_PARTITION")}
        env.update(env_extra)
        env["PYTHONPATH"] = str(ROOT)
        proc = subprocess.run([sys.executable, "-c", code], env=env,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            FAIL.append("tianhe_ai import failed under {}: {}".format(
                env_extra, proc.stderr.strip().splitlines()[-1:]))
            print("   FAIL  {}: {}".format(label, proc.stderr.strip().splitlines()[-1:]))
            continue
        got, workers, roles = proc.stdout.split()
        ok = got == want and "collect" in roles.split(",")
        print("   {}  {:<62} -> {} (collect workers {})".format(
            "ok  " if ok else "FAIL", label, got, workers))
        if got != want:
            FAIL.append("{}: got {} wanted {}".format(label, got, want))
        if "collect" not in roles.split(","):
            FAIL.append("describe() does not list the collect role")
        if got == "a100x" and workers != "12":
            FAIL.append("a100x collect workers {} (package has 12 CPUs)".format(workers))


def main():
    check_present()
    check_absent()
    check_partition()
    print()
    if FAIL:
        print("{} problem(s):".format(len(FAIL)))
        for f in FAIL:
            print("  - " + f)
        return 1
    print("a missing engine is refused up front, and the job's partition is the one recorded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
