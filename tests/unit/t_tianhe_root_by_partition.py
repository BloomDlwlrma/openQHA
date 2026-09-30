"""The root of the molecule tree is derived from the partition and the account.

UNIT. Runs bash on `hpc/env/root.sh` with a fake HOME, partition and mount base; under
a second; skipped with a message where bash is absent.

The rule
--------
    root = <prefix>/HDD_POOL/<acct>/<user>/sherwin/runs
    prefix /XYFS02  for partitions ai, temp, cn, deimos, debug (TianheXY-A, TianheXY-CN)
           /XYAIFS00 for a100x h100x hx a800x v100x (TianheXY-AI)
    acct, user from HOME = /HOME/<acct>/<user>; the tail is one variable, S0_RUNS_TAIL.

Partition from OPENQHA_PARTITION, then SLURM_JOB_PARTITION; on a login node with neither,
from which prefix is mounted -- and when both or neither are, that is an ERROR, never a
fall-back to HOME (the 2026-09-12 defect was exactly a default nobody chose). An explicit
S0_RUNS_ROOT always wins. `S0_MOUNT_BASE` prefixes the mount test so this file can fake
a mounted filesystem in a temporary directory.

Expected values are literals from the rule above, not recomputed.
"""
import os
import shutil
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
ROOT_SH = ROOT / "hpc" / "env" / "root.sh"
FAIL = []


def resolve(env_extra, mount_base):
    """Source root.sh in a clean bash and print what it resolved (or its refusal)."""
    env = {"PATH": os.environ.get("PATH", ""), "HOME": "/HOME/hku2021_fos4/hku2021_fos4xy_2",
           "S0_MOUNT_BASE": mount_base}
    env.update(env_extra)
    script = 'source "{}" && openqha_resolve_root && printf "%s\\n" "$S0_RUNS_ROOT"'.format(
        ROOT_SH.as_posix())
    p = subprocess.run(["bash", "-c", script], env=env, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.returncode, p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""


def check(label, got, want):
    ok = got == want
    print("  {:52s} {}".format(label, "ok" if ok else "FAIL\n      got  {}\n      want {}".format(got, want)))
    if not ok:
        FAIL.append(label)


def main():
    if shutil.which("bash") is None:
        print("SKIP: no bash on this machine; root.sh is a bash file")
        return 0
    if not ROOT_SH.is_file():
        print("FAIL: {} does not exist".format(ROOT_SH))
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="openqha_root_"))
    try:
        A = "/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/runs"
        AI = "/XYAIFS00/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/runs"

        print("A. by partition (no mount needed):")
        for part, want in (("ai", A), ("temp", A), ("cn", A), ("deimos", A), ("debug", A),
                           ("a100x", AI), ("h100x", AI),
                           ("hx", AI), ("a800x", AI), ("v100x", AI)):
            rc, got = resolve({"OPENQHA_PARTITION": part}, str(tmp))
            check("OPENQHA_PARTITION=" + part, (rc, got), (0, want))
        rc, got = resolve({"SLURM_JOB_PARTITION": "a100x"}, str(tmp))
        check("SLURM_JOB_PARTITION=a100x", (rc, got), (0, AI))
        rc, got = resolve({"OPENQHA_PARTITION": "ai", "SLURM_JOB_PARTITION": "a100x"}, str(tmp))
        check("OPENQHA_PARTITION beats SLURM_JOB_PARTITION", (rc, got), (0, A))

        print("B. login node, no partition: the mounted prefix decides:")
        (tmp / "XYFS02" / "HDD_POOL").mkdir(parents=True)
        rc, got = resolve({}, str(tmp))
        check("only XYFS02 mounted", (rc, got), (0, A))
        (tmp / "XYAIFS00" / "HDD_POOL").mkdir(parents=True)
        rc, _ = resolve({}, str(tmp))
        check("both mounted -> refused", rc != 0, True)
        shutil.rmtree(tmp / "XYFS02")
        rc, got = resolve({}, str(tmp))
        check("only XYAIFS00 mounted", (rc, got), (0, AI))
        shutil.rmtree(tmp / "XYAIFS00")
        rc, _ = resolve({}, str(tmp))
        check("neither mounted -> refused", rc != 0, True)

        print("C. an unknown partition is refused, not guessed:")
        rc, _ = resolve({"OPENQHA_PARTITION": "gpu_big"}, str(tmp))
        check("OPENQHA_PARTITION=gpu_big", rc != 0, True)

        print("D. an explicit root wins over everything:")
        rc, got = resolve({"OPENQHA_PARTITION": "a100x", "S0_RUNS_ROOT": "/elsewhere/runs"}, str(tmp))
        check("S0_RUNS_ROOT set", (rc, got), (0, "/elsewhere/runs"))

        print("E. the tail is one variable:")
        rc, got = resolve({"OPENQHA_PARTITION": "ai", "S0_RUNS_TAIL": "other/runs"}, str(tmp))
        check("S0_RUNS_TAIL=other/runs", (rc, got),
              (0, "/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/other/runs"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n{}".format("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL)))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
