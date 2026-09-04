"""按**分支**分配决定标识符并追加决定行 —— 撞号的机械修法.

TOOLING. Allocates decision identifiers per branch and appends the decision line.
Produces no science.

**要解决的是什么**：本仓的决定日志过去是一条平坦的序列 `D0-1 … D0-N`，
两个会话同时写就会分到同一个号。**本次会话已经因此撞号两次**
（第一次 `D0-58`，第二次 `D0-76`/`D0-77`/`D0-78`），每次都要事后改号。

**为什么"下次记得先检查"不是修法**：检查与写入之间有时间窗，
另一个会话正好在这个窗里写入就还是撞。这是竞态，不是疏忽 ——
**要用机制解决，不能用纪律解决。**

**机制有两层，缺一层都不够**：

1. **按包分支到不同文件**。包 1 的决定写 `decisions_pkg1.md`，包 2 写 `decisions_pkg2.md`，
   共同约束写 `decisions_common.md`。两个会话在不同包上工作时**根本不碰同一个文件**，
   撞号的可能性从"要靠检查避免"降到"物理上不存在"。
2. **同一个分支内用锁**。两个会话真的同时改同一个包时，用 `O_CREAT|O_EXCL` 的锁文件
   把"读最大号 → 写新行"变成一个原子操作。锁带持有者与时间戳，过期锁可被强夺，
   所以崩溃不会留下永久死锁。

分支前缀::

    D0-C-n    共同约束（跨包的治理、版本、硬件、范围、标准）
    D0-L-n    讲义 s0-1
    D0-P1-n   包 1  构象生成与去重
    D0-P2-n   包 2  Hessian / 能量与力 / 参考层级
    D0-P3-n   包 3  分子动力学
    D0-P4-n   包 4  态密度与低频（含受阻转子、MS-T）
    D0-P5-n   包 5  电子项

**主干 `D0-1 … D0-85` 已冻结**：编号不再增长，也**不重新编号** ——
26 个文件里引用着它们，改号只会制造断链，而且**改号并不能减少撞号**
（撞号只发生在新分配时）。历史行的分支归属记在
`.mem/decisions/README-decisions.md` 的索引里，供按包检索。

用法::

    python scripts/tooling/s0_mem_decide.py --branch P2 \
        --decision "包 2 的 E2 走 MP2 增量阶梯 ..." \
        --basis "用户裁定；实测项池 4 个单点 ..."

    python scripts/tooling/s0_mem_decide.py --list P2        # 看某个分支现有的号
    python scripts/tooling/s0_mem_decide.py --next P2        # 只看下一个号，不写
"""
import argparse
import datetime as _dt
import errno
import os
import re
import sys
import time
from pathlib import Path

def _repo_root():
    """Directory holding the openQHA package, found by walking up from this file.

    Depth-independent on purpose: this file keeps working wherever under the
    repository it is moved to. The earlier move into `scripts/_superseded/` broke
    every `parents[1]` in the moved files silently, which is what this replaces.
    """
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))

from openqha import S0_ROOT

MEM = S0_ROOT / ".mem" / "decisions"
LOCK_STALE_SECONDS = 300        # 超过它就认为持有者已崩溃，可强夺

# Each entry is (identifier prefix, file stem, description).
#
# 2026-09-03 (S0-G-2): the rewrite introduced a second identifier space. The old
# `D0-*` branches are FROZEN -- they are listed here so `--all` can still report
# them, but nothing new is ever appended to them. New decisions go to the `S0-*`
# branches below.
#
# The prefix letter had to change rather than the branch letter: the old space
# already contains `D0-C-*` (common constraints), so a new branch C for the
# Hessian work would have collided and destroyed the lookup for both.
BRANCHES = {
    # --- frozen (D0- space) ----------------------------------------------------
    "C": ("D0", "common", "共同约束（跨包的治理、版本、硬件、范围、标准）"),
    "L": ("D0", "lecture", "讲义 s0-1"),
    "P1": ("D0", "pkg1", "包 1  构象生成与去重"),
    "P2": ("D0", "pkg2", "包 2  Hessian / 能量与力 / 参考层级"),
    "P3": ("D0", "pkg3", "包 3  分子动力学"),
    "P4": ("D0", "pkg4", "包 4  态密度与低频（含受阻转子、MS-T）"),
    "P5": ("D0", "pkg5", "包 5  电子项"),
    # --- active (S0- space, 2026-09-03 rewrite) ---------------------------------
    "S0-G": ("S0", "G_governance", "治理、版本、硬件、范围、标准 —— 跨分路的共同约束"),
    "S0-A": ("S0", "A_conformer", "分路 A  构象搜索（CREST GFN2-xTB + MACE refine=opt）"),
    "S0-B": ("S0", "B_qha", "分路 B  准谐振分析"),
    "S0-C": ("S0", "C_hessian", "分路 C  Hessian 监督训练与 RI-MP2/RIJK/cc-pVTZ 对标"),
    "S0-D": ("S0", "D_repo", "分路 D  仓库结构重整与代码英文化"),
}

#: Frozen branches refuse new rows. Reading and `--all` still work.
FROZEN = {"C", "L", "P1", "P2", "P3", "P4", "P5"}

ROW = re.compile(r"^\|\s*\*\*([DS]0)-([A-Z0-9]+)-(\d+)\*\*\s*\|")


def branch_prefix(branch):
    """Identifier prefix for a branch: "D0" (frozen) or "S0" (active)."""
    if branch not in BRANCHES:
        raise KeyError("没有分支 {!r}；可选: {}".format(branch, sorted(BRANCHES)))
    return BRANCHES[branch][0]


def branch_letter(branch):
    """The letter that appears in the identifier, i.e. the part after the prefix.

    For the frozen branches the key IS the letter ("P1" -> D0-P1-n). For the new
    ones the key carries the prefix so that keys stay unique across both spaces
    ("S0-A" -> S0-A-n).
    """
    return branch.split("-", 1)[1] if branch.startswith("S0-") else branch


def branch_file(branch):
    if branch not in BRANCHES:
        raise KeyError("没有分支 {!r}；可选: {}".format(branch, sorted(BRANCHES)))
    return MEM / "decisions_{}.md".format(BRANCHES[branch][1])


def existing_numbers(branch):
    p = branch_file(branch)
    if not p.exists():
        return []
    prefix = branch_prefix(branch)
    letter = branch_letter(branch)
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        m = ROW.match(line)
        if m and m.group(1) == prefix and m.group(2) == letter:
            out.append(int(m.group(3)))
    return sorted(out)


def next_number(branch):
    n = existing_numbers(branch)
    return (max(n) + 1) if n else 1


class Lock:
    """`O_CREAT|O_EXCL` 的锁文件。过期锁可强夺，所以崩溃不留死锁。"""

    def __init__(self, branch, timeout=60.0):
        self.path = MEM / ".lock_{}".format(branch)
        self.timeout = timeout
        self.fd = None

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                self.fd = os.open(str(self.path),
                                  os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, "{} {}\n".format(os.getpid(), time.time())
                         .encode("utf-8"))
                return self
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                age = time.time() - self.path.stat().st_mtime
                if age > LOCK_STALE_SECONDS:
                    print("锁已过期 {:.0f} 秒（上限 {}），强夺。".format(
                        age, LOCK_STALE_SECONDS))
                    self.path.unlink(missing_ok=True)
                    continue
                if time.time() - t0 > self.timeout:
                    raise RuntimeError(
                        "等锁超时 {:.0f} 秒：{} 被别的会话持有 {:.0f} 秒。"
                        "它没有过期，所以不强夺 —— 请等它写完。".format(
                            self.timeout, self.path, age))
                time.sleep(0.2)

    def __exit__(self, *a):
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)
        return False


def append(branch, decision, basis, date=None):
    """原子地分配下一个号并追加一行。返回分配到的标识符。"""
    MEM.mkdir(parents=True, exist_ok=True)
    if branch in FROZEN:
        raise ValueError(
            "branch {!r} is frozen (S0-G-2): its identifier space no longer grows.\n"
            "Its rows live in the archive under _to_delete/stage0-discard/decisions/.\n"
            "Append to one of the active branches instead: {}".format(
                branch, sorted(b for b in BRANCHES if b not in FROZEN)))
    date = date or _dt.date.today().isoformat()
    with Lock(branch):
        n = next_number(branch)
        ident = "{}-{}-{}".format(branch_prefix(branch), branch_letter(branch), n)
        p = branch_file(branch)
        if not p.exists():
            p.write_text(header(branch), encoding="utf-8")
        row = "| **{}** | {} | {} | {} |\n".format(
            ident, date, decision.replace("|", "\\|"), basis.replace("|", "\\|"))
        with open(str(p), "a", encoding="utf-8") as fh:
            fh.write(row)
    return ident


def header(branch):
    prefix, _name, desc = BRANCHES[branch]
    ident = "{}-{}".format(prefix, branch_letter(branch))
    return (
        "# stage 0 决定日志 —— 分支 `{ident}`（{desc}）\n\n"
        "> **分支制的理由**：平坦的序列在两个会话并行时会撞号（本仓已发生两次）。"
        "按分路分文件之后，不同分路上的工作**根本不碰同一个文件**；"
        "同一分支内的并发再由 `scripts/tooling/s0_mem_decide.py` 的锁保证原子。\n"
        ">\n"
        "> **只用 `scripts/tooling/s0_mem_decide.py --branch {b}` 追加，不要手写行** ——"
        "手写就绕过了锁，撞号会回来。\n"
        ">\n"
        "> 旧的 `D0-1 … D0-85` 与 `D0-C/L/P1…P5` 已冻结（`S0-G-2`），"
        "不再增长也不重新编号；索引见 `../memory-discard.md`。\n\n"
        "| 标识符 | 日期 | 决定 | 依据 / 落在哪里 |\n"
        "|---|---|---|---|\n".format(ident=ident, b=branch, desc=desc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch",
                    help="active: S0-G / S0-A / S0-B / S0-C / S0-D | "
                         "frozen (read only): C / L / P1 / P2 / P3 / P4 / P5")
    ap.add_argument("--decision", help="决定本身，一句话说清")
    ap.add_argument("--basis", help="依据 / 落在哪里")
    ap.add_argument("--date", default=None)
    ap.add_argument("--list", dest="list_branch", default=None)
    ap.add_argument("--next", dest="next_branch", default=None)
    ap.add_argument("--all", action="store_true", help="列出所有分支的现状")
    args = ap.parse_args()

    if args.all:
        for b in BRANCHES:
            n = existing_numbers(b)
            state = "FROZEN" if b in FROZEN else "active"
            ident = "{}-{}".format(branch_prefix(b), branch_letter(b))
            nxt = "--" if b in FROZEN else "{}-{}".format(ident, next_number(b))
            print("{:8s}  {:14s}  {:6s}  {} 条  下一个 {}".format(
                ident, BRANCHES[b][1], state, len(n), nxt))
        return
    if args.list_branch:
        b = args.list_branch
        for n in existing_numbers(b):
            print("{}-{}-{}".format(branch_prefix(b), branch_letter(b), n))
        return
    if args.next_branch:
        b = args.next_branch
        print("{}-{}-{}".format(branch_prefix(b), branch_letter(b), next_number(b)))
        return
    if not (args.branch and args.decision and args.basis):
        ap.error("要么给 --all/--list/--next，要么同时给 --branch --decision --basis")
    print(append(args.branch, args.decision, args.basis, args.date))


if __name__ == "__main__":
    main()
