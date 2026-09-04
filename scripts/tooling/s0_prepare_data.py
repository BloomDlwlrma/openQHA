"""把 QM9 数据放进 stage 0 自己的 `data/` 目录.

TOOLING. Puts QM9 in place under data/. It moves data; it computes nothing.

**为什么要有这个脚本**：2026-08-28 起 stage 0 是独立的开源框架，
它不从 stage 1 / stage 2 的目录里读任何东西。但 QM9 原始数据太大（索引 119 MB、
几何 224 MB），**不进 git**。所以本脚本负责一次性把它放到位。

随仓库分发的只有 **7 个目标物种的参考几何**（`data/reference-geometries/`，11 KB），
因此 **stage 0 的核心交付（包 2）零外部数据即可复现**；
只有包 1 的大规模普查需要跑一次本脚本。

用法:
    python scripts/tooling/s0_prepare_data.py --from <某个 QM9 目录>       # 复制
    python scripts/tooling/s0_prepare_data.py --from <目录> --link         # 建符号链接, 省磁盘
    python scripts/tooling/s0_prepare_data.py --check                      # 只检查现状

`--from` 指向的目录里应当有:
    index_Chem_composition.csv
    xyz_files/dsgdb9nsd_XXXXXX.xyz

也可以完全不跑本脚本, 直接设环境变量:
    export S0_QM9_ROOT=/path/to/qm9
"""
import argparse
import hashlib
import os
import shutil
import sys
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

from openqha import S0_ROOT, config


def sha256_head(path, n_bytes=1 << 20):
    """前 1 MB 的摘要 —— 119 MB 全文摘要太慢, 前 1 MB 足以发现"换了一份文件"。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(n_bytes))
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default=None, help="QM9 数据所在目录")
    ap.add_argument("--link", action="store_true", help="建符号链接而不是复制")
    ap.add_argument("--check", action="store_true", help="只检查现状, 不动文件")
    args = ap.parse_args()

    cfg = config.load()
    root = config.qm9_root(cfg)
    index_name = cfg["data"]["qm9_index_csv"]
    xyz_name = cfg["data"]["qm9_xyz_dir"]
    vend = S0_ROOT / cfg["data"]["vendored_reference_geometries"]

    print("=" * 92)
    print("stage 0 数据准备")
    print("=" * 92)
    print("配置          {}".format(cfg["_path"]))
    print("目标目录      {}{}".format(root, "   (来自 S0_QM9_ROOT)"
                                      if os.environ.get("S0_QM9_ROOT") else ""))
    print("随仓库分发    {}  ({} 个文件)".format(
        vend.relative_to(S0_ROOT), len(list(vend.glob("*.xyz"))) if vend.exists() else 0))
    print()

    if args.check or not args.src:
        idx = root / index_name
        xyz = root / xyz_name
        print("索引表        {}  {}".format(idx, "存在" if idx.exists() else "**缺失**"))
        if idx.exists():
            print("              {:.1f} MB, 前 1 MB 摘要 {}".format(
                idx.stat().st_size / 1048576, sha256_head(idx)[:16]))
        print("几何目录      {}  {}".format(xyz, "存在" if xyz.exists() else "**缺失**"))
        if xyz.exists():
            n = sum(1 for _ in xyz.glob("dsgdb9nsd_*.xyz"))
            print("              {} 个 xyz 文件".format(n))
        print()
        if not args.src:
            if idx.exists() and xyz.exists():
                print("全量数据已就位, 包 1 可跑。")
            else:
                print("全量数据未就位 —— **包 2 仍可跑**（它只用随仓库分发的 7 个几何）,")
                print("包 1 的大规模普查需要先跑:")
                print("    python scripts/tooling/s0_prepare_data.py --from <某个 QM9 目录>")
            return

    src = Path(args.src)
    if not src.exists():
        raise FileNotFoundError("来源目录不存在: {}".format(src))
    s_idx, s_xyz = src / index_name, src / xyz_name
    for p in (s_idx, s_xyz):
        if not p.exists():
            raise FileNotFoundError("来源目录里缺 {}: {}".format(p.name, p))

    root.mkdir(parents=True, exist_ok=True)
    d_idx, d_xyz = root / index_name, root / xyz_name

    if args.link:
        for s, d in ((s_idx, d_idx), (s_xyz, d_xyz)):
            if d.exists() or d.is_symlink():
                print("已存在, 跳过: {}".format(d))
                continue
            os.symlink(s.resolve(), d)
            print("链接 {} -> {}".format(d, s.resolve()))
    else:
        if d_idx.exists():
            print("已存在, 跳过: {}".format(d_idx))
        else:
            print("复制索引表 {:.1f} MB ...".format(s_idx.stat().st_size / 1048576))
            shutil.copy2(s_idx, d_idx)
        if d_xyz.exists():
            print("已存在, 跳过: {}".format(d_xyz))
        else:
            n = sum(1 for _ in s_xyz.glob("dsgdb9nsd_*.xyz"))
            print("复制 {} 个 xyz 文件 ...".format(n))
            shutil.copytree(s_xyz, d_xyz)

    print()
    print("来源记录:")
    print("   来源目录    {}".format(src.resolve()))
    print("   索引表摘要  {} (前 1 MB)".format(sha256_head(d_idx)[:32]))
    print("   数据集      {}".format(cfg["data"]["upstream"]["dataset"]))
    print()
    print("完成。`data/qm9/` 已在 `data/.gitignore` 里, 不会进版本库。")


if __name__ == "__main__":
    main()
