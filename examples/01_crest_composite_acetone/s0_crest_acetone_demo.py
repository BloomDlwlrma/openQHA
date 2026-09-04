"""丙酮的 CREST 复合计算器工作流 —— 讲义 2 的实测来源.

跑法::
    python -m openqha.mace_server --socket /tmp/s0_mace_engrad.sock &
    S0_MACE_SOCKET=/tmp/s0_mace_engrad.sock python examples/01_crest_composite_acetone/s0_crest_acetone_demo.py
"""
import json
import os
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
from openqha import S0_ROOT, config, crest

CFG = config.load()
C = CFG["crest"]
OUT = S0_ROOT / "analysis" / "package1" / "crest_acetone"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    xyz = OUT / "acetone.xyz"
    if not xyz.exists():
        raise FileNotFoundError("起始几何不在: {}".format(xyz))

    print("=" * 92)
    print("丙酮 —— CREST 复合计算器（GFN-FF 采样 + MACE 精修）")
    print("=" * 92)
    v = crest.crest_version()
    print("CREST        {} (commit {})".format(v["version"], v["commit"]))
    print("支持 mlip 吗 {}   <- 需 CREST 3.1, 见 D0-87".format(crest.supports_mlip()))
    print("设置         runtype={} refine={} optlev={} threads={} backend={}".format(
        C["runtype"], C["refine"], C["optlev"], C["threads"], C["backend"]))
    print("套接字       {}".format(os.environ.get("S0_MACE_SOCKET", C["socket"])))
    print()

    rec = crest.run(OUT, xyz, runtype=C["runtype"], threads=int(C["threads"]),
                    optlev=C["optlev"], refine=C["refine"], backend=C["backend"],
                    engine_client=S0_ROOT / C["engine_client"], timeout_s=7200)

    print("用时         {:.0f} s = {:.2f} 小时".format(rec["seconds"], rec["seconds"] / 3600))
    print("正常收尾     {}".format(rec["terminated_normally"]))
    print("terminated EARLY   {}   <- 判据: 必须是 0".format(rec["n_terminated_early"]))
    print("completed success  {}".format(rec["n_completed_successfully"]))
    print("能量+梯度调用总数  {}".format(rec["total_engrad_calls"]))
    print("产物         {}".format(rec["products"]))
    print("总判据 ok    {}".format(rec["ok"]))

    ens = OUT / "crest_conformers.xyz"
    if ens.exists():
        frames = crest.read_ensemble(ens)
        rec["n_conformers"] = len(frames)
        print()
        print("构象数       {}".format(len(frames)))
        e = []
        for c, _ in frames:
            try:
                e.append(float(c.split()[0]))
            except (ValueError, IndexError):
                pass
        if e:
            e0 = min(e)
            print("相对能量 (kcal/mol): {}".format(
                " ".join("{:.4f}".format((x - e0) * 627.5094740631) for x in e)))

    (OUT / "run_record.json").write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("落盘: {}".format(OUT / "run_record.json"))
    if not rec["ok"]:
        print()
        print("** 未通过判据, 输出尾部:")
        print(rec.get("tail", ""))


if __name__ == "__main__":
    main()
