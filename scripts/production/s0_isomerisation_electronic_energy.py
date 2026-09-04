# -*- coding: utf-8 -*-
"""异构化反应能 dE_el：同分子式两个物种之间，绝对能量之差**是可比的**。

PRODUCTION. The isomerisation electronic energy of the edges -- the script's own
banner calls it one of stage 0's delivered quantities.
"""
import json, sys
import numpy as np
from pathlib import Path


def _repo_root():
    """Directory holding the openQHA package, found by walking up from this file.

    Replaces a bare insertion of "." on sys.path, which only worked when the script
    happened to be launched from the repository root and failed silently anywhere
    else.
    """
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


REPO_ROOT = _repo_root()
sys.path.insert(0, str(REPO_ROOT))
from openqha import config, conformers

cfg = config.load()
rows = [json.loads(l) for l in open(
    REPO_ROOT / "analysis" / "composite_energy_force_E2.partial.jsonl",
    encoding="utf-8")]
mins = [r for r in rows if r["displaced"] is None]
by_species = {}
for r in mins:
    by_species.setdefault(r["qm9_index"], []).append(r)

K = conformers.EV_TO_KCAL
print("每个物种取**最低的盆**（按 MACE 与按参考各自取，若不同则标出）")
low = {}
for qid, rs in sorted(by_species.items()):
    m = min(rs, key=lambda x: x["energy_mace_eV"])
    for rc in ["dz_base", "composite_TZstar_mp2", "composite_QZstar_mp2"]:
        rmin = min(rs, key=lambda x: x["per_recipe"][rc]["energy_ref_eV"])
        low.setdefault(rc, {})[qid] = (m, rmin, m["basin"] != rmin["basin"])
    print("   {:20s} {} 个盆   MACE 最低 = 盆 {}".format(m["name"], len(rs), m["basin"]))

print()
print("=" * 96)
print("异构化电子能 dE_el = E(乙) - E(甲)，kcal/mol   （**这是 stage 0 的交付量之一**）")
print("=" * 96)
for rc in ["dz_base", "composite_TZstar_mp2", "composite_QZstar_mp2"]:
    diffs = []
    print()
    print("### 配方 {}".format(rc))
    print("   {:24s} {:>12s} {:>12s} {:>10s}".format("边", "MACE", "参考", "差"))
    for e in cfg["edges"]:
        p = e.split("_")
        ia = "dsgdb9nsd_{:06d}".format(int(p[-2]))
        ib = "dsgdb9nsd_{:06d}".format(int(p[-1]))
        if ia not in low[rc] or ib not in low[rc]:
            continue
        ma, ra, _ = low[rc][ia]
        mb, rb, _ = low[rc][ib]
        dm = (mb["energy_mace_eV"] - ma["energy_mace_eV"]) * K
        dr = ((rb["per_recipe"][rc]["energy_ref_eV"]
               - ra["per_recipe"][rc]["energy_ref_eV"]) * K)
        diffs.append(dm - dr)
        print("   {:24s} {:12.4f} {:12.4f} {:+10.4f}".format(e, dm, dr, dm - dr))
    d = np.array(diffs)
    print("   -> {} 条边: 平均绝对偏差 {:.4f}   均方根偏差 {:.4f}   最大绝对 {:.4f} kcal/mol"
          .format(len(d), np.abs(d).mean(), float(np.sqrt((d ** 2).mean())),
                  np.abs(d).max()))
