# -*- coding: utf-8 -*-
"""Isomerisation reaction energy dE_el: between two species of the same formula the difference of absolute energies **is comparable**.

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
print("For each species the **lowest basin** is taken (separately for MACE and for the reference; a disagreement is marked)")
low = {}
for qid, rs in sorted(by_species.items()):
    m = min(rs, key=lambda x: x["energy_mace_eV"])
    for rc in ["dz_base", "composite_TZstar_mp2", "composite_QZstar_mp2"]:
        rmin = min(rs, key=lambda x: x["per_recipe"][rc]["energy_ref_eV"])
        low.setdefault(rc, {})[qid] = (m, rmin, m["basin"] != rmin["basin"])
    print("   {:20s} {} basin(s)   MACE lowest = basin {}".format(m["name"], len(rs), m["basin"]))

print()
print("=" * 96)
print("Isomerisation electronic energy dE_el = E(B) - E(A), kcal/mol   (**one of stage 0 delivered quantities**)")
print("=" * 96)
for rc in ["dz_base", "composite_TZstar_mp2", "composite_QZstar_mp2"]:
    diffs = []
    print()
    print("### recipe {}".format(rc))
    print("   {:24s} {:>12s} {:>12s} {:>10s}".format("edge", "MACE", "reference", "difference"))
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
    print("   -> {} edge(s): mean absolute deviation {:.4f}   root-mean-square deviation {:.4f}   maximum absolute {:.4f} kcal/mol"
          .format(len(d), np.abs(d).mean(), float(np.sqrt((d ** 2).mean())),
                  np.abs(d).max()))
