"""Ticket 23: enantiomer degeneracy g' by the port of CREST's intraconfRMSD.

UNIT. Pure NumPy plus RDKit for the equivalence classes; no engine; a second.

The independent source of truth is CREST itself: `tests/data/propanal_crest/` holds the
real `crest_rotamers.xyz`, `cre_members` and `crest_conformers.xyz` of propanal
(dsgdb9nsd_000035, branch A multibasin run, GFN2), whose gauche conformer's six rotamers
were shown on 2026-09-15 to be 3 methyl rotamers x 2 mirror images (heavy-atom RMSD 0.36 A,
0.000-0.008 A after x -> -x). CREST's own numbers for that file: conformer 1 (cis)
3 rotamers, conformer 2 (gauche) 6, conformer 3 one; g' = (1, 2, 1).

The hand-built cases probe the three branches of the algorithm on their own: an achiral
group whose members differ only by a methyl rotation (g' = 1 because the rotor atoms are
excluded from the RMSD), a chiral pair (g' = 2 with the mirror flag), and a single-member
group (`unsampled`, g' = 1).
"""
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from openqha.conformer_search import degeneracy             # noqa: E402
from openqha.store import layout, property as prop, report   # noqa: E402

DATA = ROOT / "tests" / "data" / "propanal_crest"
FAIL = []


def check(label, ok, detail=""):
    print("  {:66s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


# ---------------------------------------------------------------- hand-built molecules
def rotate_about_axis(points, axis_a, axis_b, angle):
    """Rotate `points` about the line a->b by `angle` (Rodrigues)."""
    k = axis_b - axis_a
    k = k / np.linalg.norm(k)
    p = points - axis_a
    c, s = math.cos(angle), math.sin(angle)
    out = p * c + np.cross(k, p) * s + np.outer(p @ k, k) * (1 - c)
    return out + axis_a


def ethanol_like():
    """CH3-CH2-OH in a fixed geometry: heavy atoms C C O, three methyl H, two CH2 H, one OH."""
    sym = ["C", "C", "O", "H", "H", "H", "H", "H", "H"]
    pos = np.array([
        [0.000, 0.000, 0.000],      # C0 methyl carbon
        [1.520, 0.000, 0.000],      # C1
        [2.050, 1.300, 0.000],      # O2
        [-0.380, 1.020, 0.000],     # H on C0
        [-0.380, -0.510, 0.880],
        [-0.380, -0.510, -0.880],
        [1.900, -0.520, 0.880],     # H on C1
        [1.900, -0.520, -0.880],
        [3.010, 1.250, 0.000],      # H on O
    ])
    return sym, pos


def write_rotamers(d, frames, members):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "crest_rotamers.xyz", "w") as f:
        for e, (sym, pos) in frames:
            f.write("{}\n{:.8f}\n".format(len(sym), e))
            for s, (x, y, z) in zip(sym, pos):
                f.write("{:2s} {:14.8f} {:14.8f} {:14.8f}\n".format(s, x, y, z))
    with open(d / "cre_members", "w") as f:
        f.write("{:4d}\n".format(len(members)))
        for n, lo, hi in members:
            f.write("{:11d}{:11d}{:11d}\n".format(n, lo, hi))


def main():
    # ------------------------------------------------------------ propanal, the real file
    res = degeneracy.conformer_degeneracies(DATA)
    g = [r["g_prime"] for r in res]
    check("propanal g' = (1, 2, 1) from CREST's own rotamer file", g == [1, 2, 1], g)
    check("propanal rotamer counts (3, 6, 1) read from cre_members",
          [r["n_rotamers"] for r in res] == [3, 6, 1])
    check("gauche conformer carries the mirror flag; cis does not",
          res[1]["mirror_flag"] is True and res[0]["mirror_flag"] is False)
    check("third conformer is 'unsampled' (one rotamer)", res[2]["g_prime_source"] == "unsampled")
    check("gauche core count is 2 and cis core count is 1",
          res[1]["n_cores"] == 2 and res[0]["n_cores"] == 1)
    check("propanal excludes the three methyl hydrogens from the RMSD",
          sorted(res[0]["excluded_atoms"]) == sorted(res[1]["excluded_atoms"])
          and len(res[0]["excluded_atoms"]) >= 3)

    # ------------------------------------------------------------ hand-built cases
    sym, pos = ethanol_like()
    with tempfile.TemporaryDirectory(prefix="degeneracy_") as tmp:
        d = Path(tmp)
        # (a) one conformer, three rotamers differing only by a 120-degree methyl rotation
        methyl_h = [3, 4, 5]
        rot = []
        for k in range(3):
            p = pos.copy()
            p[methyl_h] = rotate_about_axis(pos[methyl_h], pos[1], pos[0], k * 2 * math.pi / 3)
            rot.append((-100.0, (sym, p)))
        # (b) one conformer, a chiral pair: the OH hydrogen out of plane, and its mirror
        chir = pos.copy()
        chir[8] = [2.60, 1.90, 0.75]
        mirror = chir.copy()
        mirror[:, 2] *= -1.0
        pair = [(-99.0, (sym, chir)), (-99.0, (sym, mirror))]
        # (c) a single-member group
        single = [(-98.0, (sym, chir))]
        frames = rot + pair + single
        members = [(3, 1, 3), (2, 4, 5), (1, 6, 6)]
        write_rotamers(d / "crest", frames, members)
        res = degeneracy.conformer_degeneracies(d / "crest")
        check("methyl-rotation-only group gives g' = 1 (rotor atoms excluded)",
              res[0]["g_prime"] == 1 and res[0]["n_cores"] == 1, res[0])
        check("a chiral pair gives g' = 2 with the mirror flag",
              res[1]["g_prime"] == 2 and res[1]["mirror_flag"] is True, res[1])
        check("a single-member group is 'unsampled', g' = 1",
              res[2]["g_prime"] == 1 and res[2]["g_prime_source"] == "unsampled")

        # ------------------------------------------------------------ the Calculation
        mol = d / "mol"
        (mol / "crest").mkdir(parents=True)
        for name in ("crest_rotamers.xyz", "cre_members"):
            (mol / "crest" / name).write_text((d / "crest" / name).read_text())
        for b, conf in ((0, 0), (1, 1), (2, 2)):
            bd = layout.mace_basin_dir(mol, b)
            bd.mkdir(parents=True)
            (bd / "basin.extxyz").write_text(
                '9\nconformer={} crest_comment="x"\n'.format(conf)
                + "".join("{} 0 0 0\n".format(s) for s in sym))
        level = "mace-off23_medium"
        out = degeneracy.run_calculation(mol, level=level)
        lvl = layout.level_dir(mol, level)
        check("records land in the level folder, not _records/",
              (lvl / "degeneracy.toml").is_file() and (lvl / "degeneracy.out").is_file()
              and not (mol / "_records").exists())
        doc = prop.load(lvl / "degeneracy.toml")
        first = next(iter(doc))
        check("the Property file starts with [Calculation_Status] = NORMAL TERMINATION",
              first == "Calculation_Status" and doc["Calculation_Status"]["STATUS"] == "NORMAL TERMINATION")
        check("the Report ends with the terminal line",
              report.terminated_normally(lvl / "degeneracy.out", "degeneracy"))
        rows = doc["Basin"]
        check("one [[Basin]] row per basin with G_PRIME (1, 2, 1) mapped by conformer",
              [r["G_PRIME"] for r in rows] == [1, 2, 1] and [r["CONFORMER"] for r in rows] == [0, 1, 2])
        check("out['basins'] agrees with the file", [b["g_prime"] for b in out["basins"]] == [1, 2, 1])

        # ------------------------------------------------------------ missing input
        os.remove(mol / "crest" / "cre_members")
        try:
            degeneracy.run_calculation(mol, level=level)
            check("a missing cre_members raises", False)
        except FileNotFoundError as exc:
            check("a missing cre_members raises naming the file", "cre_members" in str(exc))

    if FAIL:
        print("FAIL: " + ", ".join(FAIL))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
