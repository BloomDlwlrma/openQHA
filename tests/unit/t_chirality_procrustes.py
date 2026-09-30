"""Continuous chirality for g' with the `procrustes` library.

UNIT. No engine. Fixtures: the library's own CHFClBr enantiomer pair
(tests/data/procrustes_chirality, from theochem/procrustes doc/notebooks), the propanal
CREST folder in tests/data/propanal_crest, the two real `crest --entropy` runs
in tests/data/propanal_crest_entropy and the propanal molecule directory.

Asserted:
  * the library's example reproduces its notebook: rotational error 26.09 and orthogonal
    error ~4e-8 -- in bohr^2, which is what IOData's `atcoords` are; in angstrom^2 the
    rotational error is 7.305 (26.09 / 1.8897^2). `kabsch_rmsd` equals
    sqrt(rotational.error / N) to 1e-8 on that pair and on every propanal pair.
  * propanal cis (Cs) has a self-mirror RMSD below the threshold (achiral), gauche above
    (chiral); the gauche rotamer pair is a mirror pair by the library's test
    (RMSD_ortho < threshold <= RMSD_rot) and g' = 2.
  * the two `--entropy` runs, whose CREST labels flipped c1 / cs for the third conformer,
    give it the same g' and self-mirror RMSDs that agree to 5e-3 A, both far below the
    threshold (it is a near-Cs saddle; the number says what the label could not).
  * a structure with more than PERMUTATION_CAP equivalent-atom permutations gets
    `unresolved` and G_PRIME_SOURCE = label_fallback, never an exception.
  * degeneracy.toml carries CHIRALITY, MIRROR_SELF_RMSD, SYMMETRY_LABEL, the RMSD pair
    and PROCRUSTES_VERSION; the expectations are unchanged.
"""
import shutil
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
from openqha.conformer_search import degeneracy as dg       # noqa: E402
from openqha.store import layout, property as prop           # noqa: E402

CHIRAL = ROOT / "tests" / "data" / "procrustes_chirality"
CREST = ROOT / "tests" / "data" / "propanal_crest"
ENTROPY = ROOT / "tests" / "data" / "propanal_crest_entropy" / "msrrho" / "crest_entropy"
MOLECULE = ROOT / "tests" / "data" / "propanal_molecule"
BOHR_PER_A = 1.8897261246
FAIL = []


def check(label, ok, detail=""):
    print("  {:76s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


def _xyz(path):
    lines = Path(path).read_text().splitlines()[2:]
    return np.array([[float(x) for x in l.split()[1:4]] for l in lines if l.strip()])


def main():
    from procrustes import orthogonal, rotational

    # ---- the library's own example -------------------------------------------------
    a, b = _xyz(CHIRAL / "enantiomer1.xyz"), _xyz(CHIRAL / "enantiomer2.xyz")
    e_rot = rotational(a, b, translate=True, scale=False).error
    e_ort = orthogonal(a, b, translate=True, scale=False).error
    check("CHFClBr: rotational error 7.305 A^2 = 26.09 bohr^2 (the notebook's number, IOData coordinates in bohr)",
          abs(e_rot - 7.3047) < 1e-3 and abs(e_rot * BOHR_PER_A ** 2 - 26.0855) < 2e-3, (e_rot, e_rot * BOHR_PER_A ** 2))
    check("CHFClBr: orthogonal error ~1e-8 (reflection allowed: the two are enantiomers)", e_ort < 1e-6, e_ort)
    r_rot, r_ort = dg.procrustes_rmsds(a, b)
    check("procrustes_rmsds returns sqrt(error / N): rot %.4f A, ortho %.2e A" % (r_rot, r_ort),
          abs(r_rot - np.sqrt(e_rot / 5)) < 1e-12 and r_ort < 1e-3)
    check("kabsch_rmsd equals the library's rotational RMSD to 1e-8 on the example",
          abs(dg.kabsch_rmsd(a, b) - r_rot) < 1e-8, (dg.kabsch_rmsd(a, b), r_rot))
    check("checked_rmsd passes (it raises when Kabsch and the library disagree)",
          abs(dg.checked_rmsd(a, b) - r_rot) < 1e-8)
    m = a.copy(); m[:, 0] *= -1.0
    check("a structure against its own reflection: rotational RMSD > 0, orthogonal RMSD = 0",
          dg.procrustes_rmsds(a, m)[0] > 0.5 and dg.procrustes_rmsds(a, m)[1] < 1e-8)

    # ---- propanal: the continuous chirality ------------------------------------------
    confs = dg.conformer_degeneracies(CREST)
    thr = dg.MIRROR_FACTOR * dg.RTHR_A
    cis, gauche, third = confs[0], confs[1], confs[2]
    print("      self-mirror RMSD: cis %.4f  gauche %.4f  third %.4f  (threshold %.4f A)"
          % (cis["mirror_self_rmsd"], gauche["mirror_self_rmsd"], third["mirror_self_rmsd"], thr))
    check("cis (Cs) is achiral by the number: self-mirror RMSD below the threshold",
          cis["chirality"] == "achiral" and cis["mirror_self_rmsd"] < thr)
    check("gauche is chiral by the number: self-mirror RMSD above the threshold",
          gauche["chirality"] == "chiral" and gauche["mirror_self_rmsd"] > thr)
    check("gauche rotamer pair is a mirror pair: RMSD_ortho < threshold <= RMSD_rot, g' = 2",
          gauche["mirror_flag"] and gauche["mirror_rmsd_ortho"] < thr <= gauche["mirror_rmsd_rot"]
          and gauche["g_prime"] == 2 and gauche["g_prime_source"] == "mirror_pair")
    check("the expectations are unchanged: g' = (1, 2, 2), cre_degen2 (3, 6, 3)",
          [c["g_prime"] for c in confs] == [1, 2, 2]
          and [c["cre_degen2_equivalent"] for c in confs] == [3, 6, 3])
    check("the point-group label is still reported beside the number (diagnostic)",
          cis["symmetry_class"] == "achiral" and gauche["symmetry_class"].startswith("chiral"))

    # ---- the two --entropy runs: the c1 / cs flip ------------------------------------
    r1 = dg.conformer_degeneracies(ENTROPY / "run01")
    r2 = dg.conformer_degeneracies(ENTROPY / "run02")
    s1, s2 = r1[2]["mirror_self_rmsd"], r2[2]["mirror_self_rmsd"]
    print("      third conformer self-mirror RMSD: run 1 %.4f  run 2 %.4f A" % (s1, s2))
    check("third conformer: same g' in both runs (%d / %d)" % (r1[2]["g_prime"], r2[2]["g_prime"]),
          r1[2]["g_prime"] == r2[2]["g_prime"] and r1[2]["chirality"] == r2[2]["chirality"])
    check("its self-mirror RMSDs agree to 5e-3 A and both sit far below the threshold (a near-Cs saddle)",
          abs(s1 - s2) < 5e-3 and max(s1, s2) < 0.5 * thr, (s1, s2))
    check("every pair of both runs passed the Kabsch == library assertion (no exception above)", True)

    # ---- the permutation cap ---------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="chirality_cap_") as tmp:
        cube = Path(tmp) / "cube"
        cube.mkdir()
        # eight equivalent carbons on a cube of edge 1.5 A: no rotor group (no pair shares
        # exactly one neighbour), one equivalence class of 8 -> 8! = 40320 permutations
        pts = [(x, y, z) for x in (0.0, 1.5) for y in (0.0, 1.5) for z in (0.0, 1.5)]
        frame = "8\ncube\n" + "".join("C {:.4f} {:.4f} {:.4f}\n".format(*p) for p in pts)
        (cube / "crest_rotamers.xyz").write_text(frame)
        (cube / "cre_members").write_text("1\n1 1 1\n")
        res = dg.conformer_degeneracies(cube, cap=10000)
        check("cube of 8 equivalent atoms: chirality unresolved above the cap, no exception",
              res[0]["chirality"] == "unresolved" and res[0]["mirror_self_rmsd"] is None)
        check("... and g' falls back to the label rule with G_PRIME_SOURCE = label_fallback",
              res[0]["g_prime_source"] == "label_fallback" and res[0]["g_prime"] == 1)
        res2 = dg.conformer_degeneracies(cube, cap=50000)
        check("with the cap raised the cube is resolved achiral (self-mirror RMSD ~0)",
              res2[0]["chirality"] == "achiral" and res2[0]["mirror_self_rmsd"] < 1e-6)

        # ---- the record ------------------------------------------------------------
        mol = Path(tmp) / "dsgdb9nsd_000035"
        shutil.copytree(MOLECULE, mol)
        out = dg.run_calculation(mol, "mace-off23_medium")
        doc = prop.load(layout.level_file(mol, "mace-off23_medium", "degeneracy.toml"))
        info = doc["Calculation_Info"]
        check("degeneracy.toml declares CHIRALITY_THRESHOLD, PERMUTATION_CAP and PROCRUSTES_VERSION",
              abs(info["CHIRALITY_THRESHOLD"] - thr) < 1e-12 and info["PERMUTATION_CAP"] == 10000
              and isinstance(info["PROCRUSTES_VERSION"], str) and info["PROCRUSTES_VERSION"] != "unknown")
        rows = {int(r["INDEX"]): r for r in doc["Basin"]}
        check("every [[Basin]] row carries CHIRALITY, MIRROR_SELF_RMSD and SYMMETRY_LABEL",
              all("CHIRALITY" in r and "MIRROR_SELF_RMSD" in r and "SYMMETRY_LABEL" in r for r in rows.values()))
        check("real propanal: g' = (1, 1, 1) with the pair counted once",
              [rows[i]["G_PRIME"] for i in (0, 1, 2)] == [1, 1, 1]
              and rows[1]["G_PRIME_SOURCE"] == "mirror_is_basin" and rows[2]["G_PRIME_SOURCE"] == "mirror_is_basin")
        text = (layout.level_file(mol, "mace-off23_medium", "degeneracy.out")).read_text(encoding="utf-8")
        check("the Report cites meng2022procrustes and prints the threshold", "meng2022procrustes" in text
              and "chirality threshold" in text)
    if FAIL:
        print("FAIL: " + ", ".join(FAIL))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
