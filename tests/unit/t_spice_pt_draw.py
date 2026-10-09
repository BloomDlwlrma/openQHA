"""`scripts/tooling/s0_spice_pt_draw.py`,
the Replay draw, on `tests/data/spice_tiny` (7 train frames: propanal x3, acetone x2, a
propanal-water dimer, 2-methyloxirane).

Asserted: with a benign membership table (the in_distribution ids mapped to SMILES absent
from the fixture) and a forgetting ids file naming an absent molecule, `--n 3 --seed 0`
writes the first three frames of `default_rng(0).permutation(7)` with `config_weight`,
`REF_energy` / `REF_forces`, no Hessian, an ids file with the per-molecule count, a
companion validation file from the END of the permutation (disjoint from the draw) and a
`[Replay]` Record; `--n 2` is the prefix of `--n 4` (nesting); `--weight 10` lands on
every frame; (a) a forgetting ids file naming propanal skips every propanal frame (the
dimer too, by fragment) and `check_disjoint` raises on a constructed overlap; (b) with
the REAL membership table (propanal and acetone are in_distribution) only the
methyloxirane frame is eligible: `--n 3` exits 2 and writes nothing, `--n 1` writes it;
an absent source exits 2 naming the DOI; an absent forgetting ids file exits 2.

Also the element filter, on a second fixture (`tests/data/spice_tiny/elements_mix.xyz`:
methane, chloromethane, water, thiophene, fluoromethane, methanol): `--elements H,C,N,O,F`
skips the Cl- and S-carrying frames only, and the draw plus its companion valid file are
the eligible frames of the same seeded permutation; nesting holds under the filter; the
Record and the stdout carry `ELEMENTS` / `N_SKIPPED_ELEMENTS`; a set no frame matches and a
malformed list both exit 2; a forgetting ids file naming water composes with the filter
(water becomes a `forgetting` skip, the Cl/S frames `element` skips); without the flag the
Record's new fields read `-` and 0.
"""
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "tooling"))
from openqha.store import dat, property as prop                  # noqa: E402
import s0_spice_pt_draw as tool                                  # noqa: E402

TOOL = ROOT / "scripts" / "tooling" / "s0_spice_pt_draw.py"
SRC = ROOT / "tests" / "data" / "spice_tiny" / "train_large_neut_no_bad_clean.xyz"
FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:300]))
    if not ok:
        FAIL.append(label)


def run(*args):
    p = subprocess.run([sys.executable, str(TOOL)] + [str(a) for a in args], capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def write_ids(path, smiles_list):
    rows = [dict(index=i, smiles=s, config_type="-", n_atoms=1) for i, s in enumerate(smiles_list)]
    cols = {"index": ("Integer", None, "i"), "smiles": ("String", None, "s"),
            "config_type": ("String", None, "c"), "n_atoms": ("Integer", None, "n")}
    dat.write_table(path, rows, list(cols), cols)


def write_membership(path, smiles_by_qid):
    from openqha_hessian import judge
    cols = {"qm9_index": ("String", None, "q"), "smiles": ("String", None, "s"), "n_heavy": ("Integer", None, "n"),
            "in_training": ("Boolean", None, "t")}
    rows = [dict(qm9_index=q, smiles=smiles_by_qid[q], n_heavy=2, in_training=True) for q in judge.IN_DISTRIBUTION]
    dat.write_table(path, rows, list(cols), cols)


def main():
    import numpy as np
    from ase.io import read
    from openqha_hessian import judge
    perm = np.random.default_rng(0).permutation(7).tolist()
    with tempfile.TemporaryDirectory(prefix="ptdraw_") as td:
        td = Path(td)
        benign = td / "membership.dat"
        write_membership(benign, {q: "CCO" for q in judge.IN_DISTRIBUTION})    # ethanol: not in the fixture
        absent_ids = td / "forget_absent.ids.dat"
        write_ids(absent_ids, ["CCCCO"])                                       # butanol: not in the fixture

        # --- the draw ----------------------------------------------------------------------------------
        out = td / "pt3.extxyz"
        rc, so, se = run("--n", 3, "--seed", 0, "--source", SRC, "--forgetting-ids", absent_ids, "--out", out,
                         "--membership-file", benign, "--n-valid", 2)
        check("--n 3 --seed 0 exits 0 and writes the file, the ids, the valid file and the Record",
              rc == 0 and out.is_file() and (td / "pt3.extxyz.ids.dat").is_file() and (td / "pt3.valid.extxyz").is_file()
              and (td / "pt3.toml").is_file(), (rc, se[-500:]))
        frames = read(str(out), index=":", format="extxyz")
        idx = [int(a.info["spice_index"]) for a in frames]
        check("the frames are the first three of default_rng(0).permutation(7), in permutation order",
              idx == perm[:3], (idx, perm))
        check("every frame carries config_weight 1.0, REF_energy, REF_forces and no Hessian",
              all(a.info.get("config_weight") == 1.0 and "REF_energy" in a.info and "REF_forces" in a.arrays
                  and "REF_hessian" not in a.info and "hessian" not in a.info for a in frames))
        ids = dat.read_table(td / "pt3.extxyz.ids.dat")
        check("the ids file: one row per frame with index, smiles, molecule_key and frames_of_molecule adding up",
              [r["index"] for r in ids] == idx and all(r["molecule_key"] != "-" for r in ids)
              and all(r["frames_of_molecule"] == sum(1 for q in ids if q["molecule_key"] == r["molecule_key"]) for r in ids), ids)
        valid = read(str(td / "pt3.valid.extxyz"), index=":", format="extxyz")
        vidx = [int(a.info["spice_index"]) for a in valid]
        check("the valid file: 2 frames from the END of the permutation, disjoint from the draw",
              vidx == perm[::-1][:2] and not set(vidx) & set(idx), (vidx, perm))
        rec = prop.load(td / "pt3.toml")["Replay"]
        check("[Replay]: DOI, SPLIT train, SEED 0, N 3, WEIGHT 1, N_MOLECULES, FRAMES_PER_MOLECULE_MAX, the exclusions",
              rec["DOI"] == "10.17863/CAM.107498" and rec["SPLIT"] == "train" and rec["SEED"] == 0 and rec["N"] == 3
              and rec["WEIGHT"] == 1.0 and rec["N_MOLECULES"] == len({r["molecule_key"] for r in ids})
              and rec["FRAMES_PER_MOLECULE_MAX"] == max(r["frames_of_molecule"] for r in ids)
              and rec["N_SKIPPED_FORGETTING"] == 0 and rec["N_SKIPPED_IN_DISTRIBUTION"] == 0 and rec["N_ELIGIBLE"] == 7
              and rec["N_VALID"] == 2, rec)
        check("the stdout names the fine-tune line with both files", "--pt-train-file" in so and "--pt-valid-file" in so)

        # --- nesting and the weight ---------------------------------------------------------------------
        rc2, _s, _e = run("--n", 2, "--seed", 0, "--source", SRC, "--forgetting-ids", absent_ids, "--out", td / "pt2.extxyz",
                          "--membership-file", benign, "--n-valid", 1)
        rc4, _s, _e = run("--n", 4, "--seed", 0, "--source", SRC, "--forgetting-ids", absent_ids, "--out", td / "pt4.extxyz",
                          "--membership-file", benign, "--n-valid", 1, "--weight", 10)
        i2 = [int(a.info["spice_index"]) for a in read(str(td / "pt2.extxyz"), index=":", format="extxyz")]
        f4 = read(str(td / "pt4.extxyz"), index=":", format="extxyz")
        i4 = [int(a.info["spice_index"]) for a in f4]
        check("nesting: --n 2 from seed 0 is the first two frames of --n 4 from seed 0", rc2 == 0 and rc4 == 0 and i4[:2] == i2 == perm[:2], (i2, i4))
        check("--weight 10 lands on every frame", all(a.info["config_weight"] == 10.0 for a in f4))
        rc1, _s, _e = run("--n", 2, "--seed", 1, "--source", SRC, "--forgetting-ids", absent_ids, "--out", td / "s1.extxyz",
                          "--membership-file", benign, "--n-valid", 1)
        i1 = [int(a.info["spice_index"]) for a in read(str(td / "s1.extxyz"), index=":", format="extxyz")]
        check("another seed is another permutation", rc1 == 0 and i1 == np.random.default_rng(1).permutation(7).tolist()[:2], i1)

        # --- (a) the forgetting draw's molecules are excluded, by fragment ---------------------------------
        propanal_ids = td / "forget_propanal.ids.dat"
        write_ids(propanal_ids, ["CCC=O"])
        rc, _s, se = run("--n", 3, "--seed", 0, "--source", SRC, "--forgetting-ids", propanal_ids, "--out", td / "fa.extxyz",
                         "--membership-file", benign, "--n-valid", 0)
        fa = read(str(td / "fa.extxyz"), index=":", format="extxyz")
        rec = prop.load(td / "fa.toml")["Replay"]
        check("(a) with propanal in the forgetting draw: the 4 propanal frames (0-2 and the dimer 3) are skipped, 3 drawn from the other 3",
              rc == 0 and len(fa) == 3 and rec["N_SKIPPED_FORGETTING"] == 4 and rec["N_ELIGIBLE"] == 3
              and sorted(int(a.info["spice_index"]) for a in fa) == [4, 5, 6],
              ([a.info["spice_index"] for a in fa], rec))
        rc, _s, se = run("--n", 4, "--seed", 0, "--source", SRC, "--forgetting-ids", propanal_ids, "--out", td / "fb.extxyz",
                         "--membership-file", benign)
        check("... and --n 4 cannot be drawn from 3 eligible frames: exit 2, nothing written",
              rc == 2 and not (td / "fb.extxyz").is_file() and "eligible" in se, (rc, se[-300:]))
        cache = {}
        try:
            tool.check_disjoint(["AAA;BBB", "CCC"], {"BBB"}, set())
            check("check_disjoint raises on a shared molecule, naming it", False)
        except AssertionError as exc:
            check("check_disjoint raises on a shared molecule, naming it", "(a)" in str(exc) and "BBB" in str(exc), str(exc))
        try:
            tool.check_disjoint(["AAA"], set(), {"AAA"})
            check("check_disjoint raises on an in_distribution molecule", False)
        except AssertionError as exc:
            check("check_disjoint raises on an in_distribution molecule", "(b)" in str(exc) and "AAA" in str(exc))
        tool.check_disjoint(["AAA;BBB"], {"CCC"}, {"DDD"})
        key_dimer = tool.molecule_key("[C:1]([C:2]([C:3](=[O:4])[H:11])([H:9])[H:10])([H:6])([H:7])[H:8].[O:5]([H:12])[H:13]", cache)
        key_prop = tool.molecule_key("CCC=O", cache)
        check("molecule_key: the propanal-water dimer names propanal as a fragment (connectivity level); an unreadable SMILES is None",
              key_prop in key_dimer.split(";") and len(key_dimer.split(";")) == 2 and tool.molecule_key("not a smiles", cache) is None)

        # --- (b) the real membership: propanal and acetone are in_distribution ------------------------------
        rc, _s, se = run("--n", 3, "--seed", 0, "--source", SRC, "--forgetting-ids", absent_ids, "--out", td / "ib.extxyz")
        check("(b) with the real membership only the methyloxirane frame is eligible: --n 3 exits 2, nothing written",
              rc == 2 and not (td / "ib.extxyz").is_file() and "in_distribution" in se, (rc, se[-300:]))
        rc, _s, se = run("--n", 1, "--seed", 0, "--source", SRC, "--forgetting-ids", absent_ids, "--out", td / "ib1.extxyz",
                         "--n-valid", 0)
        f1 = read(str(td / "ib1.extxyz"), index=":", format="extxyz")
        rec = prop.load(td / "ib1.toml")["Replay"]
        check("... --n 1 writes the methyloxirane frame (spice_index 6) and records 6 frames skipped as in_distribution",
              rc == 0 and len(f1) == 1 and int(f1[0].info["spice_index"]) == 6 and rec["N_SKIPPED_IN_DISTRIBUTION"] == 6
              and rec["N_IN_DISTRIBUTION"] == 4, (rc, rec))

        # --- the element filter: tests/data/spice_tiny/elements_mix.xyz ------------------------------------
        mix = ROOT / "tests" / "data" / "spice_tiny" / "elements_mix.xyz"
        perm6 = np.random.default_rng(0).permutation(6).tolist()
        allowed = {"H", "C", "N", "O", "F"}
        outside6 = {1, 3}                       # chloromethane carries Cl; thiophene carries S
        elig6 = [i for i in perm6 if i not in outside6]
        rc, so, se = run("--n", 4, "--seed", 0, "--source", mix, "--elements", "H,C,N,O,F",
                         "--forgetting-ids", absent_ids, "--out", td / "el4.extxyz",
                         "--membership-file", benign, "--n-valid", 0)
        el = read(str(td / "el4.extxyz"), index=":", format="extxyz")
        check("--elements H,C,N,O,F: --n 4 from seed 0 draws the first 4 eligible frames (the Cl- and S-frames skipped)",
              rc == 0 and [int(a.info["spice_index"]) for a in el] == elig6[:4], (rc, se[-300:]))
        check("every written frame's atoms stay inside the declared element set",
              all(set(a.get_chemical_symbols()) <= allowed for a in el))
        rec = prop.load(td / "el4.toml")["Replay"]
        check("the Record carries ELEMENTS and N_SKIPPED_ELEMENTS beside the other classes",
              rec["ELEMENTS"] == "H,C,N,O,F" and rec["N_SKIPPED_ELEMENTS"] == 2 and rec["N_ELIGIBLE"] == 4
              and rec["N_SOURCE"] == 6 and rec["N"] == 4, rec)
        check("the stdout names the element skips", "outside the declared elements H,C,N,O,F" in so, so[-400:])
        ids_txt = (td / "el4.extxyz.ids.dat").read_text(encoding="utf-8")
        check("the ids file's header names the filter", "# elements H,C,N,O,F" in ids_txt, ids_txt[:200])
        rc, _s, se = run("--n", 2, "--seed", 0, "--source", mix, "--elements", "H,C,N,O,F",
                         "--forgetting-ids", absent_ids, "--out", td / "el2.extxyz",
                         "--membership-file", benign, "--n-valid", 2)
        iel2 = [int(a.info["spice_index"]) for a in read(str(td / "el2.extxyz"), index=":", format="extxyz")]
        check("nesting under the filter: --n 2 from seed 0 is the prefix of --n 4",
              rc == 0 and iel2 == elig6[:2] and iel2 == [int(a.info["spice_index"]) for a in el[:2]], (iel2, elig6))
        vel2 = [int(a.info["spice_index"]) for a in read(str(td / "el2.valid.extxyz"), index=":", format="extxyz")]
        check("the companion valid file under the filter: 2 eligible frames from the end, disjoint from the draw",
              vel2 == elig6[::-1][:2] and not set(vel2) & set(iel2), (vel2, elig6))
        rc, _s, se = run("--n", 1, "--seed", 0, "--source", mix, "--elements", "N",
                         "--forgetting-ids", absent_ids, "--out", td / "elN.extxyz", "--membership-file", benign)
        check("a set no frame matches: exit 2, nothing written, the message names the element skips",
              rc == 2 and not (td / "elN.extxyz").is_file() and "outside the declared elements N" in se, (rc, se[-300:]))
        rc, _s, se = run("--n", 1, "--seed", 0, "--source", mix, "--elements", "h,C",
                         "--forgetting-ids", absent_ids, "--out", td / "elbad.extxyz", "--membership-file", benign)
        check("a malformed --elements list exits 2 naming the form",
              rc == 2 and not (td / "elbad.extxyz").is_file() and "element symbols" in se, (rc, se[-300:]))
        rc, _s, se = run("--n", 1, "--seed", 0, "--source", mix, "--elements", "H,,C",
                         "--forgetting-ids", absent_ids, "--out", td / "elempty.extxyz", "--membership-file", benign)
        check("an empty token in the list exits 2 too",
              rc == 2 and not (td / "elempty.extxyz").is_file() and "element symbols" in se, (rc, se[-300:]))
        water_ids = td / "forget_water.ids.dat"
        write_ids(water_ids, ["O"])
        rc, _s, _e = run("--n", 3, "--seed", 0, "--source", mix, "--elements", "H,C,N,O,F",
                         "--forgetting-ids", water_ids, "--out", td / "elw3.extxyz",
                         "--membership-file", benign, "--n-valid", 0)
        fw = [int(a.info["spice_index"]) for a in read(str(td / "elw3.extxyz"), index=":", format="extxyz")]
        rec = prop.load(td / "elw3.toml")["Replay"]
        check("with water in the forgetting draw the filter composes: water is `forgetting`, the Cl/S frames `element`",
              rc == 0 and fw == [i for i in perm6 if i not in (1, 2, 3)][:3]
              and rec["N_SKIPPED_FORGETTING"] == 1 and rec["N_SKIPPED_ELEMENTS"] == 2 and rec["N_ELIGIBLE"] == 3,
              (fw, rec))
        rec = prop.load(td / "pt3.toml")["Replay"]
        check("no --elements: the new Record fields read '-' and 0",
              rec["ELEMENTS"] == "-" and rec["N_SKIPPED_ELEMENTS"] == 0, rec)

        # --- the coverage rule: min1 on the 7-frame fixture ------------------------------------------------
        groups = ({0, 1, 2}, {3}, {4, 5}, {6})          # propanal, the dimer, acetone, methyloxirane
        rank = {f: r for r, f in enumerate(perm)}
        cov4 = sorted((min(g, key=rank.get) for g in groups), key=rank.get)
        rest = [i for i in perm if i not in set(cov4)]
        rc, _s, se = run("--n", 4, "--seed", 0, "--source", SRC, "--forgetting-ids", absent_ids,
                         "--out", td / "cv4.extxyz", "--membership-file", benign, "--n-valid", 0,
                         "--coverage", "min1")
        cv4 = [int(a.info["spice_index"]) for a in read(str(td / "cv4.extxyz"), index=":", format="extxyz")]
        rec = prop.load(td / "cv4.toml")["Replay"]
        check("--coverage min1: --n 4 draws each of the 4 molecules once -- its first frame in the permutation",
              rc == 0 and cv4 == cov4 and rec["COVERAGE"] == "min1" and rec["N_MOLECULES"] == 4
              and rec["N_ELIGIBLE_MOLECULES"] == 4, (cv4, cov4, rec))
        rc, _s, se = run("--n", 6, "--seed", 0, "--source", SRC, "--forgetting-ids", absent_ids,
                         "--out", td / "cv6.extxyz", "--membership-file", benign, "--n-valid", 0,
                         "--coverage", "min1")
        cv6 = [int(a.info["spice_index"]) for a in read(str(td / "cv6.extxyz"), index=":", format="extxyz")]
        check("the fill tier: --n 6 is the coverage frames then the next two of the permutation",
              rc == 0 and cv6 == cov4 + rest[:2] and set(cv4) <= set(cv6), (cv6, perm))
        rc, _s, se = run("--n", 5, "--seed", 0, "--source", SRC, "--forgetting-ids", absent_ids,
                         "--out", td / "cv5.extxyz", "--membership-file", benign, "--n-valid", 2,
                         "--coverage", "min1")
        sel5 = cov4 + rest[:1]
        vex = []
        for i in perm[::-1]:
            if len(vex) >= 2 or i in set(sel5):
                break
            vex.append(i)
        cv5 = [int(a.info["spice_index"]) for a in read(str(td / "cv5.extxyz"), index=":", format="extxyz")]
        v5 = ([int(a.info["spice_index"]) for a in read(str(td / "cv5.valid.extxyz"), index=":", format="extxyz")]
              if (td / "cv5.valid.extxyz").is_file() else [])
        rec = prop.load(td / "cv5.toml")["Replay"]
        check("min1 with a valid companion: the valid frames come from the permutation's end, disjoint from a scattered draw",
              rc == 0 and cv5 == sel5 and v5 == vex and rec["N_VALID"] == len(vex)
              and not set(v5) & set(sel5), (cv5, vex, rec))
        rc, _s, se = run("--n", 3, "--seed", 0, "--source", SRC, "--forgetting-ids", absent_ids,
                         "--out", td / "cv3.extxyz", "--membership-file", benign, "--coverage", "min1")
        check("min1 with --n below the molecule count exits 2, naming both numbers",
              rc == 2 and not (td / "cv3.extxyz").is_file() and "min1" in se
              and "the 4 molecules" in se and "got --n 3" in se, (rc, se[-300:]))
        rec = prop.load(td / "pt3.toml")["Replay"]
        check("without --coverage the Rule reads by-frame and the eligible molecules are on the Record",
              rec["COVERAGE"] == "by-frame" and rec["N_ELIGIBLE_MOLECULES"] == 4, rec)

        # --- refusals ---------------------------------------------------------------------------------------
        rc, _s, se = run("--n", 1, "--source", td / "no_such.xyz", "--forgetting-ids", absent_ids, "--out", td / "x.extxyz")
        check("an absent source exits 2 naming the DOI", rc == 2 and "10.17863/CAM.107498" in se, se[-300:])
        rc, _s, se = run("--n", 1, "--source", SRC, "--forgetting-ids", td / "no_ids.dat", "--out", td / "x.extxyz")
        check("an absent forgetting ids file exits 2 naming the test-draw tool",
              rc == 2 and "s0_spice_test_draw" in se and not (td / "x.extxyz").is_file(), se[-300:])

    print("\n{} checks, {} failed".format(36, len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
