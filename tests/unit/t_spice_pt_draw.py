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

        # --- refusals ---------------------------------------------------------------------------------------
        rc, _s, se = run("--n", 1, "--source", td / "no_such.xyz", "--forgetting-ids", absent_ids, "--out", td / "x.extxyz")
        check("an absent source exits 2 naming the DOI", rc == 2 and "10.17863/CAM.107498" in se, se[-300:])
        rc, _s, se = run("--n", 1, "--source", SRC, "--forgetting-ids", td / "no_ids.dat", "--out", td / "x.extxyz")
        check("an absent forgetting ids file exits 2 naming the test-draw tool",
              rc == 2 and "s0_spice_test_draw" in se and not (td / "x.extxyz").is_file(), se[-300:])

    print("\n{} checks, {} failed".format(19, len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
