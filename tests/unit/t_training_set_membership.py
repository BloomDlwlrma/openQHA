"""Ticket 30: is the molecule in MACE-OFF23's training set? (step 1, the shipped species)

UNIT. RDKit only, no engine, seconds. tests/data/spice_tiny holds two SPICE-format files
(atom-mapped explicit-H SMILES in extxyz headers, as the released MACE-OFF23 data carries
them): propanal x3 untagged + once as a water dimer, acetone x2 as DES370K monomers,
(R)-2-methyloxirane once; the test file holds propanal once and ethanol once.

Asserted:
  * the canonicaliser maps the SPICE form and the plain QM9 SMILES of one molecule to one
    key at all three strictnesses; the R/S pair of 2-methyloxirane matches at no_stereo
    and connectivity but not at isomeric.
  * the index is built once, cached, reused (no re-read), and a changed source file
    invalidates it.
  * propanal: IN_TRAINING, isomeric match, 3 train frames + 1 dimer frame, 1 test frame;
    acetone: 2 monomer frames, config type DES370K Monomers; (S)-2-methyloxirane: match
    at no_stereo; ethanol: test only; a molecule absent everywhere: false / none, no
    exception.
  * level_compare writes [Training_Set] with the index configured, and states the check
    was not run without it (never a silent false).
"""
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from openqha import config                                   # noqa: E402
from openqha.data import training_set as ts                  # noqa: E402
from openqha.store import layout, dat, property as prop              # noqa: E402

TINY = ROOT / "tests" / "data" / "spice_tiny"
MOLECULE = ROOT / "tests" / "data" / "propanal_molecule"
FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


def main():
    spice = "[H:5][C:1](=[C:2]([C:3]([H:7])([H:8])[H:9])[C:4]([H:10])([H:11])[H:12])[H:6]"
    plain = "CC(C)=C"
    k1, k2 = ts.canonical_keys(spice), ts.canonical_keys(plain)
    check("SPICE atom-mapped explicit-H SMILES and the plain SMILES give one key at all three levels",
          k1 == k2 and set(k1) == {"isomeric", "no_stereo", "connectivity"})
    r, s = ts.canonical_keys("C[C@H]1CO1"), ts.canonical_keys("C[C@@H]1CO1")
    check("2-methyloxirane R / S: different at isomeric, equal at no_stereo and connectivity",
          r["isomeric"] != s["isomeric"] and r["no_stereo"] == s["no_stereo"]
          and r["connectivity"] == s["connectivity"])
    check("an unparsable SMILES gives None, not an exception", ts.canonical_keys("C(C") is None)
    fr = ts.fragment_keys("CCC=O.O")
    check("a dimer SMILES yields one key set per fragment", len(fr) == 2 and fr[0]["isomeric"] == "CCC=O")

    with tempfile.TemporaryDirectory(prefix="training_set_") as tmp:
        root = Path(tmp) / "spice"
        shutil.copytree(TINY, root)
        cache = Path(tmp) / "cache" / "index.dat"
        files = {"train": root / "train_large_neut_no_bad_clean.xyz", "test": root / "test_large_neut_all.xyz"}
        check("no index yet: index_is_current is False (sources present)",
              ts.index_is_current(files=files, cache_path=cache) is False)
        p = ts.build_index(files=files, cache_path=cache, doi="test")
        tables = dat.read_tables(p)
        src = {r["FILE"]: r for r in tables["Source"]}
        check("the index records both sources with frame counts (7 train, 2 test) and sizes",
              src["train"]["N_FRAMES_TOTAL"] == 7 and src["test"]["N_FRAMES_TOTAL"] == 2
              and src["train"]["SIZE"] == files["train"].stat().st_size)
        check("index_is_current is True right after the build", ts.index_is_current(files=files, cache_path=cache) is True)
        mtime = cache.stat().st_mtime
        rec = ts.membership("CCC=O", cache_path=cache)
        check("propanal: IN_TRAINING, isomeric match, 3 monomer + 1 dimer train frames, 1 test frame",
              rec["IN_TRAINING"] and rec["MATCH_LEVEL"] == "isomeric" and rec["N_TRAIN_FRAMES"] == 3
              and rec["N_TRAIN_DIMER_FRAMES"] == 1 and rec["N_TEST_FRAMES"] == 1
              and rec["CONFIG_TYPES"] == ["DES370K Dimers", "untagged"] and rec["N_HEAVY_ATOMS"] == 4, rec)
        check("the second query reused the cached index (file untouched)", cache.stat().st_mtime == mtime)
        rec = ts.membership("CC(=O)C", cache_path=cache)
        check("acetone: 2 monomer frames under DES370K Monomers, no dimer frames",
              rec["N_TRAIN_FRAMES"] == 2 and rec["N_TRAIN_DIMER_FRAMES"] == 0
              and rec["CONFIG_TYPES"] == ["DES370K Monomers"])
        rec = ts.membership("C[C@H]1CO1", cache_path=cache)
        check("(S)-2-methyloxirane against the (R) frame: IN_TRAINING at no_stereo, not isomeric",
              rec["IN_TRAINING"] and rec["MATCH_LEVEL"] == "no_stereo"
              and rec["MATCH_ISOMERIC"] is False and rec["MATCH_NO_STEREO"] is True)
        rec = ts.membership("CCO", cache_path=cache)
        check("ethanol: in the test file only (IN_TRAINING false, IN_TEST_ONLY true)",
              rec["IN_TRAINING"] is False and rec["IN_TEST_ONLY"] is True and rec["N_TEST_FRAMES"] == 1)
        rec = ts.membership("c1ccccc1", cache_path=cache)
        check("benzene: absent everywhere -> IN_TRAINING false, MATCH_LEVEL none, no exception",
              rec["IN_TRAINING"] is False and rec["MATCH_LEVEL"] == "none" and rec["N_TRAIN_FRAMES"] == 0)
        check("the sentence says in-distribution for propanal and out-of-distribution for benzene",
              "in-distribution" in ts.sentence(ts.membership("CCC=O", cache_path=cache))
              and "out-of-distribution" in ts.sentence(rec))
        # a changed source invalidates the cache
        time.sleep(1.1)
        with open(files["test"], "a") as fh:
            fh.write("")
        os.utime(files["test"], None)
        check("a touched source file makes the cached index stale", ts.index_is_current(files=files, cache_path=cache) is False)

        # ---- level_compare: [Training_Set] present with the index, stated absent without
        from openqha.thermochem import msrrho_ensemble as me
        from openqha.thermochem import reference_level as rl
        mol = Path(tmp) / "dsgdb9nsd_000035"
        shutil.copytree(MOLECULE, mol)
        me.run_calculation(mol, level="mace-off23_medium", qm9_index="dsgdb9nsd_000035")
        cfg = config.load()
        cfg.setdefault("data", {})["training_sets"] = {"mace_off23": {
            "root": str(root), "cache_dir": str(Path(tmp) / "cache2"), "doi": "test"}}
        out = rl.level_compare(mol, qm9_index="dsgdb9nsd_000035", cfg=cfg)
        doc = prop.load(layout.thermo_file(mol, "level_compare.toml"))
        check("level_compare writes [Training_Set] for propanal from the configured files (index built on demand)",
              "Training_Set" in doc and doc["Training_Set"]["IN_TRAINING"] is True
              and doc["Training_Set"]["MATCH_LEVEL"] == "isomeric" and doc["Training_Set"]["SOURCE"] == "mace-off23_spice"
              and (Path(tmp) / "cache2" / "mace-off23_spice_index.dat").is_file())
        text = (layout.thermo_file(mol, "level_compare.out")).read_text(encoding="utf-8")
        check("the Report says the tiers are in-distribution numbers", "in-distribution" in text)
        cfg["data"]["training_sets"] = {"mace_off23": {"root": str(Path(tmp) / "nowhere"),
                                                        "cache_dir": str(Path(tmp) / "cache3")}}
        rl.level_compare(mol, qm9_index="dsgdb9nsd_000035", cfg=cfg)
        doc = prop.load(layout.thermo_file(mol, "level_compare.toml"))
        text = (layout.thermo_file(mol, "level_compare.out")).read_text(encoding="utf-8")
        check("without files or index: no [Training_Set] block, and the Report says the check was not run",
              "Training_Set" not in doc and "not checked" in text)
    if FAIL:
        print("FAIL: " + ", ".join(FAIL))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
