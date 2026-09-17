"""Ticket 31: training-set membership over the whole target QM9 set (the Dataset).

UNIT. No engine, seconds. A synthetic curated-QM9 directory (S0_CURATED_QM9) of six
QM9-format files -- propanal, acetone, ethanol, benzene, a two-fragment SMILES (fails
F1), and one file without a SMILES line -- is screened with the configured gates and
queried against the tiny SPICE index of tests/data/spice_tiny.

Asserted: the target set is the gate's (the gate list and F7 mode are in the record);
every target has one row and a failing molecule is counted by gate; the summary's
counts, fraction and per-heavy-atom arrays follow; the SPICE provenance (files, sizes,
frame counts, DOI) is in the record; the table round-trips through dat.read_table; the
shipped species' answers agree with ticket 30's.
"""
import os
import shutil
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
from openqha import config                                   # noqa: E402
from openqha.conformer_search import filters                 # noqa: E402
from openqha.data import curated_qm9, training_set as ts     # noqa: E402
from openqha.store import dat, property as prop              # noqa: E402

TINY = ROOT / "tests" / "data" / "spice_tiny"
FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


def qm9_file(path, n, smiles):
    """A QM9-format file: count, property line, n atom lines, frequencies, two SMILES,
    two InChI. Only the SMILES line matters here."""
    lines = [str(n), "gdb 1\t0 0 0"] + ["C 0.0 0.0 {:.1f} 0.0".format(0.1 * i) for i in range(n)]
    lines += ["100.0", "{}\t{}\t".format(smiles, smiles), "InChI=1S/x\tInChI=1S/x"]
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    with tempfile.TemporaryDirectory(prefix="qm9_targets_") as tmp:
        qm9 = Path(tmp) / "curated"
        qm9.mkdir()
        qm9_file(qm9 / "dsgdb9nsd_000035.xyz", 4, "CCC=O")          # in SPICE (tiny)
        qm9_file(qm9 / "dsgdb9nsd_000018.xyz", 4, "CC(=O)C")        # in SPICE (tiny)
        qm9_file(qm9 / "dsgdb9nsd_000100.xyz", 3, "CCO")            # test file only
        qm9_file(qm9 / "dsgdb9nsd_000200.xyz", 6, "c1ccccc1")       # absent
        qm9_file(qm9 / "dsgdb9nsd_000300.xyz", 5, "CCC=O.O")        # two fragments: fails F1
        (qm9 / "dsgdb9nsd_000400.xyz").write_text("2\ngdb 400\nC 0 0 0 0\nC 0 0 1 0\n")  # no SMILES line
        os.environ["S0_CURATED_QM9"] = str(qm9)
        curated_qm9._INDEX_CACHE.clear()
        cfg = config.load()
        cfg.setdefault("data", {})["training_sets"] = {"mace_off23": {
            "root": str(TINY), "cache_dir": str(Path(tmp) / "cache"), "doi": "test-doi"}}
        # the gate as configured, minus F7 (which needs the uncharacterized list and the
        # real archive): the record must say exactly which gates ran
        cfg.setdefault("species_filter", {})["f7_mode"] = "off"
        gates = tuple(g for g in filters.enabled_gates(cfg) if g != "F7")
        out = ts.qm9_targets_membership(cfg, out_dir=Path(tmp) / "out")
        sm, info = out["summary"], out["info"]
        check("six files considered: four targets, one failed a gate (F1), one unreadable",
              sm["N_CONSIDERED"] == 6 and sm["N_TARGETS"] == 4 and sm["N_FAILED_GATE"] == 1
              and out["failed"] == {"F1": 1} and sm["N_UNREADABLE"] == 1, (sm, out["failed"]))
        check("the gates applied and the F7 mode are in the record",
              tuple(info["GATES"]) == gates and info["F7_MODE"] == "off")
        check("in training: propanal and acetone (2 of 4, fraction 0.5); ethanol test-only; benzene absent",
              sm["N_IN_TRAINING"] == 2 and abs(sm["FRACTION_IN_TRAINING"] - 0.5) < 1e-12
              and sm["N_IN_TEST_ONLY"] == 1 and sm["N_IN_TRAINING_ISOMERIC"] == 2)
        check("per heavy-atom arrays: targets [0,0,1,2,0,1,0,0,0], in training [0,0,0,2,0,0,0,0,0]",
              sm["TARGETS_BY_HEAVY"] == [0, 0, 1, 2, 0, 1, 0, 0, 0]
              and sm["IN_TRAINING_BY_HEAVY"] == [0, 0, 0, 2, 0, 0, 0, 0, 0], (sm["TARGETS_BY_HEAVY"], sm["IN_TRAINING_BY_HEAVY"]))
        check("monomer-only / dimer-only counted (acetone monomer-only; propanal has both)",
              sm["N_MONOMER_ONLY"] == 1 and sm["N_DIMER_ONLY"] == 0)
        check("config_type counts: DES370K Monomers 1 (acetone), Dimers 1 and untagged 1 (propanal)",
              out["config_types"] == {"DES370K Monomers": 1, "DES370K Dimers": 1, "untagged": 1}, out["config_types"])
        doc = prop.load(out["record"])
        check("the record names the SPICE files, sizes, frame counts (7 / 2) and the DOI",
              doc["Calculation_Info"]["TRAIN_FRAMES"] == 7 and doc["Calculation_Info"]["TEST_FRAMES"] == 2
              and doc["Calculation_Info"]["DOI"] == "test-doi" and doc["Calculation_Info"]["TRAIN_SIZE"] > 0
              and doc["Calculation_Info"]["SMILES_SOURCE"] == "relaxed")
        check("the record has the four blocks and starts with [Calculation_Status]",
              set(doc) == {"Calculation_Status", "Calculation_Info", "Summary", "Failed_Gate", "Config_Type"}
              and next(iter(doc)) == "Calculation_Status")
        rows = dat.read_table(out["table"])
        check("the table has one row per target and round-trips (booleans, integers, strings)",
              sorted(r["qm9_index"] for r in rows) == ["dsgdb9nsd_000018", "dsgdb9nsd_000035", "dsgdb9nsd_000100", "dsgdb9nsd_000200"]
              and next(r for r in rows if r["qm9_index"] == "dsgdb9nsd_000035")["in_training"] is True
              and next(r for r in rows if r["qm9_index"] == "dsgdb9nsd_000035")["n_train_dimer_frames"] == 1
              and next(r for r in rows if r["qm9_index"] == "dsgdb9nsd_000200")["match_level"] == "none")
        r35 = next(r for r in rows if r["qm9_index"] == "dsgdb9nsd_000035")
        m35 = ts.membership("CCC=O", cfg)
        check("the shipped species' row agrees with ticket 30's answer",
              r35["n_train_frames"] == m35["N_TRAIN_FRAMES"] and r35["match_level"] == m35["MATCH_LEVEL"])
        out2 = ts.qm9_targets_membership(cfg, limit=2, out_dir=Path(tmp) / "out2")
        check("--limit considers only the first N molecules and records it",
              out2["summary"]["N_CONSIDERED"] == 2 and out2["info"]["LIMIT"] == 2)
        os.environ.pop("S0_CURATED_QM9", None)
        curated_qm9._INDEX_CACHE.clear()
    if FAIL:
        print("FAIL: " + ", ".join(FAIL))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
