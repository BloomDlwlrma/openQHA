"""Is the molecule in MACE-OFF23's training set? The table for the SI.

TOOLING. Builds (once) the index of the released MACE-OFF23 training and test files
(`openqha.data.training_set`) and answers for a set of molecules:

    --set shipped      every species declared in configs/edges_testset.yaml
                       -> data/training_sets/shipped_species_membership.dat
    --set qm9-targets  every curated QM9 molecule passing the branch A gate
                       -> data/training_sets/qm9_targets_membership.{dat,toml}
    --smiles S [S...]  ad hoc SMILES, printed
    --build-index      (re)build the index and stop

The index (data/training_sets/mace-off23_spice_index.dat) is in the repository; the
5.2 GB source files are not. To build it, point data.training_sets.mace_off23.root (or
S0_TRAINING_SETS_ROOT) at the directory holding train_large_neut_no_bad_clean.xyz and
test_large_neut_all.xyz (Apollo doi:10.17863/CAM.107498).

Usage:
    S0_TRAINING_SETS_ROOT=/path/to/SPICE python scripts/tooling/s0_training_set_membership.py --build-index
    python scripts/tooling/s0_training_set_membership.py --set shipped
"""
import argparse
import sys
import time
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))
from openqha import config                                   # noqa: E402
from openqha.data import training_set as ts                  # noqa: E402
from openqha.store import dat                                # noqa: E402

ROW_SCHEMA = {
    "qm9_index": ("String", None, "the molecule"),
    "name": ("String", None, "declared name"),
    "smiles": ("String", None, "declared SMILES"),
    "n_heavy": ("Integer", None, "heavy atoms"),
    "in_training": ("Boolean", None, "frames in the training file"),
    "in_test_only": ("Boolean", None, "frames in the test file only"),
    "match_level": ("String", None, "isomeric, no_stereo, connectivity or none"),
    "n_train_frames": ("Integer", None, "monomer frames, training file"),
    "n_test_frames": ("Integer", None, "monomer frames, test file"),
    "n_train_dimer_frames": ("Integer", None, "dimer frames, training file"),
    "n_test_dimer_frames": ("Integer", None, "dimer frames, test file"),
    "config_types": ("String", None, "config_type tags of the matched frames, ';'-joined"),
}


def row_for(qid, name, smiles, cfg):
    r = ts.membership(smiles, cfg)
    return {"qm9_index": qid, "name": name, "smiles": smiles, "n_heavy": r["N_HEAVY_ATOMS"],
            "in_training": r["IN_TRAINING"], "in_test_only": r["IN_TEST_ONLY"],
            "match_level": r["MATCH_LEVEL"], "n_train_frames": r["N_TRAIN_FRAMES"],
            "n_test_frames": r["N_TEST_FRAMES"], "n_train_dimer_frames": r["N_TRAIN_DIMER_FRAMES"],
            "n_test_dimer_frames": r["N_TEST_DIMER_FRAMES"],
            "config_types": ";".join(r["CONFIG_TYPES"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=("shipped", "qm9-targets"), default=None)
    ap.add_argument("--limit", type=int, default=None, help="qm9-targets: consider only the first N molecules")
    ap.add_argument("--smiles", nargs="*", default=None)
    ap.add_argument("--build-index", action="store_true")
    args = ap.parse_args()
    cfg = config.load()
    s = ts.settings(cfg)
    if args.build_index:
        t0 = time.time()
        p = ts.build_index(cfg)
        src = dat.read_tables(p)["Source"]
        print("index written  {}  ({:.0f} s)".format(p, time.time() - t0))
        for r in src:
            print("  {:5s} {:>9d} frames  {:>6d} distinct smiles  {} unparsed  {:.0f} s  {}".format(
                r["FILE"], r["N_FRAMES_TOTAL"], r["N_DISTINCT_SMILES"], r["N_UNPARSED"], r["SECONDS"], r["PATH"]))
        if not (args.set or args.smiles):
            return 0
    current = ts.index_is_current(cfg)
    if current is None and not ts.index_path(cfg).is_file():
        raise SystemExit("neither the index {} nor the source files under {} exist; see --help".format(
            ts.index_path(cfg), s["root"]))
    if current is False:
        print("note: the cached index is stale against the source files; --build-index to refresh")
    if args.smiles:
        for sm in args.smiles:
            r = ts.membership(sm, cfg)
            print("{:30s} {}".format(sm, ts.sentence(r)))
    if args.set == "shipped":
        rows = []
        for qid, spec in sorted((cfg.get("species") or {}).items()):
            rows.append(row_for(qid, spec.get("name", ""), spec["smiles"], cfg))
            r = rows[-1]
            print("  {} {:22s} {:12s} in_training {!s:5} {:12s} train {:>4d} (+{} dimer)  test {:>3d}  {}".format(
                qid, r["name"], r["smiles"], r["in_training"], r["match_level"], r["n_train_frames"],
                r["n_train_dimer_frames"], r["n_test_frames"], r["config_types"]))
        out = s["cache"] / "shipped_species_membership.dat"
        dat.write_table(out, rows, schema=ROW_SCHEMA)
        n_in = sum(1 for r in rows if r["in_training"])
        print("{} of {} declared species are in MACE-OFF23's training file; table {}".format(n_in, len(rows), out))
    if args.set == "qm9-targets":
        def progress(k, qid):
            print("  ... {} molecules screened (at {})".format(k, qid), flush=True)
        out = ts.qm9_targets_membership(cfg, limit=args.limit, progress=progress)
        sm = out["summary"]
        print("targets {} of {} QM9 molecules considered ({} failed a gate: {}; {} unreadable)".format(
            sm["N_TARGETS"], sm["N_CONSIDERED"], sm["N_FAILED_GATE"],
            ", ".join("{} {}".format(g, n) for g, n in sorted(out["failed"].items())), sm["N_UNREADABLE"]))
        print("in MACE-OFF23's training file: {} ({:.1%}); isomeric {} / no_stereo {} / connectivity {}; "
              "test only {}; monomer-only {}, dimer-only {}".format(
                  sm["N_IN_TRAINING"], sm["FRACTION_IN_TRAINING"], sm["N_IN_TRAINING_ISOMERIC"],
                  sm["N_IN_TRAINING_NO_STEREO"], sm["N_IN_TRAINING_CONNECTIVITY"], sm["N_IN_TEST_ONLY"],
                  sm["N_MONOMER_ONLY"], sm["N_DIMER_ONLY"]))
        print("by heavy atoms (1..9): targets {}  in training {}".format(sm["TARGETS_BY_HEAVY"], sm["IN_TRAINING_BY_HEAVY"]))
        for c, n in sorted(out["config_types"].items()):
            print("  config_type {:28s} {} targets".format(c, n))
        print("record  {}  table {}  ({:.0f} s)".format(out["record"], out["table"], out["info"]["SECONDS"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
