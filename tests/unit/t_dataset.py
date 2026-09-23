"""Ticket 04 of the Hessian-learning set: the Dataset -- selection, split, files, index,
OpenREACT export -- on two fake molecules made from the propanal fixture (harmonic
surrogate frames, fake reference labels written straight into the level files; no
engine, no ORCA).

Asserted: `select` finds both molecules under the tag with their stratification keys,
SPICE membership and pin status; with the pinned molecule labelled at its basin frames
only and the other fully labelled: test holds the pinned molecule's labelled frames and
nothing else of it goes to train/valid, valid frames come only from the training
molecule at the stated fraction, pool holds exactly the frames without the level and
says where they will go; the split is identical on a rebuild with the same seed and
differs with another; `index.dat` round-trips through `dat.read_table` and every row's
(file, row) is the frame it names; the OpenREACT export has one group per molecule with
coordinates / energies / forces / hessian / species in A, Eh, Eh/A, Eh/A^2 that read
back to the extxyz within 1e-10; `stratum_keys` on known SMILES.

Ticket 07 (the production split, classes, the MACE form): a third, non-pinned molecule
whose displaced frames carry no Hessian; `split_by="frame"` puts the pinned molecule's
frames all in test and draws every other labelled frame on its own at 90/5/5 (a
binomial tolerance on 30 frames), a frame's split is unchanged when other frames are
added or the previous index is dropped, a rebuild keeps every decision; the `[[Class]]`
rows add up and `classes` round-trips; the merged `mace_<name>.<level>.extxyz` reads back
with ASE with REF_energy == energy, REF_forces == forces, REF_hessian == hessian where
`has_hessian`, `split` in {train, valid, test}, frame count = the three files' sum; a
hand-written `draw.dat` makes `select` list a molecule branch A never ran for.
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


sys.path.insert(0, str(_repo_root()))
sys.path.insert(0, str(_repo_root() / "tests" / "unit"))
from t_frames import Harmonic, check, make_molecule, FAIL           # noqa: E402

from openqha.data import dataset, frames                              # noqa: E402
from openqha.qm_interfaces import orca                                # noqa: E402
from openqha.store import dat, layout, property as prop               # noqa: E402

LEVEL = "wb97m-d3bj_def2-tzvppd"
TAG = "fake"


def place(tmp, qid):
    """A fixture molecule at `<tmp>/root/<TAG>/.../<qid>` with a Frame set (3 basins x (1 + 4))."""
    root = Path(tmp) / "root"
    mol, basins = make_molecule(Path(tmp) / ("src_" + qid), with_merged=False)
    dest = layout.molecule_dir(root, TAG, qid)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(mol), str(dest))
    rec = dest / "_records" / "branchA.toml"                # the fixture is propanal; give the copy its own index
    rec.write_text(rec.read_text(encoding="utf-8").replace("dsgdb9nsd_000035", qid), encoding="utf-8")
    frames.generate(dest, calc=Harmonic(basins), engine_name="MACE-OFF23_medium", n_displaced=4)
    return root, dest


def _fake_hessian(a):
    """A reference-level Hessian for the fixture: the engine's own scaled where the frame
    carries one (basin / merged / saddle), else a deterministic surrogate of the right shape
    -- a displaced frame has no engine Hessian since ticket 29, and this fixture is about
    what the Dataset does with a LABEL Hessian, not where it came from."""
    h = a.info.get("hessian")
    if h is not None:
        return np.asarray(h) * 1.1
    n = 3 * len(a)
    return np.eye(n) * 1.1 + 0.01


def fake_labels(mol, generators, mace_level, only_basin_frames=False, gradient_only=()):
    """Reference-level files written from the engine frames with shifted values; the
    generators in `gradient_only` get no Hessian (round 5 Q7 (b): the displaced frames)."""
    for g in generators:
        src = layout.frames_file(mol, g, mace_level)
        if not src.is_file():
            continue
        out = []
        for a in frames.read_frames(src):
            if only_basin_frames and g != "basin":
                continue
            info = {k: a.info[k] for k in ("qm9_index", "basin", "generator", "k", "seed", "source_conformer", "rms_displacement_A", "smiles")}
            info.update(level=LEVEL, orca_version="6.0.1", hessian_route="analytic" if g not in gradient_only else "gradient",
                        noise_floor_cm=1.0, keywords="fake")
            out.append(dict(atoms=a, energy=float(a.get_potential_energy()) - 1.0, forces=np.asarray(a.get_forces()) * 0.5,
                            hessian=_fake_hessian(a) if g not in gradient_only else None, info=info))
        if out:
            frames._write_frames(layout.frames_file(mol, g, LEVEL), out)


def main():
    keys = dataset.stratum_keys("C1COC1")
    keys2 = dataset.stratum_keys("CC(=O)N")
    check("stratum_keys: oxetane -> 4 heavy, 1 ring, O1, r1_O1; acetamide -> r0_N1O1",
          keys == (4, 1, "O1", "r1_O1") and keys2 == (4, 0, "N1O1", "r0_N1O1"), (keys, keys2))

    with tempfile.TemporaryDirectory(prefix="dataset_") as tmp:
        root, mol_a = place(tmp, "dsgdb9nsd_000035")           # pinned: propanal
        _root, mol_b = place(tmp, "dsgdb9nsd_000036")          # the fixture again, under another index (N-methylformamide's)
        mace_level = prop.load(layout.frames_dir(mol_a) / "frames.toml")["Calculation_Info"]["LEVEL"]
        fake_labels(mol_a, frames.GENERATORS, mace_level, only_basin_frames=True)     # 3 of 15 labelled
        fake_labels(mol_b, frames.GENERATORS, mace_level)                             # 15 of 15 labelled
        pinned = ("dsgdb9nsd_000035",)

        rows = dataset.select(root, [TAG], "t", pinned=pinned)
        d = dataset.datasets_dir(root, TAG, "t")
        check("select: both molecules, branch A done, SMILES CCC=O -> r0_O1, SPICE in_training true, 035 pinned, 15 frames each",
              [r["qm9_index"] for r in rows] == ["dsgdb9nsd_000035", "dsgdb9nsd_000036"]
              and all(r["stratum"] == "r0_O1" and r["in_training"] == "true" and r["n_frames"] == 15 and r["has_frames"] for r in rows)
              and rows[0]["pinned"] and not rows[1]["pinned"] and (d / "select.dat").is_file()
              and prop.status_of(d / "select.toml") == prop.NORMAL_TERMINATION,
              [(r["qm9_index"], r["stratum"], r["in_training"], r["pinned"], r["n_frames"]) for r in rows])
        back = dat.read_table(d / "select.dat")
        check("select.dat round-trips (pinned Boolean, n_heavy Integer)",
              back[0]["pinned"] is True and back[0]["n_heavy"] == 4 and back[1]["pinned"] is False, back[0])

        out = dataset.build(root, [TAG], "t", level=LEVEL, split_by="molecule", valid_fraction=0.2, test_fraction=0.0, seed=7, pinned=pinned, train_generators=frames.GENERATORS)
        idx = out["index"]
        a_rows = [r for r in idx if r["qm9_index"] == "dsgdb9nsd_000035"]
        b_rows = [r for r in idx if r["qm9_index"] == "dsgdb9nsd_000036"]
        check("test holds the pinned molecule's 3 labelled frames; its 12 unlabelled are pool with molecule_split test",
              sorted(r["split"] for r in a_rows) == ["pool"] * 12 + ["test"] * 3
              and all(r["molecule_split"] == "test" for r in a_rows)
              and all(r["generator"] == "basin" for r in a_rows if r["split"] == "test"),
              sorted(r["split"] for r in a_rows))
        n_valid = sum(1 for r in b_rows if r["split"] == "valid")
        check("valid frames come only from the training molecule, at the fraction (round(0.2 x 15) = 3), the rest train",
              n_valid == 3 and sum(1 for r in b_rows if r["split"] == "train") == 12
              and all(r["molecule_split"] == "train" for r in b_rows)
              and not any(r["split"] == "valid" for r in a_rows), (n_valid, sorted(r["split"] for r in b_rows)))
        i = out["info"]
        check("the Record: 2 molecules (1 test), 30 frames, 18 labelled, train 12 valid 3 test 3 pool 12, one fingerprint, ORCA 6.0.1",
              (i["N_MOLECULES"], i["N_TEST_MOLECULES"], i["N_FRAMES"], i["N_LABELLED"], i["N_TRAIN"], i["N_VALID"], i["N_TEST"], i["N_POOL"])
              == (2, 1, 30, 18, 12, 3, 3, 12) and len(i["ENGINE_PARAMS_SHA256"]) == 1 and i["ORCA_VERSIONS"] == ["6.0.1"]
              and prop.status_of(d / "dataset.toml") == prop.NORMAL_TERMINATION,
              {k: i[k] for k in ("N_MOLECULES", "N_TEST_MOLECULES", "N_FRAMES", "N_LABELLED", "N_TRAIN", "N_VALID", "N_TEST", "N_POOL")})
        check("levels column: labelled frames list both levels, pool frames the engine's only",
              all(r["levels"] == mace_level + ";" + LEVEL for r in idx if r["split"] != "pool")
              and all(r["levels"] == mace_level for r in idx if r["split"] == "pool"))

        # --- reproducibility --------------------------------------------------------------
        key = lambda rows_: [(r["qm9_index"], r["generator"], r["basin"], r["k"], r["split"]) for r in rows_]   # noqa: E731
        again = dataset.build(root, [TAG], "t", level=LEVEL, split_by="molecule", valid_fraction=0.2, test_fraction=0.0, seed=7, pinned=pinned, train_generators=frames.GENERATORS)
        other = dataset.build(root, [TAG], "t", level=LEVEL, split_by="molecule", valid_fraction=0.2, test_fraction=0.0, seed=8, pinned=pinned, train_generators=frames.GENERATORS,
                              keep_previous=False)
        same, diff = key(again["index"]) == key(idx), key(other["index"]) != key(idx)
        check("the split is identical on a rebuild with the same seed and differs with another seed (previous index not kept)",
              same and diff and again["info"]["KEPT_PREVIOUS"], (same, diff))
        # a rebuild after MORE frames get labelled keeps every earlier decision (review 2026-09-18)
        before = dataset.build(root, [TAG], "t", level=LEVEL, split_by="molecule", valid_fraction=0.2, test_fraction=0.0, seed=7, pinned=pinned, train_generators=frames.GENERATORS,
                               keep_previous=False)
        prev = {(r["qm9_index"], r["generator"], r["basin"], r["k"]): r["split"] for r in before["index"] if r["split"] != "pool"}
        fake_labels(mol_a, frames.GENERATORS, mace_level)                         # propanal: now 15 of 15
        after = dataset.build(root, [TAG], "t", level=LEVEL, split_by="molecule", valid_fraction=0.2, test_fraction=0.0, seed=7, pinned=pinned, train_generators=frames.GENERATORS)
        now = {(r["qm9_index"], r["generator"], r["basin"], r["k"]): r["split"] for r in after["index"]}
        kept = all(now[k] == s for k, s in prev.items())
        check("after more frames are labelled, a rebuild keeps every earlier frame's split and the molecule sides; the new labels of the test molecule are test",
              kept and after["info"]["KEPT_PREVIOUS"] and after["info"]["N_POOL"] == 0 and after["info"]["N_TEST"] == 15
              and after["info"]["N_VALID"] == 3, (kept, {k: after["info"][k] for k in ("N_POOL", "N_TEST", "N_VALID", "N_TRAIN")}))
        # a stale label file (other coordinates) is ignored, counted, and the frame goes to pool
        stale_path = layout.frames_file(mol_b, "basin", LEVEL)
        stale = []
        for a in frames.read_frames(stale_path):
            e, f = float(a.get_potential_energy()), np.array(a.get_forces())      # before the move: ASE drops them
            a.positions[0, 0] += 0.01
            stale.append(dict(atoms=a, energy=e, forces=f, hessian=a.info["hessian"], info=dict(a.info)))
        frames._write_frames(stale_path, stale)
        st = dataset.build(root, [TAG], "t", level=LEVEL, split_by="molecule", valid_fraction=0.2, test_fraction=0.0, seed=7, pinned=pinned, train_generators=frames.GENERATORS)
        check("a label file whose geometry differs from the engine file's is stale: ignored, counted (3), its frames pool",
              st["info"]["N_STALE"] == 3 and st["info"]["N_POOL"] == 3
              and all(r["split"] == "pool" for r in st["index"] if r["qm9_index"] == "dsgdb9nsd_000036" and r["generator"] == "basin"),
              (st["info"]["N_STALE"], st["info"]["N_POOL"]))
        fake_labels(mol_b, ("basin",), mace_level)                                # restore for the checks below
        fake_labels(mol_a, frames.GENERATORS, mace_level, only_basin_frames=True)
        for g in ("displaced",):
            layout.frames_file(mol_a, g, LEVEL).unlink()
        out = dataset.build(root, [TAG], "t", level=LEVEL, split_by="molecule", valid_fraction=0.2, test_fraction=0.0, seed=7, pinned=pinned, train_generators=frames.GENERATORS,
                            keep_previous=False)
        idx = out["index"]

        # --- the index and the files ------------------------------------------------------
        back = dat.read_table(d / "index.dat")
        files = {}
        ok, bad = len(back) == 30, []
        for r in back:
            if r["file"] not in files:
                files[r["file"]] = frames.read_frames(d / r["file"])
            a = files[r["file"]][r["row"]]
            good = (a.info["qm9_index"] == r["qm9_index"] and a.info["generator"] == r["generator"]
                    and int(a.info["basin"]) == r["basin"] and int(a.info["k"]) == r["k"]
                    and a.info["level"] == (LEVEL if r["split"] != "pool" else mace_level)
                    # a labelled frame carries its Label Hessian; a pool frame carries the engine's,
                    # which a displaced frame has not had since ticket 29
                    and (np.shape(a.info["hessian"]) == (30, 30)
                         if a.info.get("has_hessian", "hessian" in a.info) else "hessian" not in a.info))
            if not good:
                bad.append((r["qm9_index"], r["generator"], r["basin"], r["k"], r["split"], a.info.get("qm9_index"), a.info.get("generator"),
                            a.info.get("basin"), a.info.get("k"), a.info.get("level"), np.shape(a.info.get("hessian"))))
            ok = ok and good
        check("index.dat round-trips and every row's (file, row) is the frame it names, at the split's level", ok,
              (len(back), bad[:2]))
        test_atoms = files["test.{}.extxyz".format(LEVEL)][0]
        eng = frames.read_frames(layout.frames_file(mol_a, "basin", mace_level))[0]
        check("a test frame carries the REFERENCE values (energy - 1, forces x 0.5, hessian x 1.1 of the engine's), same positions",
              abs(test_atoms.get_potential_energy() - (eng.get_potential_energy() - 1.0)) < 1e-6
              and np.abs(test_atoms.get_forces() - 0.5 * eng.get_forces()).max() < 1e-6
              and np.abs(test_atoms.info["hessian"] - 1.1 * eng.info["hessian"]).max() < 1e-6
              and np.abs(test_atoms.get_positions() - eng.get_positions()).max() == 0.0)

        # --- OpenREACT export -------------------------------------------------------------
        import h5py
        h5 = dataset.export_openreact(d, LEVEL, "t")
        with h5py.File(h5) as f:
            groups = sorted(f.keys())
            g = f["dsgdb9nsd_000036"]
            ga = f["dsgdb9nsd_000035"]
            nat = int(g.attrs["nat"])
            shapes = (g["coordinates"].shape, g["energies"].shape, g["forces"].shape, g["hessian"].shape, g["species"].shape)
            species = [s.decode() for s in g["species"][:]]
            # read back the first frame of 036 and compare with the extxyz row the index names
            rows36 = [r for r in back if r["qm9_index"] == "dsgdb9nsd_000036"]
            r0 = rows36[0]
            a0 = files[r0["file"]][r0["row"]]
            e_back = float(g["energies"][0]) * orca.EV_PER_HARTREE
            f_back = np.asarray(g["forces"][0]) * orca.EV_PER_HARTREE
            h_back = np.asarray(g["hessian"][0]) * orca.EV_PER_HARTREE
            x_back = np.asarray(g["coordinates"][0])
            splits = [s.decode() for s in g["split"][:]]
            molec, molec_a = int(g.attrs["molec"]), int(ga.attrs["molec"])
        check("OpenREACT export: one group per molecule, attrs nat/molec, datasets (n, nat, 3) / (n,) / (n, nat, 3) / (n, 3nat, 3nat) / (nat,), species symbols",
              groups == ["dsgdb9nsd_000035", "dsgdb9nsd_000036"] and nat == 10 and molec == 15 and molec_a == 3
              and shapes == ((15, 10, 3), (15,), (15, 10, 3), (15, 30, 30), (10,)) and species == a0.get_chemical_symbols(),
              (groups, shapes, species))
        check("read back in Eh, Eh/A, Eh/A^2, A equals the extxyz frame to 1e-10 (relative for the energy); split per frame present",
              abs(e_back - a0.get_potential_energy()) / abs(a0.get_potential_energy()) < 1e-10
              and np.abs(f_back - a0.get_forces()).max() < 1e-10 and np.abs(h_back - a0.info["hessian"]).max() < 1e-9
              and np.abs(x_back - a0.get_positions()).max() == 0.0 and sorted(set(splits)) == ["train", "valid"],
              (e_back - a0.get_potential_energy(), np.abs(h_back - a0.info["hessian"]).max(), set(splits)))

    # ================================================================== ticket 07
    with tempfile.TemporaryDirectory(prefix="dataset07_") as tmp:
        root, mol_a = place(tmp, "dsgdb9nsd_000035")           # pinned: propanal
        _r, mol_b = place(tmp, "dsgdb9nsd_000036")
        _r, mol_c = place(tmp, "dsgdb9nsd_000037")
        mace_level = prop.load(layout.frames_dir(mol_a) / "frames.toml")["Calculation_Info"]["LEVEL"]
        fake_labels(mol_a, frames.GENERATORS, mace_level)
        fake_labels(mol_b, frames.GENERATORS, mace_level)
        fake_labels(mol_c, frames.GENERATORS, mace_level, gradient_only=("displaced",))
        pinned = ("dsgdb9nsd_000035",)
        d = dataset.datasets_dir(root, TAG, "p")
        d.mkdir(parents=True)
        # a hand-written draw: the three plus one branch A never ran for, with classes
        from openqha.data import structure_classes as sc
        dat.write_table(d / "draw.dat", [
            dict(qm9_index="dsgdb9nsd_000035", smiles="CCC=O", n_heavy=4, classes="aldehyde", in_training=False, pinned=True),
            dict(qm9_index="dsgdb9nsd_000036", smiles="CCC=O", n_heavy=4, classes="aldehyde;ketone", in_training=False, pinned=False),
            dict(qm9_index="dsgdb9nsd_000037", smiles="CCC=O", n_heavy=4, classes="ketone", in_training=False, pinned=False),
            dict(qm9_index="dsgdb9nsd_000100", smiles="C1COC1", n_heavy=4, classes="heterocyclic;small_ring", in_training=False, pinned=False)],
            list(sc.ROW_SCHEMA), sc.ROW_SCHEMA)
        rows = dataset.select(root, [TAG], "p", pinned=pinned)
        by = {r["qm9_index"]: r for r in rows}
        sel_back = {r["qm9_index"]: r for r in dat.read_table(d / "select.dat")}
        sel_doc = prop.load(d / "select.toml")
        cls = {c["CLASS"]: c for c in sel_doc["Class"]}
        check("select with a draw: every drawn molecule is a row (4), classes from draw.dat, the one without branch A has has_basins / has_frames false and no smiles error; [[Class]] counts",
              len(rows) == 4 and by["dsgdb9nsd_000100"]["has_basins"] is False and by["dsgdb9nsd_000100"]["has_frames"] is False
              and by["dsgdb9nsd_000100"]["classes"] == "heterocyclic;small_ring" and by["dsgdb9nsd_000036"]["classes"] == "aldehyde;ketone"
              and all(r["drawn"] for r in rows) and by["dsgdb9nsd_000036"]["has_basins"] and by["dsgdb9nsd_000036"]["n_frames"] == 15
              and sel_back["dsgdb9nsd_000100"]["has_basins"] is False and sel_back["dsgdb9nsd_000036"]["classes"] == "aldehyde;ketone"
              and cls["aldehyde"]["N_MOLECULES"] == 2 and cls["ketone"]["N_WITH_FRAMES"] == 2 and cls["small_ring"]["N_WITH_BASINS"] == 0
              and sel_doc["Calculation_Info"]["N_DRAWN"] == 4 and sel_doc["Calculation_Info"]["N_WITH_BASINS"] == 3,
              ({q: (r["has_basins"], r["classes"]) for q, r in by.items()}, cls))
        check("select without a draw classifies from the SMILES: propanal CCC=O -> aldehyde",
              dataset.classes_of("CCC=O") == "aldehyde" and "small_ring" in dataset.classes_of("C1COC1"),
              (dataset.classes_of("CCC=O"), dataset.classes_of("C1COC1")))

        out = dataset.build(root, [TAG], "p", level=LEVEL, split_by="frame", seed=3, pinned=pinned, train_generators=frames.GENERATORS)
        i, idx = out["info"], out["index"]
        a_rows = [r for r in idx if r["qm9_index"] == "dsgdb9nsd_000035"]
        others = [r for r in idx if r["qm9_index"] != "dsgdb9nsd_000035"]
        n_tr, n_va, n_te = (sum(1 for r in others if r["split"] == s) for s in ("train", "valid", "test"))
        # 30 frames at 5 % each: P(valid + test > 8) ~ 1e-3, P(valid + test == 0) ~ 5 %; the split is fixed by seed 3
        check("split by frame: the pinned molecule's 15 frames all test (molecule_split test); the other 30 labelled frames are train / valid / test with valid + test in 1..8 (5 % + 5 % expected 3), none pool, fractions 0.05 in the Record",
              all(r["split"] == "test" and r["molecule_split"] == "test" for r in a_rows) and len(a_rows) == 15
              and n_tr + n_va + n_te == 30 and 1 <= n_va + n_te <= 8 and i["N_POOL"] == 0
              and i["SPLIT_BY"] == "frame" and i["VALID_FRACTION"] == 0.05 and i["TEST_FRACTION"] == 0.05
              and i["N_TEST"] == 15 + n_te and i["N_TEST_MOLECULES"] == 1,
              (n_tr, n_va, n_te, {k: i[k] for k in ("N_TRAIN", "N_VALID", "N_TEST", "N_POOL", "N_TEST_MOLECULES")}))
        key = lambda rows_: {(r["qm9_index"], r["generator"], r["basin"], r["k"]): r["split"] for r in rows_}   # noqa: E731
        s0 = key(others)
        # independence: a frame's draw depends on nothing else -- drop the previous index, drop a molecule, same splits
        alone = dataset.build(root, [TAG], "p", level=LEVEL, split_by="frame", seed=3, pinned=pinned, train_generators=frames.GENERATORS, keep_previous=False,
                              selection=[r for r in rows if r["qm9_index"] != "dsgdb9nsd_000037"])
        s1 = key(alone["index"])
        again = dataset.build(root, [TAG], "p", level=LEVEL, split_by="frame", seed=3, pinned=pinned, train_generators=frames.GENERATORS)
        s2 = key(again["index"])
        other_seed = dataset.build(root, [TAG], "p", level=LEVEL, split_by="frame", seed=4, pinned=pinned, train_generators=frames.GENERATORS, keep_previous=False)
        s3 = key(other_seed["index"])
        check("a frame's split is its own draw: identical without the previous index and with a molecule removed; a rebuild keeps everything; another seed differs",
              all(s1[k] == s0[k] for k in s0 if k in s1) and sum(1 for k in s0 if k in s1) == 15 and len(s1) == 30
              and all(s2[k] == v for k, v in s0.items()) and again["info"]["KEPT_PREVIOUS"]
              and any(s3[k] != v for k, v in s0.items()),
              (sum(s1[k] != s0[k] for k in s0 if k in s1), sum(s3[k] != v for k, v in s0.items())))
        cls = {c["CLASS"]: c for c in out["classes"]}
        check("[[Class]] in the Dataset Record: aldehyde = molecules 035 + 036 (30 frames), ketone = 036 + 037, every class's splits add up; classes column in index.dat round-trips",
              cls["aldehyde"]["N_MOLECULES"] == 2 and cls["aldehyde"]["N_FRAMES"] == 30 and cls["ketone"]["N_MOLECULES"] == 2
              and all(c["N_TRAIN"] + c["N_VALID"] + c["N_TEST"] + c["N_POOL"] == c["N_FRAMES"] for c in out["classes"])
              and cls["aldehyde"]["N_TEST"] >= 15
              and {r["classes"] for r in dat.read_table(d / "index.dat") if r["qm9_index"] == "dsgdb9nsd_000036"} == {"aldehyde;ketone"},
              cls)
        # the MACE form: the merged file and the REF_* keys
        from ase.io import read
        merged = read(str(dataset.merged_file(d, "p", LEVEL)), index=":", format="extxyz")
        parts = sum((read(str(d / "{}.{}.extxyz".format(s, LEVEL)), index=":", format="extxyz") for s in ("train", "valid", "test")
                     if (d / "{}.{}.extxyz".format(s, LEVEL)).is_file()), [])
        def same(a, b):
            return (abs(a.info["REF_energy"] - a.get_potential_energy()) < 1e-9
                    and np.abs(a.arrays["REF_forces"] - a.get_forces()).max() < 1e-9
                    and a.info["split"] in ("train", "valid", "test") and a.info["has_hessian"] == ("REF_hessian" in a.info)
                    and (not a.info["has_hessian"] or np.abs(np.asarray(a.info["REF_hessian"]).reshape(30, 30) - np.asarray(a.info["hessian"]).reshape(30, 30)).max() < 1e-9)
                    and a.info["qm9_index"] == b.info["qm9_index"] and a.info["split"] == b.info["split"])
        n_noh = sum(1 for a in merged if not a.info["has_hessian"])
        check("mace_<name>.<level>.extxyz: 45 frames = train + valid + test, same order; REF_energy == energy, REF_forces == forces, REF_hessian reshapes to (3N, 3N) == hessian where has_hessian, absent otherwise (037's 12 displaced frames), split in {train, valid, test}",
              len(merged) == 45 == len(parts) and all(same(a, b) for a, b in zip(merged, parts)) and n_noh == 12
              and i["N_HESSIAN_FRAMES"] == 33 and i["MERGED_FILE"].endswith("mace_p.{}.extxyz".format(LEVEL)),
              (len(merged), len(parts), n_noh, i["N_HESSIAN_FRAMES"]))
        pool_ok = True
        if (d / "pool.{}.extxyz".format(LEVEL)).is_file():
            pool_ok = all("REF_energy" not in a.info for a in read(str(d / "pool.{}.extxyz".format(LEVEL)), index=":", format="extxyz"))
        by_mol = dataset.build(root, [TAG], "p", level=LEVEL, split_by="molecule", test_fraction=0.0, valid_fraction=0.2, seed=3,
                               pinned=pinned, train_generators=frames.GENERATORS, keep_previous=False)
        check("split by molecule still works on the same tree (pinned test, others train/valid at 0.2, no frame test); pool frames carry no REF_ keys",
              by_mol["info"]["N_TEST"] == 15 and by_mol["info"]["SPLIT_BY"] == "molecule" and by_mol["info"]["N_VALID"] == 6
              and all(r["split"] != "test" for r in by_mol["index"] if r["qm9_index"] != "dsgdb9nsd_000035") and pool_ok,
              {k: by_mol["info"][k] for k in ("N_TRAIN", "N_VALID", "N_TEST", "N_POOL")})

    print("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
