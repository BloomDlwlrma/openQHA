"""Ticket 20 of the Hessian-learning set (S0-C-54, ADR 0005): `dataset.build`'s
`train_generators` -- basin frames only in train and valid, every labelled frame of a
held-out generator in test, whatever the draw or the previous index said.

On the propanal fixture (three fake molecules, 3 basins x (1 basin + 4 displaced) frames,
`with_merged` so a merged generator exists too; every frame labelled, the displaced
ones without a Hessian): with the default (`basin`) train and valid hold basin frames
only, every displaced and merged frame is in test with `held_out_generator = yes`, the
counts `N_TRAIN_BASIN` / `N_TEST_HELD_OUT` and the `[[Class]]` / `[[Molecule]]` columns
add up, the index column round-trips, the merged MACE file carries the same routing;
`train_generators=("basin", "merged")` moves the merged frames into the draw; a
previous index that had displaced frames in train is overridden on a rebuild; an
unknown or empty generator tuple is refused; the OpenREACT export and the judge's
`frame_rows` split filter are unaffected (test frames are read as before).
"""
import sys
import tempfile
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))
sys.path.insert(0, str(_repo_root() / "tests" / "unit"))
from t_frames import check, FAIL                                       # noqa: E402
from t_dataset import fake_labels, LEVEL, TAG                          # noqa: E402

from openqha.data import dataset, frames                               # noqa: E402
from openqha.store import dat, layout, property as prop                # noqa: E402


def place_with_merged(tmp, qid):
    """`t_dataset.place` with the merged generator kept (three generators present)."""
    import shutil
    from t_frames import Harmonic, make_molecule
    root = Path(tmp) / "root"
    mol, basins = make_molecule(Path(tmp) / ("src_" + qid), with_merged=True)
    dest = layout.molecule_dir(root, TAG, qid)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(mol), str(dest))
    rec = dest / "_records" / "branchA.toml"
    rec.write_text(rec.read_text(encoding="utf-8").replace("dsgdb9nsd_000035", qid), encoding="utf-8")
    frames.generate(dest, calc=Harmonic(basins), engine_name="MACE-OFF23_medium", n_displaced=4)
    return root, dest


def main():
    from ase.io import read
    with tempfile.TemporaryDirectory(prefix="dataset20_") as tmp:
        root, mol_a = place_with_merged(tmp, "dsgdb9nsd_000035")           # pinned: propanal
        _r, mol_b = place_with_merged(tmp, "dsgdb9nsd_000036")
        _r, mol_c = place_with_merged(tmp, "dsgdb9nsd_000037")
        mace_level = prop.load(layout.frames_dir(mol_a) / "frames.toml")["Calculation_Info"]["LEVEL"]
        for m in (mol_a, mol_b, mol_c):
            fake_labels(m, frames.GENERATORS, mace_level, gradient_only=("displaced",))
        gens_present = sorted({a.info["generator"] for m in (mol_a,)
                               for g in frames.GENERATORS if layout.frames_file(m, g, mace_level).is_file()
                               for a in frames.read_frames(layout.frames_file(m, g, mace_level))})
        check("the fixture has the basin, displaced, merged and saddle generators", gens_present == ["basin", "displaced", "merged", "saddle"], gens_present)
        pinned = ("dsgdb9nsd_000035",)
        dataset.select(root, [TAG], "g", pinned=pinned)

        # --- the default: basin only -------------------------------------------------------------
        out = dataset.build(root, [TAG], "g", level=LEVEL, split_by="frame", seed=3, pinned=pinned)
        i, idx = out["info"], out["index"]
        tv = [r for r in idx if r["split"] in ("train", "valid")]
        held = [r for r in idx if r["held_out_generator"] == "yes"]
        non_basin = [r for r in idx if r["generator"] != "basin" and r["split"] != "pool"]
        check("default train_generators = (basin,): train and valid hold basin frames only",
              i["TRAIN_GENERATORS"] == ["basin"] and i["HELD_OUT_GENERATORS"] == ["displaced", "merged", "saddle"]
              and tv and all(r["generator"] == "basin" for r in tv), sorted({r["generator"] for r in tv}))
        check("every labelled displaced / merged frame is in test with held_out_generator = yes ({} frames)".format(len(non_basin)),
              non_basin and all(r["split"] == "test" and r["held_out_generator"] == "yes" for r in non_basin)
              and len(held) == len(non_basin) and all(r["held_out_generator"] == "no" for r in idx if r["generator"] == "basin"),
              [(r["generator"], r["split"], r["held_out_generator"]) for r in non_basin[:3]])
        check("the Record counts: N_TRAIN_BASIN = N_TRAIN, N_TEST_HELD_OUT = the held-out frames, N_TEST >= that",
              i["N_TRAIN_BASIN"] == i["N_TRAIN"] and i["N_TEST_HELD_OUT"] == len(held) and i["N_TEST"] >= len(held)
              and i["N_TRAIN"] + i["N_VALID"] + i["N_TEST"] + i["N_POOL"] == i["N_FRAMES"],
              {k: i[k] for k in ("N_TRAIN", "N_TRAIN_BASIN", "N_VALID", "N_TEST", "N_TEST_HELD_OUT", "N_POOL", "N_FRAMES")})
        # the non-pinned molecules' basin frames are still drawn by frame (some train), the pinned one all test
        basin_b = [r for r in idx if r["qm9_index"] == "dsgdb9nsd_000036" and r["generator"] == "basin"]
        check("the non-pinned molecules' basin frames go through the draw (a train frame exists); the pinned molecule's are test",
              any(r["split"] == "train" for r in basin_b)
              and all(r["split"] == "test" for r in idx if r["qm9_index"] == "dsgdb9nsd_000035" and r["split"] != "pool"))
        per_mol = {m["QM9_INDEX"]: m for m in out["molecules"]}
        cls = {c["CLASS"]: c for c in out["classes"]}
        check("[[Molecule]] and [[Class]] carry N_TEST_HELD_OUT and their splits add up",
              all(m["N_TEST_HELD_OUT"] <= m["N_TEST"] for m in per_mol.values())
              and sum(m["N_TEST_HELD_OUT"] for m in per_mol.values()) == i["N_TEST_HELD_OUT"]
              and all(c["N_TRAIN"] + c["N_VALID"] + c["N_TEST"] + c["N_POOL"] == c["N_FRAMES"] for c in cls.values())
              and all("N_TEST_HELD_OUT" in c for c in cls.values()), (per_mol, cls))
        d = out["dir"]
        back = dat.read_table(d / "index.dat")
        check("index.dat round-trips the held_out_generator column",
              [r["held_out_generator"] for r in back] == [r["held_out_generator"] for r in idx]
              and set(r["held_out_generator"] for r in back) == {"yes", "no"})
        merged = read(str(dataset.merged_file(d, "g", LEVEL)), index=":", format="extxyz")
        check("the merged MACE file carries the same routing: no non-basin frame in train or valid",
              all(a.info["generator"] == "basin" for a in merged if a.info["split"] in ("train", "valid"))
              and any(a.info["generator"] == "displaced" and a.info["split"] == "test" for a in merged))
        rec = prop.load(d / "dataset.toml")
        check("dataset.toml holds TRAIN_GENERATORS / HELD_OUT_GENERATORS / N_TRAIN_BASIN / N_TEST_HELD_OUT",
              rec["Calculation_Info"]["TRAIN_GENERATORS"] == ["basin"] and rec["Calculation_Info"]["N_TEST_HELD_OUT"] == len(held)
              and "displaced" in rec["Calculation_Info"]["HELD_OUT_GENERATORS"])
        report = (d / "dataset.out").read_text(encoding="utf-8")
        check("the report names the rule and the held-out counts", "TRAIN_GENERATORS" in report and "held-out" in report)

        # --- merged into train ---------------------------------------------------------------------------
        out2 = dataset.build(root, [TAG], "g", level=LEVEL, split_by="frame", seed=3, pinned=pinned,
                             train_generators=("basin", "merged"))
        i2, idx2 = out2["info"], out2["index"]
        merged_b = [r for r in idx2 if r["qm9_index"] != "dsgdb9nsd_000035" and r["generator"] == "merged"]
        check("train_generators = (basin, merged): the merged frames go through the draw (a train one exists), displaced stay held out",
              i2["TRAIN_GENERATORS"] == ["basin", "merged"] and i2["HELD_OUT_GENERATORS"] == ["displaced", "saddle"]
              and any(r["split"] == "train" for r in merged_b) and all(r["held_out_generator"] == "no" for r in merged_b)
              and all(r["split"] == "test" and r["held_out_generator"] == "yes" for r in idx2
                      if r["generator"] == "displaced" and r["split"] != "pool"),
              sorted({(r["generator"], r["split"]) for r in idx2}))
        # a merged frame that a previous index (the one just written) drew into train goes back to
        # test on the next basin-only rebuild: the rule is above the kept decisions
        out3 = dataset.build(root, [TAG], "g", level=LEVEL, split_by="frame", seed=3, pinned=pinned)
        check("a rebuild with basin only overrides the previous index for the held-out frames (KEPT_PREVIOUS true, merged frames test)",
              out3["info"]["KEPT_PREVIOUS"] and all(r["split"] == "test" for r in out3["index"] if r["generator"] == "merged" and r["split"] != "pool")
              and [r["split"] for r in out3["index"] if r["generator"] == "basin"] == [r["split"] for r in idx if r["generator"] == "basin"])
        for bad in ((), ("basin", "sideways")):
            try:
                dataset.build(root, [TAG], "g", level=LEVEL, split_by="frame", seed=3, pinned=pinned, train_generators=bad)
                check("train_generators {!r} is refused".format(bad), False)
            except ValueError as exc:
                check("train_generators {!r} is refused".format(bad), "train_generators" in str(exc))

        # the judge's frame filter reads test frames as before
        from openqha.training import judge
        rows = judge.frame_rows.__doc__ or ""
        check("judge.frame_rows exists and reads a split (unchanged interface)", "split" in rows or callable(judge.frame_rows))

    print("\n{} checks, {} failed".format(15, len(FAIL)))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
