"""Ticket 12 of the Hessian-learning set: the file `04_dataset` writes is what the fork's
`config_from_atoms` reads, and the batch it makes is what ticket 11's loss slices.

No engine. From the 2-methyloxirane frame fixture (basin + 4 displaced frames, E-F-H at
both levels) a MACE-form file is written with `dataset._write_split` exactly as
`mace_<name>.<level>.extxyz` is -- the displaced frames stripped of their Hessian, as
round-5 Q7 (b) labels them -- then read back through mace: `config_from_atoms`
(default keyspec: `REF_energy` / `REF_forces` / `REF_hessian`), `AtomicData.from_config`,
a `DataLoader` batch; `phl_loss.graph_labels` slices the batch and every Hessian equals
the fixture's to 1e-10; `has_hessian` is T F F F F; `REF_energy` / `REF_forces` equal
`energy` / `forces`; `sqrt_masses` are ASE's; and the loss's full-matrix term on that
batch, fed the MACE-level Hessian of the basin frame as `pred["hessian"]`, equals
`phl.loss_full` of the two fixture matrices (eq. 1', the one target since S0-C-64).
SKIPs if the fork's key is absent (a pip-installed mace).
"""
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
FIX = ROOT / "tests" / "data" / "methyloxirane_frames"
LEVEL = "wb97m-d3bj_def2-tzvppd"
MACE_LEVEL = "mace-off23_medium"
FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


def main():
    import torch
    from ase.io import read
    from openqha.data import dataset, frames
    from openqha.training import phl, phl_loss
    try:
        from mace import data as mdata, tools as mtools
        from mace.tools import torch_geometric
        from mace.tools.default_keys import DefaultKeys
        DefaultKeys.HESSIAN
    except (ImportError, AttributeError) as exc:
        print("SKIP: the installed mace has no Hessian key ({}); install the fork".format(exc))
        return 0
    torch.set_default_dtype(torch.float64)

    ref_frames = frames.read_frames(FIX / "basin.{}.extxyz".format(LEVEL)) + \
        frames.read_frames(FIX / "displaced.{}.extxyz".format(LEVEL))
    mace_basin = frames.read_frames(FIX / "basin.{}.extxyz".format(MACE_LEVEL))[0]
    check("fixture: 1 basin + 4 displaced reference frames, all with a Hessian",
          len(ref_frames) == 5 and all(a.info["has_hessian"] for a in ref_frames))
    H_fix = [np.array(a.info["hessian"]) for a in ref_frames]
    for a in ref_frames[1:]:                                    # round-5 Q7 (b): displaced frames are E-F only
        a.info.pop("hessian")
        a.info["has_hessian"] = False

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "mace_test.{}.extxyz".format(LEVEL)
        dataset._write_split(path, [(a, "train") for a in ref_frames], reference=True)
        atoms_list = read(str(path), index=":", format="extxyz")
        check("the MACE-form file has 5 frames with split, REF_energy, REF_forces; REF_hessian on the basin frame only",
              len(atoms_list) == 5 and all("REF_energy" in a.info and "REF_forces" in a.arrays and a.info["split"] == "train" for a in atoms_list)
              and [("REF_hessian" in a.info) for a in atoms_list] == [True, False, False, False, False])
        check("REF_energy == energy and REF_forces == forces on every frame",
              all(abs(a.info["REF_energy"] - a.get_potential_energy()) < 1e-12 and np.abs(a.arrays["REF_forces"] - a.get_forces()).max() < 1e-12
                  for a in atoms_list))

        ks = mdata.KeySpecification.from_defaults()
        configs = [mdata.config_from_atoms(a, key_specification=ks) for a in atoms_list]
        check("config_from_atoms: a (3N, 3N) Hessian on the basin frame, None on the displaced ones, weights 1 / 0",
              configs[0].properties["hessian"].shape == (30, 30) and all(c.properties["hessian"] is None for c in configs[1:])
              and configs[0].property_weights["hessian"] == 1.0 and all(c.property_weights["hessian"] == 0.0 for c in configs[1:]))
        check("... the basin Hessian equals the fixture's to 1e-10", np.abs(configs[0].properties["hessian"] - H_fix[0]).max() < 1e-10)
        check("... energy and forces read under the REF_ keys",
              abs(configs[0].properties["energy"] - atoms_list[0].info["REF_energy"]) < 1e-12
              and np.abs(configs[0].properties["forces"] - atoms_list[0].arrays["REF_forces"]).max() < 1e-12)

        table = mtools.AtomicNumberTable(sorted({int(z) for a in atoms_list for z in a.numbers}))
        ads = [mdata.AtomicData.from_config(c, z_table=table, cutoff=5.0) for c in configs]
        batch = next(iter(torch_geometric.dataloader.DataLoader(dataset=ads, batch_size=5, shuffle=False)))
        check("AtomicData batch: hessian [900], has_hessian T F F F F, sqrt_masses [50] = sqrt(ASE masses)",
              tuple(batch.hessian.shape) == (900,) and batch.has_hessian.tolist() == [True, False, False, False, False]
              and tuple(batch.sqrt_masses.shape) == (50,)
              and np.abs(batch.sqrt_masses[:10].numpy() - np.sqrt(atoms_list[0].get_masses())).max() < 1e-12)

        labels = phl_loss.graph_labels(batch)
        check("phl_loss.graph_labels slices the batch: 5 graphs, the basin's Hessian to 1e-10, the others None",
              len(labels) == 5 and labels[0][2] is not None and np.abs(labels[0][2] - H_fix[0]).max() < 1e-10
              and all(l[2] is None for l in labels[1:]) and np.abs(labels[0][3] - atoms_list[0].get_masses()).max() < 1e-12)

        # the loss's full-matrix term on this batch, with the MACE-level fixture Hessian as the model's
        loss = phl_loss.WeightedEnergyForcesHessianLoss()
        n_nodes = int(batch.ptr[-1])
        H_pred = torch.zeros(3 * n_nodes, 3 * n_nodes)
        H_pred[:30, :30] = torch.tensor(mace_basin.info["hessian"])
        term = loss.hessian_error_full(batch, dict(hessian=H_pred.reshape(3 * n_nodes, n_nodes, 3)))
        exact = phl.loss_full(mace_basin.info["hessian"], H_fix[0])
        check("the loss's full-matrix term on the batch = loss_full(H_mace, H_ref) of the fixture (1e-12)",
              abs(float(term) - exact) < 1e-12 and loss.last_terms["n_labelled"] == 1, (float(term), exact))
        print("  (2-methyloxirane basin: MACE vs wB97M Cartesian loss {:.4e} eV^2/A^4 per element)".format(exact))

        # a corrupted label is refused by name
        bad = atoms_list[0].copy(); bad.info = dict(atoms_list[0].info)
        bad.info["REF_hessian"] = np.asarray(bad.info["REF_hessian"])[:-1]
        try:
            mdata.config_from_atoms(bad, key_specification=ks)
            check("a REF_hessian of the wrong length is refused naming the frame", False)
        except ValueError as exc:
            check("a REF_hessian of the wrong length is refused naming the frame", "dsgdb9nsd_000044" in str(exc) and "basin" in str(exc), str(exc))

    print("\n{} checks, {} failed".format(10, len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
