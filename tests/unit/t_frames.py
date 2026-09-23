"""Ticket 02 of the Hessian-learning set: the Frame set of one molecule at the engine
level, on the propanal fixture with a harmonic surrogate calculator (no MACE weights,
no engine): E = E_b + 1/2 dx^T H_b dx, F = -H_b dx, Hessian = H_b of the nearest basin.

Asserted: 3 basin + 12 displaced frames with the declared keys and their own seeds;
the basin frame's Hessian is the stored hessian.npy to 0 and its forces the surrogate's
zero; the displaced frames stay within the RMS ceiling and the same seed reproduces
the same positions to 1e-12 while another seed does not; `merged` and `saddle` frames
come from `DUPLICATE_MAP` / `SADDLE_CONFORMER_IDS` and `mace/confNN/conf.extxyz`; a frame
with a broken bond is kept and reported (no bond-graph filter: ANI-1 / SPICE ruling
2026-09-18), the energy window is the only filter and names the frame's dE; the extxyz
files read back with a (3N, 3N) Hessian and the Record's counts add up.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from openqha.data import frames                              # noqa: E402
from openqha.store import layout, property as prop           # noqa: E402
from openqha.thermochem import hessian as hessian_mod        # noqa: E402

SRC = ROOT / "tests" / "data" / "propanal_molecule"
FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


class Harmonic(Calculator):
    """The quadratic surface of the nearest basin's stored Hessian."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, basins, force_scale=1.0):
        super().__init__()
        self.basins = basins            # list of (x0, e0, H)
        self.force_scale = force_scale

    def _nearest(self, atoms):
        x = atoms.get_positions()
        return min(self.basins, key=lambda b: np.sqrt(((x - b[0]) ** 2).sum(1).mean()))

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x0, e0, h = self._nearest(atoms)
        dx = (atoms.get_positions() - x0).reshape(-1)
        self.results["energy"] = float(e0 + 0.5 * dx @ h @ dx)
        self.results["forces"] = (-(h @ dx)).reshape(-1, 3) * self.force_scale

    def get_hessian(self, atoms=None):
        x0, e0, h = self._nearest(atoms)
        n = len(atoms)
        return h.reshape(3 * n, n, 3)


def make_molecule(tmp, with_merged=True):
    mol = Path(tmp) / "dsgdb9nsd_000035"
    shutil.copytree(SRC, mol, ignore=shutil.ignore_patterns("orca", "levels", "msrrho", "crest", "frames"))
    basins = []
    for b in range(3):
        a = read(str(mol / "mace" / "basin{:02d}".format(b) / "basin.extxyz"), format="extxyz")
        basins.append((a.get_positions().copy(), float(a.get_potential_energy()),
                       np.load(mol / "mace" / "basin{:02d}".format(b) / "hessian.npy")))
    if with_merged:
        # input frames 0, 1, 2 became basins 0, 2, 1 (BASIN_CONFORMER_IDS = [0, 2, 1]); add
        # frame 3 merged into 0 and frame 4 a saddle, with tightened conformers on disk
        for cid, (x0, e0, h) in ((3, basins[0]), (4, basins[1])):
            a = read(str(mol / "mace" / "basin00" / "basin.extxyz"), format="extxyz")
            a.set_positions(x0 + 0.01 * (cid - 2))
            d = layout.mace_conformer_dir(mol, cid); d.mkdir(parents=True, exist_ok=True)
            write(str(d / "conf.extxyz"), a, format="extxyz")
        rec = mol / "_records" / "branchA.toml"
        t = rec.read_text(encoding="utf-8")
        t = t.replace("BASIN_CONFORMER_IDS = [0, 2, 1]",
                      "BASIN_CONFORMER_IDS = [0, 2, 1]\nDUPLICATE_MAP = [0, 1, 2, 0, 4]\nSADDLE_CONFORMER_IDS = [4]")
        rec.write_text(t, encoding="utf-8")
    return mol, basins


def main():
    with tempfile.TemporaryDirectory(prefix="frames_") as tmp:
        mol, basins = make_molecule(tmp)
        calc = Harmonic(basins)
        out = frames.generate(mol, calc=calc, engine_name="MACE-OFF23_medium")
        rows = out["frames"]
        n = {g: sum(1 for r in rows if r["GENERATOR"] == g and r["STATUS"] == "kept") for g in frames.GENERATORS}
        check("3 basin + 12 displaced + 1 merged + 1 saddle frames kept, none dropped",
              n == dict(basin=3, displaced=12, merged=1, saddle=1) and out["info"]["N_DROPPED"] == 0, (n, out["info"]["N_DROPPED"]))
        level = out["info"]["LEVEL"]
        fb = frames.read_frames(layout.frames_file(mol, "basin", level))
        h0 = np.load(mol / "mace" / "basin00" / "hessian.npy")
        check("the basin frame carries the stored hessian.npy to 0 and forces ~0, keys qm9_index/basin/generator/k/seed/level/smiles",
              np.abs(fb[0].info["hessian"] - h0).max() == 0.0 and np.abs(fb[0].get_forces()).max() < 1e-6
              and fb[0].info["generator"] == "basin" and fb[0].info["qm9_index"] == "dsgdb9nsd_000035"
              and all(k in fb[0].info for k in ("basin", "k", "seed", "level", "smiles", "engine_params_sha256")),
              sorted(fb[0].info))
        fd = frames.read_frames(layout.frames_file(mol, "displaced", level))
        x0 = basins[0][0]
        # RMS over Cartesian coordinates, hessian.thermal_displacements' definition (not per atom)
        rms = [float(np.sqrt(((a.get_positions() - basins[a.info["basin"]][0]) ** 2).mean())) for a in fd]
        check("12 displaced frames from the ANI-1 draw (the default since 2026-09-23; the Record says so), taken as they come "
              "(no ceiling: MAX_RMS_A None, the Record nan, the energy window the only filter), each with its own "
              "seed = frame_seed(qm9_index, basin, 'displaced', k)",
              len(fd) == 12 and frames.MAX_RMS_A is None and np.isnan(out["info"]["MAX_RMS_A"])
              and frames.DISTRIBUTION == "ani1" and out["info"]["DISTRIBUTION"] == "ani1"
              and all(abs(a.info["rms_displacement_A"] - r) < 1e-7 for a, r in zip(fd, rms))
              and all(a.info["seed"] == frames.frame_seed("dsgdb9nsd_000035", a.info["basin"], "displaced", a.info["k"]) for a in fd)
              and len({a.info["seed"] for a in fd}) == 12, (len(fd), max(rms), out["info"]["MAX_RMS_A"]))
        # an explicit ceiling is still honoured, and a draw that cannot meet one raises
        tight, _rec = hessian_mod.thermal_displacements(
            read(str(mol / "mace" / "basin00" / "basin.extxyz"), format="extxyz"), None, n_samples=2, hessian=h0,
            seeds=[7, 8], max_rms_displacement_A=0.05, distribution=out["info"]["DISTRIBUTION"])
        x00 = basins[0][0]
        tight_rms = [float(np.sqrt(((a.get_positions() - x00) ** 2).mean())) for a in tight]
        try:
            hessian_mod.thermal_displacements(
                read(str(mol / "mace" / "basin00" / "basin.extxyz"), format="extxyz"), None, n_samples=1, hessian=h0,
                seeds=[7], max_rms_displacement_A=1e-9, distribution=out["info"]["DISTRIBUTION"])
            raised = False
        except RuntimeError as exc:
            raised = "exceeded the displacement ceiling" in str(exc)
        check("an explicit --max-rms is still honoured (0.05 A: every draw under it) and an unreachable one raises after 50 draws",
              max(tight_rms) <= 0.05 + 1e-12 and raised, (max(tight_rms), raised))

        # --- ticket 28: ANI-1 bounds the total harmonic energy at (3/2) N_a k_B T ----------
        a00 = read(str(mol / "mace" / "basin00" / "basin.extxyz"), format="extxyz")
        KB_J, NA, CAL = 1.380649e-23, 6.02214076e23, 4184.0
        cap = 1.5 * len(a00) * KB_J * 298.15 * NA / CAL                       # kcal/mol
        ani, arec = hessian_mod.thermal_displacements(a00, None, n_samples=8, hessian=h0,
                                                      seeds=list(range(8)), max_rms_displacement_A=None,
                                                      distribution="ani1")
        dxs = [a.get_positions() - a00.get_positions() for a in ani]
        e_harm = [0.5 * float(d.reshape(-1) @ h0 @ d.reshape(-1)) * frames.EV_TO_KCAL for d in dxs]
        rho = arec["energy_fraction"]
        ani2, _ = hessian_mod.thermal_displacements(a00, None, n_samples=8, hessian=h0, seeds=list(range(8)),
                                                    max_rms_displacement_A=None, distribution="ani1")
        ani3, _ = hessian_mod.thermal_displacements(a00, None, n_samples=8, hessian=h0, seeds=list(range(100, 108)),
                                                    max_rms_displacement_A=None, distribution="ani1")
        same = max(np.abs(ani2[i].get_positions() - ani[i].get_positions()).max() for i in range(8))
        diff = np.abs(ani3[0].get_positions() - ani[0].get_positions()).max()
        gauss, grec = hessian_mod.thermal_displacements(a00, None, n_samples=8, hessian=h0, seeds=list(range(8)),
                                                        max_rms_displacement_A=None, distribution="classical")
        check("the ANI-1 draw: every frame's harmonic energy (1/2 dx^T H dx) is under the cap (3/2) N_a k_B T = {:.1f} kcal/mol "
              "and equals rho x cap to 1e-6 relative; rho in [0, 1]; the same seeds reproduce it, other seeds do not; "
              "the classical branch is untouched".format(cap),
              abs(arec["energy_cap_kcal"] - cap) < 1e-6 * cap
              and all(e <= cap * (1 + 1e-9) for e in e_harm)
              and all(0.0 <= r <= 1.0 for r in rho)
              and max(abs(e - r * cap) for e, r in zip(e_harm, rho)) < 1e-6 * cap
              and same < 1e-12 and diff > 1e-3
              and np.isnan(grec["energy_fraction"][0]) and arec["distribution"] == "ani1",
              (arec["energy_cap_kcal"], cap, max(e_harm), rho[:3], same, diff))
        check("a displaced frame's Hessian has shape (3N, 3N) and its energy sits above the basin (harmonic surrogate)",
              fd[0].info["hessian"].shape == (30, 30) and all(a.get_potential_energy() > basins[a.info["basin"]][1] for a in fd))
        # reproducibility: redraw basin 0's displacements from the recorded seeds
        seeds = [a.info["seed"] for a in fd if a.info["basin"] == 0]
        a0 = read(str(mol / "mace" / "basin00" / "basin.extxyz"), format="extxyz")
        # the draw the Record declares (round-2 Q14: classical by default), not thermal_displacements' own default
        dist = out["info"]["DISTRIBUTION"]
        again, _ = hessian_mod.thermal_displacements(a0, None, n_samples=4, hessian=h0, seeds=seeds,
                                                     max_rms_displacement_A=frames.MAX_RMS_A, distribution=dist)
        same = max(np.abs(again[i].get_positions() - [a for a in fd if a.info["basin"] == 0][i].get_positions()).max() for i in range(4))
        other, _ = hessian_mod.thermal_displacements(a0, None, n_samples=4, hessian=h0, seeds=[s + 1 for s in seeds],
                                                     max_rms_displacement_A=frames.MAX_RMS_A, distribution=dist)
        diff = np.abs(other[0].get_positions() - again[0].get_positions()).max()
        again2, _ = hessian_mod.thermal_displacements(a0, None, n_samples=4, hessian=h0, seeds=seeds,
                                                      max_rms_displacement_A=frames.MAX_RMS_A, distribution=dist)
        exact = max(np.abs(again2[i].get_positions() - again[i].get_positions()).max() for i in range(4))
        check("the same seeds reproduce the same positions: to 1e-12 in memory, to the file's 8 decimals (1e-7) on disk; shifted seeds do not",
              exact < 1e-12 and same < 1e-7 and diff > 1e-3, (exact, same, diff))
        fm = frames.read_frames(layout.frames_file(mol, "merged", level))
        fs = frames.read_frames(layout.frames_file(mol, "saddle", level))
        check("merged frame = input conformer 3 born from basin 0; saddle frame = conformer 4 (home basin by DUPLICATE_MAP)",
              fm[0].info["source_conformer"] == 3 and fm[0].info["basin"] == 0 and fs[0].info["source_conformer"] == 4
              and fs[0].info["generator"] == "saddle", (fm[0].info, fs[0].info))
        doc = prop.load(out["record"])
        gen = {r["GENERATOR"]: r for r in doc["Generator"]}
        check("the Record: [[Generator]] counts equal the files, [[Frame]] has one row per candidate, ENGINE_PIN_STATUS present",
              all(gen[g]["N_FRAMES"] == n[g] for g in n) and len(doc["Frame"]) == 17
              and "ENGINE_PIN_STATUS" in doc["Calculation_Info"], gen)
        check("the Report ends with the terminal line",
              (layout.frames_dir(mol) / "frames.out").read_text().rstrip().endswith("openQHA frames terminated normally"))

        # ---- filters: a broken bond and a huge force ------------------------------
        mol2, basins2 = make_molecule(Path(tmp) / "b", with_merged=True)
        a = read(str(layout.mace_conformer_dir(mol2, 3) / "conf.extxyz"), format="extxyz")
        pos = a.get_positions(); pos[9] += [0.0, 0.0, 1.2]                 # pull the aldehyde H off
        a.set_positions(pos); write(str(layout.mace_conformer_dir(mol2, 3) / "conf.extxyz"), a, format="extxyz")
        out2 = frames.generate(mol2, calc=Harmonic(basins2), engine_name="MACE-OFF23_medium", n_displaced=0)
        merged_row = [r for r in out2["frames"] if r["GENERATOR"] == "merged"][0]
        check("a merged conformer with a bond stretched by 1.2 A is KEPT (no bond-graph filter, as ANI-1 / SPICE) but reported "
              "as 'broken ...' and counted in N_BOND_CHANGED",
              merged_row["STATUS"] == "kept" and merged_row["BOND_CHANGE"].startswith("broken")
              and out2["info"]["N_BOND_CHANGED"] == 1 and out2["info"]["N_DROPPED"] == 0
              and layout.frames_file(mol2, "merged", level).is_file(), merged_row)
        out3 = frames.generate(mol2, calc=Harmonic(basins2, force_scale=1e4), engine_name="MACE-OFF23_medium", n_displaced=1,
                               energy_window=5.0)
        dropped = [r for r in out3["frames"] if r["STATUS"] == "dropped"]
        check("the energy window is the only filter: with a 5 kcal/mol window every off-minimum frame is dropped with its dE named, "
              "the basin frames stay",
              len(dropped) >= 3 and all("kcal/mol above the basin" in r["REASON"] for r in dropped)
              and all(r["GENERATOR"] != "basin" for r in dropped)
              and all(r["STATUS"] == "kept" for r in out3["frames"] if r["GENERATOR"] == "basin"), [(r["GENERATOR"], r["REASON"]) for r in dropped])

    check("frame_seed is stable and distinct across (basin, k)",
          frames.frame_seed("x", 0, "displaced", 0) == frames.frame_seed("x", 0, "displaced", 0)
          and len({frames.frame_seed("x", b, "displaced", k) for b in range(3) for k in range(4)}) == 12)
    if FAIL:
        print("FAIL: " + ", ".join(FAIL))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
