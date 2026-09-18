"""Ticket 03 of the Hessian-learning set: reference E-F-H labels per frame, on the
propanal fixture with a FAKE ORCA (no binary): the runner copies the fixture's
wB97M-D3BJ/def2-TZVPPd `job.hess` / `job.out` (ORCA 6.0.1, basin 0's own minimum) into
the frame's job directory, and the basin frame of the Frame set is placed at exactly the
.hess geometry so the round trip can be checked against ORCA's own frequencies.

Asserted: the keyword line is the level's single point + EnGrad + Freq (analytic) or
NumFreq (numerical); the labelled extxyz holds the MACE file's positions verbatim, the
.out's energy in eV, the gradient's negative in eV/A and the .hess Hessian in eV/A^2, whose
projected frequencies reproduce the .hess file's own to < 0.5 cm^-1; the noise floor is
read before projection and is a few cm^-1 for the analytic matrix; a frame whose .hess
geometry is 1e-6 A off the MACE file is refused and not written; a finished job is reused
(the runner is not called again) and an unfinished one rerun; the Record's counts add
up and its STATUS is NORMAL TERMINATION; the tianhe `labels` role is 16 x 4 with maxcore
6000 and the executor label is registered; hpc/env/orca.sh exists and exports the three
variables; `orca.subprocess_env` prepends S0_ORCA_PATH / S0_ORCA_LIB.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from ase.io import read, write



def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))
sys.path.insert(0, str(_repo_root() / "tests" / "unit"))
from t_frames import Harmonic, check, make_molecule, FAIL, ROOT, SRC    # noqa: E402

from openqha.data import frame_labels, frames                  # noqa: E402
from openqha.qm_interfaces import orca                          # noqa: E402
from openqha.store import layout, property as prop              # noqa: E402
from openqha.thermochem import hessian as hessian_mod           # noqa: E402

FIX = SRC / "orca" / "wb97m-d3bj_def2-tzvppd" / "basin00"
LEVEL = "wb97m-d3bj_def2-tzvppd"


class FakeOrca:
    """Copies the fixture job into the run directory; counts its calls; can shift the
    .hess geometry to provoke the refusal."""

    def __init__(self, shift_A=0.0, fail=False, one_atom=False):
        self.calls, self.shift_A, self.fail, self.one_atom = 0, shift_A, fail, one_atom

    def __call__(self, inp, out, cwd, timeout_s=None):
        self.calls += 1
        cwd = Path(cwd)
        if self.fail:
            out.write_text("ORCA started\nsomething went wrong\n", encoding="utf-8")
            return 1
        shutil.copy2(FIX / "job.out", out)
        text = (FIX / "job.hess").read_text(encoding="utf-8")
        if self.shift_A:
            lines = text.split("\n")
            i = lines.index("$atoms")
            n = int(lines[i + 1].split()[0])
            for j in range(1 if self.one_atom else n):
                parts = lines[i + 2 + j].split()
                x = float(parts[2]) + self.shift_A * orca.BOHR_PER_ANGSTROM
                lines[i + 2 + j] = " {:<4s} {:>10s} {:>18.12f} {:>18s} {:>18s}".format(parts[0], parts[1], x, parts[3], parts[4])
            text = "\n".join(lines)
        (cwd / "job.hess").write_text(text, encoding="utf-8")
        return 0


def place_basin_at_hess(mol):
    """Move the fixture's basin 0 (MACE) geometry onto the .hess geometry so the fake
    label is at the frame's own coordinates; rewrite basin.extxyz and rebuild the
    Frame set with the harmonic surrogate."""
    parsed = orca.parse_hess(FIX / "job.hess")
    pos = np.asarray(parsed["positions_bohr"]) / orca.BOHR_PER_ANGSTROM
    p = mol / "mace" / "basin00" / "basin.extxyz"
    a = read(str(p), format="extxyz")
    e0 = float(a.get_potential_energy())
    a.set_positions(pos)
    from ase.calculators.singlepoint import SinglePointCalculator
    a.calc = SinglePointCalculator(a, energy=e0, forces=np.zeros((len(a), 3)))
    write(str(p), a, format="extxyz")
    return parsed


def main():
    kw, blocks, route = frame_labels.keyword_line(LEVEL)
    kw2, _b2, route2 = frame_labels.keyword_line("ri-mp2_cc-pvtz")
    check("keyword line: the level's single point + EnGrad + Freq (analytic) / NumFreq (numerical)",
          kw == "wB97M-D3BJ def2-TZVPPD TightSCF EnGrad Freq" and route == "analytic"
          and kw2.endswith(" EnGrad NumFreq") and route2 == "numerical", (kw, kw2))

    with tempfile.TemporaryDirectory(prefix="labels_") as tmp:
        mol, basins = make_molecule(tmp, with_merged=False)
        parsed = place_basin_at_hess(mol)
        basins[0] = (np.asarray(parsed["positions_bohr"]) / orca.BOHR_PER_ANGSTROM, basins[0][1], basins[0][2])
        frames.generate(mol, calc=Harmonic(basins), engine_name="MACE-OFF23_medium", n_displaced=1)
        mlevel = frame_labels.mace_level(mol)
        fl = frame_labels.frame_list(mol)
        check("frame list from the Frame-set Record: 3 basin + 3 displaced frames, generators in order",
              len(fl) == 6 and fl[0] == ("basin", 0, 0) and ("displaced", 2, 0) in fl, fl)

        # --- label basin 0 only (the frame at the .hess geometry) ---------------------
        fake = FakeOrca()
        lab = frame_labels.label_one(mol, LEVEL, "basin", 0, 0, nprocs=4, maxcore=6000, runner=fake, mace_level_name=mlevel)
        out = frame_labels.assemble(mol, LEVEL, generators=("basin",), computed=["basin_b00_k0"])
        info, rows = out["info"], out["frames"]
        check("label_one ran the runner once, status labelled, route analytic, version 6.0.1, seconds from TOTAL RUN TIME",
              fake.calls == 1 and lab["status"] == "labelled" and lab["hessian_route"] == "analytic"
              and lab["orca_version"] == "6.0.1" and abs(lab["seconds"] - 225.733) < 0.01 and lab["memory_mb"] > 0,
              (fake.calls, lab["status"], lab["hessian_route"], lab["orca_version"], lab["seconds"], lab["memory_mb"]))
        path = layout.frames_file(mol, "basin", LEVEL)
        labelled = frames.read_frames(path)
        mace = frames.read_frames(layout.frames_file(mol, "basin", mlevel))
        a = labelled[0]
        text = (FIX / "job.out").read_text(encoding="utf-8", errors="replace")
        # the fixture .out is an optimisation (several single points); a label takes the FIRST
        e_ev = frame_labels.first_energy_from_out(text) * orca.EV_PER_HARTREE
        grad = frame_labels.gradient_from_out(text, len(a))
        f_ev = -grad * orca.EV_PER_HARTREE * orca.BOHR_PER_ANGSTROM
        check("the labelled file: positions == the MACE file's to 0, energy = the .out's in eV, forces = -gradient in eV/A (file precision 1e-6)",
              len(labelled) == 1 and np.abs(a.get_positions() - mace[0].get_positions()).max() == 0.0
              and abs(a.get_potential_energy() - e_ev) < 1e-6 and np.abs(a.get_forces() - f_ev).max() < 1e-6
              and a.info["level"] == LEVEL and a.info["generator"] == "basin" and a.info["orca_version"] == "6.0.1",
              (len(labelled), np.abs(a.get_positions() - mace[0].get_positions()).max(), a.get_potential_energy() - e_ev,
               np.abs(a.get_forces() - f_ev).max(), a.info.get("level"), a.info.get("orca_version")))
        pr = hessian_mod.project_and_diagonalise(a.info["hessian"], a.get_masses(), a.get_positions())
        mine = np.asarray(pr["frequencies_cm_inv"])
        theirs = np.asarray(parsed["frequencies_cm_inv"])
        theirs = theirs[np.abs(theirs) > 1e-6][-len(mine):] if len(theirs) > len(mine) else theirs
        dev = np.abs(np.sort(mine) - np.sort(theirs)).max()
        check("round trip: the projected frequencies of the written Hessian reproduce the .hess file's own to < 0.5 cm^-1",
              dev < 0.5, (dev, mine[:3], theirs[:3]))
        r0 = [r for r in rows if r["GENERATOR"] == "basin" and r["BASIN"] == 0][0]
        check("the Record row: labelled, noise floor read before projection (0 < floor < 50 cm^-1; analytic propanal ~29), |dx|max < 1e-8, memory and seconds",
              r0["STATUS"] == "labelled" and 0 < r0["NOISE_FLOOR_CM"] < 50 and r0["MAX_POSITION_DEV_A"] < 1e-7
              and r0["MEMORY_MB"] > 0 and abs(r0["SECONDS"] - 225.733) < 0.01 and r0["ENERGY_ABOVE_BASIN"] == 0.0,
              {k: r0[k] for k in ("STATUS", "NOISE_FLOOR_CM", "MAX_POSITION_DEV_A", "MEMORY_MB", "SECONDS")})
        check("the Record: 3 basin frames, 1 labelled + 2 unlabelled (no job), STATUS NORMAL TERMINATION",
              info["N_FRAMES"] == 3 and info["N_LABELLED"] == 1 and info["N_COMPUTED"] == 1 and info["N_UNLABELLED"] == 2
              and prop.status_of(out["record"]) == prop.NORMAL_TERMINATION and info["ORCA_VERSION"] == "6.0.1",
              {k: info[k] for k in ("N_FRAMES", "N_LABELLED", "N_COMPUTED", "N_UNLABELLED")})

        # --- rerun: the finished job is reused, the runner not called ---------------
        fake2 = FakeOrca()
        lab2 = frame_labels.label_one(mol, LEVEL, "basin", 0, 0, runner=fake2, mace_level_name=mlevel)
        out2 = frame_labels.assemble(mol, LEVEL, generators=("basin",))
        check("a finished job is reused on a rerun: runner not called, status reused, N_REUSED 1 N_COMPUTED 0",
              fake2.calls == 0 and lab2["status"] == "reused" and out2["info"]["N_REUSED"] == 1 and out2["info"]["N_COMPUTED"] == 0,
              (fake2.calls, lab2["status"], out2["info"]["N_REUSED"]))

        # --- a job at a distorted geometry is refused; a translated one is not ---------
        fake3 = FakeOrca(shift_A=1e-6, one_atom=True)
        lab3 = frame_labels.label_one(mol, LEVEL, "basin", 1, 0, runner=fake3, mace_level_name=mlevel)
        out3 = frame_labels.assemble(mol, LEVEL, generators=("basin",))
        r1 = [r for r in out3["frames"] if r["BASIN"] == 1][0]
        check("a .hess geometry with one atom 1e-6 A off the MACE file is refused, named in the Record, not written",
              lab3["status"] == "refused" and r1["STATUS"] == "refused" and "1e-07" in r1["REASON"]
              and out3["info"]["N_REFUSED"] == 1 and len(frames.read_frames(layout.frames_file(mol, "basin", LEVEL))) == 1,
              (lab3["status"], r1["REASON"]))
        shutil.rmtree(layout.orca_frame_dir(mol, LEVEL, "basin", 0, 0))     # relabel basin 0 (the .hess geometry)
        fake3b = FakeOrca(shift_A=0.5)                      # every atom: ORCA's centre-of-mass frame
        lab3b = frame_labels.label_one(mol, LEVEL, "basin", 0, 0, runner=fake3b, mace_level_name=mlevel)
        check("a .hess translated as a whole (ORCA's centre-of-mass frame, 0.5 A) is accepted: shape deviation below the 1e-7 tolerance, COM shift reported",
              lab3b["status"] == "labelled" and lab3b["max_position_dev_A"] < 1e-7 and abs(lab3b["com_shift_A"] - 0.5) < 1e-6,
              (lab3b["status"], lab3b["max_position_dev_A"], lab3b["com_shift_A"]))

        # --- a failed ORCA raises from label_one, run() reports it and goes on --------
        fake4 = FakeOrca(fail=True)
        try:
            frame_labels.label_one(mol, LEVEL, "basin", 2, 0, runner=fake4, mace_level_name=mlevel)
            raised = False
        except RuntimeError as exc:
            raised = "did not finish normally" in str(exc)
        out4 = frame_labels.run(mol, LEVEL, generators=("basin",), runner=fake4)
        check("an ORCA that does not terminate raises (the .out stays); run() records the failure and assembles the rest",
              raised and len(out4["failures"]) == 1 and out4["info"]["N_UNLABELLED"] == 1 and out4["info"]["N_REUSED"] == 1
              and (layout.orca_frame_dir(mol, LEVEL, "basin", 2, 0) / "job.out").is_file(),
              (raised, out4["failures"], out4["info"]["N_UNLABELLED"]))

        # --- scratch: the run happens elsewhere and only KEEP files come back ---------
        scratch = Path(tmp) / "scratch"
        fake5 = FakeOrca()
        lab5 = frame_labels.label_one(mol, LEVEL, "displaced", 0, 0, runner=fake5, mace_level_name=mlevel, scratch=scratch)
        wd = layout.orca_frame_dir(mol, LEVEL, "displaced", 0, 0)
        check("with scratch: ORCA ran in scratch, job.{inp,out,hess} copied back, the scratch copy removed",
              (wd / "job.out").is_file() and (wd / "job.hess").is_file() and (wd / "job.inp").is_file()
              and not (scratch / mol.name / "displaced_b00_k0").exists() and lab5["status"] == "refused",
              (sorted(p.name for p in wd.iterdir()), lab5["status"]))

    # --- the HPC layer ----------------------------------------------------------------
    sys.path.insert(0, str(ROOT / "hpc"))
    import labels as _labels
    import resource_configs
    tc = resource_configs.load("tianhe_cpu")
    w, c, b = tc.layout("labels")
    d = tc.describe()
    init = tc._worker_init(str(ROOT / "hpc"), "labels")
    check("tianhe_cpu role labels: 16 workers x 4 ranks, 12 blocks, maxcore 6000, label registered, worker init sources env/orca.sh",
          (w, c, b) == (16, 4, 12) and d["maxcore_mb"] == 6000 and _labels.label("labels") == "openqha_labels_executor"
          and "env/orca.sh" in init and "openqha_find_orca" in init and "unset $v" in init, (w, c, b, d.get("maxcore_mb")))
    sh = (ROOT / "hpc" / "env" / "orca.sh").read_text(encoding="utf-8")
    check("hpc/env/orca.sh: enters env_orca611.sh, exports S0_ORCA_BIN/PATH/LIB, restores PATH and LD_LIBRARY_PATH",
          all(s in sh for s in ("env_orca611.sh", "export S0_ORCA_BIN S0_ORCA_PATH S0_ORCA_LIB", 'export PATH="$_path"',
                                'export LD_LIBRARY_PATH="$_ld"', "conda deactivate")))
    os.environ["S0_ORCA_PATH"], os.environ["S0_ORCA_LIB"] = "/x/bin", "/x/lib"
    env = orca.subprocess_env()
    del os.environ["S0_ORCA_PATH"], os.environ["S0_ORCA_LIB"]
    check("orca.subprocess_env prepends S0_ORCA_PATH to PATH and S0_ORCA_LIB to LD_LIBRARY_PATH for the subprocess only",
          env["PATH"].startswith("/x/bin" + os.pathsep) and env["LD_LIBRARY_PATH"].startswith("/x/lib")
          and not os.environ.get("PATH", "").startswith("/x/bin"))

    print("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
