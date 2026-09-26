"""Ticket 03 of the Hessian-learning set: reference E-F-H labels per frame, on the
propanal fixture with a FAKE ORCA (no binary): the runner copies the fixture's
wB97M-D3BJ/def2-TZVPPd `job.hess` / `job.out` (ORCA 6.0.1, basin 0's own minimum) into
the frame's run directory, and the basin frame of the Frame set is placed at exactly the
.hess geometry so the round trip can be checked against ORCA's own frequencies. The
frame's files land as `<molecule>/orca.<level>.<gen>_bBB_kK.{inp,out,hess,engrad}`
(ticket 09: a file group of the molecule directory, no per-frame directory).

Asserted: the keyword line is the level's single point + EnGrad + Freq (analytic) or
NumFreq (numerical); the labelled extxyz holds the MACE file's positions verbatim, the
.out's energy in eV, the gradient's negative in eV/A and the .hess Hessian in eV/A^2, whose
projected frequencies reproduce the .hess file's own to < 0.5 cm^-1; the noise floor is
read before projection and is a few cm^-1 for the analytic matrix; a frame whose .hess
geometry is 1e-6 A off the MACE file is refused and not written; a finished job is reused
(the runner is not called again), a FAILED one (a .out without the terminal line) is not
rerun unless `retry=True` (one attempt per frame, round 11 Q3), a frame with no .out is;
the one-shot retry (ticket 02, seam B): a retry archives the failed .out as
<stem>.failed.out BEFORE ORCA runs (the archive is on disk, and the .out is gone, WHILE
the runner runs), a retry whose archive already exists is refused without running, a
failed retry keeps the archive and stays failed, a finished frame + retry is skipped with
no archive, `--force` re-runs a finished frame and still archives a failed one first; the
round's selection (ticket 04, seam C, `03_labels.py`): the default round carries the
failed frames without an archive alongside the never-run frames, marking them `retry` in
the task list's 5th column; `--retry-only` takes exactly those and nothing else;
the counts are the disk's either way, and neither the round driver nor the worker ever
passes `--force`; the printed round carries no full frame list and no per-molecule walk
lines, and each task's log prints its own retry slice (ticket 05, seam C);
the lock (ticket 24): held only while Slurm does not call its job dead AND its heartbeat
is fresh (a fake `squeue` on PATH answers RUNNING / COMPLETING / exit 1; without `squeue`
the heartbeat alone), the holder touches it every HEARTBEAT_S, a SIGTERM during ORCA
releases it and leaves no .out (a cut, `SystemExit(143)`), a `timeout_s` kill leaves a
.out ending with the TIMEOUT_S trailer (a failure); the Record's counts add
up and its STATUS is NORMAL TERMINATION; the tianhe `labels` role is 16 x 4 with maxcore
6000 and the executor label is registered; hpc/env/orca.sh exists and exports the three
variables; `orca.subprocess_env` prepends S0_ORCA_PATH / S0_ORCA_LIB.
"""
import contextlib
import importlib.util
import io
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
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

FIX = layout.msrrho_dir(SRC)
FIX_STEM = layout.orca_level_stem("wb97m-d3bj_def2-tzvppd", 0)
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
        shutil.copy2(FIX / (FIX_STEM + ".out"), out)
        if "Freq" not in Path(inp).read_text(encoding="utf-8").split("\n")[0]:
            # a GRADIENT job (displaced frame, round 5 Q7 (b)): an .engrad at the input's own
            # geometry -- energy and gradient from the fixture .out, coordinates from the .inp
            lines = [l for l in Path(inp).read_text(encoding="utf-8").split("\n")]
            i0 = next(i for i, l in enumerate(lines) if l.startswith("* xyz"))
            geo = [l.split() for l in lines[i0 + 1:] if len(l.split()) == 4]
            from ase.data import atomic_numbers
            txt = (FIX / (FIX_STEM + ".out")).read_text(encoding="utf-8", errors="replace")
            grad = frame_labels.gradient_from_out(txt, len(geo))
            e = frame_labels.first_energy_from_out(txt)
            body = ["#", "# Number of atoms", "#", " {}".format(len(geo)), "#", "# The current total energy in Eh", "#",
                    " {:.12f}".format(e), "#", "# The current gradient in Eh/bohr", "#"]
            body += ["{:.12f}".format(v) for v in grad.reshape(-1)]
            body += ["#", "# The atomic numbers and current coordinates in Bohr", "#"]
            body += ["{:4d} {:.10f} {:.10f} {:.10f}".format(atomic_numbers[s], *(float(v) * orca.BOHR_PER_ANGSTROM for v in xyz))
                     for s, *xyz in geo]
            (cwd / "job.engrad").write_text("\n".join(body) + "\n", encoding="utf-8")
            return 0
        text = (FIX / (FIX_STEM + ".hess")).read_text(encoding="utf-8")
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


def _raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


def _load_driver():
    """`workflows/hessian_learning/03_labels.py` as a module (its name is not an
    identifier): the round's default selection, its `--retry-only` sweep and its task
    list are exercised through it (ticket 04, seam C)."""
    path = ROOT / "workflows" / "hessian_learning" / "03_labels.py"
    spec = importlib.util.spec_from_file_location("hl_03_labels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def place_basin_at_hess(mol):
    """Move the fixture's basin 0 (MACE) geometry onto the .hess geometry so the fake
    label is at the frame's own coordinates; rewrite basin.extxyz and rebuild the
    Frame set with the harmonic surrogate."""
    parsed = orca.parse_hess(FIX / (FIX_STEM + ".hess"))
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
        text = (FIX / (FIX_STEM + ".out")).read_text(encoding="utf-8", errors="replace")
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
        for f in layout.frames_dir(mol).glob(layout.orca_frame_stem(LEVEL, "basin", 0, 0) + ".*"):     # relabel basin 0 (the .hess geometry)
            f.unlink()
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
        r4 = {(r["BASIN"], r["K"]): r for r in out4["frames"]}
        check("an ORCA that does not terminate raises (the .out stays) and the frame is FAILED: run() does not rerun it (runner called once), "
              "the Record counts N_FAILED 1 / N_UNLABELLED 0 with the reason naming the .out",
              raised and fake4.calls == 1 and not out4["failures"] and out4["info"]["N_FAILED"] == 1
              and out4["info"]["N_UNLABELLED"] == 0 and out4["info"]["N_REUSED"] == 1
              and r4[(2, 0)]["STATUS"] == "failed" and "not rerun" in r4[(2, 0)]["REASON"]
              and layout.orca_frame_file(mol, LEVEL, "basin", 2, 0, ".out").is_file()
              and not (layout.frames_dir(mol) / ("." + layout.orca_frame_stem(LEVEL, "basin", 2, 0))).exists(),
              (raised, fake4.calls, out4["failures"], out4["info"]["N_FAILED"], out4["info"]["N_UNLABELLED"]))
        lab4b = frame_labels.label_one(mol, LEVEL, "basin", 2, 0, runner=fake4, mace_level_name=mlevel)
        fake4c = FakeOrca()
        arc4 = frame_labels.failed_archive(layout.frames_dir(mol), layout.orca_frame_stem(LEVEL, "basin", 2, 0))
        lab4c = frame_labels.label_one(mol, LEVEL, "basin", 2, 0, runner=fake4c, mace_level_name=mlevel, retry=True)
        check("label_one on a failed frame returns status failed without running; retry=True reruns it (the fixture .hess is basin 0's geometry, so basin 2 is refused -- it ran) "
              "and the failed .out survives the retry as its archive (.failed.out, ticket 02)",
              lab4b["status"] == "failed" and fake4.calls == 1 and lab4c["status"] == "refused" and fake4c.calls == 1
              and arc4.read_text(encoding="utf-8") == "ORCA started\nsomething went wrong\n",
              (lab4b["status"], fake4.calls, lab4c["status"], fake4c.calls, arc4.read_text(encoding="utf-8") if arc4.is_file() else None))
        arc4.unlink()          # fixture cleanup: the flatten test at the end moves every orca.<level>.* file and knows only real frames

        # --- the timeout kill leaves a .out with the trailer: a failure, not a cut ----------
        def timeout_runner(inp, out, cwd, timeout_s=None):
            Path(out).write_text("ORCA started\nSCF iteration 12\n", encoding="utf-8")
            raise subprocess.TimeoutExpired(cmd="orca", timeout=timeout_s)
        stem_t = layout.orca_frame_stem(LEVEL, "displaced", 2, 0)
        try:
            frame_labels.label_one(mol, LEVEL, "displaced", 2, 0, runner=timeout_runner, mace_level_name=mlevel, timeout_s=7)
            t_raised = False
        except RuntimeError as exc:
            t_raised = "TIMEOUT_S=7" in str(exc)
        out_t = layout.frames_dir(mol) / (stem_t + ".out")
        check("timeout_s: TimeoutExpired is caught, the partial .out comes back ending with the TIMEOUT_S trailer, label_one raises with it, the frame is failed and the lock released",
              t_raised and out_t.is_file() and out_t.read_text(encoding="utf-8").rstrip().endswith("openQHA: ORCA killed after TIMEOUT_S=7 s")
              and frame_labels.failed(layout.frames_dir(mol), stem_t)
              and not frame_labels.lock_file(layout.frames_dir(mol), stem_t).exists(), (t_raised, out_t.is_file()))

        # --- the lock without Slurm: the heartbeat age alone -------------------------------
        folder = layout.frames_dir(mol)
        path0 = os.environ.get("PATH", "")
        nosq = Path(tmp) / "nosqueue"
        nosq.mkdir()
        for tool in ("python", "python3", "bash", "sh", "cp", "echo"):
            found = shutil.which(tool)
            if found and not (nosq / tool).exists():
                os.symlink(found, nosq / tool)
        os.environ["PATH"] = str(nosq)                                # a PATH without squeue
        frame_labels._squeue_cache.clear()
        lock = frame_labels.lock_file(folder, layout.orca_frame_stem(LEVEL, "displaced", 1, 0))
        lock.write_text("999 now\n", encoding="utf-8")
        fake6 = FakeOrca()
        lab6 = frame_labels.label_one(mol, LEVEL, "displaced", 1, 0, runner=fake6, mace_level_name=mlevel)
        calls_after_claim = fake6.calls
        os.utime(lock, (time.time() - 40 * 60,) * 2)                 # no heartbeat for 40 min
        lab6b = frame_labels.label_one(mol, LEVEL, "displaced", 1, 0, runner=fake6, mace_level_name=mlevel)
        check("without squeue: a lock touched now holds the frame (runner not called, status running); one not touched for 40 min "
              "(> LOCK_MAX_AGE_S 30 min) is taken over and released",
              frame_labels._job_alive("999") is None and lab6["status"] == "running" and calls_after_claim == 0
              and lab6b["status"] == "labelled" and fake6.calls == 1
              and not lock.exists() and lock.name == "orca.wb97m-d3bj_def2-tzvppd.displaced_b01_k0.running",
              (lab6["status"], fake6.calls, lab6b["status"], lock.name))

        # --- the lock with Slurm: a fake squeue answers for the job the lock names -----------
        fakebin = Path(tmp) / "fakebin"
        fakebin.mkdir()
        sq = fakebin / "squeue"
        os.environ["PATH"] = str(fakebin) + os.pathsep + path0

        def squeue_says(script):
            sq.write_text("#!/bin/bash\n" + script + "\n", encoding="utf-8")
            sq.chmod(0o755)
            frame_labels._squeue_cache.clear()

        stem_l = layout.orca_frame_stem(LEVEL, "displaced", 1, 1)
        lock2 = frame_labels.lock_file(folder, stem_l)
        lock2.write_text("4242 now\n", encoding="utf-8")            # touched now
        squeue_says('echo RUNNING')
        held_alive = frame_labels.running_elsewhere(folder, stem_l)
        os.utime(lock2, (time.time() - 40 * 60,) * 2)                # job alive, heartbeat dead
        held_alive_stale = frame_labels.running_elsewhere(folder, stem_l)
        os.utime(lock2, None)
        squeue_says('echo COMPLETING')
        held_completing = frame_labels.running_elsewhere(folder, stem_l)
        squeue_says('echo "CANCELLED by 1000"')
        held_cancelled = frame_labels.running_elsewhere(folder, stem_l)
        squeue_says('echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1')
        held_unknown = frame_labels.running_elsewhere(folder, stem_l)
        squeue_says('echo WEIRD_NEW_STATE')
        held_weird = frame_labels.running_elsewhere(folder, stem_l)
        squeue_says('sleep 30')
        frame_labels.SQUEUE_TIMEOUT_S, saved_to = 1, frame_labels.SQUEUE_TIMEOUT_S
        held_timeout = frame_labels.running_elsewhere(folder, stem_l)
        frame_labels.SQUEUE_TIMEOUT_S = saved_to
        check("with squeue: RUNNING + fresh heartbeat holds; RUNNING + 40 min silence frees; COMPLETING, 'CANCELLED by', "
              "an unknown id (exit 1) free at once; a state the dead list never saw counts as alive; a squeue that "
              "hangs past SQUEUE_TIMEOUT_S is no answer -> the heartbeat (fresh) holds",
              held_alive and not held_alive_stale and not held_completing and not held_cancelled and not held_unknown
              and held_weird and held_timeout,
              (held_alive, held_alive_stale, held_completing, held_cancelled, held_unknown, held_weird, held_timeout))
        squeue_says('echo COMPLETING')
        stem_d0 = layout.orca_frame_stem(LEVEL, "displaced", 0, 0)
        lock_d0 = frame_labels.lock_file(folder, stem_d0)
        lock_d0.write_text("4243 now\n", encoding="utf-8")
        fake6c = FakeOrca()
        lab6c = frame_labels.label_one(mol, LEVEL, "displaced", 0, 0, runner=fake6c, mace_level_name=mlevel)
        check("a lock whose job Slurm calls COMPLETING is taken over by label_one (the parsl retry after a block's time limit)",
              lab6c["status"] == "labelled" and fake6c.calls == 1 and not lock_d0.exists(), (lab6c["status"], fake6c.calls))
        os.environ["PATH"] = path0
        frame_labels._squeue_cache.clear()
        lock2.unlink()
        for f in folder.glob(stem_d0 + ".*"):                        # displaced 0 back to never run
            f.unlink()

        # --- the heartbeat: the holder touches its lock while ORCA runs ----------------------
        stem_h = stem_d0
        lock_h = frame_labels.lock_file(folder, stem_h)
        seen = {}

        def slow_runner(inp, out, cwd, timeout_s=None):
            seen["t0"] = lock_h.stat().st_mtime
            time.sleep(2.6)
            seen["t1"] = lock_h.stat().st_mtime
            return FakeOrca()(inp, out, cwd, timeout_s)
        frame_labels.HEARTBEAT_S, saved_hb = 1, frame_labels.HEARTBEAT_S
        lab_h = frame_labels.label_one(mol, LEVEL, "displaced", 0, 0, runner=slow_runner, mace_level_name=mlevel)
        frame_labels.HEARTBEAT_S = saved_hb
        for f in folder.glob(stem_d0 + ".*"):                        # and back to never run again
            f.unlink()
        check("the heartbeat: with HEARTBEAT_S = 1 the lock's mtime advanced by >= 1.5 s during a 2.6 s ORCA; released after",
              lab_h["status"] == "labelled" and seen["t1"] - seen["t0"] >= 1.5 and not lock_h.exists()
              and not any(t.name == "lock-heartbeat" and t.is_alive() for t in threading.enumerate()),
              (lab_h["status"], seen.get("t1", 0) - seen.get("t0", 0)))

        # --- SIGTERM during ORCA: a cut -- the lock is released, nothing comes back ------------
        stem_s = stem_d0                                             # the cut must leave it never run for the scratch test

        def sleepy_runner(inp, out, cwd, timeout_s=None):
            Path(out).write_text("ORCA started\n", encoding="utf-8")
            time.sleep(5)
            return 0
        threading.Timer(0.8, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
        try:
            frame_labels.label_one(mol, LEVEL, "displaced", 0, 0, runner=sleepy_runner, mace_level_name=mlevel)
            code = None
        except SystemExit as exc:
            code = exc.code
        check("SIGTERM while ORCA runs: SystemExit(143), the lock released, no <stem>.out (a cut, not a failure: the frame reads as never run), "
              "the previous handler restored",
              code == 143 and not frame_labels.lock_file(folder, stem_s).exists()
              and not (folder / (stem_s + ".out")).exists() and not frame_labels.failed(folder, stem_s)
              and signal.getsignal(signal.SIGTERM) == signal.SIG_DFL,
              (code, (folder / (stem_s + ".out")).exists(), signal.getsignal(signal.SIGTERM)))

        # --- scratch: the run happens elsewhere and only KEEP files come back ---------
        scratch = Path(tmp) / "scratch"
        fake5 = FakeOrca()
        lab5 = frame_labels.label_one(mol, LEVEL, "displaced", 0, 0, runner=fake5, mace_level_name=mlevel, scratch=scratch)
        stem5 = layout.orca_frame_stem(LEVEL, "displaced", 0, 0)
        have5 = sorted(f.name for f in layout.frames_dir(mol).glob(stem5 + ".*"))
        check("with scratch: ORCA ran in scratch, job.{inp,out,engrad} copied back as <stem>.{inp,out,engrad} (no .hess: a gradient job), the scratch copy removed",
              have5 == [stem5 + ".engrad", stem5 + ".inp", stem5 + ".out"]
              and stem5 == "orca.wb97m-d3bj_def2-tzvppd.displaced_b00_k0"
              and not (scratch / mol.name / "displaced_b00_k0").exists() and lab5["status"] == "labelled",
              (have5, lab5["status"]))
        n_dirs = sorted(d.name for d in mol.iterdir() if d.is_dir())
        f_dirs = sorted(d.name for d in layout.frames_dir(mol).iterdir() if d.is_dir())
        check("the frame labels are files of frames/: no orca/ or per-frame directory anywhere; the level with a slash is refused",
              "orca" not in n_dirs and not f_dirs and layout.orca_frame_file(mol, LEVEL, "displaced", 0, 0, ".out").parent == layout.frames_dir(mol)
              and _raises(lambda: layout.orca_frame_stem("WB97M/tz", "basin", 0, 0)), (n_dirs, f_dirs))

        # --- Q7 (b): a displaced frame is a gradient job; the mixed file and Record --------
        kw_d = frame_labels.keyword_line(LEVEL, hessian=frame_labels.wants_hessian("displaced"))
        check("a displaced frame's keyword line is the single point + EnGrad only (route gradient); basin / merged / saddle keep Freq",
              kw_d[0] == "wB97M-D3BJ def2-TZVPPD TightSCF EnGrad" and kw_d[2] == "gradient"
              and frame_labels.wants_hessian("basin") and frame_labels.wants_hessian("saddle") and not frame_labels.wants_hessian("displaced"))
        check("the gradient label: energy and forces present, hessian None, geometry check from the .engrad coordinates (dev < 1e-7), route gradient, floor nan",
              lab5["hessian"] is None and lab5["max_position_dev_A"] < 1e-7 and lab5["hessian_route"] == "gradient"
              and np.isnan(lab5["noise_floor_cm"]) and lab5["forces"].shape == (10, 3), (lab5["max_position_dev_A"], lab5["hessian_route"]))
        out7 = frame_labels.assemble(mol, LEVEL, generators=("basin", "displaced"))
        fd = frames.read_frames(layout.frames_file(mol, "displaced", LEVEL))
        r7 = {(r["GENERATOR"], r["BASIN"], r["K"]): r for r in out7["frames"]}
        check("assemble: the displaced file carries has_hessian=false and no hessian key; the Record counts N_HESSIAN_FRAMES / N_GRADIENT_FRAMES and HAS_HESSIAN per row",
              len(fd) == 2 and all(a.info["has_hessian"] is False and "hessian" not in a.info for a in fd)
              and out7["info"]["N_GRADIENT_FRAMES"] == 2 and out7["info"]["N_HESSIAN_FRAMES"] == 1
              and r7[("displaced", 0, 0)]["HAS_HESSIAN"] is False and r7[("basin", 0, 0)]["HAS_HESSIAN"] is True,
              (len(fd), out7["info"]["N_GRADIENT_FRAMES"], out7["info"]["N_HESSIAN_FRAMES"]))

        # --- tickets 09 / 09b: the flatten script moves the pre-2026-09-20 folder forms to the file groups
        before = {g: layout.frames_file(mol, g, LEVEL).read_bytes() for g in ("basin", "displaced")}
        moved_back = 0
        for f in sorted(layout.frames_dir(mol).glob("orca.{}.*".format(LEVEL))):
            engine, rest = f.name.split(".", 1)
            level, frame, ext = rest[:len(LEVEL)], rest[len(LEVEL) + 1:].rsplit(".", 1)[0], f.suffix
            old_dir = mol / "orca" / level / "frames" / frame
            old_dir.mkdir(parents=True, exist_ok=True)
            f.rename(old_dir / ("job" + ext))
            moved_back += 1
        # the msRRHO study in its old form: a basin job with a probe, two level folders, the entropy engines
        for rel, text in (("orca/{}/basin00/job.out".format(LEVEL), "x"), ("orca/{}/basin00/job.hess".format(LEVEL), "h"),
                          ("orca/{}/basin00/k0/job.out".format(LEVEL), "p"),
                          ("levels/{}/thermo_msrrho.toml".format(LEVEL), "t"), ("levels/{}/merge_map.dat".format(LEVEL), "m"),
                          ("levels/mace-off23_medium/degeneracy.out", "d"), ("levels/level_compare.toml", "c"),
                          ("crest_entropy/run01/cre_members", "e"), ("xtb/entropy_run01/conf00/hessian", "q")):
            (mol / rel).parent.mkdir(parents=True, exist_ok=True)
            (mol / rel).write_text(text, encoding="utf-8")
        sys.path.insert(0, str(ROOT / "scripts" / "tooling"))
        import s0_flatten_tree as flat
        # stage 1 on a copy of the tag in the pre-2026-09-20 shard form: <tag>/1_16000/1_1000/<qid>
        tag_dir = mol.parent
        old_mol = tag_dir / "1_16000" / "1_1000" / mol.name
        old_mol.parent.mkdir(parents=True)
        mol.rename(old_mol)
        (tag_dir / "_label" / "acetone" / "_records").mkdir(parents=True)
        smoves, sfolders = flat.plan_shards(tag_dir)
        ns, cs, rs = flat.apply(smoves, sfolders)
        smoves2, _ = flat.plan_shards(tag_dir)
        check("s0_flatten_tree stage 1: <tag>/<range>/<chunk>/<qid> and <tag>/_label/<label> move up to <tag>/<qid>, the shard folders go, a second plan is empty",
              ns == 2 and cs == 0 and rs == 3 and mol.is_dir() and (tag_dir / "acetone" / "_records").is_dir()
              and not (tag_dir / "1_16000").exists() and not (tag_dir / "_label").exists() and not smoves2
              and flat.molecules(tag_dir) == [mol], (ns, cs, rs, sorted(p.name for p in tag_dir.iterdir())))
        moves, folders = flat.plan_frames(mol)
        n_moved, n_conf, n_rm = flat.apply(moves, folders)
        moves2, _ = flat.plan_frames(mol)
        out9 = frame_labels.assemble(mol, LEVEL, generators=("basin", "displaced"))
        after = {g: layout.frames_file(mol, g, LEVEL).read_bytes() for g in ("basin", "displaced")}
        check("s0_flatten_tree stage 2: every job.<ext> of orca/<level>/frames/<frame>/ becomes frames/orca.<level>.<frame><ext>, the emptied folders go, the basinNN/ job stays for stage 3, a second plan is empty, assemble reproduces the label files byte for byte",
              n_moved == moved_back and n_conf == 0 and not (mol / "orca" / LEVEL / "frames").exists()
              and (mol / "orca" / LEVEL / "basin00" / "job.out").is_file() and not moves2
              and after == before and out9["info"]["N_LABELLED"] == 3,
              (n_moved, moved_back, n_conf, n_rm, len(moves2), out9["info"]["N_LABELLED"]))
        m3, f3 = flat.plan_msrrho(mol)
        n3, c3, r3 = flat.apply(m3, f3)
        m3b, _ = flat.plan_msrrho(mol)
        ms = layout.msrrho_dir(mol)
        want = {layout.orca_level_file(mol, LEVEL, 0, ".out"): "x", layout.orca_level_file(mol, LEVEL, 0, ".hess"): "h",
                ms / (layout.orca_level_stem(LEVEL, 0) + ".k0.out"): "p",
                layout.level_file(mol, LEVEL, "thermo_msrrho.toml"): "t", layout.level_file(mol, LEVEL, "merge_map.dat"): "m",
                layout.level_file(mol, "mace-off23_medium", "degeneracy.out"): "d", layout.thermo_file(mol, "level_compare.toml"): "c",
                layout.crest_entropy_dir(mol, 1) / "cre_members": "e", layout.xtb_entropy_dir(mol, 1, 0) / "hessian": "q"}
        got = {k: (k.read_text(encoding="utf-8") if k.is_file() else None) for k in want}
        left = sorted(d.name for d in mol.iterdir() if d.is_dir())
        check("s0_flatten_tree stage 3: the basin job and its probe become msrrho/orca.<level>.basin00[.<probe>].*, levels/ becomes msrrho/thermo/<level>.<name> and bare cross-level names, crest_entropy/ and xtb/ move whole; orca/ levels/ gone; levels_present reads the level back; a second plan is empty",
              n3 == 9 and c3 == 0 and got == want and not m3b and layout.levels_present(mol) == [LEVEL]
              and left == ["_records", "frames", "mace", "msrrho"], (n3, c3, r3, left, {str(k.relative_to(mol)): v for k, v in got.items() if v != want[k]}))

        # --- ticket 02, seam B: the one-shot retry and the archive ---------------------
        folder = layout.frames_dir(mol)
        stem_r = layout.orca_frame_stem(LEVEL, "displaced", 2, 0)    # failed: the TIMEOUT_S trailer
        arc_r = frame_labels.failed_archive(folder, stem_r)
        failed_before = (folder / (stem_r + ".out")).read_text(encoding="utf-8")
        probe = {}

        def probing_runner(inp, out, cwd, timeout_s=None):
            probe.update(archive_during_run=arc_r.is_file(),
                         archived=arc_r.read_text(encoding="utf-8") if arc_r.is_file() else None,
                         out_gone=not (folder / (stem_r + ".out")).is_file())
            return FakeOrca()(inp, out, cwd, timeout_s)

        lab_r = frame_labels.label_one(mol, LEVEL, "displaced", 2, 0, runner=probing_runner,
                                       mace_level_name=mlevel, retry=True)
        check("(i) a retry archives the failed .out BEFORE ORCA runs: WHILE the runner runs the archive holds the old output and no .out is on disk; "
              "on success the new files land and the archive stays",
              "TIMEOUT_S=7" in failed_before and probe.get("archive_during_run") is True
              and probe.get("archived") == failed_before and probe.get("out_gone") is True
              and lab_r["status"] == "labelled" and frame_labels.finished(folder, stem_r, hessian=False)
              and arc_r.read_text(encoding="utf-8") == failed_before,
              (probe, lab_r["status"], failed_before[:40]))

        stem_f = layout.orca_frame_stem(LEVEL, "displaced", 1, 0)    # plant a failure (the frame was finished)
        for ext in (".inp", ".out", ".engrad"):
            try:
                (folder / (stem_f + ext)).unlink()
            except OSError:
                pass
        (folder / (stem_f + ".out")).write_text("ORCA started\nSCF died\n", encoding="utf-8")
        fake_f = FakeOrca(fail=True)
        try:
            frame_labels.label_one(mol, LEVEL, "displaced", 1, 0, runner=fake_f, mace_level_name=mlevel, retry=True)
            raised_f = False
        except RuntimeError as exc:
            raised_f = "did not finish normally" in str(exc)
        arc_f = frame_labels.failed_archive(folder, stem_f)
        check("(iii) a retry whose ORCA fails again keeps the archive and stays failed -- and is no longer retryable",
              raised_f and fake_f.calls == 1 and arc_f.read_text(encoding="utf-8") == "ORCA started\nSCF died\n"
              and frame_labels.failed(folder, stem_f) and not frame_labels.retryable(folder, stem_f),
              (raised_f, fake_f.calls, frame_labels.retryable(folder, stem_f)))

        fake_ii = FakeOrca()
        lab_ii = frame_labels.label_one(mol, LEVEL, "displaced", 1, 0, runner=fake_ii, mace_level_name=mlevel, retry=True)
        check("(ii) a retry on a failed frame whose archive exists is refused WITHOUT running: status failed, the runner never called (the once-only cap)",
              lab_ii["status"] == "failed" and fake_ii.calls == 0 and arc_f.is_file(),
              (lab_ii["status"], fake_ii.calls))

        stem_q = layout.orca_frame_stem(LEVEL, "displaced", 0, 0)    # finished (the scratch test)
        fake_q = FakeOrca()
        lab_q = frame_labels.label_one(mol, LEVEL, "displaced", 0, 0, runner=fake_q, mace_level_name=mlevel, retry=True)
        check("(iv) a finished frame + retry: skipped as reused, nothing runs, no archive is written (only a failed .out is archived)",
              lab_q["status"] == "reused" and fake_q.calls == 0
              and not frame_labels.failed_archive(folder, stem_q).is_file()
              and frame_labels.finished(folder, stem_q, hessian=False),
              (lab_q["status"], fake_q.calls))

        fake_w = FakeOrca()
        lab_w = frame_labels.label_one(mol, LEVEL, "displaced", 0, 0, runner=fake_w, mace_level_name=mlevel, force=True)
        pre_force = (folder / (stem_f + ".out")).read_text(encoding="utf-8")
        fake_w2 = FakeOrca(fail=True)
        try:
            frame_labels.label_one(mol, LEVEL, "displaced", 1, 0, runner=fake_w2, mace_level_name=mlevel, force=True)
            raised_w = False
        except RuntimeError:
            raised_w = True
        check("--force (off by default; never wired into a round) re-runs a finished frame, and on a failed frame still archives first: the one slot is replaced by the .out it overwrites",
              lab_w["status"] == "labelled" and fake_w.calls == 1
              and not frame_labels.failed_archive(folder, stem_q).is_file()
              and raised_w and fake_w2.calls == 1 and arc_f.read_text(encoding="utf-8") == pre_force,
              (lab_w["status"], fake_w.calls, raised_w, arc_f.read_text(encoding="utf-8")[:40] if arc_f.is_file() else None))

        # --- ticket 04, seam C: the round carries the retry; --retry-only sweeps it -------
        root_c = Path(tmp) / "retry_root"
        mol_c, basins_c = make_molecule(Path(tmp) / "retry_src", with_merged=False)
        dest_c = layout.molecule_dir(root_c, "fake", "dsgdb9nsd_000042")
        dest_c.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(mol_c), str(dest_c))
        rec_c = dest_c / "_records" / "branchA.toml"
        rec_c.write_text(rec_c.read_text(encoding="utf-8").replace("dsgdb9nsd_000035", dest_c.name), encoding="utf-8")
        frames.generate(dest_c, calc=Harmonic(basins_c), engine_name="MACE-OFF23_medium", n_displaced=1)
        folder_c = layout.frames_dir(dest_c)

        def put_c(g, b, k, text):
            (folder_c / (layout.orca_frame_stem(LEVEL, g, b, k) + ".out")).write_text(text, encoding="utf-8")

        put_c("basin", 0, 0, "x\n" + frame_labels.TERMINAL + "\n")               # finished
        (folder_c / (layout.orca_frame_stem(LEVEL, "basin", 0, 0) + ".hess")).write_text("", encoding="utf-8")
        put_c("displaced", 1, 0, "y\n" + frame_labels.TERMINAL + "\n")          # finished
        (folder_c / (layout.orca_frame_stem(LEVEL, "displaced", 1, 0) + ".engrad")).write_text("", encoding="utf-8")
        put_c("basin", 1, 0, "ORCA started\ncrashed\n")                         # failed, no archive
        put_c("basin", 2, 0, "ORCA started\ncrashed\n")                         # failed, archived
        frame_labels.failed_archive(folder_c, layout.orca_frame_stem(LEVEL, "basin", 2, 0)).write_text("older failure\n", encoding="utf-8")

        # ticket 05: a second molecule, every frame finished -- its walk line ("all N frames
        # finished") used to be the only thing a round's log said about it
        mol_e, basins_e = make_molecule(Path(tmp) / "retry_src2", with_merged=False)
        dest_e = layout.molecule_dir(root_c, "fake", "dsgdb9nsd_000043")
        shutil.move(str(mol_e), str(dest_e))
        rec_e = dest_e / "_records" / "branchA.toml"
        rec_e.write_text(rec_e.read_text(encoding="utf-8").replace("dsgdb9nsd_000035", dest_e.name), encoding="utf-8")
        frames.generate(dest_e, calc=Harmonic(basins_e), engine_name="MACE-OFF23_medium", n_displaced=1)
        folder_e = layout.frames_dir(dest_e)
        for g_e, b_e, k_e in frame_labels.frame_list(dest_e):
            stem_e = layout.orca_frame_stem(LEVEL, g_e, b_e, k_e)
            (folder_e / (stem_e + ".out")).write_text("x\n" + frame_labels.TERMINAL + "\n", encoding="utf-8")
            (folder_e / (stem_e + (".hess" if frame_labels.wants_hessian(g_e) else ".engrad"))).write_text("", encoding="utf-8")

        ld = _load_driver()
        todo_default, counts_default = ld.pending([dest_c], LEVEL, None)
        todo_only, counts_only = ld.pending([dest_c], LEVEL, None, retry_only=True)
        default_ids = [(g, b, k, r) for _m, g, b, k, r in todo_default]
        only_ids = [(g, b, k, r) for _m, g, b, k, r in todo_only]
        check("seam C, pending(): the default round lists the never-run frames AND the failed frame WITHOUT an archive (retry True); --retry-only lists exactly that failure -- "
              "the archived failure, the finished and the never-run frames stay unqueued; the counts are the disk's either way",
              sorted(default_ids) == sorted([("basin", 1, 0, True), ("displaced", 0, 0, False), ("displaced", 2, 0, False)])
              and sorted(only_ids) == [("basin", 1, 0, True)]
              and counts_default == counts_only == {dest_c.name: (6, 2, 2)},
              (default_ids, only_ids, counts_default, counts_only))

        old_root = os.environ.get("S0_RUNS_ROOT")
        os.environ["S0_RUNS_ROOT"] = str(root_c)
        try:
            cap_d, cap_o = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(cap_d):
                rc_d = ld.main(["--tag", "fake", "--all", "--level", LEVEL, "--list", str(Path(tmp) / "plain_list.txt")])
            with contextlib.redirect_stdout(cap_o):
                rc_o = ld.main(["--tag", "fake", "--all", "--level", LEVEL, "--retry-only",
                                "--list", str(Path(tmp) / "only_list.txt")])
        finally:
            if old_root is None:
                os.environ.pop("S0_RUNS_ROOT", None)
            else:
                os.environ["S0_RUNS_ROOT"] = old_root
        listing_d = (Path(tmp) / "plain_list.txt").read_text(encoding="utf-8")
        listing_o = (Path(tmp) / "only_list.txt").read_text(encoding="utf-8")
        text_d, text_o = cap_d.getvalue(), cap_o.getvalue()
        want_d = sorted(["{} basin 1 0 retry".format(dest_c),
                         "{} displaced 0 0 -".format(dest_c),
                         "{} displaced 2 0 -".format(dest_c)])
        check("seam C, --list: a flagless round's task list carries the failed frame without an archive with `retry` in the 5th column and the never-run frames with `-`; "
              "--retry-only's list holds exactly the failure and nothing else; the summary states the retry rule and the retry count; "
              "ticket 05: the printed round holds NO full frame list and NO per-molecule walk lines -- no frame tag, no `(retry)` marker, "
              "no `frames finished` line, and the all-finished molecule is never named",
              rc_d == 0 and rc_o == 0
              and sorted(listing_d.splitlines()) == want_d
              and listing_o.splitlines() == ["{} basin 1 0 retry".format(dest_c)]
              and "1 of them the one retry of a failed frame without an archive" in text_d
              and "the failed .out is archived as <stem>.failed.out before ORCA starts" in text_d
              and "8 finished already" in text_d
              and "ONLY the failed frames without an archive; nothing else is queued" in text_o
              and "nothing else is queued" not in text_d
              and "frame list:" not in text_d and "frame list:" not in text_o
              and "basin_b01_k0" not in text_d and "basin_b01_k0" not in text_o
              and "displaced_b00_k0" not in text_d and "displaced_b02_k0" not in text_d
              and "(retry)" not in text_d and "(retry)" not in text_o
              and "frames finished" not in text_d and "frames finished" not in text_o
              and dest_e.name not in text_d and dest_e.name not in text_o,
              (rc_d, rc_o, listing_d, listing_o, text_d[:200], text_o[:200]))

        drv_t = (ROOT / "workflows" / "hessian_learning" / "03_labels.py").read_text(encoding="utf-8")
        fl_t = (ROOT / "openqha" / "data" / "frame_labels.py").read_text(encoding="utf-8")
        wk_t = (ROOT / "hpc" / "slurm" / "hl_label_worker.sh").read_text(encoding="utf-8")
        sl_t = (ROOT / "hpc" / "slurm" / "hl_labels.slurm").read_text(encoding="utf-8")
        dbg_t = (ROOT / "hpc" / "slurm" / "hl_pipeline_debug.slurm").read_text(encoding="utf-8")
        check("--force lives only on the frame CLI (the deliberate full relabel): the round driver and the worker never pass it; the worker maps the 5th column to --retry; "
              "hl_labels.slurm takes RETRY_ONLY (any non-empty value) into `--retry-only`, used exactly once, on the LIST call only; RETRY_FAILED / --retry-failed are gone from the driver, the script, the worker and the frame CLI; "
              "the old ordinary-round claims ('not rerun') are gone from the driver and the round script; both arrays carry the column through their awk and dispatch 6 fields",
              "--force" not in drv_t and "--force" not in wk_t and "--force" in fl_t
              and '[ "$retry" = "retry" ] && cmd+=(--retry)' in wk_t
              and "-n 6" in sl_t and "$5, s, e" in sl_t
              and "RETRY_ONLY" in sl_t and "RETRY_FAILED" not in sl_t and "--retry-failed" not in sl_t
              and sl_t.count("--retry-only") == 1 and sl_t.count("$RETRY_FLAG") == 1
              and "RETRY_FAILED" not in wk_t and "--retry-failed" not in wk_t
              and "RETRY_FAILED" not in fl_t and "--retry-failed" not in fl_t
              and "RETRY_FAILED" not in drv_t and "--retry-failed" not in drv_t and "retry_failed" not in drv_t
              and "--retry-only" in drv_t and "retry_only=args.retry_only" in drv_t
              and "not rerun" not in drv_t and "not rerun" not in sl_t
              and '-n 6' in dbg_t and "$4, $5, s, s + np - 1" in dbg_t,
              ("--force" in drv_t, "--force" in wk_t, "-n 6" in sl_t, "$5, s, e" in sl_t, "$4, $5, s, s + np - 1" in dbg_t))
        check("the scope ruling rides the round: hl_labels.slurm takes GENERATORS into BOTH the task list and the assemble, so task 0's exit 0 means every IN-SCOPE frame is labelled",
              'GENERATORS="${GENERATORS:-}"' in sl_t and 'GENERATORS_FLAG="--generators $GENERATORS"' in sl_t
              and sl_t.count("$GENERATORS_FLAG") == 2 and "$GENERATORS_FLAG --assemble" in sl_t
              and "GENERATORS=basin" in sl_t and "GENERATORS=basin RETRY_ONLY=1" in sl_t and "--generators" in drv_t,
              (sl_t.count("$GENERATORS_FLAG"), "$GENERATORS_FLAG --assemble" in sl_t, "GENERATORS=basin" in sl_t))
        check("ticket 05: each task's log prints its own retry slice after the echoes -- the count line `retries    R in this task` and one line per retry frame (molecule name + generator_bBB_kK, like the old list)",
              'echo "retries    $N_RETRY_TASK in this task"' in sl_t
              and '$5 == "retry"' in sl_t and 'n = split($1, p, "/")' in sl_t and '%s_b%02d_k%d' in sl_t,
              ('echo "retries' in sl_t, '$5 == "retry"' in sl_t, 'n = split($1' in sl_t, "%s_b%02d_k%d" in sl_t))

    # --- ticket 09 A: one campaign, one tag, one Dataset -------------------------------
    heads = "".join((ROOT / "hpc" / "slurm" / f).read_text(encoding="utf-8") for f in
                    ("hl_branchA.slurm", "hl_frames.slurm", "hl_labels.slurm", "hl_pipeline_debug.slurm"))
    readme = (ROOT / "workflows" / "hessian_learning" / "README.md").read_text(encoding="utf-8")
    check("NAME defaults to TAG in every stage script; no `NAME=draw` left in the slurm headers or the README; the steps' --name is optional",
          heads.count('NAME="${NAME:-$TAG}"') == 4 and "NAME=draw" not in heads and "NAME=draw" not in readme
          and all('"--name", default=None' in (ROOT / "workflows" / "hessian_learning" / s).read_text(encoding="utf-8")
                  for s in ("00_draw.py", "01_select.py", "04_dataset.py"))
          and '"--name", default=None' in (ROOT / "hpc" / "slurm" / "hl_list.py").read_text(encoding="utf-8"),
          heads.count('NAME="${NAME:-$TAG}"'))

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
    import providers as _prov
    check("providers.normalise_walltime: Slurm's day form becomes the HH:MM:SS parsl parses (3-00:00:00 -> 72:00:00, 7-00:00:00 -> 168:00:00, 00:30:00 kept)",
          _prov.normalise_walltime("3-00:00:00") == "72:00:00" and _prov.normalise_walltime("7-00:00:00") == "168:00:00"
          and _prov.normalise_walltime("00:30:00") == "00:30:00" and _prov.normalise_walltime("1-12:30") == "36:30:00")
    cfg_wt = tc.config(role="labels", max_blocks=1, walltime="7-00:00:00", run_dir="/tmp/parsl_render_t")
    check("the tianhe labels provider carries the normalised walltime (7-00:00:00 -> 168:00:00)",
          cfg_wt.executors[0].provider.walltime == "168:00:00", cfg_wt.executors[0].provider.walltime)
    sh = (ROOT / "hpc" / "env" / "orca.sh").read_text(encoding="utf-8")
    check("hpc/env/orca.sh: enters env_orca611.sh, reads ORCA_PATH / type -P (never an alias), exports S0_ORCA_BIN/PATH/LIB, restores PATH and LD_LIBRARY_PATH",
          all(s in sh for s in ("env_orca611.sh", "export S0_ORCA_BIN S0_ORCA_PATH S0_ORCA_LIB", 'export PATH="$_path"',
                                'export LD_LIBRARY_PATH="$_ld"', "conda deactivate", "$ORCA_PATH/orca", "type -P orca", "unalias orca")))
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
