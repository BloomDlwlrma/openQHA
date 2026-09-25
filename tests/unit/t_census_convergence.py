"""Ticket 38: the census certifies the tighten against ORCA's default line.

UNIT. No engine, no CREST, no MACE: `conformers.optimise` is stubbed with a scripted
list of (energy, residual, steps) per call and `thermochem.hessian.hessian` with a
scaled identity, so the whole tighten -> certify -> deduplicate -> floor-screen path
runs on RDKit + numpy only. The wired path stays covered by
tests/integration/t_mace_engine_folder.py.

Held:

    A. `crest_census.convergence_class`: the four outcomes and the transient that
       triggers the single second pass (pure; the prototype's `convergenceClass`,
       amended by the 2026-09-25 ruling -- the class that stays above the line is
       REJECTED where the prototype admitted it with a flag).
    B. `census_from_frames` with the stubbed optimiser: a frame above the line receives
       exactly one second pass; `converged_orca_default` is admitted and carries
       `tighten_converged = false`; every census hessian record carries its
       `convergence_class`; the census counts every class.
    C. A frame still above the line after its second pass is rejected as
       `not_certified`, listed with its residual, and never enters the basin list.
    D. A molecule all of whose candidates are not certified refuses; the message's
       certification section names the line and every residual (the two-section form
       is held by t_branch_a_crash C).
    F. With a molecule directory, the second pass reuses the same engine files -- an
       appended `Trajectory` on the conformer's `opt.traj` and the same `opt.log` -- so
       no new engine-file name is born.

The frames are propanal's real CREST ensemble (tests/data/propanal_crest); the
scripted residuals are the ticket's own numbers: 5e-5 is at or below the repository
target (1e-4 eV/A), 3e-3 is inside ORCA's default line (TolMaxG = 3e-4 Eh/bohr =
1.543e-2 eV/A), 2e-2 is above it.
"""
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
FRAMES = ROOT / "tests" / "data" / "propanal_crest" / "crest_conformers.xyz"
FAIL = []


def check(name, ok, detail=None):
    print("  {} {}".format(name, "ok" if ok else "FAIL" + (" " + repr(detail) if detail is not None else "")))
    if not ok:
        FAIL.append(name)


from openqha.conformer_search import conformers as _conformers   # noqa: E402
from openqha.conformer_search import crest_census                # noqa: E402
from openqha.thermochem import hessian as _hessian_mod           # noqa: E402
import numpy as np                                               # noqa: E402


def run_census(script, n_frames=3, molecule_dir=None):
    """`census_from_frames` on the propanal fixture with a scripted optimiser.

    `script`: one (energy_eV, residual_eV_A, steps) triple per optimiser call, in call
    order. Returns (record, basins, calls); raises like the census itself when it
    refuses. The frames are propanal's real CREST conformers; the scripted energies are
    far apart, so the CREGEN energy condition blocks every merge and each certified
    frame survives on its own. With `molecule_dir`, `_write_extxyz` is stubbed too (the
    real one would need a calculator), so the engine-file names stay assertable.
    """
    frames, comments = crest_census.read_ensemble_atoms(str(FRAMES))
    frames, comments = frames[:n_frames], comments[:n_frames]
    calls = []

    def fake_optimise(atoms, calc, fmax=1e-4, steps=500, logfile=None, trajectory=None):
        calls.append(dict(fmax=fmax, steps=steps, logfile=logfile, trajectory=trajectory))
        e, fm, ns = script[len(calls) - 1]
        return float(e), float(fm), bool(fm <= fmax), int(ns)

    def fake_hessian(atoms, calc, mode="analytic"):
        # a scaled identity: every projected mode positive, so the floor screen admits
        # every candidate -- this test is about the tighten certification, not the floor
        return np.eye(3 * len(atoms)) * 50.0, 0.0

    real_opt, real_hess = _conformers.optimise, _hessian_mod.hessian
    real_write = crest_census._write_extxyz
    _conformers.optimise = fake_optimise
    _hessian_mod.hessian = fake_hessian
    if molecule_dir is not None:
        crest_census._write_extxyz = lambda path, atoms, **info: None
    try:
        rec, basins, _mol = crest_census.census_from_frames(
            "CCC=O", frames, None, comments=comments, hessian_mode="analytic",
            molecule_dir=molecule_dir)
    finally:
        _conformers.optimise = real_opt
        _hessian_mod.hessian = real_hess
        crest_census._write_extxyz = real_write
    return rec, basins, calls


#: One (energy_eV, residual_eV_A, steps) per optimiser call: frame 0 converged on the
#: first pass; frame 1 inside ORCA's line but above our target; frame 2 above the line,
#: so its second pass runs and lands at 5e-3 (inside the line, still above our target).
SCRIPT_MIXED = [(-100.00, 5e-5, 10),
                (-99.90, 3e-3, 20),
                (-99.50, 2e-2, 30),
                (-99.45, 5e-3, 40)]
#: Frame 0 converged; frame 1 above the line and STILL above it after its second pass;
#: frame 2 inside ORCA's line.
SCRIPT_REJECT_ONE = [(-100.00, 5e-5, 10),
                     (-99.90, 2e-2, 30),
                     (-99.85, 1.9e-2, 40),
                     (-99.50, 3e-3, 20)]
#: Every frame above the line, twice: nothing can be certified.
SCRIPT_REJECT_ALL = [(-100.00, 2e-2, 30), (-99.99, 1.95e-2, 40),
                     (-99.50, 2.5e-2, 30), (-99.30, 2.3e-2, 40),
                     (-99.00, 3e-2, 30), (-98.80, 2.8e-2, 40)]


#: Frame 0 converged; frame 1 above the line, so its second pass runs (keep this
#: script's length: an extra call would be an optimiser called twice for one candidate).
SCRIPT_TWO = [(-100.00, 5e-5, 10), (-99.50, 2e-2, 30), (-99.45, 5e-3, 40)]


def main():
    # A -- the pure certification (ticket 38, prototype `convergenceClass`)
    c = crest_census.convergence_class
    check("A: at or below the target -> converged; inside ORCA's line -> "
          "converged_orca_default (the line itself is inside); above the line with no "
          "second pass -> the transient that triggers it",
          c(5e-5) == "converged" and c(1e-4) == "converged"
          and c(3e-3) == "converged_orca_default" and c(1.543e-2) == "converged_orca_default"
          and c(2e-2) == "needs_second_pass",
          (c(5e-5), c(1e-4), c(3e-3), c(1.543e-2), c(2e-2)))
    check("A2: the second pass admits -> converged_second_pass; still above the line -> "
          "not_certified (the ruling: the prototype's admitted_flagged is superseded)",
          c(2e-2, second_pass=True) == "converged_second_pass"
          and c(2e-2, second_pass=False) == "not_certified",
          (c(2e-2, second_pass=True), c(2e-2, second_pass=False)))

    # B -- the orchestration: one second pass, classes recorded, our target judged
    rec, basins, calls = run_census(SCRIPT_MIXED)
    check("B: only the frame above the line receives a second pass -- four calls for "
          "three frames, the fourth carrying the same bounds",
          len(calls) == 4 and calls[3]["fmax"] == calls[2]["fmax"] == 1e-4
          and calls[3]["logfile"] is None and calls[3]["trajectory"] is None,
          [c["fmax"] for c in calls])
    check("B2: the census counts every class and rejects none",
          rec.get("n_converged") == 1 and rec.get("n_converged_orca_default") == 1
          and rec.get("n_converged_second_pass") == 1 and rec.get("n_not_certified") == 0
          and rec.get("not_certified") == [],
          {k: rec.get(k) for k in ("n_converged", "n_converged_orca_default",
                                   "n_converged_second_pass", "n_not_certified")})
    h = rec["hessian"]
    check("B3: every census hessian record carries its convergence class; the frame "
          "inside ORCA's line and the second-pass frame missed OUR target and carry "
          "tighten_converged = false",
          h["0"]["convergence_class"] == "converged" and h["0"]["tighten_converged"] is True
          and h["1"]["convergence_class"] == "converged_orca_default"
          and h["1"]["tighten_converged"] is False
          and h["2"]["convergence_class"] == "converged_second_pass"
          and h["2"]["tighten_converged"] is False
          and rec["basin_conformer_ids"] == [0, 1, 2],
          {k: (v["convergence_class"], v["tighten_converged"]) for k, v in h.items()})

    # B4 -- the bool answers to the FINAL residual, not the first: a second pass that
    # reaches our target is a converged tighten even though its class names the pass
    rec2, _basins2, _calls2 = run_census([SCRIPT_MIXED[0], SCRIPT_MIXED[1],
                                          (-99.50, 2e-2, 30), (-99.50, 8e-5, 40)])
    check("B4: a second pass that reaches the repository target -> class names the "
          "pass, tighten_converged true",
          rec2["hessian"]["2"]["convergence_class"] == "converged_second_pass"
          and rec2["hessian"]["2"]["tighten_converged"] is True,
          (rec2["hessian"]["2"]["convergence_class"], rec2["hessian"]["2"]["tighten_converged"]))

    # C -- a double failure is rejected, listed, and never enters the basin list
    rec3, basins3, calls3 = run_census(SCRIPT_REJECT_ONE)
    check("C: the frame still above the line after its single second pass is rejected "
          "as not_certified with its residual, separate from the floor-screen saddles",
          len(calls3) == 4 and rec3.get("n_not_certified") == 1
          and rec3.get("not_certified") == [dict(conformer_id=1, residual_eV_A=1.9e-2)]
          and rec3.get("n_saddles_rejected") == 0 and rec3.get("saddles") == []
          and rec3["basin_conformer_ids"] == [0, 2] and len(basins3) == 2,
          (rec3.get("not_certified"), rec3.get("basin_conformer_ids")))

    # D -- all candidates not certified: the refusal raises, in the certification's
    # own words, with every residual listed
    try:
        run_census(SCRIPT_REJECT_ALL)
        check("D: a molecule all of whose candidates are not certified refuses", False)
    except RuntimeError as exc:
        msg = str(exc)
        check("D: the refusal carries the certification section (the line, the count, "
              "every residual -- the FINAL one, after the pass) and no floor section "
              "-- nothing was floor-screened",
              "no basin survives" in msg
              and "3 candidate(s) NOT certified" in msg
              and "TolMaxG = 3e-4 Eh/bohr = 1.543e-02 eV/A" in msg
              and "conformer   0: residual 1.95e-02 eV/A" in msg
              and "conformer   1: residual 2.30e-02 eV/A" in msg
              and "conformer   2: residual 2.80e-02 eV/A" in msg
              and "below ithr" not in msg, msg)

    # F -- with a molecule directory, the second pass appends to the SAME engine files:
    # no `opt2.traj`/`opt2.log` is born (ADR 0001's confNN file list stays true).
    with tempfile.TemporaryDirectory(prefix="t38_files_") as td:
        td = Path(td)
        _rec, _basins, calls_f = run_census(SCRIPT_TWO, n_frames=2, molecule_dir=td)
        traj_obj = calls_f[2]["trajectory"]
        check("F: the second pass reuses the conformer's own engine files -- an appended "
              "Trajectory on opt.traj and the same opt.log -- so the engine-file "
              "names do not change",
              len(calls_f) == 3 and not isinstance(traj_obj, str)
              and str(getattr(traj_obj, "filename", "")) == str(td / "mace" / "conf01" / "opt.traj")
              and calls_f[1]["logfile"] == calls_f[2]["logfile"]
              == str(td / "mace" / "conf01" / "opt.log")
              and (td / "mace" / "conf01" / "opt.traj").exists(),
              (type(traj_obj).__name__, calls_f[2].get("logfile")))

    print("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
