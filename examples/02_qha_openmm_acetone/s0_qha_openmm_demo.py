"""Branch B end to end on one molecule, both routes, in a few minutes.

EXAMPLE. It produces no scientific number and is not a production run: the trajectories
here are far too short, and the script says so where it matters rather than printing a
free energy that looks usable.

What it demonstrates, in order:

  1. the estimator is exact on a case whose answer is known in closed form;
  2. the OpenMM force is the same potential as the ASE one, to the last bit;
  3. both production routes run and write the same product contract;
  4. an independent implementation of the superposition either agrees or is shown not to.

Run it as:

    python examples/02_qha_openmm_acetone/s0_qha_openmm_demo.py

One environment runs both routes as of 2026-09-05. It used to need two, because
openmm-torch pins the pytorch it was compiled against; that pin was accepted (pytorch
2.13.0 -> 2.12.1, numpy 2.4.6 -> 1.26.4) and the cost was MEASURED at exactly zero --
energy, forces, every Hessian frequency and T*S on fixed frames all bit-identical
(scripts/calibration/s0_B_stack_fingerprint.py).

The OpenMM half therefore runs in whatever interpreter you start this with. `S0_OPENMM_PYTHON`
still overrides it, for the lean `openqha-openmm` environment from environment-openmm.yml;
and if no interpreter has the stack, that half is SKIPPED with a message saying what is
missing and what it costs, rather than quietly omitted.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def _repo_root():
    for p in Path(__file__).resolve().parents:
        if (p / "openqha" / "__init__.py").is_file():
            return p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))

from openqha import config, engine, hessian, qha           # noqa: E402

SPECIES = "dsgdb9nsd_000018"          # acetone, 10 atoms
EQUIL_PS = 0.5
PROD_PS = 3.0
SEEDS = 2
#: Below this, acceptance criteria 1 and 5 are refused outright by the analysis: a 0.4 ps
#: run once PASSED the saturation criterion, and the reason it passed was that it had not
#: begun to rise.
SMOKE_LENGTH_PS = 20.0


def rule(title):
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def openmm_python():
    """The interpreter that has OpenMM, or None.

    THIS interpreter first: one environment runs everything now. The others are kept as
    fallbacks for a machine that split them -- the lean `openqha-openmm` environment, or
    an older layout -- because a demo that dies on a working installation is worse than
    one that looks one directory further.

    Checked by IMPORT and never by `Path.exists`: a present-but-broken build must read as
    unavailable, which is the same rule openqha/capabilities.py follows.
    """
    candidates = [os.environ.get("S0_OPENMM_PYTHON"),
                  sys.executable,
                  str(Path.home() / "anaconda3" / "envs" / "openqha-openmm" / "bin"
                      / "python"),
                  shutil.which("python")]
    probe = ("import openmm, openmmtools, openmmtorch, mace; "
             "print(openmm.version.version)")
    seen = set()
    for exe in candidates:
        if not exe or exe in seen:
            continue
        seen.add(exe)
        got = subprocess.run([exe, "-c", probe], capture_output=True, text=True)
        if got.returncode == 0:
            return exe, got.stdout.strip()
    return None, None


def step_1_closed_form(atoms, calc):
    """The estimator against an answer that is written down, not against itself."""
    rule("1. The estimator, on a case where the answer is known")
    H, asym = hessian.hessian(atoms, calc, mode="analytic")
    print("Hessian max|H - H^T|          {:.3e} eV/A^2".format(asym))
    check = qha.harmonic_limit_check(H, atoms.get_masses(), atoms.get_positions(),
                                     temperature_K=298.15, n_frames=100000)
    print("closed-form T*S               {:.6f} kcal/mol".format(
        check["reference"]["closed_form_TS_kcal"]))
    print("exact-covariance error        {:.3e} kcal/mol   (the algebra)".format(
        check["exact_covariance_TS_error_kcal"]))
    print("sampled error                 {:+.4f} kcal/mol   (the estimator, budget 0.02)"
          .format(check["sampled_TS_error_kcal"]))
    print("RMS frequency error           {:.4f} cm^-1".format(
        check["rms_frequency_error_cm_inv"]))
    return check


def step_2_force_agreement(atoms, exe):
    """The OpenMM force must be the SAME potential, or nothing below means anything."""
    rule("2. Is the OpenMM force the same potential?")
    if exe is None:
        print("SKIPPED: no interpreter with openmm + openmm-torch + openmmtools.")
        print("         They are CORE for branch B; environment.yml declares them.")
        print("         conda env update -f environment.yml --prune")
        print("         or, for the lean variant: conda env create -f environment-openmm.yml")
        try:
            from openqha import capabilities
            print()
            print(capabilities.summary())
        except Exception:                                     # noqa: BLE001
            pass
        return None
    code = (
        "import sys; sys.path.insert(0, {!r})\n".format(str(ROOT)) +
        "import json\n"
        "from ase.io import read\n"
        "from openqha import config, engine, openmm_mace\n"
        "atoms = read(str(config.qm9_xyz({!r}, config.load())))\n".format(SPECIES) +
        "rec = openmm_mace.verify_against_ase(atoms, engine.model_path())\n"
        "print('OPENQHA_JSON' + json.dumps({k: rec[k] for k in "
        "('n_edges','neighbour_list','delta_energy_eV','max_delta_force_eV_per_A',"
        "'max_delta_force_traced_eV_per_A','agrees')}))\n")
    got = subprocess.run([exe, "-c", code], capture_output=True, text=True)
    line = next((l for l in got.stdout.splitlines() if l.startswith("OPENQHA_JSON")), None)
    if line is None:
        print("FAILED to verify:", (got.stderr or got.stdout)[-600:])
        return None
    rec = json.loads(line[len("OPENQHA_JSON"):])
    print("edges                         {}".format(rec["n_edges"]))
    print("neighbour list                {}".format(rec["neighbour_list"]))
    print("energy difference             {:.3e} eV".format(rec["delta_energy_eV"]))
    print("max force difference          {:.3e} eV/A".format(
        rec["max_delta_force_eV_per_A"]))
    print("traced vs eager               {:.3e} eV/A".format(
        rec["max_delta_force_traced_eV_per_A"]))
    print("agrees                        {}".format(rec["agrees"]))
    return rec


def _run(cmd, label):
    print("$ " + " ".join(str(c) for c in cmd))
    got = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if got.returncode != 0:
        print("{} FAILED:\n{}".format(label, (got.stderr or got.stdout)[-1200:]))
        return False
    for line in got.stdout.splitlines():
        if line.startswith(("  seed", "  basin", "protocol", "thermostat", "force check",
                            "written")):
            print("    " + line.strip())
    return True


def step_3_two_routes(exe):
    rule("3. Two production routes, one product contract")
    print("Trajectories are {} ps. That is a SMOKE LENGTH: acceptance criteria 1 and 5 "
          "are\nrefused below {} ps, because a 0.4 ps run once passed the saturation "
          "criterion and\nthe reason it passed was that it had not begun to rise.\n"
          .format(PROD_PS, SMOKE_LENGTH_PS))
    tags = []
    if _run([sys.executable, ROOT / "scripts/production/s0_B_qha_trajectory.py",
             "--species", SPECIES, "--tag", "ex02_ase", "--seeds", SEEDS,
             "--equil-ps", EQUIL_PS, "--prod-ps", PROD_PS], "ASE route"):
        tags.append("ex02_ase")
    if exe is not None:
        if _run([exe, ROOT / "scripts/production/s0_B_qha_trajectory_openmm.py",
                 "--species", SPECIES, "--tag", "ex02_omm", "--seeds", SEEDS,
                 "--equil-ps", EQUIL_PS, "--prod-ps", PROD_PS], "OpenMM route"):
            tags.append("ex02_omm")
    else:
        print("    OpenMM route SKIPPED (no interpreter)")
    return tags


def step_4_compare(tags):
    rule("4. Do the routes agree, and against what noise?")
    means = {}
    for tag in tags:
        root = Path.home() / "runs" / "openQHA" / "qha" / tag / SPECIES / "basin00"
        values = []
        for seed_dir in sorted(root.glob("seed*")):
            frames = np.load(seed_dir / "frames.npy")
            meta = json.loads((seed_dir / "meta.json").read_text(encoding="utf-8"))
            rec = qha.analyse(frames, meta["masses_amu"], meta["temperature_K"], meta=meta)
            values.append(rec["entropy"]["TS_QH_kcal"])
        values = np.array(values)
        means[tag] = values
        print("{:<10} T*S = {}   mean {:.4f}   seed spread {:.4f}".format(
            tag, np.round(values, 4), values.mean(),
            values.std(ddof=1) if len(values) > 1 else float("nan")))
    if len(means) == 2:
        a, b = means.values()
        print()
        print("difference of means           {:+.4f} kcal/mol".format(b.mean() - a.mean()))
        print("Read that against the seed spread above, not against zero. At {} ps the "
              "spread\nis large and this is a smoke test of the comparison, not a result."
              .format(PROD_PS))
    return means


def _cross_check_here(seed_dir):
    from openqha import mdtraj_io
    frames = np.load(seed_dir / "frames.npy")
    meta = json.loads((seed_dir / "meta.json").read_text(encoding="utf-8"))
    return mdtraj_io.cross_check(frames, meta["masses_amu"], meta["symbols"],
                                 meta["temperature_K"])


def _cross_check_there(seed_dir, exe):
    """Run the same check in the interpreter that actually has mdtraj.

    mdtraj and MDAnalysis live with OpenMM, not with ASE. Reporting "not installed" from
    the wrong interpreter would be a check that failed for a reason it does not measure --
    which is exactly how acceptance criterion 2 once spent a session reporting "no
    comparison produced" because GROMACS was in a sibling environment.
    """
    code = (
        "import sys, json; sys.path.insert(0, {!r})\n".format(str(ROOT)) +
        "import numpy as np\n"
        "from pathlib import Path\n"
        "from openqha import mdtraj_io\n"
        "d = Path({!r})\n".format(str(seed_dir)) +
        "frames = np.load(d / 'frames.npy')\n"
        "meta = json.loads((d / 'meta.json').read_text())\n"
        "rec = mdtraj_io.cross_check(frames, meta['masses_amu'], meta['symbols'],\n"
        "                            meta['temperature_K'])\n"
        "print('OPENQHA_JSON' + json.dumps(rec, default=str))\n")
    got = subprocess.run([exe, "-c", code], capture_output=True, text=True)
    line = next((l for l in got.stdout.splitlines()
                 if l.startswith("OPENQHA_JSON")), None)
    if line is None:
        raise RuntimeError((got.stderr or got.stdout)[-400:])
    return json.loads(line[len("OPENQHA_JSON"):])


def step_5_superposition(tags, exe):
    """An independent fit -- and the reason one library is not enough."""
    rule("5. Independent superposition")
    if not tags:
        print("SKIPPED: no trajectory to check.")
        return
    seed_dir = (Path.home() / "runs" / "openQHA" / "qha" / tags[0] / SPECIES
                / "basin00" / "seed00")
    rec = None
    try:
        rec = _cross_check_here(seed_dir)
    except Exception as exc:                                  # noqa: BLE001
        print("this interpreter: {}: {}".format(type(exc).__name__, exc))
    if (rec is None or all(v is None for v in rec["available"].values())) and exe:
        print("neither library here; running the check in {}".format(exe))
        try:
            rec = _cross_check_there(seed_dir, exe)
        except Exception as exc:                              # noqa: BLE001
            print("SKIPPED: {}: {}".format(type(exc).__name__, exc))
            return
    if rec is None:
        print("SKIPPED: no interpreter has mdtraj or MDAnalysis.")
        return
    print("available                     {}".format(rec["available"]))
    print("ours (mass-weighted, iterated) T*S = {:.4f}".format(rec["ours"]["TS_QH_kcal"]))
    for c in rec["comparisons"]:
        if "TS_QH_kcal" in c:
            print("{:<14} T*S = {:.4f}   diff {:+.4f} kcal/mol   mass-weighted={}".format(
                c["library"], c["TS_QH_kcal"], c["TS_difference_kcal"],
                c["mass_weighted"]))
        else:
            print("{:<14} {}".format(c["library"], c.get("skipped") or c.get("failed")))
    print()
    print("mdtraj cannot be told to weight by mass, so a difference here is the FIT")
    print("PROTOCOL and not the entropy algorithm. On a real 25 ps trajectory it differed")
    print("by 2.41 kcal/mol, while `gmx covar -mwa` -- which CAN mass-weight -- agreed to")
    print("1.27e-04. mdtraj also returned the identity rotation for 300 of 3125 frames")
    print("without raising. See openqha/mdtraj_io.py.")


def main():
    cfg = config.load()
    from ase.io import read
    from ase.optimize import BFGS

    atoms = read(str(config.qm9_xyz(SPECIES, cfg)))
    calc, name, prov = engine.calculator(device="cpu")
    atoms.calc = calc
    BFGS(atoms, logfile=None).run(fmax=1e-4, steps=2000)

    rule("Branch B, end to end on acetone")
    print("species     {}   {} atoms".format(SPECIES, len(atoms)))
    print("engine      {}".format(name))
    print("mace module {}".format(prov.get("mace_module_path")))
    patch = (prov.get("neighbour_list_patch") or {}).get("installed") or {}
    print("neighbour   {}   defect present: {}".format(
        patch.get("sizing"), patch.get("defect_present")))

    exe, omm_version = openmm_python()
    print("openmm      {}".format(
        "{}{} (openmm {})".format(
            exe, "  <- this interpreter" if exe == sys.executable else "",
            omm_version) if exe else "not available"))

    step_1_closed_form(atoms, calc)
    step_2_force_agreement(atoms, exe)
    tags = step_3_two_routes(exe)
    step_4_compare(tags)
    step_5_superposition(tags, exe)

    rule("What this did NOT show")
    print("A usable free energy. {} ps is a smoke length; the production length is set by"
          .format(PROD_PS))
    print("the saturation curve, and on the first real 25 ps run acceptance criterion 1")
    print("FAILED at +0.4684 kcal/mol against a 0.3 budget. Andricioaei and Karplus needed")
    print("about 4 ns where 200 ps was not converged.")
    print()
    print("Next: docs/tutorials/T01b_openQHA_Practice_BranchB_QHA.ipynb, then")
    print("      docs/branchB_workflow.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
