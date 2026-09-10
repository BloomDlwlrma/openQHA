"""Does a MACE committee's disagreement predict its error against RI-MP2?

CALIBRATION. Branch C, step 1 -- the gate that decides whether the ALF active-learning
scheme can work at all (S0-C-10, S0-C-13). Nothing here enters a deliverable.

Why this runs before anything else
----------------------------------
ALF selects configurations for expensive labelling by committee disagreement. That
only works if disagreement actually tracks error. The premise is NOT free: ALF assumes
independently trained members, and any committee we build by fine-tuning one base model
on one dataset will have correlated errors, so its spread will UNDERSTATE the true
error and active learning will skip exactly the blind spots it is meant to find.

This script tests the premise using published MACE-OFF models -- no training, no new
quantum chemistry -- against the RI-MP2/RIJK/cc-pVTZ (E, F, H) triples already sitting
on disk from the branch-2 production runs (S0-C-5).

WHAT A PASS AND A FAIL MEAN -- these are not symmetric
-----------------------------------------------------
The committee here is five *differently sized, independently trained* published models.
That is a MORE diverse committee than a set of fine-tunes from a common base will ever
be. So:

  * FAIL here is decisive. If even this committee cannot predict its own error, a
    correlated committee of fine-tunes certainly cannot, and the ALF scheme does not
    stand as designed.
  * PASS here is necessary, not sufficient. It does not license the fine-tuned
    committee -- that has to be re-measured once it exists.

Say this in any report of the result. A necessary condition reported as a sufficient
one is how a criterion stops being a criterion.

What is compared, and why energies are handled differently
----------------------------------------------------------
Forces and Hessians are derivatives, so MACE and RI-MP2 values are directly comparable.
Total energies are not: every model carries its own atomic reference, so absolute
energies sit on unrelated scales. Energies are therefore compared as RELATIVE energies
within one molecule, across its basins -- which is also the physically meaningful
quantity (D0-P2-11 measured exactly this) and needs a molecule with >= 2 basins.

The curvature probe
-------------------
Curvature disagreement is measured as the committee spread of the Hessian-vector
product H@v, using the corrected probe v = M^-1/2 P v0 (plan_C section 3, corrections 1
and 2): mass-weighted and projected out of the rigid-body subspace. With an isotropic
probe most of what you measure is rigid-body and mass-weighting difference rather than
curvature disagreement -- for an 8-atom molecule 6 of 24 directions are pure
translation and rotation.

Usage
-----
    python scripts/calibration/s0_C_committee_calibration.py
    python scripts/calibration/s0_C_committee_calibration.py --n-probes 8 --limit 5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

def _repo_root():
    """Directory holding the openQHA package, found by walking up from this file.

    Depth-independent on purpose: this file keeps working wherever under the
    repository it is moved to. The earlier move into `scripts/_superseded/` broke
    every `parents[1]` in the moved files silently, which is what this replaces.
    """
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))
from openqha import S0_ROOT, engine, hessian as s0hess, orca  # noqa: E402

from ase.io import read as ase_read  # noqa: E402

EV_PER_HARTREE = 27.211386245988
BOHR_PER_ANGSTROM = 1.0 / 0.529177210903
HARTREE_KCAL = 627.5094740631

#: The committee. Five published MACE-OFF models, all shipped in the repo.
#:
#: These are independently trained and of differing size, so their disagreement is a
#: GENEROUS proxy for what a fine-tuned committee would produce. See the module
#: docstring on why that asymmetry matters.
#:
#: Resolved through `openqha.engine`, NOT by writing the location down here again. The
#: reference tree moved out of this repository on 2026-09-04 (the weights are ASL and
#: openQHA is meant to be publishable), and every literal path to it broke at once. The
#: engine registry searches for it and verifies each file's SHA-256, which is also what
#: makes a silent potential swap impossible -- a second copy of the path here would have
#: neither property.
COMMITTEE_NAMES = ["MACE-OFF23_small", "MACE-OFF23_medium", "MACE-OFF23_large",
                   "MACE-OFF23b_medium", "MACE-OFF24_medium"]
COMMITTEE = [(n, engine.model_path(n)) for n in COMMITTEE_NAMES]

PROBE_SEED = 20260903


# ---------------------------------------------------------------------------------------
# Harvest the RI-MP2 reference already on disk
# ---------------------------------------------------------------------------------------
def harvest_reference(runs_root, limit=None):
    """Read (geometry, E, F, H) from every completed branch-2 ORCA work directory.

    Zero recomputation: the branch-2 driver wrote optfreq.hess and optfreq.engrad and
    then parsed only the frequencies out of the .out file (S0-C-5).
    """
    runs_root = Path(runs_root)
    records = []
    for d in sorted(runs_root.glob("*")):
        hess, engrad, xyz = d / "optfreq.hess", d / "optfreq.engrad", d / "optfreq.xyz"
        if not (hess.exists() and engrad.exists() and xyz.exists()):
            continue
        try:
            parsed = orca.parse_hess(hess)
            check = orca.verify_hess_frequencies(parsed)       # criterion: runs every time
        except (KeyError, ValueError, IndexError) as exc:
            print("  skipped {}: {}".format(d.name, exc))
            continue

        atoms = ase_read(str(xyz))
        text = engrad.read_text(encoding="utf-8", errors="replace").split("\n")
        nums = [float(t) for t in text if t.strip() and _is_float(t.strip())]
        natoms = parsed["n_atoms"]
        # .engrad layout: natoms, energy, then 3N gradient components (Eh/Bohr).
        energy_eh = nums[1]
        grad = np.array(nums[2:2 + 3 * natoms]).reshape(natoms, 3)

        name = d.name
        molecule = name.rsplit("_b", 1)[0]
        records.append({
            "workdir": str(d),
            "name": name,
            "molecule": molecule,
            "n_atoms": natoms,
            "symbols": atoms.get_chemical_symbols(),
            "positions_A": atoms.get_positions(),
            "masses_amu": parsed["masses_amu"],
            # forces = -gradient; Eh/Bohr -> eV/A
            "forces_ev_A": -grad * EV_PER_HARTREE * BOHR_PER_ANGSTROM,
            "energy_eV": energy_eh * EV_PER_HARTREE,
            "hessian_ev_A2": orca.hessian_to_ev_per_angstrom2(parsed["hessian_eh_bohr2"]),
            "hess_check": check,
        })
        if limit and len(records) >= limit:
            break
    return records


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------------------
# The corrected probe
# ---------------------------------------------------------------------------------------
def internal_probes(masses, positions, n_probes, rng):
    """Probes v = M^-1/2 P v0, normalised, in the internal (non-rigid) subspace.

    P removes the 6 rigid-body directions. Without it a fraction 6/3N of every probe --
    25% for an 8-atom molecule -- is spent on directions where the Hessian is
    identically zero at a stationary point, and where at a non-stationary point it
    encodes the reference's residual gradient rather than curvature.
    """
    m3 = np.repeat(np.asarray(masses, dtype=float), 3)
    v_rigid, _sing, _rank = s0hess.rigid_body_vectors(masses, positions)
    P = np.eye(len(m3)) - v_rigid @ v_rigid.T

    probes = []
    for _ in range(n_probes):
        v0 = rng.normal(size=len(m3))
        v = P @ v0                       # project into the internal subspace
        v = v / np.sqrt(m3)              # mass weighting (correction 2)
        nrm = np.linalg.norm(v)
        probes.append(v / nrm if nrm > 0 else v)
    return np.array(probes)


# ---------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------
def spearman(x, y):
    """Spearman rank correlation without pulling in scipy."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 3:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def _rank(a):
    order = np.argsort(a)
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(len(a), dtype=float)
    return r


def coverage(errors, stds, ks=(1.0, 2.0, 3.0)):
    """Fraction of structures whose true error falls inside k * committee std.

    A well-calibrated committee gives numbers that grow with k and are not ~0. Near-zero
    coverage at k=3 is the signature of the correlated-committee failure: the spread is
    small everywhere while the error is not.
    """
    errors, stds = np.asarray(errors, dtype=float), np.asarray(stds, dtype=float)
    out = {}
    for k in ks:
        with np.errstate(invalid="ignore"):
            out["k={:g}".format(k)] = float(np.mean(errors <= k * stds))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default="/home/ubuntu/runs/branch2_prod")
    ap.add_argument("--n-probes", type=int, default=8,
                    help="HVP probes per structure; variance of the estimator falls as 1/n")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("=" * 100)
    print("Branch C step 1 -- committee calibration gate")
    print("=" * 100)
    print("A FAIL here is decisive; a PASS is necessary but not sufficient.")
    print("(This committee is more diverse than a fine-tuned one will be -- see docstring.)")
    print()

    print("harvesting RI-MP2 reference from {} ...".format(args.runs_root))
    refs = harvest_reference(args.runs_root, limit=args.limit)
    if not refs:
        raise SystemExit("no completed ORCA work directories found -- nothing to calibrate against")
    worst_check = max(r["hess_check"]["max_deviation_cm_inv"] for r in refs)
    print("  {} structures, {} molecules".format(
        len(refs), len({r["molecule"] for r in refs})))
    print("  hess unit round-trip: worst deviation {:.4f} cm^-1 (criterion < 0.5)".format(
        worst_check))
    print()

    from mace.calculators import MACECalculator      # noqa: E402  (slow import)

    print("loading committee ...")
    members = []
    for name, path in COMMITTEE:
        if not path.exists():
            print("  MISSING {} -> {}".format(name, path))
            continue
        t0 = time.time()
        calc = MACECalculator(model_paths=str(path), device=args.device,
                              default_dtype="float64")
        members.append((name, calc))
        print("  {:22s} {:>7.1f} MB  loaded in {:5.1f} s".format(
            name, path.stat().st_size / 1e6, time.time() - t0))
    if len(members) < 2:
        raise SystemExit("need at least 2 committee members")
    print()

    rng = np.random.default_rng(PROBE_SEED)
    rows = []
    for i, ref in enumerate(refs):
        from ase import Atoms
        atoms = Atoms(symbols=ref["symbols"], positions=ref["positions_A"])
        probes = internal_probes(ref["masses_amu"], ref["positions_A"], args.n_probes, rng)

        energies, forces, hvps = [], [], []
        for _name, calc in members:
            a = atoms.copy()
            a.calc = calc
            energies.append(a.get_potential_energy())
            forces.append(a.get_forces())
            H = calc.get_hessian(atoms=a).reshape(3 * len(a), 3 * len(a))
            hvps.append(probes @ H)                       # (n_probes, 3N)

        energies = np.array(energies)
        forces = np.array(forces)
        hvps = np.array(hvps)

        ref_hvp = probes @ ref["hessian_ev_A2"]

        rows.append({
            "name": ref["name"],
            "molecule": ref["molecule"],
            "n_atoms": ref["n_atoms"],
            # committee spread
            "force_std": float(np.mean(np.std(forces, axis=0))),
            "force_std_max": float(np.max(np.std(forces, axis=0))),
            "hvp_std": float(np.mean(np.std(hvps, axis=0))),
            # true error of the committee MEAN against RI-MP2
            "force_err": float(np.sqrt(np.mean(
                (forces.mean(axis=0) - ref["forces_ev_A"]) ** 2))),
            "hvp_err": float(np.sqrt(np.mean((hvps.mean(axis=0) - ref_hvp) ** 2))),
            "ref_force_rms": float(np.sqrt(np.mean(ref["forces_ev_A"] ** 2))),
            "ref_hvp_rms": float(np.sqrt(np.mean(ref_hvp ** 2))),
            "member_energies_eV": energies.tolist(),
            "ref_energy_eV": ref["energy_eV"],
        })
        print("  [{:2d}/{:2d}] {:24s} F std {:7.4f} err {:7.4f} | HVP std {:8.4f} err {:8.4f}".format(
            i + 1, len(refs), ref["name"], rows[-1]["force_std"], rows[-1]["force_err"],
            rows[-1]["hvp_std"], rows[-1]["hvp_err"]))

    # --- relative energies, per molecule (removes each model's own atomic reference) ---
    energy_rows = []
    by_mol = {}
    for r in rows:
        by_mol.setdefault(r["molecule"], []).append(r)
    for mol, group in by_mol.items():
        if len(group) < 2:
            continue                       # a single basin carries no relative energy
        member_E = np.array([g["member_energies_eV"] for g in group])   # (n_basin, n_model)
        ref_E = np.array([g["ref_energy_eV"] for g in group])
        member_rel = member_E - member_E.mean(axis=0, keepdims=True)
        ref_rel = ref_E - ref_E.mean()
        for j, g in enumerate(group):
            energy_rows.append({
                "name": g["name"],
                "molecule": mol,
                "energy_std_kcal": float(np.std(member_rel[j]) * 23.060547830618307),
                "energy_err_kcal": float(abs(member_rel[j].mean() - ref_rel[j])
                                         * 23.060547830618307),
            })

    def block(label, stds, errs):
        rho = spearman(stds, errs)
        cov = coverage(errs, stds)
        med_s, med_e = float(np.median(stds)), float(np.median(errs))
        # Dispersion ratio: a calibrated committee has median error ~ median spread,
        # i.e. a ratio near 1. A large ratio means the committee is confidently wrong
        # -- which for active learning is worse than being uncertain, because nothing
        # gets flagged for labelling.
        #
        # Part of this ratio is IRREDUCIBLE and no committee can fix it: MACE-OFF is
        # trained toward wB97M-D3BJ/def2-TZVPPD, while our reference is
        # RI-MP2/cc-pVTZ. Every member agrees on the same systematically shifted
        # answer, so that component of the error produces no disagreement at all.
        # Committee spread can only see EPISTEMIC uncertainty (unseen regions), never
        # the offset between the model's training level and our labelling level.
        ratio = (med_e / med_s) if med_s > 0 else float("inf")
        print()
        print("  {}".format(label))
        print("    n                     {}".format(len(stds)))
        print("    Spearman rho          {:+.3f}   (criterion >= 0.5)".format(rho))
        print("    coverage              {}".format(
            "  ".join("{} {:.0%}".format(k, v) for k, v in cov.items())))
        print("    median committee std  {:.5f}".format(med_s))
        print("    median true error     {:.5f}".format(med_e))
        print("    dispersion ratio      {:.1f}x   (calibrated ~ 1; large = confidently wrong)"
              .format(ratio))
        return {"n": len(stds), "spearman": rho, "coverage": cov,
                "median_std": med_s, "median_err": med_e,
                "dispersion_ratio": ratio}

    print()
    print("=" * 100)
    print("RESULT")
    print("=" * 100)
    res = {}
    res["forces"] = block("FORCES  (eV/A)",
                          [r["force_std"] for r in rows], [r["force_err"] for r in rows])
    res["hvp"] = block("CURVATURE, H@v with the corrected probe (eV/A^2)",
                       [r["hvp_std"] for r in rows], [r["hvp_err"] for r in rows])
    if energy_rows:
        res["relative_energy"] = block(
            "RELATIVE ENERGY within a molecule (kcal/mol)",
            [r["energy_std_kcal"] for r in energy_rows],
            [r["energy_err_kcal"] for r in energy_rows])

    passed = {k: (v["spearman"] >= 0.5) for k, v in res.items()}
    print()
    for k, ok in passed.items():
        print("  {:18s} {}".format(k, "PASS" if ok else "**FAIL** (rho < 0.5)"))
    print()
    print("  Reminder: a PASS is NECESSARY, NOT SUFFICIENT. The fine-tuned committee")
    print("  will be more correlated than this one and must be re-measured.")

    out = Path(args.out) if args.out else (S0_ROOT / "analysis" / "committee_calibration.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_by": "scripts/calibration/s0_C_committee_calibration.py",
        "committee": [n for n, _ in members],
        "n_probes": args.n_probes,
        "probe": "v = M^-1/2 P v0, normalised (plan_C section 3, corrections 1 and 2)",
        "probe_seed": PROBE_SEED,
        "reference_level": "RI-MP2/cc-pVTZ + cc-pVTZ/C + cc-pVTZ/JK RIJK, TightOpt NumFreq TightSCF",
        "reference_provenance_status": "test (local ORCA 6.0.1; D0-75 reserves production for deimos 6.1.1)",
        "worst_hess_roundtrip_cm_inv": worst_check,
        "results": res,
        "passed": passed,
        "interpretation": (
            "FAIL is decisive: a committee of five independently trained, differently "
            "sized published models is MORE diverse than any set of fine-tunes from one "
            "base, so if this cannot predict its own error, a fine-tuned committee "
            "cannot either. PASS is necessary but NOT sufficient and does not license "
            "the fine-tuned committee."),
        "rows": rows,
        "energy_rows": energy_rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("written {}".format(out))


if __name__ == "__main__":
    main()
