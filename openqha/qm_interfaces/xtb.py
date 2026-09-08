"""GFN2-xTB energies, forces and analytic Hessians, through the `xtb` executable.

WHY THIS EXISTS
---------------
Branch C compares three levels on the same geometries: `MACE-OFF23_medium` (the
production potential), `GFN2-xTB` (the CREST workhorse) and `RI-MP2/RIJK/cc-pVTZ` (the
reference). Two of the three already had a Hessian route in this package; GFN2 did not,
even though `xtb` has been installed all along and its analytic Hessian costs 0.015 s on
a ten-atom molecule.

Its value in the comparison is not accuracy -- it is that GFN2 is the level branch A
uses as its workhorse. If GFN2's curvature is far from the reference, that is a fact
about the geometries branch A hands downstream, not only about GFN2.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
No thermochemistry. `xtb` prints its own G and H, and this module ignores them, exactly
as `s0_branch2_opt_freq.py` ignores ORCA's: the symmetry number and the electronic
degeneracy must be declared, and a program that guesses them silently is a source of the
0.41 kcal/mol error this repository has already been bitten by once. Frequencies come
out of here; free energies are assembled by `openqha.thermochem.thermo`.

UNITS
-----
`xtb` writes the Turbomole `$hessian` block in **Hartree/Bohr^2**, the same unit ORCA
uses in its `.hess`, so the one conversion in `orca.hessian_to_ev_per_angstrom2` is
reused rather than written again. `verify_frequencies` asserts the parse against xtb's
own `vibspectrum`, which catches wrong units, wrong masses and a broken parse at once.
"""
import os
import sys
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from . import orca

#: Environment variable takes precedence, then whatever is on PATH.
DEFAULT_BIN = "xtb"


def xtb_binary():
    """The `xtb` executable.

    Looked up in three places, in order: `S0_XTB_BIN`, PATH, and **the directory of the
    running interpreter**. The third exists because calling the environment's python by
    absolute path -- which is how every driver here is launched from a non-login shell
    -- leaves that environment's `bin` off PATH, and the failure then reads as "xtb is
    not installed" when it is installed right next to the interpreter running the code.
    `gmx_io.gmx_binary` searches sibling environments for the same reason.
    """
    p = os.environ.get("S0_XTB_BIN", DEFAULT_BIN)
    if os.path.isabs(p) and os.path.exists(p):
        return p
    found = shutil.which(p)
    if found:
        return found
    beside = Path(sys.executable).parent / p
    if beside.exists():
        return str(beside)
    raise FileNotFoundError(
        "xtb not found as {!r}: not on PATH and not beside the running interpreter "
        "({}).\nSet S0_XTB_BIN to the executable, or activate the environment that "
        "carries it.".format(p, Path(sys.executable).parent))


def version():
    """The version string xtb reports, for the provenance record."""
    out = subprocess.run([xtb_binary(), "--version"], capture_output=True, text=True)
    for line in (out.stdout + out.stderr).splitlines():
        if "version" in line.lower():
            return line.strip()
    return "unknown"


def _write_xyz(path, symbols, positions):
    lines = ["{}".format(len(symbols)), ""]
    for s, r in zip(symbols, np.asarray(positions, dtype=float)):
        lines.append("{:<3s} {:18.10f} {:18.10f} {:18.10f}".format(s, *r))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_xyz(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    n = int(lines[0].split()[0])
    symbols, pos = [], []
    for line in lines[2:2 + n]:
        parts = line.split()
        symbols.append(parts[0])
        pos.append([float(x) for x in parts[1:4]])
    return symbols, np.array(pos)


def parse_turbomole_hessian(path, n_atoms):
    """Read a Turbomole `$hessian` block: 3N x 3N, row-major, **Hartree/Bohr^2**.

    No unit conversion happens here. Conversion is the caller's explicit act, so a
    wrong assumption cannot hide inside a reader -- the same rule `orca.parse_hess`
    follows.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    start = None
    for i, line in enumerate(text):
        if line.strip().startswith("$hessian"):
            start = i + 1
            break
    if start is None:
        raise KeyError("no $hessian block in {}".format(path))
    vals = []
    need = (3 * n_atoms) ** 2
    for line in text[start:]:
        s = line.strip()
        if s.startswith("$"):
            break
        for tok in s.split():
            try:
                vals.append(float(tok))
            except ValueError:
                pass                       # Turbomole row labels, if present
        if len(vals) >= need:
            break
    if len(vals) != need:
        raise ValueError(
            "read {} numbers from the $hessian block of {}, expected {} for {} atoms"
            .format(len(vals), path, need, n_atoms))
    h = np.array(vals, dtype=float).reshape(3 * n_atoms, 3 * n_atoms)
    return 0.5 * (h + h.T)


def parse_vibspectrum(path):
    """xtb's own wavenumbers, cm^-1, in the order it printed them.

    The six (or five) rigid entries xtb writes as `-0.00` / `0.00` are kept: dropping
    them here would hide a case where the rigid modes did not separate cleanly.
    """
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        # "<mode> [symmetry] <wavenumber> <intensity> ..." -- the symmetry column is
        # absent for the rigid entries, so the wavenumber is located by position from
        # the end of the numeric fields rather than by a fixed column index.
        nums = []
        for tok in parts[1:]:
            try:
                nums.append(float(tok))
            except ValueError:
                nums.append(None)
        numeric = [x for x in nums if x is not None]
        if numeric:
            out.append(numeric[0])
    return np.array(out, dtype=float)


def optimise_and_hessian(symbols, positions, gfn=2, charge=0, uhf=0, workdir=None,
                         keep=False, accuracy=0.2, nprocs=1, opt_level="vtight"):
    """`xtb --ohess`: relax to this level's own minimum, then its analytic Hessian.

    **The geometry is re-optimised deliberately.** A Hessian is only a Hessian at a
    stationary point OF THE SAME SURFACE; evaluating GFN2's second derivatives at
    MACE's minimum measures the displacement between the two minima as much as the
    curvature, and the two cannot be separated afterwards. The displacement is
    reported (`max_displacement_A`) so the size of that difference is visible.

    Returns a dict with the relaxed geometry, the Hessian in **eV/A^2**, xtb's own
    wavenumbers, and enough provenance to reproduce the call.
    """
    tmp = workdir or tempfile.mkdtemp(prefix="openqha_xtb_")
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    _write_xyz(tmp / "in.xyz", symbols, positions)

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(int(nprocs))
    env["MKL_NUM_THREADS"] = str(int(nprocs))
    # The optimisation level is `vtight` on purpose. A loosely converged ANC optimisation
    # leaves small imaginary modes behind, and an imaginary mode is the one thing that
    # makes this whole record unusable -- `thermo.vibrational` refuses it outright. A
    # residual that is really a convergence artefact must not be reported as chemistry.
    cmd = [xtb_binary(), "in.xyz", "--ohess", str(opt_level), "--gfn", str(int(gfn)),
           "--chrg", str(int(charge)), "--uhf", str(int(uhf)),
           "--acc", str(float(accuracy))]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(tmp), capture_output=True, text=True)
    wall = time.time() - t0
    (tmp / "xtb.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            "xtb exited {} in {}. Last lines:\n{}".format(
                proc.returncode, tmp, "\n".join((proc.stdout + proc.stderr)
                                                .splitlines()[-25:])))

    sym_opt, pos_opt = _read_xyz(tmp / "xtbopt.xyz")
    if sym_opt != list(symbols):
        raise ValueError("xtb reordered the atoms: {} -> {}".format(symbols, sym_opt))
    h_eh_bohr2 = parse_turbomole_hessian(tmp / "hessian", len(symbols))
    rec = dict(
        method="GFN{}-xTB".format(int(gfn)),
        binary=xtb_binary(),
        version=version(),
        command=" ".join(cmd),
        workdir=str(tmp),
        wall_seconds=float(wall),
        symbols=list(symbols),
        positions_A=pos_opt.tolist(),
        max_displacement_A=float(np.abs(pos_opt - np.asarray(positions, float)).max()),
        hessian_eV_A2=orca.hessian_to_ev_per_angstrom2(h_eh_bohr2),
        xtb_frequencies_cm_inv=parse_vibspectrum(tmp / "vibspectrum"),
        accuracy=float(accuracy),
        charge=int(charge),
        uhf=int(uhf),
    )
    if not keep and workdir is None:
        shutil.rmtree(tmp, ignore_errors=True)
        rec["workdir"] = None
    return rec


#: xtb writes the rigid entries of `vibspectrum` as exactly `-0.00` / `0.00`, so they
#: are identified by being at zero rather than by being the smallest. That distinction
#: matters: a molecule with an imaginary mode has an entry BELOW the rigid ones, and
#: "the six smallest" would then discard the imaginary mode and keep a rigid zero.
RIGID_ENTRY_TOLERANCE_CM = 1.0e-2


def verify_frequencies(rec, masses_amu, tol_cm_inv=1.0, expect_rigid=6):
    """Assert that our parse reproduces xtb's own wavenumbers.

    Catches wrong units, wrong masses, wrong ordering and a broken parse at once, at no
    computational cost -- an assertion that runs on every sample rather than a check
    somebody did once.

    Both sides are compared as the 3N-6 VIBRATIONAL wavenumbers, signed and sorted: our
    side Eckart-projected, xtb's side with its zero entries removed by value. An earlier
    version took "the largest 3N-6" from an unprojected spectrum, which on a structure
    carrying an imaginary mode dropped that mode and kept a rigid zero in its place, and
    then reported a 62.8 cm^-1 disagreement that was entirely its own doing.
    """
    from ..quasi_harmonic import mode_match                # local: avoids a cycle
    ours = mode_match.projected_modes(rec["hessian_eV_A2"], masses_amu,
                                      rec["positions_A"], "hessian")[0]
    theirs = np.asarray(rec["xtb_frequencies_cm_inv"], dtype=float)
    rigid = np.abs(theirs) < RIGID_ENTRY_TOLERANCE_CM
    if int(rigid.sum()) != int(expect_rigid):
        raise ValueError(
            "xtb's vibspectrum has {} entries at zero, expected {} rigid modes. The "
            "spectrum is {}".format(int(rigid.sum()), expect_rigid,
                                    np.sort(theirs)[:8].round(2).tolist()))
    a = np.sort(np.asarray(ours, dtype=float))
    b = np.sort(theirs[~rigid])
    if len(b) != len(a):
        raise ValueError(
            "xtb printed {} vibrational wavenumbers, our projection gives {}".format(
                len(b), len(a)))
    dev = np.abs(a - b)
    if dev.max() > tol_cm_inv:
        raise ValueError(
            "our parse of the xtb Hessian does not reproduce xtb's own wavenumbers: "
            "max deviation {:.3f} cm^-1 > {:.3f}. Ours {} vs theirs {}".format(
                dev.max(), tol_cm_inv, a[:4].round(2).tolist(), b[:4].round(2).tolist()))
    return dict(max_deviation_cm_inv=float(dev.max()),
                rms_deviation_cm_inv=float(np.sqrt((dev ** 2).mean())),
                n_compared=int(len(a)),
                n_imaginary=int((b < 0).sum()))
