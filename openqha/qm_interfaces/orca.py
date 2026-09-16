"""Call ORCA for **energies and forces**, and the composite scheme assembled from
coefficients.

**Scope (two rulings, the second revising the first)**:

* 2026-08-29 (morning): the composite scheme is used for **energies and forces only, not
  for frequencies** -- because validating a composite frequency would mean actually
  running the 61 gradients of `CCSD(T)/cc-pVTZ` on some species (about 5.1 days), and the
  0.073 eV/A quoted in the original paper is the error of a **force**; **accuracy in the
  forces does not automatically mean accuracy in the frequencies**.
* 2026-08-29 (afternoon, on the user's instruction "change it so that it can produce
  something"): **the capability is restored and off by default**. `composite_hessian()`
  gives a composite Hessian by central differences of the composite forces. **It does not
  run by default**; a caller must ask for it explicitly, and the function first shows the
  cost of the 6N x (number of terms) single points. **The ruling that the cost was too
  high was about treating frequencies as a production reference quantity, not about
  forbidding the capability to exist** -- the capability existing, being off by default,
  and reporting its cost first are not in conflict.

The composite expression (Allen et al., equations 1 and 2) is a linear combination by
coefficient:

The recipes live in `package2.composite_recipes` in
`configs/openqha.yaml`, where each term is `[coefficient, method, basis set]`.

**Version and where it runs** (`D0-75`): production calculations use **ORCA 6.1.1 and run
only on deimos**; the local WSL 6.0.1 is **for testing only** and produces no production
numbers. This module writes the ORCA version it used verbatim into every product, so the
identity can be checked.
"""
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

# CODATA 2018, the same set as openqha.thermo
EV_PER_HARTREE = 27.211386245988
BOHR_PER_ANGSTROM = 1.0 / 0.529177210903

DEFAULT_BIN = "/home/ubuntu/packages/orca_6_0_1/orca"   # local testing; production runs
                                                        # on deimos


def orca_binary():
    """The ORCA executable. The environment variable S0_ORCA_BIN takes precedence."""
    p = os.environ.get("S0_ORCA_BIN", DEFAULT_BIN)
    if not Path(p).exists():
        raise FileNotFoundError(
            "ORCA not found: {}\nSet S0_ORCA_BIN to the executable.".format(p))
    return p


def _write_input(path, symbols, positions, method, basis, nprocs, maxcore,
                 extra="", charge=0, mult=1):
    lines = ["! {} {} EnGrad {}".format(method, basis, extra).rstrip(),
             "%maxcore {}".format(int(maxcore)),
             "%pal nprocs {} end".format(int(nprocs)),
             "* xyz {} {}".format(int(charge), int(mult))]
    for s, r in zip(symbols, positions):
        lines.append("{:2s} {:18.10f} {:18.10f} {:18.10f}".format(s, *r))
    lines.append("*")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_engrad(path, natoms):
    """Read ORCA's `.engrad`: energy in Eh, gradient in Eh/a0 (3N lines). A leading `#`
    is a comment."""
    vals = []
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        vals.append(t)
    if len(vals) < 2 + 3 * natoms:
        raise ValueError("{} has too little content: expected {} numeric lines, got "
                         "{}".format(path, 2 + 3 * natoms, len(vals)))
    n = int(vals[0])
    if n != natoms:
        raise ValueError("{} declares {} atoms, expected {}".format(path, n, natoms))
    energy_eh = float(vals[1])
    grad = np.array([float(v) for v in vals[2:2 + 3 * natoms]]).reshape(natoms, 3)
    return energy_eh, grad


def single_point(symbols, positions, method, basis, workdir=None, nprocs=8,
                 maxcore=3500, extra="", charge=0, mult=1, keep=False,
                 timeout_s=None):
    """One ORCA analytic gradient. Returns a dict: energy in eV, forces in eV/A, version,
    elapsed time.

    **Force = -gradient**, converted from Eh/a0 to eV/A, on the same footing as the ASE
    convention used by `openqha.engine`.
    """
    symbols = list(symbols)
    positions = np.asarray(positions, dtype=float)
    nat = len(symbols)
    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="s0orca_"))
    tmp.mkdir(parents=True, exist_ok=True)
    stem = tmp / "job"
    _write_input(str(stem) + ".inp", symbols, positions, method, basis,
                 nprocs, maxcore, extra, charge, mult)
    t0 = time.time()
    with open(str(stem) + ".out", "w") as fh:
        rc = subprocess.call([orca_binary(), str(stem) + ".inp"],
                             stdout=fh, stderr=subprocess.STDOUT, cwd=str(tmp),
                             timeout=timeout_s)
    seconds = time.time() - t0
    out = Path(str(stem) + ".out").read_text(encoding="utf-8", errors="replace")
    if "****ORCA TERMINATED NORMALLY****" not in out:
        tail = "\n".join(out.split("\n")[-25:])
        raise RuntimeError("ORCA did not finish normally ({} {}), exit code {}. Tail of "
                           "the output:\n{}".format(method, basis, rc, tail))
    energy_eh, grad_eh_bohr = _parse_engrad(str(stem) + ".engrad", nat)
    m = re.search(r"Program Version\s+(\S+)", out)
    version = m.group(1) if m else "unknown"
    nb = re.search(r"Number of basis functions\s+\.+\s+(\d+)", out)
    rec = dict(method=method, basis=basis,
               energy_eV=float(energy_eh * EV_PER_HARTREE),
               forces_eV_A=(-grad_eh_bohr * EV_PER_HARTREE * BOHR_PER_ANGSTROM),
               orca_version=version,
               n_basis_functions=int(nb.group(1)) if nb else None,
               seconds=seconds, nprocs=int(nprocs))
    if not keep and workdir is None:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        rec["workdir"] = str(tmp)
    return rec


#: The reference level (ADR 0004): the level MACE-OFF23 was trained to. `Freq` asks
#: for the analytic Hessian; ORCA 6.0.1 accepts it for this meta-GGA range-separated
#: hybrid (propanal basin 0, 8 cores: 226 s, "SCF Response" module, 0 imaginary modes,
#: measured 2026-09-16). `NumFreq` is the declared fallback if a build refuses.
REFERENCE_KEYWORDS = "wB97M-D3BJ def2-TZVPPD TightOpt Freq TightSCF"


def final_energy_from_out(out_text):
    """The last `FINAL SINGLE POINT ENERGY` of an ORCA output, Eh (dispersion included).
    The `.hess` file's `$act_energy` is a placeholder (0.0 in ORCA 6.0.1) and is not used."""
    vals = re.findall(r"FINAL SINGLE POINT ENERGY\s+([-+]?\d+\.\d+)", out_text)
    if not vals:
        raise ValueError("no FINAL SINGLE POINT ENERGY in the ORCA output")
    return float(vals[-1])


def hessian_route(out_text):
    """'analytic' when ORCA ran the analytic Hessian (the SCF Response module), 'numerical'
    when it fell back to or was asked for NumFreq, else 'unknown'."""
    if "NUMERICAL FREQUENCIES" in out_text or "Numerical frequency" in out_text:
        return "numerical"
    if "SCF Response" in out_text or "ANALYTICAL FREQUENCIES" in out_text:
        return "analytic"
    return "unknown"


def optimise_and_hessian(symbols, positions, workdir, keywords=REFERENCE_KEYWORDS,
                         nprocs=8, maxcore=3000, charge=0, mult=1, stem="job",
                         timeout_s=None):
    """Geometry optimisation plus Hessian at one level, in `workdir` (an engine folder).

    Skips ORCA when `workdir/<stem>.hess` exists and the `.out` terminated normally, so a
    Batch can be resumed. Returns the relaxed geometry (A), energy (Eh, the last FINAL
    SINGLE POINT ENERGY of the `.out`), the parsed Hessian record (`parse_hess`), whether the
    Hessian was analytic or numerical, the wall time, and ORCA's version.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    inp = workdir / (stem + ".inp")
    out = workdir / (stem + ".out")
    hess = workdir / (stem + ".hess")
    done = hess.is_file() and out.is_file() and \
        "****ORCA TERMINATED NORMALLY****" in out.read_text(encoding="utf-8", errors="replace")
    seconds = None
    if not done:
        lines = ["! {}".format(keywords), "%pal nprocs {} end".format(int(nprocs)),
                 "%maxcore {}".format(int(maxcore)), "* xyz {} {}".format(int(charge), int(mult))]
        for s, r in zip(symbols, positions):
            lines.append("{:2s} {:18.10f} {:18.10f} {:18.10f}".format(s, *r))
        lines.append("*")
        inp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        t0 = time.time()
        with open(out, "w") as fh:
            rc = subprocess.call([orca_binary(), str(inp)], stdout=fh, stderr=subprocess.STDOUT,
                                 cwd=str(workdir), timeout=timeout_s)
        seconds = time.time() - t0
        text = out.read_text(encoding="utf-8", errors="replace")
        if "****ORCA TERMINATED NORMALLY****" not in text:
            raise RuntimeError("ORCA did not finish normally in {} (rc {}). Tail:\n{}".format(
                workdir, rc, "\n".join(text.split("\n")[-25:])))
    text = out.read_text(encoding="utf-8", errors="replace")
    parsed = parse_hess(hess)
    if parsed["symbols"] != list(symbols):
        raise ValueError("ORCA reordered the atoms in {}: {} -> {}".format(
            workdir, list(symbols), parsed["symbols"]))
    m = re.search(r"Program Version\s+(\S+)", text)
    return dict(keywords=keywords, workdir=str(workdir), stem=stem,
                positions_A=(np.asarray(parsed["positions_bohr"]) / BOHR_PER_ANGSTROM).tolist(),
                energy_eh=final_energy_from_out(text), hess=parsed,
                hessian_route=hessian_route(text), seconds=seconds,
                orca_version=m.group(1) if m else "unknown", nprocs=int(nprocs),
                n_imaginary=int((np.asarray(parsed["frequencies_cm_inv"]) < -1.0).sum()))


def composite(symbols, positions, terms, nprocs=8, maxcore=3500, keep=False,
              timeout_s=None, progress=None, charge=0, mult=1):
    """Convenience entry point for a single recipe -- internally just "build the term pool
    and combine", with exactly the semantics of the older version."""
    pool = term_pool(symbols, positions, unique_terms([terms]), nprocs=nprocs,
                     maxcore=maxcore, timeout_s=timeout_s, progress=progress,
                     charge=charge, mult=mult)
    return combine(pool, terms)


def unique_terms(recipes):
    """Flatten the `[coefficient, method, basis set]` of several recipes into a
    **deduplicated list of (method, basis set)**.

    This is what the "term pool" is: the three-step basis ladder DZ / TZ* / QZ* shares
    CCSD(T)/DZ and MP2/DZ, and running them separately would recompute those. After
    deduplication **the intermediate steps are essentially free**.
    """
    seen, out = set(), []
    for terms in recipes:
        for _c, method, basis in terms:
            k = (method, basis)
            if k not in seen:
                seen.add(k)
                out.append(k)
    return out


def term_pool(symbols, positions, methods_bases, nprocs=8, maxcore=3500,
              timeout_s=None, progress=None, charge=0, mult=1):
    """Compute each (method, basis set) once on **the same geometry**, returning
    {(method, basis set): record}."""
    pool = {}
    for k, (method, basis) in enumerate(methods_bases):
        r = single_point(symbols, positions, method, basis, nprocs=nprocs,
                         maxcore=maxcore, timeout_s=timeout_s,
                         charge=charge, mult=mult)
        pool[(method, basis)] = r
        if progress is not None:
            progress(k + 1, len(methods_bases), method, basis, r["seconds"])
    return pool


def combine(pool, terms):
    """Linearly combine energies and forces out of the term pool according to
    `terms = [[coefficient, method, basis set], ...]`.

        E = Σ c_t E_t          F = Σ c_t F_t

    **Raises if the terms come from different ORCA versions** -- a composite is a
    combination of differences, and differences in default thresholds between versions do
    not cancel in it, which would make the whole correction meaningless (`D0-72`).
    """
    e_tot = 0.0
    f_tot = None
    per_term, versions = [], set()
    for coeff, method, basis in terms:
        r = pool.get((method, basis))
        if r is None:
            raise KeyError("the term pool has no {} / {} -- recipe and pool do not "
                           "match".format(method, basis))
        e_tot += float(coeff) * r["energy_eV"]
        f = float(coeff) * r["forces_eV_A"]
        f_tot = f if f_tot is None else f_tot + f
        versions.add(r["orca_version"])
        per_term.append(dict(coefficient=float(coeff), **{
            kk: vv for kk, vv in r.items() if kk != "forces_eV_A"}))
    if len(versions) != 1:
        raise RuntimeError(
            "the terms of the composite come from different ORCA versions {} -- "
            "differences in default thresholds between versions do not cancel in the "
            "difference, which makes the whole correction meaningless. Refusing to "
            "combine.".format(sorted(versions)))
    return dict(energy_eV=float(e_tot), forces_eV_A=f_tot,
                orca_version=versions.pop(), n_terms=len(terms),
                per_term=per_term,
                seconds=float(sum(t["seconds"] for t in per_term)))


# ---- composite Hessian: the capability exists, off by default, cost reported first ---
def hessian_cost(n_atoms, terms, seconds_per_term=None):
    """Show the cost first, then decide whether to run. Returns a dict (number of single
    points, estimated seconds, estimated days).

    A central difference needs `6N` displaced geometries, and every geometry needs **every
    term in the pool**. `seconds_per_term` is a measured {(method, basis set): seconds};
    when omitted, the estimates in the configuration are used.
    """
    methods_bases = unique_terms([terms])
    per_geom = sum((seconds_per_term or {}).get(k, 0.0) for k in methods_bases)
    n_geom = 6 * int(n_atoms)
    total = per_geom * n_geom
    return dict(n_atoms=int(n_atoms), n_displaced_geometries=n_geom,
                n_terms_per_geometry=len(methods_bases),
                n_single_points=n_geom * len(methods_bases),
                seconds_per_geometry=per_geom,
                estimated_seconds=total,
                estimated_hours=total / 3600.0,
                estimated_days=total / 86400.0)


def composite_hessian(symbols, positions, terms, delta_A=0.01, nprocs=8,
                      maxcore=3500, timeout_s=None, progress=None,
                      charge=0, mult=1):
    """A composite Hessian from central differences of the **composite forces**, in
    eV/A^2.

        H_ij = -( F_j(x + delta e_i) - F_j(x - delta e_i) ) / (2 delta)

    Same formula and same `delta` convention as
    `openqha.hessian.finite_difference_hessian`; only the source of the forces changes,
    from the neural-network potential to the composite reference.

    **This function is expensive**: `6N` geometries times every term in the pool. Look at
    `hessian_cost()` before running it. Returns (the symmetrised H, the asymmetry
    residual, a record per geometry). The asymmetry residual is a free diagnostic of the
    finite-difference error itself -- an exact Hessian is necessarily symmetric.
    """
    symbols = list(symbols)
    x0 = np.asarray(positions, dtype=float)
    n = len(symbols)
    mb = unique_terms([terms])
    h = np.zeros((3 * n, 3 * n))
    logs = []
    for i in range(3 * n):
        a, c = divmod(i, 3)
        for sign in (+1, -1):
            x = x0.copy()
            x[a, c] += sign * delta_A
            pool = term_pool(symbols, x, mb, nprocs=nprocs, maxcore=maxcore,
                             timeout_s=timeout_s, charge=charge, mult=mult)
            ref = combine(pool, terms)
            h[i] += -sign * ref["forces_eV_A"].reshape(-1) / (2.0 * delta_A)
            logs.append(dict(coordinate=i, sign=sign, seconds=ref["seconds"],
                             energy_eV=ref["energy_eV"]))
        if progress is not None:
            progress(i + 1, 3 * n)
    asym = float(np.abs(h - h.T).max())
    return 0.5 * (h + h.T), asym, logs


def force_metrics(f_ref, f_test):
    """Compare two sets of forces. Units eV/A, statistics per component (the same
    convention as Allen et al.)."""
    a = np.asarray(f_ref, dtype=float).ravel()
    b = np.asarray(f_test, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("the two force arrays have different shapes: {} vs {}".format(
            a.shape, b.shape))
    d = b - a
    return dict(n_components=int(a.size),
                mae_eV_A=float(np.abs(d).mean()),
                rmse_eV_A=float(np.sqrt((d ** 2).mean())),
                max_abs_eV_A=float(np.abs(d).max()),
                signed_mean_eV_A=float(d.mean()),
                ref_rms_eV_A=float(np.sqrt((a ** 2).mean())))


# =========================================================================================
# ORCA .hess parsing -- added 2026-09-03 for branch C (S0-C-4, S0-C-13).
#
# UNITS ARE MEASURED, NOT ASSUMED. On dsgdb9nsd_000108_b0 the `$hessian` block was
# diagonalised under both candidate unit systems and compared against the
# `$vibrational_frequencies` block ORCA writes into the same file:
#
#     assuming Eh/Bohr^2  ->  2206.86  3091.21  3139.92  3499.80 cm^-1
#     assuming Eh/A^2     ->  1167.82  1635.80  1661.57  1852.02 cm^-1
#     ORCA's own numbers  ->  2206.82  3091.16  3139.86  3499.74 cm^-1
#
# So `$hessian` is in Eh/Bohr^2, agreeing to <= 0.06 cm^-1. `.engrad` states its own
# units in a comment line: "The current gradient in Eh/bohr".
#
# `verify_hess_frequencies` turns that one-off check into an assertion that runs on
# every sample. A one-off check is not a criterion (memory-discard section 6 rule 9).
# =========================================================================================

#: Wavenumber conversion: sqrt(eV / (A^2 * amu)) -> cm^-1.
_FREQ_CONV = np.sqrt(1.602176634e-19 / (1e-20 * 1.66053906660e-27)) / (2 * np.pi * 2.99792458e10)


def _hess_section(lines, tag):
    """Index of the line holding `$<tag>`; raises if absent."""
    want = "$" + tag
    for i, line in enumerate(lines):
        if line.strip() == want:
            return i
    raise KeyError("{!r} not found in the .hess file".format(want))


def parse_hess(path):
    """Read an ORCA `.hess` file.

    Returns a dict with `hessian` (3N x 3N, **Eh/Bohr^2, as written**), `masses`
    (amu), `symbols`, `positions_bohr`, `frequencies_cm_inv` (ORCA's own), and
    `n_atoms`. No unit conversion happens here -- conversion is the caller's
    explicit act, so that a wrong assumption cannot hide inside a reader.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()

    i = _hess_section(lines, "hessian")
    n = int(lines[i + 1].split()[0])
    H = np.zeros((n, n), dtype=float)
    row_cursor = i + 2
    while True:
        cols = [int(x) for x in lines[row_cursor].split()]
        row_cursor += 1
        for _ in range(n):
            parts = lines[row_cursor].split()
            r = int(parts[0])
            for c, v in zip(cols, parts[1:]):
                H[r, int(c)] = float(v)
            row_cursor += 1
        if cols[-1] == n - 1:
            break

    i = _hess_section(lines, "atoms")
    nat = int(lines[i + 1].split()[0])
    symbols, masses, positions = [], [], []
    for k in range(nat):
        parts = lines[i + 2 + k].split()
        symbols.append(parts[0])
        masses.append(float(parts[1]))
        positions.append([float(x) for x in parts[2:5]])

    i = _hess_section(lines, "vibrational_frequencies")
    nf = int(lines[i + 1].split()[0])
    freqs = np.array([float(lines[i + 2 + k].split()[1]) for k in range(nf)])

    return {
        "hessian_eh_bohr2": H,
        "masses_amu": np.array(masses),
        "symbols": symbols,
        "positions_bohr": np.array(positions),
        "frequencies_cm_inv": freqs,
        "n_atoms": nat,
        "path": str(path),
    }


def hessian_to_ev_per_angstrom2(hessian_eh_bohr2):
    """Eh/Bohr^2 -> eV/A^2. The one place this conversion is written down."""
    bohr_per_angstrom = BOHR_PER_ANGSTROM
    return np.asarray(hessian_eh_bohr2) * EV_PER_HARTREE * (bohr_per_angstrom ** 2)


def frequencies_from_hessian(hessian_ev_a2, masses_amu):
    """Mass-weight, diagonalise, return signed wavenumbers in cm^-1 (ascending).

    Signed: a negative eigenvalue yields a negative wavenumber rather than a NaN,
    because an imaginary mode is information, not an error.
    """
    inv = np.repeat(1.0 / np.sqrt(np.asarray(masses_amu, dtype=float)), 3)
    Hm = np.asarray(hessian_ev_a2) * inv[:, None] * inv[None, :]
    w = np.linalg.eigvalsh(Hm)
    return np.sign(w) * np.sqrt(np.abs(w)) * _FREQ_CONV


#: ORCA writes the rigid-body entries of `$vibrational_frequencies` as exact zeros, so
#: they are identified by being AT zero rather than by being the smallest. The
#: distinction is not pedantic: a structure with an imaginary mode has an entry BELOW
#: the rigid ones, and "the six smallest" then discards the imaginary mode and keeps a
#: rigid zero in its place. See the note on `verify_hess_frequencies`.
RIGID_ENTRY_TOLERANCE_CM = 1.0e-8


def verify_hess_frequencies(parsed, tol_cm_inv=0.5, expect_rigid=6):
    """Assert that our parse reproduces ORCA's own printed frequencies.

    This single check catches four classes of error at once -- wrong units, wrong
    masses, wrong ordering, and a broken parse -- at zero computational cost.

    BOTH SIDES ARE THE 3N-6 VIBRATIONAL WAVENUMBERS, AND WHY THAT MATTERS
    --------------------------------------------------------------------
    ORCA prints frequencies from the mass-weighted Hessian **with translation and
    rotation projected out**. Comparing them against an unprojected spectrum happens to
    agree to about 0.05 cm^-1 at a converged stationary point -- which is why the
    original version of this function passed on 46 structures and looked correct.

    It is not correct, and it fails the moment a structure carries an imaginary mode.
    Measured 2026-09-09 on acetone at RI-MP2/RIJK/cc-pVTZ, where ORCA reports
    `[-35.74, 0, 0, 0, 0, 0]` for the six lowest: keeping "the entries that are not
    zero" retains the imaginary mode and drops only FIVE zeros, so the two lists are
    then aligned one place apart and the function reported a **48.03 cm^-1**
    disagreement that was entirely its own doing. With the Eckart projection the same
    file agrees to **0.058 cm^-1**.

    The imaginary mode itself is real and is not this function's business -- it is
    reported through `n_imaginary`, and it is for the caller to refuse thermodynamics on
    it (`thermo.vibrational` does).

    Raises ValueError past `tol_cm_inv`. Returns the diagnostic dict on success.
    """
    from ..quasi_harmonic import mode_match                # local: avoids a cycle

    H_ev = hessian_to_ev_per_angstrom2(parsed["hessian_eh_bohr2"])
    positions_A = np.asarray(parsed["positions_bohr"], dtype=float) / BOHR_PER_ANGSTROM
    ours = mode_match.projected_modes(H_ev, parsed["masses_amu"], positions_A,
                                      "hessian")[0]
    theirs = np.asarray(parsed["frequencies_cm_inv"], dtype=float)

    rigid = np.abs(theirs) <= RIGID_ENTRY_TOLERANCE_CM
    if int(rigid.sum()) != int(expect_rigid):
        raise ValueError(
            "ORCA printed {} entries at exactly zero, expected {} rigid modes. The six "
            "lowest are {} ({}).".format(
                int(rigid.sum()), expect_rigid,
                np.sort(theirs)[:6].round(4).tolist(), parsed["path"]))

    a = np.sort(ours)
    b = np.sort(theirs[~rigid])
    if a.shape != b.shape:
        raise ValueError(
            "frequency count differs: our projection gives {}, ORCA printed {} "
            "vibrational modes ({})".format(a.shape[0], b.shape[0], parsed["path"]))

    dev = np.abs(a - b)
    worst = float(dev.max()) if dev.size else 0.0
    if worst > tol_cm_inv:
        raise ValueError(
            "parsed Hessian does not reproduce ORCA's own frequencies: max deviation "
            "{:.4f} cm^-1 > {:.4f} ({}).\n"
            "Check units (the block is Eh/Bohr^2), masses, and mode ordering.".format(
                worst, tol_cm_inv, parsed["path"]))

    return {
        "max_deviation_cm_inv": worst,
        "rms_deviation_cm_inv": float(np.sqrt((dev ** 2).mean())) if dev.size else 0.0,
        "n_compared": int(len(a)),
        "symmetry_residual": float(np.abs(
            parsed["hessian_eh_bohr2"] - parsed["hessian_eh_bohr2"].T).max()),
        "n_imaginary": int((b < 0.0).sum()),
        "lowest_cm_inv": float(b.min()) if b.size else float("nan"),
    }
