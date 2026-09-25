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

#: The `%freq` block pinned into EVERY reference-level input (ticket 36). ORCA's own
#: defaults are `ProjectTR true`, `TransInvar true` and `CutOffFreq 1.0` (ORCA 6 manual
#: 6.5 for the first two, 7.27 for the third); pinning them puts the projection state of
#: every reference Hessian into the job input instead of leaving it inherited from a
#: build default. With `ProjectTR` off, transrotational modes cross `CutOffFreq` and the
#: entropy shows a finite jump (manual 6.5, "strongly discouraged"). `TransInvar`'s
#: acoustic-sum-rule correction is applied inside ORCA's frequency step and is NOT
#: written back into the `.hess` this package reads (checked once; note above
#: `parse_hess`), and `CutOffFreq 1.0` is the same 1 cm^-1 floor as the imaginary count
#: of `optimise_and_hessian`.
FREQ_BLOCK = "\n".join([
    "# projection state declared (ORCA 6 manual 6.5 / 7.27; defaults, pinned)",
    "%freq",
    "   ProjectTR true",
    "   TransInvar true",
    "   CutOffFreq 1.0",
    "end",
])

#: The reference levels ORCA runs, by level name (CONTEXT.md spelling). `keywords` is
#: the `!` line of the Opt + Hessian job, `blocks` the extra `%` input (every level pins
#: its projection state there -- `FREQ_BLOCK`, ticket 36), `route` what the Hessian is.
#: wB97M: the analytic Hessian works in ORCA 6.0.1 (ticket 26). DLPNO-CCSD(T):
#: ORCA has no analytic gradient for it, so geometry and Hessian are numerical end to end
#: (Opt NumGrad + NumFreq, (6N)^2 single points for the Hessian; ticket 32); keywords are
#: the hkuhpc convention (`core-bind/orca.md`) plus TightPNO, because the PNO truncation
#: noise between displaced geometries is the error source of a numerical curvature.
LEVELS = {
    "wb97m-d3bj_def2-tzvppd": dict(
        keywords="wB97M-D3BJ def2-TZVPPD TightOpt Freq TightSCF", blocks=FREQ_BLOCK, route="analytic",
        single_point="wB97M-D3BJ def2-TZVPPD TightSCF"),
    # the wavefunction ladder at its own minima (2026-09-18): HF has an analytic Hessian
    # (not with RIJK -- ORCA refuses; exact integrals), RI-MP2 an analytic gradient (NumFreq =
    # second differences of analytic gradients, the "semi-numerical" route; its noise floor
    # is read like a numerical one)
    "hf_cc-pvtz": dict(
        keywords="RHF cc-pVTZ TightOpt Freq TightSCF", blocks=FREQ_BLOCK, route="analytic",
        single_point="RHF cc-pVTZ TightSCF"),
    "ri-mp2_cc-pvtz": dict(
        keywords="RI-MP2 cc-pVTZ cc-pVTZ/C cc-pVTZ/JK RIJK TightOpt NumFreq TightSCF", blocks=FREQ_BLOCK,
        route="numerical", single_point="RI-MP2 cc-pVTZ cc-pVTZ/C cc-pVTZ/JK RIJK TightSCF"),
    "ri-mp2_aug-cc-pvtz": dict(
        keywords="RI-MP2 aug-cc-pVTZ aug-cc-pVTZ/C cc-pVTZ/JK RIJK TightOpt NumFreq TightSCF", blocks=FREQ_BLOCK,
        route="numerical", single_point="RI-MP2 aug-cc-pVTZ aug-cc-pVTZ/C cc-pVTZ/JK RIJK TightSCF"),
    "dlpno-ccsdt_cc-pvtz": dict(
        keywords="DLPNO-CCSD(T) cc-pVTZ cc-pVTZ/JK RIJK cc-pVTZ/C TightPNO TightSCF Opt NumGrad NumFreq",
        blocks=FREQ_BLOCK + "\n%mdci\n   TCutPairs 1e-6\nend\n%loc\n   LocMet AHFB\n   OCC true\nend",
        route="numerical",
        single_point="DLPNO-CCSD(T) cc-pVTZ cc-pVTZ/JK RIJK cc-pVTZ/C TightPNO TightSCF"),
}


def level_spec(level):
    """The ORCA specification of a reference level, or a KeyError naming the known ones."""
    if level not in LEVELS:
        raise KeyError("no ORCA specification for level {!r}; known: {}".format(
            level, ", ".join(sorted(LEVELS))))
    return dict(LEVELS[level])


def input_text(symbols, positions, keywords, nprocs, maxcore, charge=0, mult=1, blocks=""):
    """One ORCA input: the `!` line, `%pal`, `%maxcore`, any extra `%` blocks, the geometry."""
    lines = ["! {}".format(keywords), "%pal nprocs {} end".format(int(nprocs)),
             "%maxcore {}".format(int(maxcore))]
    if blocks:
        lines += [blocks.rstrip("\n")]
    lines.append("* xyz {} {}".format(int(charge), int(mult)))
    for s, r in zip(symbols, positions):
        lines.append("{:2s} {:18.10f} {:18.10f} {:18.10f}".format(s, *r))
    lines.append("*")
    return "\n".join(lines) + "\n"


#: The kinds of an ORCA job's files published from its run directory into the file group
#: `<workdir>/<stem>.<ext>` (ticket 09b, 2026-09-20): the input, the FULL `.out`, the
#: Hessian, the gradient, the optimised geometry. Everything else ORCA writes (`.gbw`,
#: `.densities`, `.tmp*`, `property.txt`) is scratch and goes with the run directory.
KEEP = (".inp", ".out", ".hess", ".engrad", ".xyz")
TERMINAL = "****ORCA TERMINATED NORMALLY****"


def _run_job(workdir, stem, inp_text, timeout_s=None, run_stem="job"):
    """Run one ORCA job: `<workdir>/.<stem>/job.inp` in that run directory, then the KEEP
    kinds copied to `<workdir>/<stem>.<ext>` (the `.out` on a failure too, for reading) and
    the run directory removed. Returns (rc, seconds, the text of the `.out`)."""
    workdir = Path(workdir)
    rundir = workdir / ("." + stem)
    rundir.mkdir(parents=True, exist_ok=True)
    inp, out = rundir / (run_stem + ".inp"), rundir / (run_stem + ".out")
    inp.write_text(inp_text, encoding="utf-8")
    t0 = time.time()
    try:
        with open(out, "w") as fh:
            rc = subprocess.call([orca_binary(), str(inp)], stdout=fh, stderr=subprocess.STDOUT,
                                 cwd=str(rundir), env=subprocess_env(), timeout=timeout_s)
    finally:
        for ext in KEEP:
            f = rundir / (run_stem + ext)
            if f.is_file():
                shutil.copy2(f, workdir / (stem + ext))
        shutil.rmtree(rundir, ignore_errors=True)
    text = (workdir / (stem + ".out")).read_text(encoding="utf-8", errors="replace") \
        if (workdir / (stem + ".out")).is_file() else ""
    return rc, time.time() - t0, text


def run_single_point(symbols, positions, workdir, keywords, nprocs=8, maxcore=3000,
                     blocks="", charge=0, mult=1, stem="job", timeout_s=None):
    """A single-point energy at any keyword line, published as the file group
    `<workdir>/<stem>.{inp,out,engrad}` (`_run_job`), skipped when a normally terminated
    `.out` is there. Returns energy_eh, seconds (None when reused), the path of the full
    `.out`."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / (stem + ".out")
    seconds = None
    if not (out.is_file() and TERMINAL in out.read_text(encoding="utf-8", errors="replace")):
        rc, seconds, text = _run_job(workdir, stem, input_text(symbols, positions, keywords, nprocs, maxcore,
                                                              charge, mult, blocks), timeout_s)
        if TERMINAL not in text:
            raise RuntimeError("ORCA did not finish normally for {} in {} (rc {}). Tail:\n{}".format(
                stem, workdir, rc, "\n".join(text.split("\n")[-25:])))
    text = out.read_text(encoding="utf-8", errors="replace")
    return dict(energy_eh=final_energy_from_out(text), seconds=seconds, out=str(out),
                keywords=keywords)


def orca_binary():
    """The ORCA executable. The environment variable S0_ORCA_BIN takes precedence."""
    p = os.environ.get("S0_ORCA_BIN", DEFAULT_BIN)
    if not Path(p).exists():
        raise FileNotFoundError(
            "ORCA not found: {}\nSet S0_ORCA_BIN to the executable.".format(p))
    return p


def subprocess_env():
    """The environment an ORCA subprocess runs in -- Slurm-blind, and the ONE seam
    every ORCA launch goes through (ADR 0008). `S0_ORCA_PATH` and `S0_ORCA_LIB`, when
    set, are prepended to PATH and LD_LIBRARY_PATH FOR THE SUBPROCESS ONLY: on tianhe
    ORCA 6.1.1 and its OpenMPI live in the conda env `orca611` (`~/env_orca611.sh`),
    which a worker running in the `openqha` env must not activate -- its libraries
    would shadow the worker's own (hpc/env/orca.sh records the two paths without
    activating).

    Every variable whose NAME starts with `SLURM` or `PMI` is deleted. The PREFIX RULE
    (not a pattern list) is deliberate: the draw300 failure of 2026-09-25 happened
    because a narrow pattern removed `SLURM_TASKS_PER_NODE` while keeping the
    `SLURM_JOBID` that arms OpenMPI 4.1's slurm components -- `ras/slurm` then
    force-terminates when a required variable is absent (research note, 2026-09-25).
    With `SLURM_JOBID` gone no `ras`/`plm`/`ess` slurm component is even eligible, and
    ORTE falls back to the local host. The worker keeps its own Slurm view; only the
    ORCA child is blind.

    `OMPI_MCA_hwloc_base_binding_policy=none` is ORCA's documented off-switch for
    OpenMPI's own CPU binding, so placement's only owner is the worker's `taskset`
    range. `OMPI_MCA_rmaps_base_oversubscribe=1` is the documented fallback only (set
    if a "not enough slots" line ever appears) and is deliberately not set here."""
    env = dict(os.environ)
    for name in [k for k in env if k.startswith(("SLURM", "PMI"))]:
        del env[name]
    env["OMPI_MCA_hwloc_base_binding_policy"] = "none"
    if env.get("S0_ORCA_PATH"):
        env["PATH"] = env["S0_ORCA_PATH"] + os.pathsep + env.get("PATH", "")
    if env.get("S0_ORCA_LIB"):
        env["LD_LIBRARY_PATH"] = env["S0_ORCA_LIB"] + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


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
                             env=subprocess_env(), timeout=timeout_s)
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


def final_rms_gradient(out_text):
    """The last `RMS gradient` of ORCA's geometry-convergence table (Eh/bohr), or None
    when the output holds no optimisation. For a numerical-gradient optimisation this is
    the figure that says how well the minimum is defined."""
    vals = re.findall(r"RMS gradient\s+([-+]?\d+\.\d+)", out_text)
    return float(vals[-1]) if vals else None


def n_single_points(out_text):
    """How many `FINAL SINGLE POINT ENERGY` lines the output holds -- the count of energy
    evaluations a numerical route spent (an analytic Freq has one per optimisation step)."""
    return len(re.findall(r"FINAL SINGLE POINT ENERGY", out_text))


def optimise_and_hessian(symbols, positions, workdir, keywords=REFERENCE_KEYWORDS,
                         nprocs=8, maxcore=3000, charge=0, mult=1, stem="job",
                         timeout_s=None, blocks="", rerun=False):
    """Geometry optimisation plus Hessian at one level, published as the file group
    `<workdir>/<stem>.{inp,out,hess,xyz}` (`_run_job`; `workdir` = `layout.msrrho_dir`,
    `stem` = `layout.orca_level_stem`).

    Skips ORCA when `workdir/<stem>.hess` exists and the `.out` terminated normally, so a
    Batch can be resumed. `rerun=True` runs anyway and REPLACES the file group: the
    ticket-39 soft-saddle retry re-runs the same job from the relaxed geometry, and the
    file group keeps its one meaning -- the basin's final job at this level. Returns the
    relaxed geometry (A), energy (Eh, the last FINAL SINGLE POINT ENERGY of the `.out`),
    the parsed Hessian record (`parse_hess`), whether the Hessian was analytic or
    numerical, the wall time (None when reused), and ORCA's version.

    `n_imaginary` counts the entries of ORCA's printed spectrum below -1 cm^-1 -- the
    1 cm^-1 floor of the `CutOffFreq 1.0` pinned in `FREQ_BLOCK` (ORCA 6 manual 7.27),
    so a negative mode inside (-1, 0) cm^-1 sits inside the floor and is not counted.

    The FULL `<stem>.out` is the engine record and is kept as ORCA wrote it -- in the file
    group and in every fixture copied from one (user ruling 2026-09-17). Never trim it
    to the lines a parser happens to read: the optimisation trajectory, SCF convergence,
    the Hessian route and the thermochemistry block are what a reader needs when a number
    looks wrong, and none of them can be recovered from a single-point line.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / (stem + ".out")
    hess = workdir / (stem + ".hess")
    done = hess.is_file() and out.is_file() and TERMINAL in out.read_text(encoding="utf-8", errors="replace")
    done = done and not rerun
    seconds = None
    if not done:
        rc, seconds, text = _run_job(workdir, stem, input_text(symbols, positions, keywords, nprocs, maxcore,
                                                              charge, mult, blocks), timeout_s)
        if TERMINAL not in text:
            raise RuntimeError("ORCA did not finish normally for {} in {} (rc {}). Tail:\n{}".format(
                stem, workdir, rc, "\n".join(text.split("\n")[-25:])))
    text = out.read_text(encoding="utf-8", errors="replace")
    parsed = parse_hess(hess)
    if parsed["symbols"] != list(symbols):
        raise ValueError("ORCA reordered the atoms in {}: {} -> {}".format(
            workdir, list(symbols), parsed["symbols"]))
    m = re.search(r"Program Version\s+(\S+)", text)
    return dict(keywords=keywords, workdir=str(workdir), stem=stem,
                opt_grad_rms=final_rms_gradient(text), n_single_points=n_single_points(text),
                positions_A=(np.asarray(parsed["positions_bohr"]) / BOHR_PER_ANGSTROM).tolist(),
                energy_eh=final_energy_from_out(text), hess=parsed,
                hessian_route=hessian_route(text), seconds=seconds,
                orca_version=m.group(1) if m else "unknown", nprocs=int(nprocs),
                # -1.0 cm^-1: the same 1 cm^-1 floor as the `CutOffFreq 1.0` pinned in
                # FREQ_BLOCK (ORCA 6 manual 7.27); a mode inside (-1, 0) is not counted
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
#
# THE STORED MATRIX IS THE PRE-ASR HESSIAN (checked once, 2026-09-25; ticket 36).
# `TransInvar true` -- ORCA's default, pinned in `FREQ_BLOCK` -- enforces translation
# invariance (the acoustic sum rule, ASR) during ORCA's frequency step, but the
# correction is not written back into the file. On the three ORCA 6.0.1 propanal
# fixtures (analytic route,
# `tests/data/propanal_molecule/msrrho/orca.wb97m-d3bj_def2-tzvppd.basin0{0,1,2}.hess`)
# the stored `$hessian`'s row sums -- the translation-mode residual sum_j H_ij -- are
# 1.14e-4, 3.89e-5 and 1.21e-4 Eh/Bohr^2 (up to 1.7e-4 of the largest element), where
# an enforced sum rule would leave ~1e-16. So the label this module reads is the raw
# matrix as written: re-applying a row-mean sum-rule correction to it shifts the
# Eckart-projected modes by <= 0.062 cm^-1 (rms <= 0.015) on those basins, inside the
# 0.5 cm^-1 round-trip tolerance, and `verify_hess_frequencies` keeps asserting that
# ORCA's own printed frequencies are reproduced from the stored matrix (there: the same
# order of agreement, <= 0.057 cm^-1). `n_imaginary`, the frequencies and everything
# downstream are computed on this raw matrix; the ASR correction is ORCA's internal
# act, not a property of the file.
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
