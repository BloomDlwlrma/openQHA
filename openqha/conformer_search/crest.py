"""Conformer sampling for package 1 -- the CREST composite calculator (GFN-FF sampling
plus MACE refinement).

**The design follows the upstream documentation** (<https://crest-lab.github.io/crest-docs/>,
"Special Calculators / Composite calculators"):

    do not use a machine-learning potential directly inside the metadynamics sampling;
    sample with a semi-empirical or force-field "workhorse" and let the machine-learning
    potential refine only the final ensemble.

The orders of magnitude settle this: one conformer search on a small molecule needs about
10^5 energy-and-gradient calls (measured here on acetone: **130 937**). Measured here,
MACE accounts for only **2514 of them, about 1.9 per cent**.

**This agrees word for word with the repository's own two-stage design.**

--------------------------------------------------------------------------------------
Two routes: `generic` and `mlip`
--------------------------------------------------------------------------------------
Upstream CREST **3.1** offers `method = "mlip"` (`fmlip-relay`: a resident server plus a
TCP socket, with native support for MACE-OFF and a custom `.model` via `mlip_modelpath`).
**But 3.1 is not released** -- checked point by point: releases go only to
v3.0.2; the continuous-release binary contains 0 occurrences of the string `mlip`;
upstream master's `src/calculator/` has no mlip source file; `subprojects/` has no
`fmlip_relay`.

So this module goes through `backend="generic"` by default, using this repository's own
socket client (`openqha/potentials/mace_server.py` plus `scripts/production/s0_mace_engrad.py`,
**architecturally the same as fmlip-relay**). The `backend="mlip"` branch is already
written, and **once 3.1 lands it is a one-parameter change**.

--------------------------------------------------------------------------------------
A known upstream defect, and why this module does not meet it
--------------------------------------------------------------------------------------
CREST 3.0.2 fails on exactly one combination, "**an external generic calculator plus
molecular dynamics or metadynamics**" (measured here: all 14 metadynamics runs
`terminated EARLY`, each in 0.019-0.021 seconds).
**This module does not meet it**, because the dynamics is run by the built-in GFN-FF and
the external calculator does only single points and optimisations -- measured, the same
external calculator ran 2514 times in the parallel `_N` subdirectories with no failure.
"""
import os
import shutil
import subprocess
import time
from pathlib import Path

from .. import S0_ROOT

#: There used to be a `TEMPLATE` constant here naming
#: `configs/crest_composite.toml.template`. The file was lost in the 2026-09-03
#: restructure (it survives in `_backup/pre_branchD_restructure_2026-09-03/`) and
#: nothing had read it for some time: `write_input` builds the TOML inline. The
#: constant is gone rather than the file restored, because two sources of truth
#: for the same content is how they drift apart. Removed 2026-09-04 after
#: t_translation_preserved_numbers reported it MISSING (that gate was retired 2026-09-09).

# --------------------------------------------------------------------------------------
# The environment every CREST call must carry
# --------------------------------------------------------------------------------------
# **OPENBLAS_NUM_THREADS = 1 is not tuning, it fixes a defect.**
#
# What conda-forge's `crest` pulls in is the **pthreads build** of OpenBLAS
# (`libopenblas-0.3.34-pthreads_h94d23a6_0`, `liblapack.so.3 -> libopenblasp-r0.3.34.so`,
# with a GOMP count of 0 in the dynamic symbol table and `pthread_create` present). CREST
# itself is OpenMP-parallel (it links libgomp). **Calling a pthreads-threaded OpenBLAS from
# inside an OpenMP parallel region** makes it print, every single time:
#
#     OpenBLAS Warning : Detect OpenMP Loop and this application may hang.
#     Please rebuild the library with USE_OPENMP=1 option.
#
# Measured (acetone, the same input, changing only this one environment variable):
#
#     unset:  4.4 s | output 4406 lines, of which 3626 are this warning (82%) | 476 475 bytes
#     =1   :  2.7 s | output  779 lines, 0 warning lines                      |  33 989 bytes
#
# Three benefits from one action: **the warning goes to zero** (and with it the risk of
# "may hang"), **the output is 14 times smaller** (which in bulk is a matter of 4000
# files), and it is **1.6 times faster** (the pthreads OpenBLAS opens threads inside each
# OpenMP thread, oversubscribing this machine's 8 cores).
#
# **Why set it to 1 rather than switch to an OpenMP build of OpenBLAS**: the molecules here
# have 10-20 atoms, the BLAS calls themselves are tiny, and threading them gains nothing;
# the parallelism belongs at the "between jobs" level (CREST's threads plus this
# repository's job pool), not inside BLAS.
#
# 2026-09-04: environment.yml now ALSO pins `libopenblas=*=openmp*`, so the mismatch does
# not arise in the first place. The variable stays regardless, for two reasons: it is what
# was actually measured, and it still holds for a CREST that came from anywhere but that
# environment file.
#
# The same date retired the two-environment split this comment used to be an argument for.
# That split was justified by a measurement that changed the environments AND set this
# variable, then credited the difference to the split. Isolated -- same binary, same input
# (dsgdb9nsd_000018, `--gfn2 --noreftopo -T 4`), only the variable differing:
#
#     nothing set              36.5 s | 4737 lines, 3992 of them this warning | 507 KB
#     OPENBLAS_NUM_THREADS=1   25.1 s |  747 lines, 0 warnings                |  32 KB
#     + OMP_NUM_THREADS=1      25.9 s |  744 lines, 0 warnings                |  32 KB
#
# **The separation was doing nothing.** One variable accounts for all of it. Retired file
# and its numbers: `_superseded/environment-crest.README.md`.
CREST_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OPENBLAS_MAIN_FREE": "1",
}

#: The runtypes listed in the upstream documentation (`page/documentation/inputfiles.html`)
RUNTYPES = ("singlepoint", "optimize", "ancopt", "numhess", "optimize_ensemble",
            "screen", "md", "mtd", "metadynamics", "imtd-gc", "nci", "entropy",
            "cregen", "basinhopping", "ttconf", "ensemblehess")

#: The convergence levels listed in the upstream documentation
OPTLEVELS = ("crude", "vloose", "loose", "normal", "tight", "vtight", "extreme")

#: Values of `refine` (checked word for word against CREST 3.0.2
#: `src/parsing/parse_calcdata.f90:343`)
REFINE_LEVELS = ("sp", "singlepoint", "add", "correction", "opt", "optimization")

#: `refine = None` (or "none") = **no quality level at all**, pure GFN-FF sampling.
#: Downstream in this repository every conformer is tightened to `fmax = 1e-4` by our own
#: criterion and then deduplicated anyway, so CREST's internal machine-learning refinement
#: **may be duplicated work** -- this branch exists to measure whether it is.
NO_QUALITY_LEVEL = ("none", "off", None, False)


def crest_binary():
    """The CREST executable. The environment variable S0_CREST_BIN overrides it."""
    b = os.environ.get("S0_CREST_BIN") or shutil.which("crest")
    if not b:
        raise FileNotFoundError(
            "cannot find the crest executable.\n"
            "For installation see https://crest-lab.github.io/crest-docs/page/installation ;\n"
            "or set the environment variable S0_CREST_BIN to point at it.")
    return b


def crest_version(binary=None):
    """Read the version and commit out of `crest --version` -- every product carries it."""
    out = subprocess.run([binary or crest_binary(), "--version"],
                         capture_output=True, text=True).stdout
    ver = commit = None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Version"):
            ver = s.split(",")[0].replace("Version", "").strip()
        if s.startswith("commit"):
            commit = s.split("(", 1)[1].split(")", 1)[0] if "(" in s else None
    return dict(version=ver, commit=commit)


def supports_mlip(binary=None):
    """Does this CREST have `method = "mlip"` (that is, the 3.1 fmlip-relay interface)?

    **It does not guess from the version number; it looks for those strings in the binary
    itself** -- the continuous-release build still reports 3.0.2, so judging by the version
    number would be wrong.
    """
    b = binary or crest_binary()
    try:
        with open(b, "rb") as fh:
            blob = fh.read()
    except OSError:
        return False
    return b"mlip_backend" in blob or b"fmlip" in blob


#: Allowed workhorse (sampling-level) methods, taken verbatim from the upstream
#: docs `page/documentation/inputfiles.html` and cross-checked against
#: `src/parsing/parse_calcdata.f90`.
#:
#: The production workhorse is **"gfn2"**, which is the
#: published iMTD-GC configuration (Pracht, Bohle & Grimme, PCCP 2020, 22, 7169;
#: upstream example 1 is literally `crest struc.xyz --gfn2`). "gfnff" was only ever
#: a cost substitute chosen for a 16000-molecule campaign; that reason is
#: gone now that stage 0 runs a few hundred molecules.
WORKHORSES = ("gfn2", "gfn1", "gfnff", "gfn0", "tblite")


#: Hydrogen mass for branch A's metadynamics, in amu. The third member of the published
#: package (SHAKE all bonds + 5 fs + 2 amu; Grimme, JCTC 2019, 15, 2847).
#:
#: WRITTEN EXPLICITLY, though CREST's default is already 2.0. Measured, not assumed: a
#: plain MD driven by the [dynamics] block this module writes prints
#: `hydrogen mass /u   :   2.00000`. Writing it changes no number.
#:
#: It is written so that criterion 11 can FAIL. A criterion called "dynamics package
#: written as one" that checks two of the three members is trusting the third to a
#: default in someone else's source tree -- and a default that changes in a future CREST
#: would take every run off the published protocol with every criterion still passing.
HYDROGEN_MASS_AMU = 2.0


def write_input(workdir, input_xyz, runtype="imtd-gc", threads=4, optlev="tight",
                refine="sp", backend="generic", engine_client=None,
                mace_model=None, shake=None, workhorse="gfn2", tstep_fs=None,
                hmass_amu=HYDROGEN_MASS_AMU, calcspace=None):
    """Write one CREST input from the template. Returns the path written.

    Every parameter is explicit and none is left to an implicit default -- each of them
    appears in the product's provenance record.
    """
    if runtype not in RUNTYPES:
        raise ValueError("runtype is not one of the values in the upstream "
                         "documentation: {!r}\nAvailable: {}".format(
                             runtype, ", ".join(RUNTYPES)))
    if optlev not in OPTLEVELS:
        raise ValueError("optlev is not one of the values in the upstream "
                         "documentation: {!r}".format(optlev))
    no_quality = refine in NO_QUALITY_LEVEL
    if not no_quality and refine not in REFINE_LEVELS:
        raise ValueError("refine is not one of the values CREST 3.0.2 supports: {!r}\n"
                         "Available: {}".format(refine, ", ".join(REFINE_LEVELS)))
    if backend not in ("generic", "mlip"):
        raise ValueError("backend must be 'generic' or 'mlip'")
    if workhorse not in WORKHORSES:
        raise ValueError("workhorse is not one of the values upstream accepts: {!r}\n"
                         "allowed: {}".format(workhorse, ", ".join(WORKHORSES)))

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if no_quality:
        quality = None
    elif backend == "generic":
        client = Path(engine_client or (S0_ROOT / "scripts" / "production"
                                / "s0_mace_engrad.py"))
        if not client.exists():
            raise FileNotFoundError("the MACE client is not at: {}".format(client))
        if not os.access(client, os.X_OK):
            # Set it rather than refuse. CREST executes this file directly, so the bit
            # is a requirement of ours, on our own script, and turning it on is one
            # syscall -- there is nothing for the operator to decide.
            #
            # It goes missing for a reason that is invisible on Windows: git tracked
            # `s0_mace_engrad.py` as mode 100644 while the working copy on a Windows
            # checkout reports `rwxr-xr-x` regardless. Every fresh clone on a cluster
            # therefore arrived non-executable, and on 2026-09-09 that killed two branch
            # A jobs on Tianhe. The index has been corrected too (`git update-index
            # --chmod=+x`); this handles the transfers that do not carry a mode at all.
            try:
                os.chmod(client, os.stat(client).st_mode | 0o111)
            except OSError as exc:
                raise PermissionError(
                    "{} has no executable bit and it could not be set ({}). CREST "
                    "**executes it directly** (not as `python <script>`), so it must be "
                    "executable and its shebang must work.\n"
                    "  chmod +x {}".format(client, exc, client))
            if not os.access(client, os.X_OK):
                raise PermissionError(
                    "{} is still not executable after chmod -- a read-only or "
                    "noexec-mounted filesystem. CREST executes it directly, so it "
                    "cannot run from here.".format(client))
        quality = ('method   = "generic"\n'
                   'binary   = "{}"\n'
                   'gradtype = "engrad"\n'
                   'gradfile = "genericinp.engrad"\n'
                   'refine   = "{}"\n'.format(client, refine))
        if calcspace:
            # `calcspace` (alias `dir`) is an upstream key: "the directory in which CREST
            # shall perform this calculation", relative or absolute. Naming it makes the
            # external calculator's working directory a PERSISTENT one instead of scratch
            # CREST removes, which is the only way to see `genericinp.xyz`,
            # `genericinp.engrad` and the client's own stderr in `generic.out` after the
            # fact. On 2026-09-09 a whole Tianhe run produced NaN energies and left not
            # one of those three files behind.
            #
            # **Diagnostic, not production.** It keeps files for every gradient call, so
            # it belongs on a run you are watching, not on a campaign.
            quality += 'calcspace = "{}"\n'.format(calcspace)
    else:
        if not mace_model:
            raise ValueError("backend='mlip' needs mace_model to point at a .model file")
        quality = ('method         = "mlip"\n'
                   'mlip_backend   = "mace"\n'
                   'mlip_modelpath = "{}"\n'
                   'refine         = "{}"\n'.format(mace_model, refine))
    head = (
        "# Generated by openqha/crest.py -- do not hand-edit; edit that module\n"
        "# Every value here is passed in explicitly -- none is left to a CREST default.\n"
        '# Composite calculator: workhorse sampling + MACE refinement\n'
        '# (upstream docs, "Composite calculators").\n'
        'input   = "{}"\n'
        'runtype = "{}"\n'
        "threads = {}\n\n"
        "[calculation]\n"
        'optlev = "{}"\n\n'
        "# Workhorse: sampling and preliminary optimisation. Built-in method,\n"
        "# starts no external process.\n"
        "[[calculation.level]]\n"
        'method = "{}"\n').format(
            Path(input_xyz).name, runtype, int(threads), optlev, workhorse)

    if no_quality:
        # **No quality level** -- pure GFN-FF sampling. All the machine-learning-potential
        # work is handed downstream (openqha/crest_census.py tightens to fmax = 1e-4,
        # deduplicates by all-atom best root-mean-square deviation, and takes the Hessian),
        # and that step happens regardless, so refining again inside CREST is duplicated work.
        text = head + (
            "\n# **No quality level** -- pure GFN-FF sampling; the machine-learning "
            "potential is used downstream only.\n"
            "# The reasoning is in the NO_QUALITY_LEVEL comment in openqha/crest.py and in "
            "the package 1 refine comparison.\n")
    else:
        text = head + (
            "\n# Quality level: applied only to the structures that survive\n"
            "[[calculation.level]]\n" + quality)


    if tstep_fs is not None:
        # CREST's own metadynamics default is 5.0 fs (it prints it as `timestep dt`
        # in crest.out). That default is paired with SHAKE on ALL bonds -- CREST's
        # own shake default is 2. We run shake = 1 (H bonds only), which
        # removes exactly the constraint that makes a 5 fs step safe on the
        # heavy-atom modes.
        #
        # Measured 2026-09-03, edge C2H5O1N1_19_36, workhorse gfn2, shake = 1,
        # CREST default step: 29 and 20 metadynamics runs `terminated EARLY`.
        text += "\n[dynamics]\ntstep = {}\n".format(float(tstep_fs))
        if shake is not None:
            text += "shake = {}\n".format(int(shake))
        if hmass_amu is not None:
            text += "hmass = {}\n".format(float(hmass_amu))
    elif shake is not None:
        # **SHAKE levels**: 2 = constrain all bonds (the CREST default), 1 = constrain
        # bonds to hydrogen only, 0 = off.
        # Measured 2026-08-31: the default of 2 makes the SHAKE iteration fail to converge
        # on strained-ring systems (shake_module.f90 allows at most 250 iterations at a
        # tolerance of 1e-7), do_shake returns non-zero, the molecular dynamics is judged to
        # have failed, and the layer above prints `terminated EARLY`.
        # Three molecules x three levels, measured (number of EARLY events):
        #   dsgdb9nsd_000607  shake=2 -> 31 | shake=1 -> 0 | shake=0 -> 0
        #   dsgdb9nsd_003163  shake=2 -> 24 | shake=1 -> 0 | shake=0 -> 0
        #   dsgdb9nsd_002334  shake=2 ->  4 | shake=1 -> 0 | shake=0 -> 0
        # 0 is not used because a timestep of 1.5 fs needs the bonds to hydrogen constrained
        # (the X-H stretching period is about 10 fs).
        text += ("\n[dynamics]\n"
                 "shake = {}\n".format(int(shake)))
        if hmass_amu is not None:
            text += "hmass = {}\n".format(float(hmass_amu))
    p = workdir / "input.toml"
    p.write_text(text, encoding="utf-8")
    return p


def run(workdir, input_xyz, timeout_s=7200, env=None, **kwargs):
    """Hand one structure to CREST for a run. Returns a complete record dict.

    **Success or failure is judged only by CREST's own output**, never by the shell exit
    code (this repository has been caught by that three times).
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    src = Path(input_xyz)
    if src.resolve() != (workdir / src.name).resolve():
        shutil.copy2(src, workdir / src.name)

    toml = write_input(workdir, src, **kwargs)
    binary = crest_binary()
    e = dict(os.environ)
    e.update(CREST_ENV)                 # see the CREST_ENV comment: this fixes an OpenBLAS
                                        # defect, it is not tuning
    e.update(env or {})

    t0 = time.time()
    with (workdir / "crest.out").open("w") as fh:
        proc = subprocess.run([binary, toml.name], cwd=workdir, stdout=fh,
                              stderr=subprocess.STDOUT, env=e, timeout=timeout_s)
    seconds = time.time() - t0

    out = (workdir / "crest.out").read_text(encoding="utf-8", errors="replace")
    rec = dict(
        workdir=str(workdir), seconds=seconds, returncode=proc.returncode,
        crest=crest_version(binary), binary=binary,
        # **Every value must be turned into a string**: `engine_client` is a Path, and
        # putting it straight into the record makes `json.dumps` raise
        # `TypeError: Object of type PosixPath is not JSON serializable` -- and it raises
        # **only after the whole batch has run**, losing a batch's worth of machine time.
        # Measured, and paid for, on 2026-08-30.
        settings={k: (str(v) if isinstance(v, Path) else v)
                  for k, v in kwargs.items()},
        terminated_normally="CREST terminated normally" in out,
        n_terminated_early=out.count("terminated EARLY"),
        n_completed_successfully=out.count("completed successfully"),
        total_engrad_calls=_grab_int(out, "Total number of energy+grad calls:"),
        products={n: (workdir / n).stat().st_size
                  for n in ("crest_conformers.xyz", "crest_best.xyz",
                            "crest_rotamers.xyz", "crestopt.xyz")
                  if (workdir / n).exists()})
    rec["ok"] = bool(rec["terminated_normally"] and rec["n_terminated_early"] == 0)
    judge_energies(workdir, out, rec)
    if not rec["ok"]:
        rec["tail"] = "\n".join(out.splitlines()[-25:])
    return rec


def run_with_shake_fallback(workdir, input_xyz, shake=2, fallback_to=1,
                            enabled=True, **kwargs):
    """Run CREST under the published protocol, retrying ONE molecule if it aborts.

    The production dynamics setting is the published package -- SHAKE on all bonds
    (`shake = 2`), 5 fs, hydrogen mass 2 amu (Grimme, JCTC 2019, 15, 2847). It has
    one measured failure mode: on strained rings the SHAKE iteration does
    not converge (`shake_module.f90`, 250 iterations, tolerance 1e-7), `do_shake`
    returns non-zero, the MD is judged failed and CREST prints `terminated EARLY`
    -- and conformers are lost with it. Measured: dsgdb9nsd_000607 / 003163 /
    002334 gave 31 / 24 / 4 EARLY at shake = 2 and zero at shake = 1.

    So the fallback is POINTWISE, not a global loosening: only a molecule that
    actually aborted is retried, once, at `fallback_to`.

    **The retry does not overwrite the first attempt.** It runs in a sibling
    directory `<workdir>_shake<fallback_to>` and the returned record carries
    `used_shake_fallback`, `shake_used` and `first_attempt`, so a fallen-back
    molecule can never be mistaken for one that ran the published protocol. A
    condition that cannot be read off the product has already cost this repo one
    dataset (defect 57).

    **The retry can be worse than the published run** (measured 2026-09-22): on a
    strained bicyclic (dsgdb9nsd_003375, ethynyl-housane) the SHAKE=2 run terminated
    normally with an ensemble and a few `terminated EARLY`, while the SHAKE=1 retry died
    in CREST's trial MTD (`Automatic MD restart failed 6 times! ERROR STOP`) -- fewer
    constraints, less integrable. When the retry leaves no ensemble and the first attempt
    has one, the FIRST attempt's record is returned, with `fallback_failed` naming the
    retry and its last line; criterion 10 then declares the early terminations. Only when
    neither attempt has an ensemble is the retry's record (its failure) returned.
    """
    rec = run(workdir, input_xyz, shake=shake, **kwargs)
    rec["shake_used"] = shake
    rec["used_shake_fallback"] = False
    if rec.get("ok") or not enabled or rec.get("n_terminated_early", 0) == 0:
        return rec

    retry_dir = Path("{}_shake{}".format(workdir, fallback_to))
    retry = run(retry_dir, input_xyz, shake=fallback_to, **kwargs)
    if not _has_ensemble(retry) and _has_ensemble(rec):
        rec["fallback_failed"] = dict(
            workdir=str(retry_dir), shake=fallback_to,
            terminated_normally=bool(retry.get("terminated_normally")),
            n_terminated_early=retry.get("n_terminated_early"),
            seconds=retry.get("seconds"), last_line=_last_line(retry.get("tail")))
        rec["fallback_reason"] = (
            "published protocol (shake={}) reported {} `terminated EARLY`; the retry at "
            "shake={} produced no ensemble ({}). The published run's ensemble is used and "
            "its early terminations are declared (criterion 10).".format(
                shake, rec.get("n_terminated_early"), fallback_to, rec["fallback_failed"]["last_line"]))
        return rec
    retry["shake_used"] = fallback_to
    retry["used_shake_fallback"] = True
    retry["first_attempt"] = dict(
        workdir=rec["workdir"], shake=shake,
        n_terminated_early=rec.get("n_terminated_early"),
        seconds=rec.get("seconds"),
        total_engrad_calls=rec.get("total_engrad_calls"))
    retry["fallback_reason"] = (
        "published protocol (shake={}) reported {} `terminated EARLY`; retried once "
        "at shake={}. This molecule did NOT run under the published "
        "protocol and must be reported separately.".format(
            shake, rec.get("n_terminated_early"), fallback_to))
    return retry


def _has_ensemble(rec):
    """CREST terminated normally and left `crest_conformers.xyz` in its work directory."""
    try:
        return bool(rec.get("terminated_normally")) and (Path(rec["workdir"]) / "crest_conformers.xyz").is_file()
    except (KeyError, TypeError, OSError):
        return False


def _last_line(tail):
    lines = [l.strip() for l in str(tail or "").splitlines() if l.strip()]
    return lines[-1] if lines else "-"


def record_from_dir(workdir, settings=None, seconds=None, returncode=None):
    """Read a record in the same format out of a CREST directory that has **already
    finished**, without rerunning it.

    Two uses: filling in the record when the process was killed by an outer timeout while
    CREST itself had finished; and inspecting after the fact a directory someone else ran.
    """
    workdir = Path(workdir)
    out_file = workdir / "crest.out"
    if not out_file.exists():
        raise FileNotFoundError("no crest.out: {}".format(workdir))
    out = out_file.read_text(encoding="utf-8", errors="replace")
    rec = dict(
        workdir=str(workdir), seconds=seconds, returncode=returncode,
        crest=crest_version(), binary=crest_binary(),
        settings={k: (str(v) if isinstance(v, Path) else v)
                  for k, v in dict(settings or {}).items()},
        reconstructed_from_directory=True,
        terminated_normally="CREST terminated normally" in out,
        n_terminated_early=out.count("terminated EARLY"),
        n_completed_successfully=out.count("completed successfully"),
        total_engrad_calls=_grab_int(out, "Total number of energy+grad calls:"),
        products={n: (workdir / n).stat().st_size
                  for n in ("crest_conformers.xyz", "crest_best.xyz",
                            "crest_rotamers.xyz", "crestopt.xyz")
                  if (workdir / n).exists()})
    # **`terminated normally` is not the same as `produced usable numbers`.** On
    # 2026-09-09 CREST ran to completion on Tianhe, printed its full wall-time summary,
    # and handed back 1814 conformers whose energies were every one of them NaN. Nothing
    # in this repository noticed, because the only thing anyone read off the ensemble was
    # how many frames it had. CREGEN's energy window is inert against NaN -- nine passes
    # discarded not one structure -- so the ensemble grows instead of shrinking and the
    # cost of the run explodes as a side effect of the same defect.
    #
    # So the energies are now part of the verdict, and `ok` is false without them.
    rec["ok"] = bool(rec["terminated_normally"] and rec["n_terminated_early"] == 0)
    judge_energies(workdir, out, rec)
    rec["settings_from_toml"] = read_input_settings(workdir / "input.toml")
    hits = [l.strip() for l in out.splitlines() if "wall-time:" in l]
    if hits:
        rec["wall_time_line"] = hits[-1]
        if rec["seconds"] is None:
            # CREST prints "* wall-time:  0 d,  0 h, 11 min, 35.682 sec"
            import re as _re
            m = _re.search(r"(\d+)\s*d,\s*(\d+)\s*h,\s*(\d+)\s*min,\s*([\d.]+)\s*sec",
                           hits[-1])
            if m:
                d, h, mi, s = (float(x) for x in m.groups())
                rec["seconds"] = d * 86400 + h * 3600 + mi * 60 + s
            rec["seconds_source"] = "the wall-time CREST printed itself (not the outer timing)"
    return rec


def _grab_int(text, key):
    for line in text.splitlines():
        if key in line:
            try:
                return int(line.split(key, 1)[1].strip().split()[0])
            except (ValueError, IndexError):
                return None
    return None


def comment_energy(comment):
    """The energy CREST writes on a frame's comment line, or None if it is not a number.

    Returns a float for `NaN` too -- `float("nan")` -- because "CREST wrote NaN there" and
    "CREST wrote something that is not a number at all" are different failures and the
    caller has to be able to tell them apart.
    """
    f = str(comment).split()
    if not f:
        return None
    try:
        return float(f[0])
    except ValueError:
        return None


def judge_energies(workdir, out, rec):
    """Add the ensemble-energy verdict to a CREST record, in place.

    **`CREST terminated normally` is not the same as `CREST produced usable numbers`.**
    On 2026-09-09 a Tianhe run finished cleanly, printed its full wall-time summary, and
    handed back **1814 conformers whose energies were every one of them NaN** (job
    7347197, propanal; the last verified acetone run gave 7 conformers). Nothing in this
    repository noticed, because the only thing anyone ever read off the ensemble was how
    many frames it had -- `read_ensemble` parses the coordinates with `float()` and
    returns the energy as a string.

    Two consequences, and the second is why this is not merely a correctness check:

      1. the numbers are unusable, and they flow downstream as if they were conformers;
      2. **CREGEN's energy window is inert against NaN.** Nine passes discarded not one
         structure (686 -> 686, 1372 -> 1372, 1814 -> 1814), so the ensemble only grows,
         and the next single-point stage runs on 1814 structures instead of a dozen.
         93% of that job's wall clock was single points on frames a working run would
         have thrown away. **The defect caused the cost.**

    So the energies are part of the verdict now, and `ok` is false without them.
    """
    inf = float("inf")
    rec["n_cregen_nan"] = sum(1 for line in out.splitlines()
                              if "E lowest" in line and "NaN" in line)
    ens = Path(workdir) / "crest_conformers.xyz"
    if not ens.exists():
        return rec
    frames = read_ensemble(ens)
    rec["n_conformers"] = len(frames)
    energies = [comment_energy(c) for c, _ in frames]
    rec["n_energies_unparsable"] = sum(1 for v in energies if v is None)
    rec["n_nonfinite_energies"] = sum(
        1 for v in energies if v is None or v != v or v in (inf, -inf))
    # ---- the failure mode that is NOT NaN, and that a finiteness check waves through --
    # Measured 2026-09-09, real CREST 3.0.2, n-butane, the production shape, with the
    # socket deliberately pointed at a path that does not exist: **every one of 1926
    # client invocations failed, and CREST exited 0, printed "terminated normally", and
    # wrote `0.00000000` as the energy of every conformer.** It never said a word about
    # the external calculator. So an external gradient program that is simply not there
    # produces a finite, plausible-looking, completely fabricated ensemble.
    #
    # The first version of this function passed that run as `ok`. It was written against
    # the NaN that Tianhe produced, and this is the neighbouring case -- so the criterion
    # is spread, not finiteness: **distinct conformers never have bit-identical energies.**
    finite = [v for v in energies if v is not None and v == v and v not in (inf, -inf)]
    rec["n_zero_energies"] = sum(1 for v in finite if v == 0.0)
    rec["energy_spread"] = (max(finite) - min(finite)) if len(finite) > 1 else None
    if not rec["n_nonfinite_energies"] and len(finite) > 1 and rec["energy_spread"] == 0.0:
        rec["ok"] = False
        rec["energy_defect"] = (
            "all {} conformer energies are bit-identical ({!r}). Distinct conformers do "
            "not have identical energies, so these were never computed. {}This is what "
            "CREST does when the external gradient program fails on every call: it exits "
            "0 and reports a normal termination (measured 2026-09-09 on real CREST "
            "3.0.2). Set S0_MACE_TRACE and read engrad_trace.log -- if the statuses are "
            "FAIL, the reason is on that line.".format(
                len(finite), finite[0],
                "They are all exactly zero, which is CREST's initial value. "
                if rec["n_zero_energies"] == len(finite) else ""))
    elif rec["n_nonfinite_energies"]:
        rec["ok"] = False
        rec["energy_defect"] = (
            "{} of {} conformer energies in crest_conformers.xyz are not finite numbers"
            "{}. CREST cannot rank or discard structures on them, so what came back is "
            "not a conformer list -- it is close to every frame CREST ever looked at. "
            "Do not read basins off this. To find out why the gradients never arrived, "
            "rerun with S0_MACE_TRACE set and a calcspace named, and compare the "
            "server's own gradient count against `Total number of energy+grad calls` "
            "in crest.out.".format(
                rec["n_nonfinite_energies"], rec["n_conformers"],
                "" if not rec["n_cregen_nan"]
                else ", and CREGEN reported a NaN lowest energy {} time(s)".format(
                    rec["n_cregen_nan"])))
    return rec


def read_ensemble(path):
    """Read a CREST multi-frame xyz. Returns [(comment line, [(element, x, y, z), ...]), ...]."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    out, i = [], 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i].split()[0])
        comment = lines[i + 1]
        atoms = []
        for k in range(n):
            f = lines[i + 2 + k].split()
            atoms.append((f[0], float(f[1]), float(f[2]), float(f[3])))
        out.append((comment, atoms))
        i += n + 2
    return out


# ======================================================================================
# In bulk: a pool of resident MACE servers plus a pool of CREST jobs
# ======================================================================================
# **Why pools**: a `method = "generic"` server serves one gradient at a time (one model
# instance, one lock). Running several CRESTs in parallel therefore requires **one server
# and one socket per parallel slot**, or they queue on the same lock and the parallelism
# is not real.
#
# **Why the working directory defaults to a native Linux disk**: CREST writes a great many
# small files in its parallel `_N` subdirectories, and going through /mnt/c (9p) is
# I/O-bound. The products are copied back into the repository once computed -- what is
# copied is the result, not the process.
def start_servers(n, socket_prefix=None, torch_threads=1,
                  python=None, timeout_s=180):
    """Start n resident MACE servers, one socket each. Returns [(process, socket path), ...].

    The sockets and their logs go in **this job's own node-local directory**,
    `<TMPDIR or /tmp>/<owner>/<job id>/` -- see `config.socket_dir`.

    They used to go under `runs_root/sockets/`, which is on the SHARED filesystem, and on
    2026-09-09 that collided on Tianhe: two branch A jobs submitted back to back both
    opened `runs_root/sockets/s0_mace_pool_0.sock` on two different compute nodes, and
    whichever bound second unlinked the first one's socket. A Unix socket is a rendezvous
    between processes on one machine; it has no business on a shared filesystem.
    """
    import sys
    from .. import config
    py = python or sys.executable
    if socket_prefix is None:
        # **The process id is in the name, and that is not decoration.** `socket_dir()`
        # is per JOB, which separates two jobs on two nodes -- the 2026-09-09 collision.
        # It does not separate two pipelines inside ONE job, and running two molecules
        # concurrently in a single allocation is exactly what
        # `examples/02ab_pair/branchA-pair.conf` does: both would otherwise open
        # `<socket_dir>/s0_mace_pool_0.sock`, and whichever bound second would unlink the
        # first one's socket. Same defect, one level down.
        #
        # `sun_path` is 108 bytes and truncates silently, so this stays short: the pid is
        # at most 7 digits and config.socket_path() checks the total against
        # SOCKET_PATH_LIMIT.
        socket_prefix = str(config.socket_dir() / "s0_mace_pool_{}".format(os.getpid()))
    servers = []
    for k in range(n):
        sock = "{}_{}.sock".format(socket_prefix, k)
        if os.path.exists(sock):
            os.unlink(sock)
        e = dict(os.environ)
        e.update(OMP_NUM_THREADS=str(torch_threads),
                 MKL_NUM_THREADS=str(torch_threads),
                 TORCH_NUM_THREADS=str(torch_threads),
                 OPENBLAS_NUM_THREADS=str(torch_threads))
        log = open("{}.log".format(sock), "w")
        p = subprocess.Popen([py, "-m", "openqha.potentials.mace_server", "--socket", sock],
                             cwd=str(S0_ROOT), env=e, stdout=log,
                             stderr=subprocess.STDOUT)
        servers.append((p, sock))
    t0 = time.time()
    for p, sock in servers:
        while not os.path.exists(sock):
            if p.poll() is not None:
                raise RuntimeError("the MACE server exited as soon as it started (return "
                                   "code {}), log: {}.log".format(p.returncode, sock))
            if time.time() - t0 > timeout_s:
                raise TimeoutError("waited more than {} s for {} to appear".format(
                    timeout_s, sock))
            time.sleep(0.2)
    return servers


def stop_servers(servers):
    for p, sock in servers:
        try:
            p.terminate()
            p.wait(timeout=30)
        except Exception:
            p.kill()
        if os.path.exists(sock):
            try:
                os.unlink(sock)
            except OSError:
                pass


def run_pool(jobs, sockets, on_done=None, timeout_s=7200, **kwargs):
    """Run a batch of CREST jobs in parallel. `jobs` is [(label, working directory, input
    xyz), ...].

    Each parallel slot is bound to one socket (that is, one resident server). Returns
    {label: record}.
    **A single job failing does not end the batch** -- its exception goes into its own
    record and the batch continues.
    """
    import queue
    import threading

    q = queue.Queue()
    for j in jobs:
        q.put(j)
    out, lock = {}, threading.Lock()

    def worker(sock):
        env = dict(S0_MACE_SOCKET=sock)
        while True:
            try:
                tag, wd, xyz = q.get_nowait()
            except queue.Empty:
                return
            try:
                rec = run(wd, xyz, timeout_s=timeout_s, env=env, **kwargs)
            except Exception as exc:                 # record it all, swallow nothing
                import traceback
                rec = dict(workdir=str(wd), ok=False,
                           error_type=type(exc).__name__, error=str(exc),
                           traceback=traceback.format_exc())
            rec["tag"] = tag
            rec["socket"] = sock
            with lock:
                out[tag] = rec
                if on_done is not None:
                    on_done(tag, rec, len(out), len(jobs))
            q.task_done()

    threads = [threading.Thread(target=worker, args=(s,), daemon=True)
               for s in sockets]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out

def read_input_settings(toml_path):
    """Read back the settings actually used from an `input.toml` that has already been
    written.

    **Why this is needed**: before reusing a working directory that has already run, one
    must confirm the settings it used then are the settings wanted now. Reusing without
    checking means mixing the results of two different settings into one data set --
    **and the product does not show it**.
    """
    toml_path = Path(toml_path)
    if not toml_path.exists():
        return None
    out = {}
    methods = []
    for line in toml_path.read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip()
        if "=" not in s:
            continue
        k, v = (x.strip() for x in s.split("=", 1))
        v = v.strip('"').strip("'")
        if k in ("runtype", "optlev", "refine", "threads", "input", "shake",
                 "tstep", "hmass", "calcspace"):
            out[k] = v
        elif k == "method":
            methods.append(v)
    out["methods"] = methods
    # The workhorse is the FIRST [[calculation.level]] method. It changes the
    # sampling itself, so it belongs in the reuse comparison (defect 57: reusing a
    # scratch directory under different settings silently mixes two datasets and
    # the products do not show it).
    out["workhorse"] = methods[0] if methods else None
    out.setdefault("refine", None)          # with no quality level there is no refine in
                                            # input.toml at all
    out.setdefault("shake", None)           # with no [dynamics] written it is CREST's default
    out.setdefault("tstep", None)           # same: absent means CREST's own 5.0 fs
    return out


#: What CREST uses when `hmass` is absent from [dynamics]. MEASURED on 3.0.2, not read
#: off the source: a plain MD with this module's [dynamics] block prints
#: `hydrogen mass /u   :   2.00000`. (The source is genuinely misleading here --
#: confparse2.f90:202 sets env%hmass = 5.0d0 and parse_calcdata.f90:1192 falls back to
#: 1.00794 -- and reading it instead of running it gives the wrong answer twice over.)
#:
#: Used ONLY to interpret directories written before hmass was written explicitly.
#: Pin it to the CREST version it was measured on if that version ever changes.
CREST_DEFAULT_HMASS_AMU = 2.0
CREST_DEFAULT_HMASS_MEASURED_ON = "crest 3.0.2"


def effective_hmass(recorded):
    """(value, source) for a directory's hydrogen mass.

    Returns the written value when the toml has one, and CREST's measured default when it
    does not -- with `source` saying which, because a value that was assumed and a value
    that was read must never look the same in a product (defect 57).
    """
    if recorded and recorded.get("hmass") not in (None, "", "none", "None"):
        return float(recorded["hmass"]), "read from input.toml"
    return (CREST_DEFAULT_HMASS_AMU,
            "not written in input.toml; CREST default, measured as {} on {}".format(
                CREST_DEFAULT_HMASS_AMU, CREST_DEFAULT_HMASS_MEASURED_ON))


def settings_match(recorded, wanted):
    """The reuse criterion: compare only the keys that **change the result**. The thread
    count does not count (it changes only the speed)."""
    if not recorded:
        return False, "the directory has no input.toml"
    # The timestep is in this list because it changes the SAMPLING, not the speed:
    # measured 2026-09-03, gfn2 + shake=1 aborts 29 metadynamics runs at 5.0 fs and
    # none at 2.0 fs. Reusing a 2 fs scratch directory under 5 fs settings would mix
    # two datasets with nothing in the products to show it (defect 57).
    # hmass is compared through effective_hmass, so a directory written before the value
    # was written explicitly still matches: it has no line, and no line means 2.0.
    for k in ("runtype", "optlev", "refine", "shake", "workhorse", "tstep", "hmass"):
        if k == "hmass":
            a = effective_hmass(recorded)[0]
            b = float(wanted.get("hmass", CREST_DEFAULT_HMASS_AMU) or
                      CREST_DEFAULT_HMASS_AMU)
            if a != b:
                return False, ("hmass differs: the directory used {!r}, now wanted "
                               "{!r}".format(a, b))
            continue
        a = recorded.get(k)
        b = wanted.get(k)
        a = None if a in ("", "none", "None") else a
        b = None if b in ("", "none", "None") else b
        # `shake` and `tstep` come back from the toml as strings and go in as
        # numbers; compare them as numbers so "2" and 2 are not called different.
        if k in ("shake", "tstep", "hmass") and a is not None and b is not None:
            try:
                a, b = float(a), float(b)
            except (TypeError, ValueError):
                pass
        if a != b:
            return False, "{} differs: the directory has {!r}, now wanted {!r}".format(
                k, a, b)
    return True, "match"
