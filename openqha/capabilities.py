"""What this installation can actually do, probed rather than assumed.

The contract, taken from ACEsuit/mace's `tests/conftest.py` capability model
(`mace/modules/extensions.py`, `tests/extensions/`, `.github/workflows/ci-extensions.yaml`):

  * locally, work whose capability is unavailable is SKIPPED;
  * a run that DECLARES it provides a capability -- by exporting `S0_REQUIRE_CAPS` --
    FAILS instead of skipping. A job can never again go green while silently skipping the
    thing it exists to test.

That second half is the whole point. openQHA has already been bitten twice by the first
half alone: acceptance criterion 2 reported "no comparison produced" because GROMACS was
installed in a sibling conda environment, and a `--no-gmx` run reported the same thing for
a completely different reason. Both times a criterion went quiet instead of red.

Probes are REAL IMPORTS and REAL EXECUTIONS, never `importlib.util.find_spec` and never
`Path.exists`. MACE's own note on this is exact: "a broken wheel (e.g. missing libmpi)
must read as unavailable". A capability that is present-but-broken is unavailable, and the
only way to know is to use it.

Core against extension
----------------------
A CORE capability is one branch B's production number depends on. An EXTENSION is
something that improves confidence or convenience and whose absence changes no number.
The split is not a matter of taste and it is enforced: `require_core()` raises when a core
capability is missing.

    core        openmm, openmm_torch, openmmtools, mdanalysis
    extension   gromacs, mdtraj, crest, xtb, orca

GROMACS is an EXTENSION as of 2026-09-05. It was branch B's only independent check of the
covariance spectrum until MDAnalysis was measured against the same trajectory and agreed
to 8.0e-09 kcal/mol -- against GROMACS' 1.27e-04 -- while needing no external binary and
no PATH. What GROMACS still uniquely provides is a check by a program that shares no
Python with us at all, which is worth having and is not worth blocking a run on.
"""
import os
import shutil
import subprocess

#: Capabilities branch B's production number depends on.
CORE = ("openmm", "openmm_torch", "openmmtools", "mdanalysis")

#: Capabilities whose absence changes no number. Each says what is lost.
EXTENSIONS = {
    "gromacs": "no `gmx covar -mwa` check of the covariance spectrum by a program that "
               "shares no Python with openQHA. MDAnalysis still provides a mass-weighted "
               "independent superposition (measured agreement 8.0e-09 kcal/mol)",
    "mdtraj": "no mdtraj superposition comparison. Note that mdtraj CANNOT mass-weight, "
              "so it was never usable as agreement -- it differed from us by 2.41 "
              "kcal/mol on a real trajectory. It is kept as a demonstration of what an "
              "unweighted fit costs",
    "crest": "no conformer search (branch A)",
    "xtb": "no GFN2 labels (branch C)",
    "orca": "no RI-MP2 reference labels (branch C)",
    "ase_route": "no ASE production route for branch B; the OpenMM route is unaffected",
}


def _import_works(name):
    """A REAL import. `find_spec` would call a broken build available."""
    try:
        __import__(name)
        return True
    except Exception:                                          # noqa: BLE001
        return False


def _binary_runs(name, env_var=None, args=("--version",)):
    """A REAL execution. On PATH but not runnable is not available."""
    exe = (os.environ.get(env_var) if env_var else None) or shutil.which(name)
    if not exe:
        return False
    try:
        got = subprocess.run([exe] + list(args), capture_output=True, timeout=60)
        return got.returncode == 0
    except Exception:                                          # noqa: BLE001
        return False


def _gromacs():
    """`gmx`, including in a SIBLING conda environment.

    The sibling search is not convenience. Acceptance criterion 2 spent a session
    reporting "no comparison produced" for no reason but which shell was active, and a
    criterion that fails for something it does not measure teaches people to ignore it.
    """
    try:
        from .quasi_harmonic import gmx_io
        gmx_io.gmx_binary()
        return True
    except Exception:                                          # noqa: BLE001
        return False


#: name -> probe. Each probe must be cheap enough to run on every startup.
PROBES = {
    "openmm": lambda: _import_works("openmm"),
    "openmm_torch": lambda: _import_works("openmmtorch"),
    "openmmtools": lambda: _import_works("openmmtools"),
    "mdanalysis": lambda: _import_works("MDAnalysis"),
    "mdtraj": lambda: _import_works("mdtraj"),
    "gromacs": _gromacs,
    "crest": lambda: _binary_runs("crest", "S0_CREST_BIN"),
    "xtb": lambda: _binary_runs("xtb"),
    "orca": lambda: _binary_runs("orca"),
    "ase_route": lambda: _import_works("ase.md.nose_hoover_chain"),
}

ALL = tuple(sorted(PROBES))

_CACHE = {}


def available(name, refresh=False):
    """Is this capability usable HERE, right now."""
    if name not in PROBES:
        raise KeyError("unknown capability {!r}; known: {}".format(name, ", ".join(ALL)))
    if refresh or name not in _CACHE:
        _CACHE[name] = bool(PROBES[name]())
    return _CACHE[name]


def state(refresh=False):
    """Every capability, for the provenance record. Products should carry this."""
    return {name: available(name, refresh=refresh) for name in ALL}


def required_caps():
    """Capabilities this run DECLARES it provides, from `S0_REQUIRE_CAPS`.

    Set it in a CI job, a cluster submission script, or a production run whose result
    would be wrong without the capability. Unknown names are an error rather than a
    typo that quietly guarantees nothing.
    """
    raw = os.environ.get("S0_REQUIRE_CAPS", "")
    caps = {c.strip() for c in raw.replace(";", ",").split(",") if c.strip()}
    unknown = caps - set(ALL)
    if unknown:
        raise ValueError(
            "S0_REQUIRE_CAPS names unknown capabilities: {}. Known: {}".format(
                ", ".join(sorted(unknown)), ", ".join(ALL)))
    return caps


def check_declared(refresh=False):
    """Raise if anything `S0_REQUIRE_CAPS` guarantees is not actually there.

    This is the half of the contract that makes the other half safe. Without it, a job
    that was set up to test GROMACS and then lost GROMACS goes green.
    """
    declared = required_caps()
    broken = sorted(c for c in declared if not available(c, refresh=refresh))
    if broken:
        raise RuntimeError(
            "S0_REQUIRE_CAPS guarantees {} but {} unavailable here. Either install it or "
            "stop declaring it: a run that declares a capability and then skips it is a "
            "run that went green without doing its job.".format(
                ", ".join(broken), "it is" if len(broken) == 1 else "they are"))
    return sorted(declared)


def require_core(names=CORE, refresh=False):
    """Raise unless every CORE capability is present. Extensions are never required."""
    missing = sorted(n for n in names if not available(n, refresh=refresh))
    if missing:
        raise RuntimeError(
            "branch B's OpenMM production route needs {} and {} not available.\n"
            "\n"
            "A current environment carries them: `environment.yml` on a workstation,\n"
            "`environment-cuda.yml` on a CUDA box (the OpenMM stack moved into both on\n"
            "2026-09-05), and `environment-tianhe-gpu.yml` on both Tianhe GPU clusters --\n"
            "an environment missing them was not built from those files:\n"
            "\n"
            "  on a workstation conda env update -f environment.yml --prune\n"
            "  on a CUDA box    conda env update -f environment-cuda.yml --prune\n"
            "  on Tianhe        the GPU environment already carries all three:\n"
            "                     bash install_dependency.sh --tianhe-cuda   # or --tianhe-a\n"
            "                     OPENQHA_ROLE=gpu source hpc/env/tianhe.sh\n"
            "                   (environment-tianhe-gpu.yml lists openmm, openmm-torch\n"
            "                   =*cuda* and openmmtools. If they are missing THERE, the\n"
            "                   environment was not built from that file.)"
            .format(", ".join(missing), "it is" if len(missing) == 1 else "they are"))
    return True


def missing_extension_note(name):
    """What is lost by this extension being absent, in one sentence."""
    return EXTENSIONS.get(name, "unknown extension {!r}".format(name))


def summary():
    """A short human-readable report. Used by check_dependency.py and the drivers."""
    s = state()
    lines = ["core (branch B's number depends on these):"]
    for n in CORE:
        lines.append("  {:<14} {}".format(n, "yes" if s[n] else "NO"))
    lines.append("extensions (absence changes no number):")
    for n in sorted(EXTENSIONS):
        lines.append("  {:<14} {}".format(n, "yes" if s.get(n) else "no"))
        if not s.get(n):
            lines.append("  {:<14}   -> {}".format("", missing_extension_note(n)))
    declared = os.environ.get("S0_REQUIRE_CAPS")
    if declared:
        lines.append("S0_REQUIRE_CAPS = {}  (a missing one of these is an ERROR here, "
                     "not a skip)".format(declared))
    return "\n".join(lines)
