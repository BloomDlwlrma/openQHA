"""GROMACS as an independent implementation of the covariance spectrum. AN EXTENSION.

Branch B's number does not depend on this module. It is one of two independent checks on
the superposition and the covariance spectrum, and since 2026-09-05 it is the OPTIONAL
one: MDAnalysis agreed with `openqha/qha.py` to 8.0e-09 kcal/mol on the same 25 ps
trajectory where `gmx covar -mwa` agreed to 1.27e-04, and it needs no external binary and
no PATH.

What GROMACS still uniquely provides is a check by a program that shares NO PYTHON with
openQHA -- not numpy, not scipy, not our linear algebra. That is worth having and it is
not worth blocking a production run on, which is precisely the definition of an extension
here. See `openqha/capabilities.py` for the contract: locally a missing extension is
skipped, and a run that DECLARES it via `S0_REQUIRE_CAPS` fails instead.

Install it with:

    conda install -c conda-forge gromacs        # note: its activation hook fails under `set -u`

`gmx_binary()` also finds it in a sibling conda environment, because acceptance criterion
2 once spent a session reporting "no comparison produced" for no reason but which shell
was active.

The original module documentation follows unchanged; every measurement in it still holds.

GROMACS as the independent implementation of the covariance spectrum (branch B).

Why GROMACS is here, and what it is not
---------------------------------------
Stage 0's potential is MACE-OFF23_medium and its trajectories are driven by ASE.
**GROMACS never evaluates a single force in this repo.** It appears in exactly one
place: given the same trajectory, `gmx covar -mwa` produces an INDEPENDENT mass-weighted
covariance spectrum, and the difference against `openqha/qha.py` is branch B acceptance
criterion 2.

The `.tpr` is therefore only a carrier of atom names and MASSES -- `-mwa` needs masses
and `-fit` needs a reference structure. Bonded and non-bonded parameters in the topology
written here are never evaluated. This has to be said in the product as well, or a reader
will assume the thermodynamics sits on some classical force field, which is exactly the
reading that would dismantle the level consistency of the protocol.

Which of the two is the reference -- and the measurement that changed the answer
-------------------------------------------------------------------------------
An earlier design made GROMACS the reference and `openqha/qha.py` the cross-check, on the
sound argument that a widely used implementation should not be checked against one
written this week.

**That arrangement does not survive contact with the tool.** `gmx anaeig -entropy`,
measured on the installed binary on 2026-09-03, refuses mass-weighted eigenvalues, uses
a formula that expects them anyway, and drops the six softest modes instead of the six
rigid ones -- a factor of 452 on a synthetic case whose answer is known in closed form.
The three findings, the upstream source lines behind them and the reproduction are in
`anaeig`'s docstring, and the tool is still run, with its output recorded as evidence and
flagged `is_reference=False`.

So the independence is split rather than lost:

  * the SPECTRUM (superposition, mass weighting, diagonalisation -- the part that could
    plausibly be wrong) is checked against `gmx covar -mwa`, which is independent and
    works;
  * the ENTROPY SUM on top of it is closed form, and `qha.harmonic_limit_check` verifies
    it against the analytic answer to machine precision (criterion 3).

No single command covers the whole chain any more. Every step still has an independent
check, and the report says which is which.

The switch that silently changes the answer
-------------------------------------------
`gmx covar -mwa` defaults to **no**. Confirmed on the installed binary (GROMACS
2026.3-conda_forge): `-[no]mwa (no)`. Every step of the quasi-harmonic derivation stands
on the MASS-WEIGHTED covariance, and without `-mwa` the tool still returns a number, with
no error and no warning, and that number is wrong. `covar()` below therefore has no way
to turn mass weighting off; `covar_plain()` exists only to feed the defective `anaeig`
the unweighted input it demands, and no repo number is computed from its output.

Trajectory format
-----------------
Frames are written as GROMOS-96 `.g96`, which GROMACS reads as a trajectory and which
stores coordinates as %15.9f in nm, i.e. 1e-8 angstrom. `.gro` was rejected: it stores 3
decimals in nm = 0.01 angstrom, and the mass-weighted amplitude of a C-H stretch at 300 K
is about 0.03 angstrom, so `.gro` quantisation would land squarely on the quantity being
measured.
"""
import os
import shutil
import sys
import subprocess
from pathlib import Path

import numpy as np

from ..thermochem import thermo

#: nm per angstrom. GROMACS works in nm throughout; this module converts at the edges
#: and nowhere else.
NM_PER_A = 0.1

#: GROMACS reports mass-weighted covariance eigenvalues in amu*nm^2, `openqha/qha.py` in
#: amu*angstrom^2. lambda[amu A^2] = 100 * lambda[amu nm^2].
AMU_A2_PER_AMU_NM2 = 100.0

#: J/(mol K) -> kcal/(mol K). `gmx anaeig -entropy` prints J/mol K.
KCAL_PER_J = 1.0 / 4184.0

#: Rigid-body modes to skip in `gmx anaeig -entropy`. 6 corresponds to a rot+trans fit;
#: it is passed explicitly even though it is also the upstream default, so that the value
#: is visible to a reader and checkable against how the trajectory was actually fitted.
NEVSKIP_ROT_TRANS = 6


def _sibling_env_binaries():
    """`gmx` in a SIBLING conda environment, if this one has none.

    GROMACS was installed into `qm9fe` while branch B runs in `openqha`, so acceptance
    criterion 2 came back "no comparison produced" -- a FAIL that says nothing about the
    science and everything about which shell was active. A criterion that fails for a
    reason it does not measure teaches people to ignore it.

    Search order still puts S0_GMX_BIN and PATH first, so an explicit choice always wins
    and this only fires when the alternative is not running the cross-check at all. Which
    binary was used is reported by `version()` and lands in the product, so a run that
    reached across environments says so.
    """
    here = Path(sys.prefix)
    envs = here.parent if here.parent.name == "envs" else here / "envs"
    found = []
    try:
        candidates = sorted(envs.iterdir())
    except OSError:
        return found
    for env in candidates:
        for name in ("gmx", "gmx_mpi"):
            exe = env / "bin" / name
            if exe.is_file() and os.access(str(exe), os.X_OK):
                found.append(str(exe))
    return found


def gmx_binary():
    """The GROMACS binary: S0_GMX_BIN, then PATH, then a sibling conda environment."""
    exe = (os.environ.get("S0_GMX_BIN") or shutil.which("gmx")
           or shutil.which("gmx_mpi"))
    if not exe:
        siblings = _sibling_env_binaries()
        exe = siblings[0] if siblings else None
    if not exe:
        raise RuntimeError(
            "no GROMACS binary. Set S0_GMX_BIN or put gmx on PATH.\n"
            "This repo installs it with `conda install -c conda-forge gromacs` into the "
            "same environment as the rest of the chain.\n"
            "NOTE: sourcing GMXRC under `set -u` breaks the conda activation hook "
            "-- never add `set -u` to hpc/env/common.sh.")
    return exe


def run_gmx(args, workdir, stdin_text="0\n0\n0\n", timeout_s=1800, check=True):
    """Run one gmx subcommand, capturing everything it said.

    Group selections are fed on stdin rather than through an index file: every tool used
    here analyses the whole system, and group 0 is System. Extra lines are harmless.
    """
    cmd = [gmx_binary(), "-quiet", "-nobackup"] + [str(a) for a in args]
    proc = subprocess.run(cmd, cwd=str(workdir), input=stdin_text, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout_s)
    rec = dict(command=" ".join(cmd), returncode=proc.returncode,
               output=proc.stdout or "")
    if check and proc.returncode != 0:
        raise RuntimeError("{} failed with code {}:\n{}".format(
            rec["command"], proc.returncode, rec["output"][-4000:]))
    return rec


# ======================================================================================
# Writers. Everything GROMACS needs, and nothing it does not.
# ======================================================================================
def _atom_names(symbols):
    """Unique, GROMACS-legal atom names: element symbol plus a running index."""
    return ["{}{}".format(s, i + 1)[:5] for i, s in enumerate(symbols)]


#: Padding around the molecule, in nm, when a box has to be declared. See
#: `box_and_shift` for why a non-periodic study ends up declaring a box at all.
BOX_PADDING_NM = 2.0


def box_and_shift(frames_A, padding_nm=BOX_PADDING_NM):
    """A box big enough that periodicity cannot reach the molecule, and the shift into it.

    Branch B studies an ISOLATED molecule and the protocol specification says `pbc = no`.
    GROMACS 2026 removed the group cutoff scheme, and the Verlet scheme it left behind
    refuses `pbc = no` outright ("With Verlet lists only full pbc or pbc=xy with walls is
    supported"), so `grompp` cannot produce a non-periodic `.tpr` at all.

    The box declared here is therefore a GROMACS formality and not a physical setting.
    It is at least 4 nm larger than the molecule in every direction, and the molecule is
    centred in it, so no atom is ever within a cutoff of an image or a boundary. Nothing
    is evaluated on this topology in any case (see the identity note at the top of this
    module) -- the box exists so that `grompp` will emit a file carrying the masses.

    Returns (box_nm, shift_nm), both in nm, to be applied identically to every file that
    goes into one GROMACS invocation. Applying different shifts to `-s` and `-f` would
    still give the right covariance -- translation is fitted away -- but it would leave
    two files that disagree about where the molecule is, which is the kind of thing that
    is discovered much later and by accident.
    """
    x = np.asarray(frames_A, dtype=float).reshape(-1, 3) * NM_PER_A
    span = x.max(axis=0) - x.min(axis=0)
    box = span + 2.0 * padding_nm
    shift = 0.5 * box - 0.5 * (x.max(axis=0) + x.min(axis=0))
    return box, shift


def write_gro(path, symbols, positions_A, title="openQHA branch B reference structure",
              box_nm=None, shift_nm=None, padding_nm=BOX_PADDING_NM):
    """Single conformation, nm, GROMACS `.gro`.

    Used only as `grompp -c`, i.e. to give the topology a geometry. The production
    coordinates travel in the `.g96` trajectory, which does not have `.gro`'s 0.01
    angstrom quantisation.
    """
    x = np.asarray(positions_A, dtype=float) * NM_PER_A
    names = _atom_names(symbols)
    if box_nm is None or shift_nm is None:
        box_nm, shift_nm = box_and_shift(np.asarray(positions_A, dtype=float)[None],
                                         padding_nm)
    box = np.asarray(box_nm, dtype=float)
    x = x + np.asarray(shift_nm, dtype=float)[None, :]
    lines = [title, "{:5d}".format(len(symbols))]
    for i, (nm, r) in enumerate(zip(names, x)):
        lines.append("{:5d}{:<5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}".format(
            1, "MOL", nm, i + 1, r[0], r[1], r[2]))
    lines.append("{:10.5f}{:10.5f}{:10.5f}".format(*box))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(path)


def write_top(path, symbols, masses, molname="MOL"):
    """Minimal topology: correct names and MASSES, zero interactions.

    Read the identity note at the top of this module before changing anything here. The
    sigma, epsilon and charge columns are zero because nothing in this branch evaluates
    them; `grompp` needs them to exist, and `gmx covar -mwa` needs the mass column, which
    is the only physically meaningful number in the file.
    """
    names = _atom_names(symbols)
    types = []
    seen = {}
    for s in symbols:
        if s not in seen:
            seen[s] = "Q{}".format(s.upper())
        types.append(seen[s])
    from ase.data import atomic_numbers
    lines = [
        "; openQHA branch B -- a MASS CARRIER, not a force field.",
        "; GROMACS evaluates no energy in this branch; it only diagonalises the position",
        "; covariance of a trajectory produced by ASE + MACE. Every interaction",
        "; parameter below is zero and is never used. See openqha/gmx_io.py.",
        "",
        "[ defaults ]",
        "; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ",
        "  1        2          no         1.0      1.0",
        "",
        "[ atomtypes ]",
        "; name  at.num  mass        charge  ptype  sigma  epsilon",
    ]
    for s, t in seen.items():
        m = float(np.asarray(masses)[[i for i, y in enumerate(symbols) if y == s][0]])
        lines.append("  {:<6s} {:<6d} {:>10.5f}  0.000   A      0.0    0.0".format(
            t, int(atomic_numbers[s]), m))
    lines += ["", "[ moleculetype ]", "; name  nrexcl", "  {}    0".format(molname), "",
              "[ atoms ]", "; nr  type  resnr  residue  atom  cgnr  charge  mass"]
    for i, (nm, t, m) in enumerate(zip(names, types, np.asarray(masses, dtype=float))):
        lines.append("  {:<4d} {:<6s} {:<5d} {:<7s} {:<5s} {:<4d} 0.000 {:>10.5f}".format(
            i + 1, t, 1, "MOL", nm, i + 1, float(m)))
    lines += ["", "[ system ]", "openQHA branch B analysis carrier", "",
              "[ molecules ]", "{} 1".format(molname), ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return Path(path)


def write_grompp_mdp(path):
    """An `.mdp` whose only job is to let `grompp` emit a `.tpr`. It integrates nothing.

    This is NOT the branch B protocol specification. That is
    configs/mdp/s0_qha_production.mdp, whose header states the three prohibitions before
    it states any parameter. This file has `nsteps = 0` and exists so `gmx covar` has
    masses to weight with.
    """
    Path(path).write_text("\n".join([
        "; openQHA branch B -- grompp input for a zero-step .tpr (a mass carrier).",
        "; The protocol specification is configs/mdp/s0_qha_production.mdp, and this",
        "; file is NOT it: nothing here is ever integrated or evaluated.",
        ";",
        "; pbc = xyz although the system is an isolated molecule: GROMACS 2026 dropped",
        "; the group cutoff scheme, and Verlet rejects pbc = no outright. The box is at",
        "; least 4 nm larger than the molecule and the molecule is centred in it (see",
        "; box_and_shift), so no image is ever within a cutoff -- and no interaction is",
        "; evaluated in the first place.",
        "integrator               = md",
        "nsteps                   = 0",
        "dt                       = 0.001",
        "continuation             = yes",
        "cutoff-scheme            = Verlet",
        "verlet-buffer-tolerance  = -1",
        "nstlist                  = 10",
        "pbc                      = xyz",
        "rlist                    = 1.0",
        "rvdw                     = 1.0",
        "rcoulomb                 = 1.0",
        "coulombtype              = Cut-off",
        "vdwtype                  = Cut-off",
        "constraints              = none",
        "tcoupl                   = no",
        "pcoupl                   = no",
        "gen-vel                  = no",
        "",
    ]), encoding="utf-8")
    return Path(path)


def write_g96_trajectory(path, symbols, frames_A, times_ps=None, box_nm=None,
                         shift_nm=None):
    """Multi-frame GROMOS-96 trajectory, nm, %15.9f (about 1e-8 angstrom).

    Format resolution matters here in a way it usually does not: the quantity being
    measured IS the fluctuation. `.gro`'s 0.01 angstrom grid is the same size as the
    amplitude of the stiffest modes, so it would bias exactly the eigenvalues that set
    the high-frequency end of the spectrum.
    """
    frames = np.asarray(frames_A, dtype=float) * NM_PER_A
    if frames.ndim != 3 or frames.shape[1] != len(symbols):
        raise ValueError("frames must be (n_frames, n_atoms, 3) matching symbols; "
                         "got {}".format(frames.shape))
    names = _atom_names(symbols)
    if box_nm is None or shift_nm is None:
        box_nm, shift_nm = box_and_shift(frames_A)
    frames = frames + np.asarray(shift_nm, dtype=float)[None, None, :]
    times = (np.arange(len(frames), dtype=float) if times_ps is None
             else np.asarray(times_ps, dtype=float))

    out = ["TITLE", " openQHA branch B trajectory (ASE + MACE; GROMACS analyses only)",
           "END"]
    for k, fr in enumerate(frames):
        out += ["TIMESTEP", "{:15d}{:15.9f}".format(k, float(times[k])), "END",
                "POSITION"]
        for i, r in enumerate(fr):
            out.append("{:5d} {:<5s} {:<5s}{:7d}{:15.9f}{:15.9f}{:15.9f}".format(
                1, "MOL", names[i], i + 1, r[0], r[1], r[2]))
        out += ["END", "BOX",
                "{:15.9f}{:15.9f}{:15.9f}".format(*[float(b) for b in box_nm]), "END"]
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")
    return Path(path)


def parse_xvg(path):
    """Columns of an `.xvg`, comments dropped. Returns (array, header lines)."""
    rows, head = [], []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        if s[0] in "#@":
            head.append(s)
            continue
        rows.append([float(v) for v in s.split()])
    return np.array(rows, dtype=float), head


# ======================================================================================
# The three GROMACS steps: grompp -> covar -> anaeig
# ======================================================================================
def grompp(workdir, gro="conf.gro", top="topol.top", mdp="analysis.mdp",
           tpr="topol.tpr"):
    """Build the `.tpr` mass carrier. `-maxwarn 2`: a topology with no interactions is
    exactly the thing grompp warns about, and it is intentional here."""
    return run_gmx(["grompp", "-f", mdp, "-c", gro, "-p", top, "-o", tpr,
                    "-maxwarn", "2"], workdir)


def covar(workdir, tpr="topol.tpr", traj="traj.g96", fit=True,
          eigenvalues="eigenvalues.xvg", eigenvectors="eigenvectors.trr",
          average="average.pdb"):
    """`gmx covar` WITH mass weighting. There is no argument to turn `-mwa` off.

    `-fit` is left at its upstream default (yes). If the trajectory was already
    superimposed upstream the second fit is idempotent and harmless -- but it happens,
    and a later reader must be able to see that it happened rather than conclude a step
    is missing and add a third one.
    """
    args = ["covar", "-s", tpr, "-f", traj, "-mwa",
            "-o", eigenvalues, "-v", eigenvectors, "-av", average]
    args += ["-fit"] if fit else ["-nofit"]
    return run_gmx(args, workdir)


def covar_plain(workdir, tpr="topol.tpr", traj="traj.g96", fit=True,
                eigenvalues="eigenvalues_nomwa.xvg",
                eigenvectors="eigenvectors_nomwa.trr", average="average_nomwa.pdb"):
    """`gmx covar` WITHOUT mass weighting -- used only to demonstrate the anaeig defect.

    Its eigenvalues are in nm^2 and carry no mass, so they are NOT quasi-harmonic
    eigenvalues and no number in this repo is computed from them. It exists because
    `gmx anaeig -entropy` refuses mass-weighted input, and the only way to show what that
    tool then does is to give it what it asks for. See `anaeig`.
    """
    args = ["covar", "-s", tpr, "-f", traj, "-nomwa",
            "-o", eigenvalues, "-v", eigenvectors, "-av", average]
    args += ["-fit"] if fit else ["-nofit"]
    return run_gmx(args, workdir)


def anaeig(workdir, eigenvectors="eigenvectors_nomwa.trr",
           eigenvalues="eigenvalues_nomwa.xvg", temperature_K=thermo.T_REF,
           nevskip=NEVSKIP_ROT_TRANS, linear=False, check=False):
    """`gmx anaeig -entropy`. **MEASURED DEFECTIVE for this use. Not a reference.**

    `gmx anaeig -entropy` was originally meant to be the reference implementation of
    branch B, with agreement as acceptance criterion 2. That is not possible, for three
    reasons measured on the installed binary (GROMACS 2026.3-conda_forge) on
    2026-09-03 and confirmed against upstream source:

      1. **It refuses mass-weighted eigenvalues.** `gmx covar -mwa` followed by
         `gmx anaeig -entropy` aborts with "Can not calculate entropies from
         mass-weighted eigenvalues, redo the analysis without mass-weighting". So the
         `-mwa` mass weighting -- which is correct, and without which the eigenvalues are
         not quasi-harmonic eigenvalues at all -- is incompatible with this tool by
         construction.

      2. **Its formula nevertheless expects mass-weighted eigenvalues.** Upstream
         `calcSchlitterEntropy` computes `1 + kteh * eigval[i] * evcorr` with
         `evcorr = c_nano^2 * c_amu`, i.e. it converts as though the eigenvalue were in
         amu*nm^2. Fed the nm^2 values the tool insists on, the mass never enters.

      3. **It skips the six LARGEST eigenvalues, and ignores `-nevskip`.** Upstream loops
         `for (i = nskip; i < eigval.ssize(); i++)` with `nskip = 6`, which is correct for
         `gmx nmeig`, where eigenvalues ascend and the first six are the rigid modes. But
         `gmx covar` writes eigenvalues in DESCENDING order, so the six that get dropped
         are the six softest, largest-amplitude modes -- precisely the ones that dominate
         the entropy. `-nevskip` is parsed and then never used in that code path.

    Reproduced exactly: for a synthetic 5-atom trajectory the tool printed a Schlitter
    entropy of 0.0436806 J/(mol K), and evaluating 0.5*R*sum ln(1 + kteh*evcorr*lambda)
    over the descending eigenvalue list with the first six dropped gives 0.04368056. The
    physically correct value for the same trajectory is 19.74 J/(mol K) -- a factor of
    452, not a rounding difference.

    So this function is kept, and it is called, but its output is recorded as EVIDENCE and
    never as a reference value. `check=False` by default because case 1 above is a hard
    abort. What survives of criterion 2 is the comparison against `gmx covar -mwa`, which
    is the independent implementation of the part that could plausibly be wrong: the
    superposition, the mass weighting and the diagonalisation. The entropy sum itself is
    closed form and is verified analytically to machine precision by
    `qha.harmonic_limit_check` (criterion 3).
    """
    args = ["anaeig", "-v", eigenvectors, "-eig", eigenvalues, "-entropy",
            "-temp", "{:.5f}".format(float(temperature_K)),
            "-nevskip", str(int(nevskip))]
    if linear:
        args.append("-linacc")
    return run_gmx(args, workdir, check=check)


def parse_entropy(output_text):
    """Pull the two entropies out of what `gmx anaeig` printed, in kcal/(mol K).

    Upstream prints, in J/(mol K):
        The Entropy due to the Schlitter formula is <x> J/mol K
        The Entropy due to the Quasi Harmonic approximation is <y> J/mol K
    Both are returned as None when absent rather than defaulted, so a parsing failure
    cannot be mistaken for a measurement.
    """
    schlitter = quasi = None
    for line in output_text.splitlines():
        low = line.lower()
        if "entropy" not in low:
            continue
        tokens = [t for t in line.replace(",", " ").split()
                  if t.replace(".", "", 1).replace("e-", "", 1)
                  .replace("e+", "", 1).lstrip("-").isdigit()]
        if not tokens:
            continue
        try:
            value = float(tokens[-1])
        except ValueError:
            continue
        if "schlitter" in low:
            schlitter = value
        elif "quasi" in low or "harmonic" in low:
            quasi = value
    return dict(
        S_Schlitter_J_per_mol_K=schlitter,
        S_QH_J_per_mol_K=quasi,
        S_Schlitter_kcal_per_K=(None if schlitter is None else schlitter * KCAL_PER_J),
        S_QH_kcal_per_K=(None if quasi is None else quasi * KCAL_PER_J),
        parsed_both=bool(schlitter is not None and quasi is not None),
    )


def cross_check(frames_A, symbols, masses, workdir, temperature_K=thermo.T_REF,
                nevskip=NEVSKIP_ROT_TRANS, keep_files=True, run_anaeig=True):
    """Branch B acceptance criterion 2, end to end, on one trajectory.

    What is being cross-checked, and what cannot be
    -----------------------------------------------
    `gmx anaeig -entropy` was supposed to be the independent implementation of the whole
    chain. It is not usable: it refuses mass-weighted eigenvalues, its
    formula nevertheless expects them, and it drops the six softest modes instead of the
    six rigid ones. All three are measured and reproduced in `anaeig`'s docstring. Its
    output is still collected here, as evidence, and marked `is_reference=False`.

    What remains is stronger than nothing and weaker than the original criterion:
    `gmx covar -mwa` is a genuinely independent implementation of the superposition, the
    mass weighting and the diagonalisation -- which is the part of this branch that could
    plausibly be wrong. The entropy sum on top of the spectrum is closed form and is
    checked to machine precision against the analytic answer by `qha.harmonic_limit_check`
    (criterion 3). Between the two, every step has an independent check; no single
    command covers the whole chain any more, and the report says so.

    Making the two comparable
    -------------------------
    `gmx covar` fits every frame ONCE, to the structure in `-s`. `openqha/qha.py` iterates
    the fit to the mean of the fitted frames. Measured on a synthetic 5-atom trajectory,
    that difference alone changes the eigenvalue sum by 1e-4 relative and individual soft
    modes by up to 2 percent -- larger than anything else in this comparison. So the `-s`
    structure written here is OUR CONVERGED MEAN, not frame 0, and then the two fits agree
    by construction and the residual difference measures the implementations rather than
    the choice of reference. `fit_protocol_difference` reports the size of the effect that
    was removed, because a difference that has been arranged away still has to be visible.

    GROMACS does not project the rigid subspace out; it relies on the fit. Our production
    number does project (Eckart, the same code the Hessian route uses). The primary
    comparison is therefore against our UNPROJECTED spectrum, like for like, and the
    projection's own effect is reported separately as `projection_effect_*`.
    """
    from ..quasi_harmonic import qha

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames_A, dtype=float)
    masses = np.asarray(masses, dtype=float)

    # ---- our side first: it supplies the reference structure GROMACS will fit to -----
    ours = qha.analyse(frames, masses, temperature_K)
    mean_structure = np.asarray(ours["mean_structure_A"], dtype=float)

    box_nm, shift_nm = box_and_shift(np.concatenate([frames, mean_structure[None]]))
    write_gro(work / "conf.gro", symbols, mean_structure, box_nm=box_nm,
              shift_nm=shift_nm,
              title="openQHA branch B: the CONVERGED MEAN structure, so that gmx covar's "
                    "single fit matches ours")
    write_top(work / "topol.top", symbols, masses)
    write_grompp_mdp(work / "analysis.mdp")
    write_g96_trajectory(work / "traj.g96", symbols, frames, box_nm=box_nm,
                         shift_nm=shift_nm)

    steps = [grompp(work), covar(work)]
    eig, _head = parse_xvg(work / "eigenvalues.xvg")
    lam_gmx_nm2 = eig[:, 1] if eig.ndim == 2 and eig.shape[1] >= 2 else np.array([])
    lam_gmx = lam_gmx_nm2 * AMU_A2_PER_AMU_NM2          # amu * angstrom^2, descending

    lam_proj = np.asarray(ours["spectrum"]["eigenvalues_amu_A2"], dtype=float)
    lam_unproj_all = np.asarray(
        ours["spectrum"]["eigenvalues_unprojected_ascending_amu_A2"], dtype=float)[::-1]
    n_keep = len(lam_proj)
    lam_unproj = lam_unproj_all[:n_keep]
    lam_gmx_kept = lam_gmx[:n_keep]

    def rel(a, b):
        if len(a) == 0 or len(b) == 0:
            return None
        n = min(len(a), len(b))
        return float(np.abs((a[:n] - b[:n]) / np.maximum(np.abs(b[:n]), 1e-30)).max())

    ent_from_gmx = qha.entropy(lam_gmx_kept, temperature_K) if len(lam_gmx_kept) else None

    # ---- the fit-protocol effect, measured rather than asserted to be small -----------
    fitted_once, mean_once, _rec_once = superimpose_once(frames, masses)
    cov_once = qha.mass_weighted_covariance(fitted_once, masses, mean_once)
    lam_once = np.sort(np.linalg.eigvalsh(cov_once))[::-1][:n_keep]

    # ---- and the anaeig evidence, which is not a reference ----------------------------
    anaeig_record = None
    if run_anaeig:
        steps.append(covar_plain(work))
        step = anaeig(work, temperature_K=temperature_K, nevskip=nevskip)
        steps.append(step)
        parsed = parse_entropy(step["output"])
        anaeig_record = dict(
            is_reference=False,
            why_not=("measured defective for this use: refuses mass weighting, its "
                     "formula expects it anyway, and it drops the six softest modes "
                     "instead of the six rigid ones. See gmx_io.anaeig."),
            returncode=step["returncode"],
            eigenvalues_were_mass_weighted=False,
            parsed=parsed,
            output_tail=step["output"][-1500:],
        )

    ts_ours = ours["entropy"]["TS_QH_kcal"]
    out = dict(
        workdir=str(work),
        gmx_binary=gmx_binary(),
        gmx_version=version(),
        our_precision="numpy float64",
        gmx_steps=[dict(command=s["command"], returncode=s["returncode"]) for s in steps],
        temperature_K=float(temperature_K),
        n_frames=int(len(frames)),
        reference_structure_for_fit="our converged mean structure (not frame 0)",

        # -- the comparison that is actually independent ------------------------------
        n_eigenvalues_gmx=int(len(lam_gmx)),
        n_eigenvalues_ours=int(n_keep),
        eigenvalues_gmx_amu_A2=[float(x) for x in lam_gmx],
        eigenvalue_max_relative_difference_vs_unprojected=rel(lam_gmx_kept, lam_unproj),
        eigenvalue_max_relative_difference_vs_projected=rel(lam_gmx_kept, lam_proj),
        eigenvalue_sum_gmx_amu_A2=float(lam_gmx_kept.sum()) if len(lam_gmx_kept) else None,
        eigenvalue_sum_ours_unprojected_amu_A2=float(lam_unproj.sum()),
        eigenvalue_sum_relative_difference=(
            float(abs(lam_gmx_kept.sum() - lam_unproj.sum()) / lam_unproj.sum())
            if len(lam_gmx_kept) else None),

        # -- what our own entropy formula makes of GROMACS' spectrum -------------------
        TS_QH_ours_kcal=ts_ours,
        TS_Schlitter_ours_kcal=ours["entropy"]["TS_Schlitter_kcal"],
        TS_QH_from_gmx_eigenvalues_kcal=(None if ent_from_gmx is None
                                         else ent_from_gmx["TS_QH_kcal"]),
        TS_Schlitter_from_gmx_eigenvalues_kcal=(None if ent_from_gmx is None
                                                else ent_from_gmx["TS_Schlitter_kcal"]),
        TS_QH_difference_kcal=(None if ent_from_gmx is None
                               else float(ts_ours - ent_from_gmx["TS_QH_kcal"])),
        TS_Schlitter_difference_kcal=(
            None if ent_from_gmx is None
            else float(ours["entropy"]["TS_Schlitter_kcal"]
                       - ent_from_gmx["TS_Schlitter_kcal"])),

        # -- the two known, deliberate methodological differences, each with its size ---
        fit_protocol_difference=dict(
            what=("gmx covar fits once to the -s structure; openQHA iterates the fit to "
                  "the mean. Both are run here on the same frames."),
            TS_QH_single_fit_kcal=float(qha.entropy(lam_once, temperature_K)["TS_QH_kcal"]),
            TS_QH_iterated_fit_kcal=float(
                qha.entropy(lam_unproj, temperature_K)["TS_QH_kcal"]),
            eigenvalue_sum_single_fit_amu_A2=float(lam_once.sum()),
            eigenvalue_sum_iterated_fit_amu_A2=float(lam_unproj.sum()),
        ),
        projection_effect=dict(
            what=("GROMACS relies on the fit alone; openQHA also projects the Eckart "
                  "rigid subspace out, using hessian.rigid_body_vectors."),
            TS_QH_unprojected_kcal=float(
                qha.entropy(lam_unproj, temperature_K)["TS_QH_kcal"]),
            TS_QH_projected_kcal=ts_ours,
        ),

        anaeig_evidence=anaeig_record,
        # look here first when the eigenvalues disagree
        rigid_to_first_vibrational_ratio=ours["spectrum"][
            "rigid_to_first_vibrational_ratio"],
        rank_check=ours["rank_check"],
        ours=ours,
    )
    if not keep_files:
        shutil.rmtree(work, ignore_errors=True)
    return out


def superimpose_once(frames_A, masses):
    """One fit to frame 0 -- GROMACS' protocol, reproduced so its effect can be measured.

    Not used for any production number. It exists because `cross_check` has to be able to
    say how much of any disagreement with `gmx covar` comes from the fit protocol rather
    than from the arithmetic, and the only honest way to say that is to run both.
    """
    from ..quasi_harmonic import qha
    return qha.superimpose(frames_A, masses, max_iterations=1, tolerance_A=0.0)


def version():
    """Version and precision of the binary that will actually run.

    Precision is part of the answer, not trivia: the conda-forge package is MIXED
    precision and ships no `gmx_d`, and the smallest covariance eigenvalues are exactly
    where that shows up first. Reporting it next to every comparison turns "is mixed
    precision good enough" from a guess into a measurement that was going to be made
    anyway.
    """
    rec = run_gmx(["--version"], ".", check=False)
    out = dict(raw=rec["output"][:2000], version=None, precision=None)
    for line in rec["output"].splitlines():
        if line.lower().startswith("gromacs version"):
            out["version"] = line.split(":", 1)[-1].strip()
        elif line.lower().startswith("precision"):
            out["precision"] = line.split(":", 1)[-1].strip()
    return out
