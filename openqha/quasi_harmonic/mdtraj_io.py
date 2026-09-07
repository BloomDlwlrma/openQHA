"""Superposition by mdtraj and MDAnalysis, as independent implementations of the fit.

Branch B's quasi-harmonic entropy is a variance over superimposed configurations, so the
superposition is not a preprocessing step -- it is half the estimator. `openqha/qha.py`
implements it one way (mass-weighted Kabsch, iterated to a converged mean structure) and
this module runs two widely used libraries over the same frames so that the number can be
checked against something that was not written here.

This is the same arrangement as `openqha/gmx_io.py`, which does it with `gmx covar`, and
it is kept for the same reason: an implementation checked only against itself is checked
against nothing.

What differs between the three, and why it is reported rather than hidden
------------------------------------------------------------------------
  * `openqha.qha.superimpose` fits MASS-WEIGHTED and ITERATES the reference to the mean of
    the fitted frames. Fitting to one arbitrary frame leaves a bias whose size depends on
    which frame was picked.
  * `mdtraj.Trajectory.superpose` fits to ONE frame and is NOT mass weighted -- it
    optionally weights by `atom_indices`, not by mass.
  * `MDAnalysis.analysis.align` can fit mass-weighted, which makes it the closer
    comparison of the two, and it is used when it is installed.

So a disagreement here is expected and is a measurement of the FIT PROTOCOL, not of the
entropy algorithm. `cross_check` therefore reports the two contributions separately, the
way `gmx_io.cross_check` reports `fit_protocol_difference`.

What it measured -- the whole difference, attributed
---------------------------------------------------
On the first real production trajectory (acetone, 25 ps, 3125 frames, MACE-OFF23_medium),
running each library both as it comes and with our protocol imposed:

    implementation                  mass-weighted   reference        T*S      difference
    openqha.qha (ours)                   yes        iterated mean   5.799829       --
    MDAnalysis 2.10.0, iterated          yes        iterated mean   5.799829   +8.020e-09
    MDAnalysis 2.10.0, as it comes       yes        one frame       5.828531   +2.870e-02
    mdtraj 1.11.1                        NO         one frame       8.207780   +2.407951

and the recovered SOFTEST MODE, which is where an unweighted fit goes wrong and which the
T*S column hides:

    openqha.qha (ours)                   52.00 cm-1     (T*S 5.7998)
    MDAnalysis 2.10.0, iterated          52.00
    MDAnalysis 2.10.0, as it comes       51.96
    mdtraj 1.11.1                        43.05 cm-1     (T*S 8.2078, difference 2.4080)

For scale, `gmx covar -mwa` -- the other mass-weighted implementation, and the one that
needs an external binary -- agreed with us to 1.27e-04 kcal/mol on this same trajectory.

Read down that column. It decomposes the disagreement into three separate things:

  * NOT MASS-WEIGHTING costs 2.41 kcal/mol -- 2.4 times the whole stage-0 accuracy target,
    from the superposition alone. This is why mdtraj cannot serve as the independent check
    that `gmx covar -mwa` serves: `gmx covar` can be told to mass-weight and
    `mdtraj.Trajectory.superpose` cannot. It is not evidence that mdtraj is wrong; it is
    fitting a different thing.
  * NOT ITERATING THE REFERENCE costs 0.029 kcal/mol. That is the bias `superimpose`'s
    docstring warns about -- fitting to one arbitrary frame leaves a residue whose size
    depends on which frame was picked -- and it is now a number rather than an argument.
  * THE IMPLEMENTATION ITSELF costs 8.0e-09 kcal/mol. Our Kabsch/SVD against MDAnalysis'
    Theobald QCP quaternion, same weighting, same reference protocol, 3125 frames: the
    frequencies agree to 1.2e-04 cm^-1 over 24 modes and the eigenvalues to 1.2e-07
    relative. That residual is float32: MDAnalysis stores positions in single precision,
    and 1e-07 relative is its epsilon.

The third line is the one worth having. Two independent algorithms, written by different
people, agree on this branch's estimator to nine significant figures.

mdtraj also printed, on stderr from its C extension:

    mdtraj/rmsd/src/theobald_rmsd.cpp UNCONVERGED ROTATION MATRIX. RETURNING IDENTITY=300

i.e. for 300 of the 3125 frames its float32 Theobald solver did not converge and returned
the IDENTITY rotation instead of a fit -- silently, as far as the Python API is concerned.
Roughly a tenth of the trajectory was not superimposed at all. Worth knowing before
trusting any RMSD-based analysis of these trajectories.

Where a library is absent `cross_check` says so, rather than quietly comparing against one
of them and calling it agreement.
"""
import numpy as np

from . import qha


def available():
    """Which of the two libraries can actually be used, checked by import."""
    found = {}
    for name in ("mdtraj", "MDAnalysis"):
        try:
            mod = __import__(name)
            found[name] = getattr(mod, "__version__", "?")
        except Exception:                                     # noqa: BLE001
            found[name] = None
    return found


def _topology(symbols):
    """A single-residue topology carrying the right elements, built by hand.

    mdtraj wants a topology before it will hold coordinates. Nothing here is used for
    anything but element identity and atom count -- there are no bonds and no force field,
    exactly as in `gmx_io.write_top`, and for the same reason: this module evaluates no
    energy.
    """
    import mdtraj
    top = mdtraj.Topology()
    chain = top.add_chain()
    residue = top.add_residue("MOL", chain)
    for i, s in enumerate(symbols):
        element = mdtraj.element.get_by_symbol(str(s))
        top.add_atom("{}{}".format(s, i + 1), element, residue)
    return top


def to_trajectory(frames_A, symbols):
    """Frames in angstrom -> an mdtraj Trajectory in nm."""
    import mdtraj
    xyz = np.asarray(frames_A, dtype=np.float32) / 10.0        # angstrom -> nm
    return mdtraj.Trajectory(xyz, _topology(symbols))


def superimpose_mdtraj(frames_A, symbols, reference_index=0):
    """mdtraj's superposition. Returns (fitted frames in angstrom, record)."""
    import mdtraj
    traj = to_trajectory(frames_A, symbols)
    traj.superpose(traj, frame=int(reference_index))
    fitted = np.asarray(traj.xyz, dtype=float) * 10.0
    return fitted, dict(
        implementation="mdtraj.Trajectory.superpose",
        version=getattr(mdtraj, "__version__", "?"),
        mass_weighted=False, iterated=False,
        reference="frame {}".format(reference_index),
        n_frames=int(len(fitted)))


def superimpose_mdanalysis(frames_A, symbols, masses, reference_index=0,
                           max_iterations=1, tolerance_A=1e-8):
    """MDAnalysis' mass-weighted superposition. Returns (fitted frames, record).

    `max_iterations=1` is MDAnalysis as it comes: fit every frame to ONE reference
    structure. Raise it to iterate the reference to the mean of the fitted frames, which
    is what `openqha.qha.superimpose` does -- and which is the only remaining protocol
    difference between the two once mass weighting is switched on. Running it both ways is
    how the comparison stops confounding two differences in one number.

    Verified against the supplied source tree (mdanalysis-develop): `weights="mass"` goes
    through `lib.util.get_weights` to `atoms.masses`, `rotation_matrix` normalises them by
    their mean before QCP ("qcp does NOT divide weights relative to the mean"), and
    `_fit_to` moves both structures to their weighted centre. That is the same fit as
    ours.
    """
    import MDAnalysis
    from MDAnalysis.analysis import align

    frames_A = np.asarray(frames_A, dtype=float)
    n_frames, n_atoms, _ = frames_A.shape
    m = np.asarray(masses, dtype=float)

    u = MDAnalysis.Universe.empty(n_atoms, trajectory=True)
    u.add_TopologyAttr("masses", m)
    u.add_TopologyAttr("names", ["{}{}".format(s, i + 1)
                                 for i, s in enumerate(symbols)])
    ref = MDAnalysis.Universe.empty(n_atoms, trajectory=True)
    ref.add_TopologyAttr("masses", m)
    ref.add_TopologyAttr("names", ["{}{}".format(s, i + 1)
                                   for i, s in enumerate(symbols)])

    reference = frames_A[int(reference_index)].copy()
    w = m / m.sum()
    history = []
    fitted = np.empty_like(frames_A)
    for it in range(int(max_iterations)):
        ref.atoms.positions = reference.astype(np.float32)
        for i, frame in enumerate(frames_A):
            u.atoms.positions = frame.astype(np.float32)
            align.alignto(u.atoms, ref.atoms, weights="mass")
            fitted[i] = u.atoms.positions
        new_reference = fitted.mean(axis=0)
        shift = float(np.sqrt((w[:, None] * (new_reference - reference) ** 2).sum()))
        history.append(shift)
        reference = new_reference
        if shift < tolerance_A:
            break

    return fitted, dict(
        implementation="MDAnalysis.analysis.align.alignto(weights='mass')",
        version=MDAnalysis.__version__,
        module_path=MDAnalysis.__file__,
        mass_weighted=True,
        iterated=bool(max_iterations > 1),
        n_iterations=len(history),
        reference_shift_A=history,
        reference=("frame {}".format(reference_index) if max_iterations == 1
                   else "iterated mean structure"),
        n_frames=int(n_frames))


def _entropy_from_fitted(fitted_A, masses, temperature_K):
    """The rest of the branch B chain, taking someone else's superposition as given."""
    mean = np.asarray(fitted_A, dtype=float).mean(axis=0)
    cov = qha.mass_weighted_covariance(fitted_A, masses, mean_structure=mean)
    spec = qha.spectrum(cov, masses, mean)
    ent = qha.entropy(np.asarray(spec["eigenvalues_amu_A2"], dtype=float), temperature_K)
    return ent, spec


def cross_check(frames_A, masses, symbols, temperature_K=None):
    """Our fit against mdtraj's and MDAnalysis', with the difference attributed.

    Returns a record with one entry per available library plus `ours`. The comparison that
    matters is `TS_difference_kcal`; `note` says what that difference is made of, because
    the fits are deliberately different protocols and pretending otherwise would turn a
    known difference into an unexplained one.
    """
    temperature_K = temperature_K or 298.15
    frames_A = np.asarray(frames_A, dtype=float)
    masses = np.asarray(masses, dtype=float)

    ours = qha.analyse(frames_A, masses, temperature_K)
    record = dict(
        temperature_K=float(temperature_K),
        n_frames=int(len(frames_A)),
        available=available(),
        ours=dict(implementation="openqha.qha.superimpose",
                  mass_weighted=True, iterated=True,
                  reference="iterated mean structure",
                  TS_QH_kcal=ours["entropy"]["TS_QH_kcal"],
                  lowest_frequency_cm_inv=ours["entropy"]["lowest_frequency_cm_inv"],
                  n_nonzero=ours["spectrum"]["n_nonzero_eigenvalues"]),
        comparisons=[])

    trials = (
        ("mdtraj", "mdtraj",
         lambda: superimpose_mdtraj(frames_A, symbols)),
        ("MDAnalysis", "MDAnalysis",
         lambda: superimpose_mdanalysis(frames_A, symbols, masses)),
        # The one that isolates the implementation from the protocol: mass-weighted AND
        # iterated to the mean, which is exactly what openqha.qha.superimpose does.
        ("MDAnalysis (iterated)", "MDAnalysis",
         lambda: superimpose_mdanalysis(frames_A, symbols, masses, max_iterations=10)),
    )
    for label, name, fn in trials:
        if record["available"].get(name) is None:
            record["comparisons"].append(dict(library=label, skipped="not installed"))
            continue
        try:
            fitted, fit_record = fn()
            ent, spec = _entropy_from_fitted(fitted, masses, temperature_K)
            fit_record.update(
                library=label,
                TS_QH_kcal=ent["TS_QH_kcal"],
                lowest_frequency_cm_inv=ent["lowest_frequency_cm_inv"],
                n_nonzero=spec["n_nonzero_eigenvalues"],
                TS_difference_kcal=float(ent["TS_QH_kcal"]
                                         - ours["entropy"]["TS_QH_kcal"]),
                note=("this difference is the FIT PROTOCOL, not the entropy algorithm: "
                      "everything after the superposition is openqha.qha, byte for byte"))
            record["comparisons"].append(fit_record)
        except Exception as exc:                              # noqa: BLE001
            record["comparisons"].append(
                dict(library=label, failed="{}: {}".format(type(exc).__name__, exc)))
    return record
