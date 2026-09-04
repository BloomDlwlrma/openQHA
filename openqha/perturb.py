"""Turn "how wrong are the forces" into **dynamics that can actually be run** -- the
perturbation construction of option B.

**The question it answers**: package 2 measures how far MACE's forces deviate from the
composite reference (some number of eV/A per component). So, **if the forces really are
wrong by that much, how much does the density-of-states free energy move?** Option B
runs two trajectories -- one with MACE, one with "MACE plus a perturbation whose size is
calibrated to the measured error" -- and compares the density-of-states free energy of
each.

**This is a convention, not a measurement, and that has to be said plainly.** The real
force error is **a function of structure, and structured**; the perturbation here is
**random**. So what this method gives is "the **typical** sensitivity of the free energy
when the forces are wrong by this much", not "MACE's free-energy error". Using it as the
former is honest; using it as the latter is wrong.

**Three hard requirements on the perturbation**, and the result means nothing without
any one of them:

1. **Conservative** (the gradient of some potential) -- otherwise the dynamics does not
   conserve energy, the total-energy drift contaminates the density of states, and the
   "free-energy change" being measured has numerical heating mixed into it.
2. **Invariant under translation and rotation** -- otherwise the perturbation becomes an
   external field applying a spurious torque to the molecule, which is exactly what the
   body-fixed velocity step exists to remove.
3. **Non-zero at the minimum, and growing with displacement** -- because the measured
   force error has precisely that shape: already non-zero at the minimum (the two
   potentials have their minima in different places) and growing away from it.

**The simplest construction satisfying all three**: write the perturbation as a
quadratic form in the **deviations of interatomic distances** --

`r_p` is the distance of the `p`-th atom pair and `r_p^0` is that same distance at the
reference minimum. A distance depends only on relative positions, so (2) holds
**exactly**; it is an explicit potential, so (1) holds **exactly**; the `b` term is
non-zero at the minimum and the `A` term grows linearly with displacement, so (3) holds.

**Calibration** (the two parts are calibrated separately, because the measurement itself
comes in two parts):

* `b` is calibrated to **the force deviation at the minimum** -- the "degenerate" number
  from package 2, which measures exactly the difference between the two potentials'
  minima;
* `A` is calibrated to **the increment brought by thermal displacement**,
  `sqrt(max(0, sigma_displaced^2 - sigma_minimum^2))`, that is, the part of the error
  that grows once one leaves the minimum.

Both parts are calibrated by **numerical measurement on real thermally sampled
configurations**, not by an analytic estimate.
"""
import numpy as np

try:                                        # ASE base classes; this module still reads
                                            # without ASE installed
    from ase.calculators.calculator import Calculator, all_changes
except Exception:                           # pragma: no cover
    Calculator, all_changes = object, None


def pair_indices(n_atoms):
    """All atom pairs (i<j), returned as two index arrays."""
    i, j = np.triu_indices(n_atoms, k=1)
    return i, j


def pair_distances(positions, i, j):
    d = positions[i] - positions[j]
    return np.linalg.norm(d, axis=1), d


class PairQuadraticPerturbation:
    """`E_p = sum_p b_p d_p + 1/2 sum_pq A_pq d_p d_q` and its analytic gradient.
    Units of eV and eV/A."""

    def __init__(self, reference_positions, b, a_matrix):
        self.x0 = np.asarray(reference_positions, dtype=float)
        self.i, self.j = pair_indices(len(self.x0))
        self.r0, _ = pair_distances(self.x0, self.i, self.j)
        self.b = np.asarray(b, dtype=float)
        self.a = np.asarray(a_matrix, dtype=float)

    def energy_forces(self, positions):
        x = np.asarray(positions, dtype=float)
        r, dvec = pair_distances(x, self.i, self.j)
        d = r - self.r0
        g = self.b + self.a @ d                     # generalised force of each atom pair
        energy = float(self.b @ d + 0.5 * d @ (self.a @ d))
        unit = dvec / r[:, None]                    # ∂r_p/∂x_i = (x_i − x_j)/r_p
        f = np.zeros_like(x)
        np.add.at(f, self.i, -(g[:, None] * unit))
        np.add.at(f, self.j, +(g[:, None] * unit))
        return energy, f                            # F = −∂E/∂x

    def forces_only_b(self, positions):
        keep = self.a
        self.a = np.zeros_like(self.a)
        try:
            return self.energy_forces(positions)[1]
        finally:
            self.a = keep

    def forces_only_a(self, positions):
        keep = self.b
        self.b = np.zeros_like(self.b)
        try:
            return self.energy_forces(positions)[1]
        finally:
            self.b = keep


def build_perturbation(reference_positions, snapshots, sigma_offset_eV_A,
                       sigma_curvature_eV_A, seed=0):
    """Build a perturbation and **calibrate both of its parts numerically** to a given
    root-mean-square force per component.

    `snapshots` are thermally sampled configurations (frames of a real trajectory are
    best). Returns (the perturbation object, a calibration record).

    The calibration is measured: the shape is built with unit coefficients, the forces it
    gives are computed at the minimum and on the snapshots, the actual root-mean-square
    per component is measured, and the whole thing is then scaled to the target.
    **No analytic estimate is made anywhere.**
    """
    x0 = np.asarray(reference_positions, dtype=float)
    n_pairs = len(pair_indices(len(x0))[0])
    rng = np.random.default_rng(seed)

    b0 = rng.normal(0.0, 1.0, n_pairs)
    g = rng.normal(0.0, 1.0, (n_pairs, n_pairs))
    a0 = (g + g.T) / np.sqrt(2.0)               # symmetric; elements still standard normal

    trial = PairQuadraticPerturbation(x0, b0, a0)

    # --- the b part: calibrated **at the minimum** ---
    f_b = trial.forces_only_b(x0)
    rms_b = float(np.sqrt((f_b ** 2).mean()))
    scale_b = 0.0 if rms_b == 0 else sigma_offset_eV_A / rms_b

    # --- the A part: calibrated **on the thermal snapshots** ---
    snaps = [np.asarray(s, dtype=float) for s in snapshots]
    if snaps:
        comp = np.concatenate([trial.forces_only_a(s).ravel() for s in snaps])
        rms_a = float(np.sqrt((comp ** 2).mean()))
    else:
        rms_a = 0.0
    scale_a = 0.0 if rms_a == 0 else sigma_curvature_eV_A / rms_a

    pert = PairQuadraticPerturbation(x0, b0 * scale_b, a0 * scale_a)

    # --- measured check after calibration (not an estimate, a second measurement) ---
    check_min = float(np.sqrt((pert.energy_forces(x0)[1] ** 2).mean()))
    if snaps:
        allf = np.concatenate([pert.energy_forces(s)[1].ravel() for s in snaps])
        check_snap = float(np.sqrt((allf ** 2).mean()))
        rms_disp = float(np.sqrt(np.mean([((s - x0) ** 2).mean() for s in snaps])))
    else:
        check_snap, rms_disp = 0.0, 0.0
    rec = dict(seed=int(seed), n_pairs=int(n_pairs), n_snapshots=len(snaps),
               target_sigma_offset_eV_A=float(sigma_offset_eV_A),
               target_sigma_curvature_eV_A=float(sigma_curvature_eV_A),
               scale_b=float(scale_b), scale_a=float(scale_a),
               measured_force_rms_at_minimum_eV_A=check_min,
               measured_force_rms_on_snapshots_eV_A=check_snap,
               snapshot_rms_displacement_A=rms_disp,
               note=("the b part is calibrated to the force deviation at the minimum; "
                     "the A part to the increment brought by thermal displacement. "
                     "Both are measured numerically on real configurations and then "
                     "scaled, with no analytic estimate."))
    return pert, rec


class PerturbedCalculator(Calculator):
    """`base potential + perturbation`. An ASE calculator, attachable to molecular
    dynamics directly."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, base, perturbation, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.base = base
        self.perturbation = perturbation

    def calculate(self, atoms=None, properties=("energy",), system_changes=None):
        Calculator.calculate(self, atoms, properties,
                             system_changes or all_changes)
        a = self.atoms.copy()
        a.calc = self.base
        e = float(a.get_potential_energy())
        f = a.get_forces()
        ep, fp = self.perturbation.energy_forces(self.atoms.get_positions())
        self.results = dict(energy=e + ep, free_energy=e + ep, forces=f + fp,
                            base_energy=e, perturbation_energy=ep)
