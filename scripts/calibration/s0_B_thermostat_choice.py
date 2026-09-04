"""Which thermostat samples the distribution quasi-harmonic analysis actually reads.

CALIBRATION. It measures a property of the TOOLING in order to justify one setting in the
branch B driver, and it produces no scientific number.

The claim under test
--------------------
It has been put to this branch that Nose-Hoover is "significantly better than Langevin for
QHA", that Berendsen "severely undercalculates the vibrational entropy", and that a
Langevin run must keep the friction below 0.1/ps or the spectrum is distorted. Two of
those three are about DYNAMICS -- the velocity autocorrelation function and the vibrational
density of states that is its Fourier transform -- and dynamics is exactly what a
stochastic thermostat perturbs. The third is about the ENSEMBLE.

Branch B reads neither velocities nor time. Its estimator is

    C = <(x - <x>)(x - <x>)^T>

a variance over CONFIGURATIONS. Nothing in it is a time correlation, so a claim about peak
broadening in a vDOS cannot be carried across to it by argument -- the two quantities do
not have to fail together, and one of them (the vDOS) left this branch when openqha/vdos.py
was retired. What CAN carry across is a claim about the ensemble: a thermostat that samples
the wrong CONFIGURATIONAL distribution biases every eigenvalue of C, and does it silently.

So the question is narrowed to the one this branch can be harmed by, and then measured:
does thermostat X reproduce the correct canonical distribution of POSITIONS?

Why the answer is exactly known here
------------------------------------
On a harmonic surface it is. In mass-weighted normal coordinates the classical canonical
distribution has <q_k^2> = k_B T / omega_k^2 exactly, so the quasi-harmonic frequency
recovered from a correctly sampled trajectory must equal the Hessian frequency it was built
from, and the entropy must equal the closed form. That turns a comparison between
thermostats -- which can only ever say they disagree -- into a measurement of the BIAS of
each one against truth. A thermostat can now be wrong on its own, with nothing to hide
behind, and this file's whole design rests on that.

The surface is built from the production potential's own analytic Hessian at the minimum,
so the frequencies, the anisotropy and the stiffness ratios are the real molecule's. What
is given up is anharmonicity, and that is the right thing to give up: anharmonicity is a
property of the SURFACE, identical for every thermostat, so it cannot change their ranking,
while it would cost a factor of a thousand in wall time and put statistical noise where the
bias is supposed to be.

Two deliberate departures from the real molecule, both stated because both matter:

  * ROTATIONS ARE TETHERED. A linear force law cannot represent a rotation -- a finite
    displacement along a rotation vector shears the molecule instead of turning it -- so
    the three rotational directions are given a finite stiffness (500 cm-1 by default) and
    the quasi-harmonic Eckart projection removes them again, exactly as it removes real
    rotations. They are not free and are not meant to be.
  * TRANSLATION IS EXACTLY FREE. The calculator matches the centre of mass to the
    reference before it measures a displacement, so a rigid shift costs nothing and the
    centre of mass random-walks under Langevin precisely as it does in production. That is
    what makes the centre-of-mass family below a fair test rather than a fixed one.

The failing examples
--------------------
A criterion that has never rejected anything is not a criterion. Two variants are here to
be rejected, and if they are not, this script is broken rather than vindicating:

  * `berendsen` scales velocities toward the target by an unphysical damping factor. It
    has no canonical stationary distribution; its fluctuations are too narrow. It must come
    back with a suppressed configurational variance and an entropy that is too low.
  * `nose_hoover_chain_1` is a chain of length one, i.e. plain Nose-Hoover. The harmonic
    oscillator is the textbook case where plain Nose-Hoover is NOT ergodic, which is the
    reason chains were invented; a 10-atom molecule near its minimum is the regime where
    that matters most. It is included so that "Nose-Hoover" cannot be adopted as a word
    without the length of the chain being part of the decision.

    python scripts/calibration/s0_B_thermostat_choice.py
    python scripts/calibration/s0_B_thermostat_choice.py --steps 1000000 --seeds 8
    python scripts/calibration/s0_B_thermostat_choice.py --surface mace --steps 10000 \
           --seeds 3 --variants langevin_1.0 bussi nose_hoover_chain_3

`--surface mace` runs the same variants on the real potential instead. There is no closed
form there, so nothing can be called biased; what it answers is the narrower question of
whether the ranking established on the harmonic surface survives anharmonicity, and it is
reported as a spread about the branch B variant rather than as an error.
"""
import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))

from openqha import config, hessian, qha, report, thermo  # noqa: E402

TARGET_K = 298.15
TIMESTEP_FS = 1.0
SAMPLE_EVERY = 8
EQUILIBRATION_FRACTION = 0.1
RIGID_TETHER_CM_INV = 500.0

# The budget this calibration is judged against. Stage 0 targets 1.0 kcal/mol on the whole
# free energy; a single tooling choice that eats a tenth of it is already expensive, and
# the thermostat is one of perhaps a dozen such choices.
BUDGET_KCAL = 0.10


# ======================================================================================
# The surface
# ======================================================================================
class HarmonicSurface:
    """E = 1/2 dx^T H dx about a fixed reference, with translation projected out.

    Written as a bare ASE-style calculator rather than a subclass so that it pickles into
    a worker process without dragging a live MACE model with it.

    The centre-of-mass match in `_displacement` is what makes translation an EXACT zero
    mode: a rigid shift of every atom changes no displacement and therefore no energy and
    no force. Without it the reference position would tether the centre of mass, and the
    centre-of-mass variants below would be comparing thermostats on a surface that had
    already answered the question for them.
    """

    implemented_properties = ["energy", "forces", "free_energy"]

    def __init__(self, reference_positions, hessian_eV_A2, masses):
        self.x0 = np.asarray(reference_positions, dtype=float)
        self.h = np.asarray(hessian_eV_A2, dtype=float)
        self.m = np.asarray(masses, dtype=float)
        self.com0 = (self.m[:, None] * self.x0).sum(0) / self.m.sum()
        self.results = {}
        self.atoms = None

    def _displacement(self, positions):
        com = (self.m[:, None] * positions).sum(0) / self.m.sum()
        return (positions - com[None, :] + self.com0[None, :] - self.x0).reshape(-1)

    def calculate(self, atoms=None, properties=("energy",), system_changes=None):
        dx = self._displacement(atoms.get_positions())
        hdx = self.h @ dx
        # The force is minus the gradient of E with respect to the RAW positions. The
        # centre-of-mass match makes that the projected gradient, not the plain one: a
        # uniform component of -H dx would otherwise push the whole molecule.
        f = -hdx.reshape(-1, 3)
        f = f - (self.m[:, None] / self.m.sum()) * f.sum(0)[None, :]
        self.results = {"energy": float(0.5 * dx @ hdx),
                        "free_energy": float(0.5 * dx @ hdx),
                        "forces": f}

    def get_property(self, name, atoms=None, allow_calculation=True):
        self.calculate(atoms)
        return self.results[name]

    def get_potential_energy(self, atoms=None, force_consistent=False):
        return self.get_property("energy", atoms)

    def get_potential_energies(self, atoms=None):
        raise NotImplementedError

    def get_forces(self, atoms=None):
        return self.get_property("forces", atoms)

    def get_stress(self, atoms=None):
        raise NotImplementedError

    def get_stresses(self, atoms=None):
        raise NotImplementedError

    def calculation_required(self, atoms, quantities):
        return True

    def check_state(self, atoms, tol=1e-15):
        return ["positions"]


def _translation_rotation_bases(masses, positions):
    """Orthonormal bases for translation and for rotation-orthogonal-to-translation.

    Built HERE rather than sliced out of `hessian.rigid_body_vectors`, and the reason is
    worth stating because getting it wrong cost this script a run. That function returns
    the LEFT SINGULAR VECTORS of the six rigid columns:

        u, s, _ = np.linalg.svd(a_mat, full_matrices=False);  return u[:, :rank]

    An SVD basis spans the same six-dimensional space but does NOT keep the first three
    columns translational -- it mixes them. Slicing `v[:, 3:6]` therefore tethers an
    arbitrary three-dimensional subspace of the rigid space and leaves a MIXTURE of
    translation and rotation free. The calculator already frees translation, so what is
    left unconstrained is partly rotational, the molecule shears without restoring force,
    and the extra variance lands in every internal mode.

    How it showed: every thermostat came back 3 to 11 kcal/mol high on T*S while the
    `exact_sampling` control -- which never touches this surface -- was right to 0.003.
    A control that passes while every measurement fails is not evidence about the
    measurements; it says the apparatus is broken. `_check_tether` below turns that into
    an assertion so the next person is told rather than left to notice.
    """
    m = np.asarray(masses, dtype=float)
    r = np.asarray(positions, dtype=float)
    com = (m[:, None] * r).sum(0) / m.sum()
    d = r - com
    sm = np.sqrt(m)
    trans = []
    for a in range(3):
        v = np.zeros((len(m), 3))
        v[:, a] = sm
        trans.append(v.reshape(-1))
    t_mat = np.stack(trans, axis=1)
    t_mat /= np.linalg.norm(t_mat, axis=0, keepdims=True)

    rots = []
    for a in range(3):
        e = np.zeros(3)
        e[a] = 1.0
        v = sm[:, None] * np.cross(np.broadcast_to(e, d.shape), d)
        rots.append(v.reshape(-1))
    r_mat = np.stack(rots, axis=1)
    r_mat = r_mat - t_mat @ (t_mat.T @ r_mat)      # orthogonal to translation
    u, sv, _ = np.linalg.svd(r_mat, full_matrices=False)
    keep = sv > 1e-8 * max(sv[0], 1.0)             # 2 for a linear molecule, 3 otherwise
    return t_mat, u[:, keep]


def _check_tether(h_tethered, masses, positions, expected_free=3):
    """The tethered Hessian's kernel must be EXACTLY translation, nothing else.

    This is the assertion that the first version of this file lacked. It costs one
    eigendecomposition of a 3N x 3N matrix and it fails loudly instead of returning a
    plausible table of wrong numbers.
    """
    m3 = np.repeat(np.asarray(masses, dtype=float), 3)
    hm = np.asarray(h_tethered, dtype=float) / np.sqrt(np.outer(m3, m3))
    lam, vec = np.linalg.eigh(0.5 * (hm + hm.T))
    scale = max(abs(lam[-1]), 1.0)
    n_free = int((lam < 1e-8 * scale).sum())
    t_mat, _r = _translation_rotation_bases(masses, positions)
    overlap = float(np.linalg.norm(t_mat.T @ vec[:, :n_free]) ** 2 / max(n_free, 1))
    if n_free != expected_free or overlap < 0.999:
        raise ValueError(
            "the tethered surface has {} free direction(s) (expected {}) and they overlap "
            "translation by {:.4f} (expected 1.0). Anything not translation that is left "
            "free lets the molecule drift without a restoring force, and the extra "
            "variance lands in every internal mode."
            .format(n_free, expected_free, overlap))
    return dict(n_free_directions=n_free, translation_overlap=overlap,
                lowest_tethered_cm_inv=float(
                    np.sqrt(max(lam[n_free], 0.0)) * qha.CM_INV_PER_SQRT_EV_A2_AMU))


def tethered_hessian(hessian_eV_A2, masses, positions, tether_cm_inv=RIGID_TETHER_CM_INV):
    """Replace the rigid-body kernel of a Hessian with a finite stiffness on ROTATION only.

    Translation is deliberately left in the kernel, because the calculator above already
    makes it an exact symmetry. The internal subspace is untouched by construction
    (P H P is the same operator either way), which is what lets the closed-form reference
    be computed from the ORIGINAL Hessian -- `main` checks that too.
    """
    m = np.asarray(masses, dtype=float)
    m3 = np.repeat(m, 3)
    hm = np.asarray(hessian_eV_A2, dtype=float) / np.sqrt(np.outer(m3, m3))
    v, _s, rank = hessian.rigid_body_vectors(m, positions)
    p = np.eye(len(m3)) - v @ v.T
    hp = p @ hm @ p
    # omega^2 in eV/(A^2 amu) for a mode at `tether_cm_inv`.
    omega2 = (tether_cm_inv / qha.CM_INV_PER_SQRT_EV_A2_AMU) ** 2
    _t_mat, r_mat = _translation_rotation_bases(m, positions)
    hp = hp + omega2 * (r_mat @ r_mat.T)
    hp = 0.5 * (hp + hp.T)
    h_out = hp * np.sqrt(np.outer(m3, m3))
    check = _check_tether(h_out, m, positions, expected_free=len(m3) - rank + 3
                          if rank < 6 else 3)
    return h_out, int(rank), check


# ======================================================================================
# The variants
# ======================================================================================
VARIANTS = [
    # name, family, spec
    ("exact_sampling", "reference",
     dict(kind="exact"),
     "closed-form Gaussian sampling, no dynamics at all"),

    ("langevin_1.0", "thermostat",
     dict(kind="langevin", friction_per_ps=1.0, fixcm=False),
     "branch B as it stands"),
    ("langevin_0.1", "thermostat",
     dict(kind="langevin", friction_per_ps=0.1, fixcm=False),
     "the friction suggested as a vDOS mitigation"),
    ("langevin_10.0", "thermostat",
     dict(kind="langevin", friction_per_ps=10.0, fixcm=False),
     "ten times branch B, to bracket the friction"),
    ("bussi", "thermostat",
     dict(kind="bussi", taut_fs=100.0),
     "stochastic velocity rescaling; GROMACS tcoupl = v-rescale, which the mdp already says"),
    ("nose_hoover_chain_3", "thermostat",
     dict(kind="nose_hoover_chain", tdamp_fs=100.0, tchain=3),
     "Nose-Hoover done properly, as a chain"),
    ("nose_hoover_chain_1", "thermostat",
     dict(kind="nose_hoover_chain", tdamp_fs=100.0, tchain=1),
     "plain Nose-Hoover: EXPECTED TO FAIL, the textbook non-ergodic case"),
    # ---- Nose-Hoover, swept rather than sampled once. See the header of the patch that
    # added these: one setting is not a test of a method.
    ("nh_tdamp5_c3_l1", "nose-hoover",
     dict(kind="nose_hoover_chain", tdamp_fs=5.0, tchain=3, tloop=1),
     "very tight coupling, 200/ps"),
    ("nh_tdamp10_c3_l1", "nose-hoover",
     dict(kind="nose_hoover_chain", tdamp_fs=10.0, tchain=3, tloop=1),
     "100/ps"),
    ("nh_tdamp30_c3_l1", "nose-hoover",
     dict(kind="nose_hoover_chain", tdamp_fs=30.0, tchain=3, tloop=1),
     "33/ps"),
    ("nh_tdamp20_c5_l1", "nose-hoover",
     dict(kind="nose_hoover_chain", tdamp_fs=20.0, tchain=5, tloop=1),
     "the tight-coupling point with a longer chain"),
    ("nh_tdamp20_c3_l1", "nose-hoover",
     dict(kind="nose_hoover_chain", tdamp_fs=20.0, tchain=3, tloop=1),
     "tight coupling"),
    ("nh_tdamp50_c3_l1", "nose-hoover",
     dict(kind="nose_hoover_chain", tdamp_fs=50.0, tchain=3, tloop=1),
     ""),
    ("nh_tdamp200_c3_l1", "nose-hoover",
     dict(kind="nose_hoover_chain", tdamp_fs=200.0, tchain=3, tloop=1),
     ""),
    ("nh_tdamp419_c3_l1", "nose-hoover",
     dict(kind="nose_hoover_chain", tdamp_fs=419.0, tchain=3, tloop=1),
     "coupling time matched to the 79.7 cm-1 lowest mode's period"),
    ("nh_tdamp1000_c3_l1", "nose-hoover",
     dict(kind="nose_hoover_chain", tdamp_fs=1000.0, tchain=3, tloop=1),
     "loose coupling"),
    ("nh_tdamp100_c3_l5", "nose-hoover",
     dict(kind="nose_hoover_chain", tdamp_fs=100.0, tchain=3, tloop=5),
     "same as nose_hoover_chain_3 but five times the chain propagation resolution"),
    ("nh_tdamp419_c5_l5", "nose-hoover",
     dict(kind="nose_hoover_chain", tdamp_fs=419.0, tchain=5, tloop=5),
     "matched coupling, longer chain, resolved propagation"),
    ("nh_tdamp1000_c5_l5", "nose-hoover",
     dict(kind="nose_hoover_chain", tdamp_fs=1000.0, tchain=5, tloop=5),
     "loose coupling, longer chain, resolved propagation"),

    ("berendsen", "thermostat",
     dict(kind="berendsen", taut_fs=100.0, fixcm=False),
     "EXPECTED TO FAIL: no canonical stationary distribution"),
    ("andersen", "thermostat",
     dict(kind="andersen", prob=0.01, fixcm=False),
     "hard collisions; correct ensemble, badly broken dynamics"),

    # ---- Nose-Hoover with rigid-motion removal, proposed 2026-09-04. Measured at the
    # ORIGINAL 100 fs coupling as well as the tight 20 fs one, so the comparison against
    # Langevin is not made only where Nose-Hoover happens to do best.
    ("nh_tdamp100_stationary", "nose-hoover + rigid removal",
     dict(kind="nose_hoover_chain", tdamp_fs=100.0, tchain=3, tloop=1, stationary=True),
     "original coupling, Stationary + ZeroRotation every step"),
    ("nh_tdamp20_stationary", "nose-hoover + rigid removal",
     dict(kind="nose_hoover_chain", tdamp_fs=20.0, tchain=3, tloop=1, stationary=True),
     "tight coupling, Stationary + ZeroRotation every step"),
    ("nh_tdamp100_recentre", "nose-hoover + rigid removal",
     dict(kind="nose_hoover_chain", tdamp_fs=100.0, tchain=3, tloop=1, recentre=True),
     "original coupling, POSITIONS re-centred: the same visual cleanliness, no momentum "
     "touched"),
    ("nh_tdamp20_recentre", "nose-hoover + rigid removal",
     dict(kind="nose_hoover_chain", tdamp_fs=20.0, tchain=3, tloop=1, recentre=True),
     "tight coupling, POSITIONS re-centred"),

    ("com_fixcm_true", "centre of mass",
     dict(kind="langevin", friction_per_ps=1.0, fixcm=True),
     "ASE's Langevin default"),
    ("com_fixcm_false", "centre of mass",
     dict(kind="langevin", friction_per_ps=1.0, fixcm=False),
     "branch B: no centre-of-mass handling in the integrator"),
    ("com_fixcom_constraint", "centre of mass",
     dict(kind="langevin", friction_per_ps=1.0, fixcm=False, fixcom=True),
     "option 1 as proposed: the FixCom constraint"),
    ("com_recentre_positions", "centre of mass",
     dict(kind="langevin", friction_per_ps=1.0, fixcm=False, recentre=True),
     "option 1 as branch B implements it: shift POSITIONS every step, touch no momentum"),
    ("com_stationary_zerorot", "centre of mass",
     dict(kind="langevin", friction_per_ps=1.0, fixcm=False, stationary=True),
     "option 1 as proposed: Stationary + ZeroRotation every step"),
]

VARIANT_BY_NAME = {v[0]: v for v in VARIANTS}


def make_dynamics(spec, atoms, rng):
    from ase import units
    dt = TIMESTEP_FS * units.fs
    kind = spec["kind"]
    if kind == "langevin":
        from ase.md.langevin import Langevin
        return Langevin(atoms, dt, temperature_K=TARGET_K,
                        friction=spec["friction_per_ps"] / (1000.0 * units.fs),
                        rng=rng, fixcm=bool(spec.get("fixcm", False)))
    if kind == "bussi":
        from ase.md.bussi import Bussi
        return Bussi(atoms, dt, temperature_K=TARGET_K,
                     taut=spec["taut_fs"] * units.fs, rng=rng)
    if kind == "nose_hoover_chain":
        from ase.md.nose_hoover_chain import NoseHooverChainNVT
        # `tloop` is the multiple-time-step count for the chain propagation. ASE defaults
        # it to 1 and uses a three-term Yoshida-Suzuki decomposition, i.e. three chain
        # evaluations per step; OpenMM's default for the same algorithm is 3 x 7 = 21.
        # It is exposed here because a thermostat judged on an under-resolved propagator
        # has been judged on the wrong thing.
        return NoseHooverChainNVT(atoms, dt, temperature_K=TARGET_K,
                                  tdamp=spec["tdamp_fs"] * units.fs,
                                  tchain=int(spec["tchain"]),
                                  tloop=int(spec.get("tloop", 1)))
    if kind == "berendsen":
        from ase.md.nvtberendsen import NVTBerendsen
        return NVTBerendsen(atoms, dt, temperature_K=TARGET_K,
                            taut=spec["taut_fs"] * units.fs,
                            fixcm=bool(spec.get("fixcm", False)))
    if kind == "andersen":
        from ase.md.andersen import Andersen
        return Andersen(atoms, dt, temperature_K=TARGET_K,
                        andersen_prob=float(spec["prob"]), rng=rng,
                        fixcm=bool(spec.get("fixcm", False)))
    raise ValueError("unknown thermostat kind {!r}".format(kind))


# ======================================================================================
# One run
# ======================================================================================
def run_one(job):
    """One (variant, seed). Returns a flat record; never raises into the pool."""
    name, spec, seed, steps, payload = job
    try:
        return _run_one(name, spec, seed, steps, payload)
    except Exception as exc:                                  # noqa: BLE001
        return dict(variant=name, seed=int(seed), failed=True,
                    error="{}: {}".format(type(exc).__name__, exc))


def _run_one(name, spec, seed, steps, payload):
    from ase import Atoms, units
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    numbers = payload["numbers"]
    x0 = payload["reference_positions"]
    masses = payload["masses"]
    h_tether = payload["hessian_tethered"]
    n_atoms = len(masses)

    if spec["kind"] == "exact":
        frames, _ref = qha.synthetic_harmonic_trajectory(
            payload["hessian"], masses, x0, TARGET_K,
            n_frames=max(1000, int(steps) // SAMPLE_EVERY), seed=seed)
        rec = qha.analyse(frames, masses, TARGET_K)
        return dict(variant=name, seed=int(seed), failed=False,
                    n_frames=int(len(frames)),
                    TS_QH_kcal=rec["entropy"]["TS_QH_kcal"],
                    lowest_frequency_cm_inv=rec["entropy"]["lowest_frequency_cm_inv"],
                    n_nonzero=rec["spectrum"]["n_nonzero_eigenvalues"],
                    kinetic_dof_effective=float("nan"),
                    kinetic_variance_ratio=float("nan"),
                    configurational_dof_effective=float("nan"),
                    configurational_variance_ratio=float("nan"),
                    com_drift_A=0.0, wall_seconds=0.0)

    atoms = Atoms(numbers=numbers, positions=x0.copy(), masses=masses)
    if payload["surface"] == "mace":
        # Each worker builds its own model. That is wasteful and it is the only correct
        # thing to do: a MACE calculator carries torch state that does not survive a fork
        # cleanly, and a shared one would serialise every force call through one lock,
        # which would make the parallelism a lie rather than merely slow.
        from openqha import engine
        calc, _name, _prov = engine.calculator(device="cpu")
        atoms.calc = calc
    else:
        atoms.calc = HarmonicSurface(x0, h_tether, masses)

    if spec.get("fixcom"):
        from ase.constraints import FixCom
        atoms.set_constraint(FixCom())

    rng = np.random.RandomState(int(seed))
    MaxwellBoltzmannDistribution(atoms, temperature_K=TARGET_K, rng=rng)
    dyn = make_dynamics(spec, atoms, rng)

    com0 = (masses[:, None] * atoms.get_positions()).sum(0) / masses.sum()
    frames, kinetic, potential, drift = [], [], [], []

    def grab():
        p = atoms.get_positions()
        frames.append(p.copy())
        kinetic.append(atoms.get_kinetic_energy())
        potential.append(atoms.get_potential_energy())
        com = (masses[:, None] * p).sum(0) / masses.sum()
        drift.append(float(np.linalg.norm(com - com0)))

    dyn.attach(grab, interval=SAMPLE_EVERY)

    if spec.get("recentre"):
        def _recentre():
            p = atoms.get_positions()
            com = (masses[:, None] * p).sum(0) / masses.sum()
            atoms.set_positions(p - com[None, :] + com0[None, :])
        dyn.attach(_recentre, interval=1)

    if spec.get("stationary"):
        from ase.md.velocitydistribution import Stationary, ZeroRotation

        def _freeze():
            Stationary(atoms)
            ZeroRotation(atoms)
        dyn.attach(_freeze, interval=1)

    import time
    t0 = time.time()
    dyn.run(int(steps))
    wall = time.time() - t0

    cut = max(1, int(EQUILIBRATION_FRACTION * len(frames)))
    pos = np.array(frames[cut:])
    ke = np.array(kinetic[cut:])
    pe = np.array(potential[cut:])

    kb_ev = thermo.KB_SI / 1.602176634e-19
    kt = kb_ev * TARGET_K

    # Effective degrees of freedom, read off the sample instead of assumed. This is the
    # lesson of the fixcm calibration in its assumption-free form: rather than decide
    # whether a variant divides by 3N or 3N-3 and then discover the choice was wrong, ask
    # the trajectory how many degrees of freedom are actually carrying kT/2.
    dof_kin = float(2.0 * ke.mean() / kt)
    dof_pot = float(2.0 * pe.mean() / kt)
    # For a canonical sample of n quadratic degrees of freedom, var/mean^2 = 2/n. The
    # ratio below is therefore 1 for any correct thermostat and less than 1 for one that
    # squeezes the distribution. THIS is the number Berendsen has to fail.
    kin_ratio = float((ke.var(ddof=1) / ke.mean() ** 2) / (2.0 / dof_kin)) if dof_kin > 0 else float("nan")
    pot_ratio = float((pe.var(ddof=1) / pe.mean() ** 2) / (2.0 / dof_pot)) if dof_pot > 0 else float("nan")

    rec = qha.analyse(pos, masses, TARGET_K)
    return dict(variant=name, seed=int(seed), failed=False,
                n_frames=int(len(pos)),
                TS_QH_kcal=rec["entropy"]["TS_QH_kcal"],
                lowest_frequency_cm_inv=rec["entropy"]["lowest_frequency_cm_inv"],
                n_nonzero=rec["spectrum"]["n_nonzero_eigenvalues"],
                kinetic_dof_effective=dof_kin,
                kinetic_variance_ratio=kin_ratio,
                configurational_dof_effective=dof_pot,
                configurational_variance_ratio=pot_ratio,
                com_drift_A=float(np.max(drift[cut:])) if len(drift) > cut else 0.0,
                wall_seconds=float(wall),
                n_atoms=int(n_atoms))


# ======================================================================================
# Driver
# ======================================================================================
def build_surface(engine_name, species, cfg, hessian_mode):
    from ase.io import read
    from ase.optimize import BFGS
    atoms = read(str(config.qm9_xyz(species, cfg)))
    if engine_name == "mace":
        from openqha import engine
        calc, label, _prov = engine.calculator(device="cpu")
    else:
        from ase.calculators.emt import EMT
        calc, label = EMT(), "ase.calculators.emt.EMT"
    atoms.calc = calc
    BFGS(atoms, logfile=None).run(fmax=1e-4, steps=2000)
    # hessian.hessian returns (H, asymmetry) -- the second element is a diagnostic,
    # not a matrix. The first version of this function fed the whole tuple to
    # np.asarray and got an inhomogeneous-shape error, which is the good failure.
    out = hessian.hessian(atoms, calc, mode=hessian_mode)
    h, asym = (out if isinstance(out, tuple) else (out, float("nan")))
    h = np.asarray(h, dtype=float)
    h = 0.5 * (h + h.T)
    return atoms, h, "{} (Hessian {}, max|H-H^T| = {:.3e} eV/A^2)".format(
        label, hessian_mode, asym)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default="dsgdb9nsd_000018")
    ap.add_argument("--engine", default="mace", choices=("mace", "emt"),
                    help="which potential supplies the Hessian the surface is built from")
    ap.add_argument("--surface", default="harmonic", choices=("harmonic", "mace"),
                    help="what the dynamics actually run on. 'harmonic' has a closed-form "
                         "answer and can therefore measure bias; 'mace' has none and can "
                         "only measure spread")
    ap.add_argument("--hessian-mode", default="analytic",
                    choices=("analytic", "finite_difference"))
    ap.add_argument("--steps", type=int, default=500000)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=20260904)
    ap.add_argument("--processes", type=int, default=0)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--tether-cm-inv", type=float, default=RIGID_TETHER_CM_INV)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = config.load()
    atoms, h_raw, engine_label = build_surface(args.engine, args.species, cfg,
                                               args.hessian_mode)
    masses = atoms.get_masses()
    x0 = atoms.get_positions()
    h_tether, rigid_rank, tether_check = tethered_hessian(h_raw, masses, x0,
                                                          args.tether_cm_inv)

    _frames, ref = qha.synthetic_harmonic_trajectory(h_raw, masses, x0, TARGET_K,
                                                     n_frames=8, seed=1)
    _f2, ref2 = qha.synthetic_harmonic_trajectory(h_tether, masses, x0, TARGET_K,
                                                  n_frames=8, seed=1)
    internal_shift = float(np.max(np.abs(
        np.array(ref["frequencies_cm_inv"]) - np.array(ref2["frequencies_cm_inv"]))))
    exact_TS = float(ref["closed_form_TS_kcal"])

    chosen = args.variants or [v[0] for v in VARIANTS]
    unknown = [c for c in chosen if c not in VARIANT_BY_NAME]
    if unknown:
        raise SystemExit("unknown variant(s): {}".format(", ".join(unknown)))

    print("=" * 92)
    print("Branch B calibration -- which thermostat samples the distribution QHA reads")
    print("=" * 92)
    print("species    {}   atoms {}   3N-6 = {}   rigid rank {}"
          .format(args.species, len(atoms), 3 * len(atoms) - 6, rigid_rank))
    if args.surface == "harmonic":
        print("surface    harmonic, from the {} analytic Hessian at the minimum"
              .format(engine_label))
        print("           rotations tethered at {:.0f} cm-1; translation exactly free"
              .format(args.tether_cm_inv))
        print("           tether shifts the internal frequencies by {:.3e} cm-1 "
              "(it must not)".format(internal_shift))
        print("           free directions {} (must be 3), overlap with translation {:.6f} "
              "(must be 1)".format(tether_check["n_free_directions"],
                                   tether_check["translation_overlap"]))
    else:
        print("surface    {} itself -- anharmonic, and therefore WITHOUT a closed form."
              .format(engine_label))
        print("           Nothing below is a bias. The column to read is the spread about "
              "the branch B variant.")
    print("protocol   {} steps of {} fs, sample every {}, first {:.0f}% dropped"
          .format(args.steps, TIMESTEP_FS, SAMPLE_EVERY, 100 * EQUILIBRATION_FRACTION))
    if args.surface == "harmonic":
        print("truth      T*S = {:.6f} kcal/mol, closed form from {} modes"
              .format(exact_TS, ref["n_modes"]))
    else:
        print("truth      none. The harmonic closed form ({:.6f} kcal/mol) is printed for "
              "scale only".format(exact_TS))
    print()

    payload = dict(numbers=atoms.get_atomic_numbers(), reference_positions=x0,
                   masses=masses, hessian=h_raw, hessian_tethered=h_tether,
                   surface=args.surface)
    jobs = [(name, VARIANT_BY_NAME[name][2], args.seed0 + i, args.steps, payload)
            for name in chosen for i in range(args.seeds)]

    n_proc = args.processes or min(len(jobs), mp.cpu_count())
    if n_proc > 1:
        with mp.Pool(n_proc) as pool:
            raw = pool.map(run_one, jobs)
    else:
        raw = [run_one(j) for j in jobs]

    rows = []
    for name in chosen:
        got = [r for r in raw if r["variant"] == name]
        bad = [r for r in got if r.get("failed")]
        ok = [r for r in got if not r.get("failed")]
        if not ok:
            rows.append(dict(variant=name, family=VARIANT_BY_NAME[name][1],
                             note=VARIANT_BY_NAME[name][3], n_seeds=0,
                             error=bad[0]["error"] if bad else "no runs"))
            print("{:<24} FAILED  {}".format(name, bad[0]["error"] if bad else ""))
            continue
        ts = np.array([r["TS_QH_kcal"] for r in ok])
        # On the harmonic surface this is an error against truth. On the real potential
        # it is an offset from an arbitrary origin, and it is relabelled below so that it
        # cannot be read as an error by someone who skipped the header.
        err = ts - exact_TS
        row = dict(
            variant=name, family=VARIANT_BY_NAME[name][1],
            note=VARIANT_BY_NAME[name][3], n_seeds=int(len(ok)),
            n_failed=int(len(bad)),
            TS_QH_kcal=float(ts.mean()),
            TS_error_kcal=float(err.mean()),
            TS_seed_sem_kcal=float(ts.std(ddof=1) / np.sqrt(len(ts))) if len(ts) > 1 else 0.0,
            kinetic_dof_effective=float(np.mean([r["kinetic_dof_effective"] for r in ok])),
            kinetic_variance_ratio=float(np.mean([r["kinetic_variance_ratio"] for r in ok])),
            configurational_dof_effective=float(
                np.mean([r["configurational_dof_effective"] for r in ok])),
            configurational_variance_ratio=float(
                np.mean([r["configurational_variance_ratio"] for r in ok])),
            com_drift_A=float(np.mean([r["com_drift_A"] for r in ok])),
            lowest_frequency_cm_inv=float(
                np.mean([r["lowest_frequency_cm_inv"] for r in ok])),
            wall_seconds=float(np.mean([r["wall_seconds"] for r in ok])))
        row["surface"] = args.surface
        if args.surface == "harmonic":
            row["within_budget"] = bool(abs(row["TS_error_kcal"]) <= BUDGET_KCAL)
        else:
            row["TS_error_kcal"] = float("nan")
            row["within_budget"] = None
        rows.append(row)
        if args.surface == "harmonic":
            print("{:<24} T*S err {:+8.4f} +- {:6.4f}   conf.var ratio {:6.3f}   "
                  "kin dof {:6.2f}   COM {:7.2f} A   {}"
                  .format(name, row["TS_error_kcal"], row["TS_seed_sem_kcal"],
                          row["configurational_variance_ratio"],
                          row["kinetic_dof_effective"], row["com_drift_A"],
                          "ok" if row["within_budget"] else "OVER BUDGET"))
        else:
            print("{:<24} T*S {:9.4f} +- {:6.4f}   conf.var ratio {:6.3f}   "
                  "kin dof {:6.2f}   COM {:7.2f} A"
                  .format(name, row["TS_QH_kcal"], row["TS_seed_sem_kcal"],
                          row["configurational_variance_ratio"],
                          row["kinetic_dof_effective"], row["com_drift_A"]))

    scored = [r for r in rows if "TS_QH_kcal" in r]
    reference_row = next((r for r in scored if r["variant"] in
                          ("langevin_1.0", "com_fixcm_false")), None)
    for r in scored:
        r["TS_minus_branchB_kcal"] = (
            float(r["TS_QH_kcal"] - reference_row["TS_QH_kcal"]) if reference_row else
            float("nan"))
    print()
    control = next((r for r in scored if r["variant"] == "exact_sampling"), None)
    others = [r for r in scored if r["variant"] != "exact_sampling"]
    if (args.surface == "harmonic" and control is not None and others
            and abs(control["TS_error_kcal"]) < BUDGET_KCAL
            and all(abs(r["TS_error_kcal"]) > 10 * BUDGET_KCAL for r in others)):
        print()
        print("STOP. `exact_sampling` -- which never touches the surface -- is right to "
              "{:+.4f} kcal/mol while EVERY variant that runs on it is off by more than "
              "{:.2f}. That pattern is not a statement about thermostats; it says the "
              "surface or the analysis is broken. Do not read the table below as a "
              "ranking.".format(control["TS_error_kcal"], 10 * BUDGET_KCAL))
    if args.surface == "harmonic":
        failing = [r for r in scored if not r["within_budget"]]
        print("{} of {} variants sit inside the {:.2f} kcal/mol budget."
              .format(len(scored) - len(failing), len(scored), BUDGET_KCAL))
        if not failing:
            print("WARNING: nothing failed. The two variants marked EXPECTED TO FAIL are "
                  "here so that a pass means something; if they passed, suspect this "
                  "script rather than the thermostats.")
    else:
        spread = [abs(r["TS_minus_branchB_kcal"]) for r in scored
                  if r["variant"] != "exact_sampling"]
        print("Largest departure from the branch B variant on the real surface: "
              "{:.4f} kcal/mol.".format(max(spread) if spread else float("nan")))
        print("This is a spread, not an error. The harmonic run is where bias is "
              "measurable.")

    out_stem = Path(args.out) if args.out else (
        _repo_root() / "analysis" / "qha" /
        ("calibration_thermostat_choice" if args.surface == "harmonic"
         else "calibration_thermostat_choice_mace"))
    rp = report.Report(
        "Branch B calibration -- thermostat choice against a known answer",
        subtitle="{}   {}   {} steps x {} seeds".format(
            args.species, engine_label, args.steps, args.seeds))
    rp.section("The question, narrowed")
    rp.note("It was put to this branch that Nose-Hoover is better than Langevin for QHA, "
            "that Berendsen undercounts vibrational entropy, and that Langevin friction "
            "must stay below 0.1/ps. Two of those are statements about DYNAMICS -- the "
            "velocity autocorrelation function and the vDOS that is its Fourier "
            "transform. Branch B's estimator is a variance over CONFIGURATIONS with no "
            "time correlation in it, and the vDOS left this branch when openqha/vdos.py "
            "was retired, so a claim about peak broadening cannot be carried across by "
            "argument. What can be carried across is the ENSEMBLE claim, and that is what "
            "is measured here.")
    rp.section("Why an answer exists to be compared against")
    rp.note("On a harmonic surface the classical canonical distribution has "
            "<q_k^2> = k_B T / omega_k^2 exactly, so a correctly sampled trajectory must "
            "return the Hessian's own frequencies. That converts a comparison BETWEEN "
            "thermostats, which can only show disagreement, into a measurement of each "
            "one's BIAS against truth.")
    rp.kv("closed_form_TS_kcal", round(exact_TS, 6), unit="kcal/mol")
    rp.kv("rigid_tether_cm_inv", args.tether_cm_inv)
    rp.kv("tether_shift_on_internal_frequencies_cm_inv", "{:.3e}".format(internal_shift),
          note="the tether must not touch the internal subspace; this is the check")
    rp.kv("timestep_fs", TIMESTEP_FS)
    rp.kv("target_temperature_K", TARGET_K)
    rp.kv("budget_kcal", BUDGET_KCAL)
    rp.section("Measured")
    if args.surface == "harmonic":
        rp.table(["variant", "family", "T*S error", "seed s.e.", "conf. var ratio",
                  "kin. dof", "COM drift", "verdict"],
                 [[r["variant"], r["family"],
                   "{:+.4f}".format(r["TS_error_kcal"]),
                   round(r["TS_seed_sem_kcal"], 4),
                   round(r["configurational_variance_ratio"], 3),
                   round(r["kinetic_dof_effective"], 2),
                   round(r["com_drift_A"], 2),
                   "ok" if r["within_budget"] else "OVER BUDGET"] for r in scored],
                 units=[None, None, "kcal/mol", "kcal/mol", None, None, "A", None])
    else:
        rp.warn("This run used --surface mace. There is no closed form on an anharmonic "
                "potential, so no entry below is an error and none of them is evidence "
                "that a thermostat is unbiased. The question it answers is narrower: does "
                "the ranking found on the harmonic surface survive anharmonicity?")
        rp.table(["variant", "family", "T*S", "seed s.e.", "vs branch B",
                  "conf. var ratio", "kin. dof", "COM drift"],
                 [[r["variant"], r["family"],
                   round(r["TS_QH_kcal"], 4),
                   round(r["TS_seed_sem_kcal"], 4),
                   "{:+.4f}".format(r["TS_minus_branchB_kcal"]),
                   round(r["configurational_variance_ratio"], 3),
                   round(r["kinetic_dof_effective"], 2),
                   round(r["com_drift_A"], 2)] for r in scored],
                 units=[None, None, "kcal/mol", "kcal/mol", "kcal/mol", None, None, "A"])
    rp.note("`conf. var ratio` is var(U)/<U>^2 divided by its canonical value 2/n, with n "
            "read off the same sample as 2<U>/kT rather than assumed. It is 1 for a "
            "correct thermostat and below 1 for one that squeezes the configurational "
            "distribution -- which is the only way a thermostat can reach branch B's "
            "number. `kin. dof` is read the same way, and is what tells apart a variant "
            "that removes three degrees of freedom from one that is simply hot.")
    rp.section("Failing examples")
    rp.note("A criterion that has never rejected anything is not a criterion. Two "
            "variants are present in order to be rejected: `berendsen`, which has no "
            "canonical stationary distribution, and `nose_hoover_chain_1`, which is plain "
            "Nose-Hoover -- the harmonic oscillator is the textbook case where it is not "
            "ergodic, and a 10-atom molecule near its minimum is that regime. The second "
            "is here so that 'Nose-Hoover' cannot be adopted as a word without the chain "
            "length being part of the decision.")
    rp.json_dump(dict(species=args.species, engine=engine_label,
                      surface=args.surface,
                      hessian_mode=args.hessian_mode,
                      steps=args.steps, seeds=args.seeds, seed0=args.seed0,
                      timestep_fs=TIMESTEP_FS, sample_every=SAMPLE_EVERY,
                      target_K=TARGET_K, budget_kcal=BUDGET_KCAL,
                      rigid_tether_cm_inv=args.tether_cm_inv,
                      tether_shift_cm_inv=internal_shift,
                      tether_check=tether_check,
                      closed_form_TS_kcal=exact_TS,
                      closed_form_frequencies_cm_inv=ref["frequencies_cm_inv"],
                      variants=rows, runs=raw))
    log = rp.write(str(out_stem) + ".log")
    written = report.write_parquet(dict(thermostat_variants=rows,
                                        thermostat_runs=[r for r in raw
                                                         if not r.get("failed")]),
                                   out_stem)
    print("\nwritten:\n  {}".format(log))
    for p, n, _c in written:
        print("  {}  ({} rows)".format(p, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
