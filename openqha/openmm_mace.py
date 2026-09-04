"""MACE as an OpenMM force, with an exact all-pairs graph. openQHA's own, no NNPOps.

Why this exists
---------------
Branch B moved to a Nose-Hoover chain thermostat, and the reference implementation of that
algorithm is OpenMM's (`openmmtools.integrators.NoseHooverChainVelocityVerletIntegrator`,
whose documented defaults are collision_frequency 50/ps, chain_length 5, num_mts 5,
num_yoshidasuzuki 5). Running branch B's potential under it needs MACE inside OpenMM.

The usual route is `openmm-ml`, whose MACE implementation builds its neighbour list with
`NNPOps.neighbors.getNeighborPairs`. Neither `openmm-ml` nor `NNPOps` is installed in this
workspace, and both are heavier dependencies than this branch needs -- because branch B's
molecules have 10 to 19 atoms.

At that size the exact neighbour list is the COMPLETE GRAPH. It needs no spatial binning,
so it cannot have the origin-anchoring defect that `openqha/mace_patch.py` guards against
-- there is no box to anchor. For this system size it is the more correct of the two
constructions, and the cheaper one.

The graph is not even filtered by the cutoff, and that is deliberate: MACE's radial cutoff
function is EXACTLY zero at and beyond `r_max`, so a pair outside the cutoff contributes
nothing at all. Measured on acetone with MACE-OFF23_medium, scaling the molecule up so
that most pairs fall outside the 5 A cutoff:

    scale   max pair    filtered edges   complete edges   dE / eV     max|dF| / (eV/A)
     1.0     4.295 A          90               90         0.000e+00      6.939e-17
     1.3     5.584 A          88               90         0.000e+00      0.000e+00
     1.6     6.873 A          72               90         0.000e+00      0.000e+00
     2.0     8.591 A          46               90         0.000e+00      0.000e+00

Bit-identical with half the edges outside the cutoff. That is what makes the edge list
STATIC, and a static edge list is what makes `torch.jit.trace` sound -- which matters
because `torch.jit.script` FAILS on this model: e3nn 0.4.4's `_activation.py` is not
scriptable ("'Tuple[int, int]' object has no attribute or method 'dim'"). openmm-ml works
around that by rebuilding the model; this module sidesteps it by having nothing
data-dependent left to script.

Against the production ASE calculator on acetone, MACE-OFF23_medium: dE = 0.000e+00 eV,
max force difference 6.939e-17 eV/A, same 90 edges as MACE's own neighbour list.

What it does not do
-------------------
No periodic boundaries. `shifts` and `unit_shifts` are zero and `cell` is zero, which is
correct for an isolated molecule and wrong for anything else, so `build_system` refuses a
periodic request rather than silently returning a number.
"""
import numpy as np

#: OpenMM works in nm and kJ/mol; MACE in angstrom and eV.
NM_TO_A = 10.0
EV_TO_KJ_PER_MOL = 96.48533212331002


def _load_model(model_path, dtype):
    import torch
    model = torch.load(str(model_path), map_location="cpu", weights_only=False)
    model = model.to(dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class _MaceForceModule:
    """Factory for the TorchScript module OpenMM calls. Kept out of the class body so the
    scripted module closes over tensors rather than over this package."""

    @staticmethod
    def build(model, atomic_numbers, dtype):
        import torch

        z_table = [int(z) for z in model.atomic_numbers]
        n = len(atomic_numbers)
        one_hot = torch.zeros((n, len(z_table)), dtype=dtype)
        for i, z in enumerate(atomic_numbers):
            if int(z) not in z_table:
                raise ValueError(
                    "element Z={} is not in the model's table {} -- refusing to run a "
                    "potential outside the chemistry it was fitted to".format(z, z_table))
            one_hot[i, z_table.index(int(z))] = 1.0

        cutoff = float(model.r_max)
        # The complete graph, built ONCE. Every ordered pair i != j, in a fixed order, so
        # the module has no data-dependent shape and traces to a fixed computation.
        rows, cols = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        edge_index = torch.tensor([rows, cols], dtype=torch.long)

        class MaceForce(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model
                self.register_buffer("one_hot", one_hot)
                self.register_buffer("batch", torch.zeros(n, dtype=torch.long))
                self.register_buffer("ptr", torch.tensor([0, n], dtype=torch.long))
                self.register_buffer("head", torch.zeros(1, dtype=torch.long))
                self.register_buffer("cell", torch.zeros((3, 3), dtype=dtype))
                self.register_buffer(
                    "nm_to_a", torch.tensor(NM_TO_A, dtype=dtype))
                self.register_buffer(
                    "ev_to_kj", torch.tensor(EV_TO_KJ_PER_MOL, dtype=dtype))
                self.register_buffer("cutoff", torch.tensor(cutoff, dtype=dtype))
                self.register_buffer("edge_index", edge_index)
                self.register_buffer(
                    "zeros3", torch.zeros((edge_index.shape[1], 3), dtype=dtype))

            def forward(self, positions: torch.Tensor) -> torch.Tensor:
                # OpenMM hands nm; MACE wants angstrom.
                pos = positions.to(self.one_hot.dtype) * self.nm_to_a
                # No neighbour search at all: the graph is complete and precomputed, and
                # pairs beyond r_max are multiplied by a cutoff function that is exactly
                # zero. See the module docstring for the measurement.
                data = {
                    "positions": pos,
                    "node_attrs": self.one_hot,
                    "edge_index": self.edge_index,
                    "shifts": self.zeros3,
                    "unit_shifts": self.zeros3,
                    "cell": self.cell,
                    "batch": self.batch,
                    "ptr": self.ptr,
                    "head": self.head,
                }
                out = self.model(data, training=False, compute_force=False)
                energy = out["energy"]
                if energy is None:
                    raise RuntimeError("the MACE model returned no energy")
                return energy[0] * self.ev_to_kj

        return MaceForce()


def build_module(model_path, atomic_numbers, dtype=None, script=True,
                 example_positions_nm=None):
    """The TorchScript module OpenMM will evaluate. Returns (module, record).

    `script=True` TRACES rather than scripts, because e3nn is not scriptable. Tracing is
    sound here only because the edge list is static; if that ever stops being true this
    silently freezes a stale graph, so `verify_against_ase` compares the traced module
    against the production calculator and is the check that would catch it.
    """
    import torch
    dtype = dtype or torch.float64
    model = _load_model(model_path, dtype)
    module = _MaceForceModule.build(model, atomic_numbers, dtype)
    n = len(atomic_numbers)
    record = dict(model_path=str(model_path), cutoff_A=float(model.r_max),
                  n_atoms=int(n),
                  n_edges=int(n * (n - 1)),
                  neighbour_list="complete graph, precomputed and static (no binning, so "
                                 "no origin anchoring; pairs beyond r_max contribute "
                                 "exactly zero)",
                  dtype=str(dtype), traced=bool(script))
    if script:
        if example_positions_nm is None:
            raise ValueError("tracing needs example positions in nm")
        example = torch.tensor(np.asarray(example_positions_nm), dtype=dtype)
        with torch.no_grad():
            pass
        module = torch.jit.trace(module, example, check_trace=False)
    return module, record


def build_system(atomic_numbers, masses_amu, model_path, dtype=None, periodic=False,
                 example_positions_nm=None):
    """An OpenMM `System` whose only force is MACE. Returns (system, record)."""
    import openmm
    import openmmtorch
    from openmm import unit

    if periodic:
        raise ValueError(
            "periodic boundaries are not supported here. The graph is built with zero "
            "shifts and a zero cell, which is exact for an isolated molecule and wrong "
            "for anything else; refusing rather than returning a plausible number.")

    if example_positions_nm is None:
        raise ValueError("build_system needs example positions in nm, to trace with")
    module, record = build_module(model_path, atomic_numbers, dtype=dtype,
                                  example_positions_nm=example_positions_nm)
    system = openmm.System()
    for m in masses_amu:
        system.addParticle(float(m) * unit.amu)
    force = openmmtorch.TorchForce(module)
    force.setUsesPeriodicBoundaryConditions(False)
    system.addForce(force)
    record["n_particles"] = system.getNumParticles()
    return system, record


def verify_against_ase(atoms, model_path, dtype=None, tolerance_eV=1e-9):
    """Prove the OpenMM force is the same potential, by comparing to the ASE calculator.

    This is the acceptance criterion for this module and it is cheap, so there is no
    excuse for running production without it. It compares ENERGY AND FORCES, because a
    graph that lost an edge could still give a plausible energy.
    """
    import torch
    from openqha import engine

    dtype = dtype or torch.float64
    module, record = build_module(model_path, atoms.get_atomic_numbers(),
                                  dtype=dtype, script=False)
    _traced, _r = build_module(model_path, atoms.get_atomic_numbers(), dtype=dtype,
                               script=True,
                               example_positions_nm=atoms.get_positions() / NM_TO_A)
    pos_nm = torch.tensor(atoms.get_positions() / NM_TO_A, dtype=dtype,
                          requires_grad=True)
    energy_kj = module(pos_nm)
    energy_kj.backward()
    e_openmm_eV = float(energy_kj) / EV_TO_KJ_PER_MOL
    # dE/dx in kJ/mol/nm -> forces in eV/A
    f_openmm = (-pos_nm.grad.detach().numpy() / EV_TO_KJ_PER_MOL / NM_TO_A)

    calc, name, _prov = engine.calculator(device="cpu")
    a = atoms.copy()
    a.calc = calc
    e_ase = float(a.get_potential_energy())
    f_ase = a.get_forces()

    # The traced module is what production runs, so it is what has to agree.
    pos_t = torch.tensor(atoms.get_positions() / NM_TO_A, dtype=dtype,
                         requires_grad=True)
    e_traced = _traced(pos_t)
    e_traced.backward()
    f_traced = (-pos_t.grad.detach().numpy() / EV_TO_KJ_PER_MOL / NM_TO_A)

    record.update(
        engine=name,
        energy_traced_eV=float(e_traced) / EV_TO_KJ_PER_MOL,
        max_delta_force_traced_eV_per_A=float(np.abs(f_traced - f_openmm).max()),
        energy_openmm_eV=e_openmm_eV, energy_ase_eV=e_ase,
        delta_energy_eV=float(e_openmm_eV - e_ase),
        max_delta_force_eV_per_A=float(np.abs(f_openmm - f_ase).max()),
        agrees=bool(abs(e_openmm_eV - e_ase) < tolerance_eV
                    and np.abs(f_openmm - f_ase).max() < 1e-6
                    and np.abs(f_traced - f_ase).max() < 1e-6))
    return record
