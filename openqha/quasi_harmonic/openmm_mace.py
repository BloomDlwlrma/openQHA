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
import os

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


def torch_device_for_platform(platform, torch=None):
    """The torch device a module traced for this OpenMM platform must be built on.

    **Tracing bakes devices in as literal constants.** MACE's AtomicEnergiesBlock does
    atomic_energies.to(x.device); torch.jit.trace sees whatever device the example
    tensor was on and writes that into the graph:

        energies = torch.to(_1, torch.device("cpu"), 7)
        return torch.matmul(x, energies)

    Run that under OpenMM's CUDA platform, where positions arrive on cuda:0, and the
    matmul gets one operand on each device. Measured on an45 2026-09-12, at the first
    force evaluation, after a 61 s trace:

        RuntimeError: Expected all tensors to be on the same device, but found at
        least two devices, cuda:0 and cpu! (... argument mat2 in wrapper_CUDA_mm)

    Nothing about the model or the versions was wrong -- the graph was built for one
    device and run on another.

    cuda:0 rather than a physical index: Slurm renumbers the allocated cards from 0
    inside the step, and openqha sets CUDA_VISIBLE_DEVICES per worker, so device 0 is
    always this process's own card.
    """
    if torch is None:
        import torch
    name = str(platform).upper()
    if name != "CUDA":
        # CPU and Reference hand the module CPU tensors, and OpenCL has no torch
        # counterpart, so openmm-torch evaluates the module on the CPU there too.
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "the OpenMM CUDA platform was requested, so the MACE module must be traced "
            "on a CUDA device, but torch.cuda.is_available() is False. Tracing on the "
            "CPU instead would succeed here and then fail at the first force evaluation "
            "with 'Expected all tensors to be on the same device'.")
    return torch.device("cuda:0")


def platform_properties_for(platform, dtype=None):
    """The Context properties that match a MACE module of this dtype. Measured.

    **OpenMM's CUDA `Precision` defaults to `single`, and a float64 module under it is
    the slowest configuration there is.** On an A800 80 GB (an45, 2026-09-12,
    MACE-OFF23_medium, 10 atoms, 200 steps), everything else held fixed:

        platform  MACE dtype  Precision          ms/step
        CUDA      float64     single (default)   74.2      <- what you get by accident
        CUDA      float64     double             39.2      <- 1.9x faster, same answer
        CUDA      float32     single             25.9
        CPU       float64     n/a                61.7      (112 threads)
        CPU       float32     n/a                34.9      (112 threads)

    So the card is 1.6x FASTER than the whole node's CPU in float64 once the precisions
    agree, and 1.3x faster in float32 -- the opposite of the conclusion drawn from the
    mismatched row alone, which is the row this function exists to stop anyone reaching
    by default.

    The double/double row is also exact against float64 on the CPU: dE = 0.0 kJ/mol and
    max|dF| = 2.3e-10 kJ/mol/nm (3.9e-14 relative), so nothing is traded for the speed.

    Why the mismatch costs so much is NOT measured here. The plausible reading is that
    openmm-torch must convert positions and forces every step when the platform's arrays
    are float32 and the module wants float64. The test that would settle it is float32
    with Precision=double -- a mismatch the other way, which should be slow for the same
    reason. Until someone runs it, treat the mechanism as unverified and the timings as
    what they are: measured.
    """
    if str(platform).upper() != "CUDA":
        # CPU exposes only Threads and DeterministicForces; Reference exposes nothing.
        return {}
    import torch
    dtype = dtype or torch.float64
    return {"Precision": "single" if dtype == torch.float32 else "double"}


def _device_constants(module):
    """Device constants in a traced graph: (hazards, all_devices).

    A hazard is a device constant inside an `aten::to` call, and ONLY that. The
    distinction is not cosmetic -- measured 2026-09-12, both ways:

      * TRACING freezes the device it saw. A module whose source says
        `self.w.to(dtype=x.dtype, device=x.device)` traces to
        `torch.to(w, torch.device("cpu"), 6)`. That is the MACE AtomicEnergiesBlock
        line that killed a CUDA run at the first force evaluation.
      * SCRIPTING keeps it dynamic: the same source scripts to
        `torch.to(w, ops.prim.device(x), _0)`, with no constant at all.

    So a `torch.device("cpu")` inside a scripted submodule was WRITTEN THAT WAY on
    purpose. e3nn's generated tensor products carry
    `torch.empty([], device=torch.device("cpu"))`, used only to broadcast shapes --
    never to hold values -- and they read cpu on every machine, which is why MACE runs
    on GPUs everywhere. A first version of this audit flagged those fourteen lines and
    refused a build whose trace was correct; blocking a working run is a worse failure
    than the one being guarded against, because it is silent about being wrong.

    Factory calls (zeros/ones/empty/arange) are therefore reported but not fatal: traced
    on CUDA they bake cuda, so a cpu one left over is deliberate source, not this bug.
    """
    hazards, seen = [], set()
    try:
        mods = list(module.named_modules())
    except Exception:                                             # noqa: BLE001
        mods = [("<top>", module)]
    for name, m in mods:
        try:
            code = str(m.code)
        except Exception:                                         # noqa: BLE001
            continue
        for line in code.splitlines():
            if 'torch.device("' not in line:
                continue
            for piece in line.split('torch.device("')[1:]:
                dev = piece.split('"')[0]
                seen.add(dev)
                if "torch.to(" in line:
                    hazards.append((name or "<top>", dev, line.strip()))
    return hazards, sorted(seen)


def build_module(model_path, atomic_numbers, dtype=None, script=True,
                 example_positions_nm=None, device=None):
    """The TorchScript module OpenMM will evaluate. Returns (module, record).

    `script=True` TRACES rather than scripts, because e3nn is not scriptable. Tracing is
    sound here only because the edge list is static; if that ever stops being true this
    silently freezes a stale graph, so `verify_against_ase` compares the traced module
    against the production calculator and is the check that would catch it.
    """
    import torch
    dtype = dtype or torch.float64
    device = torch.device(device) if device is not None else torch.device("cpu")
    model = _load_model(model_path, dtype)
    module = _MaceForceModule.build(model, atomic_numbers, dtype)
    # Weights, the precomputed graph buffers and the example positions must all be on
    # the target device BEFORE tracing -- see torch_device_for_platform().
    module = module.to(device)
    n = len(atomic_numbers)
    record = dict(model_path=str(model_path), cutoff_A=float(model.r_max),
                  n_atoms=int(n),
                  n_edges=int(n * (n - 1)),
                  neighbour_list="complete graph, precomputed and static (no binning, so "
                                 "no origin anchoring; pairs beyond r_max contribute "
                                 "exactly zero)",
                  dtype=str(dtype), traced=bool(script), trace_device=str(device))
    if script:
        if example_positions_nm is None:
            raise ValueError("tracing needs example positions in nm")
        example = torch.tensor(np.asarray(example_positions_nm), dtype=dtype,
                               device=device)
        module = torch.jit.trace(module, example, check_trace=False)
        # Read back what the trace wrote, rather than trusting that moving the module
        # was enough. A graph carrying a cpu constant cannot run on CUDA.
        hazards, devices = _device_constants(module)
        record["device_constants"] = devices
        wrong = [(n, d, line) for (n, d, line) in hazards
                 if not d.startswith(device.type)]
        record["device_mismatches"] = ["{} -> {}".format(n, d) for n, d, _ in wrong]
        if wrong and os.environ.get("S0_ALLOW_DEVICE_CONSTANTS") != "1":
            raise RuntimeError(
                "the traced graph moves tensors to a device that is not the one it will "
                "run on ({}).\n{}\n"
                "torch.jit.trace records the device it saw as a literal, so this module "
                "would fail at the first force evaluation with 'Expected all tensors to "
                "be on the same device'. Trace on the device OpenMM will use, by passing "
                "build_system(..., platform=...). S0_ALLOW_DEVICE_CONSTANTS=1 overrides "
                "this check.\n"
                "(Device constants seen anywhere in the graph: {}. Constants outside a "
                ".to() call are not counted: e3nn's scripted tensor products carry a cpu "
                "torch.empty used only for shape arithmetic.)".format(
                    device,
                    "\n".join("    {}  {}".format(n, line) for n, _, line in wrong),
                    ", ".join(devices)))
    return module, record


def build_system(atomic_numbers, masses_amu, model_path, dtype=None, periodic=False,
                 example_positions_nm=None, platform="CPU"):
    """An OpenMM `System` whose only force is MACE. Returns (system, record).

    `platform` is the OpenMM platform the caller will put its Context on, and it is not
    optional in practice: the traced module is device-specific. Passing the wrong one
    fails at the first force evaluation, not at build time.
    """
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
    import torch
    module, record = build_module(
        model_path, atomic_numbers, dtype=dtype,
        example_positions_nm=example_positions_nm,
        device=torch_device_for_platform(platform, torch))
    record["openmm_platform"] = str(platform)
    record["platform_properties"] = platform_properties_for(platform, dtype or torch.float64)
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
