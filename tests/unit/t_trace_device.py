"""A traced MACE module must be built for the device OpenMM will evaluate it on.

UNIT. Imports torch, traces a three-line module, reads two source files. Seconds.

The defect this is written on
-----------------------------
Measured on an45 (A800) 2026-09-12. Sections 1-5 of the CUDA probe passed: the driver,
the PTX, the numpy/torch ABI, a CUDA context, and a 61-second trace of MACE-OFF23_medium.
The first force evaluation then died inside TorchScript:

    energies = torch.to(_1, torch.device("cpu"), 7)
    return torch.matmul(x, energies)
    RuntimeError: Expected all tensors to be on the same device, but found at least
    two devices, cuda:0 and cpu!  (argument mat2 in wrapper_CUDA_mm)

MACE's AtomicEnergiesBlock writes `atomic_energies.to(x.device)`. `torch.jit.trace` does
not record "the device of x" -- it records the device x HAPPENED TO BE ON, as a literal.
The module was traced on the CPU and handed to OpenMM's CUDA platform, so every position
arrived on cuda:0 and met a constant that said cpu.

Nothing was wrong with the model, the versions, or the GPU. The graph was built for one
device and run on another, and the only place that is visible before a force is evaluated
is the graph itself.

What each check is for
----------------------
  * A. `torch_device_for_platform` maps platform -> device, and REFUSES rather than
    silently tracing on the CPU when CUDA is asked for and torch cannot see a card.
    A silent fallback here is exactly the 61-seconds-then-fail path above.
  * B. `_device_constants` actually reads a baked constant out of a traced graph. If it
    silently returned an empty set, the audit in `build_module` would pass everything.
  * C. **Every `build_system` call passes `platform`.** This is the check that would have
    caught the defect: the bug was not in any formula, it was a missing argument, and a
    caller added later with the default would fail the same way. Source-level, so it
    needs no GPU and no model.
"""
import ast
import sys
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from openqha.quasi_harmonic import openmm_mace              # noqa: E402

FAIL = []

#: Every file that builds an OpenMM system must name the platform it will run on.
CALLERS = [
    Path("scripts") / "production" / "s0_B_qha_trajectory_openmm.py",
    Path("scripts") / "tooling" / "s0_probe_openmm_cuda.py",
]


def check_platform_mapping():
    """A. the map, and the refusal."""
    print("A. platform -> torch device")
    import torch
    for platform in ("CPU", "Reference", "OpenCL", "cpu"):
        got = openmm_mace.torch_device_for_platform(platform, torch)
        print("  {:10s} -> {}".format(platform, got))
        if got.type != "cpu":
            FAIL.append("{} mapped to {}, expected cpu".format(platform, got))
    if torch.cuda.is_available():
        got = openmm_mace.torch_device_for_platform("CUDA", torch)
        print("  {:10s} -> {}".format("CUDA", got))
        if got.type != "cuda":
            FAIL.append("CUDA mapped to {}, expected a cuda device".format(got))
        # Index 0 and not a physical index: Slurm renumbers allocated cards from 0.
        if got.index not in (0, None):
            FAIL.append("CUDA mapped to index {}, expected 0".format(got.index))
    else:
        try:
            openmm_mace.torch_device_for_platform("CUDA", torch)
            FAIL.append("CUDA was accepted with no card visible; tracing would have "
                        "fallen back to the CPU and failed at the first force")
            print("      FAIL  CUDA accepted with no card visible")
        except RuntimeError as exc:
            print("  {:10s} -> refused: {}".format("CUDA", str(exc).split(".")[0]))


def check_device_constants():
    """B. the audit sees the hazard, and ONLY the hazard.

    Both halves matter, and the second was learned the hard way: a first version
    flagged every device constant anywhere in the graph, including the cpu
    `torch.empty` that e3nn's scripted tensor products use for shape arithmetic, and
    refused a MACE build on an A800 whose trace was correct. A guard that blocks
    working runs is worse than no guard, because it is confident while being wrong.
    """
    print("\nB. the audit separates a baked .to() from shape-only scaffolding")
    import torch

    class Baked(torch.nn.Module):
        """The shape of MACE's AtomicEnergiesBlock: a buffer moved to x's device."""

        def __init__(self):
            super().__init__()
            self.register_buffer("w", torch.ones(3, dtype=torch.float64))

        def forward(self, x):
            return (x * self.w.to(dtype=x.dtype, device=x.device)).sum()

    class ShapeOnly(torch.nn.Module):
        """The shape of e3nn's generated code: a factory tensor, never a value."""

        def forward(self, x):
            pad = torch.zeros(3, dtype=x.dtype)
            return (x + pad).sum()

    ex = torch.zeros(3, dtype=torch.float64)

    hazards, devices = openmm_mace._device_constants(
        torch.jit.trace(Baked(), ex, check_trace=False))
    print("  buffer .to(x.device)   hazards {}  devices {}".format(
        len(hazards), devices))
    if not any(d == "cpu" for _n, d, _l in hazards):
        FAIL.append("a module traced on the CPU with .to(x.device) was not flagged; "
                    "build_module would pass a graph that cannot run on a GPU")
        print("      FAIL  the audit cannot see the constant it exists to catch")

    hazards2, devices2 = openmm_mace._device_constants(
        torch.jit.trace(ShapeOnly(), ex, check_trace=False))
    print("  factory tensor only    hazards {}  devices {}".format(
        len(hazards2), devices2))
    if hazards2:
        FAIL.append("a factory tensor with no .to() was reported as a hazard; this is "
                    "the false positive that refused a correct MACE build on an A800")
        print("      FAIL  shape-only scaffolding flagged as a device mismatch")
    if "cpu" not in devices2:
        FAIL.append("the informational device list lost the factory constant; it is "
                    "reported, just not fatal")


def check_callers_pass_platform():
    """C. no caller may rely on the default."""
    print("\nC. every build_system call names its platform")
    for rel in CALLERS:
        path = ROOT / rel
        if not path.is_file():
            FAIL.append("{} is missing".format(rel))
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "build_system"]
        if not calls:
            FAIL.append("{} calls build_system nowhere; this test is watching the "
                        "wrong file".format(rel))
            print("      FAIL  {}: no build_system call found".format(rel))
            continue
        for call in calls:
            named = [k.arg for k in call.keywords]
            ok = "platform" in named
            print("  {:<52s} line {:<5d} {}".format(
                rel.name, call.lineno, "platform passed" if ok else "NO platform"))
            if not ok:
                FAIL.append("{}:{} calls build_system without platform=; it would trace "
                            "for the CPU and fail at the first force on a GPU"
                            .format(rel.as_posix(), call.lineno))


def main():
    check_platform_mapping()
    check_device_constants()
    check_callers_pass_platform()
    print()
    if FAIL:
        print("{} problem(s):".format(len(FAIL)))
        for f in FAIL:
            print("  - " + f)
        return 1
    print("traced modules are built for the platform that will evaluate them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
