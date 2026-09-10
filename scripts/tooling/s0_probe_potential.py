#!/usr/bin/env python
"""Why does this potential return a non-finite energy? Answer it in one command.

TOOLING. Read-only, runs in seconds, safe on a login node. It exists because on
2026-09-09 and again on 2026-09-10 a Tianhe branch A run produced a non-finite energy on
**every** call (1628 then 1719 of them, 100%), and the same code with the same geometry
on the workstation returned -5259.489825324396 eV. Two things can produce that and they
need completely different fixes:

  * **the file** -- what is on disk is not a usable set of weights;
  * **the stack** -- the weights are fine and torch / e3nn / mace on this machine turn
    them into NaN.

The discriminator is cheap and it is step 3 below: **load the checkpoint and look at the
parameters themselves.** A tensor full of NaN in the file is the first case. Parameters
that are all finite, followed by a forward pass that is not, is the second.

Run it on BOTH machines and put the two outputs side by side::

    python scripts/tooling/s0_probe_potential.py

Exit status is 0 only if the potential returns a finite energy and finite forces.
"""
import sys
from pathlib import Path


def _repo_root():
    for p in Path(__file__).resolve().parents:
        if (p / "openqha" / "__init__.py").is_file():
            return p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))

#: One fixed molecule so the two machines are compared on the same input. This is the
#: exact acetone CREST handed the model on Tianhe, copied from the calcspace of job
#: 7347197 -- an ordinary geometry, C-C 1.52 A, C=O 1.21 A, nothing unusual about it.
SYMBOLS = ["C", "C", "C", "O", "H", "H", "H", "H", "H", "H"]
POSITIONS = [
    [-1.2681248863, 0.7031358649, 0.0064151880],
    [0.0000958229, -0.1143359431, -0.0055449952],
    [1.2673727530, 0.7047445624, -0.0021823544],
    [0.0007795035, -1.3178067377, -0.0165894488],
    [-1.2059059001, 1.4955296623, 0.7477351398],
    [-1.3982716110, 1.1640095224, -0.9713847976],
    [-2.1252228643, 0.0673254127, 0.2109085715],
    [2.1254595878, 0.0736896073, -0.2171899694],
    [1.2046689800, 1.5102483048, -0.7292046014],
    [1.3963396460, 1.1483492205, 0.9837167461],
]

#: Measured on this project's workstation, 2026-09-10, MACE-OFF23_medium, float64,
#: numpy 1.26.4 / torch 2.12.1 / e3nn 0.4.4 / mace 0.3.16. A machine that reproduces
#: these has a working potential; one that does not is what this script is for.
REFERENCE = dict(energy_eV=-5259.489825324396, max_force_eV_A=0.3852005576922286,
                 n_tensors=77, n_parameters=2265399, bytes=18350596)


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def energy_only():
    """Print one line: the acetone energy, or `ENERGY nan`. Used by the retry in 5(d).

    A separate process is the only honest way to test an instruction-set setting:
    `ATEN_CPU_CAPABILITY` and `DNNL_MAX_CPU_ISA` are read while torch is initialising, so
    setting them from inside a process that has already imported torch proves nothing.
    """
    from ase import Atoms
    from openqha.potentials import engine
    calc, _, _ = engine.calculator(device="cpu")
    a = Atoms(symbols=SYMBOLS, positions=POSITIONS)
    a.calc = calc
    try:
        print("ENERGY {!r}".format(float(a.get_potential_energy())))
    except Exception as exc:                                              # noqa: BLE001
        print("ENERGY raised {}: {}".format(type(exc).__name__, exc))
    return 0


def main():
    import os as _os
    if _os.environ.get("S0_PROBE_ENERGY_ONLY"):
        return energy_only()
    verdicts = []

    # ---- 1. the stack ------------------------------------------------------------------
    rule("1. WHAT IS INSTALLED HERE")
    import numpy
    import torch
    mods = [numpy, torch]
    for name in ("e3nn", "mace"):
        try:
            mods.append(__import__(name))
        except ImportError as exc:
            print("  {:8s} NOT IMPORTABLE: {}".format(name, exc))
    for m in mods:
        print("  {:8s} {:12s} {}".format(
            m.__name__, str(getattr(m, "__version__", "?")),
            str(getattr(m, "__file__", ""))))
    print("  python   {}".format(sys.executable))
    # openQHA rebinds MACE's neighbour-list construction (openqha/potentials/mace_patch.py).
    # Whether that rebinding takes hold depends on the MACE tree that is installed, so it
    # is part of "the stack" and a candidate for a machine-specific NaN. Print it here
    # rather than leave it to be guessed at.
    try:
        from openqha.potentials import mace_patch
        mace_patch.apply(strict=False)
        st = mace_patch.state()
        print("  patch    applied={}  sites={}  error={}".format(
            st.get("applied"), st.get("sites"), st.get("error")))
    except Exception as exc:                                              # noqa: BLE001
        print("  patch    could not be inspected: {}: {}".format(type(exc).__name__, exc))

    # ---- 2. the file -------------------------------------------------------------------
    rule("2. WHICH FILE")
    from openqha.potentials import engine
    name = engine.engine_name()
    path = engine.model_path(name)
    size = path.stat().st_size
    print("  engine        {}".format(name))
    print("  path          {}".format(path))
    print("  bytes         {}   (workstation: {})".format(size, REFERENCE["bytes"]))
    if size != REFERENCE["bytes"]:
        print("  NOTE: a different size does NOT by itself mean different weights -- a")
        print("        torch re-serialisation changes the size while every tensor stays")
        print("        identical. Step 3 is the one that decides.")

    # ---- 3. THE DISCRIMINATOR ----------------------------------------------------------
    rule("3. ARE THE PARAMETERS IN THE FILE FINITE?")
    print("  (this is the step that separates 'bad file' from 'bad stack')")
    bad = []
    n_tensors = 0
    n_elements = 0
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
        state = obj.state_dict() if hasattr(obj, "state_dict") else obj
        for key in sorted(state):
            v = state[key]
            if not hasattr(v, "dtype") or not v.is_floating_point():
                continue
            n_tensors += 1
            n_elements += v.numel()
            if not bool(torch.isfinite(v).all()):
                n_bad = int((~torch.isfinite(v)).sum())
                bad.append((key, tuple(v.shape), n_bad, v.numel()))
    except Exception as exc:                                              # noqa: BLE001
        print("  THE FILE COULD NOT BE LOADED AT ALL: {}: {}".format(
            type(exc).__name__, exc))
        print()
        print("  VERDICT: the file is not a usable checkpoint. Re-copy it.")
        return 1

    print("  floating-point tensors  {}   (workstation: {})".format(
        n_tensors, REFERENCE["n_tensors"]))
    print("  parameters              {}".format(n_elements))
    if bad:
        print("  **NON-FINITE PARAMETERS IN THE FILE ITSELF**")
        for key, shape, n_bad, total in bad[:10]:
            print("    {:50s} {} of {} bad".format(key + " " + str(shape), n_bad, total))
        print()
        print("  VERDICT: **THE FILE IS THE PROBLEM.** The weights on disk contain NaN or")
        print("           Inf before anything is computed with them. No version of torch")
        print("           can fix that. Re-copy the model file.")
        return 1
    print("  every parameter is finite")
    verdicts.append("the file's parameters are clean")

    # ---- 4. the forward pass -----------------------------------------------------------
    rule("4. WHAT COMES OUT FOR ONE FIXED ACETONE")
    from ase import Atoms
    ok = True
    for dtype in ("float64", "float32"):
        try:
            engine._CACHE.clear()
            saved, engine.DTYPE = engine.DTYPE, dtype
            calc, _, _ = engine.calculator(device="cpu")
            engine.DTYPE = saved
            atoms = Atoms(symbols=SYMBOLS, positions=POSITIONS)
            atoms.calc = calc
            e = float(atoms.get_potential_energy())
            f = atoms.get_forces()
            fin = (e == e and abs(e) != float("inf") and bool(numpy.isfinite(f).all()))
            print("  {:8s} energy {!r:>24}   forces finite {}".format(dtype, e, fin))
            if dtype == "float64":
                if fin:
                    d = e - REFERENCE["energy_eV"]
                    print("           workstation {!r}   difference {:.3e} eV".format(
                        REFERENCE["energy_eV"], d))
                else:
                    ok = False
        except Exception as exc:                                          # noqa: BLE001
            print("  {:8s} RAISED {}: {}".format(dtype, type(exc).__name__, exc))
            ok = False

    # ---- 5. where does it first go wrong -----------------------------------------------
    # Runs on failure, and on demand: S0_PROBE_ALWAYS_DEEP=1 exercises it on a machine
    # where the potential works, which is how the checks below are known to be correct
    # rather than merely written.
    import os as _os
    if not ok or _os.environ.get("S0_PROBE_ALWAYS_DEEP"):
        rule("5. WHERE THE NaN FIRST APPEARS")
        print("  The parameters are clean, so something between the coordinates and the")
        print("  energy produces it. These three checks say which layer.")

        # (0) THE FLOOR: plain arithmetic, no MACE, no e3nn, no model.
        #     Added 2026-09-10 after numpy, torch, e3nn, mace, matscipy, the BLAS provider
        #     and the OpenMP runtime were all matched to a machine that works and the NaN
        #     stayed. When every library version agrees, what is left below them is the
        #     CPU and the kernel OpenBLAS selects for it AT RUNTIME. If a matmul is wrong,
        #     nothing above it can be right, and no version pin will fix it.
        print()
        print("  (0) plain arithmetic -- no MACE, no e3nn, no model")
        try:
            cpu, flags = "", set()
            try:
                with open("/proc/cpuinfo") as fh:
                    for line in fh:
                        low = line.lower()
                        if not cpu and low.startswith("model name"):
                            cpu = line.split(":", 1)[1].strip()
                        elif not flags and low.startswith("flags"):
                            flags = set(line.split(":", 1)[1].split())
                        if cpu and flags:
                            break
            except OSError:
                pass
            print("      cpu                  {}".format(cpu or "unknown"))
            # AVX-512 is the difference that survived every version comparison: this
            # project's workstation (Raptor Lake i7) does not have it; a Xeon Platinum
            # 8358P does, and torch dispatches different kernels on it.
            avx512 = sorted(f for f in flags if f.startswith("avx512"))
            print("      avx512               {}".format(
                " ".join(avx512[:6]) + (" ..." if len(avx512) > 6 else "")
                if avx512 else "absent"))
            for var in ("ATEN_CPU_CAPABILITY", "DNNL_MAX_CPU_ISA",
                        "OPENBLAS_CORETYPE", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
                print("      {:20s} {}".format(var, _os.environ.get(var, "[not set]")))
            info = torch.__config__.parallel_info()
            for line in info.splitlines():
                if "parallel backend" in line or "get_num_threads" in line:
                    print("      {}".format(line.strip()))

            torch.manual_seed(0)
            a = torch.randn(256, 256, dtype=torch.float64)
            b = torch.randn(256, 256, dtype=torch.float64)
            t_mm = a @ b
            n_mm = torch.from_numpy(numpy.asarray(a.numpy()) @ numpy.asarray(b.numpy()))
            mm_finite = bool(torch.isfinite(t_mm).all())
            mm_err = float((t_mm - n_mm).abs().max()) if mm_finite else float("nan")
            print("      torch matmul finite  {}".format(mm_finite))
            print("      vs numpy matmul      max|diff| = {:.3e}".format(mm_err))
            v = torch.randn(1000, dtype=torch.float64)
            print("      exp/sum/sqrt finite  {}".format(bool(
                torch.isfinite(v.exp().sum()).all()
                and torch.isfinite(v.abs().sqrt().sum()).all())))
            ev = torch.linalg.eigvalsh(a @ a.T)
            print("      eigvalsh finite      {}".format(bool(torch.isfinite(ev).all())))
            if not mm_finite or not (mm_err == mm_err and mm_err < 1e-8):
                print()
                print("      **THE LINEAR ALGEBRA ITSELF IS WRONG ON THIS MACHINE.**")
                print("      No package version above this can fix it. This is OpenBLAS")
                print("      picking a kernel for this CPU. Try, in order:")
                print("        OPENBLAS_CORETYPE=Haswell python ...   (force a safe kernel)")
                print("        OPENBLAS_NUM_THREADS=1     python ...   (rule out threading)")
                print("        mamba install 'libopenblas=*=pthreads*' (other build)")
        except Exception as exc:                                          # noqa: BLE001
            print("      could not be tested: {}: {}".format(type(exc).__name__, exc))

        # (a) the neighbour list. r = 0 in a radial basis is the classic NaN source, and
        #     the neighbour search is the one part openQHA replaces (mace_patch).
        print()
        print("  (a) neighbour list")
        try:
            from mace.data.neighborhood import get_neighborhood
            pos = numpy.array(POSITIONS, dtype=float)
            edge_index, shifts, unit_shifts, _cell = get_neighborhood(
                positions=pos, cutoff=5.0, pbc=(False, False, False), cell=None)
            ei = numpy.asarray(edge_index)
            d = numpy.linalg.norm(pos[ei[1]] - pos[ei[0]], axis=1)
            print("      edges {}   r_min {:.6f} A   r_max {:.6f} A".format(
                ei.shape[1], d.min(), d.max()))
            print("      zero-length edges {}   non-finite {}".format(
                int((d == 0).sum()), int((~numpy.isfinite(d)).sum())))
            if (d == 0).any():
                print("      **r = 0 EDGES** -- the radial basis divides by this. This is")
                print("      the defect; it is in the neighbour search, not in torch.")
        except Exception as exc:                                          # noqa: BLE001
            print("      could not be built: {}: {}".format(type(exc).__name__, exc))

        # (b) e3nn on its own, with no MACE and no weights involved at all. If these are
        #     NaN then the equivariant primitives are broken against this torch, and
        #     nothing about the model or the molecule matters.
        print()
        print("  (b) e3nn primitives, no MACE, no weights")
        try:
            from e3nn import o3
            torch.manual_seed(0)
            vec = torch.randn(8, 3, dtype=torch.float64)
            sh = o3.spherical_harmonics([0, 1, 2, 3], vec, normalize=True,
                                        normalization="component")
            print("      spherical_harmonics  finite={}  max|.|={:.6g}".format(
                bool(torch.isfinite(sh).all()), float(sh.abs().max())))
            irr = o3.Irreps("4x0e + 4x1o")
            tp = o3.FullyConnectedTensorProduct(irr, irr, irr).to(torch.float64)
            x = torch.randn(8, irr.dim, dtype=torch.float64)
            out = tp(x, x)
            print("      tensor product       finite={}  max|.|={:.6g}".format(
                bool(torch.isfinite(out).all()), float(out.abs().max())))
            # MACE builds every `U_matrix_*` buffer from these Clebsch-Gordan
            # coefficients (mace.tools.cg.U_matrix_real -> e3nn.o3.wigner_3j), and e3nn
            # reads them from a cached `constants.pt` with torch.load -- the same load the
            # TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD warning fires on. If these are wrong, every
            # symmetric contraction is wrong, and no weight file is implicated.
            w3 = [o3.wigner_3j(a_, b_, c_) for a_, b_, c_ in
                  ((1, 1, 0), (1, 1, 2), (2, 2, 2), (1, 2, 3))]
            w3_ok = all(bool(torch.isfinite(x).all()) for x in w3)
            print("      wigner_3j            finite={}  max|.|={:.6g}".format(
                w3_ok, max(float(x.abs().max()) for x in w3) if w3_ok else float("nan")))
            if not w3_ok:
                print("      **THE CLEBSCH-GORDAN COEFFICIENTS THEMSELVES ARE NaN.**")
                print("      e3nn loads them from its cached constants.pt. Delete e3nn's")
                print("      cache and reinstall e3nn; the weight file is not involved.")
            if not bool(torch.isfinite(sh).all()) or not bool(torch.isfinite(out).all()):
                print("      **e3nn ITSELF PRODUCES NaN on this torch.** The model is")
                print("      irrelevant -- change torch, not the weights.")
        except Exception as exc:                                          # noqa: BLE001
            print("      raised: {}: {}".format(type(exc).__name__, exc))

        # (c) which layer emits the first non-finite value.
        #     Part of the model CANNOT be hooked: e3nn compiles its tensor products with
        #     `torch.jit.script` **at load time**, using whatever torch is installed, and
        #     hooks are not allowed on a ScriptModule. That is worth knowing on its own --
        #     those compiled pieces are generated fresh against this torch, so they are
        #     the part of the stack most exposed to a torch version change, and they are
        #     also the part this section cannot see inside. Check (b) is the test for
        #     them: it exercises the same e3nn machinery with no MACE and no weights.
        print()
        print("  (c) where the NaN is BORN, not merely where it is seen")
        try:
            from ase import Atoms as _Atoms
            engine._CACHE.clear()
            calc, _, _ = engine.calculator(device="cpu")
            model = calc.models[0]
            names = {id(m): n for n, m in model.named_modules()}

            # (c1) the live model's own tensors. Section 3 read the FILE; this reads what
            #      is in memory after loading, dtype conversion and e3nn's code
            #      regeneration. MACE builds `U_matrix_*` from Wigner coefficients and
            #      registers it as a buffer, so a buffer that is finite on disk and
            #      non-finite here would be produced by the load, not by the file.
            badt = []
            for kind, it in (("param", model.named_parameters()),
                             ("buffer", model.named_buffers())):
                for nm, v in it:
                    if (torch.is_tensor(v) and v.is_floating_point()
                            and not bool(torch.isfinite(v).all())):
                        badt.append("{} {} {}".format(kind, nm, tuple(v.shape)))
            print("      live tensors non-finite: {}".format(
                len(badt) if badt else "none"))
            for b in badt[:6]:
                print("        {}".format(b))

            # (c2) input AND output, so the ORIGIN can be told from the propagation.
            hookable, scripted = [], 0
            for m in model.modules():
                if m is model:
                    continue
                if isinstance(m, torch.jit.ScriptModule):
                    scripted += 1
                else:
                    hookable.append(m)
            seen = []

            def anybad(o):
                if torch.is_tensor(o):
                    return o.is_floating_point() and not bool(torch.isfinite(o).all())
                if isinstance(o, (list, tuple)):
                    return any(anybad(x) for x in o)
                if isinstance(o, dict):
                    return any(anybad(x) for x in o.values())
                return False

            def hook(mod, inp, out):
                seen.append((mod, anybad(inp), anybad(out)))

            handles = [m.register_forward_hook(hook) for m in hookable]
            a = _Atoms(symbols=SYMBOLS, positions=POSITIONS)
            a.calc = calc
            try:
                a.get_potential_energy()
            except Exception:                                             # noqa: BLE001
                pass
            for h in handles:
                h.remove()

            print("      modules: {} hookable, {} TorchScript (e3nn codegen, opaque)"
                  .format(len(hookable), scripted))
            born = [(m, i, o) for (m, i, o) in seen if o and not i]
            propagated = [(m, i, o) for (m, i, o) in seen if o and i]
            if born:
                print("      **NaN IS BORN HERE** (input finite, output not):")
                for m, _i, _o in born[:3]:
                    print("        {}  ({})".format(
                        names.get(id(m), "?"), type(m).__name__))
                print("      That module computes it. Everything after is propagation.")
            elif propagated:
                print("      every non-finite module ALREADY RECEIVED a non-finite input.")
                print("      The first one seen was:")
                for m, _i, _o in propagated[:2]:
                    print("        {}  ({})".format(
                        names.get(id(m), "?"), type(m).__name__))
                print("      So the origin is UPSTREAM of it, inside the {} TorchScript"
                      .format(scripted))
                print("      modules this hook cannot see into.")
            else:
                print("      no hookable module emitted a non-finite value at all.")

        except Exception as exc:                                          # noqa: BLE001
            print("      could not be traced: {}: {}".format(type(exc).__name__, exc))

        # (d) THE FIX, TESTED RATHER THAN SUGGESTED.
        #     The failing layer is MACE's SymmetricContraction, whose contractions are
        #     `torch.fx` GraphModules built by opt_einsum_fx -- large einsums, not the
        #     small matmul check (0) exercises. When every package version matches a
        #     machine that works, the remaining difference is the CPU: this project's
        #     workstation is a Raptor Lake i7 with **no AVX-512**, and a Xeon Platinum
        #     8358P (Ice Lake-SP) has it. torch dispatches different kernels for it.
        #
        #     Both variables below are read while torch initialises, so each has to be
        #     tried in a FRESH PROCESS. That is what makes this a test and not a guess.
        #
        #     Only when the baseline actually failed. On a machine where section 4 already
        #     returned a finite energy, every candidate would come back finite too and the
        #     word FIXED would mean nothing.
        print()
        print("  (d) does forcing a narrower instruction set fix it?")
        if ok:
            print("      skipped -- section 4 already returned a finite energy here, so")
            print("      there is nothing for these settings to fix.")
        else:
            try:
                import subprocess
                fixed = None
                for var, val in (("ATEN_CPU_CAPABILITY", "avx2"),
                                 ("ATEN_CPU_CAPABILITY", "default"),
                                 ("DNNL_MAX_CPU_ISA", "AVX2"),
                                 ("OMP_NUM_THREADS", "1")):
                    env = dict(_os.environ)
                    env[var] = val
                    env["S0_PROBE_ENERGY_ONLY"] = "1"
                    try:
                        r = subprocess.run([sys.executable, __file__], env=env,
                                           capture_output=True, text=True, timeout=900)
                        line = [x for x in r.stdout.splitlines()
                                if x.startswith("ENERGY")]
                        got = line[-1][7:] if line else "(no answer)"
                    except Exception as exc:                          # noqa: BLE001
                        got = "{}: {}".format(type(exc).__name__, exc)
                    good = ("nan" not in got.lower() and "raised" not in got
                            and "no answer" not in got)
                    print("      {:34s} -> {}{}".format(
                        var + "=" + val, got, "   **FIXED**" if good else ""))
                    if good:
                        fixed = (var, val)
                        break
                if fixed:
                    print()
                    print("      Put it in hpc/env/common.sh so every job gets it:")
                    print("        export {}={}".format(*fixed))
                else:
                    print()
                    print("      None of them fixed it. The instruction set is not the")
                    print("      difference, or not the only one.")
            except Exception as exc:                                  # noqa: BLE001
                print("      could not be tested: {}: {}".format(
                    type(exc).__name__, exc))

    # ---- 6. the verdict ----------------------------------------------------------------
    rule("6. VERDICT")
    if ok:
        print("  The potential works on this machine. If a run still produces non-finite")
        print("  energies, the difference is not the model -- look at what is being sent")
        print("  to it (S0_MACE_TRACE records every call).")
        return 0
    print("  **THE STACK IS THE PROBLEM, NOT THE FILE.**")
    print("  Every parameter in the checkpoint is finite, and the forward pass still is")
    print("  not. That is torch / e3nn / mace on this machine, not the weights.")
    print()
    print("  Known-good on this project's workstation:")
    print("    numpy 1.26.4   torch 2.12.1   e3nn 0.4.4   mace 0.3.16")
    print("  Section 5 says which layer to blame. Change ONE thing and rerun this")
    print("  script -- it takes seconds, and it is the whole test.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
