# -*- coding: utf-8 -*-
"""Can this driver run the CUDA kernels OpenMM is about to build? Ask before, not after.

THE RULE, AND THE HALF OF IT THIS PROJECT HAD WRONG
---------------------------------------------------
CUDA "minor version compatibility" says an application built with a newer 12.x toolkit
runs on an older 12.x driver. `environment-tianhe-gpu.yml` cited it to justify
`cuda-version=12.3` against TianheXY-A's 12.2 driver, and for PyTorch and MACE it is
correct -- they ship **cubins**, already compiled for sm_80, and the driver loads them.

**It does not cover PTX.** A driver can load a cubin from a newer toolkit; it cannot
JIT-compile PTX from one, because the PTX ISA version is newer than its JIT compiler
knows. NVIDIA's own answer to this is "upgrade the driver" -- there is no compatibility
mode for it.

OpenMM's CUDA platform compiles **every kernel at run time**: nvrtc turns its CUDA C
into PTX and the driver JITs that PTX. So OpenMM is exactly the case minor version
compatibility excludes, and on 2026-09-12 all twelve branch B trajectories of three jobs
died identically at `openmm.Context(...)`:

    openmm.OpenMMException: Error loading CUDA module: CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)

after three minutes each of loading torch and tracing the model. Nothing in the job log
said so; the driver.log had it.

WHAT THIS MODULE MEASURES
-------------------------
Two numbers that can be read without a GPU context and without OpenMM:

    nvrtc     `nvrtcVersion()` from the libnvrtc that this environment will actually use
              -- the thing that decides which PTX ISA is emitted.
    driver    the CUDA version the installed driver supports, from
              `cuDriverGetVersion()` in libcuda, else the "CUDA Version:" field of
              `nvidia-smi`.

If nvrtc is newer than the driver, `check()` raises with both numbers and the fix.
Equal or older passes. Neither readable -> pass with a note, because refusing to run on
a machine we could not interrogate is worse than letting OpenMM produce its own error.

`S0_SKIP_GPU_PREFLIGHT=1` bypasses it.
"""
import ctypes
import os
import re
import subprocess

_SKIP_ENV = "S0_SKIP_GPU_PREFLIGHT"


def _nvrtc_version():
    """(major, minor) of the nvrtc this process would load, or None.

    The soname list is ordered newest-first only so that a machine carrying several is
    reported by the one a fresh `dlopen("libnvrtc.so")` would find first anyway.
    """
    for name in ("libnvrtc.so", "libnvrtc.so.12", "libnvrtc.so.11"):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        try:
            major = ctypes.c_int()
            minor = ctypes.c_int()
            if lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)) == 0:
                return (major.value, minor.value)
        except AttributeError:
            continue
    return None


def _driver_cuda_version():
    """(major, minor) of the CUDA version the DRIVER supports, or None.

    `cuDriverGetVersion` returns 1000*major + 10*minor and needs no context, no card and
    no initialisation beyond the library load -- so it works on a login node too.
    """
    for name in ("libcuda.so", "libcuda.so.1"):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        try:
            v = ctypes.c_int()
            if lib.cuDriverGetVersion(ctypes.byref(v)) == 0 and v.value:
                return (v.value // 1000, (v.value % 1000) // 10)
        except AttributeError:
            continue
    # nvidia-smi prints it in the header: "CUDA Version: 12.2"
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30)
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out.stdout or "")
        if m:
            return (int(m.group(1)), int(m.group(2)))
    except Exception:                                              # noqa: BLE001
        pass
    return None


def describe():
    """Both versions and the verdict, as data. Never raises."""
    nvrtc = _nvrtc_version()
    driver = _driver_cuda_version()
    ok, why = True, "not checked"
    if nvrtc and driver:
        ok = nvrtc <= driver
        why = "nvrtc {}.{} {} driver {}.{}".format(
            nvrtc[0], nvrtc[1], "<=" if ok else ">", driver[0], driver[1])
    elif not nvrtc:
        why = "libnvrtc not loadable here"
    elif not driver:
        why = "no CUDA driver readable here (a login node, usually)"
    return dict(nvrtc=nvrtc, driver_cuda=driver, ok=ok, reason=why,
                skipped=os.environ.get(_SKIP_ENV) == "1")


def check(platform="CUDA"):
    """Raise unless this driver can JIT the PTX nvrtc will emit. Returns describe().

    `platform` other than CUDA is a no-op: the CPU platform compiles nothing.
    """
    d = describe()
    if str(platform).upper() != "CUDA" or d["skipped"]:
        return d
    if d["ok"]:
        return d
    nv, dr = d["nvrtc"], d["driver_cuda"]
    raise RuntimeError(
        "this driver cannot run OpenMM's CUDA kernels.\n"
        "  nvrtc in this environment   {}.{}   <- emits PTX at this ISA version\n"
        "  CUDA supported by driver    {}.{}   <- its JIT compiler is older\n"
        "\n"
        "  OpenMM compiles every kernel at run time, so the driver must JIT that PTX.\n"
        "  CUDA minor version compatibility does NOT cover PTX JIT -- it covers cubins,\n"
        "  which is why torch and MACE load on this machine and OpenMM does not. Without\n"
        "  this check the failure is CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222) at\n"
        "  openmm.Context(), once per trajectory, minutes in.\n"
        "\n"
        "  Fix: bring the CUDA toolkit down to the driver's version --\n"
        "      mamba install -n openqha-gpu cuda-version={}.{}\n"
        "  and load the matching module (module load CUDA/{}.{}). Raising the driver is\n"
        "  the site's call, not ours.\n"
        "  {}=1 skips this check.".format(
            nv[0], nv[1], dr[0], dr[1], dr[0], dr[1], dr[0], dr[1], _SKIP_ENV))
