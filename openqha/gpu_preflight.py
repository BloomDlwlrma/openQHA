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


#: Filled by `_nvrtc_version()`: the file `dlopen` actually resolved to.
#: **Which one it is matters more than it looks.** `module load CUDA/12.3` puts that
#: toolkit's lib directory ahead of the conda environment's on LD_LIBRARY_PATH, so the
#: nvrtc in use can be the module's while the conda pin says something else. On
#: 2026-09-12 the fix that mattered was changing the MODULE from 12.3 to 12.2; the conda
#: `cuda-version` pin never got installed (the network was down) and the PTX error went
#: away regardless. A version with no path is half a measurement.
NVRTC_PATH = None


def _nvrtc_version():
    """(major, minor) of the nvrtc this process would load, or None.

    The soname list is ordered newest-first only so that a machine carrying several is
    reported by the one a fresh `dlopen("libnvrtc.so")` would find first anyway.
    """
    global NVRTC_PATH
    for name in ("libnvrtc.so", "libnvrtc.so.12", "libnvrtc.so.11"):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        try:
            major = ctypes.c_int()
            minor = ctypes.c_int()
            if lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)) == 0:
                NVRTC_PATH = _loaded_path("nvrtc")
                return (major.value, minor.value)
        except AttributeError:
            continue
    # Loaded but the call failed (a stub returns an error and prints "You are running
    # using the stub version of nvrtc"): still say WHICH file it was.
    NVRTC_PATH = _loaded_path("nvrtc")
    return None


def _loaded_path(fragment):
    """The mapped file whose name contains `fragment`, from /proc/self/maps, or None."""
    try:
        seen = []
        with open("/proc/self/maps", "r") as fh:
            for line in fh:
                parts = line.rstrip("\n").split()
                if len(parts) >= 6 and fragment in parts[-1] and parts[-1] not in seen:
                    seen.append(parts[-1])
        return seen[0] if seen else None
    except OSError:
        return None


#: The libcuda this process mapped, once _driver_cuda_version() has tried to load one.
LIBCUDA_PATH = None


def _stubs_on_ld_path():
    """LD_LIBRARY_PATH entries that are a CUDA `stubs` directory.

    A toolkit's lib64/stubs/ (and conda's targets/x86_64-linux/lib/stubs/) hold link-time
    STUBS of libcuda.so and libnvrtc.so: every entry point returns an error, so a program
    can be linked without a driver present. They are never meant to be on the runtime
    search path. Some site CUDA modules export them anyway, for the linker's sake, and
    then every CUDA program on that node fails at initialisation -- with CUDA error 34,
    CUDA_ERROR_STUB_LIBRARY, which is the driver API saying exactly this. Measured on
    an104 (TianheXY-AI) 2026-09-13: the probe passed sections 1-4 with the environment's
    own nvrtc (too new, but real), and after a module load the same sections reported
    "stub version of nvrtc" and error 34.
    """
    return [d for d in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
            if d and ("/stubs" in d or d.endswith("stubs"))]


def _driver_cuda_version():
    """(major, minor) of the CUDA version the DRIVER supports, or None.

    `cuDriverGetVersion` returns 1000*major + 10*minor and needs no context, no card and
    no initialisation beyond the library load -- so it works on a login node too.
    """
    global LIBCUDA_PATH
    for name in ("libcuda.so.1", "libcuda.so"):
        # .so.1 first: the real driver library installs that soname; the stub is the
        # bare libcuda.so. Asking for .so.1 first finds the driver even with a stubs
        # directory earlier on the path -- but the process that matters (OpenMM) does
        # not get that choice, so the stub is still reported below.
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        LIBCUDA_PATH = LIBCUDA_PATH or _loaded_path("libcuda.so")
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
    stubs = _stubs_on_ld_path()
    stub_loaded = any("/stubs/" in (p or "") for p in (NVRTC_PATH, LIBCUDA_PATH))
    if stub_loaded or (stubs and not nvrtc):
        # This outranks the version comparison: a stub has no version worth comparing.
        ok = False
        why = "a CUDA STUB library is on the load path -- every CUDA call will fail"
    elif nvrtc and driver:
        ok = nvrtc <= driver
        why = "nvrtc {}.{} {} driver {}.{}".format(
            nvrtc[0], nvrtc[1], "<=" if ok else ">", driver[0], driver[1])
    elif not nvrtc:
        why = "libnvrtc not loadable here"
    elif not driver:
        why = "no CUDA driver readable here (a login node, usually)"
    return dict(nvrtc=nvrtc, nvrtc_path=NVRTC_PATH, libcuda_path=LIBCUDA_PATH,
                driver_cuda=driver, ld_stubs=stubs, stub_loaded=stub_loaded,
                ok=ok, reason=why, skipped=os.environ.get(_SKIP_ENV) == "1")


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
    if d.get("stub_loaded") or (d.get("ld_stubs") and not nv):
        raise RuntimeError(
            "a CUDA STUB library is on this process's load path, so every CUDA call will\n"
            "  fail (CUDA error 34, CUDA_ERROR_STUB_LIBRARY).\n"
            "  libcuda loaded from     {}\n"
            "  libnvrtc loaded from    {}\n"
            "  stubs on LD_LIBRARY_PATH:\n      {}\n"
            "\n"
            "  A toolkit's lib64/stubs/ exists so a program can be LINKED without a driver;\n"
            "  it must never be on the runtime search path. A site CUDA module that exports\n"
            "  it does this to every CUDA program on the node. Fix: remove that entry --\n"
            "      module show CUDA/<ver>           # which module set it\n"
            "      module unload CUDA/<ver>         # or load one that does not\n"
            "      export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr : '\\n' | grep -v stubs | paste -sd:)\n"
            "  {}=1 skips this check.".format(
                d.get("libcuda_path") or "(not mapped)", d.get("nvrtc_path") or "(not mapped)",
                "\n      ".join(d.get("ld_stubs") or ["(none -- the stub came from a default path)"]),
                _SKIP_ENV))
    if nv is None or dr is None:
        raise RuntimeError(
            "cannot establish that this driver can run OpenMM's CUDA kernels: "
            "nvrtc {} / driver CUDA {} ({}). {}=1 skips this check.".format(
                nv, dr, d["reason"], _SKIP_ENV))
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
