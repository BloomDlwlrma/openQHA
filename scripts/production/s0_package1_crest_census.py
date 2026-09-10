"""Package 1, the **CREST branch**, batch version -- **one worker process per molecule**.

PRODUCTION. The batch CREST census, one worker process per molecule.

--------------------------------------------------------------------------------------
Architecture (rewritten after the user ruling of 2026-08-31)
--------------------------------------------------------------------------------------
The user: "one worker per molecule, each loading CREST and the GFN-FF conformer search on its own,
**loading the MACE model serially**, with as many molecules at once as there are CPUs, or as maximum memory / the memory one MACE load needs".

    main process (**loads no MACE model**)
      +- process pool, N = min(CPU count, available memory / memory per worker [, VRAM / VRAM per worker])
           +- worker (one molecule from start to finish)
                1. load one MACE model serially, **under a lock**   <- see "why serially" below
                2. start a MACE socket server thread in this process (reusing that same model)
                3. native QM9 geometry --coarse MACE relaxation--> the CREST starting point
                4. run CREST (GFN-FF sampling + refine="sp", asking itself for MACE single points over the socket)
                5. tighten to fmax=1e-4 -> pool with the ETKDG census -> deduplicate -> Hessian
                6. write to disk, return one record

**A worker holds exactly one MACE model** (measured steady state 821 MB): the socket server
thread and the tightening/Hessian use **the same** calculator. Pointing CREST at a separate
server process would put two models inside one worker.

**Why the model load is serial**: one load takes about 5-10 s and is disk-I/O and
deserialisation heavy. N processes loading at once fight over I/O; loading serially takes the
same total time, but each worker starts work the moment its own load finishes. Implemented with a cross-process lock.

**Why each worker is single-threaded**: measured, MACE scales very poorly with threads on small
molecules (acetone: 1 thread 111 ms, 4 threads 72 ms, **8 threads 101 ms, slower than 4**).
The molecules are too small; thread start-up and synchronisation cost more than the computation.
**So the parallelism goes between processes, never inside threads.**

--------------------------------------------------------------------------------------
GPU
--------------------------------------------------------------------------------------
`--device cuda` works, but **measured on this machine it does not pay**. Production precision
here is `float64` (fixed in the configuration; geometry optimisation and finite-difference
Hessians need it), and an entry-level Turing card has 1/32 the double-precision throughput of single:

    NVIDIA T400 4GB, per force call
        acetone, 10 atoms    CPU 101.6 ms   GPU  82.9 ms   (GPU 1.23x faster)
        CCOCCCO, 19 atoms    CPU 151.1 ms   GPU 142.4 ms   (GPU 1.06x faster)
        -- float32 on the same card is 39.3 ms (2.1x faster), **but changing the precision changes the science, so it is not done**

And one card cannot hold many workers (a few hundred MB per CUDA context),
**so the default on this machine is multi-process CPU**.

**On a card with full-speed double precision (an H100, where FP64 is about 1/2 of FP32) the
conclusion reverses** -- at that point it **must be re-measured**; do not carry this one over.

Usage::

    python scripts/production/s0_package1_crest_census.py                 # all of 1-4000
    python scripts/production/s0_package1_crest_census.py --limit 40
    python scripts/production/s0_package1_crest_census.py --workers 8
    python scripts/production/s0_package1_crest_census.py --device cuda
    python scripts/production/s0_package1_crest_census.py --plan-only      # report the concurrency plan only; run nothing
"""
import argparse
import csv
import gzip
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

# **Must be set before importing torch / numpy**: once the OpenMP thread pool exists it cannot be changed.
# One thread per worker (the reason is at the top of this module).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

def _repo_root():
    """Directory holding the openQHA package, found by walking up from this file.

    Depth-independent on purpose: this file keeps working wherever under the
    repository it is moved to. The earlier move into `scripts/_superseded/` broke
    every `parents[1]` in the moved files silently, which is what this replaces.
    """
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))
from openqha import (S0_ROOT, config, conformers, crest, crest_census, engine,
                  filters, record, report)

CFG = config.load()
P1 = config.package(1, CFG)
CB = P1["crest_batch"]
C = CFG["crest"]
T_REF = config.temperature(CFG)

FULL_RANGE = tuple(P1["qm9_range"])
CHUNK = (FULL_RANGE[0], FULL_RANGE[0] + int(P1["chunk_size"]) - 1)
DEDUP_A = float(P1["dedup_rmsd_A"])
FMAX_COARSE = float(P1["fmax_census_eV_A"])
FMAX_TIGHT = float(CB["tighten_fmax_eV_A"])
ORDER_SEED = int(CB["order_seed"])

# Result root: the user specified `{root}/result-s0tf_1_16000/1_4000` on 2026-08-31.
# The root is overridable by S0_RESULT_ROOT (pointed at shared storage on a cluster); the default stays under analysis/ in the repository.
_RES = os.environ.get("S0_RESULT_ROOT")
SAVE_ROOT = (Path(_RES).expanduser() if _RES
             else S0_ROOT / "analysis" / "package1" / "crest")
ETKDG_ROOT = S0_ROOT / "analysis" / "package1" / "{}_{}".format(*FULL_RANGE)

#: Measured (`VmRSS`, the steady state after a 19-atom molecule). Used to plan concurrency.
WORKER_HOST_MB = 821.0
#: A rough estimate of CUDA context + model + activations (measured peak VRAM allocation 42-75 MB, with about 300 MB of context on top)
WORKER_GPU_MB = 450.0
#: Memory headroom: do not consume all of the available memory
MEM_MARGIN = 0.80


def qid_of(i):
    return "dsgdb9nsd_{:06d}".format(i)


def _manifest(obj, stem, title=None):
    """Manifests and summaries are not written as JSON either: a `.log` (for people) plus a `.parquet` (the flat scalar part)."""
    stem = Path(stem)
    r = report.Report(title or stem.name, subtitle="package 1 - CREST branch")
    r.json_dump(obj, title="contents")
    r.write(stem.with_suffix(".log"))
    flat = {k: v for k, v in obj.items()
            if isinstance(v, (str, int, float, bool)) or v is None}
    if flat:
        record.write_parquet_rows([flat], stem.with_suffix(".parquet"))
    return stem.with_suffix(".log")


# ======================================================================================
# concurrency planning
# ======================================================================================
def mem_available_mb():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def gpu_free_mb(device):
    if not str(device).startswith("cuda"):
        return None
    try:
        import torch
        idx = int(device.split(":")[1]) if ":" in device else 0
        free, _total = torch.cuda.mem_get_info(idx)
        return free / 2 ** 20
    except Exception:
        return None


def plan_workers(device="cpu", requested=0, reserve_cpus=0):
    """N = min(CPU count, available memory / memory per worker [, available VRAM / VRAM per worker]).

    That is the formula the user specified on 2026-08-31. **Every term is reported**, so that
    "why this number" can be checked, rather than being a constant out of nowhere.
    """
    n_cpu_total = os.cpu_count() or 1
    n_cpu = max(1, n_cpu_total - int(reserve_cpus))
    host = mem_available_mb()
    n_mem = int(host * MEM_MARGIN / WORKER_HOST_MB) if host == host else n_cpu
    limits = [("CPU count os.cpu_count() = {} minus the reserve {}".format(
                  n_cpu_total, int(reserve_cpus)), n_cpu),
              ("available memory {:.0f} MB x {:.0%} / {:.0f} MB per worker".format(
                  host, MEM_MARGIN, WORKER_HOST_MB), n_mem)]
    n = min(n_cpu, n_mem)
    g = gpu_free_mb(device)
    if g is not None:
        n_gpu = max(1, int(g * MEM_MARGIN / WORKER_GPU_MB))
        limits.append(("available VRAM {:.0f} MB x {:.0%} / {:.0f} MB per worker".format(
            g, MEM_MARGIN, WORKER_GPU_MB), n_gpu))
        n = min(n, n_gpu)
    if requested:
        limits.append(("--workers on the command line", requested))
        n = min(n, requested)
    return max(1, int(n)), limits


# ======================================================================================
# the worker side
# ======================================================================================
_W = {}          # global state inside a worker process: calculator, socket, server thread


def _worker_init(lock, device):
    """Run once as each worker process starts. **The model load is inside the lock**, so it is serial."""
    import torch
    torch.set_num_threads(1)
    with lock:                                     # <- serial load
        t0 = time.time()
        calc, name, prov = engine.calculator(device=device)
        try:                                       # warm-up: build the graph, select kernels, allocate VRAM
            from ase import Atoms
            a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
            a.calc = calc
            a.get_potential_energy()
        except Exception:
            pass
        load_s = time.time() - t0
    # Node-local, per job: see openqha.config.socket_dir. A socket under runs_root is on
    # the shared filesystem, where two jobs collide (measured on Tianhe 2026-09-09).
    sock = str(config.socket_path("s0_mace_w", os.getpid()))
    from openqha import mace_server
    th, stop = mace_server.serve_in_thread(sock, calc)
    _W.update(calc=calc, engine=name, prov=prov, socket=sock, stop=stop,
              device=device, load_seconds=load_s)


def _worker_one(task):
    """One molecule from start to finish. Runs in a worker process and returns one serialisable record."""
    qid, smiles, args = task
    # ---- wall-clock budget: past the deadline no new molecule is **started** ------------
    # Nothing already running is cut short (that machine time would be wasted); it simply takes no new work.
    dl = args.get("deadline")
    if dl and time.time() >= dl:
        return dict(ok=False, skipped=True, qm9_index=qid)
    t_start = time.time()
    calc = _W["calc"]
    wd = Path(args["scratch"]) / qid
    out_mol, out_xyz = Path(args["out_mol"]), Path(args["out_xyz"])
    out_log = Path(args["out_log"])
    phase = args.get("phase", "full")
    try:
        if phase in ("full", "crest"):
            run_rec, start_info = _crest_stage(qid, smiles, wd, calc, args)
            if phase == "crest":
                # stage one ends here: hand the **ensemble** and the native output to stage two
                return _finish_crest_only(qid, smiles, wd, run_rec, start_info,
                                          args, t_start)
        else:
            run_rec, start_info = _load_crest_stage(qid, args)
        rec, basins = analyse_one(qid, smiles, wd, calc, start_info, run_rec,
                                  do_hessian=not args["no_hessian"],
                                  ensemble=args.get("_ens"))
        rec["total_seconds"] = time.time() - t_start
        rec["worker"] = dict(pid=os.getpid(), device=_W["device"],
                             model_load_seconds=_W["load_seconds"],
                             socket=_W["socket"])
        rec["stage"] = args.get("stage") or "full"
        if not args["pilot"]:
            # **No JSON** (user, 2026-08-31): the .log is for people, the .parquet carries the precision,
            # and the native CREST output is only renamed.
            record.write_molecule(rec, args["dirs"], qid, basins=basins,
                                  crest_out=wd / "crest.out",
                                  keep_crest_out=bool(CB["keep_crest_out"]))
        return dict(ok=True, qm9_index=qid, record=_slim(rec))
    except Exception as exc:
        fr = dict(qm9_index=qid, smiles=smiles, error_type=type(exc).__name__,
                  error=str(exc), traceback=traceback.format_exc(),
                  seconds=time.time() - t_start)
        if not args["pilot"]:
            record.write_failure(fr, args["dirs"], qid)
            # On failure the native CREST output matters **more**, not less -- the root cause is in it
            if CB["keep_crest_out"] and (wd / "crest.out").exists():
                try:
                    d = Path(args["dirs"]["out"])
                    d.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(wd / "crest.out", d / "crest_{}.out".format(qid))
                except OSError:
                    pass
        return dict(ok=False, qm9_index=qid, failure=fr)


def _finish_crest_only(qid, smiles, wd, run_rec, start_info, args, t_start):
    """The end of stage one: ensemble + native output + run record (no MACE tightening or Hessian at all)."""
    import shutil as _sh
    d = Path(args["dirs"]["ens"])
    d.mkdir(parents=True, exist_ok=True)
    src = Path(wd) / "crest_conformers.xyz"
    if not src.exists():
        raise RuntimeError("stage one finished but there is no crest_conformers.xyz")
    _sh.copy2(src, d / "{}_ensemble.xyz".format(qid))
    for extra in ("crest_rotamers.xyz", "start.xyz"):
        if (Path(wd) / extra).exists():
            _sh.copy2(Path(wd) / extra, d / "{}_{}".format(qid, extra))
    if CB["keep_crest_out"] and (Path(wd) / "crest.out").exists():
        o = Path(args["dirs"]["out"])
        o.mkdir(parents=True, exist_ok=True)
        _sh.copy2(Path(wd) / "crest.out", o / "crest_{}.out".format(qid))
    n_conf = len(crest_census._read_raw(d / "{}_ensemble.xyz".format(qid)))
    rec = dict(qm9_index=qid, smiles=smiles, stage="stage1-crest",
               crest_run=dict(seconds=run_rec.get("seconds"),
                              terminated_normally=run_rec.get("terminated_normally"),
                              n_terminated_early=run_rec.get("n_terminated_early"),
                              total_engrad_calls=run_rec.get("total_engrad_calls"),
                              reused_scratch=run_rec.get("reused_scratch", False),
                              shake_fallback_used=run_rec.get("shake_fallback_used"),
                              crest=run_rec.get("crest"),
                              settings=run_rec.get("settings")),
               start_geometry=start_info,
               n_conformers_reported_by_crest=n_conf,
               ensemble_path=str(d / "{}_ensemble.xyz".format(qid)),
               total_seconds=time.time() - t_start,
               note="the product of stage one (CREST sampling). Tightening / deduplication / Hessian are done by stage two.")
    if not args["pilot"]:
        record.write_molecule(rec, args["dirs"], qid, basins=None,
                              crest_out=None, keep_crest_out=False)
    return dict(ok=True, qm9_index=qid, record=_slim(rec))


def _load_crest_stage(qid, args):
    """Stage two: read back the ensemble and run record stage one saved."""
    ens = Path(args["dirs"]["ens"]) / "{}_ensemble.xyz".format(qid)
    if not ens.exists():
        raise RuntimeError("stage two cannot find the stage one ensemble: {}".format(ens))
    args["_ens"] = str(ens)
    return (dict(seconds=None, terminated_normally=True, n_terminated_early=0,
                 from_stage1=True),
            dict(source="stage1-crest", path=str(ens)))


def _crest_stage(qid, smiles, wd, calc, args):
    """Prepare the starting point and run CREST. Reuses a working directory only if it **finished and its settings match** (defect 57)."""
    if wd.exists() and not args["redo_crest"] and (wd / "crest.out").exists():
        try:
            r = crest.record_from_dir(wd)
        except Exception:
            r = {}
        same, _why = crest.settings_match(
            r.get("settings_from_toml"),
            dict(runtype=C["runtype"], optlev=C["optlev"], refine=C["refine"]))
        if r.get("ok") and same and (wd / "crest_conformers.xyz").exists():
            r["reused_scratch"] = True
            return r, dict(source="reused_scratch", path=str(wd / "start.xyz"))
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)

    atoms, start_info = starting_geometry(qid, smiles, calc)
    conformers.write_xyz(atoms, wd / "start.xyz",
                         "{} CREST start ({})".format(qid, start_info["source"]))
    def _go(shake):
        return crest.run(wd, wd / "start.xyz", timeout_s=int(CB["timeout_s"]),
                         env=dict(S0_MACE_SOCKET=_W["socket"]),
                         runtype=C["runtype"], threads=int(args["crest_threads"]),
                         optlev=C["optlev"], refine=C["refine"],
                         backend=C["backend"], shake=shake,
                         engine_client=S0_ROOT / C["engine_client"])

    run_rec = _go(C.get("shake"))          # null => use CREST own default
    # ---- targeted retry for `terminated EARLY` ------------------------------------------
    # Root cause (measured 2026-08-31; see the shake comment in openqha/crest.py): the CREST default
    # `shake = 2` (constrain every bond) fails to converge on **strained rings**, so the molecular dynamics is judged to have failed.
    # `shake = 1` (constrain only bonds to hydrogen) measurably cleared EARLY on all three failing molecules.
    # **Retried only on failure**, so the sampling conditions of molecules that already worked are untouched.
    fb = CB.get("shake_fallback")
    if (not run_rec.get("ok") and fb is not None
            and (run_rec.get("n_terminated_early") or 0) > 0):
        early0 = run_rec.get("n_terminated_early")
        run_rec = _go(int(fb))
        run_rec["shake_fallback_used"] = int(fb)
        run_rec["n_terminated_early_before_fallback"] = early0
    if not run_rec.get("ok"):
        raise RuntimeError(
            "CREST did not pass the criteria: finished normally {}, terminated EARLY {}, return code {}{}".format(
                run_rec.get("terminated_normally"), run_rec.get("n_terminated_early"),
                run_rec.get("returncode"),
                " (already retried with shake={})".format(fb) if fb is not None else ""))
    return run_rec, start_info


def _slim(rec):
    """The compact record returned to the main process -- the full record is already on disk and need not cross the pipe again."""
    keep = ("qm9_index", "smiles", "n_conformers_reported_by_crest", "n_basins",
            "n_basins_crest_only", "n_basins_etkdg_only", "n_basins_both",
            "n_saddles_rejected", "conformational_correction_kcal",
            "total_seconds", "analysis_seconds")
    out = {k: rec.get(k) for k in keep}
    out["crest_seconds"] = (rec.get("crest_run") or {}).get("seconds")
    out["crest_reused"] = (rec.get("crest_run") or {}).get("reused_scratch", False)
    out["weight_of_lowest"] = (rec.get("populations") or {}).get("weight_of_lowest")
    out["tighten_steps_total"] = (rec.get("tighten_crest") or {}).get("opt_steps_total")
    out["n_graph_changed"] = (rec.get("tighten_crest") or {}).get("n_graph_changed")
    out["worker_pid"] = (rec.get("worker") or {}).get("pid")
    return out


# ======================================================================================
# The steps for one molecule (numerically identical to before the architecture change)
# ======================================================================================
def starting_geometry(qid, smiles, calc):
    from ase.io import read
    try:
        ref = config.qm9_xyz(qid, CFG)
    except FileNotFoundError:
        ref = None
    if ref is not None and CB["source_geometry"] == "qm9_native_relaxed":
        atoms = read(str(ref))
        e, fm, ok, ns = conformers.optimise(atoms, calc, fmax=FMAX_COARSE,
                                            steps=int(P1["max_opt_steps"]))
        return atoms, dict(source="qm9_native_relaxed", path=str(ref),
                           relaxed_energy_eV=float(e), max_force_eV_A=float(fm),
                           converged=bool(ok), opt_steps=int(ns))
    rec, basins, _ = conformers.census(
        smiles, calc, name=qid, n_embed=int(P1["n_embed"]), seed=int(P1["embed_seed"]),
        fmax=FMAX_COARSE, threshold_A=DEDUP_A, scan_thresholds=False,
        reference_xyz=None, include_reference_as_seed=False)
    return basins[0], dict(source="etkdg_lowest", n_basins_coarse=rec["n_basins"],
                           relaxed_energy_eV=float(rec["basin_energies_eV"][0]))


def etkdg_basins_of(qid):
    from ase.io import read
    p = ETKDG_ROOT / "{}_{}".format(*CHUNK) / "xyz" / (qid + "_basins.xyz")
    j = ETKDG_ROOT / "{}_{}".format(*CHUNK) / "mol" / (qid + ".json")
    if not p.exists() or not j.exists():
        return None
    return dict(frames=read(str(p), index=":"), path=str(p),
                record=json.loads(j.read_text(encoding="utf-8")))


def analyse_one(qid, smiles, workdir, calc, start_info, run_rec, do_hessian=True,
                ensemble=None):
    ens = Path(ensemble) if ensemble else Path(workdir) / "crest_conformers.xyz"
    if not ens.exists():
        raise RuntimeError("CREST produced no crest_conformers.xyz")
    frames, comments = crest_census.read_ensemble_atoms(ens)
    t0 = time.time()
    tight_c, basins_c, e_c = crest_census.tighten_frames(
        smiles, frames, calc, fmax=FMAX_TIGHT)

    groups, group_e, labels = [basins_c], [e_c], ["crest"]
    etk = etkdg_basins_of(qid) if CB["combine_with_etkdg"] else None
    tight_e = None
    if etk is not None:
        tight_e, basins_e, e_e = crest_census.tighten_frames(
            smiles, etk["frames"], calc, fmax=FMAX_TIGHT)
        groups.append(basins_e); group_e.append(e_e); labels.append("etkdg")

    gb, pooled = crest_census.global_basins(smiles, groups, group_e,
                                            threshold_A=DEDUP_A)
    hess, keep, saddles = ([], list(range(len(pooled))), [])
    if do_hessian:
        hess, keep, saddles = crest_census.hessian_screen(pooled, calc)
    if not keep:
        raise RuntimeError("no basin survives pooling, deduplication and the imaginary-frequency filter")

    e_pool = np.asarray(gb["global_energies_eV"])[keep]
    order = np.argsort(e_pool)
    keep = [keep[i] for i in order]
    e_pool = e_pool[order]
    rel = (e_pool - e_pool.min()) * conformers.EV_TO_KCAL

    member = {lab: set(m) for lab, m in zip(labels, gb["membership_per_group"])}
    prov = [sorted(lab for lab in labels if gi in member[lab]) for gi in keep]

    rec = dict(
        qm9_index=qid, smiles=smiles, temperature_K=T_REF,
        start_geometry=start_info,
        crest_run=dict(seconds=run_rec.get("seconds"),
                       terminated_normally=run_rec.get("terminated_normally"),
                       n_terminated_early=run_rec.get("n_terminated_early"),
                       total_engrad_calls=run_rec.get("total_engrad_calls"),
                       reused_scratch=run_rec.get("reused_scratch", False),
                       shake_fallback_used=run_rec.get("shake_fallback_used"),
                       n_terminated_early_before_fallback=run_rec.get(
                           "n_terminated_early_before_fallback"),
                       crest=run_rec.get("crest"), settings=run_rec.get("settings")),
        n_conformers_reported_by_crest=len(frames),
        crest_relative_kcal=crest_census._crest_relative_kcal(comments),
        tighten_crest=tight_c, tighten_etkdg=tight_e,
        etkdg_source=(etk["path"] if etk else None),
        n_etkdg_basins_in=(len(etk["frames"]) if etk else 0),
        pooling=dict(labels=labels, n_pooled_frames=gb["n_frames_pooled"],
                     n_global_before_hessian=gb["n_global_basins"],
                     membership_per_group=gb["membership_per_group"],
                     merge_energy_warnings=gb["merge_energy_warnings"],
                     global_energies_eV=list(gb["global_energies_eV"]),
                     global_relative_kcal=list(gb["global_relative_kcal"])),
        hessian_all_pooled=hess, n_saddles_rejected=len(saddles), saddles=saddles,
        basin_hessian=[hess[i] for i in keep] if hess else [],
        n_basins=len(keep),
        basin_energies_eV=[float(x) for x in e_pool],
        basin_relative_kcal=[float(x) for x in rel],
        basin_provenance=prov,
        n_basins_crest_only=sum(1 for p in prov if p == ["crest"]),
        n_basins_etkdg_only=sum(1 for p in prov if p == ["etkdg"]),
        n_basins_both=sum(1 for p in prov if len(p) == 2),
        populations=conformers.basin_populations(rel, T_REF),
        conformational_correction_kcal=crest_census.conformational_correction_kcal(
            rel, T_REF),
        dedup_threshold_A=DEDUP_A, tighten_fmax_eV_A=FMAX_TIGHT,
        analysis_seconds=time.time() - t0,
        free_energy_note="no free energy is computed -- the symmetry number and electronic degeneracy must be declared explicitly; this repository does not derive them automatically.")
    if etk is not None:
        er = etk["record"]
        rec["etkdg_reference"] = dict(
            n_basins_coarse=er.get("n_basins"),
            basin_relative_kcal_coarse=er.get("basin_relative_kcal"),
            correction_coarse_kcal=crest_census.conformational_correction_kcal(
                er.get("basin_relative_kcal", [0.0]), T_REF))
    return rec, [pooled[i] for i in keep]


def write_basins_xyz(path, basins, energies, prov, qid):
    lines = []
    for k, (a, e) in enumerate(zip(basins, energies)):
        lines.append(str(len(a)))
        lines.append("{} basin {} E = {:.8f} eV  from={}  (MACE-OFF23-SC)".format(
            qid, k, e, "+".join(prov[k])))
        for s, p in zip(a.get_chemical_symbols(), a.positions):
            lines.append("{:2s} {:18.10f} {:18.10f} {:18.10f}".format(s, *p))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ======================================================================================
# the main process
# ======================================================================================
def load_index(lo, hi):
    want = {qid_of(i) for i in range(lo, hi + 1)}
    rows = {}
    with config.qm9_index_csv(CFG).open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["qm9_index"] in want:
                rows[r["qm9_index"]] = r
    return rows


def processing_order(ids, schedule="random"):
    """The default is a **random permutation with a fixed seed** -- any prefix is an unbiased sample.

    `schedule="lpt"` switches to longest-processing-time first (descending predicted cost by conformational
    flexibility): it shortens the total wall clock **but destroys the unbiasedness of a prefix** (a prefix
    is then systematically biased towards flexible molecules), so it is not the default.
    """
    if schedule == "lpt":
        try:
            import pandas as pd
            df = pd.read_parquet(
                S0_ROOT / "analysis" / "package1_flexibility_ranking.parquet")
            rank = {q: i for i, q in enumerate(df["qm9_index"])}
            return sorted(ids, key=lambda q: rank.get(q, 10 ** 9))
        except Exception as exc:
            print("   the ranking table longest-processing-time-first needs could not be read ({}); falling back to the random permutation".format(exc))
    rng = np.random.default_rng(ORDER_SEED)
    return [ids[i] for i in rng.permutation(len(ids))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=CHUNK[0])
    ap.add_argument("--hi", type=int, default=CHUNK[1])
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0,
                    help="upper bound; defaults to min(CPU count, available memory / memory per worker[, VRAM / ...])")
    ap.add_argument("--crest-threads", type=int, default=1,
                    help="threads per CREST; should be 1 when there is one worker per molecule")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | cuda:N")
    ap.add_argument("--schedule", choices=("random", "lpt"), default="random")
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--no-hessian", action="store_true")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--redo-crest", action="store_true")
    ap.add_argument("--plan-only", action="store_true", help="report the concurrency plan only; run nothing")
    ap.add_argument("--reserve-cpus", type=int, default=0,
                    help="logical cores to leave for another session (for example one running GPU molecular dynamics)")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="wall-clock budget in hours. Past it no new molecule starts; whatever is running finishes")
    ap.add_argument("--phase", choices=("full", "crest", "analyse"), default="full",
                    help="full=one worker does everything end to end (the default on this machine); "
                         "crest=CREST sampling only (stage one, on a CPU node); "
                         "analyse=tightening + deduplication + Hessian only (stage two, reading the stage one ensemble)")
    ap.add_argument("--ens-from", default="",
                    help="which stage directory stage two reads the ensemble from (defaults to the same --stage directory)")
    ap.add_argument("--stage", default="",
                    help="subdirectory name when running in stages (for example stage1-gfnff / stage2-mace). "
                         "Leave it empty for no subdirectory")
    ap.add_argument("--shard", default="0/1",
                    help="k/n -- shard k of n in a job array. "
                         "Taken modulo the processing order, so each shard is still an unbiased random sample")
    args = ap.parse_args()

    chunk_dir = (SAVE_ROOT / "result-s0tf_{}_{}".format(*FULL_RANGE)
                 / "{}_{}".format(args.lo, args.hi))
    if args.stage:
        chunk_dir = chunk_dir / args.stage
    dirs = {k: chunk_dir / k for k in ("log", "mol", "xyz", "out", "ens")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    out_mol, out_xyz, out_log = dirs["mol"], dirs["xyz"], dirs["out"]
    scratch = (Path(args.scratch).expanduser() if args.scratch
               else config.runs_dir(CB["scratch_dir"], CFG))

    n_workers, limits = plan_workers(args.device, args.workers, args.reserve_cpus)
    v = crest.crest_version()
    prov = engine.provenance()          # **hashes the weights only; loads no model**

    print("=" * 104)
    print("package 1 - CREST branch   indices {}-{}   one worker per molecule".format(args.lo, args.hi))
    print("=" * 104)
    print("CREST        {} (commit {})  runtype={} refine={} optlev={}".format(
        v["version"], v["commit"], C["runtype"], C["refine"], C["optlev"]))
    print("engine       {}  dtype={}  device={}".format(
        prov["engine"], prov["dtype"], args.device))
    print()
    print("[concurrency plan]  N = min(the terms below)")
    for label, val in limits:
        print("   {:<56} {:>5}".format(label, val))
    print("   {:<56} {:>5}".format("**chosen**", n_workers))
    print("   per worker: 1 torch thread, {} CREST thread(s), one MACE model (about {:.0f} MB)"
          .format(args.crest_threads, WORKER_HOST_MB))
    print("   the model load is **serial** (a cross-process lock); a worker starts as soon as its own load finishes")
    if args.plan_only:
        return

    rows = load_index(args.lo, args.hi)
    all_ids = [qid_of(i) for i in range(args.lo, args.hi + 1) if qid_of(i) in rows]
    gates = filters.enabled_gates(CFG)
    ids, rejected, gate_counts = filters.screen_many(
        ((q, rows[q]["qm9_smiles"]) for q in all_ids), CFG)
    order = processing_order(ids, args.schedule)

    print()
    print("gates {} -> {} passed / {} in the range".format("/".join(gates), len(ids), len(all_ids)))
    print("order        {}".format(
        "a random permutation with fixed seed {} (any prefix is an unbiased sample)".format(ORDER_SEED)
        if args.schedule == "random" else
        "**longest processing time first** -- shorter total wall clock, but a prefix is no longer unbiased"))
    try:
        shard_k, shard_n = (int(x) for x in args.shard.split("/"))
    except ValueError:
        raise SystemExit("--shard must be written k/n, for example 3/16")
    if not (0 <= shard_k < shard_n):
        raise SystemExit("k in --shard must lie in [0, n)")
    if shard_n > 1:
        order = [q for i, q in enumerate(order) if i % shard_n == shard_k]
        print("shard        {}/{} -> {} molecule(s) in this shard (taken modulo the processing "
              "order, so this shard is still an unbiased random sample)".format(shard_k, shard_n, len(order)))

    # Resume criterion: the product is <index>.parquet (the one carrying the precision). .json is no longer consulted.
    todo = [q for q in order
            if args.redo or not (dirs["mol"] / (q + ".parquet")).exists()]
    n_done_before = len(order) - len(todo)
    if args.pilot:
        todo = todo[:args.pilot]
    elif args.limit:
        todo = todo[:max(0, args.limit - n_done_before)]
    print("{} product(s) already present; {} to run this time".format(n_done_before, len(todo)))
    if not todo:
        print("there is no molecule to run.")
        return

    if not args.pilot:
        _manifest(dict(
            package="package 1 - CREST branch (one worker per molecule)",
            generated_by="scripts/production/s0_package1_crest_census.py",
            qm9_range=[args.lo, args.hi], config_path=CFG["_path"],
            engine=prov, crest=v, crest_settings=dict(C),
            crest_batch_settings=dict(CB),
            concurrency=dict(n_workers=n_workers, device=args.device,
                             limits=[dict(label=l, value=x) for l, x in limits],
                             worker_host_mb=WORKER_HOST_MB,
                             model_loading="serial (a cross-process lock)",
                             torch_threads_per_worker=1,
                             crest_threads_per_worker=args.crest_threads),
            shard=args.shard,
            schedule=args.schedule, processing_order_seed=ORDER_SEED,
            processing_order=order,
            species_filter=dict(gates_enabled=list(gates), n_in_range=len(all_ids),
                                n_passed=len(ids), counts=gate_counts)),
            chunk_dir / ("manifest" if shard_n == 1
                         else "manifest_shard{}of{}".format(shard_k, shard_n)))

    if args.phase == "analyse" and args.ens_from:
        dirs = dict(dirs)
        dirs["ens"] = (chunk_dir.parent / args.ens_from / "ens")
    wargs = dict(scratch=str(scratch), dirs={k: str(v) for k, v in dirs.items()},
                 stage=args.stage, phase=args.phase,
                 out_mol=str(out_mol), out_xyz=str(out_xyz),
                 out_log=str(out_log), no_hessian=args.no_hessian,
                 pilot=bool(args.pilot), redo_crest=args.redo_crest,
                 crest_threads=args.crest_threads,
                 deadline=(time.time() + args.max_hours * 3600.0
                           if args.max_hours > 0 else None))
    tasks = [(q, rows[q]["qm9_smiles"], wargs) for q in todo]
    if args.max_hours > 0:
        print("wall budget  {:.1f} hours; past it no new molecule starts (whatever is running finishes)".format(
            args.max_hours))
        print("             the number expected to finish at the currently measured throughput is reported in the summary")

    ctx = mp.get_context("fork")
    lock = ctx.Lock()
    done, failed, skipped, t0 = [], [], [], time.time()
    print()
    with ctx.Pool(processes=n_workers, initializer=_worker_init,
                  initargs=(lock, args.device)) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker_one, tasks), 1):
            el = time.time() - t0
            if res["ok"]:
                r = res["record"]
                done.append(r)
                # **Branch on the phase**: stage one has no basin-level fields yet (they are None), and printing
                # them with one generic format raises TypeError and **kills the main loop**, taking the molecule
                # still in flight with it (measured in the end-to-end test on 2026-08-31).
                if args.phase == "crest":
                    print("   [{:5d}/{:5d}] {}  CREST {:3d} conformer(s) / {:6.0f} s; "
                          "EARLY {}; {:6.0f} s for this molecule; throughput {:.0f} s/molecule".format(
                              i, len(tasks), r["qm9_index"],
                              r.get("n_conformers_reported_by_crest") or 0,
                              _f(r.get("crest_seconds")),
                              r.get("crest_terminated_early"),
                              _f(r.get("total_seconds")), el / i), flush=True)
                else:
                    print("   [{:5d}/{:5d}] {}  CREST {:3d} conformer(s)/{:5.0f} s -> {:3d} basin(s) "
                          "(C only {:2d}/E only {:2d}); correction {:+.4f}; {:5.0f} s for this molecule; "
                          "throughput {:.0f} s/molecule".format(
                              i, len(tasks), r["qm9_index"],
                              r.get("n_conformers_reported_by_crest") or 0,
                              _f(r.get("crest_seconds")), r.get("n_basins") or 0,
                              r.get("n_basins_crest_only") or 0,
                              r.get("n_basins_etkdg_only") or 0,
                              r.get("conformational_correction_kcal") or 0.0,
                              _f(r.get("total_seconds")), el / i), flush=True)
            elif res.get("skipped"):
                skipped.append(res["qm9_index"])
                continue
            else:
                f = res["failure"]
                failed.append(f)
                print("   [{:5d}/{:5d}] failed {} {} -- {}: {}".format(
                    i, len(tasks), f["qm9_index"], f["smiles"], f["error_type"],
                    str(f["error"])[:70]), flush=True)

    if skipped:
        print()
        print("   the wall budget is spent; {} molecule(s) were not started (the next run continues from here)".format(len(skipped)))
    _summarise(done, failed, order, time.time() - t0, chunk_dir, args, prov, v,
               n_workers, limits, skipped)


def _summarise(done, failed, order, elapsed, chunk_dir, args, prov, v,
               n_workers, limits, skipped=()):
    print()
    print("=" * 104)
    print("summary")
    print("=" * 104)
    print("{} succeeded, {} failed, wall clock {:.0f} s = {:.2f} hours, {} worker(s)".format(
        len(done), len(failed), elapsed, elapsed / 3600, n_workers))
    payload = dict(n_success=len(done), n_failed=len(failed),
                   n_skipped_by_budget=len(skipped),
                   skipped_by_budget=list(skipped),
                   max_hours=args.max_hours, reserve_cpus=args.reserve_cpus,
                   seconds=elapsed,
                   failures=failed, engine=prov, crest=v,
                   concurrency=dict(n_workers=n_workers, device=args.device,
                                    limits=[dict(label=l, value=x) for l, x in limits]),
                   schedule=args.schedule, rows=done)
    if done:
        arr = lambda k: np.array([r.get(k) if r.get(k) is not None else np.nan
                                  for r in done], dtype=float)
        if args.phase == "crest":
            # stage one reports only the quantities it computed itself
            nc = arr("n_conformers_reported_by_crest")
            cr, tot = arr("crest_seconds"), arr("total_seconds")
            ee = arr("crest_terminated_early")
            per = elapsed / max(len(done) + len(failed), 1)
            print("CREST conformers  mean {:.2f} median {:.0f} maximum {:.0f}".format(
                np.nanmean(nc), np.nanmedian(nc), np.nanmax(nc)))
            print("time per molecule  mean {:.0f} s median {:.0f} s maximum {:.0f} s".format(
                np.nanmean(tot), np.nanmedian(tot), np.nanmax(tot)))
            print("**throughput** {:.0f} s/molecule; parallel speed-up {:.1f}x ({} worker(s))".format(
                per, np.nanmean(tot) / max(per, 1e-9), n_workers))
            n_bad = int(np.nansum(ee > 0))
            print("molecules with a non-zero terminated EARLY count: {}   <- criterion: must be 0".format(n_bad))
            remaining = len(order) - len(done)
            print()
            print("cost extrapolation  {} left -> {:.1f} hours = {:.1f} days".format(
                remaining, per * remaining / 3600, per * remaining / 86400))
            payload["statistics"] = dict(
                phase="crest",
                seconds_per_molecule_wall=float(per),
                seconds_per_molecule_serial=float(np.nanmean(tot)),
                n_conformers_mean=float(np.nanmean(nc)),
                n_molecules_with_early=n_bad,
                projected_hours_remaining=float(per * remaining / 3600))
            _finish_summary(payload, failed, chunk_dir, args)
            return
        nb, nc = arr("n_basins"), arr("n_conformers_reported_by_crest")
        tot, cr, an = arr("total_seconds"), arr("crest_seconds"), arr("analysis_seconds")
        co, eo = arr("n_basins_crest_only"), arr("n_basins_etkdg_only")
        corr, w0 = arr("conformational_correction_kcal"), arr("weight_of_lowest")
        per = elapsed / max(len(done) + len(failed), 1)
        print("CREST conformers  mean {:.2f} maximum {:.0f}; basins after pooling mean {:.2f} maximum {:.0f}".format(
            np.nanmean(nc), np.nanmax(nc), np.nanmean(nb), np.nanmax(nb)))
        print("basin source  CREST only mean {:.2f}; ETKDG only mean {:.2f}".format(
            np.nanmean(co), np.nanmean(eo)))
        print("time per molecule  mean {:.0f} s median {:.0f} s maximum {:.0f} s "
              "(of which CREST mean {:.0f} s, analysis mean {:.0f} s)".format(
                  np.nanmean(tot), np.nanmedian(tot), np.nanmax(tot),
                  np.nanmean(cr), np.nanmean(an)))
        print("**throughput** {:.0f} s/molecule (wall clock / molecule count); {:.0f} s serially per molecule "
              "=> parallel speed-up {:.1f}x ({} worker(s))".format(
                  per, np.nanmean(tot), np.nanmean(tot) / max(per, 1e-9), n_workers))
        print("lowest-basin weight  mean {:.3f}; fraction below 0.9 {:.1%}".format(
            np.nanmean(w0), float(np.nanmean(w0 < 0.9))))
        print("conformational correction  mean {:+.4f} median {:+.4f} most negative {:+.4f} kcal/mol".format(
            np.nanmean(corr), np.nanmedian(corr), np.nanmin(corr)))
        remaining = len(order) - len(done)
        print()
        print("cost extrapolation  {} left -> {:.1f} hours = {:.1f} days".format(
            remaining, per * remaining / 3600, per * remaining / 86400))
        payload["statistics"] = dict(
            seconds_per_molecule_wall=float(per),
            seconds_per_molecule_serial=float(np.nanmean(tot)),
            parallel_speedup=float(np.nanmean(tot) / max(per, 1e-9)),
            n_workers=n_workers,
            n_basins_mean=float(np.nanmean(nb)),
            crest_seconds_mean=float(np.nanmean(cr)),
            analysis_seconds_mean=float(np.nanmean(an)),
            projected_hours_remaining=float(per * remaining / 3600))
    _finish_summary(payload, failed, chunk_dir, args)


def _f(v):
    """None -> nan, so that a format like {:.0f} does not blow up."""
    return float("nan") if v is None else float(v)


def _finish_summary(payload, failed, chunk_dir, args):
    if failed:
        print()
        print("failures ({}):".format(len(failed)))
        for f in failed[:20]:
            print("   {} {:24s} {}: {}".format(f["qm9_index"], f["smiles"],
                                               f["error_type"], str(f["error"])[:70]))
    if not args.pilot:
        name = ("summary" if args.shard == "0/1"
                else "summary_shard{}".format(args.shard.replace("/", "of")))
        _manifest(payload, chunk_dir / name, title="package 1 - CREST branch   summary of this run")
        print()
        print("written: {} and the .log of the same name".format(chunk_dir))


if __name__ == "__main__":
    main()
