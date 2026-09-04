"""包 1 的 **CREST 支路**, 批量版 —— **每个分子一个 worker 进程**.

PRODUCTION. The batch CREST census, one worker process per molecule.

--------------------------------------------------------------------------------------
架构（2026-08-31 用户裁定后重写）
--------------------------------------------------------------------------------------
用户：「每个分子单独 worker 且单独加载 CREST 与 GFN-FF 构象搜索，**串行加载 MACE 模型**，
一共 CPU 数个分子、或 最大内存/加载 MACE 所需内存 个分子同时运行」。

    主进程（**不加载 MACE 模型**）
      +- 进程池, N = min(CPU 数, 可用内存/单 worker 内存 [, 显存/单 worker 显存])
           +- worker（一个分子从头到尾）
                1. **加锁**串行加载一份 MACE 模型      <- 见下"为什么串行"
                2. 在本进程内起一个 MACE 套接字服务线程（复用同一份模型）
                3. QM9 原生几何 --MACE 粗弛豫--> CREST 起点
                4. 跑 CREST（GFN-FF 采样 + refine="sp" 经套接字问自己要 MACE 单点）
                5. 收紧到 fmax=1e-4 -> 与 ETKDG 普查并池 -> 去重 -> Hessian
                6. 落盘, 返回一份记录

**一个 worker 只有一份 MACE 模型**（实测稳态 821 MB）：套接字服务线程与
收紧/Hessian 用的是**同一个** calculator。若让 CREST 去连一个独立的服务端进程，
同一个 worker 里就会有两份模型。

**为什么模型加载要串行**：加载一次约 5–10 秒，是磁盘 I/O + 反序列化密集的。
N 个进程同时加载会互相抢 I/O；串行加载的总时长不变，但每个 worker
一旦加载完就立刻开工。用一把跨进程的锁实现。

**为什么每个 worker 单线程**：实测 MACE 在小分子上的线程标度极差
（丙酮：1 线程 111 ms、4 线程 72 ms、**8 线程 101 ms，比 4 线程还慢**）。
分子太小，线程启动与同步的开销超过计算本身。
**所以并行度要放在"进程之间"，不是"线程之内"。**

--------------------------------------------------------------------------------------
GPU
--------------------------------------------------------------------------------------
`--device cuda` 可用，但**本机实测不划算**。本仓生产精度是 `float64`
（配置写死；几何优化与有限差分 Hessian 需要），而入门级 Turing 卡的双精度吞吐
只有单精度的 1/32：

    NVIDIA T400 4GB, 每次力计算
        丙酮 10 原子      CPU 101.6 ms   GPU  82.9 ms   （GPU 快 1.23 倍）
        CCOCCCO 19 原子   CPU 151.1 ms   GPU 142.4 ms   （GPU 快 1.06 倍）
        —— 同卡 float32 是 39.3 ms（快 2.1 倍），**但改精度是改科学，不做**

一块卡上又开不了多少 worker（每个 CUDA 上下文几百 MB），
**所以本机默认 CPU 多进程**。

**换到有全速双精度的卡（如 H100，FP64 约为 FP32 的 1/2）结论会反过来** ——
届时**必须重测**，不要沿用这里的结论。

用法::

    python scripts/production/s0_package1_crest_census.py                 # 全 1-4000
    python scripts/production/s0_package1_crest_census.py --limit 40
    python scripts/production/s0_package1_crest_census.py --workers 8
    python scripts/production/s0_package1_crest_census.py --device cuda
    python scripts/production/s0_package1_crest_census.py --plan-only      # 只报并发规划, 不跑
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

# **必须在 import torch / numpy 之前设**：OpenMP 线程池一旦建起来就改不了。
# 每个 worker 单线程（理由见模块开头）。
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

# 结果根：用户 2026-08-31 指定 `{根}/result-s0tf_1_16000/1_4000`。
# 根可由环境变量 S0_RESULT_ROOT 覆盖（集群上指向共享盘），默认仍在仓内 analysis/ 下。
_RES = os.environ.get("S0_RESULT_ROOT")
SAVE_ROOT = (Path(_RES).expanduser() if _RES
             else S0_ROOT / "analysis" / "package1" / "crest")
ETKDG_ROOT = S0_ROOT / "analysis" / "package1" / "{}_{}".format(*FULL_RANGE)

#: 实测（`VmRSS`，算过 19 原子分子之后的稳态）。用于并发规划。
WORKER_HOST_MB = 821.0
#: CUDA 上下文 + 模型 + 激活的粗估（实测显存分配峰值 42–75 MB，上下文另计约 300 MB）
WORKER_GPU_MB = 450.0
#: 内存留白：不把可用内存吃满
MEM_MARGIN = 0.80


def qid_of(i):
    return "dsgdb9nsd_{:06d}".format(i)


def _manifest(obj, stem, title=None):
    """清单/小结也不写 JSON：写 `.log`（人读）+ `.parquet`（扁平的标量部分）。"""
    stem = Path(stem)
    r = report.Report(title or stem.name, subtitle="包 1 · CREST 支路")
    r.json_dump(obj, title="内容")
    r.write(stem.with_suffix(".log"))
    flat = {k: v for k, v in obj.items()
            if isinstance(v, (str, int, float, bool)) or v is None}
    if flat:
        record.write_parquet_rows([flat], stem.with_suffix(".parquet"))
    return stem.with_suffix(".log")


# ======================================================================================
# 并发规划
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
    """N = min(CPU 数, 可用内存/单 worker 内存 [, 可用显存/单 worker 显存])。

    用户 2026-08-31 指定的就是这个式子。这里把**每一项都报出来**，
    好让"为什么是这个数"可查，而不是一个凭空的常数。
    """
    n_cpu_total = os.cpu_count() or 1
    n_cpu = max(1, n_cpu_total - int(reserve_cpus))
    host = mem_available_mb()
    n_mem = int(host * MEM_MARGIN / WORKER_HOST_MB) if host == host else n_cpu
    limits = [("CPU 数 os.cpu_count() = {} 减去预留 {}".format(
                  n_cpu_total, int(reserve_cpus)), n_cpu),
              ("可用内存 {:.0f} MB x {:.0%} / 单 worker {:.0f} MB".format(
                  host, MEM_MARGIN, WORKER_HOST_MB), n_mem)]
    n = min(n_cpu, n_mem)
    g = gpu_free_mb(device)
    if g is not None:
        n_gpu = max(1, int(g * MEM_MARGIN / WORKER_GPU_MB))
        limits.append(("可用显存 {:.0f} MB x {:.0%} / 单 worker {:.0f} MB".format(
            g, MEM_MARGIN, WORKER_GPU_MB), n_gpu))
        n = min(n, n_gpu)
    if requested:
        limits.append(("命令行 --workers", requested))
        n = min(n, requested)
    return max(1, int(n)), limits


# ======================================================================================
# worker 侧
# ======================================================================================
_W = {}          # worker 进程内的全局状态：calculator、套接字、服务线程


def _worker_init(lock, device):
    """每个 worker 进程起来时跑一次。**模型加载在锁里**，所以是串行的。"""
    import torch
    torch.set_num_threads(1)
    with lock:                                     # <- 串行加载
        t0 = time.time()
        calc, name, prov = engine.calculator(device=device)
        try:                                       # 预热：建图、选 kernel、分配显存
            from ase import Atoms
            a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
            a.calc = calc
            a.get_potential_energy()
        except Exception:
            pass
        load_s = time.time() - t0
    sock = str(config.runs_dir("sockets", CFG) / "s0_mace_w{}.sock".format(os.getpid()))
    from openqha import mace_server
    th, stop = mace_server.serve_in_thread(sock, calc)
    _W.update(calc=calc, engine=name, prov=prov, socket=sock, stop=stop,
              device=device, load_seconds=load_s)


def _worker_one(task):
    """一个分子从头到尾。在 worker 进程里跑，返回一份可序列化的记录。"""
    qid, smiles, args = task
    # ---- 墙钟预算：到点之后不再**开始**新分子 ------------------------------------------
    # 已经在跑的不腰斩（那份机时会白花）；只是不再领新活。
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
                # 阶段一到此为止：把**系综**与原生输出交给阶段二
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
            # **不写 JSON**（用户 2026-08-31）：.log 给人读、.parquet 带精度、
            # CREST 原生输出只改名。
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
            # 失败时 CREST 的原生输出**更要留**——根因就在里面
            if CB["keep_crest_out"] and (wd / "crest.out").exists():
                try:
                    d = Path(args["dirs"]["out"])
                    d.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(wd / "crest.out", d / "crest_{}.out".format(qid))
                except OSError:
                    pass
        return dict(ok=False, qm9_index=qid, failure=fr)


def _finish_crest_only(qid, smiles, wd, run_rec, start_info, args, t_start):
    """阶段一的收尾：系综 + 原生输出 + 运行记录（不做任何 MACE 的收紧/Hessian）。"""
    import shutil as _sh
    d = Path(args["dirs"]["ens"])
    d.mkdir(parents=True, exist_ok=True)
    src = Path(wd) / "crest_conformers.xyz"
    if not src.exists():
        raise RuntimeError("阶段一结束但没有 crest_conformers.xyz")
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
               note="阶段一（CREST 采样）产物。收紧 / 去重 / Hessian 由阶段二完成。")
    if not args["pilot"]:
        record.write_molecule(rec, args["dirs"], qid, basins=None,
                              crest_out=None, keep_crest_out=False)
    return dict(ok=True, qm9_index=qid, record=_slim(rec))


def _load_crest_stage(qid, args):
    """阶段二：把阶段一存下来的系综与运行记录读回来。"""
    ens = Path(args["dirs"]["ens"]) / "{}_ensemble.xyz".format(qid)
    if not ens.exists():
        raise RuntimeError("阶段二找不到阶段一的系综: {}".format(ens))
    args["_ens"] = str(ens)
    return (dict(seconds=None, terminated_normally=True, n_terminated_early=0,
                 from_stage1=True),
            dict(source="stage1-crest", path=str(ens)))


def _crest_stage(qid, smiles, wd, calc, args):
    """准备起点 + 跑 CREST。复用**已跑完且设置相符**的工作目录（缺陷 57）。"""
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

    run_rec = _go(C.get("shake"))          # null => 用 CREST 自己的默认
    # ---- `terminated EARLY` 的定点重试 ------------------------------------------------
    # 根因（2026-08-31 实测，见 openqha/crest.py 的 shake 注释）：CREST 默认
    # `shake = 2`（约束全部键）在**张力环**体系上迭代不收敛 -> 分子动力学被判失败。
    # `shake = 1`（只约束含氢键）实测把三个失败分子的 EARLY 全部清零。
    # **只在失败时重试**，不改动已经跑通的分子的采样条件。
    fb = CB.get("shake_fallback")
    if (not run_rec.get("ok") and fb is not None
            and (run_rec.get("n_terminated_early") or 0) > 0):
        early0 = run_rec.get("n_terminated_early")
        run_rec = _go(int(fb))
        run_rec["shake_fallback_used"] = int(fb)
        run_rec["n_terminated_early_before_fallback"] = early0
    if not run_rec.get("ok"):
        raise RuntimeError(
            "CREST 未通过判据: 正常收尾 {}, terminated EARLY {}, 返回码 {}{}".format(
                run_rec.get("terminated_normally"), run_rec.get("n_terminated_early"),
                run_rec.get("returncode"),
                "（已用 shake={} 重试过）".format(fb) if fb is not None else ""))
    return run_rec, start_info


def _slim(rec):
    """回传给主进程的精简记录 —— 完整记录已落盘，不必再走一次管道。"""
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
# 单分子的各步（数值行为与改架构之前完全一致）
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
        raise RuntimeError("CREST 没有产出 crest_conformers.xyz")
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
        raise RuntimeError("并池去重 + 虚频筛选之后一个盆都不剩")

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
        free_energy_note="没有算自由能 —— 对称数与电子简并度必须显式声明，本仓不自动推导。")
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
# 主进程
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
    """默认**固定种子的随机置换** —— 任何前缀都是无偏样本。

    `schedule="lpt"` 改成"最长作业优先"（按构象柔性预测的成本降序）：
    能缩短总墙钟，**但会破坏前缀的无偏性**（前缀会系统性偏向高柔性分子），
    所以它不是默认。
    """
    if schedule == "lpt":
        try:
            import pandas as pd
            df = pd.read_parquet(
                S0_ROOT / "analysis" / "package1_flexibility_ranking.parquet")
            rank = {q: i for i, q in enumerate(df["qm9_index"])}
            return sorted(ids, key=lambda q: rank.get(q, 10 ** 9))
        except Exception as exc:
            print("   最长作业优先所需的排名表读不到（{}），退回随机置换".format(exc))
    rng = np.random.default_rng(ORDER_SEED)
    return [ids[i] for i in rng.permutation(len(ids))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=CHUNK[0])
    ap.add_argument("--hi", type=int, default=CHUNK[1])
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0,
                    help="上限；默认 min(CPU 数, 可用内存/单 worker 内存[, 显存/…])")
    ap.add_argument("--crest-threads", type=int, default=1,
                    help="每个 CREST 的线程数；每分子一个 worker 时应为 1")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | cuda:N")
    ap.add_argument("--schedule", choices=("random", "lpt"), default="random")
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--no-hessian", action="store_true")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--redo-crest", action="store_true")
    ap.add_argument("--plan-only", action="store_true", help="只报并发规划, 不跑")
    ap.add_argument("--reserve-cpus", type=int, default=0,
                    help="留给别的会话的逻辑核数（例如另一个会话要跑 GPU 分子动力学）")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="墙钟预算（小时）。到点后不再开始新分子；在跑的让它跑完")
    ap.add_argument("--phase", choices=("full", "crest", "analyse"), default="full",
                    help="full=一个 worker 从头做到尾（本机默认）; "
                         "crest=只做 CREST 采样（阶段一，CPU 节点）; "
                         "analyse=只做收紧+去重+Hessian（阶段二，读阶段一的系综）")
    ap.add_argument("--ens-from", default="",
                    help="阶段二从哪个阶段的目录读系综（默认同一个 --stage 目录）")
    ap.add_argument("--stage", default="",
                    help="分阶段跑时的子目录名（例如 stage1-gfnff / stage2-mace）。"
                         "留空则不分子目录")
    ap.add_argument("--shard", default="0/1",
                    help="k/n —— 作业阵列的第 k 个分片（共 n 片）。"
                         "按处理顺序取模，所以每片仍是无偏随机样本")
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
    prov = engine.provenance()          # **只读权重做哈希，不加载模型**

    print("=" * 104)
    print("包 1 · CREST 支路   编号 {}-{}   每分子一个 worker".format(args.lo, args.hi))
    print("=" * 104)
    print("CREST        {} (commit {})  runtype={} refine={} optlev={}".format(
        v["version"], v["commit"], C["runtype"], C["refine"], C["optlev"]))
    print("引擎         MACE-OFF23-SC  SHA-256 {}...  dtype={}  device={}".format(
        prov["sha256"][:16], prov["dtype"], args.device))
    print()
    print("[并发规划]  N = min(下列各项)")
    for label, val in limits:
        print("   {:<56} {:>5}".format(label, val))
    print("   {:<56} {:>5}".format("**采用**", n_workers))
    print("   每 worker: 1 个 torch 线程, {} 个 CREST 线程, 一份 MACE 模型（约 {:.0f} MB）"
          .format(args.crest_threads, WORKER_HOST_MB))
    print("   模型加载**串行**（跨进程锁），加载完即开工")
    if args.plan_only:
        return

    rows = load_index(args.lo, args.hi)
    all_ids = [qid_of(i) for i in range(args.lo, args.hi + 1) if qid_of(i) in rows]
    gates = filters.enabled_gates(CFG)
    ids, rejected, gate_counts = filters.screen_many(
        ((q, rows[q]["qm9_smiles"]) for q in all_ids), CFG)
    order = processing_order(ids, args.schedule)

    print()
    print("门 {} -> 通过 {} 个 / 区间 {} 个".format("/".join(gates), len(ids), len(all_ids)))
    print("顺序         {}".format(
        "固定种子 {} 的随机置换（任何前缀都是无偏样本）".format(ORDER_SEED)
        if args.schedule == "random" else
        "**最长作业优先** —— 缩短总墙钟, 但前缀不再无偏"))
    try:
        shard_k, shard_n = (int(x) for x in args.shard.split("/"))
    except ValueError:
        raise SystemExit("--shard 要写成 k/n，例如 3/16")
    if not (0 <= shard_k < shard_n):
        raise SystemExit("--shard 的 k 必须在 [0, n) 内")
    if shard_n > 1:
        order = [q for i, q in enumerate(order) if i % shard_n == shard_k]
        print("分片         {}/{} -> 本片 {} 个分子（按处理顺序取模，"
              "所以本片仍是无偏随机样本）".format(shard_k, shard_n, len(order)))

    # 续跑判据：产物是 <编号>.parquet（带精度的那一份）。不再看 .json。
    todo = [q for q in order
            if args.redo or not (dirs["mol"] / (q + ".parquet")).exists()]
    n_done_before = len(order) - len(todo)
    if args.pilot:
        todo = todo[:args.pilot]
    elif args.limit:
        todo = todo[:max(0, args.limit - n_done_before)]
    print("已有产物 {} 个; 本次要跑 {} 个".format(n_done_before, len(todo)))
    if not todo:
        print("没有要跑的分子。")
        return

    if not args.pilot:
        _manifest(dict(
            package="包 1 · CREST 支路（每分子一个 worker）",
            generated_by="scripts/production/s0_package1_crest_census.py",
            qm9_range=[args.lo, args.hi], config_path=CFG["_path"],
            engine=prov, crest=v, crest_settings=dict(C),
            crest_batch_settings=dict(CB),
            concurrency=dict(n_workers=n_workers, device=args.device,
                             limits=[dict(label=l, value=x) for l, x in limits],
                             worker_host_mb=WORKER_HOST_MB,
                             model_loading="串行（跨进程锁）",
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
        print("墙钟预算     {:.1f} 小时；到点后不再开始新分子（在跑的让它跑完)".format(
            args.max_hours))
        print("             按当前实测吞吐，预计能完成的分子数会在小结里报出")

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
                # **按 phase 分支**：阶段一还没有盆级字段（它们是 None），
                # 用一行通用格式去打会抛 TypeError 并**杀掉主循环**，
                # 连带丢掉还在飞的那个分子（2026-08-31 端到端测试实测）。
                if args.phase == "crest":
                    print("   [{:5d}/{:5d}] {}  CREST {:3d} 构象 / {:6.0f} s; "
                          "EARLY {}; 单分子 {:6.0f} s; 吞吐 {:.0f} s/分子".format(
                              i, len(tasks), r["qm9_index"],
                              r.get("n_conformers_reported_by_crest") or 0,
                              _f(r.get("crest_seconds")),
                              r.get("crest_terminated_early"),
                              _f(r.get("total_seconds")), el / i), flush=True)
                else:
                    print("   [{:5d}/{:5d}] {}  CREST {:3d} 构象/{:5.0f} s -> {:3d} 盆 "
                          "(仅C {:2d}/仅E {:2d}); 修正 {:+.4f}; 单分子 {:5.0f} s; "
                          "吞吐 {:.0f} s/分子".format(
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
                print("   [{:5d}/{:5d}] 失败 {} {} —— {}: {}".format(
                    i, len(tasks), f["qm9_index"], f["smiles"], f["error_type"],
                    str(f["error"])[:70]), flush=True)

    if skipped:
        print()
        print("   墙钟预算到点，未开始的分子 {} 个（下次运行会接着跑）".format(len(skipped)))
    _summarise(done, failed, order, time.time() - t0, chunk_dir, args, prov, v,
               n_workers, limits, skipped)


def _summarise(done, failed, order, elapsed, chunk_dir, args, prov, v,
               n_workers, limits, skipped=()):
    print()
    print("=" * 104)
    print("小结")
    print("=" * 104)
    print("成功 {} 个, 失败 {} 个, 墙钟 {:.0f} s = {:.2f} 小时, {} 个 worker".format(
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
            # 阶段一只报它自己算出来的量
            nc = arr("n_conformers_reported_by_crest")
            cr, tot = arr("crest_seconds"), arr("total_seconds")
            ee = arr("crest_terminated_early")
            per = elapsed / max(len(done) + len(failed), 1)
            print("CREST 构象  均值 {:.2f} 中位 {:.0f} 最大 {:.0f}".format(
                np.nanmean(nc), np.nanmedian(nc), np.nanmax(nc)))
            print("单分子耗时  均值 {:.0f} s 中位 {:.0f} s 最大 {:.0f} s".format(
                np.nanmean(tot), np.nanmedian(tot), np.nanmax(tot)))
            print("**吞吐**    {:.0f} s/分子; 并行加速 {:.1f} 倍（{} 个 worker）".format(
                per, np.nanmean(tot) / max(per, 1e-9), n_workers))
            n_bad = int(np.nansum(ee > 0))
            print("terminated EARLY 不为 0 的分子: {} 个   <- 判据: 必须是 0".format(n_bad))
            remaining = len(order) - len(done)
            print()
            print("成本外推    剩余 {} 个 -> {:.1f} 小时 = {:.1f} 天".format(
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
        print("CREST 构象  均值 {:.2f} 最大 {:.0f}; 并池后盆数 均值 {:.2f} 最大 {:.0f}".format(
            np.nanmean(nc), np.nanmax(nc), np.nanmean(nb), np.nanmax(nb)))
        print("盆的来源    仅 CREST 均值 {:.2f}; 仅 ETKDG 均值 {:.2f}".format(
            np.nanmean(co), np.nanmean(eo)))
        print("单分子耗时  均值 {:.0f} s 中位 {:.0f} s 最大 {:.0f} s "
              "(其中 CREST 均 {:.0f} s, 分析 均 {:.0f} s)".format(
                  np.nanmean(tot), np.nanmedian(tot), np.nanmax(tot),
                  np.nanmean(cr), np.nanmean(an)))
        print("**吞吐**    {:.0f} s/分子（墙钟/分子数）; 单分子串行 {:.0f} s "
              "=> 并行加速 {:.1f} 倍（{} 个 worker）".format(
                  per, np.nanmean(tot), np.nanmean(tot) / max(per, 1e-9), n_workers))
        print("最低盆权重  均值 {:.3f}; < 0.9 的占 {:.1%}".format(
            np.nanmean(w0), float(np.nanmean(w0 < 0.9))))
        print("构象修正    均值 {:+.4f} 中位 {:+.4f} 最负 {:+.4f} kcal/mol".format(
            np.nanmean(corr), np.nanmedian(corr), np.nanmin(corr)))
        remaining = len(order) - len(done)
        print()
        print("成本外推    剩余 {} 个 -> {:.1f} 小时 = {:.1f} 天".format(
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
    """None -> nan，好让 {:.0f} 这类格式不炸。"""
    return float("nan") if v is None else float(v)


def _finish_summary(payload, failed, chunk_dir, args):
    if failed:
        print()
        print("失败清单 ({} 个):".format(len(failed)))
        for f in failed[:20]:
            print("   {} {:24s} {}: {}".format(f["qm9_index"], f["smiles"],
                                               f["error_type"], str(f["error"])[:70]))
    if not args.pilot:
        name = ("summary" if args.shard == "0/1"
                else "summary_shard{}".format(args.shard.replace("/", "of")))
        _manifest(payload, chunk_dir / name, title="包 1 · CREST 支路   本次运行小结")
        print()
        print("落盘: {} 与同名 .log".format(chunk_dir))


if __name__ == "__main__":
    main()
