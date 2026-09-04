"""把 MACE-OFF23-SC 的**能量与力**对标复合参考；**频率能力存在，默认关闭**。

CALIBRATION. Benchmarks energies and forces against the composite reference.

范围经过两次裁定，后一次修订前一次，两条都写在这里，不覆盖历史：

* **2026-08-29（上午）**：我们关心的是**能量与力**的精度；**频率的验证代价过高**
  （要验证复合频率就得真算一次 `CCSD(T)/cc-pVTZ` 的 61 次梯度，约 5.1 天），
  而原文给的 0.073 eV/Å 是**力**的误差，**力的精度不自动等于频率的精度**。
* **2026-08-29（下午）**：用户指示「改为能够产出」。于是 `--hessian` 恢复了这个能力：
  由**复合力**做中心差分得到复合 Hessian，再走与包 2 相同的 Eckart 投影与对角化。
  **默认关闭**；打开时先把 `6N ×（项池大小）` 次单点的代价算给你看，
  不加 `--hessian-confirm-cost` 就只报代价、不开跑。
  裁定「代价过高」针对的是**把频率当作生产参考量**，不是禁止这个能力存在。

两个度量，口径不同，必须分开：

* **力** —— 逐分量比，单位 eV/Å。与 Allen 等原文同口径（他们报的 0.073 eV/Å
  就是「每个力分量的均方根误差」）。**只能在离开极小点的结构上读** ——
  在 MACE 自己的极小点上，MACE 的力按构造约等于零，「偏差」恒等于参考力本身，
  量到的是两个极小点的几何差，不是力的精度（`D0-81`）。
* **能量** —— **只能比相对量**。MACE 的绝对能量与耦合簇的绝对能量零点不同，
  相减没有意义；能比的是**同一物种不同构象之间的能量差**、
  **热位移结构相对它自己母极小点的能量差**，以及同分子式两个物种之间的反应能。

**项池**：一条基组阶梯（DZ → TZ* → QZ*）里，低几级的单点是高一级的子集。
本脚本按**去重后的项池**求解，所以量出整条阶梯的代价 = 量出最高一级的代价，
中间几级是**免费**的 —— 「阶梯收敛了没有」于是从假设变成一个可读的数。

用法::

    # 冒烟：单项 MP2/cc-pVDZ
    python scripts/calibration/s0_composite_energy_force.py --recipe smoke_mp2_dz --limit 3 --displace 3

    # E2（2026-08-29 用户裁定的精度「MP2 DZ* -> QZ*」）：44 个结构、一条阶梯
    python scripts/calibration/s0_composite_energy_force.py --ladder mp2_dz_to_qz --displace 3 \
        --nprocs 12 --tag E2 --resume

    # 频率能力（默认关闭；先看代价）
    python scripts/calibration/s0_composite_energy_force.py --recipe dz_base --limit 1 --hessian

产物::

    analysis/composite_energy_force_<tag>.json           最终汇总
    analysis/composite_energy_force_<tag>.partial.jsonl  每算完一个结构就落一行（可续跑）
"""
import argparse
import json
import sys
import time
from pathlib import Path

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

from openqha import S0_ROOT, config, conformers, engine, hessian, orca

OUT = S0_ROOT / "analysis"


def load_structures(cfg, limit=None):
    """取包 2 已经落盘的每个物种的每个盆（真极小点）。"""
    pkg2 = OUT / "package2"
    if not pkg2.exists():
        raise FileNotFoundError(
            "{} 不存在 —— 先跑 scripts/calibration/s0_package2_hessian_benchmark.py".format(pkg2))
    from ase.io import read
    items = []
    for qid, spec in sorted(cfg["species"].items()):
        d = pkg2 / spec["name"]
        for xyz in sorted(d.glob("basin*.xyz")):
            items.append(dict(qm9_index=qid, name=spec["name"], zh=spec["zh"],
                              basin=int(xyz.stem.replace("basin", "")),
                              path=str(xyz), atoms=read(str(xyz))))
    if not items:
        raise FileNotFoundError("在 {} 下没有找到任何 basin*.xyz".format(pkg2))
    return items[:limit] if limit else items


def resolve_recipes(p2, args):
    """把 `--recipe` / `--recipes` / `--ladder` 化成 [(名字, terms), ...]。"""
    recipes = p2["composite_recipes"]
    if args.ladder:
        ladders = p2.get("recipe_ladders", {})
        if args.ladder not in ladders:
            raise KeyError("配置里没有阶梯 {!r}；可选: {}".format(
                args.ladder, sorted(ladders)))
        names = list(ladders[args.ladder]["recipes"])
    elif args.recipes:
        names = [x.strip() for x in args.recipes.split(",") if x.strip()]
    else:
        names = [args.recipe or p2["reference_level"]]
    out = []
    for n in names:
        if n not in recipes:
            raise KeyError("配置里没有配方 {!r}；可选: {}".format(n, sorted(recipes)))
        out.append((n, [[t[0], t[1], t[2]] for t in recipes[n]["terms"]]))
    return out


def key_of(it):
    return "{}|{}|{}".format(it["name"], it["basin"],
                             "min" if it.get("displaced") is None
                             else "d{}".format(it["displaced"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default=None,
                    help="单个配方名；缺省用配置里的 reference_level")
    ap.add_argument("--recipes", default=None, help="逗号分隔的多个配方，共用一个项池")
    ap.add_argument("--ladder", default=None, help="配置里 recipe_ladders 的名字")
    ap.add_argument("--limit", type=int, default=None, help="只算前 N 个极小点（冒烟用）")
    ap.add_argument("--nprocs", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=None, help="单次 ORCA 的秒数上限")
    ap.add_argument("--displace", type=int, default=0,
                    help="每个极小点额外采 N 个热位移结构。"
                         "**力的对标必须用它** —— 在 MACE 自己的极小点上比力是退化的")
    ap.add_argument("--temperature", type=float, default=298.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="产物文件名的后缀；缺省用配方/阶梯名")
    ap.add_argument("--resume", action="store_true",
                    help="读已有的 .partial.jsonl，跳过已算完的结构")
    ap.add_argument("--hessian", action="store_true",
                    help="**默认关闭的能力**：由复合力做中心差分给出复合 Hessian 与频率。"
                         "只对极小点做。不加 --hessian-confirm-cost 就只报代价、不开跑")
    ap.add_argument("--hessian-confirm-cost", action="store_true",
                    help="看过代价之后，真的开跑复合 Hessian")
    ap.add_argument("--hessian-delta", type=float, default=hessian.DELTA_A,
                    help="复合 Hessian 的中心差分位移，Å")
    args = ap.parse_args()

    cfg = config.load()
    p2 = config.package(2, cfg)
    chosen = resolve_recipes(p2, args)
    pool_terms = orca.unique_terms([t for _n, t in chosen])
    tag = args.tag or (args.ladder or "_".join(n for n, _t in chosen))

    calc, engine_name, prov = engine.calculator()
    items = load_structures(cfg, args.limit)

    # ---- 热位移: 力的对标必须离开极小点 ----------------------------------------------
    displaced_rec = None
    if args.displace:
        print("采 {} 个 {:.2f} K 热位移结构/极小点 ...".format(
            args.displace, args.temperature))
        extra, displaced_rec = [], []
        for it in items:
            s_list, rec = hessian.thermal_displacements(
                it["atoms"], calc, temperature_K=args.temperature,
                n_samples=args.displace, seed=args.seed)
            rec["parent"] = "{} basin {}".format(it["name"], it["basin"])
            displaced_rec.append(rec)
            for j, st in enumerate(s_list):
                extra.append(dict(qm9_index=it["qm9_index"], name=it["name"],
                                  zh=it["zh"], basin=it["basin"],
                                  path=it["path"] + "#displaced{}".format(j),
                                  displaced=j, atoms=st))
            print("   {:20s} 盆 {}  均方根位移 {} A  拒绝 {} 次".format(
                it["name"], it["basin"],
                [round(x, 3) for x in rec["rms_displacement_A"]],
                rec["n_draws_rejected"]))
        items = items + extra
        print()

    print("=" * 100)
    print("能量与力的对标   {}  vs  {}".format(engine_name, tag))
    print("=" * 100)
    for name, terms in chosen:
        print("配方 {}".format(name))
        for c, m, b in terms:
            print("      {:+d}  {:8s} {}".format(int(c), m, b))
    print("项池（去重后每个几何只跑这些）: {}".format(
        ", ".join("{}/{}".format(m, b) for m, b in pool_terms)))
    print("自旋 {}   ORCA {}".format(p2.get("reference_spin", "?"), orca.orca_binary()))
    print("结构 {} 个（{} 个真极小点 + {} 个热位移）".format(
        len(items), sum(1 for i in items if i.get("displaced") is None),
        sum(1 for i in items if i.get("displaced") is not None)))
    print()

    # ---- 断点续跑 ---------------------------------------------------------------------
    part = OUT / "composite_energy_force_{}.partial.jsonl".format(tag)
    part.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if args.resume and part.exists():
        for line in part.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r
        print("续跑：已有 {} 个结构的结果，跳过它们。".format(len(done)))
        print()

    results, t0 = [], time.time()
    for k, it in enumerate(items):
        kk = key_of(it)
        if kk in done:
            results.append(done[kk])
            continue
        a = it["atoms"]
        a.calc = calc
        e_mace = float(a.get_potential_energy())
        f_mace = a.get_forces()

        print("   {:>2}/{:<2} {:20s} 盆 {} {}".format(
            k + 1, len(items), it["name"], it["basin"],
            "" if it.get("displaced") is None
            else "位移 {}".format(it["displaced"])), flush=True)

        def prog(i, n, m, b, s):
            print("      [{}/{}] {:8s} {:9s} {:7.1f} s".format(i, n, m, b, s),
                  flush=True)

        pool = orca.term_pool(a.get_chemical_symbols(), a.get_positions(),
                              pool_terms, nprocs=args.nprocs,
                              timeout_s=args.timeout, progress=prog)

        per_recipe = {}
        for name, terms in chosen:
            ref = orca.combine(pool, terms)
            fm = orca.force_metrics(ref["forces_eV_A"], f_mace)
            per_recipe[name] = dict(
                energy_ref_eV=ref["energy_eV"],
                forces_ref_eV_A=ref["forces_eV_A"].tolist(),
                force_metrics=fm, orca_version=ref["orca_version"],
                per_term=ref["per_term"])
            print("      {:24s} 力: 平均绝对偏差 {:.4f}  均方根偏差 {:.4f}  "
                  "最大 {:.4f}   （参考力本身的均方根 {:.4f}）eV/A".format(
                      name, fm["mae_eV_A"], fm["rmse_eV_A"],
                      fm["max_abs_eV_A"], fm["ref_rms_eV_A"]), flush=True)

        rec = dict(key=kk, qm9_index=it["qm9_index"], name=it["name"],
                   basin=it["basin"], displaced=it.get("displaced"),
                   path=it["path"], n_atoms=len(a),
                   energy_mace_eV=e_mace, forces_mace_eV_A=f_mace.tolist(),
                   pool_seconds=float(sum(r["seconds"] for r in pool.values())),
                   per_recipe=per_recipe)
        results.append(rec)
        # **每算完一个结构就落盘** —— 8 小时的跑不能只在结尾写一次
        with open(str(part), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- 汇总 -------------------------------------------------------------------------
    at_min = [r for r in results if r.get("displaced") is None]
    off_min = [r for r in results if r.get("displaced") is not None]

    def pool_forces(subset, name):
        if not subset:
            return None
        aa = np.concatenate([np.array(r["per_recipe"][name]["forces_ref_eV_A"]).ravel()
                             for r in subset])
        bb = np.concatenate([np.array(r["forces_mace_eV_A"]).ravel() for r in subset])
        return orca.force_metrics(aa, bb)

    def relative_energies(name):
        """(a) 构象相对能量：只用极小点；(b) 位移能量：相对各自的母极小点。"""
        conf, by_species = [], {}
        for r in at_min:
            by_species.setdefault(r["qm9_index"], []).append(r)
        for qid, rs in by_species.items():
            if len(rs) < 2:
                continue
            e0m = min(x["energy_mace_eV"] for x in rs)
            e0r = min(x["per_recipe"][name]["energy_ref_eV"] for x in rs)
            for x in rs:
                dm = (x["energy_mace_eV"] - e0m) * conformers.EV_TO_KCAL
                dr = ((x["per_recipe"][name]["energy_ref_eV"] - e0r)
                      * conformers.EV_TO_KCAL)
                conf.append(dict(qm9_index=qid, name=x["name"], basin=x["basin"],
                                 rel_mace_kcal=dm, rel_ref_kcal=dr,
                                 diff_kcal=dm - dr))
        disp, parent = [], {(r["qm9_index"], r["basin"]): r for r in at_min}
        for r in off_min:
            pa = parent.get((r["qm9_index"], r["basin"]))
            if pa is None:
                continue
            dm = (r["energy_mace_eV"] - pa["energy_mace_eV"]) * conformers.EV_TO_KCAL
            dr = ((r["per_recipe"][name]["energy_ref_eV"]
                   - pa["per_recipe"][name]["energy_ref_eV"]) * conformers.EV_TO_KCAL)
            disp.append(dict(qm9_index=r["qm9_index"], name=r["name"],
                             basin=r["basin"], displaced=r["displaced"],
                             rel_mace_kcal=dm, rel_ref_kcal=dr, diff_kcal=dm - dr))
        return conf, disp

    def stat(rows):
        if not rows:
            return None
        d = np.array([x["diff_kcal"] for x in rows])
        return dict(n=len(rows), mae_kcal=float(np.abs(d).mean()),
                    rmse_kcal=float(np.sqrt((d ** 2).mean())),
                    max_abs_kcal=float(np.abs(d).max()),
                    signed_mean_kcal=float(d.mean()))

    summary = {}
    print()
    print("=" * 100)
    print("汇总")
    print("=" * 100)
    for name, _terms in chosen:
        fa, fo = pool_forces(at_min, name), pool_forces(off_min, name)
        conf, disp = relative_energies(name)
        summary[name] = dict(force_at_minima=fa, force_displaced=fo,
                             relative_energy_conformers=conf,
                             relative_energy_displacements=disp,
                             energy_stat_conformers=stat(conf),
                             energy_stat_displacements=stat(disp))
        print()
        print("### 配方 {}".format(name))
        for tagline, pm, degenerate in (
                ("极小点上（**退化，不可当作力的精度**）", fa, True),
                ("热位移上（**这才是力的精度**）", fo, False)):
            if pm is None:
                continue
            print("  力 —— {}".format(tagline))
            print("     {} 个分量: 平均绝对偏差 {:.4f}  **均方根偏差 {:.4f}**  "
                  "最大 {:.4f} eV/A".format(
                      pm["n_components"], pm["mae_eV_A"], pm["rmse_eV_A"],
                      pm["max_abs_eV_A"]))
            print("     有符号均值 {:+.4f}；参考力本身的均方根 {:.4f} eV/A".format(
                pm["signed_mean_eV_A"], pm["ref_rms_eV_A"]))
            if degenerate:
                print("     ** MACE 的力在这里按构造约等于零，偏差恒等于参考力本身。")
        for lab, st in (("构象相对能量（只用极小点）",
                         summary[name]["energy_stat_conformers"]),
                        ("热位移能量（相对各自母极小点）",
                         summary[name]["energy_stat_displacements"])):
            if st is None:
                print("  {}: 无可比对象，跳过。".format(lab))
                continue
            print("  {}（{} 个）: 平均绝对偏差 {:.4f}  均方根偏差 {:.4f}  "
                  "最大 {:.4f} kcal/mol".format(
                      lab, st["n"], st["mae_kcal"], st["rmse_kcal"],
                      st["max_abs_kcal"]))
    print()
    print("   参照：Allen 等原文的复合式相对真 CCSD(T)/QZ 力的均方根误差是 0.073 eV/A")

    # ---- 阶梯：相邻两级之间的差 ---------------------------------------------------------
    ladder = None
    if len(chosen) > 1:
        ladder = []
        names = [n for n, _t in chosen]
        for lo, hi in zip(names[:-1], names[1:]):
            de, df = [], []
            for r in results:
                de.append((r["per_recipe"][hi]["energy_ref_eV"]
                           - r["per_recipe"][lo]["energy_ref_eV"])
                          * conformers.EV_TO_KCAL)
                df.append(np.array(r["per_recipe"][hi]["forces_ref_eV_A"])
                          - np.array(r["per_recipe"][lo]["forces_ref_eV_A"]))
            dfa = np.concatenate([x.ravel() for x in df])
            ladder.append(dict(
                lower=lo, upper=hi,
                energy_shift_mean_kcal=float(np.mean(de)),
                energy_shift_spread_kcal=float(np.std(de)),
                force_rms_change_eV_A=float(np.sqrt((dfa ** 2).mean())),
                force_max_change_eV_A=float(np.abs(dfa).max())))
        print()
        print("阶梯：相邻两级**参考本身**的差（不是与 MACE 比，是参考收敛到哪了）")
        for r in ladder:
            print("   {:22s} -> {:22s}  力的均方根变化 {:.4f} eV/A（最大 {:.4f}）；"
                  "能量平移 {:+.3f} ± {:.3f} kcal/mol".format(
                      r["lower"], r["upper"], r["force_rms_change_eV_A"],
                      r["force_max_change_eV_A"], r["energy_shift_mean_kcal"],
                      r["energy_shift_spread_kcal"]))
        print("   能量平移的**均值**无意义（绝对零点变了），有意义的是它的**离散**：")
        print("   离散小 = 这一级基组修正对不同结构几乎是同一个常数 = 相对量已经收敛。")

    # ---- 复合 Hessian（默认关闭的能力）-------------------------------------------------
    hess = None
    if args.hessian:
        hess = run_hessian(args, chosen, at_min, calc)

    payload = dict(
        generated_by="scripts/calibration/s0_composite_energy_force.py",
        scope=("能量与力为主（2026-08-29 上午裁定）；频率能力存在但默认关闭"
               "（2026-08-29 下午用户指示「改为能够产出」）"),
        engine=prov, tag=tag,
        recipes={n: t for n, t in chosen},
        pool_terms=[list(x) for x in pool_terms],
        reference_spin=p2.get("reference_spin"),
        orca_binary=orca.orca_binary(),
        n_structures=len(results),
        summary_per_recipe=summary,
        ladder_between_levels=ladder,
        displacement_sampling=displaced_rec,
        composite_hessian=hess,
        note_forces=("在 MACE 自己的极小点上比力是**退化的**: MACE 的力约等于零, "
                     "偏差恒等于参考力本身。力的精度只能在**热位移结构**上读。"),
        note_energy=("绝对能量不可比（MACE 与耦合簇零点不同）；"
                     "这里比的是同一物种内部的相对量。"),
        literature_reference=("Allen 等, Reactive Chemistry at Unrestricted Coupled "
                              "Cluster Level: 复合式相对真 CCSD(T)/QZ 力的均方根误差 "
                              "0.073 eV/Å"),
        per_structure=results, total_seconds=time.time() - t0)
    out = OUT / "composite_energy_force_{}.json".format(tag)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("落盘:", out)
    print("逐结构续跑文件:", part)
    print("总用时 {:.0f} s".format(time.time() - t0))


def run_hessian(args, chosen, at_min, calc):
    """复合 Hessian —— 先报代价，确认过才跑。只对极小点做（位移结构不是极小点）。"""
    from ase.io import read
    name, terms = chosen[-1]        # 阶梯里最高的一级
    print()
    print("=" * 100)
    print("复合 Hessian（**默认关闭的能力**，本次被 --hessian 打开）  配方 {}".format(name))
    print("=" * 100)
    # 用**本次实测**的单点耗时做代价估计，而不是用配置里的估计值
    spt = {}
    for r in at_min:
        for t in r["per_recipe"][name]["per_term"]:
            spt.setdefault((t["method"], t["basis"]), []).append(t["seconds"])
    spt = {k: float(np.mean(v)) for k, v in spt.items()}
    total_s = 0.0
    for r in at_min:
        c = orca.hessian_cost(r["n_atoms"], terms, spt)
        total_s += c["estimated_seconds"]
        print("   {:20s} 盆 {}  {} 个原子 -> {} 个位移几何 × {} 项 = {} 次单点，"
              "约 {:.1f} 小时".format(
                  r["name"], r["basin"], c["n_atoms"], c["n_displaced_geometries"],
                  c["n_terms_per_geometry"], c["n_single_points"],
                  c["estimated_hours"]))
    print("   —— 合计约 {:.1f} 小时（{:.2f} 天），用的是**本次实测**的单点耗时".format(
        total_s / 3600.0, total_s / 86400.0))
    if not args.hessian_confirm_cost:
        print("   **未加 --hessian-confirm-cost，只报代价，不开跑。**")
        return dict(mode="cost_only", recipe=name,
                    estimated_seconds=total_s,
                    estimated_hours=total_s / 3600.0,
                    seconds_per_term_measured={"{}/{}".format(*k): v
                                               for k, v in spt.items()},
                    note="能力存在、默认关闭、代价先报。加 --hessian-confirm-cost 才真跑。")
    out = []
    for r in at_min:
        a = read(r["path"])
        print("   跑 {} 盆 {} ...".format(r["name"], r["basin"]), flush=True)
        h, asym, logs = orca.composite_hessian(
            a.get_chemical_symbols(), a.get_positions(), terms,
            delta_A=args.hessian_delta, nprocs=args.nprocs,
            timeout_s=args.timeout,
            progress=lambda i, n: print("      坐标 {}/{}".format(i, n), flush=True))
        rec = hessian.project_and_diagonalise(h, a.get_masses(), a.get_positions())
        rec.update(name=r["name"], basin=r["basin"], recipe=name,
                   hessian_asymmetry_eV_A2=asym, delta_A=args.hessian_delta,
                   n_single_points=len(logs) * len(orca.unique_terms([terms])))
        # 同一几何上的 MACE 频率，供直接对照
        a2 = a.copy()
        a2.calc = calc
        hm, _asym_m = hessian.finite_difference_hessian(a2, calc,
                                                        delta=args.hessian_delta)
        rec["mace_frequencies_cm_inv"] = hessian.project_and_diagonalise(
            hm, a.get_masses(), a.get_positions())["frequencies_cm_inv"]
        d = (np.array(rec["frequencies_cm_inv"])
             - np.array(rec["mace_frequencies_cm_inv"]))
        rec["frequency_deviation"] = dict(
            mae_cm_inv=float(np.abs(d).mean()),
            rmse_cm_inv=float(np.sqrt((d ** 2).mean())),
            max_abs_cm_inv=float(np.abs(d).max()),
            signed_mean_cm_inv=float(d.mean()))
        print("      频率偏差: 平均绝对 {:.2f}  均方根 {:.2f}  最大 {:.2f} cm^-1".format(
            rec["frequency_deviation"]["mae_cm_inv"],
            rec["frequency_deviation"]["rmse_cm_inv"],
            rec["frequency_deviation"]["max_abs_cm_inv"]))
        out.append(rec)
    return dict(mode="computed", recipe=name, per_structure=out)


if __name__ == "__main__":
    main()
