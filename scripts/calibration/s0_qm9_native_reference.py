"""QM9 原生数据构成的**免费独立对比样**: 刚转子-谐振子热力学项.

CALIBRATION. The docstring below settles it: a comparison sample at a different
level of theory, explicitly not a stage 0 result.

QM9 对每个分子都给了 B3LYP/6-31G(2df,p) 下的:
  * 三个转动常数 A, B, C (GHz)  -> 转动项, 连几何都不用读
  * 全部 3N-6 个谐振频率 (cm^-1) -> 振动项 (刚转子-谐振子)
  * 零点能 zpve, 以及 U0 / U / H / G (Hartree)

于是可以零成本地得到一个 `(G - E_el)` 的**谐振参考值**, 用来和 stage 0 的
态密度路线对照。两者之差 = 非谐 + 构象 + 势面之差, 这正是 stage 0 要量的东西。

**这不是 stage 0 的结果，是对比样。** 它的层级是 B3LYP/6-31G(2df,p)，
与讲义的 MACE-OFF23-SC 不同势面, 所以差值里混着势面之差 —— 必须写明。

用法:
    python scripts/calibration/s0_qm9_native_reference.py
产物:
    analysis/qm9_native_reference.json
"""
import csv
import json
import math
from pathlib import Path

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


# ---------------------------------------------------------------- 物理常数
KB_KCAL = 1.987204259e-3            # kcal/(mol*K)
HC_KCAL = 2.85914308e-3             # kcal/mol per cm^-1
T_REF = 298.15
KT = KB_KCAL * T_REF
H_SI = 6.62607015e-34
KB_SI = 1.380649e-23
HARTREE_KCAL = 627.5094740631

ROOT = _repo_root()
INDEX = (ROOT.parent / "stage1-qm9-alchemical-reaction-energy" / "source-data"
         / "index_Chem_composition.csv")

EDGE = "C2H5O1N1_19_36"
# 物种 -> QM9 编号. 边名里的 19 / 36 就是 QM9 索引 (已核对 index_Chem_composition.csv)
SPECIES_QM9 = {"acetamide": "dsgdb9nsd_000019",
               "N-methylformamide": "dsgdb9nsd_000036"}
SYMMETRY_NUMBER = {"acetamide": 1, "N-methylformamide": 1}   # 显式声明, 绝不自动推导
REACTANT, PRODUCT = list(SPECIES_QM9)


def a_harmonic_kcal(nu_cm, temperature_K=T_REF):
    """量子谐振子亥姆霍兹自由能 (含零点能), 零点取势阱底."""
    kt = KB_KCAL * temperature_K
    x = HC_KCAL * nu_cm / kt
    return kt * (0.5 * x + math.log1p(-math.exp(-x)))


def a_rot_from_constants_kcal(a_ghz, b_ghz, c_ghz, sigma, temperature_K=T_REF):
    """由三个转动常数 (GHz) 直接给刚转子转动自由能.

        Theta_i = h * nu_i / k_B      (nu_i 是以 Hz 计的转动常数)
        q_rot   = (sqrt(pi)/sigma) * sqrt(T^3 / (Theta_A Theta_B Theta_C))
    """
    theta = [H_SI * g * 1e9 / KB_SI for g in (a_ghz, b_ghz, c_ghz)]     # K
    q = (math.sqrt(math.pi) / sigma) * math.sqrt(
        temperature_K ** 3 / (theta[0] * theta[1] * theta[2]))
    return -KB_KCAL * temperature_K * math.log(q), theta, q


def load_qm9():
    rows = {}
    want = set(SPECIES_QM9.values())
    with INDEX.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["qm9_index"] in want:
                rows[r["qm9_index"]] = r
    missing = want - set(rows)
    if missing:
        raise KeyError("QM9 索引表里缺: {} —— 拒绝用估计值顶替".format(sorted(missing)))
    return rows


def main():
    qm9 = load_qm9()
    out = dict(edge=EDGE, temperature_K=T_REF,
               level="B3LYP/6-31G(2df,p)  (QM9 原生)",
               source=str(INDEX), species={})

    print("=" * 92)
    print("QM9 原生 B3LYP/6-31G(2df,p) 的刚转子-谐振子参考   边 {}".format(EDGE))
    print("=" * 92)
    for name, qid in SPECIES_QM9.items():
        r = qm9[qid]
        freqs = [float(v) for v in r["frequencies"].split()]
        n_at = sum(int(r[k + "_num"]) for k in ("C", "H", "O", "N", "F"))
        expect = 3 * n_at - 6
        if len(freqs) != expect:
            raise ValueError("{}: 频率 {} 个, 应为 3N-6 = {}".format(name, len(freqs), expect))
        n_imag = sum(1 for f in freqs if f <= 0.0)
        a_vib = sum(a_harmonic_kcal(f) for f in freqs)
        a_rot, theta, q_rot = a_rot_from_constants_kcal(
            float(r["A"]), float(r["B"]), float(r["C"]), SYMMETRY_NUMBER[name])
        rec = dict(qm9_index=qid, smiles=r["qm9_smiles"], n_atoms=n_at,
                   n_frequencies=len(freqs), n_imaginary=n_imag,
                   lowest_frequency_cm_inv=min(freqs),
                   rotational_constants_GHz=[float(r[k]) for k in ("A", "B", "C")],
                   rotational_temperatures_K=theta,
                   symmetry_number=SYMMETRY_NUMBER[name],
                   A_vib_kcal=a_vib, A_rot_kcal=a_rot,
                   G_minus_E_el_kcal=a_vib + a_rot,
                   qm9_zpve_hartree=float(r["zpve"]),
                   qm9_U0_hartree=float(r["U0"]), qm9_G_hartree=float(r["G"]))
        out["species"][name] = rec
        print()
        print("  {:20s} {}   {}   {} 个原子".format(name, qid, r["qm9_smiles"], n_at))
        print("     频率 {} 个 (3N-6 = {}), 虚频 {} 个, 最低 {:.2f} cm^-1".format(
            len(freqs), expect, n_imag, min(freqs)))
        print("     转动常数 (GHz)  {:8.5f} {:8.5f} {:8.5f}".format(*rec["rotational_constants_GHz"]))
        print("     转动温度 (K)    {:8.5f} {:8.5f} {:8.5f}   T/Theta_max = {:.0f}".format(
            *theta, T_REF / max(theta)))
        print("     A_vib = {:+9.4f}   A_rot = {:+9.4f}   (G - E_el) = {:+9.4f} kcal/mol".format(
            a_vib, a_rot, a_vib + a_rot))
        print("     QM9 自带: zpve = {:.6f} Eh = {:.4f} kcal/mol".format(
            float(r["zpve"]), float(r["zpve"]) * HARTREE_KCAL))

    s = out["species"]
    d_vib = s[PRODUCT]["A_vib_kcal"] - s[REACTANT]["A_vib_kcal"]
    d_rot = s[PRODUCT]["A_rot_kcal"] - s[REACTANT]["A_rot_kcal"]
    d_therm = s[PRODUCT]["G_minus_E_el_kcal"] - s[REACTANT]["G_minus_E_el_kcal"]
    d_G_qm9 = (s[PRODUCT]["qm9_G_hartree"] - s[REACTANT]["qm9_G_hartree"]) * HARTREE_KCAL
    d_U0_qm9 = (s[PRODUCT]["qm9_U0_hartree"] - s[REACTANT]["qm9_U0_hartree"]) * HARTREE_KCAL

    print()
    print("=" * 92)
    print("差值   {} -> {}".format(REACTANT, PRODUCT))
    print("=" * 92)
    print("   Delta A_vib   (谐振)            = {:+9.4f} kcal/mol".format(d_vib))
    print("   Delta A_rot   (刚转子)          = {:+9.4f} kcal/mol".format(d_rot))
    print("   Delta (G - E_el)  合计          = {:+9.4f} kcal/mol   <-- 本脚本的产品".format(
        d_therm))
    print()
    print("   对照 QM9 自带的完整量 (同为 B3LYP/6-31G(2df,p)):")
    print("      Delta G  (QM9 的 G 列)       = {:+9.4f} kcal/mol".format(d_G_qm9))
    print("      Delta U0 (QM9 的 U0 列)      = {:+9.4f} kcal/mol".format(d_U0_qm9))
    print("      两者之差 = Delta(G - U0)     = {:+9.4f} kcal/mol".format(d_G_qm9 - d_U0_qm9))
    print()
    print("   注意: QM9 的 G 列含电子能, 本脚本的 (G - E_el) 不含 —— 两者不可直接比。")
    print("         能比的是: Delta G(QM9) - Delta(G-E_el)(本脚本) 应约等于 Delta E_el(B3LYP),")
    print("         即 {:+.4f} kcal/mol。".format(d_G_qm9 - d_therm))
    out["delta"] = dict(A_vib=d_vib, A_rot=d_rot, G_minus_E_el=d_therm,
                        qm9_G=d_G_qm9, qm9_U0=d_U0_qm9,
                        implied_delta_E_el_b3lyp=d_G_qm9 - d_therm)

    outdir = ROOT / "analysis"
    outdir.mkdir(exist_ok=True)
    p = outdir / "qm9_native_reference.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("落盘:", p)
    print()
    print("**这是对比样, 不是 stage 0 的结果。** 层级是 B3LYP/6-31G(2df,p), 与讲义的")
    print("MACE-OFF23-SC 不同势面; 且它是纯谐振的, 不含非谐与构象贡献 —— 而那正是")
    print("stage 0 的态密度路线要捕捉的东西。两者之差 = 非谐 + 构象 + 势面之差。")


if __name__ == "__main__":
    main()
