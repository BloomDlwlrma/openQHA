"""stage 0 —— 可分离项的显式断言, 与低频模的四种处理法之比较.

CALIBRATION. Explicit assertions on the separable terms, plus four treatments of a
low-frequency mode. Its output is the size of a contribution to the error bar.

回答两个问题:

  问题一  Delta G 必须含平动 / 转动 / 振动 / 电子四项. 讲义 s0-1 是否只有振动?
          -> 不是. 转动在 s0-1 第 6.6 节已算 (刚转子 + 显式外对称数);
             平动与电子简并在第 2.1 节被声称"精确相消", 但那只是散文里的**声称**.
             skills 第 2.1 条要求"必须显式断言, 不能只在注释里说" ——
             本脚本 A 部分把这两条声称变成会失败的断言.

  问题二  低频模的权重约为高频模的 4 倍, 低频处怎么解?
          -> B / C / D 部分: 把一个甲基转子的自由能用四种模型各算一遍
             (简谐振子 / 自由转子 / 一维受阻转子精确解 / Grimme 准刚转子-谐振子),
             并给出两个极限检验. 差值即"低频处理"对误差棒的贡献.

用法:
    python scripts/calibration/s0_lowfreq_and_separable_terms.py

产物:
    analysis/lowfreq_and_separable_terms.json
    analysis/lowfreq_hindered_rotor.png

参数分类 (skills 第 1.4 条) 见文件末尾的 PARAMETER_TABLE.
本脚本不调用任何势函数, 因此不受 Egret-1 未安装的阻塞.
"""
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")               # 本脚本只落盘, 不显示; 讲义里则不得设 Agg
import matplotlib.pyplot as plt

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
KB_KCAL = 1.987204259e-3            # 玻尔兹曼常数, kcal/(mol*K)
HC_KCAL = 2.85914308e-3             # h*c, kcal/mol per cm^-1  (1 cm^-1 的能量)
T_REF = 298.15                      # K
KT = KB_KCAL * T_REF                # kcal/mol
H_SI = 6.62607015e-34
KB_SI = 1.380649e-23
NA = 6.02214076e23
AMU_KG = 1.66053906660e-27
P_STD = 1.0e5                       # Pa, 标准态 1 bar
J_TO_KCAL = 1.0 / 4184.0
# I 以 amu*A^2 计时的转动常数 B[cm^-1] = ROT_CONST_AMU_A2 / I
ROT_CONST_AMU_A2 = 16.857629046     # = h / (8 pi^2 c), 换算到 amu*A^2 与 cm^-1

# ---------------------------------------------------------------- 体系定义
EDGE = "C2H5O1N1_19_36"
SPECIES = {"acetamide": "CC(N)=O", "N-methylformamide": "CNC=O"}

# 外对称数: 显式声明, 绝不自动推导 (skills 第 2.2 条)
SYMMETRY_NUMBER = {"acetamide": 1, "N-methylformamide": 1}

# 电子简并度 g0 = 自旋多重度. 与对称数同一原则: 显式声明, 缺失即拒绝运行.
# 两者都是闭壳层单重态基态, g0 = 1, 故 -kT ln g0 = 0.
ELECTRONIC_DEGENERACY = {"acetamide": 1, "N-methylformamide": 1}

# 甲基转子的内对称数 (顶部自身的三重轴)
SIGMA_INTERNAL_METHYL = 3


# ================================================================================
# A 部分  可分离项 —— 把散文里的"精确相消"变成会失败的断言
# ================================================================================

def sackur_tetrode_kcal(mass_amu, temperature_K=T_REF, pressure_Pa=P_STD,
                        kind="gibbs"):
    """理想气体的平动自由能 (每摩尔, kcal/mol), 标准态体积 v = kT/p.

        Lambda   = h / sqrt(2 pi m kB T)                     热德布罗意波长
        a_trans  = -kT [ ln(kT / (p Lambda^3)) + 1 ]         亥姆霍兹, 含 1/N! 的 Stirling 项
        g_trans  = a_trans + kT = -kT ln(kT / (p Lambda^3))   吉布斯

    **两者相差恰好一个 kT, 那就是 pV 项本身**, 也就是 Delta n * RT.
    本项目 Delta n = 0, 故在差值里两种口径完全等价; 但**单个物种的绝对值差 0.5925
    kcal/mol**, 所以必须写明用的是哪一个 —— 本函数早先只实现了吉布斯口径却叫作
    "亥姆霍兹", 是一处命名错误, 已更正.

    只依赖质量与温度/压强 —— 与几何、成键方式、势函数全部无关.
    """
    if kind not in ("gibbs", "helmholtz"):
        raise ValueError("kind 必须是 'gibbs' 或 'helmholtz', 收到 {!r}".format(kind))
    m = mass_amu * AMU_KG
    lam = H_SI / math.sqrt(2.0 * math.pi * m * KB_SI * temperature_K)
    v = KB_SI * temperature_K / pressure_Pa          # 每分子体积, m^3
    ln_q_over_n = math.log(v / lam ** 3)
    stirling = 0.0 if kind == "gibbs" else 1.0
    a_j = -KB_SI * temperature_K * (ln_q_over_n + stirling)
    return a_j * NA * J_TO_KCAL


def molecular_mass_amu(smiles):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    return Descriptors.MolWt(Chem.AddHs(Chem.MolFromSmiles(smiles)))


def part_a():
    from rdkit import Chem
    out = {}

    # --- 缺失即拒绝运行, 不给默认值 -------------------------------------------
    for name in SPECIES:
        if name not in SYMMETRY_NUMBER:
            raise KeyError("外对称数未声明: {} —— 拒绝运行".format(name))
        if name not in ELECTRONIC_DEGENERACY:
            raise KeyError("电子简并度未声明: {} —— 拒绝运行".format(name))

    masses = {n: molecular_mass_amu(s) for n, s in SPECIES.items()}
    formulas = {n: Chem.rdMolDescriptors.CalcMolFormula(
        Chem.AddHs(Chem.MolFromSmiles(s))) for n, s in SPECIES.items()}
    g_trans = {n: sackur_tetrode_kcal(m, kind="gibbs") for n, m in masses.items()}
    a_trans = {n: sackur_tetrode_kcal(m, kind="helmholtz") for n, m in masses.items()}
    a_elec = {n: -KT * math.log(ELECTRONIC_DEGENERACY[n]) for n in SPECIES}

    names = list(SPECIES)
    d_trans = a_trans[names[1]] - a_trans[names[0]]
    d_elec = a_elec[names[1]] - a_elec[names[0]]

    print("=" * 88)
    print("A 部分  四项里哪些精确相消 —— 断言, 不是声称")
    print("=" * 88)
    print("边: {}".format(EDGE))
    for n in names:
        print("  {:20s} 式 {:10s} M = {:10.6f} amu   g0 = {:d}   sigma_ext = {:d}".format(
            n, formulas[n], masses[n], ELECTRONIC_DEGENERACY[n], SYMMETRY_NUMBER[n]))
    print()
    print("  1) 平动 (Sackur-Tetrode, 298.15 K, 1 bar)")
    print("       {:22s} {:>16s} {:>16s} {:>10s}".format(
        "物种", "a_trans(亥姆霍兹)", "g_trans(吉布斯)", "g - a"))
    for n in names:
        print("       {:22s} {:16.8f} {:16.8f} {:10.4f}".format(
            n, a_trans[n], g_trans[n], g_trans[n] - a_trans[n]))
    print("       g - a = kT = {:.4f} kcal/mol —— 这就是 pV 项, 即 Delta n * RT".format(KT))
    print("       Delta a_trans          = {:14.8e} kcal/mol".format(d_trans))
    assert g_trans[names[1]] - g_trans[names[0]] == 0.0, "Delta g_trans 非零"

    # 断言一: 同分子式 => 质量逐位相同 => 平动项按 IEEE-754 逐位相消
    assert formulas[names[0]] == formulas[names[1]], \
        "分子式不同, 平动项不再相消 —— 本脚本的相消证明作废"
    assert masses[names[0]] == masses[names[1]], "质量不逐位相同"
    assert d_trans == 0.0, "Delta A_trans 非零: {!r}".format(d_trans)
    print("       断言通过: 分子式相同 -> 质量逐位相同 -> Delta A_trans 恒等于 0")
    print("       故不必另写脚本算 F_trans —— 它在 Delta 里不是很小, 是恒等于 0.")
    print()
    print("  2) 电子 (q_el = g0 * exp(-beta E_el); E_el 单列, 此处只余 -kT ln g0)")
    for n in names:
        print("       A_elec({:20s}) = {:14.8f} kcal/mol   (g0 = {:d})".format(
            n, a_elec[n], ELECTRONIC_DEGENERACY[n]))
    assert d_elec == 0.0, "Delta A_elec 非零: {!r}".format(d_elec)
    print("       断言通过: 两者皆闭壳层单重态 -> g0 = 1 -> 该项恒等于 0")
    print()
    print("  3) 转动: 不相消, 已在讲义 s0-1 第 6.6 节按刚转子 + 显式 sigma 算出")
    print("  4) 振动/构象: 不相消, 正是 stage 0 要算的东西 (第 6.5 节, 态密度加权)")
    print()

    # --- 这条断言应当失败的例子 (skills 第 2.4(a) 条) ------------------------
    #
    # 本检验第一次写时用了一个拍出来的阈值 |Delta| > 0.1 kcal/mol, 它当场报警.
    # 按 skills 第 2.4(b) 条追查: 报的不是真错, 是**判据自己的口径问题** ——
    # 平动项对质量的依赖只有 -(3/2) kT ln(m'/m), 1 amu 仅值 0.0149 kcal/mol.
    # 改成与解析式对拍: 反例必须 (i) 非零, 因为上面的断言是逐位相等; 且
    # (ii) 精确等于 -(3/2) kT ln(m'/m). 这是推导判据, 不是拍的阈值.
    fake = sackur_tetrode_kcal(masses[names[0]] + 1.0, kind="helmholtz")
    delta_fake = fake - a_trans[names[0]]
    analytic = -1.5 * KT * math.log((masses[names[0]] + 1.0) / masses[names[0]])
    print("  反例检验 (证明上面的断言不是空的):")
    print("     若两侧质量差 1 amu, Delta A_trans = {:+.8f} kcal/mol".format(delta_fake))
    print("     解析式 -(3/2) kT ln(m'/m)        = {:+.8f} kcal/mol".format(analytic))
    print("     两者之差                          = {:+.3e} kcal/mol".format(
        delta_fake - analytic))
    assert delta_fake != 0.0, "反例没能让逐位相等的断言失败 -> 该断言是空的"
    assert abs(delta_fake - analytic) < 1e-9, "Sackur-Tetrode 实现与解析式不符"
    print("     通过: 断言有分辨力 (非零), 且实现与解析式一致.")
    print("     注意量级: 平动项对质量极不敏感, 1 amu 只有 {:.4f} kcal/mol,".format(
        abs(delta_fake)))
    print("               远小于 1.0 kcal/mol 的目标 —— 但断言仍取逐位相等, 因为在")
    print("               同分子式下它本来就该是 0, 任何非零都说明分子式搞错了.")
    print()

    out["formulas"] = formulas
    out["mass_amu"] = masses
    out["a_trans_helmholtz_kcal_per_mol"] = a_trans
    out["g_trans_gibbs_kcal_per_mol"] = g_trans
    out["A_elec_kcal_per_mol"] = a_elec
    out["delta_A_trans_kcal_per_mol"] = d_trans
    out["delta_A_elec_kcal_per_mol"] = d_elec
    out["counterexample_delta_A_trans_mass_plus_1amu"] = delta_fake
    return out


# ================================================================================
# B 部分  甲基转子的约化转动惯量 —— Pitzer-Gwinn I^(2,3) 方案
# ================================================================================

def embed(smiles, seed=0xC0FFEE):
    """与讲义 s0-1 同一配方: ETKDGv3 + MMFF94. 只用来取几何, 不产出能量."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    if AllChem.EmbedMolecule(mol, p) != 0:
        raise RuntimeError("ETKDGv3 嵌入失败: {}".format(smiles))
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    conf = mol.GetConformer()
    xyz = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    m = np.array([a.GetMass() for a in mol.GetAtoms()])
    return mol, xyz, m


def find_methyl_tops(mol):
    """返回 [(top_carbon_idx, anchor_idx, [top atom idx...]), ...]"""
    from rdkit import Chem
    patt = Chem.MolFromSmarts("[CX4H3]-[!#1]")
    tops = []
    for c_idx, anchor in mol.GetSubstructMatches(patt):
        atoms = [c_idx] + [nb.GetIdx() for nb in mol.GetAtomWithIdx(c_idx).GetNeighbors()
                           if nb.GetIdx() != anchor]
        tops.append((c_idx, anchor, atoms))
    return tops


def reduced_moment_I23(xyz, masses, top_atoms, axis_a, axis_b):
    """Pitzer-Gwinn 的 I^(2,3) 约化转动惯量, amu*A^2.

        I_red = I_top * ( 1 - sum_g  lambda_g^2 * I_top / I_g )

    I_top    顶部绕其自身转轴的转动惯量
    lambda_g 转轴相对整分子三个主惯量轴的方向余弦
    I_g      整分子的三个主转动惯量

    出处: K. S. Pitzer, W. D. Gwinn, J. Chem. Phys. 10 (1942) 428, doi:10.1063/1.1723744;
          方案编号与命名见 A. L. L. East, L. Radom, J. Chem. Phys. 106 (1997) 6655,
          doi:10.1063/1.473958 (该文把 Pitzer 的诸方案统一记作 I^(m,n)).
    """
    axis = xyz[axis_b] - xyz[axis_a]
    axis = axis / np.linalg.norm(axis)

    # I_top: 顶部各原子到转轴的垂距平方乘质量
    d = xyz[top_atoms] - xyz[axis_a]
    perp = d - np.outer(d @ axis, axis)
    i_top = float((masses[top_atoms] * (perp ** 2).sum(1)).sum())

    # 整分子主惯量与主轴
    com = (masses[:, None] * xyz).sum(0) / masses.sum()
    r = xyz - com
    r2 = (r ** 2).sum(1)
    inertia = (masses[:, None, None] * (r2[:, None, None] * np.eye(3)
                                        - r[:, :, None] * r[:, None, :])).sum(0)
    i_princ, axes = np.linalg.eigh(inertia)
    lam = axes.T @ axis                                 # 方向余弦
    i_red = i_top * (1.0 - float((lam ** 2 * i_top / i_princ).sum()))
    return i_red, i_top, i_princ, lam


def part_b():
    print("=" * 88)
    print("B 部分  甲基转子的约化转动惯量 (Pitzer-Gwinn I^(2,3))")
    print("=" * 88)
    out = {}
    for name, smi in SPECIES.items():
        mol, xyz, m = embed(smi)
        tops = find_methyl_tops(mol)
        rec = []
        print("  {:20s} 找到 {} 个甲基顶".format(name, len(tops)))
        for c_idx, anchor, atoms in tops:
            i_red, i_top, i_princ, lam = reduced_moment_I23(xyz, m, atoms, c_idx, anchor)
            b_cm = ROT_CONST_AMU_A2 / i_red
            sym = mol.GetAtomWithIdx(anchor).GetSymbol()
            print("     C{:d}-{}{:d}   I_top = {:6.3f}   I_red = {:6.3f} amu A^2"
                  "   -> B = {:6.3f} cm^-1".format(c_idx, sym, anchor, i_top, i_red, b_cm))
            rec.append(dict(top_carbon=c_idx, anchor=anchor, anchor_element=sym,
                            I_top_amu_A2=i_top, I_red_amu_A2=i_red,
                            B_cm_inv=b_cm,
                            I_principal_amu_A2=[float(v) for v in i_princ]))
        out[name] = rec
    print()
    return out


# ================================================================================
# C 部分  低频模的四种处理法
# ================================================================================

def a_harmonic_kcal(nu_cm):
    """量子谐振子亥姆霍兹自由能 (含零点能), 零点取势阱底. kcal/mol."""
    x = HC_KCAL * np.asarray(nu_cm, float) / KT
    return KT * (0.5 * x + np.log1p(-np.exp(-x)))


def s_harmonic_kcal_per_k(nu_cm):
    x = HC_KCAL * np.asarray(nu_cm, float) / KT
    return KB_KCAL * (x / np.expm1(x) - np.log1p(-np.exp(-x)))


def q_free_rotor(b_cm, sigma_int, temperature_K=T_REF):
    """一维自由转子配分函数 q = (1/sigma) sqrt(pi kT / (h c B))."""
    kt_cm = KB_KCAL * temperature_K / HC_KCAL
    return math.sqrt(math.pi * kt_cm / b_cm) / sigma_int


def a_free_rotor_kcal(b_cm, sigma_int, temperature_K=T_REF):
    return -KB_KCAL * temperature_K * math.log(q_free_rotor(b_cm, sigma_int, temperature_K))


def s_free_rotor_kcal_per_k(b_cm, sigma_int, temperature_K=T_REF):
    """一维自由转子熵: S = k [ ln q + 1/2 ]  (因 E = kT/2)."""
    return KB_KCAL * (math.log(q_free_rotor(b_cm, sigma_int, temperature_K)) + 0.5)


def mathieu_levels(b_cm, v_n_cm, n_fold=3, m_max=200):
    """一维受阻转子 H = B (-i d/dphi)^2 + (V_n/2)(1 - cos n phi) 的本征值, cm^-1.

    在自由转子基 {exp(i m phi)} 上对角化. 势的零点取在阱底 (phi = 0), 故
        <m|V|m>       = V_n / 2
        <m|V|m ± n>   = -V_n / 4
    这是标准的 Mathieu 方程; 见 Pitzer & Gwinn 1942 (doi:10.1063/1.1723744).
    """
    m = np.arange(-m_max, m_max + 1)
    h = np.diag(b_cm * m.astype(float) ** 2 + v_n_cm / 2.0)
    off = -v_n_cm / 4.0
    idx = np.arange(len(m) - n_fold)
    h[idx, idx + n_fold] = off
    h[idx + n_fold, idx] = off
    return np.linalg.eigvalsh(h)


def a_hindered_rotor_kcal(b_cm, v_n_cm, sigma_int=SIGMA_INTERNAL_METHYL, n_fold=3,
                          m_max=200, temperature_K=T_REF):
    """受阻转子精确解的自由能, 零点取势阱底 (与谐振子一致, 含零点能)."""
    e = mathieu_levels(b_cm, v_n_cm, n_fold, m_max)
    kt_cm = KB_KCAL * temperature_K / HC_KCAL
    q = float(np.exp(-e / kt_cm).sum()) / sigma_int
    return -KB_KCAL * temperature_K * math.log(q), q, e


def torsional_harmonic_frequency_cm(b_cm, v_n_cm, n_fold=3):
    """阱底简谐近似的扭转频率:  nu = n * sqrt(V_n * B).

    由 V = (V_n/2)(1 - cos n phi) 在 phi=0 展开得 k = n^2 V_n / 2, omega = sqrt(k/I);
    代入 B = h/(8 pi^2 c I) 化简即得. 两边都以 cm^-1 计.
    """
    return n_fold * math.sqrt(v_n_cm * b_cm)


def a_grimme_qrrho_kcal(nu_cm, b_cm_of_mode, temperature_K=T_REF, nu0_cm=100.0):
    """Grimme 准刚转子-谐振子: 只对**熵**做阻尼插值, 自由能 = E_harm - T*S_qRRHO.

    w(nu) = 1 / (1 + (nu0/nu)^4)                       Head-Gordon 型阻尼
    S_qRRHO = w * S_HO(nu) + (1-w) * S_FR
    出处: S. Grimme, Chem. Eur. J. 18 (2012) 9955, doi:10.1002/chem.201200497.

    实现说明: 原方案用 mu' = mu*B_av/(mu+B_av) 的"平均转动惯量"防止 mu -> inf 时熵发散;
    这里的低频模就是一个真实的转子, 其 B 已由 B 部分算出, 故直接用它 ——
    这比原方案更贴近本体系, 但**属于可修改约定**, 改了会移动低频端的熵.
    """
    nu = float(nu_cm)
    w = 1.0 / (1.0 + (nu0_cm / nu) ** 4)
    s_ho = float(s_harmonic_kcal_per_k(nu))
    s_fr = s_free_rotor_kcal_per_k(b_cm_of_mode, SIGMA_INTERNAL_METHYL, temperature_K)
    s = w * s_ho + (1.0 - w) * s_fr
    x = HC_KCAL * nu / (KB_KCAL * temperature_K)
    e = KB_KCAL * temperature_K * (0.5 * x + x / math.expm1(x))   # 内能不插值
    return e - temperature_K * s, w


def part_c(b_cm):
    print("=" * 88)
    print("C 部分  一个甲基转子的自由能: 四种模型, B = {:.4f} cm^-1".format(b_cm))
    print("=" * 88)
    print("  零点一律取在扭转势阱底; 所有模型都含零点能; sigma_int = {}".format(
        SIGMA_INTERNAL_METHYL))
    print("  kT = {:.4f} kcal/mol = {:.1f} cm^-1".format(KT, KT / HC_KCAL))
    print()

    # ---- 极限检验一: V3 -> 0 时受阻转子必须回到自由转子 ---------------------
    a_fr = a_free_rotor_kcal(b_cm, SIGMA_INTERNAL_METHYL)
    a_hr0, q0, _ = a_hindered_rotor_kcal(b_cm, 0.0)
    print("  极限检验 1  V3 -> 0:")
    print("     A_free_rotor            = {:+10.6f} kcal/mol".format(a_fr))
    print("     A_hindered(V3 = 0)      = {:+10.6f} kcal/mol".format(a_hr0))
    print("     差                      = {:+10.3e} kcal/mol".format(a_hr0 - a_fr))
    assert abs(a_hr0 - a_fr) < 5e-3, "V3=0 时未回到自由转子"
    print("     通过 (自由转子解析式是连续化近似, 残差来自离散能级求和)")
    print()

    # ---- 极限检验二: V3 -> 大 时受阻转子必须回到谐振子 ----------------------
    v_big = 8000.0                                     # cm^-1, 约 22.9 kcal/mol
    nu_big = torsional_harmonic_frequency_cm(b_cm, v_big)
    a_hr_big, _, _ = a_hindered_rotor_kcal(b_cm, v_big)
    a_ho_big = float(a_harmonic_kcal(nu_big))
    print("  极限检验 2  V3 = {:.0f} cm^-1 ({:.1f} kcal/mol), nu_harm = {:.1f} cm^-1:".format(
        v_big, v_big * HC_KCAL, nu_big))
    print("     A_hindered              = {:+10.6f} kcal/mol".format(a_hr_big))
    print("     A_harmonic(nu_harm)     = {:+10.6f} kcal/mol".format(a_ho_big))
    print("     差                      = {:+10.6f} kcal/mol".format(a_hr_big - a_ho_big))
    assert abs(a_hr_big - a_ho_big) < 0.05, "高势垒极限未回到谐振子"
    print("     通过 (残差 = 余弦阱相对抛物阱的非谐性, 本来就该有)")
    print()

    # ---- 扫势垒 -------------------------------------------------------------
    v3_kcal = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    rows = []
    print("  {:>8s} {:>9s} {:>11s} {:>11s} {:>11s} {:>11s} {:>10s}".format(
        "V3", "nu_harm", "A_harmonic", "A_free", "A_qRRHO", "A_hindered", "HO-HR"))
    print("  {:>8s} {:>9s} {:>11s} {:>11s} {:>11s} {:>11s} {:>10s}".format(
        "kcal/mol", "cm^-1", "kcal/mol", "kcal/mol", "kcal/mol", "kcal/mol", "kcal/mol"))
    print("  " + "-" * 76)
    for v in v3_kcal:
        v_cm = v / HC_KCAL
        nu = torsional_harmonic_frequency_cm(b_cm, v_cm) if v > 0 else 0.0
        a_hr, _, _ = a_hindered_rotor_kcal(b_cm, v_cm)
        if v > 0:
            a_ho = float(a_harmonic_kcal(nu))
            a_qr, _ = a_grimme_qrrho_kcal(nu, b_cm)
            s_ho = "{:+11.4f}".format(a_ho)
            s_qr = "{:+11.4f}".format(a_qr)
            gap = "{:+10.4f}".format(a_ho - a_hr)
        else:
            a_ho = a_qr = None
            s_ho = "{:>11s}".format("-inf")
            s_qr = "{:>11s}".format("-inf")
            gap = "{:>10s}".format("-inf")
        print("  {:8.2f} {:9.1f} {} {:+11.4f} {} {:+11.4f} {}".format(
            v, nu, s_ho, a_fr, s_qr, a_hr, gap))
        rows.append(dict(V3_kcal_per_mol=float(v), nu_harmonic_cm_inv=float(nu),
                         A_harmonic=a_ho, A_free_rotor=float(a_fr),
                         A_qRRHO=a_qr, A_hindered_rotor=float(a_hr)))
    # ---- 误差符号的过零点: 实测, 不靠眼看 -----------------------------------
    def gap(v_cm):
        return float(a_harmonic_kcal(torsional_harmonic_frequency_cm(b_cm, v_cm))) \
            - a_hindered_rotor_kcal(b_cm, v_cm)[0]

    lo, hi = 1.0, 4000.0                              # cm^-1, gap(lo) < 0 < gap(hi)
    assert gap(lo) < 0.0 < gap(hi), "过零点不在括号内, 下面的二分无效"
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if gap(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    v_zero = 0.5 * (lo + hi)
    nu_zero = torsional_harmonic_frequency_cm(b_cm, v_zero)

    print()
    print("  读表 —— 简谐近似的误差有两个截然不同的区间, 分界点实测如下:")
    print("     误差过零点  V3 = {:.4f} kcal/mol,  nu_harm = {:.1f} cm^-1".format(
        v_zero * HC_KCAL, nu_zero))
    print("     (i)  nu < {:.0f} cm^-1: 简谐**低估** A —— 因为 S_HO ~ -k ln(beta h c nu)".format(
        nu_zero))
    print("          在 nu -> 0 时发散到 +inf, A_HO -> -inf. 转子的熵其实是有限的.")
    print("     (ii) nu > {:.0f} cm^-1: 简谐**高估** A, 幅度先增后减, 峰值约 0.10 kcal/mol.".format(
        nu_zero))
    print("     两个区间的物理含义不同: (i) 是发散, (ii) 是余弦阱的非谐性.")
    print("     只有 (i) 会威胁 1.0 kcal/mol 的目标 —— 而甲基转子正好落在 (i) 里.")
    print()
    return rows, dict(sign_change_V3_kcal=float(v_zero * HC_KCAL),
                      sign_change_nu_cm_inv=float(nu_zero))


def part_c_figure(b_cm, path):
    v_cm = np.logspace(math.log10(3.0), math.log10(8000.0), 220)     # cm^-1
    a_hr = np.array([a_hindered_rotor_kcal(b_cm, v)[0] for v in v_cm])
    nu = np.array([torsional_harmonic_frequency_cm(b_cm, v) for v in v_cm])
    a_ho = a_harmonic_kcal(nu)
    a_qr = np.array([a_grimme_qrrho_kcal(n, b_cm)[0] for n in nu])
    a_fr = a_free_rotor_kcal(b_cm, SIGMA_INTERNAL_METHYL)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    ax.plot(nu, a_hr, lw=2.2, color="#1f3b73", label="1D hindered rotor (exact, Mathieu)")
    ax.plot(nu, a_ho, lw=1.6, ls="--", color="#b03030", label="harmonic oscillator")
    ax.plot(nu, a_qr, lw=1.6, ls="-.", color="#2e8b57", label="Grimme quasi-RRHO")
    ax.axhline(a_fr, lw=1.2, ls=":", color="#888888", label="1D free rotor (limit)")
    ax.set_xscale("log")
    ax.set_xlabel("harmonic torsional frequency  $\\tilde\\nu$  /  cm$^{-1}$")
    ax.set_ylabel("$A$ of one methyl torsion  /  kcal mol$^{-1}$")
    ax.set_title("(a) One methyl torsion, four models   (298.15 K, $B$ = "
                 "{:.2f} cm$^{{-1}}$)".format(b_cm), fontsize=10)
    ax.grid(alpha=.25)
    ax.legend(frameon=False, fontsize=8)

    ax2.plot(nu, a_ho - a_hr, lw=2.2, color="#b03030", label="harmonic $-$ exact")
    ax2.plot(nu, a_qr - a_hr, lw=2.0, color="#2e8b57", label="quasi-RRHO $-$ exact")
    ax2.axhline(0, lw=0.8, color="k")
    ax2.axhline(1.0, lw=1.0, ls=":", color="#444444")
    ax2.axhline(-1.0, lw=1.0, ls=":", color="#444444")
    ax2.set_xscale("log")
    ax2.set_xlabel("harmonic torsional frequency  $\\tilde\\nu$  /  cm$^{-1}$")
    ax2.set_ylabel("error in $A$  /  kcal mol$^{-1}$")
    ax2.set_title("(b) Error of each approximation; dotted lines = the 1 kcal/mol target",
                  fontsize=10)
    ax2.grid(alpha=.25)
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"harmonic_minus_exact_max_kcal": float(np.abs(a_ho - a_hr).max()),
            "qrrho_minus_exact_max_kcal": float(np.abs(a_qr - a_hr).max())}


# ================================================================================
# D 部分  低频处理对本条边的净影响
# ================================================================================

def part_d(tops):
    print("=" * 88)
    print("D 部分  低频处理法之差在本条边上不相消")
    print("=" * 88)
    names = list(SPECIES)
    print("  {:20s} {:>8s} {:>14s}".format("物种", "甲基数", "B (cm^-1)"))
    for n in names:
        bs = ", ".join("{:.3f}".format(t["B_cm_inv"]) for t in tops[n])
        print("  {:20s} {:>8d} {:>14s}".format(n, len(tops[n]), bs if bs else "-"))
    print()
    print("  两侧各有一个甲基转子, 但锚定原子不同 ({} vs {}), 约化转动惯量因此不同,".format(
        tops[names[0]][0]["anchor_element"], tops[names[1]][0]["anchor_element"]))
    print("  势垒也不同. 于是 A_rotor 之差不为零, 模型选择的误差只**部分**相消.")
    print()
    print("  下表: 若两侧真实势垒分别为 (V_A, V_B), 用简谐近似代替精确受阻转子,")
    print("        在 Delta A 上留下多大的残差 (kcal/mol).")
    print()
    grid = [0.05, 0.5, 1.0, 2.0, 3.0]
    b_a = tops[names[0]][0]["B_cm_inv"]
    b_b = tops[names[1]][0]["B_cm_inv"]
    print("       V_B \\ V_A " + "".join("{:>9.2f}".format(v) for v in grid))
    cells = {}
    for vb in grid:
        row = []
        for va in grid:
            ea = float(a_harmonic_kcal(torsional_harmonic_frequency_cm(b_a, va / HC_KCAL))) \
                - a_hindered_rotor_kcal(b_a, va / HC_KCAL)[0]
            eb = float(a_harmonic_kcal(torsional_harmonic_frequency_cm(b_b, vb / HC_KCAL))) \
                - a_hindered_rotor_kcal(b_b, vb / HC_KCAL)[0]
            row.append(eb - ea)
        cells["V_B={:.2f}".format(vb)] = {"V_A={:.2f}".format(va): float(r)
                                          for va, r in zip(grid, row)}
        print("       {:9.2f} ".format(vb) + "".join("{:+9.4f}".format(r) for r in row))
    print()
    worst = max(abs(v) for r in cells.values() for v in r.values())
    print("  对角线 (两侧势垒相同) 上残差接近相消; 离对角线越远残差越大.")
    print("  最坏格点 |残差| = {:.4f} kcal/mol.".format(worst))
    print("  结论: 低频处理法必须在两侧用同一个模型, 且该模型的残差要进误差棒.")
    print()
    return {"B_cm_inv": {names[0]: b_a, names[1]: b_b},
            "harmonic_minus_hindered_residual_grid_kcal": cells,
            "worst_grid_residual_kcal": worst}


# ================================================================================
PARAMETER_TABLE = [
    dict(name="T_REF", value=298.15, unit="K", classification="可修改约定",
         note="参考温度; 改了它整张表都变, 但两侧同改, Delta 的变化是物理的"),
    dict(name="P_STD", value=1.0e5, unit="Pa", classification="文献值",
         note="IUPAC 标准态 1 bar; 只进平动项, 而平动项在本条边上恒等于 0"),
    dict(name="SIGMA_INTERNAL_METHYL", value=3, unit="1", classification="推导判据",
         note="甲基顶自身的三重轴; 与外对称数分开计, 见 skills 第 2.2 条"),
    dict(name="ELECTRONIC_DEGENERACY", value="显式声明", unit="1",
         classification="可修改约定",
         note="自旋多重度. 缺失即拒绝运行, 不给默认值 —— 与外对称数同一原则"),
    dict(name="m_max", value=200, unit="1", classification="数值容差",
         note="Mathieu 矩阵的自由转子基截断; B*200^2 远大于 kT, 配分函数已收敛"),
    dict(name="nu0_cm", value=100.0, unit="cm^-1", classification="文献值",
         note="Grimme 阻尼函数的转折频率; Chem. Eur. J. 18 (2012) 9955"),
    dict(name="v_big", value=8000.0, unit="cm^-1", classification="推导判据",
         note="高势垒极限检验点; 需 V3 远大于 kT (207 cm^-1) 才谈得上回到谐振子"),
    dict(name="seed", value="0xC0FFEE", unit="1", classification="资源预算",
         note="ETKDGv3 随机种子; 只影响约化转动惯量的第三位小数, 不得为让结果好看而挑选"),
]


def main():
    root = _repo_root()
    outdir = root / "analysis"
    outdir.mkdir(exist_ok=True)

    a = part_a()
    tops = part_b()
    b_cm = tops["acetamide"][0]["B_cm_inv"]
    rows, signchange = part_c(b_cm)
    figpath = outdir / "lowfreq_hindered_rotor.png"
    figstat = part_c_figure(b_cm, figpath)
    print("图已落盘: {}".format(figpath))
    print("  |A_harmonic - A_exact| 在扫描区间上的最大值 = {:.4f} kcal/mol".format(
        figstat["harmonic_minus_exact_max_kcal"]))
    print("  |A_qRRHO   - A_exact| 在扫描区间上的最大值 = {:.4f} kcal/mol".format(
        figstat["qrrho_minus_exact_max_kcal"]))
    print()
    d = part_d(tops)

    payload = dict(edge=EDGE, temperature_K=T_REF,
                   species=SPECIES, symmetry_number=SYMMETRY_NUMBER,
                   electronic_degeneracy=ELECTRONIC_DEGENERACY,
                   part_a_separable_terms=a,
                   part_b_reduced_moments=tops,
                   part_c_rotor_models=rows,
                   part_c_sign_change=signchange,
                   part_c_figure_summary=figstat,
                   part_d_residual_on_this_edge=d,
                   parameter_table=PARAMETER_TABLE,
                   geometry_source="RDKit ETKDGv3 + MMFF94, seed 0xC0FFEE —— "
                                   "只用于约化转动惯量, 不产出任何能量",
                   references=[
                       "K. S. Pitzer, W. D. Gwinn, J. Chem. Phys. 10 (1942) 428, "
                       "doi:10.1063/1.1723744",
                       "A. L. L. East, L. Radom, J. Chem. Phys. 106 (1997) 6655, "
                       "doi:10.1063/1.473958",
                       "P. Y. Ayala, H. B. Schlegel, J. Chem. Phys. 108 (1998) 2314, "
                       "doi:10.1063/1.475616",
                       "S. Grimme, Chem. Eur. J. 18 (2012) 9955, "
                       "doi:10.1002/chem.201200497",
                       "S.-T. Lin, M. Blanco, W. A. Goddard III, J. Chem. Phys. 119 (2003) "
                       "11792, doi:10.1063/1.1624057",
                       "S.-T. Lin, P. K. Maiti, W. A. Goddard III, J. Phys. Chem. B 114 "
                       "(2010) 8191, doi:10.1021/jp103120q",
                   ])
    jpath = outdir / "lowfreq_and_separable_terms.json"
    jpath.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("落盘: {}".format(jpath))


if __name__ == "__main__":
    main()
