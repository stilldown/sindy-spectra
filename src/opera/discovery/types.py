"""发现算法的配置与结果数据类型。

数学符号约定
-----------
* N  — 样本数（即控制条件数）
* M  — 波长点数
* P  — 正频率数，P = M//2 + 1
* K  — 保留的谱主成分数（"组分数"）
* J  — 算子库特征总数
* d  — 控制维度数（即控制变量个数）
* c  — 控制矩阵，形状 (N, d)，每行为一个样本的控制向量
* D̂  — 频域观测张量，形状 (N, P)，D̂ = rfft(S)
* D† — D̂ 的 Moore-Penrose 伪逆，形状 (P, N)
* A  — 投影系数矩阵；SVD 路径为 (N, K)，伪逆路径为 (N, N)
* ω  — 归一化频率轴，形状 (P,)，ω ∈ [0, 1]
* ξ  — SINDy-PI 系数向量
* Θ  — Euler 算子特征矩阵，形状 (N, J)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class MatrixConfig:
    """矩阵数值稳定性参数。

    Attributes
    ----------
    epsilon : float
        伪逆截断阈值和 log 下界，防止零值取对数。默认 1e-9。
    """
    epsilon: float = 1e-9


@dataclass
class DiscoveryConfig:
    """发现算法的全局配置。

    组分数选择
    ----------
    k_mode : {"fixed", "auto", "capped-auto"}
        "fixed"       — 严格使用 k_value，忽略能量阈值。
        "auto"        — 按 rank_energy_threshold 自动估计。
        "capped-auto" — 自动估计后截断到 k_max。
    k_value : int
        fixed 模式下的组分数 K。
    k_max : int
        auto / capped-auto 模式下的组分数上限。
    rank_energy_threshold : float
        auto 模式下的累积奇异值能量阈值（0 < thr ≤ 1），默认 0.99。

    稀疏化与数值
    ------------
    sparsity_threshold : float
        SINDy-PI 系数稀疏化阈值，低于此值视为零。
    nullspace_tol : float
        零空间 SVD 截断容差。
    zero_anchor_tol : float
        判定样本为零浓度锚点的控制变量绝对值上界。
    min_component_energy_ratio : float
        最小组分能量占比，低于此值的组分将被丢弃（当前未启用）。
    min_component_score : float
        最小组分质量评分（当前未启用）。
    matrix : MatrixConfig
        矩阵级数值参数。

    管线模式开关
    ------------
    多个标志同时为 True 时，优先级为：
    ``use_weak_form > use_direct_euler > use_inverse_operator > 默认 SVD 路径``。

    use_inverse_operator : bool
        True → 使用伪逆算子管线（``operator.construct_inverse_library``，K = N）。
        False（默认） → 使用 SVD 谱基投影管线（``pipeline_utils.construct_pure_library``）。
    use_weak_form : bool
        True → 使用真正弱形式（IBP，``operator.build_weak_form_library``，K = N），
        无需对含噪数据求导，优先级最高。
    weak_form_test_degree : int
        弱形式多项式测试函数的最高阶数，默认 2。
    use_direct_euler : bool
        True → 使用无 SVD 的直接 Euler 算子路径（``pipeline_utils.build_direct_euler_library``）。
        算子直接在全频域张量 D̂(c, ω) 上计算（含 c_i 缩放），用 W(ω)=[1,−iω] 线性拟合
        分离 f/g，输出 K=1 全局方程，不进行组分分离。
    """
    # ---------- 组分数选择 ----------
    k_mode: str = "fixed"
    k_value: int = 3
    k_max: int = 8
    rank_energy_threshold: float = 0.99

    # ---------- 稀疏化与数值 ----------
    sparsity_threshold: float = 1e-2
    nullspace_tol: float = 1e-8
    zero_anchor_tol: float = 1e-12
    min_component_energy_ratio: float = 0.03
    min_component_score: float = 0.2
    matrix: MatrixConfig = field(default_factory=MatrixConfig)

    # ---------- 管线模式开关 ----------
    use_inverse_operator: bool = False
    use_weak_form: bool = False
    weak_form_test_degree: int = 2
    use_direct_euler: bool = False

@dataclass
class DiscoveryResult:
    """发现算法的输出结果。

    核心输出
    --------
    S_real : ndarray, shape (M, K)
        K 个纯组分光谱（波长域实值），由 irfft(谱基) 得到。
    f_response_eval : ndarray, shape (N, K)
        各样本对各组分的强度响应系数 A = D̂ @ P†，
        其中 P ∈ ℂ^{K×P} 为 SVD 谱基。
    Xi : ndarray, shape (1, J, K)
        隐式约束系数张量：Xi[0, :, k] 是第 k 个组分的 SINDy-PI 系数向量 ξ_k，
        满足 Θ(c) ξ_k ≈ 0，其中 Θ ∈ ℝ^{N×J} 为算子特征矩阵。
    operator_names : list of str, length J
        与 Xi 第二维对齐的算子名称列表。
    A_matrix : ndarray, shape (J, K), optional
        Xi[0] 的别名，方便直接访问系数矩阵。
    f_response : ndarray, shape (N, K), optional
        f_response_eval 的别名。
    pure_spectra_complex : ndarray, shape (P, K), optional
        谱基在频域的复值表示 P^T。
    reconstruction_error : float, optional
        重建误差（当前固定返回 0.0，未实际计算）。

    辅助诊断
    --------
    xi_by_control : dict[str, ndarray]
        按组分名称索引的 Xi 列向量：{"component_1": ξ_1, ...}。
    quality_flags : list of str
        质量标记，"ok" 表示无异常。
    diagnostics : dict[str, float]
        数值诊断指标：k_eff, k_selected, nullspace_energy, sigma_gap_min, anchor_count。
    metadata : dict[str, object]
        算法元数据：models, equations, anchor_indices, k_source, J_tot, operator_block_ranges。
    latex_blocks : list of str
        每个组分对应的 LaTeX 方程字符串，供 GUI 渲染。
    """
    # ---------- 核心输出 ----------
    S_real: np.ndarray
    f_response_eval: np.ndarray
    Xi: np.ndarray
    operator_names: List[str]
    A_matrix: Optional[np.ndarray] = None
    f_response: Optional[np.ndarray] = None
    pure_spectra_complex: Optional[np.ndarray] = None
    reconstruction_error: Optional[float] = None

    # ---------- 辅助诊断 ----------
    xi_by_control: Dict[str, np.ndarray] = field(default_factory=dict)
    component_scores: Optional[np.ndarray] = None
    component_energy_ratio: Optional[np.ndarray] = None
    component_nonzero_ratio: Optional[np.ndarray] = None
    quality_flags: List[str] = field(default_factory=list)
    latex_blocks: List[str] = field(default_factory=list)
    diagnostics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)
