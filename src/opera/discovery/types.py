from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class MatrixConfig:
    """用于所有矩阵构造/求解的参数集合。"""
    epsilon: float = 1e-9


@dataclass
class DiscoveryConfig:
    # 组件数选择模式:
    # - fixed: 严格使用 k_value
    # - auto: 基于谱能量阈值自动估计
    # - capped-auto: 自动估计后截断到 k_max
    k_mode: str = "fixed"
    k_value: int = 3
    k_max: int = 8
    rank_energy_threshold: float = 0.99
    calibrate_pure_spectra_once: bool = True
    calibration_ridge: float = 1e-8
    # Gamma 逐频稀疏回归参数
    gamma_sparse_iters: int = 4
    gamma_pivot_operator: str = "D̂(c,ω)"

    # 兼容旧参数（保留字段，不用于 fixed 模式决策）
    max_components: int = 5

    sparsity_threshold: float = 1e-2
    nullspace_tol: float = 1e-8
    zero_anchor_tol: float = 1e-12
    min_component_energy_ratio: float = 0.03
    min_component_score: float = 0.2

    # 频移估计：默认关闭，开启后会基于频域相位非线性拟合 g_shift
    estimate_g_shift: bool = False
    # 可选的频移幅度约束（绝对值），用于 least-squares bounds，None 表示不约束
    g_shift_max_abs: float | None = None

    # 集中的矩阵配置
    matrix: MatrixConfig = field(default_factory=MatrixConfig)
    # 如果为 True 则使用伪逆算子管线而不是基投影
    use_inverse_operator: bool = False


@dataclass
class DiscoveryResult:
    S_real: np.ndarray
    f_response_eval: np.ndarray
    # 主输出：按算子块组织的 Xi 张量，形状 (n_blocks, J_max, K)
    Xi: np.ndarray
    operator_names: List[str]
    # 展开的算子系数矩阵（即 Gamma≈A·P^T 中的 A），形状 (J_tot, K)
    A_matrix: Optional[np.ndarray] = None
    f_response: Optional[np.ndarray] = None
    g_shift: Optional[np.ndarray] = None
    pure_spectra_complex: Optional[np.ndarray] = None
    reconstruction_error: Optional[float] = None
    # 语义更新：按组分提供 Xi 列向量（component_k -> Xi[:, k]）
    xi_by_control: Dict[str, np.ndarray] = field(default_factory=dict)
    component_scores: Optional[np.ndarray] = None
    component_energy_ratio: Optional[np.ndarray] = None
    component_nonzero_ratio: Optional[np.ndarray] = None
    quality_flags: List[str] = field(default_factory=list)
    latex_blocks: List[str] = field(default_factory=list)
    diagnostics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)
