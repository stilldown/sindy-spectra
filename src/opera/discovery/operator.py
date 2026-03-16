"""备选算子库：伪逆路径与真弱形式（IBP）路径。

两种算子路径
-----------
本模块提供两条与 :mod:`opera.discovery.pipeline_utils` 不同的算子构造路径：

**路径 A — 伪逆算子（construct_inverse_library）**

不做 SVD 谱基投影，直接计算::

    L_j^{inv}(ω) = diag(D† · ∂D/∂c_j)   ∈ ℂ^P

其中 D† = pinv(D̂) ∈ ℂ^{P×N} 是 Moore-Penrose 伪逆。
取对角线相当于对每个频率 ω 独立估计算子标量值。
截断到前 k_eff 个频率以控制维度。

**路径 B — 真弱形式（build_weak_form_library）**

无需 ∂D̂/∂c_j（避免对含噪数据求导）。利用分部积分（IBP）将
导数从 D̂ 转移到光滑测试函数 ψ_m(c)::

    ⟨c_i ∂_{c_i} ln D, ψ_m⟩(ω)
        = -Σ_n (∂_{c_i}ψ_m(c_n)·c_i(n) + ψ_m(c_n)) · ln D̂(c_n, ω)

测试函数 ψ_m 取多项式基（最高 ``test_func_degree`` 阶），梯度解析计算。
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict

from .types import DiscoveryConfig


def construct_inverse_library(
    d_hat: np.ndarray,
    d_d_c: np.ndarray,
    d2_d_c: np.ndarray,
    omega: np.ndarray,
    factors: np.ndarray,
    config: DiscoveryConfig,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    r"""伪逆算子库（不做 SVD 谱基投影）。

    对每个频率 ω 独立计算::

        L_j^{inv}(ω) = diag(D† · ∂D̂/∂c_j)            （一阶）
        Ξ_{ij}^{inv}(ω) = diag(D† · ∂²D̂/(∂c_i∂c_j))  （二阶）

    其中 D† = pinv(D̂) ∈ ℂ^{P×N} 是 Moore-Penrose 伪逆。
    取对角线 → 截断到前 k_eff 个频率 → 在 N 个样本上广播为 (N, k_eff) 形状。

    同时以 SVD 谱基作为输出的 ``basis``/``A``/``omega_means``
    （仅用于下游重建，不影响算子库的计算逻辑）。

    Returns
    -------
    library        : dict[str, ndarray(N, k_eff)]
    basis          : ndarray (k_eff, P)   SVD 谱基（占位用）
    A              : ndarray (N, k_eff)   投影系数
    omega_means    : ndarray (k_eff,)     各谱基有效频率
    """
    # D† = pinv(D̂) ∈ ℂ^{P×N}
    D_dag = np.linalg.pinv(d_hat)

    n_samples, n_freq = d_hat.shape
    n_controls = factors.shape[1]

    # 确定有效组分数 k_eff（与 construct_pure_library 逻辑一致）
    k_max = int(config.k_max)
    k_eff = min(k_max, n_freq)
    if config.k_mode == "fixed":
        k_eff = int(config.k_value)

    library: Dict[str, np.ndarray] = {}

    # 零阶：复对数（截断到前 k_eff 个频率）
    d_hat_k = d_hat[:, :k_eff]
    omega_k  = omega[:k_eff]
    ln_f = np.real(np.log(d_hat_k + 1e-12))
    g    = -np.imag(np.log(d_hat_k + 1e-12)) / (omega_k + 1e-9)
    library["ln_f"] = ln_f
    library["g"]    = g

    # 一阶：diag(D† · ∂D̂/∂c_j)，广播为 (N, k_eff)
    # D_dag: (P, N)，d_d_c[:,j,:]: (N, P)  →  D_dag @ d_d_c = (P, P)
    for j in range(n_controls):
        term = D_dag @ d_d_c[:, j, :]            # (P, P)
        diag = np.diag(term)[:k_eff]             # (k_eff,)
        lib_j = np.tile(diag, (n_samples, 1))    # (N, k_eff)
        library[f"L1_c{j+1}_f"] = np.real(lib_j)
        library[f"L1_c{j+1}_g"] = -np.imag(lib_j) / (omega[:k_eff] + 1e-9)

    # 二阶：diag(D† · ∂²D̂/(∂c_i∂c_j))
    for i in range(n_controls):
        for j in range(n_controls):
            term = D_dag @ d2_d_c[:, i, j, :]    # (P, P)
            diag = np.diag(term)[:k_eff]
            lib_f = np.tile(np.real(diag), (n_samples, 1))
            lib_g = np.tile(-np.imag(diag) / (omega[:k_eff] + 1e-9), (n_samples, 1))
            library[f"Xi2_c{i+1}c{j+1}_f"] = lib_f
            library[f"Xi2_c{i+1}c{j+1}_g"] = lib_g

    # SVD 谱基（仅用于重建输出，不影响算子计算）
    from scipy.linalg import svd as _svd
    _, _, Vt = _svd(d_hat, full_matrices=False)
    spectral_basis = Vt[:k_eff, :]              # (k_eff, P)
    A_proj = d_hat @ spectral_basis.conj().T    # (N, k_eff)
    omega_means = np.real(
        np.diag(spectral_basis @ np.diag(omega) @ spectral_basis.conj().T)
    )                                           # (k_eff,)

    return library, spectral_basis, A_proj, omega_means


from .preprocess import _detect_cartesian_grid


def _build_polynomial_test_functions_with_grads(
    factors: np.ndarray,
    degree: int = 2,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    r"""构造多项式测试函数及其解析梯度。

    对控制变量 ``factors`` 构建最高阶数为 ``degree`` 的多项式基函数，
    并同时计算每个函数关于每个控制维度的**解析**一阶偏导。

    Parameters
    ----------
    factors : ndarray, shape (N, n_controls)
        所有样本点的控制变量取值。
    degree : int
        多项式最高阶数（1 或 2），默认 2。

    Returns
    -------
    Psi : ndarray, shape (M, N)
        测试函数矩阵——每行为一个测试函数在 N 个样本点上的取值。
    dPsi : ndarray, shape (n_controls, M, N)
        ``dPsi[i, m, n] = ∂_{c_i} ψ_m(c_n)``
    names : list of str, length M
        各测试函数的名称。
    """
    N, n_controls = factors.shape
    psi_list: List[np.ndarray] = []
    dpsi_list: List[np.ndarray] = []   # each: (n_controls, N)
    names: List[str] = []

    # 常数函数 ψ_0 = 1
    psi_list.append(np.ones(N))
    dpsi_list.append(np.zeros((n_controls, N)))
    names.append("1")

    # 一阶: ψ = c_i,  ∂_{c_j}ψ = δ_{ij}
    for i in range(n_controls):
        psi_list.append(factors[:, i].copy())
        dp = np.zeros((n_controls, N))
        dp[i] = 1.0
        dpsi_list.append(dp)
        names.append(f"c_{i + 1}")

    if degree >= 2:
        # 自身平方: ψ = c_i^2,  ∂_{c_j}ψ = 2c_i δ_{ij}
        for i in range(n_controls):
            psi_list.append(factors[:, i] ** 2)
            dp = np.zeros((n_controls, N))
            dp[i] = 2.0 * factors[:, i]
            dpsi_list.append(dp)
            names.append(f"c_{i + 1}^2")

        # 交叉项: ψ = c_i c_j (i < j),  ∂_{c_k}ψ = δ_{ki}c_j + δ_{kj}c_i
        for i in range(n_controls):
            for j in range(i + 1, n_controls):
                psi_list.append(factors[:, i] * factors[:, j])
                dp = np.zeros((n_controls, N))
                dp[i] = factors[:, j]
                dp[j] = factors[:, i]
                dpsi_list.append(dp)
                names.append(f"c_{i + 1}*c_{j + 1}")

    Psi = np.array(psi_list)                      # (M, N)
    dPsi = np.array(dpsi_list).transpose(1, 0, 2) # (n_controls, M, N)
    return Psi, dPsi, names


def build_weak_form_library(
    d_hat: np.ndarray,
    factors: np.ndarray,
    omega: np.ndarray,
    test_func_degree: int = 2,
    k_eff: int | None = None,
) -> Tuple[Dict[str, np.ndarray], List[str], np.ndarray]:
    r"""**真正的弱形式算子库**——通过分部积分（IBP）构造，无需对含噪数据求导。

    与 :func:`compute_weak_operators` 的关键区别
    -----------------------------------------
    ``compute_weak_operators`` 仍以 ``d_d_c``（数值微分结果）作为输入，
    只是事后将结果乘以测试函数权重，并不是真正的弱形式。

    本函数**不需要** ``d_d_c``，仅使用频域数据 ``d_hat`` 本身：

    .. math::

        \langle L_i, \psi_m \rangle(\omega)
            = -\sum_n \bigl(
                \partial_{c_i}\psi_m(c_n)\cdot c_i(n)
                + \psi_m(c_n)
              \bigr) \cdot \ln D(c_n, \omega)

    其中 :math:`\partial_{c_i}[\psi_m(c)\,c_i]` 是对**光滑测试函数**的导数，
    可以解析计算，因此完全绕开了对含噪数据 :math:`D` 的数值微分。

    库的形状约定
    -----------
    每个库条目的形状为 ``(M, k_eff)``，其中 M = 测试函数数量。
    该形状与 :func:`~opera.discovery.pipeline_utils.solve_nullspace`
    直接兼容（M 充当"样本"轴，k_eff 充当"组分"轴）。

    Parameters
    ----------
    d_hat : ndarray, shape (N, n_freq)
        频域观测数据（对波长/空间维做 rfft 后的结果）。
    factors : ndarray, shape (N, n_controls)
        所有样本点的控制变量取值矩阵。
    omega : ndarray, shape (n_freq,)
        归一化频率轴（例如 ``np.linspace(0, 1, n_freq)``）。
    test_func_degree : int
        测试函数的多项式阶数，默认 2。
    k_eff : int or None
        保留的频率分量数；None = 保留全部。

    Returns
    -------
    library : dict[str, ndarray(M, k_eff)]
        弱形式算子库，键名遵循 ``solve_nullspace`` 的 ``_f``/``_g`` 过滤约定：

        * ``"wln_f"``, ``"wg"`` — 零阶项（``⟨ln D, ψ_m⟩`` 的实/虚部）
        * ``"wL_{i}_f"``, ``"wL_{i}_g"`` — 第 i 个控制维的一阶弱 Euler 算子
    names : list[str], length M
        测试函数名称列表。
    Psi : ndarray, shape (M, N)
        测试函数在所有样本点的取值矩阵。
    """
    N, n_freq = d_hat.shape
    n_controls = factors.shape[1]

    if k_eff is None:
        k_eff = n_freq
    k_eff = min(k_eff, n_freq)
    omega_k = omega[:k_eff]

    # 复数对数：ln D(c, ω)，shape (N, k_eff)
    ln_D = np.log(d_hat[:, :k_eff] + 1e-12)

    # 构造多项式测试函数及其解析梯度
    Psi, dPsi, names = _build_polynomial_test_functions_with_grads(
        factors, degree=test_func_degree
    )
    # Psi:  (M, N)
    # dPsi: (n_controls, M, N)

    library: Dict[str, np.ndarray] = {}

    # -------------------------------------------------------------------
    # 零阶项: ⟨ln D(·, ω), ψ_m⟩ = Psi @ ln_D  → (M, k_eff)
    # -------------------------------------------------------------------
    w0 = Psi @ ln_D                                              # (M, k_eff)
    library["wln_f"] = np.real(w0)
    library["wg"] = -np.imag(w0) / (omega_k[None, :] + 1e-9)

    # -------------------------------------------------------------------
    # 一阶弱 Euler 算子 (IBP, 无需 d_d_c):
    #   ⟨c_i ∂_{c_i} ln D, ψ_m⟩(ω)
    #       = -Σ_n (∂_{c_i}ψ_m(c_n)·c_i(n) + ψ_m(c_n)) · ln D(c_n, ω)
    # -------------------------------------------------------------------
    for i in range(n_controls):
        ci = factors[:, i]                                       # (N,)
        # IBP 核: ∂_{c_i}[ψ_m(c) · c_i] = ∂_{c_i}ψ_m · c_i + ψ_m
        ibp_i = dPsi[i] * ci[None, :] + Psi                     # (M, N)
        wLi = -(ibp_i @ ln_D)                                   # (M, k_eff)
        library[f"wL_{i + 1}_f"] = np.real(wLi)
        library[f"wL_{i + 1}_g"] = -np.imag(wLi) / (omega_k[None, :] + 1e-9)

    return library, names, Psi


def _compute_control_gradient(field: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """在控制空间上计算标量场 ``field`` 的梯度。

    仅支持因子构成笛卡尔网格的情况，返回形状 ``(N, d)``。
    """
    y = np.asarray(field)
    c = np.asarray(factors, dtype=float)
    n_samples, n_dims = c.shape

    is_grid, uniq_vals, grid_shape, sort_idx = _detect_cartesian_grid(c)
    if not is_grid:
        # 退化为零梯度
        return np.zeros((n_samples, n_dims), dtype=float)

    # 重排并重塑
    y_sorted = y[sort_idx]
    y_grid = y_sorted.reshape(*grid_shape)

    grads = np.gradient(y_grid, *uniq_vals, axis=tuple(range(n_dims)))
    if n_dims == 1:
        grads = [grads]

    # 恢复原始顺序
    inv_idx = np.empty_like(sort_idx)
    inv_idx[sort_idx] = np.arange(n_samples)

    grad_flat = np.zeros((n_samples, n_dims), dtype=float)
    for j in range(n_dims):
        grad_flat[:, j] = grads[j].reshape(n_samples)[inv_idx]
    return grad_flat


def compute_weak_operators(
    d_hat: np.ndarray,
    d_d_c: np.ndarray,
    d2_d_c: np.ndarray,
    factors: np.ndarray,
    psi: np.ndarray,
) -> Dict[str, np.ndarray]:
    """构造弱形式算子矩阵（加权积分版本）。

    本实现采用离散积分与分部积分近似：

    1. 首先用伪逆管道生成原始算子库 ``lib``；
    2. 用 ``psi`` 对每个算子直接加权；
    3. 对一阶算子再减去 ``psi`` 的控制梯度贡献以实现分部积分。

    返回的库格式与 ``construct_pure_library`` 兼容，形状均为 ``(N,K)``。
    """
    lib, basis, A, omega_means = construct_inverse_library(
        d_hat=d_hat,
        d_d_c=d_d_c,
        d2_d_c=d2_d_c,
        omega=np.linspace(0,1,d_hat.shape[1]),
        factors=factors,
        config=DiscoveryConfig(),
    )

    n_samples = d_hat.shape[0]
    psi = np.asarray(psi, dtype=float).reshape(n_samples)
    psi_grad = _compute_control_gradient(psi, factors)

    # rebuild pseudoinverse since we need it here as well
    D_dag = np.linalg.pinv(d_hat)
    n_controls = factors.shape[1]

    # determine effective component count from library shapes
    sample_mat = next(iter(lib.values()))
    k_eff = sample_mat.shape[1]

    weak: Dict[str, np.ndarray] = {}

    # helper to tile diag for all samples
    def tile_diag(mat: np.ndarray) -> np.ndarray:
        diag = np.diag(mat)[:k_eff]
        return np.tile(diag, (n_samples, 1))

    # compute first-order weak operators and store basic L1 matrices
    L1_basic = []
    for j in range(n_controls):
        raw1 = D_dag @ d_d_c[:, j, :]  # shape (n_freq, n_freq)
        m = tile_diag(raw1)
        L1_basic.append(m)
        # weak: psi*m - psi_grad_j（分部积分修正，对每个频率分量减去标量梯度）
        weak[f"L1_c{j+1}_weak"] = psi[:, None] * m - psi_grad[:, j, None] * np.ones((1, k_eff))

    # compute second-order and cross operators
    for i in range(n_controls):
        for j in range(n_controls):
            raw2 = D_dag @ d2_d_c[:, i, j, :]  # shape (n_freq, n_freq)
            m2 = tile_diag(raw2)
            cross_term = L1_basic[i] * L1_basic[j]
            if i == j:
                cross_term = cross_term - L1_basic[i]
            weak[f"L2_c{i+1}c{j+1}_weak"] = psi[:, None] * m2 - psi[:, None] * cross_term

    # include original library entries weighted as a fallback
    for name, mat in lib.items():
        weak[f"orig_{name}"] = psi[:, None] * mat
    return weak
