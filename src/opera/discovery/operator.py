"""备选算子库：伪逆路径与弱形式路径（均通过完整 N×N 帽矩阵实现，不依赖 SVD 分离）。

两种算子路径
-----------
本模块提供两条与 :mod:`opera.discovery.pipeline_utils` 不同的算子构造路径，
两条路径均通过直接乘以 D 逆（Moore-Penrose 伪逆 D†）来计算完整的 N×N 帽矩阵：

    D† = pinv(D̂)  ∈ ℂ^{P×N}
    A  = D̂ @ D†   ∈ ℂ^{N×N}   （完整帽矩阵，方阵，无需截断）

D̂ 为 (N, P)，D† 为 (P, N)，二者相乘得到 **N×N 方阵**（帽矩阵 / 正交投影算子）。
组分数 K = N，由矩阵乘法自然确定，不依赖用户指定的 k_value。

**路径 A — 伪逆算子（construct_inverse_library）**

对 D† @ ∂D̂/∂c_j 这一 (P, P) 方阵取对角线前 N 个元素::

    L_j^{inv}(k) = (D† · ∂D̂/∂c_j)_{kk},   k = 0..N-1

零阶项 ln A 对帽矩阵逐元素取复数对数：ln A = log(D̂ @ D†)。

**路径 B — 弱形式算子（build_weak_form_library）**

无需 ∂D̂/∂c_j（避免对含噪数据求导）。关键步骤：

1. **直接乘以 D 逆得到完整 N×N 帽矩阵（不用 SVD，无需截断）**::

       A = D̂ @ D†  ∈ ℂ^{N×N}   （方阵）

2. **对帽矩阵的对数 ln A 做 IBP 内积**::

       ⟨c_i ∂_{c_i} ln A_k, ψ_m⟩
           = -Σ_n (∂_{c_i}ψ_m(c_n)·c_i(n) + ψ_m(c_n)) · ln A_k(c_n)

   其中 ∂_{c_i}[ψ_m(c)·c_i] 对多项式测试函数 ψ_m 解析可得，
   完全绕开了对含噪数据 D̂ 的数值微分。

测试函数 ψ_m 取多项式基（最高 ``test_func_degree`` 阶），梯度解析计算。
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict

from .types import DiscoveryConfig


def construct_inverse_library(
    d_hat: np.ndarray,
    dD_dc: np.ndarray,
    d2D_dc2: np.ndarray,
    omega: np.ndarray,
    c: np.ndarray,
    config: DiscoveryConfig,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    r"""伪逆算子库（不做 SVD 截断，使用完整 D†-基作为谱基）。

    数学基础
    --------
    D̂ 形状为 (N, P)，其 Moore-Penrose 伪逆 D† 形状为 (P, N)。

    与 SVD 路径（``construct_pure_library``）完全相同的公式，只是谱基不同::

        SVD 路径：spectral_basis = Vt[:K, :]       （前 K 个右奇异向量，截断）
        伪逆路径：spectral_basis = D†.T              （完整 D†-基，K = N，无截断）

    投影系数 A（帽矩阵，方阵，K = N）::

        A(n, k) = D̂(n, :) · D†(:, k)  = (D̂ @ D†)[n, k] ∈ ℂ^{N×N}

    一阶对数导数（按元素，各样本独立）::

        α_j(n, k) = ∂A(n,k)/∂c_j(n) / A(n,k)
                  = (∂D̂(n,:)/∂c_j · D†(:,k)) / A(n,k)

    一阶 Euler 算子（含 c_j 缩放，来自 1.md）::

        L_j(n, k) = c_j(n) · α_j(n, k)

    二阶对数导数（按元素）::

        β_{ij}(n,k) = ∂²A(n,k)/∂c_i∂c_j / A(n,k)  -  α_i(n,k) · α_j(n,k)

    二阶 Euler 算子（含 c 缩放，1.md 公式）::

        Ξ_{ii}(n,k) = c_i(n)² · β_{ii}(n,k) + L_i(n,k)   （对角）
        Ξ_{ij}(n,k) = c_i(n) · c_j(n) · β_{ij}(n,k)       （非对角，i ≠ j）

    .. note::
        与旧实现的区别：旧版用 ``diag(D† @ ∂D̂/∂c_j)`` 计算一阶算子，
        这是对所有样本求和的频域全局量（形如 ``Σ_n D†[k,n] * ∂D̂[n,j,k]``），
        而非每个样本点的独立对数导数。新版改为与 SVD 路径相同的按元素投影::

            ∂A[:,j,:] = (∂D̂[:,j,:] @ spectral_basis.conj().T)
                        （N×N 矩阵，每行对应一个样本，每列对应一个 D†-基组分）

        从而保证 L_i / c_i = 常数（Euler 算子核心特性），零锚点处 L_i = 0。

    Returns
    -------
    library        : dict[str, ndarray(N, N)]
    spectral_basis : ndarray (N, P)   D†-基（D†全部 N 列的转置）
    A              : ndarray (N, N)   完整帽矩阵 D̂ @ D†
    omega_means    : ndarray (N,)     各分量有效频率
    """
    from scipy.linalg import svd as _svd

    # D† = pinv(D̂) ∈ ℂ^{P×N}
    D_dag = np.linalg.pinv(d_hat)

    n_samples, n_freq = d_hat.shape
    n_controls = c.shape[1]

    # 帽矩阵：D̂ @ D† ∈ ℂ^{N×N}，组分数 K = N
    A             = d_hat @ D_dag                                # (N, N)
    k_eff         = n_samples
    spectral_basis = D_dag.T                                      # (N, P)
    omega_means   = np.real(
        np.diag(spectral_basis @ np.diag(omega) @ spectral_basis.conj().T)
    )                                                             # (N,)

    # ── Euler 算子（与 construct_pure_library 相同公式，只是谱基不同）────────
    # dA[n, j, k] = ∂D̂(n,:)/∂c_j · D†(:,k) = dD_dc[n,j,:] @ spectral_basis[k,:].conj()
    dA = np.zeros((n_samples, n_controls, k_eff), dtype=complex)
    for j in range(n_controls):
        dA[:, j, :] = dD_dc[:, j, :] @ spectral_basis.conj().T  # (N,P)@(P,N) = (N,N)

    d2A = np.zeros((n_samples, n_controls, n_controls, k_eff), dtype=complex)
    for i in range(n_controls):
        for j in range(n_controls):
            d2A[:, i, j, :] = d2D_dc2[:, i, j, :] @ spectral_basis.conj().T

    # 屏蔽极小 A 避免数值爆炸
    weights = np.abs(A)
    mask    = weights > 1e-9 * np.max(weights)

    # alpha_j(n,k) = dA[n,j,k] / A[n,k]（无 c 缩放的一阶对数导数）
    alpha  = np.zeros_like(dA)
    L1_mask = mask[:, None, :].repeat(n_controls, axis=1)
    A_exp1  = A[:, None, :].repeat(n_controls, axis=1)
    alpha[L1_mask] = dA[L1_mask] / A_exp1[L1_mask]

    # beta_ij（无 c 缩放的二阶对数导数）
    beta   = np.zeros_like(d2A)
    term1  = np.zeros_like(d2A)
    tm     = mask[:, None, None, :].repeat(n_controls, axis=1).repeat(n_controls, axis=2)
    A_exp2 = A[:, None, None, :].repeat(n_controls, axis=1).repeat(n_controls, axis=2)
    term1[tm] = d2A[tm] / A_exp2[tm]
    for i in range(n_controls):
        for j in range(n_controls):
            beta[:, i, j, :] = term1[:, i, j, :] - alpha[:, i, :] * alpha[:, j, :]

    # L_j = c_j · alpha_j（含 c_j 缩放的一阶 Euler 算子）
    L1 = np.zeros_like(dA)
    for j in range(n_controls):
        L1[:, j, :] = c[:, j, np.newaxis] * alpha[:, j, :]

    # Ξ_ii = c_i² β_ii + L_i（对角）；Ξ_ij = c_i c_j β_ij（非对角）
    Xi2 = np.zeros_like(d2A)
    for i in range(n_controls):
        for j in range(n_controls):
            if i == j:
                Xi2[:, i, j, :] = (c[:, i, np.newaxis] ** 2 * beta[:, i, j, :]
                                    + L1[:, i, :])
            else:
                Xi2[:, i, j, :] = (c[:, i, np.newaxis] * c[:, j, np.newaxis]
                                    * beta[:, i, j, :])

    # ── 库组装 ────────────────────────────────────────────────────────────────
    library: Dict[str, np.ndarray] = {}

    # 零阶：对帽矩阵 A 逐元素取复数对数
    ln_A = np.log(A + 1e-12)                                     # (N, N)
    library["ln_f"] = np.real(ln_A)
    library["g"]    = -np.imag(ln_A) / (omega_means[None, :] + 1e-9)

    # 一阶 Euler 算子
    for j in range(n_controls):
        library[f"L1_c{j+1}_f"] = np.real(L1[:, j, :])
        library[f"L1_c{j+1}_g"] = -np.imag(L1[:, j, :]) / (omega_means[None, :] + 1e-9)

    # 二阶 Euler 算子
    for i in range(n_controls):
        for j in range(n_controls):
            val = Xi2[:, i, j, :]
            library[f"Xi2_c{i+1}c{j+1}_f"] = np.real(val)
            library[f"Xi2_c{i+1}c{j+1}_g"] = -np.imag(val) / (omega_means[None, :] + 1e-9)

    return library, spectral_basis, A, omega_means


from .preprocess import _detect_cartesian_grid


def _build_polynomial_test_functions_with_grads(
    c: np.ndarray,
    degree: int = 2,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    r"""构造多项式测试函数及其解析梯度。

    对控制变量 ``c`` 构建最高阶数为 ``degree`` 的多项式基函数，
    并同时计算每个函数关于每个控制维度的**解析**一阶偏导。

    Parameters
    ----------
    c : ndarray, shape (N, n_controls)
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
    N, n_controls = c.shape
    psi_list: List[np.ndarray] = []
    dpsi_list: List[np.ndarray] = []   # each: (n_controls, N)
    names: List[str] = []

    # 常数函数 ψ_0 = 1
    psi_list.append(np.ones(N))
    dpsi_list.append(np.zeros((n_controls, N)))
    names.append("1")

    # 一阶: ψ = c_i,  ∂_{c_j}ψ = δ_{ij}
    for i in range(n_controls):
        psi_list.append(c[:, i].copy())
        dp = np.zeros((n_controls, N))
        dp[i] = 1.0
        dpsi_list.append(dp)
        names.append(f"c_{i + 1}")

    if degree >= 2:
        # 自身平方: ψ = c_i^2,  ∂_{c_j}ψ = 2c_i δ_{ij}
        for i in range(n_controls):
            psi_list.append(c[:, i] ** 2)
            dp = np.zeros((n_controls, N))
            dp[i] = 2.0 * c[:, i]
            dpsi_list.append(dp)
            names.append(f"c_{i + 1}^2")

        # 交叉项: ψ = c_i c_j (i < j),  ∂_{c_k}ψ = δ_{ki}c_j + δ_{kj}c_i
        for i in range(n_controls):
            for j in range(i + 1, n_controls):
                psi_list.append(c[:, i] * c[:, j])
                dp = np.zeros((n_controls, N))
                dp[i] = c[:, j]
                dp[j] = c[:, i]
                dpsi_list.append(dp)
                names.append(f"c_{i + 1}*c_{j + 1}")

    Psi = np.array(psi_list)                      # (M, N)
    dPsi = np.array(dpsi_list).transpose(1, 0, 2) # (n_controls, M, N)
    return Psi, dPsi, names


def build_weak_form_library(
    d_hat: np.ndarray,
    c: np.ndarray,
    omega: np.ndarray,
    test_func_degree: int = 2,
    k_eff: int | None = None,
    spectral_basis: np.ndarray | None = None,
) -> Tuple[Dict[str, np.ndarray], List[str], np.ndarray, np.ndarray, np.ndarray]:
    r"""**弱形式算子库**——直接乘以 D 逆得到完整 N×N 帽矩阵，再做 IBP 内积，无需 SVD 分离和数值微分。

    数学原理（帽矩阵版：D̂ @ D† 为方阵，无需截断）
    -----------------------------------------------
    D̂ 形状为 (N, P)，其伪逆 D† 形状为 (P, N)。
    二者相乘 **D̂ @ D† ∈ ℂ^{N×N}** 是方阵（帽矩阵），无需截断到组分数::

        D† = pinv(D̂)  ∈ ℂ^{P×N}
        A  = D̂ @ D†   ∈ ℂ^{N×N}   （完整帽矩阵，方阵，N = 样本数）

    组分数 K = N，由矩阵乘法自然确定，不依赖用户指定的 k_eff。

    对 ln A_k(c) 做 IBP 内积::

        ⟨L_i^{(k)}, ψ_m⟩ = -Σ_n (∂_{c_i}ψ_m(c_n)·c_i(n) + ψ_m(c_n)) · ln A_k(c_n)

    IBP 核 ``∂_{c_i}[ψ_m(c)·c_i]`` 对多项式测试函数 ψ_m 解析可得，
    完全绕开对含噪 D̂ 的数值微分。

    库的形状约定
    -----------
    每个库条目形状为 ``(M, N)``，其中

    * M = 测试函数数量
    * N = 样本数（帽矩阵列数，即 D̂ @ D† 的列数）

    Parameters
    ----------
    d_hat : ndarray, shape (N, P)
        频域观测数据（rfft 结果）。
    c : ndarray, shape (N, n_controls)
        控制变量矩阵。
    omega : ndarray, shape (P,)
        归一化频率轴 ω ∈ [0, 1]（例如 ``np.linspace(0, 1, P)``）。
    test_func_degree : int
        测试函数的多项式阶数，默认 2。
    k_eff : int or None
        **保留参数，已弃用**。D̂ @ D† 是 N×N 方阵，组分数由矩阵乘法自然确定，
        此参数不再生效。
    spectral_basis : ndarray or None
        **保留参数，已弃用**。

    Returns
    -------
    library : dict[str, ndarray(M, N)]
        弱形式算子库，键名遵循 ``solve_nullspace`` 的 ``_f``/``_g`` 过滤约定：

        * ``"wln_f"``, ``"wg"``       — 零阶项（⟨ln A, ψ_m⟩ 的实/虚部）
        * ``"wL_{i}_f"``, ``"wL_{i}_g"`` — 第 i 个控制维的一阶弱 Euler 算子
    names : list[str], length M
        测试函数名称列表。
    Psi : ndarray, shape (M, N)
        测试函数在所有样本点的取值矩阵。
    spectral_basis : ndarray, shape (N, P)
        D†-基（D†全部 N 列的转置），供 pipeline 重用。
    A : ndarray, shape (N, N)
        完整帽矩阵 D̂ @ D†。
    """
    N, n_freq = d_hat.shape
    n_controls = c.shape[1]

    # k_eff 和 spectral_basis 参数已弃用：D̂ @ D† 是 N×N 方阵，组分数由矩阵乘法自然确定
    if k_eff is not None or spectral_basis is not None:
        import warnings
        warnings.warn(
            "build_weak_form_library：k_eff 和 spectral_basis 参数已弃用并被忽略。"
            "D̂ @ D† 是 N×N 方阵，组分数由矩阵乘法自然确定（K = N）。",
            DeprecationWarning,
            stacklevel=2,
        )

    # ── 步骤 1/2：直接乘以 D 逆，得到完整 N×N 帽矩阵（无需 SVD，无需截断）──
    # D̂ @ D† 是方阵，D† = pinv(D̂) ∈ ℂ^{P×N}
    D_dag = np.linalg.pinv(d_hat)                   # (n_freq, N)

    # D̂ @ D† 是 N×N 方阵，组分数 = N（样本数），无需截断
    spectral_basis = D_dag.T                         # (N, n_freq)
    A = d_hat @ D_dag                               # (N, N) — 完整帽矩阵（hat matrix），方阵

    # 有效频率 ω_k = Re(diag(P diag(ω) P†))，用于 f/g 分离的归一化因子
    # spectral_basis 为 (N, n_freq)，omega_means 为 (N,)
    omega_means = np.real(
        np.diag(spectral_basis @ np.diag(omega) @ spectral_basis.conj().T)
    )                                                # (N,)

    # ── 步骤 3：对完整帽矩阵 A 逐元素取复数对数（lnA 的另一种实现方式）──────
    # A = D̂ @ D† 是 N×N 方阵；ln_A ∈ ℂ^{N×N}
    ln_A = np.log(A + 1e-12)                        # (N, N)

    # ── 步骤 4：多项式测试函数及其解析梯度 ────────────────────────────────
    Psi, dPsi, names = _build_polynomial_test_functions_with_grads(
        c, degree=test_func_degree
    )
    # Psi:  (M, N)
    # dPsi: (n_controls, M, N)

    library: Dict[str, np.ndarray] = {}

    # -------------------------------------------------------------------
    # 零阶项: ⟨ln A_k, ψ_m⟩ = Psi @ ln_A  → (M, N)
    # Re(·) = 对数幅度内积（f 分量），-Im(·)/ω_k = 相位内积（g 分量）
    # -------------------------------------------------------------------
    w0 = Psi @ ln_A                                     # (M, N)
    library["wln_f"] = np.real(w0)
    library["wg"]    = -np.imag(w0) / (omega_means[None, :] + 1e-9)

    # -------------------------------------------------------------------
    # 一阶弱 Euler 算子 (IBP, 无需 dD_dc):
    #
    #   ⟨c_i ∂_{c_i} ln A_k, ψ_m⟩
    #       = -Σ_n (∂_{c_i}ψ_m(c_n)·c_i(n) + ψ_m(c_n)) · ln A_k(c_n)
    #
    # 导数转移到光滑测试函数 ψ_m 上，完全避免对含噪数据 D̂ 求导。
    # -------------------------------------------------------------------
    for i in range(n_controls):
        ci = c[:, i]                                    # (N,)
        # IBP 核: ∂_{c_i}[ψ_m(c) · c_i] = ∂_{c_i}ψ_m · c_i + ψ_m
        ibp_i = dPsi[i] * ci[None, :] + Psi            # (M, N)
        wLi = -(ibp_i @ ln_A)                          # (M, N)
        library[f"wL_{i + 1}_f"] = np.real(wLi)
        library[f"wL_{i + 1}_g"] = -np.imag(wLi) / (omega_means[None, :] + 1e-9)

    return library, names, Psi, spectral_basis, A


def _compute_control_gradient(field: np.ndarray, c: np.ndarray) -> np.ndarray:
    """在控制空间上计算标量场 ``field`` 的梯度。

    仅支持 c 构成笛卡尔网格的情况，返回形状 ``(N, d)``。
    """
    y = np.asarray(field)
    c = np.asarray(c, dtype=float)
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
    dD_dc: np.ndarray,
    d2D_dc2: np.ndarray,
    c: np.ndarray,
    psi: np.ndarray,
) -> Dict[str, np.ndarray]:
    """构造弱形式算子矩阵（加权积分版本）。

    本实现采用离散积分与分部积分近似：

    1. 首先用伪逆管道生成原始算子库 ``lib``；
    2. 用 ``psi`` 对每个算子直接加权；
    3. 对一阶算子再减去 ``psi`` 的控制梯度贡献以实现分部积分。

    返回的库格式与 ``construct_pure_library`` 兼容，形状均为 ``(N,N)``。
    """
    lib, basis, A, omega_means = construct_inverse_library(
        d_hat=d_hat,
        dD_dc=dD_dc,
        d2D_dc2=d2D_dc2,
        omega=np.linspace(0,1,d_hat.shape[1]),
        c=c,
        config=DiscoveryConfig(),
    )

    n_samples = d_hat.shape[0]
    psi = np.asarray(psi, dtype=float).reshape(n_samples)
    psi_grad = _compute_control_gradient(psi, c)

    # rebuild pseudoinverse since we need it here as well
    D_dag = np.linalg.pinv(d_hat)
    n_controls = c.shape[1]

    # 组分数 = 样本数（与 construct_inverse_library 保持一致，帽矩阵为 N×N 方阵）
    k_eff = n_samples

    weak: Dict[str, np.ndarray] = {}

    # helper：取 (P, P) 方阵的对角线并补零/截断到 k_eff = n_samples
    def tile_diag(mat: np.ndarray, target_len: int, nrows: int) -> np.ndarray:
        d = np.diag(mat)                              # (P,)
        # 补零到 target_len（当 P < target_len 时）或截断（当 P >= target_len 时）
        padded = np.zeros(target_len, dtype=d.dtype)
        take = min(len(d), target_len)
        padded[:take] = d[:take]
        return np.tile(padded, (nrows, 1))            # (N, N)

    # compute first-order weak operators and store basic L1 matrices（含 c_j 缩放）
    # 使用与 construct_inverse_library 相同的按元素投影公式：
    #   alpha_j(n,k) = (dD_dc[n,j,:] @ D†[:,k]) / A[n,k]
    # D_dag: (P,N), spectral_basis = D_dag.T: (N,P)
    # A = D̂ @ D†: (N,N)
    A_hat = d_hat @ D_dag                            # (N, N) hat matrix
    spectral_basis_w = D_dag.T                        # (N, P)

    weights_w = np.abs(A_hat)
    mask_w    = weights_w > 1e-9 * np.max(weights_w)

    alpha_basic = []
    L1_basic    = []
    for j in range(n_controls):
        dA_j = dD_dc[:, j, :] @ spectral_basis_w.conj().T     # (N, N)
        alpha_j = np.zeros_like(dA_j)
        alpha_j[mask_w] = dA_j[mask_w] / A_hat[mask_w]        # element-wise log deriv
        alpha_basic.append(alpha_j)
        L_j = c[:, j, np.newaxis] * alpha_j                    # c_j * alpha_j（含缩放）
        L1_basic.append(L_j)
        # weak: psi * L_j - psi_grad_j（分部积分修正）
        weak[f"L1_c{j+1}_weak"] = psi[:, None] * L_j - psi_grad[:, j, None] * np.ones((1, k_eff))

    # 二阶 Euler 算子（1.md 公式，含 c 缩放）
    for i in range(n_controls):
        for j in range(n_controls):
            d2A_ij = d2D_dc2[:, i, j, :] @ spectral_basis_w.conj().T  # (N, N)
            beta_ij_full = np.zeros_like(d2A_ij)
            beta_ij_full[mask_w] = d2A_ij[mask_w] / A_hat[mask_w]
            beta_ij = beta_ij_full - alpha_basic[i] * alpha_basic[j]   # (N, N)
            if i == j:
                Xi2_ij = c[:, i, np.newaxis] ** 2 * beta_ij + L1_basic[i]
            else:
                Xi2_ij = c[:, i, np.newaxis] * c[:, j, np.newaxis] * beta_ij
            weak[f"L2_c{i+1}c{j+1}_weak"] = psi[:, None] * Xi2_ij

    # include original library entries weighted as a fallback
    for name, mat in lib.items():
        weak[f"orig_{name}"] = psi[:, None] * mat
    return weak
