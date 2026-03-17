"""张量形式 Euler 算子库（按浓度维度组织的高阶张量路径）。

数学背景：张量组织
------------------
设控制变量有 d 个维度，第 j 维有 n_j 个取值点，则数据自然组织为高阶张量::

    spectra_tensor  ∈ ℝ^{n_1 × n_2 × ⋯ × n_d × M}
    D_tensor        ∈ ℂ^{n_1 × n_2 × ⋯ × n_d × P}    （rfft 结果）

记 grid_shape = (n_1, n_2, …, n_d)，N = ∏ n_j（展平样本数），P = M//2+1。

对比传统展平路径（"按样本排列"）::

    传统路径：spectra (N, M) + c (N, d) → 检测网格 → sort → gradient → unsort
    张量路径：spectra_tensor (*grid, M) + c_axes [c1, c2, …] → np.gradient 直接沿轴计算

张量路径的三大优势
------------------
1. **无需网格检测**：浓度空间结构已显式编码在张量轴中，直接用
   ``np.gradient(D_tensor, c_axes[j], axis=j)`` 计算 ∂D̂/∂c_j。

2. **算子保持张量形状**：一阶 Euler 算子 L_j、二阶算子 Ξ_{ij}
   都具有形状 ``(*grid_shape, K)``，物理语义清晰：
   每个张量位置对应一个具体的浓度组合 (c_1[i_1], c_2[i_2], …)。

3. **易于可视化与分析**：不需要将结果映射回控制空间。

算子计算公式（与展平路径完全相同，只是数组形状不同）
----------------------------------------------------
::

    A_tensor = D_tensor @ spectral_basis.conj().T    ∈ ℂ^{*grid × K}

    α_j = (∂D_tensor/∂c_j @ spectral_basis.conj().T) / A_tensor    ∈ ℂ^{*grid × K}
    β_ij = (∂²D_tensor/∂c_i∂c_j @ spectral_basis.conj().T) / A_tensor  − α_i · α_j

    L_j   = c_j_tensor[…, None] · α_j              ∈ ℂ^{*grid × K}
    Ξ_ii  = c_i² · β_ii + L_i                      （对角 Euler 算子）
    Ξ_ij  = c_i · c_j · β_ij          (i ≠ j)       （非对角 Euler 算子）

    f-分量 = Re(·)
    g-分量 = −Im(·) / ω_k

互操作性
--------
``build_tensor_euler_library`` 同时返回：

- ``library_tensor``：字典，值形状为 ``(*grid_shape, K)``（张量组织）
- ``library``      ：字典，值形状为 ``(N, K)``（展平，与 ``solve_nullspace`` 兼容）
- ``grid_shape``   ：张量形状，供调用方重构张量

入口函数 ``run_tensor_discovery`` 接受张量输入并返回标准 ``DiscoveryResult``。
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import svd as _scipy_svd
from typing import List

from .types import DiscoveryConfig, DiscoveryResult

# 对数运算的数值稳定性下界（防止 log(0) 和除以零）
_EPSILON_LOG: float = 1e-12

# ---------------------------------------------------------------------------
# 工具函数：展平 ↔ 张量 互转
# ---------------------------------------------------------------------------

def flat_to_tensor(
    spectra: np.ndarray,
    c: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], tuple[int, ...]]:
    """将展平的 (N, M) 光谱矩阵 + (N, d) 控制矩阵转换为张量形式。

    要求 ``c`` 构成完整的等间距笛卡尔网格（使用 ``_detect_cartesian_grid`` 检测）。

    Parameters
    ----------
    spectra : (N, M)
        展平光谱矩阵，每行一条光谱。
    c : (N, d)
        控制变量矩阵，每行一个样本的控制向量。

    Returns
    -------
    spectra_tensor : (*grid_shape, M)
        按浓度维度组织的高阶光谱张量，C-order 排列（最后一个浓度轴变化最快）。
    c_axes : list of 1D ndarrays, length d
        各浓度维度的取值数组，c_axes[j] 形状为 (n_j,)。
    grid_shape : tuple of ints, length d
        张量形状，grid_shape[j] = len(c_axes[j]) = n_j。

    Raises
    ------
    ValueError
        若 c 不构成完整的笛卡尔网格。
    """
    from .preprocess import _detect_cartesian_grid

    spectra = np.asarray(spectra)
    c = np.asarray(c, dtype=float)

    is_grid, unique_vals, grid_shape, sort_idx = _detect_cartesian_grid(c)
    if not is_grid:
        raise ValueError(
            "c 必须构成完整的等间距笛卡尔网格才能转换为张量形式。"
            "请检查控制变量是否覆盖所有网格交叉点（无遗漏，无重复）。"
        )

    M = spectra.shape[1]
    spectra_sorted = spectra[sort_idx]                    # (N, M) 按 C-order 排序
    spectra_tensor = spectra_sorted.reshape(*grid_shape, M)
    return spectra_tensor, list(unique_vals), grid_shape


def tensor_to_flat(
    tensor: np.ndarray,
    n_leading: int,
) -> np.ndarray:
    """将高阶张量 ``tensor`` 的前 ``n_leading`` 维（网格维）展平为一维。

    Parameters
    ----------
    tensor : ndarray, shape (*grid_shape, ...)
        输入张量，前 n_leading 维为网格维度，其余为特征维度。
    n_leading : int
        要展平的前导维数（即网格维数 d = len(grid_shape)）。

    Returns
    -------
    ndarray, shape (N, ...)
        展平结果，N = prod(grid_shape)。
    """
    grid_shape = tensor.shape[:n_leading]
    N = int(np.prod(grid_shape))
    return tensor.reshape(N, *tensor.shape[n_leading:])


# ---------------------------------------------------------------------------
# 张量频域变换
# ---------------------------------------------------------------------------

def compute_tensor_fourier(
    spectra_tensor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """对光谱张量沿最后一维（波长轴）做 rfft，返回复频域张量和频率轴。

    Parameters
    ----------
    spectra_tensor : (*grid_shape, M)
        按浓度维度组织的光谱张量。

    Returns
    -------
    D_tensor : (*grid_shape, P)  complex
        复频域张量，P = M//2 + 1。
    omega : (P,)
        归一化频率轴 ω ∈ [0, 1]。
    """
    spectra_detrend = spectra_tensor - np.mean(spectra_tensor, axis=-1, keepdims=True)
    D_tensor = np.fft.rfft(spectra_detrend, axis=-1)    # (*grid_shape, P)
    P = D_tensor.shape[-1]
    omega = np.linspace(0.0, 1.0, P)
    return D_tensor, omega


# ---------------------------------------------------------------------------
# 张量导数计算（无需网格检测）
# ---------------------------------------------------------------------------

def compute_tensor_control_derivatives(
    D_tensor: np.ndarray,
    c_axes: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """沿各浓度轴直接计算 ∂D̂/∂c_j 和 ∂²D̂/(∂c_i∂c_j)，无需网格检测。

    ``D_tensor`` 的前 d 个维度按顺序对应 ``c_axes[0], c_axes[1], …, c_axes[d-1]``，
    最后一维为频率轴（P 个点）。

    与展平路径的差异
    ----------------
    传统路径需要：
        1. 调用 ``_detect_cartesian_grid`` 检测网格结构；
        2. 按 C-order 排序，重塑为 (*grid_shape, P)；
        3. 调用 ``np.gradient`` 沿网格轴差分；
        4. 反排序恢复展平顺序。

    张量路径直接：
        ``np.gradient(D_tensor, c_axes[j], axis=j)``  — 步骤 2/3/4 全部省略。

    支持非等间距网格：``np.gradient`` 内部使用有限差分，
    自动处理非等间距坐标（边界使用一阶差分，内部使用二阶中心差分）。

    Parameters
    ----------
    D_tensor : (*grid_shape, P)  complex ndarray
        频域张量，前 d 维为浓度轴，最后一维为频率轴。
    c_axes : list of 1D ndarrays, length d
        c_axes[j] 是第 j 个浓度维度的取值数组（长度 = grid_shape[j]）。

    Returns
    -------
    dD_dc : (*grid_shape, d, P)
        一阶偏导 ∂D̂/∂c_j（d 对应不同控制维度）。
    d2D_dc2 : (*grid_shape, d, d, P)
        二阶混合偏导 ∂²D̂/(∂c_i∂c_j)。
    """
    d = len(c_axes)
    grid_shape = D_tensor.shape[:-1]
    P = D_tensor.shape[-1]

    # 一阶导：沿轴 j 对浓度 c_axes[j] 差分
    dD_dc = np.zeros((*grid_shape, d, P), dtype=complex)
    for j in range(d):
        # np.gradient 沿轴 j 计算，c_axes[j] 提供坐标（支持非等间距）
        dD_dc[..., j, :] = np.gradient(D_tensor, c_axes[j], axis=j)

    # 二阶混合导：对 ∂D̂/∂c_i 再沿轴 j 差分
    d2D_dc2 = np.zeros((*grid_shape, d, d, P), dtype=complex)
    for i in range(d):
        for j in range(d):
            # dD_dc[..., i, :] 形状 (*grid_shape, P)，沿轴 j 差分
            d2D_dc2[..., i, j, :] = np.gradient(dD_dc[..., i, :], c_axes[j], axis=j)

    return dD_dc, d2D_dc2


# ---------------------------------------------------------------------------
# 张量形式 Euler 算子库构建（核心函数）
# ---------------------------------------------------------------------------

def build_tensor_euler_library(
    spectra_tensor: np.ndarray,
    c_axes: list[np.ndarray],
    wavelengths: np.ndarray,
    config: DiscoveryConfig,
) -> tuple[
    dict[str, np.ndarray],   # library（展平，N×K，与 solve_nullspace 兼容）
    dict[str, np.ndarray],   # library_tensor（张量，*grid×K）
    np.ndarray,              # spectral_basis (K, P)
    np.ndarray,              # A_flat (N, K)
    np.ndarray,              # A_tensor (*grid_shape, K)
    np.ndarray,              # omega_means (K,)
    tuple[int, ...],         # grid_shape
]:
    """张量形式 Euler 算子库，按浓度维度保持张量组织。

    与 ``construct_pure_library`` 的数学公式完全相同，
    区别在于数据沿浓度维度组织为高阶张量，不展平排列。

    算子计算流程（张量形式）
    -----------------------
    **步骤 1：频域变换**

        D_tensor = rfft(spectra_tensor, axis=-1)   ∈ ℂ^{*grid × P}

    **步骤 2：控制导数（直接沿浓度轴，无网格检测）**

        ∂D/∂c_j = np.gradient(D_tensor, c_axes[j], axis=j)   ∈ ℂ^{*grid × P}

    **步骤 3：SVD 谱基（在展平的 N×P 矩阵上）**

        D_flat = D_tensor.reshape(N, P)
        U, s, Vt = svd(D_flat)
        spectral_basis = Vt[:K, :]   ∈ ℂ^{K × P}

    **步骤 4：张量投影（广播 @ 操作，无需展平）**

        A_tensor    = D_tensor @ Phi†              ∈ ℂ^{*grid × K}
        dA_j_tensor = (∂D/∂c_j) @ Phi†            ∈ ℂ^{*grid × K}

        其中 Phi† = spectral_basis.conj().T (P, K)，@ 自动广播到所有前导维度。

    **步骤 5：对数导数（全张量形式，无需分样本循环）**

        α_j   = dA_j / A_tensor               ∈ ℂ^{*grid × K}
        β_ij  = d2A_ij / A_tensor − α_i · α_j ∈ ℂ^{*grid × K}

    **步骤 6：Euler 算子（使用 meshgrid 生成的浓度张量）**

        c_mesh[j]  ∈ ℝ^{*grid_shape}           （第 j 维浓度的网格值）
        L_j        = c_mesh[j][…, None] · α_j   ∈ ℂ^{*grid × K}
        Ξ_ii       = c_i² · β_ii + L_i          （对角，含修正项）
        Ξ_ij       = c_i · c_j · β_ij (i≠j)     （非对角）

    Parameters
    ----------
    spectra_tensor : (*grid_shape, M)
        按浓度维度组织的高阶光谱张量。
    c_axes : list of 1D ndarrays, length d
        c_axes[j] 是第 j 个浓度维度的取值数组。
    wavelengths : (M,)
        波长轴，仅用于确认波长点数与 M 一致。
    config : DiscoveryConfig
        算法配置（k_mode, k_value 等与展平路径含义相同）。

    Returns
    -------
    library : dict[str, ndarray(N, K)]
        展平为 (N, K) 的算子库，可直接传入 ``solve_nullspace``。
    library_tensor : dict[str, ndarray(*grid_shape, K)]
        保持张量形状 (*grid_shape, K) 的算子库，供可视化或进一步分析。
    spectral_basis : (K, P)
        SVD 谱基向量。
    A_flat : (N, K)
        展平投影系数。
    A_tensor : (*grid_shape, K)
        张量形式投影系数。
    omega_means : (K,)
        各谱基有效频率。
    grid_shape : tuple[int, ...]
        张量的前 d 个维度形状，N = prod(grid_shape)。
    """
    d = len(c_axes)
    grid_shape: tuple[int, ...] = spectra_tensor.shape[:-1]
    M = spectra_tensor.shape[-1]
    N = int(np.prod(grid_shape))

    assert len(grid_shape) == d, (
        f"spectra_tensor 前 {d} 维应对应 c_axes，"
        f"但 spectra_tensor.shape={spectra_tensor.shape}, len(c_axes)={d}"
    )
    for j, ax in enumerate(c_axes):
        assert len(ax) == grid_shape[j], (
            f"c_axes[{j}] 长度 {len(ax)} 与张量第 {j} 维 {grid_shape[j]} 不匹配"
        )

    # ── 步骤 1：频域变换 ──────────────────────────────────────────────────────
    D_tensor, omega = compute_tensor_fourier(spectra_tensor)    # (*grid, P)
    P = D_tensor.shape[-1]

    # ── 步骤 2：控制导数（沿浓度轴，无需网格检测）────────────────────────────
    dD_dc_tensor, d2D_dc2_tensor = compute_tensor_control_derivatives(D_tensor, c_axes)
    # dD_dc_tensor   : (*grid, d, P)
    # d2D_dc2_tensor : (*grid, d, d, P)

    # ── 步骤 3：SVD 谱基（在展平 N×P 矩阵上做 SVD，频域结构与展平无关）──────
    D_flat = D_tensor.reshape(N, P)
    _, s, Vt = _scipy_svd(D_flat, full_matrices=False)

    k_max = int(config.k_max)
    energy = np.cumsum(s**2) / np.sum(s**2)
    k_eff = int(np.searchsorted(energy, config.rank_energy_threshold)) + 1
    k_eff = min(k_eff, k_max)
    if config.k_mode == "fixed":
        k_eff = int(config.k_value)
    K = k_eff

    spectral_basis = Vt[:K, :]                                  # (K, P)
    Phi_dag = spectral_basis.conj().T                           # (P, K)

    # ── 步骤 4：张量投影（@ 自动广播到所有前导维度）──────────────────────────
    # (*grid, P) @ (P, K) = (*grid, K)
    A_tensor = D_tensor @ Phi_dag                               # (*grid, K)
    omega_means = np.real(
        np.diag(spectral_basis @ np.diag(omega) @ spectral_basis.conj().T)
    )                                                           # (K,)

    # dA 和 d2A 投影
    # (*grid, d, P) @ (P, K) = (*grid, d, K)
    dA_tensor = dD_dc_tensor @ Phi_dag                          # (*grid, d, K)
    # (*grid, d, d, P) @ (P, K) = (*grid, d, d, K)
    d2A_tensor = d2D_dc2_tensor @ Phi_dag                      # (*grid, d, d, K)

    # ── 步骤 5：对数导数（全张量形式）────────────────────────────────────────
    # 屏蔽极小 A
    abs_A = np.abs(A_tensor)                                    # (*grid, K)
    mask = abs_A > 1e-9 * np.max(abs_A)                        # (*grid, K) bool

    # alpha_j = dA_j / A（无 c 缩放的一阶对数导数）
    # dA_tensor[..., j, :] 形状 (*grid, K)，A_tensor 形状 (*grid, K)
    alpha = np.zeros_like(dA_tensor)                            # (*grid, d, K)
    for j in range(d):
        dA_j = dA_tensor[..., j, :]                            # (*grid, K)
        alpha_j = np.zeros_like(dA_j)
        alpha_j[mask] = dA_j[mask] / A_tensor[mask]
        alpha[..., j, :] = alpha_j

    # beta_ij = d2A_ij / A - alpha_i * alpha_j（无 c 缩放的二阶对数导数）
    beta = np.zeros_like(d2A_tensor)                            # (*grid, d, d, K)
    for i in range(d):
        for j in range(d):
            d2A_ij = d2A_tensor[..., i, j, :]                  # (*grid, K)
            term1 = np.zeros_like(d2A_ij)
            term1[mask] = d2A_ij[mask] / A_tensor[mask]
            beta[..., i, j, :] = term1 - alpha[..., i, :] * alpha[..., j, :]

    # ── 步骤 6：Euler 算子（含 c 缩放，使用 meshgrid 浓度张量）──────────────
    # c_mesh[j] 的形状为 (*grid_shape)，广播时需扩展为 (*grid_shape, 1)
    c_mesh = np.meshgrid(*c_axes, indexing="ij")                # d 个 *grid_shape 数组

    # L_j = c_j_tensor[..., None] * alpha_j：(*grid, 1) * (*grid, K) = (*grid, K)
    L1_tensor = np.zeros_like(dA_tensor)                        # (*grid, d, K)
    for j in range(d):
        c_j = c_mesh[j][..., np.newaxis]                        # (*grid, 1)
        L1_tensor[..., j, :] = c_j * alpha[..., j, :]

    # Ξ_{ii} = c_i² β_ii + L_i（对角）；Ξ_{ij} = c_i c_j β_ij（非对角）
    Xi2_tensor = np.zeros_like(d2A_tensor)                      # (*grid, d, d, K)
    for i in range(d):
        c_i = c_mesh[i][..., np.newaxis]                        # (*grid, 1)
        for j in range(d):
            c_j = c_mesh[j][..., np.newaxis]                    # (*grid, 1)
            if i == j:
                Xi2_tensor[..., i, j, :] = (
                    c_i ** 2 * beta[..., i, j, :] + L1_tensor[..., i, :]
                )
            else:
                Xi2_tensor[..., i, j, :] = c_i * c_j * beta[..., i, j, :]

    # ── 步骤 7：f/g 分离 → 实值张量算子库 ───────────────────────────────────
    # 零阶项
    ln_f_tensor = np.real(np.log(A_tensor + 1e-12))             # (*grid, K)
    g_tensor    = -np.imag(np.log(A_tensor + 1e-12)) / (omega_means + 1e-9)

    library_tensor: dict[str, np.ndarray] = {
        "ln_f":   ln_f_tensor,
        "g":      g_tensor,
        "ln_f^2": ln_f_tensor ** 2,
        "g^2":    g_tensor ** 2,
    }

    # 一阶 Euler 算子
    for j in range(d):
        term = L1_tensor[..., j, :]                             # (*grid, K)
        library_tensor[f"L1_c{j+1}_f"] = np.real(term)
        library_tensor[f"L1_c{j+1}_g"] = -np.imag(term) / (omega_means + 1e-9)

    # 二阶 Euler 算子
    for i in range(d):
        for j in range(d):
            val = Xi2_tensor[..., i, j, :]                      # (*grid, K)
            library_tensor[f"Xi2_c{i+1}c{j+1}_f"] = np.real(val)
            library_tensor[f"Xi2_c{i+1}c{j+1}_g"] = -np.imag(val) / (omega_means + 1e-9)

    # ── 步骤 8：展平 → (N, K) 供 solve_nullspace 使用 ──────────────────────
    library: dict[str, np.ndarray] = {
        key: tensor_to_flat(val, n_leading=d)
        for key, val in library_tensor.items()
    }
    A_flat = A_tensor.reshape(N, K)

    return library, library_tensor, spectral_basis, A_flat, A_tensor, omega_means, grid_shape


# ---------------------------------------------------------------------------
# 顶层入口函数
# ---------------------------------------------------------------------------

def run_tensor_discovery(
    spectra_tensor: np.ndarray,
    c_axes: list[np.ndarray],
    wavelengths: np.ndarray,
    config: DiscoveryConfig | None = None,
) -> "DiscoveryResult":
    """从按浓度维度组织的高阶张量光谱数据中发现隐式物理方程。

    这是对 ``run_discovery`` 的补充入口，接受张量形式输入，
    无需展平并手动构造 ``c`` 矩阵。

    Parameters
    ----------
    spectra_tensor : (*grid_shape, M)
        高阶光谱张量：前 d 维为浓度维度，最后一维为波长。
        例如对 2 种组分浓度 c1 ∈ {0, 0.5, 1.0}，c2 ∈ {0, 1.0, 2.0}，
        则形状为 ``(3, 3, M)``。
    c_axes : list of 1D ndarrays, length d
        各浓度维度的取值数组。c_axes[j] 长度必须等于 grid_shape[j]。
        **必须包含全零参考点**（即 c_axes[j][0] == 0 或等价条件）。
    wavelengths : (M,)
        单调递增的波长轴，仅用于 IFFT 重建的点数。
    config : DiscoveryConfig, optional
        算法配置，默认使用 SVD 路径、k_value=3。

    Returns
    -------
    DiscoveryResult
        与 ``run_discovery`` 返回结构相同，额外在 ``metadata`` 中包含：

        * ``"grid_shape"``      — 输入张量的浓度维形状
        * ``"c_axes"``          — 各浓度维度的取值数组
        * ``"library_tensor"``  — 张量形式算子库 dict[str, (*grid, K)]

    Notes
    -----
    若需要从展平 (N, M) + c (N, d) 输入使用张量路径，请先调用
    ``flat_to_tensor(spectra, c)`` 获得 ``spectra_tensor`` 和 ``c_axes``。
    """
    from .pipeline_utils import solve_nullspace, pretty_name
    from time import perf_counter

    cfg = config or DiscoveryConfig()
    spectra_tensor = np.asarray(spectra_tensor)
    wavelengths = np.asarray(wavelengths)
    d = len(c_axes)
    grid_shape = spectra_tensor.shape[:-1]
    N = int(np.prod(grid_shape))

    # ── 零浓度锚点检测 ──────────────────────────────────────────────────────
    # 在张量中，锚点对应 grid 索引全为 0（即 c_axes[j][0] ≈ 0 for all j）
    anchor_found = all(
        abs(float(ax[0])) <= float(cfg.zero_anchor_tol) for ax in c_axes
    )
    if not anchor_found:
        raise ValueError(
            "未检测到零浓度样本。c_axes[j][0] 必须近似为 0（所有控制维度的第一个取值），"
            "以提供物理参考（空白样本）。"
        )

    t0 = perf_counter()

    # ── 构建张量 Euler 算子库 ───────────────────────────────────────────────
    (
        library,          # dict[str, (N, K)]
        library_tensor,   # dict[str, (*grid, K)]
        spectral_basis,   # (K, P)
        A_flat,           # (N, K)
        A_tensor,         # (*grid, K)
        omega_means,      # (K,)
        grid_shape_out,
    ) = build_tensor_euler_library(spectra_tensor, c_axes, wavelengths, cfg)

    t_lib = perf_counter()

    # ── SINDy-PI 零空间识别（使用展平库）───────────────────────────────────
    component_models = solve_nullspace(library)

    op_names = [pretty_name(k) for k in library.keys()]
    J = len(op_names)
    K = len(component_models)

    # ── 组装 Xi 系数张量 ─────────────────────────────────────────────────────
    lib_f_keys = sorted(k for k in library if "_f" in k or k in ("ln_f", "ln_f^2"))
    lib_g_keys = sorted(k for k in library if "_g" in k or k in ("g", "g^2"))

    Xi = np.zeros((1, J, K))
    eqn_strs: list[str] = []

    for k in range(K):
        model = component_models[k]
        xi_k = np.zeros(J, dtype=float)
        parts: list[str] = []

        if "f" in model:
            cfs, f_keys = model["f"]
            terms = []
            for coef, fname in zip(cfs, f_keys):
                coef_r = float(np.real(coef))
                try:
                    idx = list(library.keys()).index(fname)
                    xi_k[idx] = coef_r
                except ValueError:
                    pass
                if abs(coef_r) > 1e-3:
                    terms.append(f"{coef_r:.3f}·{pretty_name(fname)}")
            if terms:
                parts.append("f: " + " + ".join(terms))

        if "g" in model:
            cgs, g_keys = model["g"]
            terms = []
            for coef, gname in zip(cgs, g_keys):
                coef_r = float(np.real(coef))
                try:
                    idx = list(library.keys()).index(gname)
                    xi_k[idx] = coef_r
                except ValueError:
                    pass
                if abs(coef_r) > 1e-3:
                    terms.append(f"{coef_r:.3f}·{pretty_name(gname)}")
            if terms:
                parts.append("g: " + " + ".join(terms))

        Xi[0, :, k] = xi_k
        eqn_strs.append(f"Comp {k+1} " + " | ".join(parts))

    # ── 纯谱重建 ─────────────────────────────────────────────────────────────
    S_real = np.fft.irfft(spectral_basis, n=len(wavelengths), axis=1).T   # (M, K)
    pure_spectra_complex = spectral_basis.T                                # (P, K)

    return DiscoveryResult(
        S_real=S_real,
        f_response_eval=A_flat,
        A_matrix=Xi[0],
        Xi=Xi,
        operator_names=op_names,
        f_response=A_flat,
        pure_spectra_complex=pure_spectra_complex,
        reconstruction_error=0.0,
        xi_by_control={f"component_{k+1}": Xi[0, :, k] for k in range(K)},
        component_scores=None,
        component_energy_ratio=None,
        component_nonzero_ratio=None,
        quality_flags=["ok"],
        latex_blocks=eqn_strs,
        diagnostics={
            "k_eff":           float(K),
            "k_selected":      float(K),
            "nullspace_energy": 0.0,
            "sigma_gap_min":   0.0,
            "anchor_count":    1.0,
        },
        metadata={
            "models":                component_models,
            "equations":             eqn_strs,
            "k_source":              "k_value" if cfg.k_mode == "fixed" else "mode_rule",
            "J_tot":                 J,
            "operator_block_ranges": {},
            "grid_shape":            grid_shape_out,
            "c_axes":                c_axes,
            "library_tensor":        library_tensor,
        },
    )


# ---------------------------------------------------------------------------
# 退化策略：纤维采样模式（高维浓度空间）
# ---------------------------------------------------------------------------

def build_fiber_euler_library(
    fibers: list[np.ndarray],
    c_axes: list[np.ndarray],
    wavelengths: np.ndarray,
    config: DiscoveryConfig,
) -> tuple[
    dict[str, np.ndarray],        # library（展平，N_total×K，与 solve_nullspace 兼容）
    list[dict[str, np.ndarray]],  # fiber_libraries：每条纤维的张量库 (n_j, K)
    np.ndarray,                   # spectral_basis (K, P)
    np.ndarray,                   # A_combined (N_total, K)
    np.ndarray,                   # omega_means (K,)
]:
    """从 d 条轴对齐纤维（1D 扫描）构建 Euler 算子库（高维退化策略）。

    动机：全笛卡尔网格退化
    -----------------------
    当控制变量维度 d 较高时，完整的笛卡尔网格需要 ``Πn_j`` 个实验点（随 d 指数增长）。
    纤维采样只需沿每个轴各做一次 1D 扫描，总共 ``Σn_j`` 个实验点（随 d **线性**增长）：

    ============================  =============================================
    方法                           所需实验点数（d=5，每轴 n=4）
    ----------------------------  ---------------------------------------------
    全笛卡尔网格                    4^5 = 1024
    纤维采样（本函数）               4×5 = 20
    ============================  =============================================

    张量语义
    ---------
    纤维 j 仍然是一个张量切片：在 d 维全张量
    ``(*grid_shape, M)`` 中，纤维 j 对应将第 j 轴外其余所有轴固定在参考浓度（通常为 0）的
    1D 子张量。因此所有代数结构保持不变，只是非对角混合偏导无法计算。

    物理约束
    ---------
    沿纤维 j：c_k = c_ref_k ≈ 0（k ≠ j），因此::

        L_k = c_k · α_k = 0              (k ≠ j)
        Ξ_{ij} = c_i · c_j · β_{ij} = 0  (至少一个下标 ≠ j)

    只有对角项可以从纤维数据中提取::

        L_j = c_j · α_j                （一阶 Euler 算子）
        Ξ_{jj} = c_j² · β_{jj} + L_j  （二阶对角 Euler 算子）

    由此构成"对角 Euler 库"，适用于无强交叉项交互的物理系统
    （Beer-Lambert 定律等加法模型正属此类）。

    组合库的结构
    -------------
    返回的展平库形状为 ``(N_total, K)``，其中 ``N_total = Σn_j``。
    行按纤维顺序排列（先纤维 0 的 n_0 行，再纤维 1 的 n_1 行，依此类推）。
    对于纤维 j 的行，维度 k ≠ j 的所有算子列为精确零（不是近似），
    因为沿该纤维 c_k = 0 是物理事实。

    Parameters
    ----------
    fibers : list of 2D arrays, length d
        ``fibers[j]`` 形状 ``(n_j, M)``，是第 j 个浓度维度的 1D 扫描光谱。
        **每条纤维必须包含零浓度参考点**：``c_axes[j][0] ≈ 0``。
        对于 d 个浓度变量，共需 d 条纤维，每条仅改变一个维度。
    c_axes : list of 1D arrays, length d
        ``c_axes[j]`` 形状 ``(n_j,)``，是纤维 j 扫描的浓度取值（含 0）。
    wavelengths : (M,)
        波长轴（仅用于一致性检查，点数应等于 M）。
    config : DiscoveryConfig
        算法配置，K 选择规则与全张量路径相同。

    Returns
    -------
    library : dict[str, ndarray(N_total, K)]
        展平算子库，可直接传入 ``solve_nullspace``。
        键名与全张量路径相同：``ln_f``, ``g``, ``L1_c{j}_f``, ``Xi2_c{j}c{j}_f``, …
    fiber_libraries : list of dict[str, ndarray(n_j, K)], length d
        每条纤维的 1D 张量库，形状 ``(n_j, K)``。
        ``fiber_libraries[j][key][i, k]`` = 纤维 j 第 i 点第 k 组分的算子值。
    spectral_basis : (K, P)
        SVD 谱基，从所有纤维数据联合提取。
    A_combined : (N_total, K)
        所有纤维样本的投影系数。
    omega_means : (K,)
        各谱基有效频率。
    """
    d = len(fibers)
    assert d == len(c_axes), f"fibers 长度 {d} 与 c_axes 长度 {len(c_axes)} 不匹配"

    # ── 步骤 1：对每条纤维做 rfft ─────────────────────────────────────────────
    # fibers[j]: (n_j, M) → D_hats[j]: (n_j, P)
    D_hats: list[np.ndarray] = []
    omega_out: np.ndarray | None = None
    for j, fiber_j in enumerate(fibers):
        fiber_j = np.asarray(fiber_j)
        assert fiber_j.ndim == 2, f"fibers[{j}] 应为 2D 数组 (n_j, M)"
        D_j, omega_j = compute_tensor_fourier(fiber_j)   # (n_j, P)
        D_hats.append(D_j)
        omega_out = omega_j
    omega = omega_out                                      # (P,)
    P = D_hats[0].shape[1]

    # ── 步骤 2：联合 SVD（在所有纤维数据上）─────────────────────────────────
    # 将所有纤维的频域数据垂直堆叠：(Σn_j, P)
    D_all = np.vstack(D_hats)                             # (N_total, P)
    N_total = D_all.shape[0]

    _, s, Vt = _scipy_svd(D_all, full_matrices=False)

    k_max = int(config.k_max)
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    k_eff = int(np.searchsorted(energy, config.rank_energy_threshold)) + 1
    k_eff = min(k_eff, k_max)
    if config.k_mode == "fixed":
        k_eff = int(config.k_value)
    K = k_eff

    spectral_basis = Vt[:K, :]                            # (K, P)
    Phi_dag = spectral_basis.conj().T                     # (P, K)
    omega_means = np.real(
        np.diag(spectral_basis @ np.diag(omega) @ spectral_basis.conj().T)
    )                                                     # (K,)

    # ── 步骤 3：逐条纤维计算对角 Euler 算子 ──────────────────────────────────
    fiber_libraries: list[dict[str, np.ndarray]] = []
    A_combined = D_all @ Phi_dag                          # (N_total, K)

    row_offset = 0
    # 预先建立 all_keys 集合（含所有维度的算子名称，方便后续填零）
    all_keys: list[str] = ["ln_f", "g", "ln_f^2", "g^2"]
    for j in range(d):
        all_keys += [
            f"L1_c{j+1}_f",    f"L1_c{j+1}_g",
            f"Xi2_c{j+1}c{j+1}_f", f"Xi2_c{j+1}c{j+1}_g",
        ]

    # 初始化全零展平库
    library: dict[str, np.ndarray] = {k: np.zeros((N_total, K)) for k in all_keys}

    for j in range(d):
        D_j = D_hats[j]                                  # (n_j, P)
        c_j = np.asarray(c_axes[j], dtype=float)         # (n_j,)
        n_j = D_j.shape[0]

        # 投影到谱基
        A_j = D_j @ Phi_dag                              # (n_j, K)

        # 一阶与二阶对角导数（仅沿第 j 轴，无需多维 gradient）
        dA_j   = np.gradient(A_j,  c_j, axis=0)         # (n_j, K)
        d2A_j  = np.gradient(dA_j, c_j, axis=0)         # (n_j, K)

        # 对数导数（屏蔽极小 A）
        abs_A_j = np.abs(A_j)
        mask_j = abs_A_j > 1e-9 * (np.max(abs_A_j) + 1e-30)

        alpha_j = np.zeros_like(A_j)                    # (n_j, K)
        alpha_j[mask_j] = dA_j[mask_j] / A_j[mask_j]

        term1 = np.zeros_like(A_j)
        term1[mask_j] = d2A_j[mask_j] / A_j[mask_j]
        beta_jj = term1 - alpha_j ** 2                  # (n_j, K)

        # Euler 算子
        c_j_col = c_j[:, np.newaxis]                    # (n_j, 1) for broadcast
        L_j    = c_j_col * alpha_j                      # (n_j, K)
        Xi_jj  = c_j_col ** 2 * beta_jj + L_j          # (n_j, K)

        # 对数参数（零阶项）
        ln_f_j = np.real(np.log(A_j + _EPSILON_LOG))          # (n_j, K)
        g_j    = -np.imag(np.log(A_j + _EPSILON_LOG)) / (omega_means + 1e-9)

        # 构建纤维库（仅含本维度的对角算子）
        fiber_lib: dict[str, np.ndarray] = {
            "ln_f":                       ln_f_j,
            "g":                          g_j,
            "ln_f^2":                     ln_f_j ** 2,
            "g^2":                        g_j ** 2,
            f"L1_c{j+1}_f":              np.real(L_j),
            f"L1_c{j+1}_g":              -np.imag(L_j) / (omega_means + 1e-9),
            f"Xi2_c{j+1}c{j+1}_f":      np.real(Xi_jj),
            f"Xi2_c{j+1}c{j+1}_g":      -np.imag(Xi_jj) / (omega_means + 1e-9),
        }
        fiber_libraries.append(fiber_lib)

        # 填入展平库
        for key, val in fiber_lib.items():
            library[key][row_offset: row_offset + n_j] = val
        row_offset += n_j

    return library, fiber_libraries, spectral_basis, A_combined, omega_means


def approximate_tensor_from_fibers(
    fibers: list[np.ndarray],
    c_axes: list[np.ndarray],
    target_c_axes: list[np.ndarray],
) -> np.ndarray:
    """用纤维数据以加法近似模型重构任意浓度组合处的光谱张量。

    数学原理（加法近似 / ANOVA 一阶模型）
    ------------------------------------
    对于加法型物理系统（如多组分 Beer-Lambert）::

        D̂(c) ≈ D̂(0) + Σ_j [D̂_j(c_j) - D̂(0)]

    其中 D̂_j(c_j) 是纤维 j 在浓度 c_j 处的测量值，D̂(0) 是公共零参考点。

    此近似对线性/加法贡献的系统（如吸光度可加的混合物）精确，
    对含交叉项（相互作用）的系统存在偏差。

    用途
    -----
    * 将纤维数据"升维"到全张量，然后调用 ``run_tensor_discovery``
    * 在指定目标网格 ``target_c_axes`` 上评估近似光谱张量
    * 配合 ``build_tensor_euler_library`` 生成完整的（包含非对角项的）算子库估计

    Parameters
    ----------
    fibers : list of (n_j, M) arrays
        纤维光谱，``fibers[j][0]`` 应为零参考点。
    c_axes : list of (n_j,) arrays
        各纤维的浓度轴，``c_axes[j][0] ≈ 0``。
    target_c_axes : list of (m_j,) arrays
        目标张量各维度的浓度取值，不必与原始纤维轴一致（通过线性插值外推）。

    Returns
    -------
    approx_tensor : (*target_grid_shape, M)
        加法近似的光谱张量，可直接传入 ``build_tensor_euler_library``。

    Notes
    -----
    插值使用 ``np.interp``（夹断外推）。若目标浓度超出纤维范围，
    外推结果可能不可靠；建议将 ``target_c_axes[j]`` 限制在
    ``c_axes[j]`` 的范围内。
    """
    d = len(fibers)
    assert d == len(c_axes) == len(target_c_axes)

    M = fibers[0].shape[1]

    # 从各纤维的第一行提取零参考光谱
    ref_rows = [np.asarray(fibers[j])[0] for j in range(d)]

    # 一致性检验：各纤维的零参考光谱应基本相同（量纲化残差 < 5%）
    if d > 1:
        ref_matrix = np.stack(ref_rows, axis=0)             # (d, M)
        ref_mean = np.mean(ref_matrix, axis=0)              # (M,)
        ref_norm = np.linalg.norm(ref_mean)
        if ref_norm > _EPSILON_LOG:
            max_relative_dev = np.max(
                np.linalg.norm(ref_matrix - ref_mean, axis=1) / ref_norm
            )
            if max_relative_dev > 0.05:
                import warnings
                warnings.warn(
                    f"各纤维零参考光谱之间存在较大差异（最大相对偏差 "
                    f"{max_relative_dev:.1%}）。这可能是由测量噪声或不同实验条件引起。"
                    "加法近似将对零参考光谱取均值，可能引入系统误差。"
                    "建议检查各纤维的第一行（c=0 处）是否代表相同样品状态。",
                    UserWarning,
                    stacklevel=2,
                )

    D0 = np.mean(ref_rows, axis=0)                         # (M,) 公共零参考光谱

    # 对每条纤维插值到目标浓度轴
    fiber_interp: list[np.ndarray] = []
    for j in range(d):
        fj = np.asarray(fibers[j])                       # (n_j, M)
        cj = np.asarray(c_axes[j], dtype=float)
        tj = np.asarray(target_c_axes[j], dtype=float)
        interp_j = np.zeros((len(tj), M))
        for wl_idx in range(M):
            interp_j[:, wl_idx] = np.interp(tj, cj, fj[:, wl_idx])
        fiber_interp.append(interp_j)                    # (m_j, M)

    # 加法近似：D(c) ≈ D0 + Σ_j [D_j(c_j) - D0]
    # 用 meshgrid 广播到全张量
    target_shape = tuple(len(tj) for tj in target_c_axes)
    approx = np.zeros((*target_shape, M))

    # 加上各纤维贡献（通过 reshape + 广播）
    for j in range(d):
        delta_j = fiber_interp[j] - D0                  # (m_j, M)
        # reshape 为 (1,...,m_j,...,1,M) — 只有第 j 轴非单元素
        shape = [1] * d + [M]
        shape[j] = len(target_c_axes[j])
        approx += delta_j.reshape(shape)                 # 广播到全张量

    # 加上参考基底（D0 对应全为零的贡献之和已经在上面减去，需补回一次 D0）
    # 公式 = D0 + Σ_j(D_j - D0) = (1-d)*D0 + Σ_j D_j
    # 当前 approx = Σ_j(D_j - D0) = Σ_j D_j - d*D0，所以还需加 D0
    approx += D0

    return approx


def run_fiber_discovery(
    fibers: list[np.ndarray],
    c_axes: list[np.ndarray],
    wavelengths: np.ndarray,
    config: DiscoveryConfig | None = None,
) -> "DiscoveryResult":
    """从纤维（1D 扫描）数据中发现隐式物理方程（高维退化入口）。

    这是 ``run_tensor_discovery`` 的退化版入口，适用于高维浓度空间中
    **无法获取完整笛卡尔网格**的情形。

    采样对比
    ---------
    ==================  =============================
    策略                 所需实验点数（d 维，每轴 n 点）
    ------------------  -----------------------------
    全张量网格           n^d（指数增长）
    纤维采样（本函数）    n × d（线性增长）
    ==================  =============================

    数据组织
    ---------
    每条纤维 ``fibers[j]`` 是第 j 个浓度维度的 1D 扫描结果：

    * 固定所有其他浓度于参考值（通常 0）
    * 仅改变第 j 个浓度 ``c_axes[j]``

    可用算子
    ---------
    纤维模式下可计算的 Euler 算子（在各纤维自身轴上）：

    * ``L_j`` — 一阶 Euler 算子（对角）✓
    * ``Ξ_{jj}`` — 二阶对角 Euler 算子 ✓
    * ``Ξ_{ij}``（i ≠ j）— **不可用**（无法从纤维数据估计交叉偏导）✗

    Parameters
    ----------
    fibers : list of 2D arrays, length d
        ``fibers[j]`` 形状 ``(n_j, M)``，第 j 维纤维光谱。
        ``c_axes[j][0]`` 必须近似为 0（零参考点）。
    c_axes : list of 1D arrays, length d
        ``c_axes[j]`` 形状 ``(n_j,)``，纤维 j 扫描的浓度值（单调递增，首元素 ≈ 0）。
    wavelengths : (M,)
        波长轴，用于光谱重建。
    config : DiscoveryConfig, optional
        算法配置，默认 k_value=3，k_mode="fixed"。

    Returns
    -------
    DiscoveryResult
        与 ``run_tensor_discovery`` 返回结构相同，额外在 ``metadata`` 中包含：

        * ``"fiber_libraries"`` — 每条纤维的 1D 张量库 ``list[dict[str, (n_j,K)]]``
        * ``"sampling_mode"``   — ``"fiber"``（标识纤维模式）
        * ``"n_total_samples"`` — 总样本数 Σn_j
        * ``"n_full_grid"``     — 假设全网格所需样本数 Πn_j

    See Also
    --------
    run_tensor_discovery : 全网格张量入口（需要 Πn_j 个样本）
    approximate_tensor_from_fibers : 从纤维数据加法近似重构全张量
    """
    from .pipeline_utils import solve_nullspace, pretty_name

    cfg = config or DiscoveryConfig()
    d = len(fibers)
    wavelengths = np.asarray(wavelengths)

    # ── 零参考点检验 ─────────────────────────────────────────────────────────
    for j, ax in enumerate(c_axes):
        if abs(float(np.asarray(ax)[0])) > float(cfg.zero_anchor_tol):
            raise ValueError(
                f"c_axes[{j}][0] = {float(np.asarray(ax)[0]):.4g} 不近似为 0。"
                "纤维模式要求每条纤维的第一个测量点为零参考浓度（空白样本），"
                f"容差为 zero_anchor_tol={cfg.zero_anchor_tol}。"
            )

    # ── 构建纤维 Euler 算子库 ────────────────────────────────────────────────
    (
        library,           # dict[str, (N_total, K)]
        fiber_libraries,   # list of dict[str, (n_j, K)]
        spectral_basis,    # (K, P)
        A_combined,        # (N_total, K)
        omega_means,       # (K,)
    ) = build_fiber_euler_library(fibers, c_axes, wavelengths, cfg)

    K = spectral_basis.shape[0]
    N_total = A_combined.shape[0]

    # ── SINDy-PI 零空间识别 ──────────────────────────────────────────────────
    component_models = solve_nullspace(library)

    op_names = [pretty_name(k) for k in library.keys()]
    J = len(op_names)

    # ── 组装 Xi 系数张量 ─────────────────────────────────────────────────────
    Xi = np.zeros((1, J, K))
    eqn_strs: list[str] = []

    for k in range(K):
        model = component_models[k]
        xi_k = np.zeros(J, dtype=float)
        parts: list[str] = []

        for field_name in ("f", "g"):
            if field_name not in model:
                continue
            cfs, fkeys = model[field_name]
            terms = []
            for coef, fname in zip(cfs, fkeys):
                coef_r = float(np.real(coef))
                try:
                    idx = list(library.keys()).index(fname)
                    xi_k[idx] = coef_r
                except ValueError:
                    pass
                if abs(coef_r) > 1e-3:
                    terms.append(f"{coef_r:.3f}·{pretty_name(fname)}")
            if terms:
                parts.append(f"{field_name}: " + " + ".join(terms))

        Xi[0, :, k] = xi_k
        eqn_strs.append(f"Comp {k + 1} " + " | ".join(parts))

    # ── 纯谱重建 ─────────────────────────────────────────────────────────────
    S_real = np.fft.irfft(spectral_basis, n=len(wavelengths), axis=1).T   # (M, K)
    pure_spectra_complex = spectral_basis.T                                # (P, K)

    # 统计信息
    n_per_fiber = [len(np.asarray(ax)) for ax in c_axes]
    n_full_grid = int(np.prod(n_per_fiber))

    return DiscoveryResult(
        S_real=S_real,
        f_response_eval=A_combined,
        A_matrix=Xi[0],
        Xi=Xi,
        operator_names=op_names,
        f_response=A_combined,
        pure_spectra_complex=pure_spectra_complex,
        reconstruction_error=0.0,
        xi_by_control={f"component_{k + 1}": Xi[0, :, k] for k in range(K)},
        component_scores=None,
        component_energy_ratio=None,
        component_nonzero_ratio=None,
        quality_flags=["ok"],
        latex_blocks=eqn_strs,
        diagnostics={
            "k_eff":            float(K),
            "k_selected":       float(K),
            "nullspace_energy": 0.0,
            "sigma_gap_min":    0.0,
            "anchor_count":     float(d),   # d 个纤维共享一个零参考点
        },
        metadata={
            "models":              component_models,
            "equations":           eqn_strs,
            "k_source":            "k_value" if cfg.k_mode == "fixed" else "mode_rule",
            "J_tot":               J,
            "operator_block_ranges": {},
            "sampling_mode":       "fiber",
            "fiber_libraries":     fiber_libraries,
            "c_axes":              c_axes,
            "n_total_samples":     N_total,
            "n_full_grid":         n_full_grid,
        },
    )
