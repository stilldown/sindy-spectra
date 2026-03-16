"""强形式 Euler 算子库构建与 SINDy-PI 零空间求解。

数学基础：Euler 算子分解
-----------------------
设观测的频域投影系数为 A_k(c) = D̂(c, ·) · v_k*，其中 v_k 是第 k 个 SVD 谱基向量。

将投影系数写成幅度-相位形式::

    A_k(c) = |A_k(c)| · exp(i · φ_k(c))

在复对数域::

    ln A_k(c) = ln|A_k(c)| + i·φ_k(c)

定义两个实值函数::

    f_k(c) := ln|A_k(c)|          ("f-分量"，对数幅度)
    g_k(c) := φ_k(c) / ω_k        ("g-分量"，归一化相位)

其中 ω_k 是谱基 v_k 的有效频率（谱能量加权均值）。

**一阶 Euler 算子**（对数域 Euler-Lagrange 偏导）::

    L_i(c) = c_i · ∂_{c_i} ln A_k(c)   ∈ ℂ

分解为 f/g 两个实值分量::

    L_i^f := Re(L_i)              = c_i · ∂_{c_i} f_k
    L_i^g := -Im(L_i) / ω_k       = c_i · ∂_{c_i} g_k

**二阶 Euler 算子**（修正 Itô 交叉项）::

    Ξ_{ij}(c) = c_i c_j ∂²_{c_i c_j} ln A_k  -  L_i(c)·L_j(c)   ∈ ℂ

同样分解为 f/g 两个实值分量。

SINDy-PI 隐式方程识别
-----------------------
将所有算子排列成特征矩阵 Θ ∈ ℝ^{N×J}，对 f 和 g 子空间分别求
SVD 最小奇异值对应的右奇异向量 ξ，满足隐式约束 Θ ξ ≈ 0。
"""
from __future__ import annotations

import re
import numpy as np
from scipy.linalg import svd

from .types import DiscoveryConfig


def pretty_name(name: str) -> str:
    """将内部算子标识符转换为简洁的数学符号字符串（供 GUI / LaTeX 使用）。"""
    if name == "ln_f":
        return "ln_f"
    if name == "g":
        return "g"

    m = re.match(r"L1_c(\d+)_([fg])", name)
    if m:
        ci, fg = m.groups()
        if fg == "f":
            return f"c{ci}d_ln_f/dc{ci}"
        else:
            return f"c{ci}d_g/dc{ci}"
    m2 = re.match(r"Xi2_c(\d+)c(\d+)_([fg])", name)
    if m2:
        i, j, fg = m2.groups()
        if i == j:
            # 对角项为完整二阶 Euler 算子：c_i ∂_{c_i}(c_i ∂_{c_i} f)
            # = c_i² ∂²f/∂c_i² + c_i ∂f/∂c_i（含一阶修正项）
            return f"c{i}d(c{i}d_{fg}/dc{i})/dc{i}"
        else:
            return f"c{i}c{j}d2_{fg}/dc{i}dc{j}"
    # fallback 保留原名
    return name


def construct_pure_library(
    d_hat: np.ndarray,
    dD_dc: np.ndarray,
    d2D_dc2: np.ndarray,
    omega: np.ndarray,
    c: np.ndarray,
    config: DiscoveryConfig,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """构建 Euler 算子特征库（SVD 谱基投影，强形式）。

    算法步骤
    --------
    **步骤 1：SVD 谱基**

        D̂ ≈ U Σ Vᵀ，取前 K 列 → 谱基 Phi = Vt[:K, :] ∈ ℂ^{K×P}

    **步骤 2：投影系数**

        A(c) = D̂(c, ·) · Phi† = D̂ @ Phi.conj().T,   A ∈ ℂ^{N×K}

    **步骤 3：有效频率**

        ω_k = Re(diag(Phi diag(ω) Phi†))   （谱能量加权均值频率）

    **步骤 4：Euler 算子**

        L_i = (∂D̂/∂c_i @ Phi†) ⊘ A   （按元素复数除法，大振幅处才可靠）
        Ξ_{ij} = (∂²D̂/(∂c_i∂c_j) @ Phi†) ⊘ A  -  L_i · L_j

    **步骤 5：f/g 分离**

        算子复值 → 实值库条目：
            f-分量 = Re(·)
            g-分量 = -Im(·) / ω_k   （归一化相位分量）

    特征库结构（f 和 g 子空间严格分离，确保 J_f + J_g == J）::

        f 子空间: ln_f, ln_f^2, L1_c{j}_f, Xi2_c{i}c{j}_f
        g 子空间: g,    g^2,    L1_c{j}_g, Xi2_c{i}c{j}_g

    .. warning::
        库中**不得**加入 f×g 交叉项（如 ``ln_f*g``）。交叉项既不属于 f 子空间
        也不属于 g 子空间，会导致 J_f + J_g < J（库内出现"孤立项"），
        进而使 ``pipeline.py`` 中按键名查找填入 Xi 的循环漏填该位置，
        令 Xi 部分列永远为零，破坏系数张量完整性。

    Parameters
    ----------
    d_hat   : (N, P)
    dD_dc   : (N, n_controls, P)   — 一阶偏导 ∂D̂/∂c_j
    d2D_dc2 : (N, n_controls, n_controls, P)   — 二阶偏导 ∂²D̂/(∂c_i∂c_j)
    omega   : (P,)  归一化频率轴 ω ∈ [0, 1]
    c       : (N, n_controls)   — 控制变量矩阵
    config  : DiscoveryConfig

    Returns
    -------
    library        : dict[str, ndarray(N, K)]
    spectral_basis : ndarray (K, P)
    A              : ndarray (N, K)   投影系数
    omega_means    : ndarray (K,)     各谱基有效频率
    """
    n_samples, n_freq = d_hat.shape
    n_controls = c.shape[1]

    # ── 步骤 1/2/3：SVD → 谱基 P → 投影系数 A → 有效频率 ω_k ──────────────
    U, s, Vt = svd(d_hat, full_matrices=False)

    k_max = int(config.k_max)
    energy = np.cumsum(s**2) / np.sum(s**2)
    k_eff = np.searchsorted(energy, config.rank_energy_threshold) + 1
    k_eff = min(k_eff, k_max)
    if config.k_mode == "fixed":
        k_eff = int(config.k_value)

    spectral_basis = Vt[:k_eff, :]                               # (K, P)
    A = d_hat @ spectral_basis.conj().T                          # (N, K)
    omega_means = np.real(
        np.diag(spectral_basis @ np.diag(omega) @ spectral_basis.conj().T)
    )                                                             # (K,)

    # ── 步骤 4：Euler 算子 L_i 和 Ξ_{ij} ────────────────────────────────
    # dA[:,j,:] = (∂D̂/∂c_j) @ Phi†
    dA = np.zeros((n_samples, n_controls, k_eff), dtype=complex)
    for j in range(n_controls):
        dA[:, j, :] = dD_dc[:, j, :] @ spectral_basis.conj().T

    # d2A[:,i,j,:] = (∂²D̂/∂c_i∂c_j) @ Phi†
    d2A = np.zeros((n_samples, n_controls, n_controls, k_eff), dtype=complex)
    for i in range(n_controls):
        for j in range(n_controls):
            d2A[:, i, j, :] = d2D_dc2[:, i, j, :] @ spectral_basis.conj().T

    # 屏蔽极小 A 避免数值爆炸
    weights = np.abs(A)
    mask = weights > 1e-9 * np.max(weights)

    # alpha_i = ∂ ln A_k / ∂c_i（无 c 缩放的一阶对数导数）
    alpha = np.zeros_like(dA)
    L1_mask = mask[:, None, :].repeat(n_controls, axis=1)
    A_exp1 = A[:, None, :].repeat(n_controls, axis=1)
    alpha[L1_mask] = dA[L1_mask] / A_exp1[L1_mask]

    # beta_ij = ∂² ln A_k / ∂c_i ∂c_j（无 c 缩放的二阶对数导数）
    beta = np.zeros_like(d2A)
    term1 = np.zeros_like(d2A)
    term1_mask = mask[:, None, None, :].repeat(n_controls, axis=1).repeat(n_controls, axis=2)
    A_exp2 = A[:, None, None, :].repeat(n_controls, axis=1).repeat(n_controls, axis=2)
    term1[term1_mask] = d2A[term1_mask] / A_exp2[term1_mask]
    for i in range(n_controls):
        for j in range(n_controls):
            beta[:, i, j, :] = term1[:, i, j, :] - alpha[:, i, :] * alpha[:, j, :]

    # 一阶 Euler 算子：L_i = c_i * ∂ ln A_k / ∂c_i（含 c_i 缩放）
    L1 = np.zeros_like(dA)
    for j in range(n_controls):
        L1[:, j, :] = c[:, j, np.newaxis] * alpha[:, j, :]

    # 二阶 Euler 算子（1.md 公式）：
    #   对角 Ξ_{ii} = c_i² β_{ii} + L_i  = c_i ∂_{c_i}(c_i ∂_{c_i} ln A_k)
    #   非对角 Ξ_{ij} = c_i c_j β_{ij}    = c_i c_j ∂² ln A_k / ∂c_i ∂c_j
    Xi2 = np.zeros_like(d2A)
    for i in range(n_controls):
        for j in range(n_controls):
            if i == j:
                Xi2[:, i, j, :] = (
                    c[:, i, np.newaxis] ** 2 * beta[:, i, j, :]
                    + L1[:, i, :]
                )
            else:
                Xi2[:, i, j, :] = (
                    c[:, i, np.newaxis] * c[:, j, np.newaxis] * beta[:, i, j, :]
                )

    # ── 步骤 5：f/g 分离 → 实值算子库（f 和 g 子空间严格不相交） ──────────
    library: dict[str, np.ndarray] = {}

    # 零阶项
    ln_f = np.real(np.log(A + 1e-12))                            # ln|A|
    g    = -np.imag(np.log(A + 1e-12)) / (omega_means + 1e-9)   # -arg(A)/ω

    library["ln_f"]   = ln_f           # f 子空间
    library["g"]      = g              # g 子空间
    library["ln_f^2"] = ln_f ** 2     # f 子空间
    library["g^2"]    = g ** 2        # g 子空间
    # ⚠ 不加 ln_f*g 交叉项：若加入，该项不属于 f 也不属于 g 子空间，
    #   导致 J_f + J_g < J，pipeline.py 中按 key 查找填入 Xi 的循环会
    #   漏填该项对应位置，使 Xi 部分列永远为零。保持严格二分以确保 J_f + J_g == J。

    # 一阶 Euler 算子
    for j in range(n_controls):
        term = L1[:, j, :]
        library[f"L1_c{j+1}_f"] = np.real(term)
        library[f"L1_c{j+1}_g"] = -np.imag(term) / (omega_means + 1e-9)

    # 二阶 Euler 算子
    for i in range(n_controls):
        for j in range(n_controls):
            val = Xi2[:, i, j, :]
            library[f"Xi2_c{i+1}c{j+1}_f"] = np.real(val)
            library[f"Xi2_c{i+1}c{j+1}_g"] = -np.imag(val) / (omega_means + 1e-9)

    return library, spectral_basis, A, omega_means


def solve_nullspace(
    library: dict[str, np.ndarray],
) -> list[dict[str, tuple[np.ndarray, list[str]]]]:
    """对每个谱分量独立求解 f/g 子空间中的 SINDy-PI 隐式约束。

    数学原理
    --------
    对 f 子空间（含所有 f-分量条目），构建特征矩阵 X_f ∈ ℝ^{N×J_f}。
    SVD 最小奇异值对应的右奇异向量 ξ_f 满足隐式方程::

        X_f ξ_f ≈ 0

    g 子空间同理。

    过滤规则
    --------
    * f 子空间：键名包含 "_f"，或键名为 "ln_f"、"ln_f^2"。
    * g 子空间：键名包含 "_g"，或键名为 "g"、"g^2"。

    (construct_pure_library 保证 J_f + J_g == J，无孤立项。)

    Returns
    -------
    component_models : list[dict], length K
        每项格式：{"f": (coefs_array, key_list), "g": (coefs_array, key_list)}
    """
    lib_f = {k: v for k, v in library.items()
             if "_f" in k or k in ("ln_f", "ln_f^2")}
    lib_g = {k: v for k, v in library.items()
             if "_g" in k or k in ("g", "g^2")}

    k_eff = list(library.values())[0].shape[1]   # 组分数 K
    component_models: list[dict] = []

    for comp_idx in range(k_eff):
        results: dict[str, tuple[np.ndarray, list[str]]] = {}
        for subspace_name, sub_lib in (("f", lib_f), ("g", lib_g)):
            if not sub_lib:
                continue
            keys = sorted(sub_lib.keys())
            J = len(keys)
            N = sub_lib[keys[0]].shape[0]

            # 特征矩阵 X ∈ ℝ^{N×J}（取第 comp_idx 列）
            X = np.zeros((N, J))
            for i, k in enumerate(keys):
                X[:, i] = sub_lib[k][:, comp_idx]

            # 列归一化；剔除零列
            norms = np.linalg.norm(X, axis=0)
            valid = norms > 1e-9
            X_norm = X[:, valid]

            if X_norm.shape[1] > 1:
                _, _, Vt = svd(X_norm, full_matrices=False)
                coefs_norm = Vt[-1, :]          # 最小奇异值方向

                coefs = np.zeros(J, dtype=complex)
                coefs[valid] = coefs_norm / norms[valid]   # 还原到未归一化坐标

                # 最大系数归一，便于比较量级
                max_idx = np.argmax(np.abs(coefs))
                if np.abs(coefs[max_idx]) > 0:
                    coefs /= coefs[max_idx]

                results[subspace_name] = (coefs, keys)
        component_models.append(results)
    return component_models


def build_direct_euler_library(
    d_hat: np.ndarray,
    dD_dc: np.ndarray,
    d2D_dc2: np.ndarray,
    omega: np.ndarray,
    c: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """无 SVD 直接 Euler 算子库（1.md 完整路径）。

    算法步骤（对应 1.md 第 2–3 节）
    ----------------------------------
    **步骤 1：逐元素 Euler 算子（无子空间降维）**

    对每个控制维 c_i，直接在全频域张量 D̂(c, ω) ∈ ℂ^{N×P} 上计算：

        alpha_i(c, ω) = ∂D̂/∂c_i(c, ω) / D̂(c, ω)          （一阶对数导数）
        L_i(c, ω)     = c_i · alpha_i(c, ω)                 （一阶 Euler 算子）

    二阶（按 1.md 公式）：

        beta_ij = ∂²D̂/(∂c_i∂c_j) / D̂ - alpha_i · alpha_j  （二阶对数导数）
        Ξ_{ii}  = c_i² · beta_ii + L_i                      （对角 Euler 算子）
        Ξ_{ij}  = c_i · c_j · beta_ij        (i ≠ j)        （非对角 Euler 算子）

    **步骤 2：W(ω) 线性拟合分离 f/g**

    对每个算子 A(c, ω)（N×P 复矩阵），在每个 c 点上跨所有 ω 作最小二乘拟合：

        A(c_n, ω) ≈ f(c_n) + (−i·ω) · g(c_n)
        W = [[1, −iω₁], [1, −iω₂], ..., [1, −iωₚ]]ᵀ  ∈ ℂ^{P×2}
        [f(c_n), g(c_n)]ᵀ = pinv(W) · A(c_n, :)

    取 f = Re 分量，g = Re 分量（拟合保证两者均近实数）。

    **步骤 3：返回库**

    库条目形状为 (N, 1)，与 solve_nullspace 兼容（K=1 全局方程）。

    Parameters
    ----------
    d_hat   : (N, P)   频域观测数据（rfft）
    dD_dc   : (N, d, P)   一阶偏导 ∂D̂/∂c_j
    d2D_dc2 : (N, d, d, P)   二阶偏导 ∂²D̂/(∂c_i∂c_j)
    omega   : (P,)   归一化频率轴 ω ∈ [0, 1]
    c       : (N, d)   控制变量矩阵

    Returns
    -------
    library       : dict[str, ndarray(N, 1)]   算子库，键名与 SVD 路径兼容
    spectral_basis : ndarray (1, P)             参考谱（零锚点均值）
    A              : ndarray (N, 1)             响应系数（exp(f_0)）
    omega_means    : ndarray (1,)               参考有效频率
    """
    n_samples, n_freq = d_hat.shape
    n_controls = c.shape[1]

    # W(ω) = [1, -iω]ᵀ，用于跨频率拟合 f 和 g
    W = np.stack([np.ones(n_freq), -1j * omega], axis=1)    # (P, 2)
    # pinv_W: (2, P)，使得 [f; g] = pinv_W @ A(c, :)
    pinv_W = np.linalg.pinv(W)                              # (2, P)

    def _fg_fit(op: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """对 (N, P) 复算子矩阵跨 ω 做 W(ω) 拟合，返回 f(N,) 和 g(N,)。"""
        fg = op @ pinv_W.T    # (N, 2)
        return np.real(fg[:, 0]), np.real(fg[:, 1])

    # ── 步骤 1：逐元素对数导数（屏蔽极小 D̂ 避免数值爆炸）─────────────────
    eps_abs = 1e-9 * np.max(np.abs(d_hat))
    dhat_safe = d_hat.copy()
    dhat_safe[np.abs(dhat_safe) < eps_abs] = eps_abs * (1.0 + 0j)

    # alpha_i(c, ω) = ∂D̂/∂c_i / D̂（N, d, P）
    alpha = dD_dc / dhat_safe[:, np.newaxis, :]

    # beta_ij(c, ω) = ∂²D̂/(∂c_i∂c_j) / D̂ - alpha_i · alpha_j（N, d, d, P）
    beta = (d2D_dc2 / dhat_safe[:, np.newaxis, np.newaxis, :]
            - alpha[:, :, np.newaxis, :] * alpha[:, np.newaxis, :, :])

    # 一阶 Euler 算子：L_i = c_i · alpha_i（N, d, P）
    L_euler = c[:, :, np.newaxis] * alpha      # broadcast: (N, d, P)

    # 二阶 Euler 算子（N, d, d, P）
    Xi2_direct = np.zeros_like(d2D_dc2)
    for i in range(n_controls):
        for j in range(n_controls):
            if i == j:
                Xi2_direct[:, i, j, :] = (
                    c[:, i, np.newaxis] ** 2 * beta[:, i, j, :]
                    + L_euler[:, i, :]
                )
            else:
                Xi2_direct[:, i, j, :] = (
                    c[:, i, np.newaxis] * c[:, j, np.newaxis] * beta[:, i, j, :]
                )

    # ── 步骤 2：W(ω) 拟合 → f/g 标量场 ─────────────────────────────────────
    library: dict[str, np.ndarray] = {}

    # 零阶项：对 ln D̂ 做 W(ω) 拟合
    ln_dhat = np.log(np.abs(dhat_safe) + 1e-12) + 1j * np.angle(dhat_safe)  # (N, P)
    f0, g0 = _fg_fit(ln_dhat)
    library["ln_f"]   = f0[:, np.newaxis]           # (N, 1)
    library["g"]      = g0[:, np.newaxis]            # (N, 1)
    library["ln_f^2"] = f0[:, np.newaxis] ** 2
    library["g^2"]    = g0[:, np.newaxis] ** 2

    # 一阶 Euler 算子
    for j in range(n_controls):
        fj, gj = _fg_fit(L_euler[:, j, :])
        library[f"L1_c{j+1}_f"] = fj[:, np.newaxis]     # (N, 1)
        library[f"L1_c{j+1}_g"] = gj[:, np.newaxis]

    # 二阶 Euler 算子
    for i in range(n_controls):
        for j in range(n_controls):
            fi, gi = _fg_fit(Xi2_direct[:, i, j, :])
            library[f"Xi2_c{i+1}c{j+1}_f"] = fi[:, np.newaxis]
            library[f"Xi2_c{i+1}c{j+1}_g"] = gi[:, np.newaxis]

    # ── 步骤 3：参考谱与响应系数 ──────────────────────────────────────────────
    # 参考谱：D̂ 沿样本轴均值（作为单一"组分"谱基）
    spectral_basis = np.mean(d_hat, axis=0, keepdims=True)   # (1, P)

    # 响应系数：用 W(ω) 拟合的对数幅度重建
    A = np.exp(f0)[:, np.newaxis].astype(complex)            # (N, 1)

    # 有效频率
    omega_means = np.array([np.real(
        np.dot(spectral_basis[0], omega * spectral_basis[0].conj())
        / (np.dot(spectral_basis[0], spectral_basis[0].conj()) + 1e-12)
    )])                                                       # (1,)

    return library, spectral_basis, A, omega_means
