from __future__ import annotations

import numpy as np
from typing import Tuple, Dict

from .types import DiscoveryConfig


def construct_inverse_library(
    d_hat: np.ndarray,
    d_d_c: np.ndarray,
    d2_d_c: np.ndarray,
    omega: np.ndarray,
    factors: np.ndarray,
    config: DiscoveryConfig,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    r"""Build operator library using pseudoinverse formulas.

    This routine uses expressions such as D^\dagger \partial_c D
    without performing an SVD basis projection.  The returned dictionary
    has the same shape convention as construct_pure_library so downstream
    code can handle either pipeline interchangeably.

    Results dimensions:
    library[name] is shape (N, K) where K is chosen by config.k_max or
    config.k_value.  basis is merely a placeholder (original d_hat may be
    returned).  A may equal d_hat or its projection.  omega_means behaves
    as in construct_pure_library.
    """
    # 伪逆算子
    # 对矩阵 d_hat (N x M) 取 Moore-Penrose 伪逆
    D_dag = np.linalg.pinv(d_hat)

    n_samples, n_freq = d_hat.shape
    n_controls = factors.shape[1]

    # 这里我们暂时仍保留 K=rank 作为列数，方便与现有求 nullspace 兼容
    # 确定 K 与 config 逻辑一致
    k_max = int(config.k_max)
    k_eff = min(k_max, n_freq)
    if config.k_mode == "fixed":
        k_eff = int(config.k_value)

    # operator 值的直接计算；与原有 L1/Xi2 结构类似，但未投影
    library: Dict[str, np.ndarray] = {}

    # 0阶项 - 截断到 k_eff 使得所有算子具有一致的列数 (N_samples, k_eff)
    d_hat_k = d_hat[:, :k_eff]
    omega_k = omega[:k_eff]
    ln_f = np.real(np.log(d_hat_k + 1e-12))
    g = -np.imag(np.log(d_hat_k + 1e-12)) / (omega_k + 1e-9)
    library["ln_f"] = ln_f
    library["g"] = g

    # 计算一阶伪逆算子 L1_cj = D^\dagger (d/dc_j D)
    # D_dag 形状为 (n_freq, n_samples)，d_d_c[:, j, :] 形状为 (n_samples, n_freq)
    # 矩阵乘积：(n_freq, n_samples) @ (n_samples, n_freq) = (n_freq, n_freq)
    for j in range(n_controls):
        term = D_dag @ d_d_c[:, j, :]  # shape (n_freq, n_freq)
        # 这里只取对角线作为每个频率的响应
        diag = np.diag(term)[:k_eff]
        # 为兼容，重复成 (N,k_eff) 形状
        lib_j = np.tile(diag, (n_samples, 1))
        library[f"L1_c{j+1}_f"] = np.real(lib_j)
        library[f"L1_c{j+1}_g"] = -np.imag(lib_j) / (omega[:k_eff] + 1e-9)

    # 二阶伪逆算子类似
    for i in range(n_controls):
        for j in range(n_controls):
            term = D_dag @ d2_d_c[:, i, j, :]  # shape (n_freq, n_freq)
            diag = np.diag(term)[:k_eff]
            lib_f = np.tile(np.real(diag), (n_samples, 1))
            lib_g = np.tile(-np.imag(diag) / (omega[:k_eff] + 1e-9), (n_samples, 1))
            library[f"Xi2_c{i+1}c{j+1}_f"] = lib_f
            library[f"Xi2_c{i+1}c{j+1}_g"] = lib_g

    # 计算 SVD 谱基（仅用于构造输出，不影响算子库的伪逆构造逻辑）
    # 这使得 basis 形状与 construct_pure_library 一致，为 (k_eff, n_freq)
    from scipy.linalg import svd as _svd
    _, _, Vt = _svd(d_hat, full_matrices=False)
    spectral_basis = Vt[:k_eff, :]              # (k_eff, n_freq)
    A_proj = d_hat @ spectral_basis.conj().T    # (N, k_eff)
    omega_means = np.real(
        np.diag(spectral_basis @ np.diag(omega) @ spectral_basis.conj().T)
    )                                            # (k_eff,)

    basis = spectral_basis
    A = A_proj

    return library, basis, A, omega_means


# ----- weak-form helpers (placeholders) -----

from .preprocess import _detect_cartesian_grid


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
