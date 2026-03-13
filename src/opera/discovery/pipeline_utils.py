from __future__ import annotations

import re
import numpy as np
from scipy.linalg import svd

from .types import DiscoveryConfig


def pretty_name(name: str) -> str:
    """统一将内部算子标识转换为简洁一致的样式。

    这个工具由管线和 GUI 共享，用于把底层名称映射到最终展示结果。
    """
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
            return f"c{i}^2d2_{fg}/dc{i}^2"
        else:
            return f"c{i}c{j}d2_{fg}/dc{i}dc{j}"
    # fallback 保留原名
    return name



def construct_pure_library(
    d_hat: np.ndarray,
    d_d_c: np.ndarray,
    d2_d_c: np.ndarray,
    omega: np.ndarray,
    factors: np.ndarray,
    config: DiscoveryConfig,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """按照“纯 Euler”逻辑构建算子库。

    输入为频域数据及其一阶/二阶控制比例导数。
    返回 ``(library, spectral_basis, A, omega_means)``：

    * ``library`` 是一个字典，键为算子名称，值为(N,K) 数组样本；
    * ``spectral_basis`` 是频域联合谱基 (K, M)；
    * ``A`` 是将数据投影到谱基后的系数 (N, K)；
    * ``omega_means`` 是每个谱基的有效频率，用于 f/g 解耦。
    """
    n_samples, n_freq = d_hat.shape
    n_controls = factors.shape[1]

    # 通过 SVD 提取谱基
    U, s, Vt = svd(d_hat, full_matrices=False)

    k_max = int(config.k_max)
    energy = np.cumsum(s**2) / np.sum(s**2)
    k_eff = np.searchsorted(energy, config.rank_energy_threshold) + 1
    k_eff = min(k_eff, k_max)
    if config.k_mode == "fixed":
        k_eff = int(config.k_value)

    spectral_basis = Vt[:k_eff, :]

    # 投影得到分量系数 A
    A = d_hat @ spectral_basis.conj().T

    omega_means = np.real(
        np.diag(spectral_basis @ np.diag(omega) @ spectral_basis.conj().T)
    )

    # 计算各算子样本值
    dA = np.zeros((n_samples, n_controls, k_eff), dtype=complex)
    for j in range(n_controls):
        dA[:, j, :] = d_d_c[:, j, :] @ spectral_basis.conj().T

    d2A = np.zeros((n_samples, n_controls, n_controls, k_eff), dtype=complex)
    for i in range(n_controls):
        for j in range(n_controls):
            d2A[:, i, j, :] = d2_d_c[:, i, j, :] @ spectral_basis.conj().T

    # 基于 A 和导数构造 L1、Xi2
    weights = np.abs(A)
    mask = weights > 1e-9 * np.max(weights)

    L1 = np.zeros_like(dA)
    Xi2 = np.zeros_like(d2A)

    L1_mask = mask[:, None, :].repeat(n_controls, axis=1)
    A_expanded = A[:, None, :].repeat(n_controls, axis=1)
    L1[L1_mask] = dA[L1_mask] / A_expanded[L1_mask]

    term1_mask = mask[:, None, None, :].repeat(n_controls, axis=1).repeat(
        n_controls, axis=2
    )
    A_expanded_2 = A[:, None, None, :].repeat(n_controls, axis=1).repeat(
        n_controls, axis=2
    )

    term1 = np.zeros_like(d2A)
    term1[term1_mask] = d2A[term1_mask] / A_expanded_2[term1_mask]

    for i in range(n_controls):
        for j in range(n_controls):
            Xi2[:, i, j, :] = term1[:, i, j, :] - L1[:, i, :] * L1[:, j, :]

    # f/g 分离算子
    library: dict[str, np.ndarray] = {}

    # 0阶项
    ln_f = np.real(np.log(A + 1e-12))
    g = -np.imag(np.log(A + 1e-12)) / (omega_means + 1e-9)

    library["ln_f"] = ln_f
    library["g"] = g
    library["ln_f^2"] = ln_f**2
    library["g^2"] = g**2
    library["ln_f*g"] = ln_f * g

    # 一阶项 L1
    for j in range(n_controls):
        term = L1[:, j, :]
        library[f"L1_c{j+1}_f"] = np.real(term)
        library[f"L1_c{j+1}_g"] = -np.imag(term) / (omega_means + 1e-9)

    # 二阶 Xi2
    for i in range(n_controls):
        for j in range(n_controls):
            val = Xi2[:, i, j, :]
            term_f = np.real(val)
            term_g = -np.imag(val) / (omega_means + 1e-9)
            library[f"Xi2_c{i+1}c{j+1}_f"] = term_f
            library[f"Xi2_c{i+1}c{j+1}_g"] = term_g

    return library, spectral_basis, A, omega_means


def solve_nullspace(
    library: dict[str, np.ndarray]
) -> list[dict[str, tuple[np.ndarray, list[str]]]]:
    """对每个分量单独求解 f/g 空间的零空间系数。

    输出格式为

        [ {'f': (coeffs, names), 'g': (coeffs, names)}, ... ]
    """
    lib_f = {k: v for k, v in library.items() if "_f" in k or k == "ln_f" or k == "ln_f^2"}
    lib_g = {k: v for k, v in library.items() if "_g" in k or k == "g" or k == "g^2"}

    n_k = list(library.values())[0].shape[1]
    component_models: list[dict] = []

    for comp_idx in range(n_k):
        results: dict[str, tuple[np.ndarray, list[str]]] = {}
        for name, lib in [("f", lib_f), ("g", lib_g)]:
            if not lib:
                continue
            keys = sorted(lib.keys())
            n_features = len(keys)
            n_samples = lib[keys[0]].shape[0]

            X = np.zeros((n_samples, n_features))
            for i, k in enumerate(keys):
                X[:, i] = lib[k][:, comp_idx]

            norms = np.linalg.norm(X, axis=0)
            valid = norms > 1e-9
            X_sub = X[:, valid]

            if X_sub.shape[1] > 1:
                U, S, Vt = svd(X_sub, full_matrices=False)
                coefs_sub = Vt[-1, :]

                coefs = np.zeros(n_features, dtype=complex)
                coefs[valid] = coefs_sub / norms[valid]

                max_idx = np.argmax(np.abs(coefs))
                if np.abs(coefs[max_idx]) > 0:
                    coefs /= coefs[max_idx]

                results[name] = (coefs, keys)
        component_models.append(results)
    return component_models
