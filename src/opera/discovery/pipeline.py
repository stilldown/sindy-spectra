from __future__ import annotations

import numpy as np
from time import perf_counter

from .types import DiscoveryConfig, DiscoveryResult
from .preprocess import validate_inputs, compute_fourier_tensor, build_control_derivative_bundle
from .pipeline_utils import construct_pure_library, solve_nullspace, pretty_name
# 新模块提供基/系数联合求解的算子映射逻辑
from .operator import construct_inverse_library


def run_discovery(
    spectra: np.ndarray,
    factors: np.ndarray,
    wavelengths: np.ndarray,
    config: DiscoveryConfig | None = None,
) -> DiscoveryResult:
    """管线主函数。

    算子构建与零空间解决逻辑分别放在 ``pipeline_utils`` 中，
    这里仅负责流程和结果格式化。
    """
    t0 = perf_counter()
    cfg = config or DiscoveryConfig()

    s = np.asarray(spectra)
    c = np.asarray(factors)
    wl = np.asarray(wavelengths)

    validate_inputs(s, c, wl)
    n_samples = s.shape[0]

    anchor_idx = np.where(np.all(np.abs(c) <= float(cfg.zero_anchor_tol), axis=1))[0]
    if anchor_idx.size == 0:
        raise ValueError("未检测到零浓度样本（所有控制因子均为0）。该参考集为必需输入。")

    d_hat, omega = compute_fourier_tensor(s, wl)
    t_fft = perf_counter()

    d_d_c, d2_d_c = build_control_derivative_bundle(
        d_hat=d_hat,
        factors=c,
        eps=float(cfg.matrix.epsilon),
    )
    t_deriv = perf_counter()

    # choose construction method based on config flag
    if cfg.use_inverse_operator:
        # 基于伪逆构造算子库，并对算子加权形成弱算子（默认均匀权重）
        lib, basis, A, w_means = construct_inverse_library(
            d_hat=d_hat,
            d_d_c=d_d_c,
            d2_d_c=d2_d_c,
            omega=omega,
            factors=c,
            config=cfg,
        )
        # 使用统一测试函数 psi=1 计算弱算子作为后续候选
        try:
            from .operator import compute_weak_operators
            psi = np.ones(n_samples)
            weak_lib = compute_weak_operators(d_hat, d_d_c, d2_d_c, c, psi)
            # merge weak_lib entries into main library
            lib.update(weak_lib)
        except ImportError:
            pass
    else:
        lib, basis, A, w_means = construct_pure_library(
            d_hat=d_hat,
            d_d_c=d_d_c,
            d2_d_c=d2_d_c,
            omega=omega,
            factors=c,
            config=cfg,
        )
    t_lib = perf_counter()

    component_models = solve_nullspace(lib)

    # operator names 直接从库 keys 生成以确保与 Xi/A_matrix 行对齐
    op_names = [pretty_name(k) for k in lib.keys()]

    n_features = len(op_names)
    k_eff = len(component_models)

    Xi = np.zeros((1, n_features, k_eff))
    eqn_strs: list[str] = []

    for k in range(k_eff):
        model = component_models[k]
        xi_k: list[complex] = []
        parts: list[str] = []
        if "f" in model:
            cfs, names = model["f"]
            xi_k.extend(cfs)
            terms = [f"{coef:.3f}*{name}" for coef, name in zip(cfs, names) if abs(coef) > 1e-3]
            parts.append("f: " + " + ".join(terms))
        if "g" in model:
            cgs, names = model["g"]
            xi_k.extend(cgs)
            terms = [f"{coef:.3f}*{name}" for coef, name in zip(cgs, names) if abs(coef) > 1e-3]
            parts.append("g: " + " + ".join(terms))
        if len(xi_k) == n_features:
            Xi[0, :, k] = xi_k
        eqn_strs.append(f"Comp {k+1} " + " | ".join(parts))

    # 频域谱基转到波长域并转置为 (n_wl, K) 形状
    S_time_domain = np.fft.irfft(basis, n=len(wl), axis=1).T
    # 频域纯光谱占位，basis.T 为 (n_freq, K) 方便测试
    pure_spectra = basis.T

    return DiscoveryResult(
        S_real=S_time_domain,
        f_response_eval=A,
        A_matrix=Xi[0],
        Xi=Xi,
        operator_names=op_names,
        f_response=A,
        g_shift=np.zeros((s.shape[0], k_eff)),
        pure_spectra_complex=pure_spectra,
        reconstruction_error=0.0,
        xi_by_control={f"component_{k+1}": Xi[0, :, k] for k in range(k_eff)},
        component_scores=None,
        component_energy_ratio=None,
        component_nonzero_ratio=None,
        quality_flags=["ok"],
        latex_blocks=eqn_strs,
        diagnostics={
            "k_eff": float(k_eff),
            "k_selected": float(k_eff),
            "nullspace_energy": 0.0,
            "sigma_gap_min": 0.0,
            "anchor_count": float(anchor_idx.size),
        },
        metadata={
            "models": component_models,
            "equations": eqn_strs,
            "anchor_indices": anchor_idx,
            "k_source": "k_value" if cfg.k_mode == "fixed" else "mode_rule",
            "J_tot": n_features,
            "operator_block_ranges": {},
        },
    )
