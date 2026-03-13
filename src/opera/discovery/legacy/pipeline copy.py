from __future__ import annotations

import numpy as np
from time import perf_counter
from scipy.optimize import least_squares

from .types import DiscoveryConfig, DiscoveryResult
from .preprocess import validate_inputs, compute_fourier_tensor, build_control_derivative_bundle
from .library import build_observable_library
from .factorization import select_component_count, find_joint_nullspace, extract_sparse_physical_coefficients, calibrate_pure_spectra_once
from .decoder import decode_physical_manifolds
from .symbolic import build_latex_blocks_from_xi


def _resolve_k(cfg: DiscoveryConfig, matrix: np.ndarray) -> int:
    # 约定：fixed 模式严格仅使用 k_value，不再由 legacy max_components 覆盖。
    k_value = int(cfg.k_value)

    return select_component_count(
        matrix=matrix,
        mode=cfg.k_mode,
        k_value=k_value,
        k_max=int(cfg.k_max),
        energy_threshold=float(cfg.rank_energy_threshold),
    )


def _operator_blocks_from_tags(operator_tags: list[str]) -> dict[str, tuple[int, int]]:
    """按可观测算子族的 tags 直接划分 block，返回 {block_name: (start, end)}，end 为开区间。
    由于同一个类的 tags 可能分散（这取决于 library 的拼接顺序），我们需要收集最小和最大索引。
    """
    groups = {}
    for i, tag in enumerate(operator_tags):
        if tag not in groups:
            groups[tag] = []
        groups[tag].append(i)
    
    out = {}
    for k, idx in groups.items():
        out[k] = (min(idx), max(idx) + 1)
    return out


def _xi_matrix_to_tensor(
    xi_matrix: np.ndarray,
    block_ranges: dict[str, tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """将算子系数矩阵按分块规则重排为张量。

    返回:
    - xi_tensor: (n_blocks, J_max, K) 按块填充的系数张量
    - mask: (n_blocks, J_max) 有效项掩码
    - block_names: 块名称列表（与 xi_tensor 第一维对齐）
    """
    x = np.asarray(xi_matrix)
    items = sorted(block_ranges.items(), key=lambda kv: kv[1][0])
    if len(items) == 0:
        # 退化为单块
        return x.reshape(1, x.shape[0], x.shape[1]), np.ones((1, x.shape[0]), dtype=bool), ["all"]

    block_names = [name for name, _ in items]
    lengths = [int(end - start) for _, (start, end) in items]
    j_max = max(lengths)
    n_blocks = len(items)
    k_eff = x.shape[1]

    tensor = np.zeros((n_blocks, j_max, k_eff), dtype=x.dtype)
    mask = np.zeros((n_blocks, j_max), dtype=bool)
    for b, (_, (start, end)) in enumerate(items):
        seg = x[start:end, :]
        l = seg.shape[0]
        tensor[b, :l, :] = seg
        mask[b, :l] = True

    return tensor, mask, block_names


def _find_zero_anchor_indices(factors: np.ndarray, tol: float) -> np.ndarray:
    c = np.asarray(factors, dtype=float)
    mask = np.all(np.abs(c) <= float(tol), axis=1)
    return np.where(mask)[0]


def _first_iw_index(operator_names: list[str]) -> int | None:
    for i, name in enumerate(operator_names):
        if name == "iω·D̂(c,ω)":
            return i
    for i, name in enumerate(operator_names):
        if name.startswith("iω_") and "·D̂(c,ω)" in name:
            return i
    return None


def _build_gamma_from_theta(
    theta: np.ndarray,
    n_samples: int,
    n_freq: int,
    sparsity_threshold: float,
    operator_names: list[str],
    pivot_operator: str,
    sparse_iters: int,
    spectral_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """逐频点从观测库 Theta 求解提升矩阵 Γ（最小右奇异向量）。

    对每个频点 n：构造 Theta_n (N, J_tot)，解 min ||Theta_n γ_n||₂，||γ_n||₂=1，
    取最小奇异向量作为 γ_n，并批量归一化。若 pivot 未命中则按列范数回退。
    不再执行稀疏迭代，仅用轻微正则保障可解性。
    """
    th = np.asarray(theta, dtype=complex)
    j_tot = th.shape[1]

    # 还原每个频点的样本块: theta_3d[m, n, j]
    theta_3d = th.reshape(n_samples, n_freq, j_tot)
    
    if pivot_operator in operator_names:
        pivot_idx = operator_names.index(pivot_operator)
    elif "D̂(c,ω)" in operator_names:
        pivot_idx = operator_names.index("D̂(c,ω)")
    else:
        # 回退：选择全局列范数最大的列作为锚点
        pivot_idx = int(np.argmax(np.linalg.norm(th, axis=0)))

    # -------- 逐频枢轴 OLS（无稀疏迭代） --------
    # X 形状: (n_freq, n_samples, j_tot - 1)
    keep = [i for i in range(j_tot) if i != pivot_idx]
    X = theta_3d.transpose(1, 0, 2)[:, :, keep]
    Y = -theta_3d.transpose(1, 0, 2)[:, :, pivot_idx:pivot_idx+1]
    
    # 构造正规方程 Xt X coef = Xt Y
    # Xt 形状: (n_freq, j_tot - 1, n_samples)
    Xt = X.transpose(0, 2, 1).conj()
    A = Xt @ X
    B = Xt @ Y
    
    # 加入微小正则化保证满秩，防止 LinAlgError
    idx = np.arange(len(keep))
    A[:, idx, idx] += 1e-12
    
    # 批量求解
    try:
        coef = np.linalg.solve(A, B)  # (n_freq, j_tot - 1, 1)
    except np.linalg.LinAlgError:
        # 退化情况回退到批量伪逆
        coef = np.linalg.pinv(A) @ B

    # 组装 Gamma，形状 (n_freq, j_tot)
    gamma = np.zeros((n_freq, j_tot), dtype=complex)
    gamma[:, pivot_idx] = 1.0 + 0j
    gamma[:, keep] = coef[:, :, 0]

    # 归一化每一行
    ng = np.linalg.norm(gamma, axis=1, keepdims=True)
    mask = ng[:, 0] > 0
    gamma[mask] = gamma[mask] / ng[mask]

    # 计算残差与支撑集
    # pred 形状: (n_samples, n_freq)
    pred = np.einsum('mnj,nj->mn', theta_3d, gamma)
    
    num = np.linalg.norm(pred, axis=0)
    # 取各频点 Theta_n 的 Frobenius 范数
    den = np.linalg.norm(theta_3d, axis=(0, 2)) + 1e-12
    residuals = num / den
    
    supports = np.sum(np.abs(gamma) > 0, axis=1)

    # 恢复频点幅值尺度
    if spectral_scale is not None:
        gamma = gamma * np.asarray(spectral_scale)[:, None]

    # 还原为 (J_tot, n_freq)
    gamma = gamma.T

    diag = {
        "gamma_linear_residual_mean": float(np.mean(residuals)),
        "gamma_linear_residual_max": float(np.max(residuals)),
        "gamma_sparse_support_mean": float(np.mean(supports)),
        "gamma_sparse_support_max": float(np.max(supports)),
        "gamma_pivot_index": float(pivot_idx),
    }
    return gamma, diag


def _reconstruction_error_from_fg_p(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    f_response: np.ndarray,
    g_shift: np.ndarray,
    p_complex: np.ndarray,
) -> float:
    d = np.asarray(spectra, dtype=float)
    wl = np.asarray(wavelengths, dtype=float)
    n_samples, n_wl = d.shape

    d_wl = float(np.mean(np.diff(wl))) if wl.size > 1 else 1.0
    omega = 2.0 * np.pi * np.fft.fftfreq(n_wl, d=d_wl)

    d_hat_model = np.zeros((n_samples, n_wl), dtype=complex)
    k_eff = p_complex.shape[1]
    for k in range(k_eff):
        a_k = f_response[:, [k]] * np.exp(1j * omega.reshape(1, -1) * g_shift[:, [k]])
        d_hat_model += a_k * p_complex[:, k].reshape(1, -1)

    recon = np.fft.ifft(d_hat_model, axis=1).real
    rel_err = float(np.linalg.norm(recon - d) / (np.linalg.norm(d) + 1e-12))
    return rel_err


def _enforce_hermitian_frequency_spectra(p_freq_complex: np.ndarray) -> np.ndarray:
    """对频域纯谱系数施加厄米对称，保证 IFFT 后为实值光谱。"""
    p = np.asarray(p_freq_complex).copy()
    n_freq = p.shape[0]

    # DC 分量实数化
    p[0, :] = np.real(p[0, :])

    half = n_freq // 2
    for k in range(1, half + (0 if n_freq % 2 == 0 else 1)):
        kn = (-k) % n_freq
        avg = 0.5 * (p[k, :] + np.conj(p[kn, :]))
        p[k, :] = avg
        p[kn, :] = np.conj(avg)

    # 偶数长度下 Nyquist 分量需实数
    if n_freq % 2 == 0:
        p[half, :] = np.real(p[half, :])

    return p


def _lambda_spectra_norm_from_freq(p_freq_complex: np.ndarray) -> float:
    p_sym = _enforce_hermitian_frequency_spectra(p_freq_complex)
    p_lambda = np.fft.ifft(p_sym, axis=0)
    return float(np.linalg.norm(np.real(p_lambda)))


def _estimate_g_shift_from_phase(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    f_response: np.ndarray,
    p_complex: np.ndarray,
    max_abs: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    基于频域相位的非线性最小二乘拟合，估计每个样本/组分的 g_shift。

    模型：d_hat[i](ω) ≈ Σ_k f_i,k · exp(i·ω·g_i,k) · p_k(ω)

    返回 (g_shift, 平均残差范数)。
    """
    d = np.asarray(spectra, dtype=float)
    wl = np.asarray(wavelengths, dtype=float)
    n_samples, n_wl = d.shape
    k_eff = f_response.shape[1]

    d_wl = float(np.mean(np.diff(wl))) if wl.size > 1 else 1.0
    omega = 2.0 * np.pi * np.fft.fftfreq(n_wl, d=d_wl)
    d_hat = np.fft.fft(d, axis=1)

    g_shift = np.zeros((n_samples, k_eff), dtype=float)
    residuals = []
    # bounds for least_squares
    if max_abs is None:
        bounds = (-np.inf, np.inf)
    else:
        bounds = (-float(max_abs), float(max_abs))

    for i in range(n_samples):
        f_row = f_response[i]

        def residual(g_vec: np.ndarray) -> np.ndarray:
            phase = np.exp(1j * omega.reshape(-1, 1) * g_vec.reshape(1, -1))
            model = (f_row.reshape(1, -1) * phase * p_complex).sum(axis=1)
            diff = model - d_hat[i]
            return np.concatenate([diff.real, diff.imag])

        res = least_squares(
            residual,
            x0=np.zeros(k_eff, dtype=float),
            bounds=bounds,
            method="trf",
        )
        g_shift[i] = res.x
        residuals.append(float(np.linalg.norm(res.fun)))

    avg_residual = float(np.mean(residuals)) if residuals else 0.0
    return g_shift, avg_residual


def run_discovery(
    spectra: np.ndarray,
    factors: np.ndarray,
    wavelengths: np.ndarray,
    config: DiscoveryConfig | None = None,
) -> DiscoveryResult:
    t0 = perf_counter()
    cfg = config or DiscoveryConfig()

    s = np.asarray(spectra)
    c = np.asarray(factors)
    wl = np.asarray(wavelengths)

    validate_inputs(s, c, wl)

    # 硬约束：必须存在零浓度参考集
    anchor_idx = _find_zero_anchor_indices(c, tol=cfg.zero_anchor_tol)
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

    theta, operator_names, scales, operator_tags = build_observable_library(
        d_hat,
        d_d_c,
        d2_d_c,
        omega,
        c,
    )
    t_library = perf_counter()

    # 严格代数路径：先提取物理零空间，然后旋转稀疏化
    n_samples, n_freq = d_hat.shape
    
    # 动态评估有效的物理组分数 K
    k_eff = _resolve_k(cfg, theta)

    # 第一步：计算在全量观测张量特征空间下的物理零流形
    v_null, svd_diag = find_joint_nullspace(theta, k_eff)
    
    # 第二步：使用 DEIM (QR-Pivoting) 和 L1 硬阈值进行算子系数量化分离
    xi_matrix_raw = extract_sparse_physical_coefficients(
        V_null=v_null,
        sparsity_threshold=float(cfg.sparsity_threshold),
    )
    
    # 根据之前的算子库缩放比例（scales），重新恢复出具备物理量纲的独立方程矩阵 Xi
    xi = xi_matrix_raw / scales.reshape(-1, 1)

    # 组分波长域复原与 f/g 的提取
    k_eff = xi.shape[1]
    
    # 结合全局网格解析流形
    f_response, g_shift_bar = decode_physical_manifolds(
        xi=xi,
        names=operator_names,
        factors=c,
        anchor_idx=anchor_idx,
    )
    
    # 恢复相移单位
    # 因为 phase operator 是 i * bar_omega * D, g_shift_bar 是 [0, 1] 频域宽度的系数
    # bar_omega 的最大值对应的真实频率上限是 np.pi / d_wl (奈奎斯特频率)
    d_wl = float(np.mean(np.diff(wl))) if wl.size > 1 else 1.0
    omega_max = np.pi / d_wl
    g_shift = g_shift_bar / omega_max

    t_decomp = perf_counter()
    
    # Phase 4 后半段：直接从流形计算光谱
    # 我们知道 D_hat(c, omega) \approx f(c) e^{-i omega g(c)} P(omega)
    # 对于每个频率点 omega_n，求解 n_samples 对 K 的线性超定系统
    
    p_freq_complex = np.zeros((n_freq, k_eff), dtype=complex)
    half_n = (n_freq + 1) // 2
    
    for n in range(half_n):
        # f_response: (N, K), g_shift: (N, K)
        A_n = f_response * np.exp(-1j * omega_max * omega[n] * g_shift)
        # s 是时域 spectra, 我们直接拟合频域 d_hat 的一半频率
        Y_n = d_hat[:, n]
        
        # OLS
        m_mat = A_n.conj().T @ A_n + 1e-12 * np.eye(k_eff)
        rhs_vec = A_n.conj().T @ Y_n
        
        try:
            p = np.linalg.solve(m_mat, rhs_vec)
        except:
            p, *_ = np.linalg.lstsq(m_mat, rhs_vec, rcond=None)
            
        p_freq_complex[n, :] = p
        
    for n in range(half_n, n_freq):
        p_freq_complex[n, :] = np.conj(p_freq_complex[n_freq - n, :])

    diagnostics = {
        "nullspace_energy": svd_diag["nullspace_energy"],
        "sigma_gap_min": svd_diag["sigma_gap_min"],
        "k_selected": float(k_eff),
        "anchor_count": float(anchor_idx.size),
        "gamma_shape": theta.shape,
    }

    t_field = perf_counter()


    # 单次纯谱校准（非迭代）可显著改善组分可分性
    p_freq_base = p_freq_complex.copy()
    if cfg.calibrate_pure_spectra_once:
        p_freq_real_cal, p_freq_cal, recon_error_cal = calibrate_pure_spectra_once(
            spectra=s,
            wavelengths=wl,
            f_response=f_response,
            g_shift=g_shift,
            ridge=max(float(cfg.calibration_ridge), 1e-12),
        )

        base_norm = _lambda_spectra_norm_from_freq(p_freq_base)
        cal_norm = _lambda_spectra_norm_from_freq(p_freq_cal)
        collapsed = cal_norm < max(1e-12, 1e-3 * (base_norm + 1e-12))

        if collapsed:
            p_freq_complex = p_freq_base
            p_freq_real = np.real(p_freq_base)
            recon_error = _reconstruction_error_from_fg_p(
                spectra=s,
                wavelengths=wl,
                f_response=f_response,
                g_shift=g_shift,
                p_complex=p_freq_complex,
            )
            diagnostics["pure_spectra_calibration_collapsed"] = 1.0
            diagnostics["pure_spectra_base_norm"] = base_norm
            diagnostics["pure_spectra_cal_norm"] = cal_norm
        else:
            p_freq_complex = p_freq_cal
            p_freq_real = p_freq_real_cal
            recon_error = recon_error_cal
            diagnostics["pure_spectra_calibration_collapsed"] = 0.0
            diagnostics["pure_spectra_base_norm"] = base_norm
            diagnostics["pure_spectra_cal_norm"] = cal_norm
    else:
        recon_error = _reconstruction_error_from_fg_p(
            spectra=s,
            wavelengths=wl,
            f_response=f_response,
            g_shift=g_shift,
            p_complex=p_freq_complex,
        )
        p_freq_real = np.real(p_freq_complex)
        diagnostics["pure_spectra_calibration_collapsed"] = 0.0
        diagnostics["pure_spectra_base_norm"] = _lambda_spectra_norm_from_freq(p_freq_complex)
        diagnostics["pure_spectra_cal_norm"] = diagnostics["pure_spectra_base_norm"]
    diagnostics["重建相对误差"] = float(recon_error)

    # 对外输出使用波长域纯光谱（而非频域系数）
    p_freq_complex_sym = _enforce_hermitian_frequency_spectra(p_freq_complex)
    p_lambda_complex = np.fft.ifft(p_freq_complex_sym, axis=0)
    p_real = np.real(p_lambda_complex)
    t_calib = perf_counter()

    # 单次流程：仅基于结果质量决定展示策略，不做迭代回代
    symbolic_low_conf = (
        recon_error > 0.25
        or diagnostics.get("nullspace_energy", 0.0) > 0.1
    )

    # 统一输出：直接利用解析基于 xi 的特征生成 PDE 方程
    try:
        latex_blocks = build_latex_blocks_from_xi(xi, operator_names, top_terms=8)
    except Exception:
        latex_blocks = []

    t_symbolic = perf_counter()

    quality_flags = []
    if diagnostics.get("nullspace_energy", 0.0) > 0.1:
        quality_flags.append("high_nullspace_residual")
    if symbolic_low_conf:
        quality_flags.append("symbolic_low_confidence")
    if len(quality_flags) == 0:
        quality_flags.append("ok")

    block_ranges = _operator_blocks_from_tags(operator_tags)
    xi_tensor, xi_tensor_mask, xi_block_names = _xi_matrix_to_tensor(xi, block_ranges)

    meta = {
        "omega": omega,
        "theta_shape": theta.shape,
        "gamma_shape": theta.shape,
        "pure_spectra_freq_complex": p_freq_complex,
        "pure_spectra_freq_complex_hermitian": p_freq_complex_sym,
        "pure_spectra_freq_real": p_freq_real,
        "J_tot": int(len(operator_names)),
        "operator_block_ranges": block_ranges,
        "xi_tensor_block_names": xi_block_names,
        "xi_tensor_mask": xi_tensor_mask,
        "symbol_semantics": {
            "Gamma": "lifting matrix from observable operators, shape (J_tot, M)",
            "A": "operator coefficient matrix in Gamma≈A·P^T (a.k.a. A_matrix), shape (J_tot, K)",
            "P": "pure spectra coefficient matrix in Gamma≈A·P^T, shape (M, K)",
            "Xi": "tensorized A grouped by operator blocks",
            "Xi_by_component": "dict mapping component_k -> A column (J_tot,)",
        },
        "selected_component_indices": np.arange(k_eff, dtype=int),
        "anchor_indices": anchor_idx,
        "k_mode": cfg.k_mode,
        "k_source": "k_value" if cfg.k_mode == "fixed" else "mode_rule",
        "library_config": {
            "mode": "12_class_rigorous"
        },
        "timing_seconds": {
            "fft": float(t_fft - t0),
            "derivatives": float(t_deriv - t_fft),
            "library": float(t_library - t_deriv),
            "gamma_decompose": float(t_decomp - t_library),
            "field_reconstruct": float(t_field - t_decomp),
            "pure_spectra_calib": float(t_calib - t_field),
            "symbolic": float(t_symbolic - t_calib),
            "total": float(t_symbolic - t0),
        },
    }

    return DiscoveryResult(
        S_real=p_real,
        f_response_eval=f_response,
        Xi=xi_tensor,
        operator_names=operator_names,
        A_matrix=xi,
        f_response=f_response,
        g_shift=g_shift,
        pure_spectra_complex=p_lambda_complex,
        reconstruction_error=float(recon_error),
        xi_by_control={f"component_{k+1}": xi[:, k] for k in range(k_eff)},
        component_scores=None,
        component_energy_ratio=None,
        component_nonzero_ratio=None,
        quality_flags=quality_flags,
        latex_blocks=latex_blocks,
        diagnostics=diagnostics,
        metadata=meta,
    )
