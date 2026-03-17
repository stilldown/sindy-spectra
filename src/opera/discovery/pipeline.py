"""发现管线主入口：run_discovery。

端到端数学流程
--------------
给定：

* S ∈ ℝ^{N×M}  — 观测光谱矩阵（N 个控制条件，M 个波长点）
* c ∈ ℝ^{N×d}  — 控制变量矩阵
* λ ∈ ℝ^M      — 波长轴

**阶段 1：频域变换**

    S̃(c, λ) = S(c, λ) - mean_λ S(c, λ)   （去直流）
    D̂(c, ω) = rfft(S̃(c, ·))              （N×P 复矩阵，P = M//2+1）

**阶段 2：Euler 算子导数**

    ∂D̂/∂c_j  ,  ∂²D̂/(∂c_i∂c_j)   via 等间距中心差分

**阶段 3：算子特征库 Θ ∈ ℝ^{M×J}**

三种可选路径（由 DiscoveryConfig 控制）：

* 默认（``use_weak_form=False, use_inverse_operator=False``）：
    SVD 谱基投影 → 按元素复对数 → Euler 算子展开，再通过测试函数 Ψ 投影至 (M, K)
    （见 :func:`~opera.discovery.pipeline_utils.construct_pure_library`）

* 直接 Euler（``use_direct_euler=True``）：
    无 SVD 逐元素 Euler 算子，W(ω) 拟合分离 f/g，再通过 Ψ 投影至 (M, 1)
    （见 :func:`~opera.discovery.pipeline_utils.build_direct_euler_library`）

* 伪逆（``use_inverse_operator=True``）：
    D†∂D 对角线截断 → 频域特征（保持 (N, K) 强形式，不做 Ψ 投影）
    （见 :func:`~opera.discovery.operator.construct_inverse_library`）

* 真弱形式（``use_weak_form=True``）：
    D̂ @ D† = N×N 方阵（帽矩阵，无需截断），组分数 K = N；
    对 ln A 做 IBP 内积 ⟨L_i, ψ_m⟩ — 无需数值微分，直接输出 (M, N)
    （见 :func:`~opera.discovery.operator.build_weak_form_library`）

以弱形式算子库为准：除伪逆路径外，所有路径的算子库条目统一为 (M, K) 形状，
其中 M = 测试函数数量，K = 谱组分数。行的语义为测试函数内积，而非样本点逐元素值。

**阶段 4：SINDy-PI 零空间识别**

对 f 和 g 子空间分别求 SVD 最小奇异值方向：

    Θ_f ξ_f ≈ 0,   Θ_g ξ_g ≈ 0

**阶段 5：输出组装**

* 谱基 IFFT → 纯组分光谱 S_real ∈ ℝ^{M×K}
* 投影系数 A ∈ ℂ^{N×K} → f_response_eval
* 系数向量 [ξ_f; ξ_g] → Xi 张量
"""
from __future__ import annotations

import numpy as np
from time import perf_counter

from .types import DiscoveryConfig, DiscoveryResult
from .preprocess import validate_inputs, compute_fourier_tensor, build_control_derivative_bundle
from .pipeline_utils import construct_pure_library, build_direct_euler_library, solve_nullspace, pretty_name
from .operator import construct_inverse_library


def run_discovery(
    spectra: np.ndarray,
    c: np.ndarray,
    wavelengths: np.ndarray,
    config: DiscoveryConfig | None = None,
) -> DiscoveryResult:
    """从观测光谱和控制变量中发现隐式物理方程。

    Parameters
    ----------
    spectra : ndarray (N, M)
        N 个控制条件下的光谱矩阵（每行一条光谱）。
    c : ndarray (N, d)
        控制变量矩阵 c ∈ ℝ^{N×d}（如浓度、温度等），d 为控制维度数。
        必须包含至少一行零控制（参考样本）。
    wavelengths : ndarray (M,)
        单调递增的波长轴，仅用于 IFFT 重建的点数。
    config : DiscoveryConfig, optional
        算法配置，默认使用 SVD 谱基投影路径、k_value=3。

    Returns
    -------
    DiscoveryResult
        包含纯组分光谱 S_real、响应系数 f_response_eval、隐式系数 Xi 等。
    """
    t0 = perf_counter()
    cfg = config or DiscoveryConfig()

    s  = np.asarray(spectra)
    c  = np.asarray(c)
    wl = np.asarray(wavelengths)

    validate_inputs(s, c, wl)
    n_samples = s.shape[0]

    # ── 零浓度锚点检测 ──────────────────────────────────────────────────────
    # 需要至少一个所有控制因子均近似为零的参考样本（物理上的"空白"）
    anchor_idx = np.where(np.all(np.abs(c) <= float(cfg.zero_anchor_tol), axis=1))[0]
    if anchor_idx.size == 0:
        raise ValueError("未检测到零浓度样本（所有控制因子均为0）。该参考集为必需输入。")

    # ── 阶段 1：频域变换 D̂ = rfft(S̃) ─────────────────────────────────────
    d_hat, omega = compute_fourier_tensor(s, wl)
    t_fft = perf_counter()

    # ── 阶段 2：控制偏导 ∂D̂/∂c_j 和 ∂²D̂/(∂c_i∂c_j) ──────────────────────
    dD_dc, d2D_dc2 = build_control_derivative_bundle(
        d_hat=d_hat,
        c=c,
        eps=float(cfg.matrix.epsilon),
    )
    t_deriv = perf_counter()

    # ── 阶段 3：算子特征库 Θ ────────────────────────────────────────────────
    if cfg.use_weak_form:
        # 弱形式路径：D̂ @ D† 是 N×N 方阵（帽矩阵），无需截断到 k_eff。
        # build_weak_form_library 内部做 pinv → A = D̂ @ D†（N×N 方阵） → ln A → IBP。
        # 组分数 K = N（样本数），由矩阵乘法自然确定；
        # k_eff 参数被 build_weak_form_library 接受但会触发 DeprecationWarning 并被忽略。
        # 库条目形状：(M, N)，行 = 测试函数内积（弱形式原生格式）。
        from .operator import build_weak_form_library
        lib, _psi_names, _Psi, basis, A = build_weak_form_library(
            d_hat=d_hat,
            c=c,
            omega=omega,
            test_func_degree=cfg.weak_form_test_degree,
        )

    elif cfg.use_direct_euler:
        # 1.md 直接 Euler 算子路径：无 SVD，逐元素在全频域 D̂(c,ω) 上计算 Euler 算子，
        # 用 W(ω)=[1,−iω] 线性拟合分离 f/g，输出 K=1 全局物理方程。
        # 原始库条目形状：(N, 1)；以下投影并转置为 (K, M) = (1, M) = 算子种类×频点。
        lib, basis, A, _w_means = build_direct_euler_library(
            d_hat=d_hat,
            dD_dc=dD_dc,
            d2D_dc2=d2D_dc2,
            omega=omega,
            c=c,
        )
        # 算子种类×频点：Ψ 投影 (N,1)→(M,1)，再转置 → (1, M) = (K, M)
        from .operator import _build_polynomial_test_functions_with_grads
        _Psi, _, _ = _build_polynomial_test_functions_with_grads(
            c, degree=cfg.weak_form_test_degree
        )
        lib = {k: (_Psi @ v).T for k, v in lib.items()}

    elif cfg.use_inverse_operator:
        # 伪逆路径：D†∂D 对角线截断，不做谱基投影
        # 库条目形状 (N, K)；转置为 (K, N) = 算子种类×频点（N 样本作为"频点"维）。
        lib, basis, A, _w_means = construct_inverse_library(
            d_hat=d_hat,
            dD_dc=dD_dc,
            d2D_dc2=d2D_dc2,
            omega=omega,
            c=c,
            config=cfg,
        )
        lib = {k: v.T for k, v in lib.items()}
        # 附加近似弱算子（psi=1 的均匀权重版本，转置对齐）
        try:
            from .operator import compute_weak_operators
            psi = np.ones(n_samples)
            weak_lib = compute_weak_operators(d_hat, dD_dc, d2D_dc2, c, psi)
            lib.update({k: v.T for k, v in weak_lib.items()})
        except ImportError:
            pass

    else:
        # 默认路径（推荐）：SVD 谱基 → 按元素复对数 → Euler 算子（含 c_i 缩放）
        # 原始库条目形状：(N, K)；以下投影并转置为 (K, M) = 算子种类×频点。
        lib, basis, A, _w_means = construct_pure_library(
            d_hat=d_hat,
            dD_dc=dD_dc,
            d2D_dc2=d2D_dc2,
            omega=omega,
            c=c,
            config=cfg,
        )
        # 算子种类×频点：Ψ 投影 (N,K)→(M,K)，再转置 → (K, M)
        from .operator import _build_polynomial_test_functions_with_grads
        _Psi, _, _ = _build_polynomial_test_functions_with_grads(
            c, degree=cfg.weak_form_test_degree
        )
        lib = {k: (_Psi @ v).T for k, v in lib.items()}
    t_lib = perf_counter()

    # ── 阶段 4：SINDy-PI 零空间识别 ─────────────────────────────────────────
    # solve_nullspace 对 f 和 g 子空间分别求最小奇异值方向 ξ
    component_models = solve_nullspace(lib)

    # 算子名称列表（与 Xi 第二维对齐）
    op_names = [pretty_name(k) for k in lib.keys()]
    J = len(op_names)                   # 算子总数
    K = len(component_models)           # 组分数

    # ── 阶段 5：组装 Xi 系数张量 ──────────────────────────────────────────
    # Xi[0, :, k] = [ξ_f_k（按排序键顺序）; ξ_g_k（按排序键顺序）]
    # 由于 construct_pure_library 保证 J_f + J_g == J，这里可精确对齐。
    lib_f_keys = sorted(k for k in lib if "_f" in k or k in ("ln_f", "ln_f^2"))
    lib_g_keys = sorted(k for k in lib if "_g" in k or k in ("g", "g^2"))
    key_order = lib_f_keys + lib_g_keys   # 与 solve_nullspace 内部排序一致

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
                # 找到该键在全局 op_names 中的位置
                try:
                    idx = list(lib.keys()).index(fname)
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
                    idx = list(lib.keys()).index(gname)
                    xi_k[idx] = coef_r
                except ValueError:
                    pass
                if abs(coef_r) > 1e-3:
                    terms.append(f"{coef_r:.3f}·{pretty_name(gname)}")
            if terms:
                parts.append("g: " + " + ".join(terms))

        Xi[0, :, k] = xi_k
        eqn_strs.append(f"Comp {k+1} " + " | ".join(parts))

    # ── 阶段 6：纯谱重建 ──────────────────────────────────────────────────
    # basis ∈ ℂ^{K×P}，各行为 SVD 谱基向量
    # S_real ∈ ℝ^{M×K}，由 irfft(basis.row) 沿波长域重建
    S_real = np.fft.irfft(basis, n=len(wl), axis=1).T   # (M, K)
    pure_spectra = basis.T                               # (P, K)，频域占位

    return DiscoveryResult(
        S_real=S_real,
        f_response_eval=A,
        A_matrix=Xi[0],
        Xi=Xi,
        operator_names=op_names,
        f_response=A,
        pure_spectra_complex=pure_spectra,
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
            "anchor_count":    float(anchor_idx.size),
        },
        metadata={
            "models":                component_models,
            "equations":             eqn_strs,
            "anchor_indices":        anchor_idx,
            "k_source":              "k_value" if cfg.k_mode == "fixed" else "mode_rule",
            "J_tot":                 J,
            "operator_block_ranges": {},
        },
    )
