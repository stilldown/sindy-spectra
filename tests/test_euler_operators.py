"""测试 1.md 完整 Euler 算子逻辑。

验证要点
--------
1. **c 缩放正确性**：Euler 算子 L_i = c_i * ∂ ln A / ∂c_i
   - 在零锚点（c_i=0）处严格为零
   - L_i / c_i = 常数（Beer-Lambert 型数据）

2. **对角 Xi2 公式**：Ξ_{ii} = c_i² β_{ii} + L_i（含一阶修正项）
   - 对 D = exp(alpha*c1)*P(ω)：Xi2_11 = L1（因为 beta_11=0）
   - 1.md 公式：Ξ_{ii} = D^{-1}(c_i² ∂²D/∂c_i²) - (L_i² - L_i)

3. **非对角 Xi2 公式**：Ξ_{ij} = c_i c_j β_{ij}（无修正项）

4. **W(ω) 拟合路径**（`build_direct_euler_library`）：
   - 库条目形状 (N, 1)，K=1
   - 同样满足 c 缩放特性

5. **管线集成**：`use_direct_euler=True` 路径端到端运行
"""
from __future__ import annotations

import numpy as np
import pytest

from opera.discovery.pipeline_utils import (
    construct_pure_library,
    build_direct_euler_library,
    pretty_name,
)
from opera.discovery.types import DiscoveryConfig
from opera.discovery import run_discovery


# ---------------------------------------------------------------------------
# 辅助函数：构造精确可解析的测试数据
# ---------------------------------------------------------------------------

def _beer_lambert_data_1d(alpha: float = 2.0, n_c: int = 4, n_wl: int = 32):
    """D(c1, omega) = exp(alpha * c1) * P(omega) 精确解析数据（1D 控制）。

    Euler 算子精确结果：
        L1_c1 = c1 * alpha
        Xi2_c1c1 = c1 * alpha  (= L1_c1，因 beta_11 = 0 on this model)
    """
    c1_vals = np.linspace(0.0, 1.5, n_c)
    c = c1_vals.reshape(-1, 1)
    omega_wl = np.linspace(0.0, 1.0, n_wl)
    P_spectral = np.exp(-2 * omega_wl) + 0.5
    spectra = np.exp(alpha * c1_vals[:, None]) * P_spectral[None, :]  # (n_c, n_wl)
    d_hat = np.fft.rfft(spectra, axis=1)
    n_freq = d_hat.shape[1]
    omega = np.linspace(0.0, 1.0, n_freq)
    h = 1e-6
    dD_dc = np.zeros((n_c, 1, n_freq), dtype=complex)
    d2D_dc2 = np.zeros((n_c, 1, 1, n_freq), dtype=complex)
    for n in range(n_c):
        sp = np.fft.rfft(np.exp(alpha * (c1_vals[n] + h)) * P_spectral)
        sm = np.fft.rfft(np.exp(alpha * (c1_vals[n] - h)) * P_spectral)
        dD_dc[n, 0, :] = (sp - sm) / (2 * h)
        d2D_dc2[n, 0, 0, :] = np.fft.rfft(alpha ** 2 * np.exp(alpha * c1_vals[n]) * P_spectral)
    return c, d_hat, omega, dD_dc, d2D_dc2, c1_vals, alpha


def _cartesian_grid_spectra(seed: int = 42, n_wl: int = 48):
    """构造笛卡尔网格光谱（含零锚点）用于端到端测试。"""
    rng = np.random.default_rng(seed)
    c1_vals = np.linspace(0.0, 2.0, 4)
    c2_vals = np.linspace(0.0, 1.0, 4)
    c1_g, c2_g = np.meshgrid(c1_vals, c2_vals, indexing="ij")
    c = np.column_stack([c1_g.flatten(), c2_g.flatten()])
    wl = np.linspace(380.0, 730.0, n_wl)
    x = wl[None, :]
    s = (np.exp(-0.5 * ((x - (500 + 20 * c[:, 0, None])) / 18.0) ** 2)
         + 0.6 * np.exp(-0.5 * ((x - (580 - 15 * c[:, 1, None])) / 22.0) ** 2)
         + 0.01 * rng.normal(size=(c.shape[0], n_wl)))
    return s, c, wl


# ---------------------------------------------------------------------------
# 1. Euler 算子 c 缩放正确性
# ---------------------------------------------------------------------------

class TestEulerOperatorCScaling:
    """验证 L_i = c_i * ∂ ln A / ∂c_i 的 c 缩放特性。"""

    def test_l1_zero_at_anchor_svd(self):
        """SVD 路径：零锚点处 L1 = 0（c_i=0 时 Euler 算子自然为零）。"""
        c, d_hat, omega, dD_dc, d2D_dc2, c1_vals, alpha = _beer_lambert_data_1d()
        cfg = DiscoveryConfig(k_mode="fixed", k_value=1)
        lib, _, _, _ = construct_pure_library(d_hat, dD_dc, d2D_dc2, omega, c, cfg)
        anchor_mask = c1_vals == 0.0
        assert np.allclose(lib["L1_c1_f"][anchor_mask, :], 0.0, atol=1e-10)

    def test_l1_zero_at_anchor_direct(self):
        """直接 Euler 路径：零锚点处 L1 = 0。"""
        c, d_hat, omega, dD_dc, d2D_dc2, c1_vals, alpha = _beer_lambert_data_1d()
        lib, _, _, _ = build_direct_euler_library(d_hat, dD_dc, d2D_dc2, omega, c)
        anchor_mask = c1_vals == 0.0
        assert np.allclose(lib["L1_c1_f"][anchor_mask, :], 0.0, atol=1e-10)

    def test_l1_proportional_to_c_svd(self):
        """SVD 路径：L1 / c1 = alpha（常数）对所有 c1 > 0 成立。"""
        c, d_hat, omega, dD_dc, d2D_dc2, c1_vals, alpha = _beer_lambert_data_1d()
        cfg = DiscoveryConfig(k_mode="fixed", k_value=1)
        lib, _, _, _ = construct_pure_library(d_hat, dD_dc, d2D_dc2, omega, c, cfg)
        non_anchor = c1_vals > 0
        ratios = lib["L1_c1_f"][non_anchor, 0] / c1_vals[non_anchor]
        np.testing.assert_allclose(ratios, alpha, rtol=1e-5)

    def test_l1_proportional_to_c_direct(self):
        """直接 Euler 路径：L1 / c1 = alpha（常数）对所有 c1 > 0 成立。"""
        c, d_hat, omega, dD_dc, d2D_dc2, c1_vals, alpha = _beer_lambert_data_1d()
        lib, _, _, _ = build_direct_euler_library(d_hat, dD_dc, d2D_dc2, omega, c)
        non_anchor = c1_vals > 0
        ratios = lib["L1_c1_f"][non_anchor, 0] / c1_vals[non_anchor]
        np.testing.assert_allclose(ratios, alpha, rtol=1e-4)


# ---------------------------------------------------------------------------
# 2. 对角 Xi2 含一阶修正项
# ---------------------------------------------------------------------------

class TestXi2DiagonalFormula:
    """验证 Ξ_{ii} = c_i² β_{ii} + L_i（1.md 公式，含修正项）。"""

    def test_xi2_diagonal_equals_l1_for_beer_lambert_svd(self):
        """对于 D = exp(alpha*c1)*P(ω)，beta_11 = 0，故 Ξ_{11} = L_1。"""
        c, d_hat, omega, dD_dc, d2D_dc2, c1_vals, alpha = _beer_lambert_data_1d()
        cfg = DiscoveryConfig(k_mode="fixed", k_value=1)
        lib, _, _, _ = construct_pure_library(d_hat, dD_dc, d2D_dc2, omega, c, cfg)
        np.testing.assert_allclose(
            lib["Xi2_c1c1_f"], lib["L1_c1_f"], atol=1e-8,
            err_msg="Xi2 diagonal should equal L1 when D = exp(alpha*c1)*P(omega)"
        )

    def test_xi2_diagonal_equals_l1_for_beer_lambert_direct(self):
        """直接 Euler 路径同理：Ξ_{11} = L_1。"""
        c, d_hat, omega, dD_dc, d2D_dc2, c1_vals, alpha = _beer_lambert_data_1d()
        lib, _, _, _ = build_direct_euler_library(d_hat, dD_dc, d2D_dc2, omega, c)
        np.testing.assert_allclose(
            lib["Xi2_c1c1_f"], lib["L1_c1_f"], atol=1e-8,
            err_msg="Direct Euler: Xi2 diagonal should equal L1 for Beer-Lambert data"
        )

    def test_xi2_zero_at_anchor_svd(self):
        """SVD 路径：c_i=0 时 Xi2 对角项为零（因含 L_i 修正，L_i(c=0)=0）。"""
        c, d_hat, omega, dD_dc, d2D_dc2, c1_vals, alpha = _beer_lambert_data_1d()
        cfg = DiscoveryConfig(k_mode="fixed", k_value=1)
        lib, _, _, _ = construct_pure_library(d_hat, dD_dc, d2D_dc2, omega, c, cfg)
        anchor_mask = c1_vals == 0.0
        assert np.allclose(lib["Xi2_c1c1_f"][anchor_mask, :], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. 直接 Euler 库形状
# ---------------------------------------------------------------------------

class TestDirectEulerLibraryShapes:
    """验证 build_direct_euler_library 返回值形状正确。"""

    def test_library_entry_shape_is_N1(self):
        """库条目形状应为 (N, 1)（K=1 全局方程）。"""
        c, d_hat, omega, dD_dc, d2D_dc2, c1_vals, _ = _beer_lambert_data_1d()
        N = len(c1_vals)
        lib, basis, A, omega_means = build_direct_euler_library(
            d_hat, dD_dc, d2D_dc2, omega, c
        )
        for key, val in lib.items():
            assert val.shape == (N, 1), f"{key}: expected ({N}, 1), got {val.shape}"

    def test_spectral_basis_and_A_shapes(self):
        """basis 形状 (1, P)，A 形状 (N, 1)，omega_means 形状 (1,)。"""
        c, d_hat, omega, dD_dc, d2D_dc2, _, _ = _beer_lambert_data_1d()
        N, P = d_hat.shape
        lib, basis, A, omega_means = build_direct_euler_library(
            d_hat, dD_dc, d2D_dc2, omega, c
        )
        assert basis.shape == (1, P)
        assert A.shape == (N, 1)
        assert omega_means.shape == (1,)

    def test_expected_keys_present(self):
        """库中必须包含 ln_f, g, L1_c1_f, L1_c1_g, Xi2_c1c1_f, Xi2_c1c1_g。"""
        c, d_hat, omega, dD_dc, d2D_dc2, _, _ = _beer_lambert_data_1d()
        lib, _, _, _ = build_direct_euler_library(d_hat, dD_dc, d2D_dc2, omega, c)
        for key in ["ln_f", "g", "L1_c1_f", "L1_c1_g", "Xi2_c1c1_f", "Xi2_c1c1_g"]:
            assert key in lib, f"Missing key: {key}"

    def test_two_controls_key_count(self):
        """二维控制下库条目数量正确（1 zeroth + 2 L1 pairs + 4 Xi2 pairs = 14 entries）。"""
        rng = np.random.default_rng(0)
        N, d, P = 8, 2, 10
        d_hat = rng.standard_normal((N, P)) + 1j * rng.standard_normal((N, P))
        c = rng.uniform(0.1, 2.0, (N, d))
        omega = np.linspace(0, 1, P)
        dD_dc = rng.standard_normal((N, d, P)) + 1j * rng.standard_normal((N, d, P))
        d2D_dc2 = rng.standard_normal((N, d, d, P)) + 1j * rng.standard_normal((N, d, d, P))
        lib, _, _, _ = build_direct_euler_library(d_hat, dD_dc, d2D_dc2, omega, c)
        # ln_f, g, ln_f^2, g^2 = 4; L1 × 2 × 2 = 4; Xi2 × 4 × 2 = 8 → total 16
        assert len(lib) == 16, f"Expected 16 keys for 2D control, got {len(lib)}"


# ---------------------------------------------------------------------------
# 4. pretty_name 对角 Xi2 格式
# ---------------------------------------------------------------------------

class TestPrettyNameDiagonalXi2:
    """验证对角 Xi2 的 pretty_name 反映完整二阶 Euler 算子。"""

    def test_diagonal_xi2_name_contains_euler_structure(self):
        """对角 Xi2 名称应体现 c_i ∂(c_i ∂/∂c_i) 的 Euler 结构。"""
        name_f = pretty_name("Xi2_c1c1_f")
        # 应包含外层 c1d(...)/dc1 形式，并且括号内也含 c1d...
        assert name_f.count("c1d") >= 2, (
            f"Diagonal Xi2 name should show nested Euler structure c1d(c1d.../dc1)/dc1, "
            f"got: {name_f}"
        )

    def test_off_diagonal_xi2_name_unchanged(self):
        """非对角 Xi2 名称应保持 c_i c_j 乘法形式。"""
        name_f = pretty_name("Xi2_c1c2_f")
        assert "c1c2" in name_f, f"Unexpected off-diagonal name: {name_f}"


# ---------------------------------------------------------------------------
# 5. 管线集成测试（use_direct_euler=True）
# ---------------------------------------------------------------------------

class TestDirectEulerPipeline:
    """验证 use_direct_euler=True 路径端到端运行正确。"""

    def test_pipeline_runs_without_error(self):
        """use_direct_euler=True 路径应无错误地完成。"""
        s, c, wl = _cartesian_grid_spectra()
        cfg = DiscoveryConfig(use_direct_euler=True)
        out = run_discovery(s, c, wl, cfg)
        assert out.Xi.ndim == 3
        assert out.S_real is not None
        assert np.isfinite(out.reconstruction_error)

    def test_pipeline_k_equals_1(self):
        """直接 Euler 路径的组分数 K 应为 1（全局物理方程，无组分分离）。"""
        s, c, wl = _cartesian_grid_spectra()
        cfg = DiscoveryConfig(use_direct_euler=True)
        out = run_discovery(s, c, wl, cfg)
        assert out.Xi.shape[2] == 1, f"Expected K=1, got K={out.Xi.shape[2]}"
        assert out.A_matrix.shape[1] == 1
        assert out.S_real.shape[1] == 1

    def test_pipeline_output_shapes(self):
        """直接 Euler 路径输出形状：S_real=(M,1)，f_response=(N,1)，Xi=(1,J,1)。"""
        s, c, wl = _cartesian_grid_spectra()
        n_samples = c.shape[0]
        n_wl = len(wl)
        cfg = DiscoveryConfig(use_direct_euler=True)
        out = run_discovery(s, c, wl, cfg)
        assert out.S_real.shape[0] == n_wl
        assert out.S_real.shape[1] == 1
        assert out.f_response_eval.shape[0] == n_samples
        assert out.f_response_eval.shape[1] == 1

    def test_pipeline_has_latex_blocks(self):
        """直接 Euler 路径应输出 1 个 LaTeX 方程块。"""
        s, c, wl = _cartesian_grid_spectra()
        cfg = DiscoveryConfig(use_direct_euler=True)
        out = run_discovery(s, c, wl, cfg)
        assert len(out.latex_blocks) == 1
        assert isinstance(out.latex_blocks[0], str)
        assert len(out.latex_blocks[0]) > 0

    def test_pipeline_operator_names_match_pretty_name(self):
        """直接 Euler 路径的算子名称应通过 pretty_name 正确渲染。"""
        s, c, wl = _cartesian_grid_spectra()
        cfg = DiscoveryConfig(use_direct_euler=True)
        out = run_discovery(s, c, wl, cfg)
        # 所有名称均应非空
        assert all(len(n) > 0 for n in out.operator_names)
        # 应包含 L1 算子名称
        assert any("d_ln_f" in n or "d_g" in n for n in out.operator_names)

    def test_use_direct_euler_priority_over_inverse_operator(self):
        """当 use_direct_euler=True 时，即使同时设 use_inverse_operator=True，
        也应使用 direct_euler 路径（K=1，不是 N）。"""
        s, c, wl = _cartesian_grid_spectra()
        cfg = DiscoveryConfig(use_direct_euler=True, use_inverse_operator=True)
        out = run_discovery(s, c, wl, cfg)
        # direct_euler 路径 K=1
        assert out.Xi.shape[2] == 1
