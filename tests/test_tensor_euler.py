"""测试张量形式 Euler 算子库（tensor_utils.py）。

验证要点
--------
1. **flat_to_tensor / tensor_to_flat 互逆性**
2. **compute_tensor_fourier 与展平 rfft 结果一致**
3. **compute_tensor_control_derivatives 与 build_control_derivative_bundle 结果一致**
4. **build_tensor_euler_library 与 construct_pure_library 算子值一致**
   （相同数据，张量路径与展平路径给出相同 Euler 算子）
5. **run_tensor_discovery 端到端运行**
6. **1D/2D 控制变量均支持**
7. **非等间距网格（np.gradient 内部支持）**
"""
from __future__ import annotations

import numpy as np
import pytest

from opera.discovery.tensor_utils import (
    flat_to_tensor,
    tensor_to_flat,
    compute_tensor_fourier,
    compute_tensor_control_derivatives,
    build_tensor_euler_library,
    run_tensor_discovery,
)
from opera.discovery.pipeline_utils import construct_pure_library
from opera.discovery.preprocess import build_control_derivative_bundle, compute_fourier_tensor
from opera.discovery.types import DiscoveryConfig


# ---------------------------------------------------------------------------
# 辅助数据生成
# ---------------------------------------------------------------------------

def _beer_lambert_1d(alpha: float = 2.0, n_c: int = 5, n_wl: int = 32):
    """D(c1, ω) = exp(alpha * c1) * P(ω)，1 种浓度变量。"""
    c1_vals = np.linspace(0.0, 1.5, n_c)
    wl = np.linspace(400.0, 700.0, n_wl)
    P_spec = np.exp(-2 * np.linspace(0, 1, n_wl)) + 0.5
    # 张量形式：(n_c, n_wl)
    spectra_tensor = np.exp(alpha * c1_vals[:, None]) * P_spec[None, :]
    c_axes = [c1_vals]
    return spectra_tensor, c_axes, wl, alpha


def _beer_lambert_2d(alpha: float = 1.5, beta: float = 0.8,
                     n_c1: int = 4, n_c2: int = 3, n_wl: int = 48):
    """D(c1, c2, ω) = exp(alpha*c1 + beta*c2) * P(ω)，2 种浓度变量。"""
    c1_vals = np.linspace(0.0, 1.2, n_c1)
    c2_vals = np.linspace(0.0, 0.9, n_c2)
    wl = np.linspace(380.0, 730.0, n_wl)
    P_spec = np.exp(-np.linspace(0, 2, n_wl)) + 0.3
    # 张量形式：(n_c1, n_c2, n_wl)
    c1_g, c2_g = np.meshgrid(c1_vals, c2_vals, indexing="ij")
    spectra_tensor = np.exp(alpha * c1_g[..., None] + beta * c2_g[..., None]) * P_spec
    c_axes = [c1_vals, c2_vals]
    return spectra_tensor, c_axes, wl, alpha, beta


def _flat_from_tensor_1d(spectra_tensor, c_axes):
    """从 1D 张量生成展平的 (N, M) + c(N, 1) 格式。"""
    N, M = spectra_tensor.shape
    spectra_flat = spectra_tensor.reshape(N, M)
    c_flat = c_axes[0].reshape(-1, 1)
    return spectra_flat, c_flat


def _flat_from_tensor_2d(spectra_tensor, c_axes):
    """从 2D 张量生成展平的 (N, M) + c(N, 2) 格式。"""
    n1, n2, M = spectra_tensor.shape
    N = n1 * n2
    spectra_flat = spectra_tensor.reshape(N, M)
    c1_g, c2_g = np.meshgrid(c_axes[0], c_axes[1], indexing="ij")
    c_flat = np.stack([c1_g.flatten(), c2_g.flatten()], axis=-1)
    return spectra_flat, c_flat


# ---------------------------------------------------------------------------
# 1. flat_to_tensor 和 tensor_to_flat
# ---------------------------------------------------------------------------

class TestFlatTensorConversion:
    def test_round_trip_1d(self):
        """1D 展平 → 张量 → 展平应与原始一致。"""
        st, c_axes, wl, _ = _beer_lambert_1d()
        sf, cf = _flat_from_tensor_1d(st, c_axes)
        st2, axes2, gs = flat_to_tensor(sf, cf)
        assert st2.shape == st.shape
        np.testing.assert_allclose(st2, st)
        assert len(axes2) == 1
        np.testing.assert_allclose(axes2[0], c_axes[0])

    def test_round_trip_2d(self):
        """2D 展平 → 张量 → 展平应与原始一致。"""
        st, c_axes, wl, _, _ = _beer_lambert_2d()
        sf, cf = _flat_from_tensor_2d(st, c_axes)
        st2, axes2, gs = flat_to_tensor(sf, cf)
        assert st2.shape == st.shape
        np.testing.assert_allclose(st2, st, atol=1e-12)
        assert len(axes2) == 2
        np.testing.assert_allclose(axes2[0], c_axes[0])
        np.testing.assert_allclose(axes2[1], c_axes[1])

    def test_tensor_to_flat_1d(self):
        """tensor_to_flat(1 leading dim) 展平正确。"""
        st, c_axes, wl, _ = _beer_lambert_1d()
        sf = tensor_to_flat(st, n_leading=1)
        assert sf.shape == (st.shape[0], st.shape[1])
        np.testing.assert_allclose(sf, st.reshape(-1, st.shape[-1]))

    def test_tensor_to_flat_2d(self):
        """tensor_to_flat(2 leading dims) 展平正确。"""
        st, c_axes, wl, _, _ = _beer_lambert_2d()
        sf = tensor_to_flat(st, n_leading=2)
        n1, n2, M = st.shape
        assert sf.shape == (n1 * n2, M)

    def test_non_cartesian_raises(self):
        """非笛卡尔网格应抛出 ValueError。"""
        spectra = np.random.default_rng(0).random((5, 20))
        c_bad = np.random.default_rng(1).random((5, 2))
        with pytest.raises(ValueError):
            flat_to_tensor(spectra, c_bad)


# ---------------------------------------------------------------------------
# 2. compute_tensor_fourier
# ---------------------------------------------------------------------------

class TestTensorFourier:
    def test_consistent_with_compute_fourier_tensor_1d(self):
        """张量 rfft 结果应与 compute_fourier_tensor 在展平数据上一致。"""
        st, c_axes, wl, _ = _beer_lambert_1d()
        sf, _ = _flat_from_tensor_1d(st, c_axes)

        D_tensor, omega_t = compute_tensor_fourier(st)
        D_flat, omega_f = compute_fourier_tensor(sf, wl)

        np.testing.assert_allclose(D_tensor.reshape(-1, D_tensor.shape[-1]), D_flat, atol=1e-12)
        np.testing.assert_allclose(omega_t, omega_f, atol=1e-12)

    def test_consistent_with_compute_fourier_tensor_2d(self):
        """2D 张量 rfft 结果与展平路径一致。"""
        st, c_axes, wl, _, _ = _beer_lambert_2d()
        sf, _ = _flat_from_tensor_2d(st, c_axes)

        D_tensor, omega_t = compute_tensor_fourier(st)
        D_flat, omega_f = compute_fourier_tensor(sf, wl)

        n1, n2, P = D_tensor.shape
        np.testing.assert_allclose(D_tensor.reshape(n1 * n2, P), D_flat, atol=1e-12)

    def test_output_shape(self):
        """输出形状：(*grid_shape, P)，P = M//2 + 1。"""
        st, c_axes, wl, _ = _beer_lambert_1d(n_c=6, n_wl=40)
        D, omega = compute_tensor_fourier(st)
        assert D.shape == (6, 40 // 2 + 1)
        assert omega.shape == (40 // 2 + 1,)
        assert omega[0] == pytest.approx(0.0)
        assert omega[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. compute_tensor_control_derivatives
# ---------------------------------------------------------------------------

class TestTensorControlDerivatives:
    def test_1d_consistent_with_build_control_derivative_bundle(self):
        """1D 张量路径偏导与展平路径 build_control_derivative_bundle 一致。"""
        st, c_axes, wl, _ = _beer_lambert_1d(n_c=6)
        sf, cf = _flat_from_tensor_1d(st, c_axes)

        D_tensor, omega_t = compute_tensor_fourier(st)
        dD_dc_tensor, d2D_tensor = compute_tensor_control_derivatives(D_tensor, c_axes)

        D_flat, _ = compute_fourier_tensor(sf, wl)
        dD_flat, d2D_flat = build_control_derivative_bundle(D_flat, cf)

        # 展平张量导数并比较
        N = len(c_axes[0])
        P = D_tensor.shape[-1]
        dD_t_flat = dD_dc_tensor[..., 0, :].reshape(N, P)   # (N, P)
        np.testing.assert_allclose(dD_t_flat, dD_flat[:, 0, :], atol=1e-8)

    def test_2d_derivative_shape(self):
        """2D 张量偏导形状：(*grid, d, P) 和 (*grid, d, d, P)。"""
        st, c_axes, wl, _, _ = _beer_lambert_2d(n_c1=4, n_c2=3)
        D_tensor, _ = compute_tensor_fourier(st)
        dD, d2D = compute_tensor_control_derivatives(D_tensor, c_axes)
        n1, n2, P = D_tensor.shape
        assert dD.shape == (n1, n2, 2, P)
        assert d2D.shape == (n1, n2, 2, 2, P)

    def test_second_deriv_symmetry(self):
        """混合偏导应对称：∂²D/∂c_i∂c_j = ∂²D/∂c_j∂c_i（数值差分近似对称）。"""
        st, c_axes, wl, _, _ = _beer_lambert_2d()
        D_tensor, _ = compute_tensor_fourier(st)
        _, d2D = compute_tensor_control_derivatives(D_tensor, c_axes)
        # d2D[..., 0, 1, :] ≈ d2D[..., 1, 0, :]（数值差分在等间距网格上精确对称）
        np.testing.assert_allclose(
            d2D[..., 0, 1, :], d2D[..., 1, 0, :], atol=1e-6,
            err_msg="二阶混合偏导应近似对称"
        )

    def test_gradient_along_correct_axis(self):
        """c1 轴的导数应沿轴 0 变化，c2 轴的导数应沿轴 1 变化。"""
        st, c_axes, wl, alpha, beta = _beer_lambert_2d()
        D_tensor, _ = compute_tensor_fourier(st)
        dD, _ = compute_tensor_control_derivatives(D_tensor, c_axes)

        # D = exp(alpha*c1 + beta*c2)*P(ω)，去直流后 ∂D/∂c1 = alpha * D
        # 验证 dD[..., 0, :] / D_tensor ≈ alpha（在所有 c 点上）
        # 仅检查中间样本（边界用一阶差分，精度略低）
        ratio_c1 = np.abs(dD[1:-1, 1:-1, 0, :]) / (np.abs(D_tensor[1:-1, 1:-1, :]) + 1e-12)
        # 对数导数大小约为 alpha（Beer-Lambert 模型的精确解）
        assert np.mean(ratio_c1) == pytest.approx(alpha, rel=0.1)


# ---------------------------------------------------------------------------
# 4. build_tensor_euler_library：与 construct_pure_library 结果一致
# ---------------------------------------------------------------------------

class TestTensorEulerLibraryConsistency:
    """验证张量路径与展平路径给出相同的 Euler 算子值。"""

    def _compare_1d(self, alpha, n_c, n_wl, k_value):
        st, c_axes, wl, _ = _beer_lambert_1d(alpha=alpha, n_c=n_c, n_wl=n_wl)
        sf, cf = _flat_from_tensor_1d(st, c_axes)
        cfg = DiscoveryConfig(k_mode="fixed", k_value=k_value)

        # 张量路径
        lib_t, lib_tensor, basis_t, A_t, A_tensor, w_t, gs = build_tensor_euler_library(
            st, c_axes, wl, cfg
        )

        # 展平路径（参考实现）
        D_flat, omega = compute_fourier_tensor(sf, wl)
        dD_flat, d2D_flat = build_control_derivative_bundle(D_flat, cf)
        lib_f, basis_f, A_f, w_f = construct_pure_library(D_flat, dD_flat, d2D_flat, omega, cf, cfg)

        return lib_t, lib_f, basis_t, basis_f, A_t, A_f

    def test_l1_f_matches_1d(self):
        """1D：L1_c1_f（一阶 Euler 算子 f 分量）与展平路径一致。"""
        lib_t, lib_f, _, _, _, _ = self._compare_1d(alpha=2.0, n_c=6, n_wl=32, k_value=1)
        np.testing.assert_allclose(lib_t["L1_c1_f"], lib_f["L1_c1_f"], atol=1e-8)

    def test_l1_g_matches_1d(self):
        """1D：L1_c1_g（一阶 Euler 算子 g 分量）与展平路径一致。"""
        lib_t, lib_f, _, _, _, _ = self._compare_1d(alpha=2.0, n_c=6, n_wl=32, k_value=1)
        np.testing.assert_allclose(lib_t["L1_c1_g"], lib_f["L1_c1_g"], atol=1e-8)

    def test_xi2_matches_1d(self):
        """1D：Xi2_c1c1_f 与展平路径一致。"""
        lib_t, lib_f, _, _, _, _ = self._compare_1d(alpha=2.0, n_c=6, n_wl=32, k_value=1)
        np.testing.assert_allclose(lib_t["Xi2_c1c1_f"], lib_f["Xi2_c1c1_f"], atol=1e-8)

    def test_ln_f_matches_1d(self):
        """1D：ln_f（零阶项）与展平路径一致。"""
        lib_t, lib_f, _, _, _, _ = self._compare_1d(alpha=2.0, n_c=6, n_wl=32, k_value=1)
        np.testing.assert_allclose(lib_t["ln_f"], lib_f["ln_f"], atol=1e-8)

    def test_all_keys_present(self):
        """库中应包含所有 f/g 分量键名（1D case）。"""
        st, c_axes, wl, _ = _beer_lambert_1d()
        cfg = DiscoveryConfig(k_mode="fixed", k_value=1)
        lib_t, lib_tensor, *_ = build_tensor_euler_library(st, c_axes, wl, cfg)
        expected_keys = {
            "ln_f", "g", "ln_f^2", "g^2",
            "L1_c1_f", "L1_c1_g",
            "Xi2_c1c1_f", "Xi2_c1c1_g",
        }
        assert expected_keys.issubset(set(lib_t.keys()))

    def test_2d_library_keys(self):
        """2D case：库中应包含 c1, c2 的一阶和二阶算子。"""
        st, c_axes, wl, _, _ = _beer_lambert_2d()
        cfg = DiscoveryConfig(k_mode="fixed", k_value=2)
        lib_t, lib_tensor, *_ = build_tensor_euler_library(st, c_axes, wl, cfg)
        for j in range(1, 3):
            assert f"L1_c{j}_f" in lib_t
            assert f"L1_c{j}_g" in lib_t
        for i in range(1, 3):
            for j in range(1, 3):
                assert f"Xi2_c{i}c{j}_f" in lib_t

    def test_2d_l1_matches_flat(self):
        """2D：L1_c1_f 和 L1_c2_f 与展平路径一致。"""
        st, c_axes, wl, _, _ = _beer_lambert_2d(n_c1=4, n_c2=3)
        sf, cf = _flat_from_tensor_2d(st, c_axes)
        cfg = DiscoveryConfig(k_mode="fixed", k_value=1)

        lib_t, _, _, A_t, _, _, _ = build_tensor_euler_library(st, c_axes, wl, cfg)

        D_flat, omega = compute_fourier_tensor(sf, wl)
        dD_flat, d2D_flat = build_control_derivative_bundle(D_flat, cf)
        lib_f, *_ = construct_pure_library(D_flat, dD_flat, d2D_flat, omega, cf, cfg)

        np.testing.assert_allclose(lib_t["L1_c1_f"], lib_f["L1_c1_f"], atol=1e-8)
        np.testing.assert_allclose(lib_t["L1_c2_f"], lib_f["L1_c2_f"], atol=1e-8)

    def test_tensor_library_shape(self):
        """library_tensor 的每个条目形状应为 (*grid_shape, K)。"""
        st, c_axes, wl, _, _ = _beer_lambert_2d(n_c1=4, n_c2=3)
        cfg = DiscoveryConfig(k_mode="fixed", k_value=2)
        _, lib_tensor, _, _, _, _, gs = build_tensor_euler_library(st, c_axes, wl, cfg)
        for key, val in lib_tensor.items():
            assert val.shape[:2] == (4, 3), f"key={key}: shape {val.shape} 前两维应为 (4,3)"
            assert val.shape[2] == 2, f"key={key}: 最后一维应为 K=2"

    def test_l1_c_scaling_in_tensor_library(self):
        """张量路径：零锚点处 L1=0，L1/c1 对内部点为常数（c 缩放正确）。

        边界点因 np.gradient 使用一阶单侧差分精度较低，此处仅检测内部点。
        """
        st, c_axes, wl, alpha = _beer_lambert_1d(n_c=7)   # 使用更多点确保有足够内部点
        cfg = DiscoveryConfig(k_mode="fixed", k_value=1)
        lib_t, lib_tensor, *_ = build_tensor_euler_library(st, c_axes, wl, cfg)

        L1 = lib_t["L1_c1_f"][:, 0]   # (N,) 第一个组分
        c1 = c_axes[0]
        # 零锚点处 L1 = 0
        assert abs(L1[0]) < 1e-10, f"零锚点处 L1 应为 0，实为 {L1[0]}"
        # 内部点（排除边界，np.gradient 边界用一阶差分精度较低）L1/c1 应约为常数
        interior = (c1 > 0) & (np.arange(len(c1)) > 0) & (np.arange(len(c1)) < len(c1) - 1)
        if interior.sum() >= 2:
            ratios = L1[interior] / c1[interior]
            assert np.ptp(ratios) < 0.05 * abs(np.mean(ratios)), (
                f"内部点 L1/c1 应约为常数（容差 5%），实为 {ratios}"
            )


# ---------------------------------------------------------------------------
# 5. run_tensor_discovery 端到端
# ---------------------------------------------------------------------------

class TestRunTensorDiscovery:
    def test_1d_runs_without_error(self):
        """1D 张量路径端到端应无错误完成。"""
        st, c_axes, wl, _ = _beer_lambert_1d()
        cfg = DiscoveryConfig(k_mode="fixed", k_value=1)
        result = run_tensor_discovery(st, c_axes, wl, cfg)
        assert result.Xi.ndim == 3
        assert result.S_real is not None

    def test_2d_runs_without_error(self):
        """2D 张量路径端到端应无错误完成。"""
        st, c_axes, wl, _, _ = _beer_lambert_2d()
        cfg = DiscoveryConfig(k_mode="fixed", k_value=2)
        result = run_tensor_discovery(st, c_axes, wl, cfg)
        assert result.Xi.shape[2] == 2   # K=2

    def test_metadata_contains_grid_shape(self):
        """结果 metadata 应包含 grid_shape 和 library_tensor。"""
        st, c_axes, wl, _ = _beer_lambert_1d()
        cfg = DiscoveryConfig(k_mode="fixed", k_value=1)
        result = run_tensor_discovery(st, c_axes, wl, cfg)
        assert "grid_shape" in result.metadata
        assert "library_tensor" in result.metadata
        assert result.metadata["grid_shape"] == st.shape[:-1]

    def test_output_shapes(self):
        """S_real (M, K)，f_response (N, K)，Xi (1, J, K)。"""
        n_c, n_wl = 5, 32
        st, c_axes, wl, _ = _beer_lambert_1d(n_c=n_c, n_wl=n_wl)
        cfg = DiscoveryConfig(k_mode="fixed", k_value=1)
        result = run_tensor_discovery(st, c_axes, wl, cfg)
        assert result.S_real.shape == (n_wl, 1)
        assert result.f_response_eval.shape == (n_c, 1)
        assert result.Xi.shape[2] == 1

    def test_no_zero_anchor_raises(self):
        """c_axes[j][0] 非零（无零锚点）时应抛出 ValueError。"""
        st, c_axes, wl, _ = _beer_lambert_1d()
        # 修改 c_axes 使其不包含 0
        c_axes_bad = [c_axes[0] + 0.5]   # 所有浓度偏移 0.5
        with pytest.raises(ValueError, match="零浓度"):
            run_tensor_discovery(st, c_axes_bad, wl)

    def test_tensor_matches_flat_discovery(self):
        """张量路径 run_tensor_discovery 与展平路径 run_discovery 给出相同算子名称和 K。"""
        from opera.discovery import run_discovery

        st, c_axes, wl, _ = _beer_lambert_1d(n_c=6)
        sf, cf = _flat_from_tensor_1d(st, c_axes)
        cfg = DiscoveryConfig(k_mode="fixed", k_value=1)

        result_tensor = run_tensor_discovery(st, c_axes, wl, cfg)
        result_flat   = run_discovery(sf, cf, wl, cfg)

        assert result_tensor.operator_names == result_flat.operator_names
        assert result_tensor.Xi.shape == result_flat.Xi.shape

    def test_library_tensor_structure(self):
        """library_tensor 中每个条目的形状应为 (*grid_shape, K)。"""
        st, c_axes, wl, _, _ = _beer_lambert_2d(n_c1=3, n_c2=4)
        cfg = DiscoveryConfig(k_mode="fixed", k_value=1)
        result = run_tensor_discovery(st, c_axes, wl, cfg)
        lib_tensor = result.metadata["library_tensor"]
        for key, val in lib_tensor.items():
            assert val.shape[:2] == (3, 4), (
                f"key={key}: shape {val.shape}, 期望前两维为 (3,4)"
            )


# ---------------------------------------------------------------------------
# 6. 非等间距网格
# ---------------------------------------------------------------------------

class TestNonUniformGrid:
    def test_nonuniform_1d_derivatives(self):
        """非等间距 1D 网格下导数计算不报错，且内部点对数导数近似正确。"""
        # 非等间距网格：c1 = 0, 0.1, 0.3, 0.6, 1.0
        c1_vals = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
        c_axes = [c1_vals]
        n_wl = 24
        # 使用非平凡光谱（避免去直流后全零）
        P_spec = np.exp(-2 * np.linspace(0, 1, n_wl)) + 0.5
        alpha = 1.5
        spectra_tensor = np.exp(alpha * c1_vals[:, None]) * P_spec[None, :]

        D_tensor, omega = compute_tensor_fourier(spectra_tensor)
        dD, _ = compute_tensor_control_derivatives(D_tensor, c_axes)

        # dD 形状：(n_c, d=1, P)，取 c1 方向导数（d=0）
        dD_c1 = dD[..., 0, :]           # (n_c, P)
        D_abs = np.abs(D_tensor)         # (n_c, P)
        # 取内部点（排除边界），跨所有频率取均值
        interior = slice(1, -1)
        D_interior = D_abs[interior]
        dD_interior = np.abs(dD_c1[interior])
        # 只在 D 较大的频率上计算比值（排除接近零的频率分量）
        mask_freq = D_interior > 0.01 * np.max(D_interior)
        ratio_vals = dD_interior[mask_freq] / D_interior[mask_freq]
        # 放宽到 30% 容差（非等间距差分精度较低）
        assert np.mean(ratio_vals) == pytest.approx(alpha, rel=0.3)

    def test_nonuniform_2d_library_runs(self):
        """非等间距 2D 网格下 build_tensor_euler_library 不报错。"""
        c1 = np.array([0.0, 0.2, 0.5, 1.0])
        c2 = np.array([0.0, 0.3, 1.0])
        c_axes = [c1, c2]
        n_wl = 20
        P_spec = np.ones(n_wl)
        c1g, c2g = np.meshgrid(c1, c2, indexing="ij")
        spectra_tensor = np.exp(c1g[..., None] + 0.5 * c2g[..., None]) * P_spec
        wl = np.linspace(400, 700, n_wl)
        cfg = DiscoveryConfig(k_mode="fixed", k_value=1)
        lib, lib_t, *_ = build_tensor_euler_library(spectra_tensor, c_axes, wl, cfg)
        assert "L1_c1_f" in lib
        assert "L1_c2_f" in lib
        assert np.all(np.isfinite(lib["L1_c1_f"]))
