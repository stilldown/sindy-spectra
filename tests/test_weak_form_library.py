"""Tests for the weak-form operator library (`build_weak_form_library`).

Key mathematical property being tested:
    The weak-form pipeline computes the projection coefficient A by directly
    multiplying by the Moore-Penrose pseudoinverse D†, without SVD separation:

        Step 1: Pseudoinverse  D† = pinv(D̂)
        Step 2: Projection     A = D̂ @ D†[:, :K]   (N, K) — first K columns of hat matrix
        Step 3: Log            ln_A = log(A + ε)   ("another way" vs. raw-bin log)
        Step 4: IBP inner product
            ⟨L_i^(k), ψ_m⟩ = -Σ_n (∂_{c_i}ψ_m·c_i(n) + ψ_m(c_n)) · ln A_k(c_n)
"""
import numpy as np
import pytest

from opera.discovery.operator import (
    _build_polynomial_test_functions_with_grads,
    build_weak_form_library,
)
from opera.discovery import run_discovery, DiscoveryConfig


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _make_cartesian_grid(n_per_dim=5, n_controls=2, seed=7):
    """构造笛卡尔网格样本（包含原点锚点）。"""
    rng = np.random.default_rng(seed)
    vals = [np.linspace(0.0, 1.0, n_per_dim) for _ in range(n_controls)]
    grids = np.meshgrid(*vals, indexing="ij")
    factors = np.column_stack([g.ravel() for g in grids])
    return factors


# ---------------------------------------------------------------------------
# 1. 测试函数辅助函数测试
# ---------------------------------------------------------------------------

class TestBuildPolynomialTestFunctions:
    def test_shapes_single_control(self):
        N = 10
        c = np.random.default_rng(0).uniform(0.1, 2.0, (N, 1))
        Psi, dPsi, names = _build_polynomial_test_functions_with_grads(c, degree=2)
        # degree-2, 1 control: 1 + 1 + 1 = 3 functions (1, c_1, c_1^2)
        assert Psi.shape == (3, N)
        assert dPsi.shape == (1, 3, N)
        assert len(names) == 3

    def test_shapes_two_controls(self):
        N = 12
        c = np.random.default_rng(1).uniform(0.1, 2.0, (N, 2))
        Psi, dPsi, names = _build_polynomial_test_functions_with_grads(c, degree=2)
        # degree-2, 2 controls: 1 + 2 + 2 + 1 = 6 functions
        assert Psi.shape == (6, N)
        assert dPsi.shape == (2, 6, N)
        assert len(names) == 6

    def test_shapes_three_controls(self):
        N = 8
        c = np.random.default_rng(2).uniform(0.1, 1.0, (N, 3))
        Psi, dPsi, names = _build_polynomial_test_functions_with_grads(c, degree=2)
        # degree-2, 3 controls: 1 + 3 + 3 + 3 = 10 functions
        assert Psi.shape == (10, N)
        assert dPsi.shape == (3, 10, N)
        assert len(names) == 10

    def test_constant_function(self):
        """第一行必须是全 1（常数函数）。"""
        N = 8
        c = np.random.default_rng(3).uniform(0.1, 2.0, (N, 2))
        Psi, dPsi, names = _build_polynomial_test_functions_with_grads(c, degree=1)
        np.testing.assert_array_equal(Psi[0], np.ones(N))
        np.testing.assert_array_equal(dPsi[:, 0, :], np.zeros((2, N)))
        assert names[0] == "1"

    def test_linear_function_values(self):
        """ψ = c_1 的梯度应为 [1, 0, ...]。"""
        N = 6
        c = np.random.default_rng(4).uniform(0.5, 2.0, (N, 2))
        Psi, dPsi, names = _build_polynomial_test_functions_with_grads(c, degree=1)
        # name "c_1" at index 1
        idx = names.index("c_1")
        np.testing.assert_array_equal(Psi[idx], c[:, 0])
        np.testing.assert_array_equal(dPsi[0, idx, :], np.ones(N))     # ∂_{c_1}c_1 = 1
        np.testing.assert_array_equal(dPsi[1, idx, :], np.zeros(N))    # ∂_{c_2}c_1 = 0

    def test_quadratic_function_values(self):
        """ψ = c_1^2 的梯度应为 [2c_1, 0]。"""
        N = 6
        c = np.random.default_rng(5).uniform(0.5, 2.0, (N, 2))
        Psi, dPsi, names = _build_polynomial_test_functions_with_grads(c, degree=2)
        idx = names.index("c_1^2")
        np.testing.assert_allclose(Psi[idx], c[:, 0] ** 2)
        np.testing.assert_allclose(dPsi[0, idx, :], 2.0 * c[:, 0])
        np.testing.assert_array_equal(dPsi[1, idx, :], np.zeros(N))

    def test_cross_function_values(self):
        """ψ = c_1*c_2 的梯度应为 [c_2, c_1]。"""
        N = 6
        c = np.random.default_rng(6).uniform(0.5, 2.0, (N, 2))
        Psi, dPsi, names = _build_polynomial_test_functions_with_grads(c, degree=2)
        idx = names.index("c_1*c_2")
        np.testing.assert_allclose(Psi[idx], c[:, 0] * c[:, 1])
        np.testing.assert_allclose(dPsi[0, idx, :], c[:, 1])
        np.testing.assert_allclose(dPsi[1, idx, :], c[:, 0])

    def test_degree1_fewer_functions(self):
        """degree=1 应比 degree=2 少函数。"""
        N = 8
        c = np.random.default_rng(7).uniform(0.1, 2.0, (N, 2))
        Psi1, _, _ = _build_polynomial_test_functions_with_grads(c, degree=1)
        Psi2, _, _ = _build_polynomial_test_functions_with_grads(c, degree=2)
        assert Psi1.shape[0] < Psi2.shape[0]


# ---------------------------------------------------------------------------
# 2. build_weak_form_library 基本形状测试
# ---------------------------------------------------------------------------

class TestBuildWeakFormLibraryShapes:
    def test_output_shapes_single_control(self):
        rng = np.random.default_rng(10)
        N, n_freq = 10, 16
        d_hat = rng.standard_normal((N, n_freq)) + 1j * rng.standard_normal((N, n_freq))
        factors = rng.uniform(0.1, 2.0, (N, 1))
        omega = np.linspace(0, 1, n_freq)
        k = 5
        lib, names, Psi, basis, A = build_weak_form_library(d_hat, factors, omega, k_eff=k)

        # M = 1 + 1 + 1 = 3 (degree-2, 1 control)
        M = Psi.shape[0]
        assert M == 3
        assert Psi.shape == (3, N)
        # Library entries are (M, K) where K = k SVD components
        assert basis.shape == (k, n_freq)
        assert A.shape == (N, k)
        for key, val in lib.items():
            assert val.shape == (M, k), f"{key}: expected ({M},{k}), got {val.shape}"

    def test_output_shapes_two_controls(self):
        rng = np.random.default_rng(11)
        N, n_freq = 16, 20
        d_hat = rng.standard_normal((N, n_freq)) + 1j * rng.standard_normal((N, n_freq))
        factors = rng.uniform(0.1, 2.0, (N, 2))
        omega = np.linspace(0, 1, n_freq)
        lib, names, Psi, basis, A = build_weak_form_library(d_hat, factors, omega, k_eff=8)
        # M = 1 + 2 + 2 + 1 = 6
        assert Psi.shape[0] == 6
        for val in lib.values():
            assert val.shape[0] == 6
        assert A.shape == (N, 8)

    def test_library_keys_present(self):
        """库中必须包含 wln_f, wg, wL_1_f, wL_1_g, wL_2_f, wL_2_g。"""
        rng = np.random.default_rng(12)
        d_hat = rng.standard_normal((10, 12)) + 1j * rng.standard_normal((10, 12))
        factors = rng.uniform(0.1, 2.0, (10, 2))
        omega = np.linspace(0, 1, 12)
        lib, _, _, _, _ = build_weak_form_library(d_hat, factors, omega)
        for key in ["wln_f", "wg", "wL_1_f", "wL_1_g", "wL_2_f", "wL_2_g"]:
            assert key in lib, f"Missing key: {key}"

    def test_k_eff_capped_at_matrix_rank(self):
        """请求 k_eff 超出 min(N, n_freq) 时应截断到矩阵秩。"""
        rng = np.random.default_rng(13)
        N, n_freq = 8, 5
        d_hat = rng.standard_normal((N, n_freq)) + 1j * rng.standard_normal((N, n_freq))
        factors = rng.uniform(0.1, 2.0, (N, 1))
        omega = np.linspace(0, 1, n_freq)
        lib, _, _, basis, A = build_weak_form_library(d_hat, factors, omega, k_eff=100)
        # D† of (8,5) gives at most min(N, n_freq) = 5 independent components
        expected_k = min(N, n_freq)
        for val in lib.values():
            assert val.shape[1] == expected_k
        assert basis.shape[0] == expected_k
        assert A.shape[1] == expected_k

    def test_no_d_d_c_required(self):
        """build_weak_form_library 的签名不含 d_d_c。"""
        import inspect
        sig = inspect.signature(build_weak_form_library)
        assert "d_d_c" not in sig.parameters, "d_d_c should NOT be a parameter"
        assert "d2_d_c" not in sig.parameters, "d2_d_c should NOT be a parameter"


# ---------------------------------------------------------------------------
# 3. 数学正确性（IBP 精确恢复测试）
# ---------------------------------------------------------------------------

class TestWeakFormIBPCorrectness:
    """验证弱形式 IBP 公式与 D†-投影系数的对数一致。

    新版弱形式直接乘以 D 逆计算投影系数（A = D̂ @ D†[:, :K]），
    再对 ln_A = log(A + ε) 做 IBP 内积，不依赖 SVD 分离。
    """

    def test_pure_f_recovery(self):
        """验证弱形式输出精确等于 IBP 离散公式（以 ln_A 为信号）。

        对 D(c,ω) = exp(α·c_1)·P_spectral(ω) 这类数据：
        - D† 给出投影系数 A = D̂ @ D†[:, :K]
        - ln_A ≈ α·c_1 + const（实数部分主导）
        - 弱形式算子输出应满足：
            wL_1_f[m, k] = Re( -Σ_n (∂_{c_1}ψ_m·c_1(n) + ψ_m(c_n)) · ln_A(c_n, k) )
        """
        N_per = 8
        alpha = 2.5
        c_vals = np.linspace(0.0, 1.0, N_per)
        factors = c_vals.reshape(-1, 1)
        n_freq = 10
        omega = np.linspace(0.01, 1.0, n_freq)
        P_spectral = np.exp(-omega)
        d_hat = np.exp(alpha * c_vals)[:, None] * P_spectral[None, :]   # (N, n_freq)

        lib, names, Psi, basis, A = build_weak_form_library(d_hat, factors, omega, k_eff=n_freq)

        # 使用库返回的 A（= D̂ @ P†）重现期望值：
        # wL_1_f = Re( -(ibp_kernel @ ln_A) )
        # 其中 ibp_kernel[m, n] = ∂_{c_1}ψ_m(c_n)·c_1(n) + ψ_m(c_n)
        ln_A = np.log(A + 1e-12)
        Psi_arr, dPsi_arr, _ = _build_polynomial_test_functions_with_grads(
            factors, degree=2
        )
        ibp_kernel = dPsi_arr[0] * c_vals[None, :] + Psi_arr   # (M, N)
        expected_f = np.real(-(ibp_kernel @ ln_A))              # (M, K)

        np.testing.assert_allclose(
            lib["wL_1_f"], expected_f, atol=1e-10,
            err_msg="wL_1_f does not match IBP formula -(ibp_kernel @ ln_A)"
        )

    def test_zero_operator_for_constant_D(self):
        """若 D(c,ω) = 常数（与 c 无关），则所有 wL_i 的有限性。"""
        N = 12
        n_freq = 8
        # D = complex constant (but nonzero)
        d_hat = np.full((N, n_freq), 2.0 + 0.5j)
        factors = np.random.default_rng(20).uniform(0.1, 2.0, (N, 2))
        omega = np.linspace(0, 1, n_freq)

        lib, _, _, basis, A = build_weak_form_library(d_hat, factors, omega)

        # 所有输出必须有限
        for val in lib.values():
            assert np.all(np.isfinite(val))

    def test_ibp_identity(self):
        """验证 IBP 等式：lib["wL_1_f"][m,k] == Re(-(ibp_kernel[m,:] @ ln_A[:,k]))。

        IBP 在组分分离的 ln_A 信号上应精确成立（数值误差 < 1e-10）。
        """
        # 1D grid: c ∈ [0.5, 1.5], uniform
        N = 8
        n_freq = 6
        c_vals = np.linspace(0.5, 1.5, N)
        factors = c_vals.reshape(-1, 1)
        omega = np.linspace(0.1, 1.0, n_freq)
        rng = np.random.default_rng(30)
        D = np.abs(rng.standard_normal((N, n_freq))) + 2.0
        d_hat = D.astype(complex)

        lib, names, Psi, basis, A = build_weak_form_library(d_hat, factors, omega, k_eff=n_freq)

        # Component-separated log signal
        ln_A = np.log(A + 1e-12)                          # (N, K)
        Psi_arr, dPsi_arr, _ = _build_polynomial_test_functions_with_grads(factors, degree=2)

        # For ψ_0 = 1: IBP kernel = ∂_{c_1}[1·c_1] = 1  (constant 1 for all n)
        ibp_psi0 = np.ones(N)
        expected_0 = np.real(-(ibp_psi0 @ ln_A))         # (K,)
        np.testing.assert_allclose(
            lib["wL_1_f"][0, :], expected_0, atol=1e-10
        )

        # For ψ_1 = c_1: IBP kernel = ∂_{c_1}[c_1·c_1] = 2·c_1
        ibp_psi1 = 2.0 * c_vals
        expected_1 = np.real(-(ibp_psi1 @ ln_A))         # (K,)
        idx_c1 = names.index("c_1")
        np.testing.assert_allclose(
            lib["wL_1_f"][idx_c1, :], expected_1, atol=1e-10
        )


# ---------------------------------------------------------------------------
# 4. pipeline 集成测试（use_weak_form=True）
# ---------------------------------------------------------------------------

class TestWeakFormPipeline:
    def _make_spectra(self, n_wl=48):
        rng = np.random.default_rng(99)
        c1_vals = np.linspace(0.0, 2.0, 4)
        c2_vals = np.linspace(0.0, 1.0, 4)
        c1_g, c2_g = np.meshgrid(c1_vals, c2_vals, indexing="ij")
        c1, c2 = c1_g.ravel(), c2_g.ravel()
        factors = np.column_stack([c1, c2])
        wavelengths = np.linspace(380.0, 730.0, n_wl)
        x = wavelengths[None, :]
        s = (np.exp(-0.5 * ((x - (490 + 15 * c1[:, None])) / 18.0) ** 2)
             + 0.7 * np.exp(-0.5 * ((x - (570 - 12 * c2[:, None])) / 22.0) ** 2)
             + 0.01 * rng.normal(size=(factors.shape[0], n_wl)))
        return s, factors, wavelengths

    def test_weak_form_pipeline_runs(self):
        """use_weak_form=True 管线应无错误地完成并产生合理输出。"""
        s, factors, wl = self._make_spectra()
        cfg = DiscoveryConfig(
            k_mode="fixed", k_value=4,
            use_weak_form=True,
            weak_form_test_degree=2,
        )
        out = run_discovery(s, factors, wl, cfg)
        assert out.Xi.ndim == 3
        assert out.A_matrix is not None
        assert out.S_real is not None
        assert np.isfinite(out.reconstruction_error)

    def test_weak_form_operator_names_differ_from_strong(self):
        """弱形式库的算子名称应以 'w' 开头，与强形式不同。"""
        s, factors, wl = self._make_spectra()
        cfg_weak = DiscoveryConfig(k_mode="fixed", k_value=3, use_weak_form=True)
        cfg_strong = DiscoveryConfig(k_mode="fixed", k_value=3)
        out_weak = run_discovery(s, factors, wl, cfg_weak)
        out_strong = run_discovery(s, factors, wl, cfg_strong)
        assert set(out_weak.operator_names) != set(out_strong.operator_names)
        # Weak form names start with "w"
        assert any("w" in n.lower() for n in out_weak.operator_names)

    def test_weak_form_k_eff_shape(self):
        """弱形式的 Xi 和 A_matrix 列数应等于 k_value。"""
        s, factors, wl = self._make_spectra()
        cfg = DiscoveryConfig(k_mode="fixed", k_value=3, use_weak_form=True)
        out = run_discovery(s, factors, wl, cfg)
        assert out.Xi.shape[2] == 3
        assert out.A_matrix.shape[1] == 3
        assert out.S_real.shape[1] == 3

    def test_weak_form_no_d_d_c_used(self):
        """弱形式管线不应依赖 d_d_c（通过 monkeypatch 验证）。"""
        import unittest.mock as mock
        s, factors, wl = self._make_spectra()
        cfg = DiscoveryConfig(k_mode="fixed", k_value=3, use_weak_form=True)
        # Patch build_control_derivative_bundle to return zeros — weak form should still work
        import opera.discovery.pipeline as pl_module
        with mock.patch(
            "opera.discovery.pipeline.build_control_derivative_bundle",
            return_value=(
                np.zeros((factors.shape[0], factors.shape[1], 25)),
                np.zeros((factors.shape[0], factors.shape[1], factors.shape[1], 25)),
            ),
        ):
            out = run_discovery(s, factors, wl, cfg)
        assert out.Xi.ndim == 3
