"""Tests for joint_diag.py – 谱-物理联合解析流算法（1.md 算法逻辑）。"""
import numpy as np
import pytest

from opera.discovery.joint_diag import (
    build_euler_operators,
    separate_fg,
    separate_all_operators_fg,
    joint_diagonalize,
    joint_diag_residual,
    build_physical_candidate_library,
    sindy_pi_nullspace,
    run_joint_diag_pipeline,
)


# ---------------------------------------------------------------------------
# 辅助：生成合成数据
# ---------------------------------------------------------------------------

def _make_synthetic_data(n_samples=12, n_freq=16, n_controls=2, seed=42):
    rng = np.random.default_rng(seed)
    d_hat = rng.standard_normal((n_samples, n_freq)) + 1j * rng.standard_normal((n_samples, n_freq))
    d_d_c = rng.standard_normal((n_samples, n_controls, n_freq)) + 1j * rng.standard_normal(
        (n_samples, n_controls, n_freq)
    )
    d2_d_c = rng.standard_normal((n_samples, n_controls, n_controls, n_freq)) + 1j * rng.standard_normal(
        (n_samples, n_controls, n_controls, n_freq)
    )
    factors = rng.uniform(0.1, 2.0, size=(n_samples, n_controls))
    omega = np.linspace(0, 1, n_freq)
    return d_hat, d_d_c, d2_d_c, factors, omega


# ---------------------------------------------------------------------------
# 阶段 1：Euler 算子测试
# ---------------------------------------------------------------------------

class TestBuildEulerOperators:
    def test_output_shapes(self):
        d_hat, d_d_c, d2_d_c, factors, omega = _make_synthetic_data(
            n_samples=8, n_freq=10, n_controls=2
        )
        ops = build_euler_operators(d_hat, d_d_c, d2_d_c, factors)

        # n_controls=2: 2 一阶(L_1,L_2) + 2 二阶(Xi_11,Xi_22) + 1 交叉(Xi_12) = 5 个算子
        assert len(ops) == 5
        for name, arr in ops.items():
            assert arr.shape == (8, 10), f"{name} 形状错误: {arr.shape}"

    def test_first_order_keys(self):
        d_hat, d_d_c, d2_d_c, factors, _ = _make_synthetic_data(n_controls=2)
        ops = build_euler_operators(d_hat, d_d_c, d2_d_c, factors)
        assert "L_1" in ops
        assert "L_2" in ops

    def test_second_order_keys(self):
        d_hat, d_d_c, d2_d_c, factors, _ = _make_synthetic_data(n_controls=2)
        ops = build_euler_operators(d_hat, d_d_c, d2_d_c, factors)
        assert "Xi_11" in ops
        assert "Xi_22" in ops
        assert "Xi_12" in ops

    def test_three_controls(self):
        d_hat, d_d_c, d2_d_c, factors, _ = _make_synthetic_data(n_controls=3)
        ops = build_euler_operators(d_hat, d_d_c, d2_d_c, factors)
        # 3 一阶 + 3 二阶 + 3 交叉 = 9 个算子
        assert len(ops) == 9

    def test_element_wise_formula(self):
        """验证 L_1 = D^{-1} ⊙ (c_1 * dD/dc_1) 的逐点正确性。"""
        n_samples, n_freq, n_controls = 5, 8, 1
        rng = np.random.default_rng(0)
        d_hat = rng.standard_normal((n_samples, n_freq)) + 0.1  # 避免接近零
        d_d_c = rng.standard_normal((n_samples, 1, n_freq))
        d2_d_c = rng.standard_normal((n_samples, 1, 1, n_freq))
        factors = rng.uniform(0.5, 2.0, size=(n_samples, 1))

        ops = build_euler_operators(d_hat, d_d_c, d2_d_c, factors)
        L1_expected = (1.0 / (d_hat + 1e-12)) * (factors[:, 0:1] * d_d_c[:, 0, :])
        np.testing.assert_allclose(ops["L_1"], L1_expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# 阶段 2：f/g 频域分离测试
# ---------------------------------------------------------------------------

class TestSeparateFG:
    def test_output_shapes(self):
        n_samples, n_freq = 10, 20
        rng = np.random.default_rng(1)
        op = rng.standard_normal((n_samples, n_freq)) + 1j * rng.standard_normal((n_samples, n_freq))
        omega = np.linspace(0, 1, n_freq)
        tf, tg = separate_fg(op, omega)
        assert tf.shape == (n_samples,)
        assert tg.shape == (n_samples,)

    def test_recovery_pure_f(self):
        """若算子为纯 f 分量（与 ω 无关），则 term_f 应还原输入，term_g 应接近零。"""
        n_samples, n_freq = 8, 30
        rng = np.random.default_rng(2)
        omega = np.linspace(0, 1, n_freq)
        # 构造 op(c, ω) = f_true(c)（与 ω 无关）
        f_true = rng.standard_normal(n_samples)
        op = np.outer(f_true, np.ones(n_freq))  # (N, n_freq)

        tf, tg = separate_fg(op, omega)
        np.testing.assert_allclose(tf, f_true, atol=1e-8)
        np.testing.assert_allclose(tg, np.zeros(n_samples), atol=1e-8)

    def test_recovery_pure_g(self):
        """若算子为纯 g 分量（op = -iω * g_true），则 term_g 应还原输入，term_f 应接近零。"""
        n_samples, n_freq = 8, 30
        rng = np.random.default_rng(3)
        omega = np.linspace(0.01, 1, n_freq)
        # 构造 op(c, ω) = (-iω) * g_true(c)
        g_true = rng.standard_normal(n_samples)
        op = np.outer(g_true, -1j * omega)  # (N, n_freq)

        tf, tg = separate_fg(op, omega)
        np.testing.assert_allclose(tf, np.zeros(n_samples), atol=1e-8)
        np.testing.assert_allclose(tg, g_true, atol=1e-8)

    def test_mixed_recovery(self):
        """混合 f/g 情况下能分离。"""
        n_samples, n_freq = 10, 40
        rng = np.random.default_rng(4)
        omega = np.linspace(0.0, 1.0, n_freq)
        f_true = rng.standard_normal(n_samples)
        g_true = rng.standard_normal(n_samples)
        # op(c, ω) = f_true(c) + (-iω) * g_true(c)
        op = np.outer(f_true, np.ones(n_freq)) + np.outer(g_true, -1j * omega)

        tf, tg = separate_fg(op, omega)
        np.testing.assert_allclose(tf, f_true, atol=1e-8)
        np.testing.assert_allclose(tg, g_true, atol=1e-8)

    def test_separate_all_operators_fg_shapes(self):
        d_hat, d_d_c, d2_d_c, factors, omega = _make_synthetic_data()
        ops = build_euler_operators(d_hat, d_d_c, d2_d_c, factors)
        f_ops, g_ops = separate_all_operators_fg(ops, omega)
        assert len(f_ops) == len(ops)
        assert len(g_ops) == len(ops)
        for v in f_ops.values():
            assert v.shape == (d_hat.shape[0],)


# ---------------------------------------------------------------------------
# 阶段 3：联合对角化测试
# ---------------------------------------------------------------------------

class TestJointDiagonalize:
    def test_output_shapes(self):
        d_hat, d_d_c, d2_d_c, factors, omega = _make_synthetic_data()
        ops = build_euler_operators(d_hat, d_d_c, d2_d_c, factors)
        op_list = list(ops.values())
        k = 3
        B, Lambda = joint_diagonalize(op_list, k=k)
        assert B.shape == (d_hat.shape[1], k)
        assert Lambda.shape == (d_hat.shape[0], k)

    def test_B_column_orthogonality(self):
        """联合对角化基的各列应正交（来自特征分解）。"""
        d_hat, d_d_c, d2_d_c, factors, omega = _make_synthetic_data()
        ops = build_euler_operators(d_hat, d_d_c, d2_d_c, factors)
        B, _ = joint_diagonalize(list(ops.values()), k=4)
        gram = B.conj().T @ B  # 应近似为 I_k
        np.testing.assert_allclose(
            np.abs(gram), np.eye(4), atol=1e-10,
            err_msg="联合对角化基未正交"
        )

    def test_k_capped_at_n_freq(self):
        """k 不能超过 n_freq。"""
        d_hat, d_d_c, d2_d_c, factors, omega = _make_synthetic_data(n_freq=5)
        ops = build_euler_operators(d_hat, d_d_c, d2_d_c, factors)
        B, Lambda = joint_diagonalize(list(ops.values()), k=100)
        assert B.shape[1] <= 5

    def test_residual_is_nonneg(self):
        d_hat, d_d_c, d2_d_c, factors, omega = _make_synthetic_data()
        ops = build_euler_operators(d_hat, d_d_c, d2_d_c, factors)
        op_list = list(ops.values())
        B, _ = joint_diagonalize(op_list, k=3)
        res = joint_diag_residual(op_list, B)
        assert res >= 0.0


# ---------------------------------------------------------------------------
# 阶段 4：物理候选库测试
# ---------------------------------------------------------------------------

class TestBuildPhysicalCandidateLibrary:
    def test_output_shapes_single_control(self):
        N = 20
        c = np.random.default_rng(0).uniform(0.1, 2.0, (N, 1))
        Theta, names = build_physical_candidate_library(c, max_degree=2)
        # 期望：[1, c1, c1^2] = 3 列
        assert Theta.shape == (N, 3), f"实际列数：{Theta.shape[1]}"
        assert len(names) == 3

    def test_output_shapes_two_controls(self):
        N = 15
        c = np.random.default_rng(1).uniform(0.1, 2.0, (N, 2))
        Theta, names = build_physical_candidate_library(c, max_degree=2)
        # 期望：[1, c1, c2, c1^2, c2^2, c1*c2] = 6 列
        assert Theta.shape[1] == 6
        assert "1" in names
        assert "c1" in names
        assert "c2" in names
        assert "c1*c2" in names

    def test_include_log(self):
        N = 10
        c = np.abs(np.random.default_rng(2).standard_normal((N, 1))) + 0.5
        Theta, names = build_physical_candidate_library(c, max_degree=1, include_log=True)
        assert any("ln" in n for n in names)

    def test_include_inv(self):
        N = 10
        c = np.random.default_rng(3).uniform(0.1, 2.0, (N, 1))
        Theta, names = build_physical_candidate_library(c, max_degree=1, include_inv=True)
        assert any("1/(1+" in n for n in names)

    def test_theta_first_column_ones(self):
        """第一列应为全 1（常数项）。"""
        N = 12
        c = np.random.default_rng(4).uniform(0.5, 2.0, (N, 2))
        Theta, _ = build_physical_candidate_library(c)
        np.testing.assert_allclose(Theta[:, 0], np.ones(N))


# ---------------------------------------------------------------------------
# 阶段 5：SINDy-PI 零空间求解测试
# ---------------------------------------------------------------------------

class TestSindyPiNullspace:
    def test_output_shapes(self):
        N, J, K = 20, 6, 3
        rng = np.random.default_rng(10)
        Theta = rng.standard_normal((N, J))
        Lambda = rng.standard_normal((N, K)) + 1j * rng.standard_normal((N, K))
        Xi, residuals = sindy_pi_nullspace(Theta, Lambda, sparsity_threshold=0.01)
        assert Xi.shape == (J, K)
        assert len(residuals) == K

    def test_sparsity_applied(self):
        """以较高阈值运行时，输出应有明显的零值。"""
        N, J, K = 30, 8, 2
        rng = np.random.default_rng(20)
        Theta = rng.standard_normal((N, J))
        Lambda = rng.standard_normal((N, K))
        Xi, _ = sindy_pi_nullspace(Theta, Lambda, sparsity_threshold=0.5)
        # 至少有一个零
        assert np.sum(Xi == 0.0) > 0


# ---------------------------------------------------------------------------
# 完整管线测试
# ---------------------------------------------------------------------------

class TestRunJointDiagPipeline:
    def test_pipeline_runs_without_error(self):
        d_hat, d_d_c, d2_d_c, factors, omega = _make_synthetic_data(
            n_samples=16, n_freq=12, n_controls=2, seed=99
        )
        result = run_joint_diag_pipeline(
            d_hat, d_d_c, d2_d_c, omega, factors, k=3
        )
        # 检查所有键都存在
        for key in ["operators", "B", "Lambda", "f_ops", "g_ops",
                    "Theta", "theta_names", "Xi", "residuals", "jd_residual"]:
            assert key in result, f"缺少键：{key}"

    def test_pipeline_output_shapes(self):
        n_samples, n_freq, n_controls, k = 12, 10, 2, 3
        d_hat, d_d_c, d2_d_c, factors, omega = _make_synthetic_data(
            n_samples=n_samples, n_freq=n_freq, n_controls=n_controls
        )
        result = run_joint_diag_pipeline(
            d_hat, d_d_c, d2_d_c, omega, factors, k=k
        )
        assert result["B"].shape == (n_freq, k)
        assert result["Lambda"].shape == (n_samples, k)
        J = result["Theta"].shape[1]
        assert result["Xi"].shape == (J, k)
        assert len(result["residuals"]) == k

    def test_pipeline_jd_residual_finite(self):
        d_hat, d_d_c, d2_d_c, factors, omega = _make_synthetic_data()
        result = run_joint_diag_pipeline(d_hat, d_d_c, d2_d_c, omega, factors, k=2)
        assert np.isfinite(result["jd_residual"])

    def test_pipeline_three_controls(self):
        d_hat, d_d_c, d2_d_c, factors, omega = _make_synthetic_data(
            n_samples=20, n_freq=14, n_controls=3
        )
        result = run_joint_diag_pipeline(
            d_hat, d_d_c, d2_d_c, omega, factors, k=2
        )
        assert result["B"].shape[1] == 2
