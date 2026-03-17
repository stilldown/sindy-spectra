"""验证算法的核心物理原理：Euler 算子是微分方程候选项，SINDy-PI 零空间发现隐式 ODE。

理论背景（来源：1.md）
--------------------
算法将光谱分解约束为满足隐式微分方程的组分：

    D̂(c, ω) ≈ Σ_k  A_k(c) · φ_k(ω)
    A_k(c)  = f_k(c) · exp(−i ω_k · g_k(c))

Euler 算子库（每列 = 微分方程候选项在所有样本上的取值）：
    ln_f     = Re(ln A_k)                      ← 零阶（含规范化常数 C）
    L1_j_f   = c_j · ∂(ln f_k)/∂c_j           ← 一阶 Euler 算子（无常数）
    Xi2_ij_f = c_i² ∂²(ln f_k)/∂c_i² + L1_i  ← 二阶 Euler 算子（含一阶修正项）

SINDy-PI 零空间找 ξ 使 Σ ξ_j Ω_j(ln_f) ≈ 0，即发现 f(c) 满足的隐式 ODE。

注意：ln_f 包含谱基规范化常数 C = ln‖φ_k‖，因此 L1_f ≠ ln_f（差一个常数），
但 L1_f 和 Xi2_f 都不含 C（因为它们是对 ln_f 求导后再乘 c）。

Euler ODE 恒等式：f = exp(−ε c^n)
--------------------------------------
    L1_f   = −n ε c^n        （= n · (ln_f − C)，无常数项）
    Xi2_f  = n · L1_f         ← 关键恒等式，与 n 直接对应
即：发现方程 Xi2_f − n · L1_f = 0 等价于 c²∂²(ln f)/∂c² = (n−1)c∂(ln f)/∂c
    → 解为 ∂(ln f)/∂c = A c^(n−1) → f = exp(Bc^n)。

具体情况：
    n=1 (Beer-Lambert):  Xi2_f = 1 · L1_f  ↔  c²∂²(ln f)/∂c² = 0
    n=2 (Gaussian):       Xi2_f = 2 · L1_f  ↔  c²∂²(ln f)/∂c² = c∂(ln f)/∂c
    n=3 (Cubic-exp):      Xi2_f = 3 · L1_f  ↔  c²∂²(ln f)/∂c² = 2c∂(ln f)/∂c
"""
from __future__ import annotations

import numpy as np
import pytest

from opera.discovery.pipeline_utils import construct_pure_library, build_direct_euler_library
from opera.discovery.types import DiscoveryConfig


# ---------------------------------------------------------------------------
# 辅助：构造精确解析光谱数据 f = exp(−ε c^power) · P(λ)
# ---------------------------------------------------------------------------

def _make_exp_power_data(power: int, eps: float = 0.5, n_c: int = 10, n_wl: int = 32):
    """构造 f(c) = exp(−eps * c^power) 的单组分精确光谱数据。

    零锚点 c=0：f(0) = 1（满足算法要求）。
    返回 (c, c_vals, d_hat, omega, dD_dc, d2D_dc2)。
    """
    c_vals = np.linspace(0.0, 1.5, n_c)
    f_vals = np.exp(-eps * c_vals ** power)
    wl = np.linspace(380.0, 730.0, n_wl)
    P = np.exp(-0.5 * ((wl - 520.0) / 25.0) ** 2)   # 固定纯光谱

    d_hat = np.fft.rfft(f_vals[:, None] * P[None, :], axis=1)
    omega = np.linspace(0.0, 1.0, d_hat.shape[1])

    # 解析一、二阶导数
    dD_dc = np.zeros((n_c, 1, d_hat.shape[1]), dtype=complex)
    d2D_dc2 = np.zeros((n_c, 1, 1, d_hat.shape[1]), dtype=complex)
    for idx in range(n_c):
        cn, fn = c_vals[idx], f_vals[idx]
        if cn < 1e-12 or power == 1:
            # 一阶：f'(c) = -eps*power*c^(power-1) * f(c)
            # 对于 c=0 且 power>1 时导数为 0
            df = -eps * power * (cn ** max(0, power - 1)) * fn if power == 1 else \
                 (-eps * power * cn ** (power - 1) * fn if cn > 0 else 0.0)
            d2f = (eps ** 2 * power ** 2 * cn ** (2 * (power - 1)) - eps * power * (power - 1) * cn ** max(0, power - 2)) * fn \
                  if cn > 1e-12 else 0.0
        else:
            df = -eps * power * cn ** (power - 1) * fn
            d2f = (eps ** 2 * power ** 2 * cn ** (2 * (power - 1))
                   - eps * power * (power - 1) * cn ** (power - 2)) * fn
        dD_dc[idx, 0, :] = np.fft.rfft(df * P)
        d2D_dc2[idx, 0, 0, :] = np.fft.rfft(d2f * P)

    c = c_vals.reshape(-1, 1)
    return c, c_vals, d_hat, omega, dD_dc, d2D_dc2


# ===========================================================================
# 1. Euler ODE 恒等式：Xi2_f = n · L1_f  for  f = exp(−ε c^n)
# ===========================================================================

class TestEulerODEPowerIdentity:
    """核心微分方程恒等式：Xi2_f / L1_f = n（非零锚点处）。

    物理含义：比值 n 唯一决定了 f(c) 是哪种次幂的指数函数。
    算法可以通过这个比值"发现"浓度响应的次幂阶数。
    """

    @pytest.mark.parametrize("power", [1, 2, 3])
    def test_xi2_over_l1_equals_power_svd(self, power):
        """SVD 路径：Xi2_f / L1_f = power（精确到 rtol=1e-4）。"""
        c, c_vals, d_hat, omega, dD_dc, d2D_dc2 = _make_exp_power_data(power)
        cfg = DiscoveryConfig(k_mode="fixed", k_value=1)
        lib, _, _, _ = construct_pure_library(d_hat, dD_dc, d2D_dc2, omega, c, cfg)

        non_anchor = c_vals > 1e-9
        ratio = lib["Xi2_c1c1_f"][non_anchor, 0] / lib["L1_c1_f"][non_anchor, 0]
        np.testing.assert_allclose(
            ratio, float(power),
            rtol=1e-4,
            err_msg=(
                f"f=exp(-eps*c^{power}): Xi2_f/L1_f should be {power} "
                f"(Euler ODE: c²∂²(ln f)/∂c² + c∂(ln f)/∂c = {power}·c∂(ln f)/∂c)"
            ),
        )

    @pytest.mark.parametrize("power", [1, 2, 3])
    def test_xi2_over_l1_equals_power_direct(self, power):
        """直接 Euler 路径（W=[1,-iω]）：Xi2_f / L1_f = power（内点，rtol 5%）。

        注：直接路径在样本集合端点处有 np.gradient 边界误差，因此仅检查中间段。
        """
        c, c_vals, d_hat, omega, dD_dc, d2D_dc2 = _make_exp_power_data(power)
        lib, _, _, _ = build_direct_euler_library(d_hat, dD_dc, d2D_dc2, omega, c)

        # 仅检查内部点（排除端点边界误差）
        interior = (c_vals > 1e-9) & (c_vals < 1.3)
        ratio = lib["Xi2_c1c1_f"][interior, 0] / lib["L1_c1_f"][interior, 0]
        np.testing.assert_allclose(
            ratio, float(power),
            rtol=0.05,
            err_msg=f"Direct Euler: Xi2_f/L1_f should be {power} for f=exp(-eps*c^{power})",
        )

    def test_xi2_minus_n_l1_is_zero(self):
        """SVD 路径：Xi2_f − n·L1_f = 0（精确到机器精度）。

        这是 SINDy-PI 零空间应该发现的方程：
        Xi2_c1c1_f − n · L1_c1_f = 0  ↔  c²∂²(ln f)/∂c² = (n−1)c∂(ln f)/∂c
        """
        for power in [1, 2, 3]:
            c, c_vals, d_hat, omega, dD_dc, d2D_dc2 = _make_exp_power_data(power)
            cfg = DiscoveryConfig(k_mode="fixed", k_value=1)
            lib, _, _, _ = construct_pure_library(d_hat, dD_dc, d2D_dc2, omega, c, cfg)
            residual = lib["Xi2_c1c1_f"][:, 0] - float(power) * lib["L1_c1_f"][:, 0]
            assert np.allclose(residual, 0.0, atol=1e-10), (
                f"f=exp(-eps*c^{power}): Xi2_f - {power}*L1_f should be 0, "
                f"max residual={np.max(np.abs(residual)):.2e}"
            )


# ===========================================================================
# 2. Euler 算子零锚点物理约束：所有 Euler 算子在 c=0 处精确为零
# ===========================================================================

class TestEulerOperatorZeroAtAnchor:
    """验证所有 Euler 导数算子在零浓度锚点处为零（c=0 → L_i=0，Xi2=0）。

    物理意义：c_i=0 时 Euler 缩放 c_i·∂/∂c_i 自然消失。
    这保证了零浓度空白样品不贡献任何方程约束（作为纯参考点）。
    """

    @pytest.mark.parametrize("power", [1, 2])
    def test_l1_zero_at_anchor_svd(self, power):
        c, c_vals, d_hat, omega, dD_dc, d2D_dc2 = _make_exp_power_data(power, n_c=8)
        cfg = DiscoveryConfig(k_mode="fixed", k_value=1)
        lib, _, _, _ = construct_pure_library(d_hat, dD_dc, d2D_dc2, omega, c, cfg)
        anchor = c_vals < 1e-10
        assert np.allclose(lib["L1_c1_f"][anchor, :], 0.0, atol=1e-10), \
            "L1_f must vanish at c=0 (Euler scaling: c·∂/∂c = 0 when c=0)"

    @pytest.mark.parametrize("power", [1, 2])
    def test_xi2_zero_at_anchor_svd(self, power):
        c, c_vals, d_hat, omega, dD_dc, d2D_dc2 = _make_exp_power_data(power, n_c=8)
        cfg = DiscoveryConfig(k_mode="fixed", k_value=1)
        lib, _, _, _ = construct_pure_library(d_hat, dD_dc, d2D_dc2, omega, c, cfg)
        anchor = c_vals < 1e-10
        assert np.allclose(lib["Xi2_c1c1_f"][anchor, :], 0.0, atol=1e-10), \
            "Xi2_f must vanish at c=0"

    @pytest.mark.parametrize("power", [1, 2])
    def test_l1_zero_at_anchor_direct(self, power):
        c, c_vals, d_hat, omega, dD_dc, d2D_dc2 = _make_exp_power_data(power, n_c=8)
        lib, _, _, _ = build_direct_euler_library(d_hat, dD_dc, d2D_dc2, omega, c)
        anchor = c_vals < 1e-10
        assert np.allclose(lib["L1_c1_f"][anchor, :], 0.0, atol=1e-10)


# ===========================================================================
# 3. Euler 算子定义恒等式：L1_f = c · ∂(ln f)/∂c
# ===========================================================================

class TestEulerOperatorDefinition:
    """验证 L1_f = c · ∂(ln f)/∂c（Euler 算子的定义恒等式）。

    此恒等式验证代码的算子计算路径（α = dA/A, L1 = c·α）是否
    与直接数值微分 ∂(ln f)/∂c 一致。
    """

    @pytest.mark.parametrize("power", [1, 2])
    def test_l1_matches_c_times_log_derivative(self, power):
        """L1_f[n] 应等于 c[n] * numerical_gradient(ln_f)[n]（内点，rtol 5%）。"""
        c, c_vals, d_hat, omega, dD_dc, d2D_dc2 = _make_exp_power_data(
            power, eps=0.6, n_c=12
        )
        cfg = DiscoveryConfig(k_mode="fixed", k_value=1)
        lib, _, _, _ = construct_pure_library(d_hat, dD_dc, d2D_dc2, omega, c, cfg)

        ln_f_vals = lib["ln_f"][:, 0]
        dc = c_vals[1] - c_vals[0]
        # 中心差分（内点更准确）
        d_ln_f = np.gradient(ln_f_vals, dc)
        expected_l1 = c_vals * d_ln_f

        # 仅比较远离边界的内部点（边界差分误差大）
        interior = (c_vals > 0.3) & (c_vals < 1.2)
        np.testing.assert_allclose(
            lib["L1_c1_f"][interior, 0],
            expected_l1[interior],
            rtol=0.05,
            err_msg=f"L1_f should equal c * d(ln_f)/dc (power={power})",
        )


# ===========================================================================
# 4. 二阶 Euler 公式结构：Xi2_f = c² · ∂²(ln f)/∂c² + L1_f
# ===========================================================================

class TestXi2FormulaStructure:
    """验证 Xi2_f = c² · β + L1_f（代码公式的数学结构正确性）。

    推导：
        β = ∂²(ln A)/∂c²  =  ∂²(ln f)/∂c² （纯粹二阶对数导数）
        Xi2_f = c² · β + L1_f = c²·∂²(ln f)/∂c² + c·∂(ln f)/∂c
                              = （Euler 二阶算子作用于 ln f）
    """

    def test_xi2_structure_beer_lambert(self):
        """Beer-Lambert f=exp(-εc)：β=0，故 Xi2_f = L1_f（无二阶项）。"""
        c, c_vals, d_hat, omega, dD_dc, d2D_dc2 = _make_exp_power_data(1, n_c=8)
        cfg = DiscoveryConfig(k_mode="fixed", k_value=1)
        lib, _, _, _ = construct_pure_library(d_hat, dD_dc, d2D_dc2, omega, c, cfg)
        # β = 0 because ∂²(-εc)/∂c² = 0 → Xi2_f = 0·c² + L1_f = L1_f
        np.testing.assert_allclose(
            lib["Xi2_c1c1_f"], lib["L1_c1_f"], atol=1e-10,
            err_msg="Beer-Lambert: Xi2_f should equal L1_f (β₁₁ = 0)",
        )

    def test_xi2_structure_gaussian(self):
        """Gaussian f=exp(-εc²)：β = ∂²(ln f)/∂c² = -2ε（常数）。

        Xi2_f = c²·(-2ε) + L1_f = L1_f + (L1_f) = 2·L1_f
        （因为 L1_f = -2εc² 且 c²·(-2ε) = L1_f）
        """
        c, c_vals, d_hat, omega, dD_dc, d2D_dc2 = _make_exp_power_data(2, n_c=10)
        cfg = DiscoveryConfig(k_mode="fixed", k_value=1)
        lib, _, _, _ = construct_pure_library(d_hat, dD_dc, d2D_dc2, omega, c, cfg)
        # Xi2_f - L1_f = c² · β = c² · (-2ε)
        # L1_f = -2εc², so c²·(-2ε) = L1_f
        # → Xi2_f - L1_f = L1_f → Xi2_f = 2 L1_f
        np.testing.assert_allclose(
            lib["Xi2_c1c1_f"], 2.0 * lib["L1_c1_f"], atol=1e-10,
            err_msg="Gaussian: Xi2_f should equal 2·L1_f",
        )

    def test_xi2_minus_l1_equals_c2_beta(self):
        """Xi2_f − L1_f = c² · ∂²(ln f)/∂c²（数值验证，内部点）。

        此恒等式验证代码的 beta 计算路径（beta = d²A/A - alpha²）正确。
        """
        c, c_vals, d_hat, omega, dD_dc, d2D_dc2 = _make_exp_power_data(2, eps=0.7, n_c=10)
        cfg = DiscoveryConfig(k_mode="fixed", k_value=1)
        lib, _, _, _ = construct_pure_library(d_hat, dD_dc, d2D_dc2, omega, c, cfg)

        ln_f_vals = lib["ln_f"][:, 0]
        dc = c_vals[1] - c_vals[0]
        # 二阶数值微分
        d2_ln_f = np.gradient(np.gradient(ln_f_vals, dc), dc)
        expected_c2_beta = c_vals ** 2 * d2_ln_f

        # Xi2_f - L1_f = c² · β
        diff = lib["Xi2_c1c1_f"][:, 0] - lib["L1_c1_f"][:, 0]

        # 内点比较（边界有数值微分误差）
        interior = (c_vals > 0.2) & (c_vals < 1.3)
        np.testing.assert_allclose(
            diff[interior], expected_c2_beta[interior],
            rtol=0.05,
            err_msg="Xi2_f - L1_f should equal c² · d²(ln_f)/dc²",
        )


# ===========================================================================
# 5. 两组分系统：零锚点约束与形状验证
# ===========================================================================

class TestTwoComponentSystem:
    """双组分系统：Euler 算子在零锚点处严格为零，库形状正确。"""

    def test_l1_xi2_zero_at_anchor_two_components(self):
        """任何组分数下，零锚点处 L1 和 Xi2 严格为零。"""
        rng = np.random.default_rng(13)
        n_c = 8
        c_vals = np.linspace(0.0, 2.0, n_c)
        n_wl = 48
        wl = np.linspace(380.0, 730.0, n_wl)
        P1 = np.exp(-0.5 * ((wl - 500) / 20) ** 2)
        P2 = np.exp(-0.5 * ((wl - 580) / 25) ** 2)
        eps1, eps2 = 1.0, 0.8
        spectra = (np.exp(-eps1 * c_vals[:, None]) * P1
                   + np.exp(-eps2 * c_vals[:, None]) * P2)

        c = c_vals.reshape(-1, 1)
        d_hat = np.fft.rfft(spectra, axis=1)
        omega = np.linspace(0.0, 1.0, d_hat.shape[1])

        # 精确导数（两组分叠加）
        dD_dc = np.zeros((n_c, 1, d_hat.shape[1]), dtype=complex)
        d2D_dc2 = np.zeros((n_c, 1, 1, d_hat.shape[1]), dtype=complex)
        for idx in range(n_c):
            cn = c_vals[idx]
            df = -eps1 * np.exp(-eps1 * cn) * P1 - eps2 * np.exp(-eps2 * cn) * P2
            d2f = (eps1 ** 2 * np.exp(-eps1 * cn) * P1
                   + eps2 ** 2 * np.exp(-eps2 * cn) * P2)
            dD_dc[idx, 0, :] = np.fft.rfft(df)
            d2D_dc2[idx, 0, 0, :] = np.fft.rfft(d2f)

        cfg = DiscoveryConfig(k_mode="fixed", k_value=2)
        lib, _, _, _ = construct_pure_library(d_hat, dD_dc, d2D_dc2, omega, c, cfg)

        anchor = c_vals < 1e-10
        assert np.allclose(lib["L1_c1_f"][anchor, :], 0.0, atol=1e-10), \
            "L1_f must be exactly 0 at c=0 (two-component system)"
        assert np.allclose(lib["Xi2_c1c1_f"][anchor, :], 0.0, atol=1e-10), \
            "Xi2_f must be exactly 0 at c=0 (two-component system)"

        # 库形状 (N, K=2)
        for key, val in lib.items():
            assert val.shape == (n_c, 2), f"{key}: expected ({n_c}, 2), got {val.shape}"

