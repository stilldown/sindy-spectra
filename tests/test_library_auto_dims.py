import numpy as np

from opera.discovery.library import build_observable_library


def test_library_auto_dims_control_and_spectral():
    rng = np.random.default_rng(123)
    n_samples = 12
    n_freq = 20
    n_controls = 3
    n_spec_dims = 2

    d_hat = rng.normal(size=(n_samples, n_freq)) + 1j * rng.normal(size=(n_samples, n_freq))
    d_d_c = rng.normal(size=(n_samples, n_controls, n_freq)) + 1j * rng.normal(size=(n_samples, n_controls, n_freq))
    d2_d_c = rng.normal(size=(n_samples, n_controls, n_controls, n_freq)) + 1j * rng.normal(size=(n_samples, n_controls, n_controls, n_freq))
    factors = rng.uniform(size=(n_samples, n_controls))
    omega = rng.normal(size=(n_freq, n_spec_dims))

    theta, names, scales, tags = build_observable_library(
        d_hat=d_hat,
        d_d_c=d_d_c,
        d2_d_c=d2_d_c,
        omega=omega[:, 0], # 目前仅支持一维 omega
        factors=factors,
        include_c_times_s=True,
        include_iomega_control_terms=True,
        include_control_coupling=True,
    )

    # tags 只是每列的物理类别标签，长度应与 names 一致
    assert len(tags) == len(names)

    assert theta.shape[0] == n_samples * n_freq
    assert theta.shape[1] == len(names)

    assert "D̂" in names
    assert "iω·D̂" in names
    # 可能带有 c_j 因子，只要包含偏导符号即可
    assert any("∂D̂/∂c_1" in n for n in names)
    assert any("iω" in n and "∂²D̂/(∂c_2∂c_3)" in n for n in names)

    # 验证是否做好了比例归一化
    norms = np.linalg.norm(theta, axis=0)
    np.testing.assert_allclose(norms, np.ones(theta.shape[1]), rtol=1e-5)
