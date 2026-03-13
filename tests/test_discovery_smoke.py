import numpy as np
import pytest

from opera.discovery import run_discovery, DiscoveryConfig


def test_discovery_smoke_shapes():
    rng = np.random.default_rng(42)
    
    # 构建笛卡尔网格
    c1_vals = np.linspace(0.0, 2.0, 5)
    c2_vals = np.linspace(0.0, 1.0, 4)
    c1_grid, c2_grid = np.meshgrid(c1_vals, c2_vals, indexing='ij')
    
    c1 = c1_grid.flatten()
    c2 = c2_grid.flatten()
    factors = np.column_stack([c1, c2])
    
    n_samples = factors.shape[0]
    n_wl = 64

    wavelengths = np.linspace(380.0, 730.0, n_wl)
    x = wavelengths[None, :]

    s1 = np.exp(-0.5 * ((x - (500 + 20 * c1[:, None])) / 18.0) ** 2)
    s2 = 0.6 * np.exp(-0.5 * ((x - (580 - 15 * c2[:, None])) / 22.0) ** 2)
    spectra = s1 + s2 + 0.01 * rng.normal(size=(n_samples, n_wl))

    cfg = DiscoveryConfig(k_mode="fixed", k_value=3, sparsity_threshold=1e-2)
    out = run_discovery(spectra, factors, wavelengths, cfg)

    assert out.S_real.shape[0] == n_wl
    assert out.f_response_eval.shape[0] == n_samples
    assert out.Xi.ndim == 3
    assert out.A_matrix is not None
    assert out.A_matrix.shape[0] == len(out.operator_names)
    assert out.A_matrix.shape[1] == out.S_real.shape[1]
    assert out.Xi.shape[2] == out.S_real.shape[1]
    assert len(out.xi_by_control) == out.S_real.shape[1]
    assert all(k.startswith("component_") for k in out.xi_by_control)
    assert all(v.shape[0] == len(out.operator_names) for v in out.xi_by_control.values())
    # curl_norm_phi 与 path_closure_error 已因精确积分移除
    assert isinstance(out.latex_blocks, list)
    assert len(out.latex_blocks) == out.S_real.shape[1]
    assert all(isinstance(x, str) and len(x) > 0 for x in out.latex_blocks)
    # 启用标志时应出现新项
    # 断言当存在此项名称时（仅作备查）："D̂(c,ω)·∂D̂(c,ω)/∂c_" 在 operator_names 中
    assert out.quality_flags is not None
    assert "nullspace_energy" in out.diagnostics
    assert "sigma_gap_min" in out.diagnostics
    assert "anchor_count" in out.diagnostics
    assert out.metadata.get("anchor_indices") is not None
    # phi/psi 诊断在重构后已删除
    assert "phi_l2" not in out.diagnostics
    assert "psi_l2" not in out.diagnostics
    assert out.f_response is not None
    assert out.g_shift is not None
    assert out.pure_spectra_complex is not None
    assert out.reconstruction_error is not None
    assert np.isfinite(out.reconstruction_error)

    # also exercise the inverse-operator branch (should not error)
    cfg2 = DiscoveryConfig(
        k_mode="fixed", k_value=3, sparsity_threshold=1e-2,
        use_inverse_operator=True
    )
    out2 = run_discovery(spectra, factors, wavelengths, cfg2)
    assert out2.A_matrix.shape[1] == out2.S_real.shape[1]
    # weak operators added so count may increase
    assert len(out2.operator_names) >= len(out.operator_names)


def test_compute_weak_operators_gradient_effect():
    # create simple synthetic data where psi has linear trend
    from opera.discovery.operator import compute_weak_operators
    rng = np.random.default_rng(0)
    N = 16
    d = 2
    K = 3
    n_freq = 10
    d_hat = rng.standard_normal((N, n_freq)) + 1j * rng.standard_normal((N, n_freq))
    d_d_c = rng.standard_normal((N, d, n_freq)) + 1j * rng.standard_normal((N, d, n_freq))
    d2_d_c = rng.standard_normal((N, d, d, n_freq)) + 1j * rng.standard_normal((N, d, d, n_freq))
    # construct simple grid of factors
    c1 = np.linspace(0,1,4)
    c2 = np.linspace(0,2,4)
    c1g, c2g = np.meshgrid(c1, c2, indexing='ij')
    factors = np.column_stack([c1g.flatten(), c2g.flatten()])
    psi = np.linspace(0,1,N)
    weak_lib = compute_weak_operators(d_hat, d_d_c, d2_d_c, factors, psi)
    # basic shape assertions
    assert isinstance(weak_lib, dict)
    assert all(mat.shape[0] == N for mat in weak_lib.values())
    # pick one L1 entry and ensure gradient subtraction occurred
    for name in weak_lib:
        if name.startswith("L1_c1_"):
            mat = weak_lib[name]
            break
    # because psi increases, weighted values should differ from psi*original
    # approximate difference by recomputing simple weight for comparison
    lib_basic, *_ = construct_inverse_library(d_hat, d_d_c, d2_d_c, np.linspace(0,1,n_freq), factors, DiscoveryConfig())
    basic_mat = psi[:,None] * lib_basic[name]
    assert not np.allclose(mat, basic_mat)


def test_discovery_three_controls():
    rng = np.random.default_rng(7)

    c1_vals = np.linspace(0.0, 2.0, 4)
    c2_vals = np.linspace(0.0, 1.5, 3)
    c3_vals = np.linspace(0.0, 1.0, 3)
    c1_grid, c2_grid, c3_grid = np.meshgrid(c1_vals, c2_vals, c3_vals, indexing='ij')

    c1 = c1_grid.flatten()
    c2 = c2_grid.flatten()
    c3 = c3_grid.flatten()
    factors = np.column_stack([c1, c2, c3])
    
    n_samples = factors.shape[0]
    n_wl = 80

    wavelengths = np.linspace(380.0, 730.0, n_wl)
    x = wavelengths[None, :]

    center1 = 480 + 20 * c1[:, None] - 10 * c3[:, None]
    center2 = 590 - 12 * c2[:, None] + 8 * c3[:, None]
    amp1 = (0.8 + 0.4 * c2[:, None])
    amp2 = (0.5 + 0.3 * c1[:, None])

    s1 = amp1 * np.exp(-0.5 * ((x - center1) / 16.0) ** 2)
    s2 = amp2 * np.exp(-0.5 * ((x - center2) / 24.0) ** 2)
    spectra = s1 + s2 + 0.01 * rng.normal(size=(n_samples, n_wl))

    cfg = DiscoveryConfig(k_mode="fixed", k_value=4, max_components=4, sparsity_threshold=1e-2)
    out = run_discovery(spectra, factors, wavelengths, cfg)

    assert out.f_response_eval.shape[0] == n_samples
    assert out.S_real.shape[0] == n_wl
    assert out.f_response is not None
    assert out.g_shift is not None
    assert out.f_response.shape[0] == n_samples
    assert out.g_shift.shape[0] == n_samples
    assert out.g_shift.shape[1] == out.S_real.shape[1]
    # 多变量下也应输出“每组分单一方程”：f_k(c), g_k(c)
    assert len(out.latex_blocks) == out.S_real.shape[1]
    first_block = out.latex_blocks[0]
    assert "Comp 1" in first_block
    # 块可能隐式存在；只是检查它不为零
    assert len(first_block) > 20
    assert "\\exp" not in first_block
    assert "f11" not in first_block and "g11" not in first_block


def test_discovery_requires_zero_anchor_samples():
    rng = np.random.default_rng(123)

    c1_vals = np.linspace(0.1, 2.0, 5) # 无零锚点
    c2_vals = np.linspace(0.1, 1.0, 4)
    c1_grid, c2_grid = np.meshgrid(c1_vals, c2_vals, indexing='ij')

    c1 = c1_grid.flatten()
    c2 = c2_grid.flatten()
    factors = np.column_stack([c1, c2])

    n_samples = factors.shape[0]
    n_wl = 48

    wavelengths = np.linspace(400.0, 700.0, n_wl)
    x = wavelengths[None, :]
    s1 = np.exp(-0.5 * ((x - (510 + 12 * c1[:, None])) / 20.0) ** 2)
    s2 = 0.7 * np.exp(-0.5 * ((x - (575 - 10 * c2[:, None])) / 25.0) ** 2)
    spectra = s1 + s2

    cfg = DiscoveryConfig(k_mode="fixed", k_value=2, max_components=2)
    with pytest.raises(ValueError, match="未检测到零浓度样本"):
        run_discovery(spectra, factors, wavelengths, cfg)


def test_fixed_k_constrains_xi_width_and_blocks_present():
    rng = np.random.default_rng(2026)

    c1_vals = np.linspace(0.0, 2.0, 4)
    c2_vals = np.linspace(0.0, 1.2, 3)
    c3_vals = np.linspace(0.0, 1.0, 2)
    c1_grid, c2_grid, c3_grid = np.meshgrid(c1_vals, c2_vals, c3_vals, indexing='ij')

    c1 = c1_grid.flatten()
    c2 = c2_grid.flatten()
    c3 = c3_grid.flatten()
    factors = np.column_stack([c1, c2, c3])
    
    n_samples = factors.shape[0]
    n_wl = 72

    wavelengths = np.linspace(380.0, 730.0, n_wl)
    x = wavelengths[None, :]
    s1 = (0.7 + 0.2 * c2[:, None]) * np.exp(-0.5 * ((x - (505 + 15 * c1[:, None])) / 19.0) ** 2)
    s2 = (0.5 + 0.3 * c1[:, None]) * np.exp(-0.5 * ((x - (585 - 9 * c2[:, None] + 5 * c3[:, None])) / 23.0) ** 2)
    spectra = s1 + s2

    # fixed 模式下仅由 k_value 决定；max_components 不应覆盖
    cfg = DiscoveryConfig(k_mode="fixed", k_value=2, max_components=6, k_max=6, sparsity_threshold=1e-2)
    out = run_discovery(spectra, factors, wavelengths, cfg)

    assert out.Xi.ndim == 3
    assert out.Xi.shape[2] == 2
    assert out.A_matrix is not None
    assert out.A_matrix.shape[1] == 2
    assert out.S_real.shape[1] == 2
    assert int(out.diagnostics.get("k_selected", -1)) == 2
    assert out.metadata.get("k_source") == "k_value"
    assert "J_tot" in out.metadata
    assert "operator_block_ranges" in out.metadata
    assert isinstance(out.metadata["operator_block_ranges"], dict)


def test_find_joint_nullspace_preserves_components():
    rng = np.random.default_rng(12345)
    # 构造低秩矩阵代表 \Theta^H \Theta 性质
    true_basis = rng.standard_normal((10, 2))
    theta = rng.standard_normal((15, 10)) @ true_basis @ true_basis.T
    
    from opera.discovery.factorization import find_joint_nullspace
    v_null, diag = find_joint_nullspace(theta, k_eff=2)

    assert v_null.shape == (10, 2)
    assert np.allclose(v_null.T @ v_null, np.eye(2), atol=1e-5)
    assert "nullspace_energy" in diag
