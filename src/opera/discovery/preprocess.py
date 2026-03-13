from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

def validate_inputs(spectra: np.ndarray, factors: np.ndarray, wavelengths: np.ndarray) -> None:
    if spectra.ndim != 2:
        raise ValueError("spectra 必须为二维数组: (n_samples, n_wavelengths)")
    if factors.ndim != 2:
        raise ValueError("factors 必须为二维数组: (n_samples, n_controls)")
    if wavelengths.ndim != 1:
        raise ValueError("wavelengths 必须为一维数组")
    if spectra.shape[0] != factors.shape[0]:
        raise ValueError("spectra 与 factors 的样本数必须一致")
    if spectra.shape[1] != wavelengths.shape[0]:
        raise ValueError("spectra 的光谱长度必须等于 wavelengths 长度")


def compute_analytic_signal(spectra: np.ndarray) -> np.ndarray:
    """去均值并计算复解析信号 D_H"""
    spectra_detrend = spectra - np.mean(spectra, axis=1, keepdims=True)
    return hilbert(spectra_detrend, axis=1)

def compute_fourier_tensor(spectra: np.ndarray, wavelengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """计算归一化频率轴的 Fourier 张量。此处返回的是 rfft 下的频域数据（等价于只保留非负频率）"""
    # 按照新架构，我们更推荐在频域用 rfft 实现半频带截断，这与解析信号的频域表现完全一致。
    wl = np.asarray(wavelengths, dtype=float)
    # 去直流
    spectra_detrend = spectra - np.mean(spectra, axis=1, keepdims=True)
    
    # 解析信号的频域等效：只取非负频率 (rfft) 并将正频率振幅翻倍
    d_hat = np.fft.rfft(spectra_detrend, axis=1)
    # 不强制翻倍也可以，因为后续会执行列级范数归一化，尺度不影响 SVD 的零空间。
    
    n_freq = d_hat.shape[1]
    # 将频率轴严格归一化到 [0, 1]
    omega_bar = np.linspace(0.0, 1.0, n_freq)
    
    return d_hat, omega_bar


def _detect_cartesian_grid(factors: np.ndarray, tol: float = 1e-5) -> tuple[bool, list[np.ndarray], tuple[int, ...], np.ndarray]:
    """Detect if the scatter points form a complete cartesian grid and return necessary metadata.

    Returns:
        is_grid: bool
        unique_vals: list of 1D arrays for each dimension
        grid_shape: tuple of ints (size along each dim)
        sort_idx: indices that sort the flattened factors into grid order
    """
    n_samples, n_dims = factors.shape
    unique_vals = []
    
    for dim in range(n_dims):
        vals = np.unique(factors[:, dim])
        # 对数值进行排序
        vals.sort()
        unique_vals.append(vals)
        
    grid_shape = tuple(len(u) for u in unique_vals)
    expected_samples = np.prod(grid_shape)
    
    if expected_samples != n_samples:
        return False, [], (), np.array([])
        
    # 检查点是否精确匹配网格
    mesh = np.meshgrid(*unique_vals, indexing='ij')
    mesh_points = np.stack([m.flatten() for m in mesh], axis=-1)
    
    # 我们需要找出原始因子到排序网格点的映射
    # 使用 lexsort：最后一列先排序
    sort_idx = np.lexsort([factors[:, i] for i in range(n_dims-1, -1, -1)])
    sorted_factors = factors[sort_idx]
    
    if np.max(np.abs(sorted_factors - mesh_points)) > tol:
        return False, [], (), np.array([])
        
    return True, unique_vals, grid_shape, sort_idx


def estimate_control_derivatives_scattered(
    field: np.ndarray,
    factors: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """在散点控制空间上估计偏导，返回形状 (n_samples, n_controls, n_freq)。"""
    y = np.asarray(field)
    c = np.asarray(factors, dtype=float)
    n_samples, n_freq = y.shape
    n_controls = c.shape[1]
    
    is_grid, uniq_vals, grid_shape, sort_idx = _detect_cartesian_grid(c)
    
    if is_grid:
        # 将数据重塑为网格形式
        y_sorted = y[sort_idx]
        y_grid = y_sorted.reshape(*grid_shape, n_freq)
        
        # 沿控制方向计算梯度
        # np.gradient 需要间隔或坐标信息
        grads_grid = np.gradient(y_grid, *uniq_vals, axis=tuple(range(n_controls)))
        
        # 处理 n_controls == 1（返回单个数组）和 >1（返回数组列表）的情况
        if n_controls == 1:
            grads_grid = [grads_grid]
            
        grads = np.zeros((n_samples, n_controls, n_freq), dtype=complex)
        
        # 反转排序索引
        inv_sort_idx = np.empty_like(sort_idx)
        inv_sort_idx[sort_idx] = np.arange(n_samples)
        
        for i in range(n_controls):
            grads[:, i, :] = grads_grid[i].reshape(n_samples, n_freq)[inv_sort_idx]
            
        return grads
        
    # 非网格情形不再提供 kNN 估计，强制要求笛卡尔网格以使用中心差分
    raise ValueError("控制变量必须构成等间距笛卡尔网格，以使用中心差分估计导数")


def estimate_control_second_derivatives_scattered(
    first_derivatives: np.ndarray,
    factors: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """基于一阶导估计控制二阶导。

    输入 first_derivatives 形状为 (n_samples, n_controls, n_freq)，
    输出 second_derivatives 形状为 (n_samples, n_controls, n_controls, n_freq)，
    其中 second_derivatives[:, i, j, :] 对应 ∂²/∂c_i∂c_j。
    """
    d1 = np.asarray(first_derivatives)
    n_samples, n_controls, n_freq = d1.shape
    d2 = np.zeros((n_samples, n_controls, n_controls, n_freq), dtype=complex)

    is_grid, uniq_vals, grid_shape, sort_idx = _detect_cartesian_grid(np.asarray(factors))
    
    if is_grid:
        inv_sort_idx = np.empty_like(sort_idx)
        inv_sort_idx[sort_idx] = np.arange(n_samples)
        
        for i in range(n_controls):
            d1_i = d1[:, i, :]
            # 排序并重塑
            d1_i_sorted = d1_i[sort_idx]
            d1_i_grid = d1_i_sorted.reshape(*grid_shape, n_freq)
            
            grad_grid = np.gradient(d1_i_grid, *uniq_vals, axis=tuple(range(n_controls)))
            if n_controls == 1:
                grad_grid = [grad_grid]
                
            for j in range(n_controls):
                d2[:, i, j, :] = grad_grid[j].reshape(n_samples, n_freq)[inv_sort_idx]
                
        return d2

    # 非网格情形不再提供 kNN 估计
    raise ValueError("控制变量必须构成等间距笛卡尔网格，以使用中心差分估计导数")


def build_control_derivative_bundle(
    d_hat: np.ndarray,
    factors: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """统一构建一阶与二阶控制导数。"""
    c = np.asarray(factors, dtype=float)
    y = np.asarray(d_hat)
    n_samples, n_freq = y.shape
    n_controls = c.shape[1]

    # 允许存在全零锚点：对非零样本构网格做中心差分，锚点导数设为 0
    anchor_mask = np.all(np.isclose(c, 0.0), axis=1)
    
    is_fully_grid, _, _, _ = _detect_cartesian_grid(c)
    if is_fully_grid:
        keep_mask = np.ones(n_samples, dtype=bool)
    else:
        keep_mask = ~anchor_mask

    if not np.any(keep_mask):
        raise ValueError("全部样本都是零浓度，无法构建网格导数")

    c_keep = c[keep_mask]
    y_keep = y[keep_mask]

    d1_keep = estimate_control_derivatives_scattered(
        field=y_keep,
        factors=c_keep,
        eps=eps,
    )
    d2_keep = estimate_control_second_derivatives_scattered(
        first_derivatives=d1_keep,
        factors=c_keep,
        eps=eps,
    )

    # 回填到完整数组，锚点导数为 0
    d1 = np.zeros((n_samples, n_controls, n_freq), dtype=complex)
    d2 = np.zeros((n_samples, n_controls, n_controls, n_freq), dtype=complex)
    d1[keep_mask] = d1_keep
    d2[keep_mask] = d2_keep
    return d1, d2
