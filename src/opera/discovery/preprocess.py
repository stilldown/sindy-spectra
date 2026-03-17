"""预处理层：输入验证与频域张量计算。

数学背景
--------
设观测矩阵 S ∈ ℝ^{N×M}，其中 N 为样本数，M 为波长数。
每行 S(c, λ) 是在控制条件 c 下测量的光谱。

步骤 1 — 去直流::

    S̃(c, λ) = S(c, λ) - ⟨S(c, ·)⟩_λ

步骤 2 — 半频带 FFT（rfft）::

    D̂(c, ω) = Σ_{λ} S̃(c, λ) e^{-iωλ},   ω ∈ {0, 1/(M-1), …, 1}

得到频域张量 D̂ ∈ ℂ^{N×P}，P = M//2 + 1 为正频率数量。
归一化频率轴 ω̄ ∈ [0, 1] 消除对波长单位的依赖。

步骤 3 — 控制导数（等间距笛卡尔网格，中心差分）::

    ∂D̂/∂c_j ≈ (D̂(c + h_j ê_j, ·) - D̂(c - h_j ê_j, ·)) / (2h_j)
    ∂²D̂/(∂c_i ∂c_j) ≈ 逐步对 ∂D̂/∂c_i 再做 c_j 方向差分

这些导数供 ``pipeline_utils.construct_pure_library`` 构造 Euler 算子使用。
"""
from __future__ import annotations

import numpy as np


def validate_inputs(spectra: np.ndarray, c: np.ndarray, wavelengths: np.ndarray) -> None:
    """校验 run_discovery 的三个必要输入的形状与一致性。"""
    if spectra.ndim != 2:
        raise ValueError("spectra 必须为二维数组: (N, M)")
    if c.ndim != 2:
        raise ValueError("c 必须为二维数组: (N, d)，其中 d 为控制维度数")
    if wavelengths.ndim != 1:
        raise ValueError("wavelengths 必须为一维数组")
    if spectra.shape[0] != c.shape[0]:
        raise ValueError("spectra 与 c 的样本数必须一致")
    if spectra.shape[1] != wavelengths.shape[0]:
        raise ValueError("spectra 的光谱长度必须等于 wavelengths 长度")


def compute_fourier_tensor(spectra: np.ndarray, wavelengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """将光谱矩阵变换到频域，返回复频域张量 D̂ 和归一化频率轴 ω̄。

    变换步骤
    --------
    1. 去直流：S̃(c, λ) = S(c, λ) - mean_λ S(c, λ)
    2. rfft：D̂(c, ω) = FFT(S̃(c, ·))，保留非负频率半频带

    返回值
    ------
    d_hat : ndarray, shape (N, P)
        复频域张量，P = M//2 + 1（正频率数），M 为波长点数。
    omega : ndarray, shape (P,)
        归一化频率轴 ω̄ ∈ [0, 1]，与波长单位无关。
    """
    # 去直流
    spectra_detrend = spectra - np.mean(spectra, axis=1, keepdims=True)
    # 半频带 FFT：只保留非负频率成分
    d_hat = np.fft.rfft(spectra_detrend, axis=1)
    n_freq = d_hat.shape[1]
    # 归一化频率轴，严格覆盖 [0, 1]
    omega = np.linspace(0.0, 1.0, n_freq)
    return d_hat, omega


def _detect_cartesian_grid(c: np.ndarray, tol: float = 1e-5) -> tuple[bool, list[np.ndarray], tuple[int, ...], np.ndarray]:
    """检测散点控制变量是否构成完整的等间距笛卡尔网格。

    笛卡尔网格条件：沿每个控制维度的唯一取值个数之积 = 总样本数，
    且所有点均精确匹配网格交叉点（误差 < tol）。

    Returns
    -------
    is_grid : bool
    unique_vals : list of 1-D arrays, one per control dimension
    grid_shape : tuple[int, ...]
    sort_idx : ndarray
        将散点 c 重排为 C-order 网格顺序的索引。
    """
    n_samples, n_dims = c.shape
    unique_vals = []
    
    for dim in range(n_dims):
        vals = np.unique(c[:, dim])
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
    
    # 我们需要找出原始控制变量到排序网格点的映射
    # 使用 lexsort：最后一列先排序
    sort_idx = np.lexsort([c[:, i] for i in range(n_dims-1, -1, -1)])
    sorted_c = c[sort_idx]
    
    if np.max(np.abs(sorted_c - mesh_points)) > tol:
        return False, [], (), np.array([])
        
    return True, unique_vals, grid_shape, sort_idx


def estimate_control_derivatives_scattered(
    field: np.ndarray,
    c: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """在等间距笛卡尔网格上用中心差分估计偏导 ∂D̂/∂c_j。

    要求 ``c`` 必须构成完整的等间距笛卡尔网格。

    Returns
    -------
    grads : ndarray, shape (N, n_controls, P)
        grads[n, j, k] = (∂D̂/∂c_j)(c_n, ω_k)
    """
    y = np.asarray(field)
    c = np.asarray(c, dtype=float)
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
    raise ValueError("控制变量 c 必须构成等间距笛卡尔网格，以使用中心差分估计导数")


def estimate_control_second_derivatives_scattered(
    first_derivatives: np.ndarray,
    c: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """在等间距笛卡尔网格上用中心差分估计二阶混合偏导 ∂²D̂/(∂c_i ∂c_j)。

    输入 ``first_derivatives`` 形状 (N, n_controls, P)，
    输出 shape (N, n_controls, n_controls, P)，其中索引 [n, i, j, k]
    对应 (∂²D̂/∂c_i∂c_j)(c_n, ω_k)。
    """
    d1 = np.asarray(first_derivatives)
    n_samples, n_controls, n_freq = d1.shape
    d2 = np.zeros((n_samples, n_controls, n_controls, n_freq), dtype=complex)

    is_grid, uniq_vals, grid_shape, sort_idx = _detect_cartesian_grid(np.asarray(c))
    
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
    raise ValueError("控制变量 c 必须构成等间距笛卡尔网格，以使用中心差分估计导数")


def build_control_derivative_bundle(
    d_hat: np.ndarray,
    c: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """统一构建一阶与二阶控制偏导数束。

    对零浓度锚点（c = 0）：若整体不构成笛卡尔网格，则将其剔除后
    对剩余网格差分，锚点处导数补零（物理上 c=0 处无 Euler 算子意义）。

    Returns
    -------
    dD_dc : ndarray, shape (N, n_controls, P)
        一阶偏导 ∂D̂/∂c_j
    d2D_dc2 : ndarray, shape (N, n_controls, n_controls, P)
        二阶偏导 ∂²D̂/(∂c_i∂c_j)
    """
    c = np.asarray(c, dtype=float)
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

    dD_dc_keep = estimate_control_derivatives_scattered(
        field=y_keep,
        c=c_keep,
        eps=eps,
    )
    d2D_dc2_keep = estimate_control_second_derivatives_scattered(
        first_derivatives=dD_dc_keep,
        c=c_keep,
        eps=eps,
    )

    # 回填到完整数组，锚点导数为 0
    dD_dc = np.zeros((n_samples, n_controls, n_freq), dtype=complex)
    d2D_dc2 = np.zeros((n_samples, n_controls, n_controls, n_freq), dtype=complex)
    dD_dc[keep_mask] = dD_dc_keep
    d2D_dc2[keep_mask] = d2D_dc2_keep
    return dD_dc, d2D_dc2
