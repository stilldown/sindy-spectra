from __future__ import annotations

import numpy as np


def build_observable_library(
    d_hat: np.ndarray,
    d_d_c: np.ndarray,
    d2_d_c: np.ndarray,
    omega: np.ndarray,
    factors: np.ndarray,
    *,
    # legacy flags kept for compatibility; currently ignored
    include_c_times_s: bool = True,
    include_iomega_control_terms: bool = True,
    include_control_coupling: bool = True,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """构建严格的 12 类物理受约束观测算子库（Physics-Constrained Operator Library）。
    
    采用张量架构与 Euler 算子构建维度不变特征空间：
    四大驱动源（Base, 1阶, 2阶自身, 2阶交叉） 分别与三大频域变换 (1, iω, (iω)²) 组合。
    
    驱动源包含：
    1. Base: D̂
    2. 1阶浓度驱动: c_j · ∂D̂/∂c_j                   (纯浓度演化)
    3. 2阶浓度驱动: c_j² · ∂²D̂/∂c_j²                (非线性演化形态)
    4. 交叉耦合驱动: c_i c_j · ∂²D̂/(∂c_i ∂c_j)      (组分间交互，当 n_controls >= 2)
    
    返回：
    - theta: 经过列级 L2 范数归一化的算子矩阵, 形状 (N_samples * N_freq, N_operators)
    - names: 算子名称列表 (严格匹配各分类)
    - scales: 每一列对应的归一化缩放因子，用于解码时还原量纲
    - tags: 每一列对应的物理分类标签
    """
    n_samples, n_freq = d_hat.shape
    n_controls = d_d_c.shape[1]

    omega_arr = np.asarray(omega)
    if omega_arr.ndim == 1:
        omega_cols = omega_arr.reshape(-1, 1)
    else:
        raise ValueError("omega 必须为 1D 数组 (n_freq,)")
        
    iw1 = 1j * omega_cols.reshape(1, -1)   # (1, n_freq)
    iw2 = -(omega_cols ** 2).reshape(1, -1) # (1, n_freq), 即 (iω)²

    base_terms = []
    base_names = []
    block_tags = []  # 记录当前这个算子属于哪个物理类别，方便后续流水线解析

    # 1. Base 项 (D̂)
    base_terms.append(d_hat)
    base_names.append("D̂")
    block_tags.append("D")

    # 2. 1阶浓度驱动 (c_j * ∂D̂/∂c_j)
    for j in range(n_controls):
        cj = factors[:, j].reshape(-1, 1)
        base_terms.append(cj * d_d_c[:, j, :])
        base_names.append(f"c_{j+1}·∂D̂/∂c_{j+1}")
        block_tags.append("c_dDdc")

    # 3. 2阶浓度驱动 (c_j² * ∂²D̂/∂c_j²)
    for j in range(n_controls):
        cj2 = (factors[:, j]**2).reshape(-1, 1)
        base_terms.append(cj2 * d2_d_c[:, j, j, :])
        base_names.append(f"c_{j+1}²·∂²D̂/∂c_{j+1}²")
        block_tags.append("c2_d2Ddc2")

    # 4. 交叉耦合驱动 (c_i * c_j * ∂²D̂/(∂c_i ∂c_j)), 不重复添加 i,j 组合
    for i in range(n_controls):
        for j in range(i + 1, n_controls):
            cicj = (factors[:, i] * factors[:, j]).reshape(-1, 1)
            # 在 preprocess.py 中 d2_d_c 是对称的，直接取混合偏导
            base_terms.append(cicj * d2_d_c[:, i, j, :])
            base_names.append(f"c_{i+1}c_{j+1}·∂²D̂/(∂c_{i+1}∂c_{j+1})")
            block_tags.append("cc_d2Ddcidcj")

    # 进行 3 种频域变换扩展: *1, *iω, *(iω)²
    terms = []
    names = []
    
    # 我们按照基准组、iω组、(iω)²组的顺序严格排列，这能产生非常漂亮的 12 类矩阵结构
    # 基准组 [* 1]
    for term, name in zip(base_terms, base_names):
        terms.append(term)
        names.append(name)
        
    # 平移组 [* iω]
    for term, name, tag in zip(base_terms, base_names, block_tags):
        terms.append(iw1 * term)
        names.append(f"iω·{name}")
        
    # 形态/展宽组 [* (iω)²]
    for term, name, tag in zip(base_terms, base_names, block_tags):
        terms.append(iw2 * term)
        names.append(f"(iω)²·{name}")
        
    # 我们对应地扩展 tags (加前缀表示是 iω 还是 (iω)²)
    final_tags = []
    # 基础组
    final_tags.extend(block_tags)
    # iω 组
    final_tags.extend([f"iw_{t}" for t in block_tags])
    # (iω)² 组
    final_tags.extend([f"iw2_{t}" for t in block_tags])

    # 将张量拉直为矩阵: (n_samples * n_freq, n_operators)
    theta_raw = np.stack([t.reshape(-1) for t in terms], axis=1)
    
    # 执行特征缩放 (Column Scaling)
    scales = np.linalg.norm(theta_raw, axis=0) + 1e-12
    theta = theta_raw / scales

    return theta, names, scales, final_tags

