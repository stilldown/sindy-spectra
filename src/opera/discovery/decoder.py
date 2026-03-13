from __future__ import annotations

import re
from typing import Callable
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, lsqr

from .preprocess import _detect_cartesian_grid

def build_1d_first_derivative_matrix(n: int, coords: np.ndarray) -> sp.csr_matrix:
    """构建一维非均匀或均匀网格的中心差分矩阵（二阶截断误差）"""
    if n <= 1:
        return sp.csr_matrix((1, 1))
        
    h = np.diff(coords)
    
    data = []
    row = []
    col = []
    
    # 左边界（前向差分）
    data.extend([-1.0/h[0], 1.0/h[0]])
    row.extend([0, 0])
    col.extend([0, 1])
    
    # 内部点（中心差分）
    for i in range(1, n - 1):
        h1 = h[i-1]
        h2 = h[i]
        # f'(x) 近似 (-h2 * f(x-1) + (h2 - h1) * f(x) + h1 * f(x+1)) / (h1 * h2 * (h1 + h2))
        # 上述为非均匀网格的标准中心差分形式：
        denom = h1 * h2 * (h1 + h2)
        c_prev = -h2 * h2 / denom
        c_next = h1 * h1 / denom
        c_mid = (h2 * h2 - h1 * h1) / denom

        data.extend([c_prev, c_mid, c_next])
        row.extend([i, i, i])
        col.extend([i-1, i, i+1])
        
    # 右边界（后向差分）
    data.extend([-1.0/h[-1], 1.0/h[-1]])
    row.extend([n-1, n-1])
    col.extend([n-2, n-1])
    
    return sp.csr_matrix((data, (row, col)), shape=(n, n))

from typing import Optional

def build_grid_derivative_matrices(factors: np.ndarray, keep_mask: Optional[np.ndarray] = None) -> list[sp.csr_matrix]:
    """为各个控制变量生成全局张量拉平后的偏导数稀疏矩阵，支持忽略非网格的锚点。"""
    n_samples, n_controls = factors.shape
    if keep_mask is None:
        keep_mask = np.ones(n_samples, dtype=bool)

    factors_grid = factors[keep_mask]
    
    is_grid, uniq_vals, grid_shape, sort_idx = _detect_cartesian_grid(factors_grid)
    if not is_grid:
        raise ValueError("物理方程全局解码要求非锚点样本必须构成完整的笛卡尔网格。")
    
    n_grid = len(factors_grid)

    # 建立从 sort_idx (网格顺序) 到 original_idx (网格数据的原始顺序) 的映射矩阵
    row_idx = np.arange(n_grid)
    col_idx = sort_idx
    P_grid = sp.csr_matrix((np.ones(n_grid), (row_idx, col_idx)), shape=(n_grid, n_grid))
    P_orig = P_grid.T # P_orig @ x_grid = x_orig  # 反转映射

    # 网格数据内索引到全体全量样本索引的映射矩阵
    # E @ x_grid_orig_order = x_full_size (用0填充锚点)
    # E.T @ x_full_size = x_grid_orig_order (抽取网格点)
    grid_indices = np.where(keep_mask)[0]
    E_mat = sp.csr_matrix(
        (np.ones(n_grid), (grid_indices, np.arange(n_grid))),
        shape=(n_samples, n_grid)
    )

    D_matrices = []

    for j in range(n_controls):
        # 针对第 j 个维度构建 1D 差分矩阵
        n_j = grid_shape[j]
        coords_j = uniq_vals[j]
        D_1d = build_1d_first_derivative_matrix(n_j, coords_j)

        pre_dim = np.prod(grid_shape[:j], dtype=int) if j > 0 else 1
        post_dim = np.prod(grid_shape[j+1:], dtype=int) if j < n_controls - 1 else 1

        I_pre = sp.eye(pre_dim, format='csr')
        I_post = sp.eye(post_dim, format='csr')

        D_grid = sp.kron(I_pre, sp.kron(D_1d, I_post, format='csr'), format='csr')
        
        # 转换回局部顺序：D_orig = P_orig @ D_grid @ P_grid
        D_local = P_orig @ D_grid @ P_grid
        
        # 抬升为全局尺寸矩阵 (n_samples, n_samples)
        D_global = E_mat @ D_local @ E_mat.T
        D_matrices.append(D_global)
        
    return D_matrices

def _parse_term_operator(
    name: str, 
    factors: np.ndarray, 
    D_matrices: list[sp.csr_matrix]
) -> tuple[sp.csr_matrix | None, np.ndarray | None, Callable | None]:
    """解析单个算子字符串，返回对应的 矩阵(L) 或 向量(b) 或非线性求值函数(f) 算子
    如果算子包含导数(U)，返回 D 矩阵；
    如果算子只是常数(1, c_j)，返回常数向量；
    如果是不可线性化的非线性项，则返回关于 U 的函数。
    """
    n_samples = factors.shape[0]
    
    # 纯 D̂
    if name == "D̂":
        return None, np.ones(n_samples), None
        
    # c_j·D̂
    m = re.match(r"^c_(\d+)·D̂$", name)
    if m:
        j = int(m.group(1)) - 1
        return None, factors[:, j], None
        
    # ∂D̂/∂c_j
    m = re.match(r"^∂D̂/∂c_(\d+)$", name)
    if m:
        j = int(m.group(1)) - 1
        return D_matrices[j], None, None
        
    # c_i·∂D̂/∂c_j
    m = re.match(r"^c_(\d+)·∂D̂/∂c_(\d+)$", name)
    if m:
        i = int(m.group(1)) - 1
        j = int(m.group(2)) - 1
        Diag_C = sp.diags(factors[:, i], offsets=0, format="csr")
        return Diag_C @ D_matrices[j], None, None
        
    # ∂²D̂/∂c_i²
    m = re.match(r"^∂²D̂/∂c_(\d+)²$", name)
    if m:
        i = int(m.group(1)) - 1
        return D_matrices[i] @ D_matrices[i], None, None

    # ∂²D̂/(∂c_i∂c_j)
    m = re.match(r"^∂²D̂/\(∂c_(\d+)∂c_(\d+)\)$", name)
    if m:
        i = int(m.group(1)) - 1
        j = int(m.group(2)) - 1
        return D_matrices[i] @ D_matrices[j], None, None

    # c_j²·∂²D̂/∂c_j²
    m = re.match(r"^c_(\d+)²·∂²D̂/∂c_\1²$", name)
    if m:
        j = int(m.group(1)) - 1
        Diag_C2 = sp.diags(factors[:, j]**2, offsets=0, format="csr")
        return Diag_C2 @ (D_matrices[j] @ D_matrices[j]), None, None

    # c_i c_j·∂²D̂/(∂c_i ∂c_j)
    m = re.match(r"^c_(\d+)c_(\d+)·∂²D̂/\(∂c_\1∂c_\2\)$", name)
    if m:
        i = int(m.group(1)) - 1
        j = int(m.group(2)) - 1
        Diag_CC = sp.diags(factors[:, i] * factors[:, j], offsets=0, format="csr")
        return Diag_CC @ (D_matrices[i] @ D_matrices[j]), None, None

    # D̂·∂D̂/∂c_j (非线性耦合) - 将原始空间方程除以D后，等价于 f * dU
    m = re.match(r"^D̂·∂D̂/∂c_(\d+)$", name)
    if m:
        j = int(m.group(1)) - 1
        return None, None, lambda U: np.exp(U) * (D_matrices[j] @ U)

    # D̂² - 除以D后等价于 f = exp(U)
    m = re.match(r"^D̂²$", name)
    if m:
        return None, None, lambda U: np.exp(U)

    raise ValueError(f"无法解析的方程基本项: {name}")


def decode_physical_manifolds(
    xi: np.ndarray, 
    names: list[str], 
    factors: np.ndarray,
    anchor_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    将稀疏复数系数矩阵 Xi 解析为非负流形 f(c) 和 相位漂移 g(c)
    
    返回:
      f_response: (n_samples, K)
      g_shift: (n_samples, K)
    """
    n_samples, n_controls = factors.shape
    K = xi.shape[1]
    
    f_response = np.zeros((n_samples, K))
    g_shift = np.zeros((n_samples, K))
    
    from .preprocess import _detect_cartesian_grid
    
    # 判断完整的 factors 是否已经是网格
    is_fully_grid, _, _, _ = _detect_cartesian_grid(factors)
    
    keep_mask = np.ones(n_samples, dtype=bool)
    if not is_fully_grid and anchor_idx is not None and len(anchor_idx) > 0:
        keep_mask[anchor_idx] = False

    # 构建微分算子矩阵
    D_matrices = build_grid_derivative_matrices(factors, keep_mask=keep_mask)

    # 拆分项
    base_indices = []
    p1_indices = []
    p2_indices = []
    
    # 因为存在多种前缀，我们需要归一化取出真正的"基础算子名"
    # 以便把对同一个基础算子的各阶频率组合起来
    base_name_to_idx = {}
    for i, name in enumerate(names):
        if name.startswith("iω·"):
            pass
        elif name.startswith("(iω)²·"):
            pass
        else:
            base_indices.append(i)
            base_name_to_idx[name] = i

    # 提前生成各个约束矩阵的 anchor 相关缓存
    weight = 1e6
    anchor_rows = []
    anchor_cols = []
    anchor_vals = []
    b_anchor = np.zeros(len(anchor_idx) if anchor_idx is not None else 0)
    if anchor_idx is not None:
        for r_idx, a_idx in enumerate(anchor_idx):
            anchor_rows.append(r_idx)
            anchor_cols.append(a_idx)
            anchor_vals.append(weight)
    
    # 构建复用的惩罚块
    Anchor_Penalty = sp.csr_matrix((anchor_vals, (anchor_rows, anchor_cols)), shape=(len(b_anchor), n_samples))

    for k in range(K):
        xi_k = xi[:, k]

        # 为分量 k 组装算子矩阵
        M_U_deriv = sp.csr_matrix((n_samples, n_samples), dtype=complex)
        b_U = np.zeros(n_samples, dtype=complex)
        
        nonlin_funcs = []

        # 将无omega和(iω)²的实数项合并映射到 U
        for idx in base_indices:
            name = names[idx]
            
            coef = xi_k[idx]
            
            # 若包含二阶频率项
            name_w2 = f"(iω)²·{name}"
            if name_w2 in names:
                coef += xi_k[names.index(name_w2)]

            if np.abs(coef) < 1e-12:
                continue

            L_mat, L_vec, L_func = _parse_term_operator(name, factors, D_matrices)
            if L_mat is not None:
                M_U_deriv += coef * L_mat
            if L_vec is not None:
                b_U -= coef * L_vec
            if L_func is not None:
                nonlin_funcs.append((coef, L_func))

        # 组装完整的 M_U = M_U_deriv (注意，因为我们要叠成实数偏微分方程，分别把实虚部压一起)
        # 解 U: min || M_U_deriv U - b_U ||

        if M_U_deriv.nnz == 0:
            # 该组分没有任何导数项，说明 f 是常数
            # f = exp(U) = 1
            U_sol = np.zeros(n_samples)
        else:
            # 扩展为实数最小二乘
            M_U_real = M_U_deriv.real if hasattr(M_U_deriv, 'real') else np.real(M_U_deriv)
            M_U_imag = M_U_deriv.imag if hasattr(M_U_deriv, 'imag') else np.imag(M_U_deriv)
            M_U_stack = sp.vstack([M_U_real, M_U_imag], format="csr")
            b_U_stack_base = np.concatenate([np.real(b_U), np.imag(b_U)])

            M_U_stack = sp.vstack([M_U_stack, Anchor_Penalty])
            
            # 采用 Picard 迭代法求解包含非线性项的系统
            U_sol = np.zeros(n_samples)
            
            for _ in range(5): # 迭代 5 次以收敛非线性项
                b_U_eval = np.copy(b_U_stack_base)
                
                # 计算当前U下的非线性部分残差，并移到右侧 (b - L_nonlin(U))
                if len(nonlin_funcs) > 0:
                    b_nonlin = np.zeros(n_samples, dtype=complex)
                    for c_nl, f_nl in nonlin_funcs:
                        # evaluate non-linear function
                        # f_nl returns real array like exp(U_sol)
                        nl_val = f_nl(U_sol) 
                        b_nonlin -= c_nl * nl_val
                    b_U_eval += np.concatenate([np.real(b_nonlin), np.imag(b_nonlin)])

                # 加入 Dirichlet 锚点条件
                b_U_stack_eval = np.concatenate([b_U_eval, b_anchor])

                # 求解稀疏最小二乘
                res = lsqr(M_U_stack, b_U_stack_eval, atol=1e-8, btol=1e-8)
                U_new = res[0]
                
                if np.linalg.norm(U_new - U_sol) < 1e-6:
                    U_sol = U_new
                    break
                U_sol = U_new

        f_response[:, k] = np.exp(U_sol)

        # ------------------ 解 V (相位 g_shift) ------------------
        rhs_V_complex = np.zeros(n_samples, dtype=complex)
        for b_idx in base_indices:
            name = names[b_idx]
            
            p_name = f"iω·{name}"
            if p_name not in names:
                continue

            p_idx = names.index(p_name)
            phase_coef = xi_k[p_idx]
            if np.abs(phase_coef) < 1e-12:
                continue

            # 计算 L^U(U)
            L_mat, L_vec, L_func = _parse_term_operator(name, factors, D_matrices)
            if L_mat is not None:
                val = L_mat @ U_sol
            elif L_vec is not None:
                val = L_vec
            elif L_func is not None:
                val = L_func(U_sol)
            else:
                val = 0.0

            rhs_V_complex += phase_coef * val

        if M_U_deriv.nnz == 0:
            V_sol = np.zeros(n_samples)
        else:
            M_V_real = M_U_deriv.real if hasattr(M_U_deriv, 'real') else np.real(M_U_deriv)
            M_V_imag = M_U_deriv.imag if hasattr(M_U_deriv, 'imag') else np.imag(M_U_deriv)
            M_V_stack = sp.vstack([M_V_real, M_V_imag], format="csr")
            b_V_stack = np.concatenate([np.real(rhs_V_complex), np.imag(rhs_V_complex)])

            M_V_stack = sp.vstack([M_V_stack, Anchor_Penalty])
            b_V_stack = np.concatenate([b_V_stack, b_anchor])
            
            res_V = lsqr(M_V_stack, b_V_stack, atol=1e-8, btol=1e-8)
            V_sol = res_V[0]
            
        g_shift[:, k] = V_sol
        
    return f_response, g_shift
