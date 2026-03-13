from __future__ import annotations

import numpy as np
from scipy.linalg import qr

def select_component_count(
    matrix: np.ndarray,
    mode: str,
    k_value: int,
    k_max: int,
    energy_threshold: float,
) -> int:
    """根据张量奇异值能量选择组分数。"""
    m = np.asarray(matrix)
    # 对于极大的 M x P 矩阵，只计算 P x P 协方差的特征值
    cov = m.conj().T @ m
    s2, _ = np.linalg.eigh(cov)
    s2 = np.maximum(s2, 0.0)
    s = np.sqrt(s2)[::-1]  # 降序排列

    if s.size == 0:
        return 1

    if mode == "fixed":
        return max(1, min(int(k_value), s.size))

    e = np.cumsum(s ** 2)
    e = e / (e[-1] + 1e-12)
    k_auto = int(np.searchsorted(e, float(energy_threshold), side="left") + 1)
    k_auto = max(1, min(k_auto, s.size))

    if mode == "auto":
        return k_auto
    if mode == "capped-auto":
        return max(1, min(k_auto, int(k_max), s.size))

    return max(1, min(int(k_value), s.size))


def find_joint_nullspace(theta: np.ndarray, k_eff: int) -> tuple[np.ndarray, dict]:
    """提取观测张量 Theta 的物理零空间。
    
    返回:
      V_null: (P, K) 形状的矩阵，包含使得 Theta*v 能量最小的 K 个基向。
    """
    P_dim = theta.shape[1]
    # 利用协方差矩阵求 SVD 的极小右奇异向量
    cov = theta.conj().T @ theta
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # argsort 默认是升序，所以前 k_eff 个特征值及对应的特征向量就是零空间
    k_use = max(1, min(k_eff, P_dim))
    V_null = eigenvectors[:, :k_use]
    
    diagnostics = {
        "nullspace_energy": float(np.sum(eigenvalues[:k_use])),
        "total_energy": float(np.sum(eigenvalues)),
        "sigma_gap_min": float(eigenvalues[k_use] / (eigenvalues[k_use - 1] + 1e-12)) if P_dim > k_use else 0.0
    }
    
    return V_null, diagnostics


def extract_sparse_physical_coefficients(
    V_null: np.ndarray,
    sparsity_threshold: float,
) -> np.ndarray:
    """使用列主元 QR 分解 (DEIM 启发) 以及 SINDy-like 截断，对零空间进行稀疏旋转。
    
    使得解耦后的组分系数 Xi 具有物理唯一标识且高度稀疏。
    """
    P_dim, K = V_null.shape
    if K == 0:
        return np.zeros((P_dim, 0), dtype=complex)
        
    # 1. 寻找最能独立标识物理成分的 K 个“锚点”测度
    # 通过对 V_null 转置进行 QR 列主元分解获得
    _, _, p_idx = qr(V_null.T, pivoting=True)
    pivot_indices = p_idx[:K]
    
    # 2. 旋转 V_null，使得这 K 个锚点成为单位阵 I_K
    # 从而强制每个分离出的分量都严格建立在不同的基础演化方向上
    anchor_matrix = V_null[pivot_indices, :]
    try:
        Q = np.linalg.inv(anchor_matrix)
    except np.linalg.LinAlgError:
        Q = np.linalg.pinv(anchor_matrix)
        
    Xi = V_null @ Q  # 形状 (P_dim, K)
    
    # 3. L1/L0 刚性稀疏截断过滤
    for k in range(K):
        abs_xi = np.abs(Xi[:, k])
        # 使用当前最大系数（通常就是锚点位置，值为1）作为缩放参考
        max_val = np.max(abs_xi)
        if max_val > 0:
            mask = abs_xi > (sparsity_threshold * max_val)
            # 严格保留锚点以防数值截断
            mask[pivot_indices[k]] = True
            Xi[~mask, k] = 0.0
            
    return Xi


def calibrate_pure_spectra_once(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    f_response: np.ndarray,
    g_shift: np.ndarray,
    ridge: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, float]:
    """单次线性校准纯光谱（非迭代）。

    给定 f,g 后，对每个频点独立解复线性系统，得到 p_k(ω)。
    """
    d = np.asarray(spectra, dtype=float)
    wl = np.asarray(wavelengths, dtype=float)
    n_samples, n_wl = d.shape
    k_eff = f_response.shape[1]

    d_wl = float(np.mean(np.diff(wl))) if wl.size > 1 else 1.0
    if wl.size > 1:
        assert np.allclose(np.diff(wl), d_wl, atol=1e-6), "Wavelength dimensions must be evenly spaced for direct FFT."

    omega = 2.0 * np.pi * np.fft.fftfreq(n_wl, d=d_wl)
    d_hat = np.fft.fft(d, axis=1)

    p_complex = np.zeros((n_wl, k_eff), dtype=complex)
    recon_hat = np.zeros_like(d_hat, dtype=complex)

    half_n = n_wl // 2 + 1
    for n in range(half_n):
        a = np.real(f_response) * np.exp(1j * omega[n] * np.real(g_shift))
        y = d_hat[:, n]
        ah = a.conj().T
        lhs = ah @ a + float(ridge) * float(n_samples) * np.eye(k_eff)
        rhs = ah @ y
        
        # 强制 DC 和 Nyquist 频率的重建分量为实数
        if n == 0 or (n_wl % 2 == 0 and n == n_wl // 2):
            lhs = np.real(lhs)
            rhs = np.real(rhs)

        try:
            p = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            # 数值退化时回退到最小二乘/伪逆，保证流程稳定
            p, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
            
        p_complex[n] = p
        
        if n == 0 or (n_wl % 2 == 0 and n == n_wl // 2):
            recon_hat[:, n] = np.real(a @ p)
        else:
            recon_hat[:, n] = a @ p

    # 强制负频率的复共轭对称性
    for n in range(half_n, n_wl):
        n_conj = n_wl - n
        p_complex[n] = np.conj(p_complex[n_conj])
        recon_hat[:, n] = np.conj(recon_hat[:, n_conj])

    recon_ifft = np.fft.ifft(recon_hat, axis=1)
    if np.max(np.abs(recon_ifft.imag)) > 1e-4:
        import warnings
        warnings.warn("Reconstructed spectra before taking real part has significant imaginary components.")
    recon = recon_ifft.real
    
    rel_err = float(np.linalg.norm(recon - d) / (np.linalg.norm(d) + 1e-12))
    return np.real(p_complex), p_complex, rel_err
