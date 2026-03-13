from __future__ import annotations

import numpy as np


def nullspace_sparse_vectors(
    theta: np.ndarray,
    max_components: int,
    sparsity_threshold: float,
    nullspace_tol: float,
) -> np.ndarray:
    """从 Theta 的零空间中提取稀疏向量，返回 Xi: (n_features, k_eff)。"""
    u, s, vh = np.linalg.svd(theta, full_matrices=True)
    if s.size == 0:
        raise ValueError("Theta 为空，无法求解")

    scale = s[0] if s[0] > 0 else 1.0
    zero_mask = s <= (nullspace_tol * scale)
    nullity = int(np.sum(zero_mask))
    unobserved = max(0, vh.shape[0] - s.size)
    total_nullity = nullity + unobserved

    # 构造目标列数：固定为请求的 max_components（若可用），
    # 严格零空间不足时，用最小奇异方向补足，保证 Xi 尺寸受控。
    target_k = max(1, min(int(max_components), vh.shape[0]))

    if total_nullity == 0:
        basis = vh[-target_k:].T
    else:
        null_basis = vh[-total_nullity:].T
        if total_nullity >= target_k:
            basis = null_basis[:, :target_k]
        else:
            # 追加最小奇异值方向（排除已在 null_basis 中的尾部）
            extra_needed = target_k - total_nullity
            extra = vh[-(total_nullity + extra_needed) : -total_nullity].T if extra_needed > 0 else np.zeros((vh.shape[1], 0))
            basis = np.hstack([null_basis, extra])

    k_eff = basis.shape[1]
    xi = np.array(basis, dtype=complex, copy=True)

    for k in range(k_eff):
        idx = np.argmax(np.abs(xi[:, k]))
        y = -theta[:, idx]
        X = np.delete(theta, idx, axis=1)

        # 初始最小二乘拟合
        w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        # STRidge 迭代
        for _ in range(5):
            small = np.abs(w) < sparsity_threshold
            if not np.any(small):
                break
            w[small] = 0
            big = ~small
            if np.sum(big) == 0:
                break
            # 在保留的支持集上重拟合
            w_new = np.zeros_like(w)
            w_new[big], _, _, _ = np.linalg.lstsq(X[:, big], y, rcond=None)
            w = w_new

        # 重建完整向量
        x_sparse = np.zeros(theta.shape[1], dtype=complex)
        X_cols = [c for c in range(theta.shape[1]) if c != idx]
        x_sparse[X_cols] = w
        x_sparse[idx] = 1.0

        mx = np.max(np.abs(x_sparse))
        if mx > 0:
            x_sparse = x_sparse / mx
        xi[:, k] = x_sparse

    return xi
