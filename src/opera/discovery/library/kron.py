from __future__ import annotations

import numpy as np


def build_psi(phi: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Kronecker 风格特征展开，输出 (n_samples*n_freq, n_phi*n_w)。"""
    n_samples, n_phi = phi.shape
    n_freq, n_w = w.shape
    psi = np.einsum("sp,fw->sfpw", phi, w, optimize=True)
    return psi.reshape(n_samples * n_freq, n_phi * n_w)
