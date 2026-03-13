from __future__ import annotations

import math
import numpy as np


def build_lie_basis(omega: np.ndarray, orders=(0, 1, 2)) -> tuple[np.ndarray, list[str]]:
    cols = []
    names = []
    for o in orders:
        scale = 1.0 / math.factorial(o)
        if o == 0:
            cols.append(np.ones_like(omega, dtype=complex) * scale)
            names.append("W0=1")
        elif o == 1:
            cols.append((-1j * omega) * scale)
            names.append("W1=-iω")
        elif o == 2:
            cols.append(-(omega ** 2) * scale)
            names.append("W2=-ω²")
        else:
            cols.append(((-1j * omega) ** o) * scale)
            names.append(f"W{o}=(-iω)^{o}")
    return np.stack(cols, axis=1), names
