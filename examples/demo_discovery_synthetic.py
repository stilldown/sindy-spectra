"""
Discovery 合成数据演示。

此脚本演示：
1. 构造具有已知控制机理的多变量合成光谱；
2. 使用 discovery pipeline 自动识别隐式约束；
3. 输出诊断指标与 LaTeX 方程块。
"""

import sys
import os
import numpy as np

# 添加 src 到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from opera.discovery import run_discovery, DiscoveryConfig


def generate_synthetic_data():
    c1 = np.linspace(0, 5, 20)
    c2 = np.linspace(0, 2, 20)
    x = np.linspace(-10, 10, 128)

    c1_grid, c2_grid = np.meshgrid(c1, c2, indexing='ij')
    width = 2.0

    x_grid = x.reshape(1, 1, -1)
    exponent = -((x_grid - c1_grid[..., np.newaxis]) ** 2) / width
    s_tensor = c2_grid[..., np.newaxis] * np.exp(exponent)

    return s_tensor, [c1, c2], x


def main():
    s_tensor, c_grids, wavelengths = generate_synthetic_data()

    d = s_tensor.reshape(-1, s_tensor.shape[-1])
    c1, c2 = np.meshgrid(c_grids[0], c_grids[1], indexing='ij')
    factors = np.column_stack([c1.reshape(-1), c2.reshape(-1)])

    cfg = DiscoveryConfig(max_components=2, sparsity_threshold=1e-2)
    out = run_discovery(d, factors, wavelengths, cfg)

    print("\n发现的方程项 (系数 > 0.01):")
    for k in range(out.Xi.shape[1]):
        print(f"\nComponent {k+1}:")
        for i, coeff in enumerate(out.Xi[:, k]):
            # 兼容 array 值
            if isinstance(coeff, np.ndarray):
                arr = coeff.flatten()
                val = np.max(np.abs(arr))
                coeff_str = ", ".join(f"{x:.3g}" for x in arr)
            else:
                val = abs(coeff)
                coeff_str = f"{coeff:.3g}"
            if val > 0.01:
                print(f"{out.operator_names[i]}: {coeff_str}")

    print("\n--- Diagnostics ---")
    for key, value in out.diagnostics.items():
        print(f"{key}: {value}")

    print("\n--- LaTeX ---")
    for block in out.latex_blocks:
        print(block)


if __name__ == "__main__":
    main()
