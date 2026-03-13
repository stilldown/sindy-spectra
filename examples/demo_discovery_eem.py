import os
import sys
import numpy as np
import pandas as pd

# 添加 src 到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from opera.discovery import run_discovery, DiscoveryConfig


def load_eem_data(filepath):
    df = pd.read_csv(filepath)
    spectral_cols = [col for col in df.columns if col.isdigit()]

    df_all = df.groupby(['c1', 'c2'], as_index=False)[spectral_cols].mean()
    df_grid = df_all[(df_all['c1'] > 0) & (df_all['c2'] > 0)].copy()
    df_grid = df_grid.groupby(['c1', 'c2'], as_index=False)[spectral_cols].mean()

    c1_vals = np.sort(df_grid['c1'].unique())
    c2_vals = np.sort(df_grid['c2'].unique())
    wavelengths = np.array([float(col) for col in spectral_cols])

    s_tensor = np.zeros((len(c1_vals), len(c2_vals), len(wavelengths)))
    for i, c1 in enumerate(c1_vals):
        for j, c2 in enumerate(c2_vals):
            row = df_grid[(df_grid['c1'] == c1) & (df_grid['c2'] == c2)]
            if not row.empty:
                s_tensor[i, j, :] = row[spectral_cols].iloc[0].values.astype(float)

    s_decomp = df_all[spectral_cols].to_numpy(dtype=float)
    sample_pairs = df_all[['c1', 'c2']].to_numpy(dtype=float)
    return s_tensor, [c1_vals, c2_vals], wavelengths, s_decomp, sample_pairs


def main():
    filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '2_corrected_final.csv'))
    print(f"正在加载数据: {filepath}")

    s_tensor, c_grids, x_grid, s_decomp, sample_pairs = load_eem_data(filepath)

    print(f"生成的张量形状: {s_tensor.shape}")
    print(f"直接分解样本数(含0,0): {s_decomp.shape[0]}")

    print("正在运行 Discovery pipeline...")
    cfg = DiscoveryConfig(max_components=4, sparsity_threshold=1e-2)
    out = run_discovery(s_decomp, sample_pairs, x_grid, cfg)

    print("\n--- Discovery 结果 ---")
    print(f"组分数 K: {out.S_real.shape[1]}")
    print(f"S_real 形状: {out.S_real.shape}")
    print(f"响应倍数 形状: {out.f_response_eval.shape}")
    print(f"Xi 形状: {out.Xi.shape}")

    has_zero_zero = np.any((sample_pairs[:, 0] == 0.0) & (sample_pairs[:, 1] == 0.0))
    print(f"(0,0) 是否在原始观测样本中: {has_zero_zero}")

    print("诊断:")
    for key, value in out.diagnostics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
