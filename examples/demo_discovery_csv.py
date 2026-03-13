import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# 将 src 加入 sys.path 方便示例直接运行
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from opera.discovery import run_discovery, DiscoveryConfig


def load_data(filepath):
    df = pd.read_csv(filepath)

    wavelength_cols = [col for col in df.columns if col not in ['Sample', 'c1', 'c2']]
    wavelengths = np.array([float(w) for w in wavelength_cols])

    df_grouped = df.groupby(['c1', 'c2'])[wavelength_cols].mean().reset_index()
    c_factors = df_grouped[['c1', 'c2']].to_numpy(dtype=float)
    d = df_grouped[wavelength_cols].to_numpy(dtype=float)

    return d, c_factors, wavelengths


def main():
    print("Loading data...")
    filepath = os.path.join(os.path.dirname(__file__), '..', '2_corrected_final.csv')
    d, c_factors, wavelengths = load_data(filepath)

    print(f"Data shape: {d.shape}")
    print(f"Concentration factors shape: {c_factors.shape}")

    cfg = DiscoveryConfig(max_components=6, sparsity_threshold=1e-2)
    result = run_discovery(d, c_factors, wavelengths, cfg)

    s_real, c_real, xi, operator_names = result.S_real, result.f_response_eval, result.A_matrix, result.operator_names
    k_eff = s_real.shape[1]

    print("\n--- Discovery Results ---")
    print("Sparse Physical Equation Coefficients (Xi):")
    for k in range(k_eff):
        print(f"\nComponent {k + 1}:")
        for p, name in enumerate(operator_names):
            val = xi[p, k]
            if np.any(np.abs(val) > 0):
                if isinstance(val, np.ndarray) and val.size == 1:
                    val = val.item()
                if isinstance(val, (int, float, complex, np.number)):
                    print(f"  {name:20s} : {np.real(val):.4f} + {np.imag(val):.4f}j")
                else:
                    print(f"  {name:20s} : {val}")

    print("\n--- Diagnostics ---")
    for key, value in result.diagnostics.items():
        print(f"{key}: {value}")

    print("\n--- LaTeX blocks ---")
    for block in result.latex_blocks:
        print(block)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Discovery Pipeline")

    for k in range(k_eff):
        axes[0].plot(wavelengths, s_real[:, k], label=f'Component {k + 1}')
    axes[0].set_title('Extracted Pure Spectra (S)')
    axes[0].set_xlabel('Wavelength')
    axes[0].set_ylabel('Intensity')
    axes[0].legend()

    for k in range(k_eff):
        idx = np.argsort(c_factors[:, 0])
        axes[1].scatter(c_factors[idx, 0], c_real[idx, k], label=f'Component {k + 1}', alpha=0.7)
    axes[1].set_title('Extracted Concentration Profiles (C)')
    axes[1].set_xlabel('c1')
    axes[1].set_ylabel('Intensity')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
