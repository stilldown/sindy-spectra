import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from opera.discovery import run_discovery, DiscoveryConfig


def main():
    filepath = "2_corrected_final.csv"
    df = pd.read_csv(filepath)

    wavelength_cols = [col for col in df.columns if col not in ['Sample', 'c1', 'c2']]
    wavelengths = np.array([float(w) for w in wavelength_cols])

    grouped = df.groupby(['c1', 'c2'])[wavelength_cols].mean().reset_index()
    factors = grouped[['c1', 'c2']].to_numpy(dtype=float)
    spectra = grouped[wavelength_cols].to_numpy(dtype=float)

    print("开始运行 Discovery 流程...")
    cfg = DiscoveryConfig(max_components=2, sparsity_threshold=1e-2)
    out = run_discovery(spectra, factors, wavelengths, cfg)

    print("\n=== 提取的物理控制方程（近似）===")
    for block in out.latex_blocks:
        print(block)

    s_time = out.S_real.T
    n_freq = s_time.shape[1]

    plt.figure(figsize=(10, 6))
    w_idx = np.arange(n_freq)
    for k in range(s_time.shape[0]):
        plt.plot(w_idx, s_time[k, :], label=f'Component {k+1}')
    plt.title('Extracted Pure Spectra (Discovery)')
    plt.xlabel('Wavelength Index')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True)
    plt.savefig('discovery_pure_spectra.png')
    print("纯光谱图像已保存至 discovery_pure_spectra.png")


if __name__ == "__main__":
    main()
