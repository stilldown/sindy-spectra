
import sys; sys.path.insert(0, './src'); import numpy as np, pandas as pd; 
import opera.discovery.pipeline as py_pipe
py_pipe.build_latex_blocks_from_xi = lambda *args, **kwargs: ['MOCK']
from opera.discovery.types import DiscoveryConfig

filepath = '2_corrected_final.csv'
df = pd.read_csv(filepath)
wavelength_cols = [col for col in df.columns if col not in ['Sample', 'c1', 'c2']]
wavelengths = np.array([float(w) for w in wavelength_cols])
c = df.groupby(['c1', 'c2'])[wavelength_cols].mean().index.to_frame().to_numpy(dtype=float)
spectra = df.groupby(['c1', 'c2'])[wavelength_cols].mean().to_numpy(dtype=float)

cfg = DiscoveryConfig(max_components=2, sparsity_threshold=0.01, k_mode='fixed', k_value=2)
out = py_pipe.run_discovery(spectra, c, wavelengths, cfg)

with open('test_out.txt', 'w', encoding='utf-8') as f:
    f.write('C_real max: ' + str(np.max(np.abs(out.C_real), axis=0)) + '\n')
    f.write('C_real min: ' + str(np.min(np.abs(out.C_real), axis=0)) + '\n')
    for k in range(out.Xi.shape[1]):
        f.write(f'Comp {k+1} nnz: ' + str(np.sum(np.abs(out.A_matrix[:, k]) > 1e-4)) + '\n')
<<<<<<< HEAD

=======

>>>>>>> 2d9cf06 (Initial commit)
