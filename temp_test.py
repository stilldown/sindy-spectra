import sys, os
sys.path.append(r'd:\OneDrive\桌面\OPERA\src')
from opera.discovery import run_discovery, DiscoveryConfig
import numpy as np
from opera.gui.main_window import MainWindow
from examples.demo_discovery_synthetic import generate_synthetic_data

s_tensor, c_grids, wavelengths = generate_synthetic_data()
d = s_tensor.reshape(-1, s_tensor.shape[-1])
c1, c2 = np.meshgrid(c_grids[0], c_grids[1], indexing='ij')
factors = np.column_stack([c1.reshape(-1), c2.reshape(-1)])

cfg = DiscoveryConfig()
out = run_discovery(d, factors, wavelengths, cfg)

from PySide6.QtWidgets import QApplication
app = QApplication([])

mw = MainWindow()
mw.cur_data = {'wavelengths': wavelengths, 'factors': factors, 'spectra': d, 'factor_names': ['c1','c2']}

mw.on_fit_complete(out)
print('---- LOG VIEW ----')
print(mw.log_view.toPlainText())
print('---- MODEL HTML ----')
if hasattr(mw.model_view,'toHtml'):
    print(mw.model_view.toHtml()[:1000])
else:
    print(mw.model_view.toPlainText()[:1000])
