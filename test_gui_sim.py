<<<<<<< HEAD
import numpy as np
from src.opera.gui.main_window import MainWindow

mw = MainWindow()
class Dummy: pass
result = Dummy()
result.S_real = np.tile(np.linspace(0,1,50)[:,None], (1,2))
result.C_real = np.zeros((10,2))

mw.cur_data = {'wavelengths': np.linspace(400,500,50), 'factors': np.zeros((10,2)), 'factor_names':['c1','c2']}

S_real = result.S_real
k = S_real.shape[1]
labels = [f"Comp {i+1}" for i in range(k)]
mw.comp_plot.plot_spectra(mw.cur_data['wavelengths'], S_real.T, labels=labels)
if k > 1:
    sims=[]
    for i in range(k):
        for j in range(i+1,k):
            num = np.dot(S_real[:,i], S_real[:,j])
            denom = (np.linalg.norm(S_real[:,i])*np.linalg.norm(S_real[:,j]) + 1e-12)
            sims.append(num/denom)
    if any(abs(s-1.0)<1e-3 for s in sims):
        mw.log_view.append("<font color='orange'><b>[提示]</b> 纯组分光谱高度相似，可能在图上重叠。</font>")
for idx in range(k):
    if np.std(S_real[:, idx]) < 1e-8:
        mw.log_view.append(f"<font color='orange'><b>[提示]</b> 组分{idx+1}的光谱近似平坦，可能仅包含噪声或分离失败。</font>")

print('logs:')
print(mw.log_view.toPlainText())
=======
import numpy as np
from src.opera.gui.main_window import MainWindow

mw = MainWindow()
class Dummy: pass
result = Dummy()
result.S_real = np.tile(np.linspace(0,1,50)[:,None], (1,2))
result.C_real = np.zeros((10,2))

mw.cur_data = {'wavelengths': np.linspace(400,500,50), 'factors': np.zeros((10,2)), 'factor_names':['c1','c2']}

S_real = result.S_real
k = S_real.shape[1]
labels = [f"Comp {i+1}" for i in range(k)]
mw.comp_plot.plot_spectra(mw.cur_data['wavelengths'], S_real.T, labels=labels)
if k > 1:
    sims=[]
    for i in range(k):
        for j in range(i+1,k):
            num = np.dot(S_real[:,i], S_real[:,j])
            denom = (np.linalg.norm(S_real[:,i])*np.linalg.norm(S_real[:,j]) + 1e-12)
            sims.append(num/denom)
    if any(abs(s-1.0)<1e-3 for s in sims):
        mw.log_view.append("<font color='orange'><b>[提示]</b> 纯组分光谱高度相似，可能在图上重叠。</font>")
for idx in range(k):
    if np.std(S_real[:, idx]) < 1e-8:
        mw.log_view.append(f"<font color='orange'><b>[提示]</b> 组分{idx+1}的光谱近似平坦，可能仅包含噪声或分离失败。</font>")

print('logs:')
print(mw.log_view.toPlainText())
>>>>>>> 2d9cf06 (Initial commit)
