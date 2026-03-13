import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtWidgets import QGridLayout
import numpy as np

class SpectrumPlot(QWidget):
    """
    使用 pyqtgraph 进行高性能光谱绘制。
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.plot_item = pg.PlotWidget(title="光谱预览")
        self.layout.addWidget(self.plot_item)
        
        self.plot_item.showGrid(x=True, y=True)
        self.plot_item.setLabel('left', '强度 (Intensity)')
        self.plot_item.setLabel('bottom', '波长 (Wavelength / nm)')
        
        self.curves = []

    def plot_spectra(self, wavelengths, spectra, max_display=1000, labels=None):
        """
        绘制多条光谱曲线。

        参数:
        - wavelengths: 1D array of x values
        - spectra: 2D array of shape (n_curves, len(wavelengths))
        - max_display: 最多绘制的曲线数；若曲线数量超过则等距抽样
        - labels: 可选的字符串列表，与曲线一一对应，用于添加图例
        """
        self.plot_item.clear()
        self.curves = []
        if labels is not None:
            # 清除老图例以避免叠加
            self.plot_item.addLegend()

        n_samples = spectra.shape[0]
        # 允许在不跳步的情况下绘制多达 1000 条曲线
        step = max(1, n_samples // max_display)

        # 使用更加明显的颜色梯度为曲线着色
        for idx, i in enumerate(range(0, n_samples, step)):
            # 基于样本索引的色相梯度
            hue = int(180 * i / n_samples)  # 蓝 -> 绿 -> 黄 -> 红
            color = pg.hsvColor(hue / 360, 0.8, 0.9)
            if labels is not None and idx < len(labels):
                curve = self.plot_item.plot(wavelengths, spectra[i], pen=pg.mkPen(color, width=1), name=labels[idx])
            else:
                curve = self.plot_item.plot(wavelengths, spectra[i], pen=pg.mkPen(color, width=1))
            self.curves.append(curve)

    def plot_spectra_with_label(self, x, spectra, x_label='X'):
        """
        便捷包装：允许自定义底轴标签，并兼容任意 x 作为横轴。
        """
        # 设置底部标签
        self.plot_item.setLabel('bottom', x_label)
        # 复用 plot_spectra，要求 spectra 形状为 (n_curves, len(x))
        self.plot_spectra(x, spectra)

    def plot_concentration(self, c1, obs, fit=None, c2=None, xlabel='c1'):
        """
        绘制观测浓度散点，可选叠加拟合曲线。
        - c1: 一维横坐标
        - obs: 一维观测值
        - fit: 一维拟合值（长度与 c1 相同）
        - c2: 可选，用于给散点上色的数组
        """
        self.plot_item.clear()
        self.curves = []

        # 绘制观测散点，若提供 c2 则按 c2 着色
        if c2 is None:
            pen = pg.mkPen('w')
            for i in range(len(c1)):
                self.plot_item.plot([c1[i]], [obs[i]], pen=None, symbol='o', symbolBrush='b')
        else:
            # 使用 ScatterPlotItem 支持彩色标记
            spots = [{'pos': (c1[i], obs[i]), 'data': i, 'brush': pg.intColor(int(255 * (c2[i] - np.min(c2)) / (np.ptp(c2)+1e-12)))} for i in range(len(c1))]
            scatter = pg.ScatterPlotItem(size=8, spots=spots)
            self.plot_item.addItem(scatter)

        # 若提供拟合曲线则叠加绘制
        if fit is not None:
            # 按 c1 排序后画线
            idx = np.argsort(c1)
            self.plot_item.plot(c1[idx], fit[idx], pen=pg.mkPen('r', width=2))

        self.plot_item.setLabel('bottom', xlabel)

class ResultPlot(QWidget):
    """
    用于显示重构结果的图形控件。
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.plot_item = pg.PlotWidget(title="性能检查")
        self.layout.addWidget(self.plot_item)

    def plot_surface(self, X, Y, Z):
        # pyqtgraph 3D 是一个独立的模块 (GLViewWidget)
        # 对于简单的 2D 预览：
        self.plot_item.clear()
        # 预览残差或重构的 1D 切片
        pass


class ComponentGridPlot(QWidget):
    """用于按组分绘制独立子图的面板。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(8)
        self.plot_widgets = []

    def clear(self):
        while self.layout.count():
            item = self.layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self.plot_widgets = []

    def plot_component_concentrations(self, factors, concentrations, factor_names):
        """二维控制下，每个组分一个子图散点（x=c1, y=组分浓度, 颜色=c2）。"""
        self.clear()

        c = np.asarray(factors)
        cc = np.asarray(concentrations)
        if c.ndim != 2 or c.shape[1] < 2:
            return

        c1 = c[:, 0]
        c2 = c[:, 1]
        n_comp = cc.shape[1]

        n_cols = 2 if n_comp > 1 else 1
        n_rows = int(np.ceil(n_comp / n_cols))

        c2_min = float(np.min(c2))
        c2_ptp = float(np.ptp(c2)) + 1e-12

        for k in range(n_comp):
            r = k // n_cols
            col = k % n_cols

            pw = pg.PlotWidget(title=f"组分 {k+1}")
            pw.showGrid(x=True, y=True)
            pw.setLabel('left', '强度响应 (f)')
            pw.setLabel('bottom', f"{factor_names[0]} (颜色: {factor_names[1]})")

            spots = []
            for i in range(len(c1)):
                color_idx = int(255 * (c2[i] - c2_min) / c2_ptp)
                spots.append({
                    'pos': (float(c1[i]), float(cc[i, k])),
                    'data': i,
                    'brush': pg.intColor(color_idx, hues=256),
                })

            scatter = pg.ScatterPlotItem(size=7, spots=spots)
            pw.addItem(scatter)

            self.layout.addWidget(pw, r, col)
            self.plot_widgets.append(pw)
