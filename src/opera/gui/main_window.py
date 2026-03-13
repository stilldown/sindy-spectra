from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QSplitter, QGroupBox,
    QLineEdit, QFormLayout, QTabWidget, QTextEdit, QLabel, QSpinBox, QComboBox, QCheckBox
)
from PySide6.QtCore import Qt
import numpy as np
import os
import sys
import re

# 直接运行本文件时，确保项目 `src` 加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from opera.discovery import DiscoveryConfig
from opera.gui.plots import SpectrumPlot, ComponentGridPlot
from opera.gui.backend import DiscoveryService

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    WEBENGINE_AVAILABLE = True
except Exception:
    QWebEngineView = None
    WEBENGINE_AVAILABLE = False



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OPERA - 物理演化重构算法")
        self.resize(1200, 800)

        self.cur_data = None
        # 后端服务实例，负责文件加载、配置生成和分析线程
        self.backend = DiscoveryService()

        # 主布局
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)

        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)

        # 左侧控件
        self.side_panel = QWidget()
        self.side_layout = QVBoxLayout(self.side_panel)
        self.splitter.addWidget(self.side_panel)

        self.import_gb = QGroupBox("数据导入")
        self.import_layout = QVBoxLayout(self.import_gb)
        self.btn_import = QPushButton("导入光谱 (CSV/Excel)")
        self.btn_import.clicked.connect(self.import_data)
        self.btn_demo = QPushButton("加载演示数据")
        self.btn_demo.clicked.connect(self.load_demo)
        self.import_layout.addWidget(self.btn_import)
        self.import_layout.addWidget(self.btn_demo)
        self.side_layout.addWidget(self.import_gb)

        # 参数
        self.params_gb = QGroupBox("OPERA 参数")
        self.params_layout = QFormLayout(self.params_gb)

        self.comp_count = QSpinBox()
        self.comp_count.setRange(1, 10)
        self.comp_count.setValue(3)
        self.comp_count.valueChanged.connect(self.update_library_from_ui)
        self.params_layout.addRow("固定组分数 K（fixed 生效）:", self.comp_count)

        self.k_mode = QComboBox()
        self.k_mode.addItems(["fixed", "auto", "capped-auto"])
        self.k_mode.setCurrentText("fixed")
        self.params_layout.addRow("K 选择模式:", self.k_mode)

        self.k_max = QSpinBox()
        self.k_max.setRange(1, 20)
        self.k_max.setValue(8)
        self.params_layout.addRow("K 上限 (k_max，仅 auto/capped-auto):", self.k_max)

        self.rank_energy_threshold = QLineEdit("0.99")
        self.params_layout.addRow("秩能量阈值:", self.rank_energy_threshold)

        self.threshold_v = QLineEdit("0.01")
        self.params_layout.addRow("稀疏阈值 (Sparsity):", self.threshold_v)

        self.zero_anchor_tol = QLineEdit("1e-12")
        self.params_layout.addRow("零浓度容差:", self.zero_anchor_tol)

        self.calibration_ridge = QLineEdit("1e-8")
        self.params_layout.addRow("单次校准 Ridge:", self.calibration_ridge)

        self.calibrate_once = QCheckBox("启用单次纯谱校准")
        self.calibrate_once.setChecked(True)
        self.params_layout.addRow("纯谱校准:", self.calibrate_once)

        self.epsilon = QLineEdit("1e-9")
        self.params_layout.addRow("数值容差 ε:", self.epsilon)
        self.side_layout.addWidget(self.params_gb)

        # 算子库选项 - 基础算子
        self.library_gb = QGroupBox("算子库设定")
        self.library_layout = QVBoxLayout(self.library_gb)

        self.lbl_base_op = QLabel("<b>物理纯净算子架构 (Pure Euler)</b><br>基于观测矩阵的对数导数构造纯净算子库：<br>f'/f, g', (ln f)'', g'', 交互耦合项<br>通过联合对角化与物理自洽约束识别")
        self.library_layout.addWidget(self.lbl_base_op)

        self.side_layout.addWidget(self.library_gb)

        self.side_layout.addStretch()
        self.btn_run = QPushButton("运行 Discovery")
        self.btn_run.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white; padding: 10px;")
        self.btn_run.clicked.connect(self.run_identification)
        self.side_layout.addWidget(self.btn_run)

        # 右侧显示
        self.display_panel = QWidget()
        self.display_layout = QVBoxLayout(self.display_panel)
        self.splitter.addWidget(self.display_panel)

        # 选项卡
        self.tabs = QTabWidget()
        self.display_layout.addWidget(self.tabs)

        self.spectrum_plot = SpectrumPlot()
        self.tabs.addTab(self.spectrum_plot, "光谱预览")

        self.comp_plot = SpectrumPlot()
        self.tabs.addTab(self.comp_plot, "纯组分光谱 (S)")

        self.conc_plot = SpectrumPlot()
        self.tabs.addTab(self.conc_plot, "强度响应 (f)")

        self.conc_grid_plot = ComponentGridPlot()
        self.tabs.addTab(self.conc_grid_plot, "组分子图")

        # 日志选项卡
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFontPointSize(12)
        self.tabs.addTab(self.log_view, "日志")

        # 模型视图（单一选项卡，渲染 + 源）
        if WEBENGINE_AVAILABLE:
            self.model_view = QWebEngineView()
        else:
            self.model_view = QTextEdit()
            self.model_view.setReadOnly(True)
            self.model_view.setFontPointSize(12)
        self.tabs.addTab(self.model_view, "Discovery 模型")

        self.splitter.setSizes([300, 900])

    # 数据解析逻辑已迁移到 backend.DiscoveryService

    def import_data(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "导入光谱数据", "", "数据文件 (*.csv *.xlsx *.xls)")
        if not filepath:
            return
        try:
            data = self.backend.load_file(filepath)
            self.cur_data = data
            self.spectrum_plot.plot_spectra(data['wavelengths'], data['spectra'])
            self.log_view.append(f"<b>[成功]</b> 已加载: {os.path.basename(filepath)}")
            self.log_view.append(f"  样本数: {data['spectra'].shape[0]}")
            self.log_view.append(f"  波长点数: {len(data['wavelengths'])}")
            self.log_view.append(f"  识别出的因子: {', '.join(data['factor_names'])}")
        except Exception as e:
            self.log_view.append(f"<font color='red'><b>[错误]</b> 加载失败: {str(e)}</font>")

    def load_demo(self):
        try:
            data = self.backend.load_demo()
            self.cur_data = data
            self.spectrum_plot.plot_spectra(data['wavelengths'], data['spectra'])
            names = data['factor_names']
            self.log_view.append(f"演示数据已加载。识别到控制维: {len(names)} ({', '.join(names)})")
        except Exception as e:
            self.log_view.append(f"<font color='red'><b>[错误]</b> {e}</font>")

    def update_library_from_ui(self):
        pass

    def run_identification(self):
        if self.cur_data is None:
            return
        self.btn_run.setEnabled(False)
        self.log_view.clear()
        self.log_view.append("<b>[日志]</b> 开始 Discovery 识别...")

        # 从界面收集参数并交给后端生成配置
        try:
            th = float(self.threshold_v.text())
        except ValueError:
            th = 0.01
        try:
            rank_thr = float(self.rank_energy_threshold.text())
        except ValueError:
            rank_thr = 0.99
        try:
            anchor_tol = float(self.zero_anchor_tol.text())
        except ValueError:
            anchor_tol = 1e-12
        try:
            ridge = float(self.calibration_ridge.text())
        except ValueError:
            ridge = 1e-8

        cfg = self.backend.make_config(
            k_mode=self.k_mode.currentText(),
            k_value=int(self.comp_count.value()),
            k_max=int(self.k_max.value()),
            rank_energy_threshold=float(rank_thr),
            sparsity_threshold=float(th),
            zero_anchor_tol=float(anchor_tol),
            calibration_ridge=float(ridge),
            calibrate_once=bool(self.calibrate_once.isChecked()),
            epsilon=float(self.epsilon.text() or 1e-9),
        )

        self.thread = self.backend.create_thread(cfg)
        self.thread.finished.connect(self.on_fit_complete)
        self.thread.error.connect(lambda e: self.log_view.append(f"<font color='red'>错误: {e}</font>"))
        self.thread.start()

    def _format_operator_name(self, name: str) -> str:
        """将内部算子标识符映射为易读文本/LaTeX。"""
        # 基础映射
        if name == "ln_f":
            return "\\ln f"
        if name == "g":
            return "g"

        # L1 形式: L1_c1_f, L1_c2_g 等
        m = re.match(r"L1_c(\d+)_([fg])", name)
        if m:
            ci = m.group(1)
            var = "\\ln f" if m.group(2) == "f" else "g"
            if m.group(2) == "f":
                return f"c{ci}\\partial {var}/\\partial c{ci}"
            else:
                return f"c{ci}\\partial {var}/\\partial c{ci}"

        # Xi2 形式: Xi2_c1c2_f 等
        m2 = re.match(r"Xi2_c(\d+)c(\d+)_([fg])", name)
        if m2:
            i, j, fg = m2.groups()
            var = "\\ln f" if fg == "f" else "g"
            if i == j:
                return f"c{i}^2\\partial^2 {var}/\\partial c{i}^2"
            else:
                return f"c{i}c{j}\\partial^2 {var}/\\partial c{i}\\partial c{j}"

        # fallback: 用 name 本身
        return name

    def _render_model_html(self, latex_blocks, diagnostics, quality_flags, component_scores, component_energy_ratio):
        metric_name_map = {
            "nullspace_energy": "零空间能量 (越小越好)",
            "sigma_gap_min": "奇异值间隙 (越大越稳定)",
            "k_selected": "分析截断维度 K",
            "anchor_count": "发现参考零点数",
            "pure_spectra_base_norm": "重建纯谱投影 L2 范数",
            "pure_spectra_cal_norm": "校准后纯谱范数",
            "重建相对误差": "观测场重建相对误差",
            "gamma_shape": "观测张量形状",
        }
        quality_flag_map = {
            "ok": "质量良好",
            "high_nullspace_residual": "零空间残差偏高",
            "symbolic_low_confidence": "符号重建置信度偏低（回退隐式评估）",
        }

        component_scores = np.asarray(component_scores if component_scores is not None else np.array([]))
        component_energy_ratio = np.asarray(component_energy_ratio if component_energy_ratio is not None else np.array([]))

        blocks_html = "\n".join([
            """
            <div class='eq-card'>
                <div class='eq-body'>$$ %s $$</div>
            </div>
            """ % latex
            for latex in latex_blocks
        ])

        diag_rows = "\n".join([
            f"<tr><td>{metric_name_map.get(k, k)}</td><td>{v:.6g}</td></tr>"
            if isinstance(v, (float, int, np.floating, np.integer))
            else f"<tr><td>{metric_name_map.get(k, k)}</td><td>{v}</td></tr>"
            for k, v in diagnostics.items()
        ])

        if component_scores.size > 0 and component_energy_ratio.size > 0:
            score_rows = "\n".join([
                f"<tr><td>组分 {idx + 1}</td><td>{float(sc):.4f}</td><td>{float(en):.4f}</td></tr>"
                for idx, (sc, en) in enumerate(zip(component_scores, component_energy_ratio))
            ])
        else:
            score_rows = "<tr><td colspan='3'>当前逻辑未输出组分评分（联合单次识别模式）</td></tr>"

        if "ok" in quality_flags and len(quality_flags) == 1:
            badge_text = "质量良好"
            badge_cls = "good"
        else:
            badge_text = "存在质量风险: " + ", ".join([quality_flag_map.get(x, x) for x in quality_flags])
            badge_cls = "warn"

        return f"""
<!doctype html>
<html>
    <head>
        <meta charset='utf-8'/>
        <script src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'></script>
        <style>
            :root {{
                --bg: #f7f9fc;
                --card: #ffffff;
                --border: #d9e1ec;
                --title: #1f2d3d;
                --sub: #4a5b6c;
            }}
            body {{
                margin: 0;
                background: var(--bg);
                font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
                color: var(--title);
            }}
            .wrap {{ padding: 16px; }}
            .header {{
                background: linear-gradient(120deg, #2f80ed, #56ccf2);
                color: #fff;
                border-radius: 12px;
                padding: 14px 16px;
                margin-bottom: 14px;
            }}
            .header h2 {{ margin: 0 0 4px 0; font-size: 18px; }}
            .header p {{ margin: 0; opacity: 0.95; font-size: 13px; }}
            .grid {{
                display: grid;
                grid-template-columns: 1fr;
                gap: 10px;
            }}
            .eq-card {{
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 12px 14px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.04);                overflow-x: auto;
                overflow-y: hidden;            }}
            .eq-title {{ font-weight: 700; color: var(--sub); margin-bottom: 8px; }}
            .diag {{
                margin-top: 14px;
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 12px 14px;
            }}
            .diag h3 {{ margin: 0 0 8px 0; font-size: 15px; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
            td {{ padding: 6px 4px; border-bottom: 1px dashed #e2e8f0; }}
            td:first-child {{ color: var(--sub); width: 45%; }}
            .badge {{
                display: inline-block;
                margin-top: 8px;
                border-radius: 999px;
                padding: 4px 10px;
                font-size: 12px;
                font-weight: 700;
            }}
            .badge.good {{ background: #e8f8ef; color: #1c7c47; }}
            .badge.warn {{ background: #fff3e6; color: #b04a00; }}
        </style>
    </head>
    <body>
        <div class='wrap'>
            <div class='header'>
                <h2>Discovery 识别结果</h2>
                <p>从观测张量构建隐式约束，执行 Γ≈A·P^T 分解并恢复可解释公式。</p>
                <div class='badge {badge_cls}'>{badge_text}</div>
            </div>
            <div class='grid'>
                {blocks_html}
            </div>
            <div class='diag'>
                <h3>组件质量评分</h3>
                <table>
                    <tr><td><b>组分</b></td><td><b>评分</b></td><td><b>能量占比</b></td></tr>
                    {score_rows}
                </table>
            </div>
            <div class='diag'>
                <h3>一致性与稳定性指标</h3>
                <table>
                    {diag_rows}
                </table>
            </div>
        </div>
    </body>
</html>
"""

    def on_fit_complete(self, result):
        metric_name_map = {
            "nullspace_energy": "零空间能量 (越小越好)",
            "sigma_gap_min": "奇异值间隙 (越大越稳定)",
            "k_selected": "分析截断维度 K",
            "anchor_count": "发现参考零点数",
            "pure_spectra_base_norm": "重建纯谱投影 L2 范数",
            "pure_spectra_cal_norm": "校准后纯谱范数",
            "重建相对误差": "观测场重建相对误差",
            "gamma_shape": "观测张量形状",
        }
        quality_flag_map = {
            "ok": "质量良好",
            "high_nullspace_residual": "零空间残差偏高",
            "symbolic_low_confidence": "符号重建置信度偏低（回退隐式评估）",
        }

        self.btn_run.setEnabled(True)
        self.tabs.setCurrentIndex(1)
        self.log_view.append("<font color='green'><b>[成功]</b> Discovery 识别完成。</font>")

        S_real = result.S_real
        f_response_eval = result.f_response_eval

        try:
            # 绘制纯组分，同时传递标签以便显示图例
            # 新逻辑下 S_real 形状为 (K, 波长数)
            k = S_real.shape[0]  
            labels = [f"Comp {i+1}" for i in range(k)]
            # 直接传入 (K, 波长数) 为 spectra，plot_spectra 会绘制 K 条光谱曲线
            self.comp_plot.plot_spectra(self.cur_data['wavelengths'], S_real, labels=labels)
            
            # 如果多个组分的光谱高度相似，给出提示（因为它们会覆盖）
            if k > 1:
                sims = []
                for i in range(k):
                    for j in range(i + 1, k):
                        row_i = S_real[i, :]
                        row_j = S_real[j, :]
                        num = np.dot(row_i, row_j)
                        denom = (np.linalg.norm(row_i) * np.linalg.norm(row_j) + 1e-12)
                        sims.append(num / denom)
                if any(abs(s - 1.0) < 1e-3 for s in sims):
                    self.log_view.append(
                        "<font color='orange'><b>[提示]</b> 纯组分光谱高度相似，可能在图上重叠。</font>"
                    )
            # 检查是否有组分几乎平坦
            for idx in range(k):
                if np.std(S_real[idx, :]) < 1e-8:
                    self.log_view.append(
                        f"<font color='orange'><b>[提示]</b> 组分{idx+1}的光谱近似平坦，可能仅包含噪声或分离失败。</font>"
                    )

            factors = np.asarray(self.cur_data['factors'])
            factor_names = self.cur_data.get('factor_names', [f'c{i+1}' for i in range(factors.shape[1])])

            if factors.shape[1] == 1:
                x = factors[:, 0]
                idx = np.argsort(x)
                # 绘制 K 条曲线与控制变量的关系
                # f_response_eval 的形状是 (N, K)，要转置为 (K, N)
                self.conc_plot.plot_spectra_with_label(x[idx], f_response_eval[idx].T, x_label=f'{factor_names[0]}')
                self.conc_plot.plot_item.setLabel('left', '强度响应倍数 (f)')
                self.log_view.append("<b>[响应分布]</b> 1维控制：按控制变量排序绘制各组分曲线。")
            elif factors.shape[1] == 2:
                # 在二维控制情况下：
                # 1) 传统页显示组分1
                # 2) 新增子图页显示每个组分
                c1 = factors[:, 0]
                c2 = factors[:, 1]
                self.conc_plot.plot_concentration(
                    c1=c1,
                    obs=f_response_eval[:, 0],
                    fit=None,
                    c2=c2,
                    xlabel=f'{factor_names[0]} (颜色: {factor_names[1]})',
                )
                self.conc_plot.plot_item.setLabel('left', '组分1强度响应倍数 (f)')

                self.conc_grid_plot.plot_component_concentrations(
                    factors=factors,
                    concentrations=f_response_eval,
                    factor_names=factor_names,
                )
                self.log_view.append("<b>[响应分布]</b> 2维控制：已在“组分子图”标签页中按组分分别展示。")
            else:
                # 高维控制：默认显示第一控制变量切片
                x = factors[:, 0]
                idx = np.argsort(x)
                # 绘制 K 条响应曲线
                self.conc_plot.plot_spectra_with_label(x[idx], f_response_eval[idx].T, x_label=f'{factor_names[0]}')
                self.conc_plot.plot_item.setLabel('left', '强度响应倍数 (f)')
                self.log_view.append(
                    f"<b>[响应分布]</b> {factors.shape[1]}维控制：当前显示沿 {factor_names[0]} 的投影曲线。"
                )

            self.log_view.append("<b>[提示]</b> 纯组分光谱和响应分布已重构至对应选项卡。")
        except Exception as e:
            self.log_view.append(f"<font color='orange'><b>[警告]</b> 重构失败: {str(e)}</font>")

        # 结果概览写入日志
        self.log_view.append(f"<b>[模型]</b> 组分数: {S_real.shape[0]}，特征项数: {len(result.operator_names)}")
        # 处理元数据中的模型并漂亮地打印每个组件的f/g法则
        models = result.metadata.get("models", [])
        if models:
            self.log_view.append("<b>[识别方程]</b>")
            for idx, model in enumerate(models, start=1):
                for var in ("f", "g"):
                    if var not in model:
                        continue
                    coefs, keys = model[var]
                    terms = []
                    for coef, name in zip(coefs, keys):
                        if abs(coef) < 1e-3:
                            continue
                        pretty = self._format_operator_name(name)
                        terms.append(f"{coef:.3g}*{pretty}")
                    if terms:
                        self.log_view.append(f"  Comp {idx} {var}-model: 0 = " + " + ".join(terms))

        if getattr(result, "quality_flags", None):
            self.log_view.append(
                f"<b>[质量状态]</b> {', '.join([quality_flag_map.get(x, x) for x in result.quality_flags])}"
            )
        for k, v in result.diagnostics.items():
            k_zh = metric_name_map.get(k, k)
            if isinstance(v, (float, int, np.floating, np.integer)):
                self.log_view.append(f"  - {k_zh}: {float(v):.6g}")
            else:
                self.log_view.append(f"  - {k_zh}: {v}")


        if WEBENGINE_AVAILABLE and hasattr(self, 'model_view') and isinstance(self.model_view, QWebEngineView):
            comp_scores = getattr(result, "component_scores", None)
            comp_energy = getattr(result, "component_energy_ratio", None)
            
            # 从结构化模型渲染方程
            eq_blocks = []
            models = result.metadata.get("models", [])
            for k, model in enumerate(models, start=1):
                for var in ("f", "g"):
                    if var not in model:
                        continue
                    coefs, keys = model[var]
                    terms = []
                    for coef, name in zip(coefs, keys):
                        if abs(coef) < 1e-3:
                            continue
                        pretty = self._format_operator_name(name)
                        terms.append(f"{coef:.3g} {pretty}")
                    if terms:
                        eq_blocks.append(f"Comp {k} {var}-model: 0 = " + " + ".join(terms))
            
            html = self._render_model_html(
                eq_blocks,
                result.diagnostics,
                getattr(result, "quality_flags", ["unknown"]),
                comp_scores if comp_scores is not None else np.array([]),
                comp_energy if comp_energy is not None else np.array([]),
            )
            self.model_view.setHtml(html)
        else:
            if isinstance(self.model_view, QTextEdit):
                self.model_view.clear()
                self.model_view.append('=== Discovery 公式（渲染不可用，显示源码）===')
                if getattr(result, "metadata", {}) and "equations" in result.metadata:
                     for eq in result.metadata["equations"]:
                        self.model_view.append(eq)
                self.model_view.append('\n=== 诊断指标 ===')
                for k, v in result.diagnostics.items():
                    self.model_view.append(f"- {k}: {v}")


        # 结束 on_fit_complete

if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())