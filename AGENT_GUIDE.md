<<<<<<< HEAD
# OPERA 项目智能体速览

## 项目概况
- 领域：光谱分解与符号发现（SINDy-PI 风格），含 GUI（PySide6）。
- 语言/依赖：Python；核心依赖见 `requirements.txt`（numpy、scipy、sympy、pandas、scikit-learn、matplotlib、PySide6 等）。
- 核心目录：
  - `src/opera/discovery/`: 管线、算子库、稀疏回归与谱分解。
  - `src/opera/gui/`: 主窗口与绘图。
  - `examples/`: CSV/EEM/可视化示例。
  - `tests/`: pytest 冒烟与语义测试。

## 关键概念与变量
- `f_response`：样本×组分的强度响应（替代旧的 `C_real`/`f_est`）。
- `g_shift`：样本×组分的光谱位移（目前默认置零，可外部提供）。
- 纯谱：频域系数 `p_freq_complex`（经厄米对称）与波长域 `p_real`。
- 发现结果：`DiscoveryResult`（见 `types.py`）包含 `f_response`, `g_shift`, `p_real`, 诊断信息等。

## 核心流程（`pipeline.run_discovery`）
1. 预处理、算子矩阵生成（符号/算子见 `symbolic.py`, `library/*`）。
2. 求解 Gamma（线性+稀疏 ALS），再做分解得到算子系数 `xi` 与纯谱频域 `p_freq_complex`。
3. 对频谱施加厄米对称 → IFFT 得到 `p_real`。
4. 线性最小二乘求 `f_response`；`g_shift` 当前置零（弃用 phi/psi 路径）。
5. 可选一次性纯谱校准 `calibrate_pure_spectra_once`（ridge 约束，检测谱崩塌）。
6. 诊断与输出封装为 `DiscoveryResult`。

## 光谱平移处理
- 表示方式：在频域用相位 `exp(i * ω * g_shift)` 表示位移，IFFT 后对应波长平移。
- 现状：积分路径已移除，`g_shift` 默认全零，但接口保留；若外部提供 `g_shift`，所有重建/校准逻辑会自动使用。
- 新增：`DiscoveryConfig.estimate_g_shift=True` 可启用频域相位的非线性 least-squares 拟合，估计每个样本/组分的 `g_shift`（可用 `g_shift_max_abs` 限制幅度）。
- 若需更复杂估计，可在 Gamma 分解后，通过频域线性拟合估计 `g_shift`，或复原旧 phi/psi 逻辑。

## 算子库与符号发现
- 基本算子：`1, c, f, f', c f, c f', f f'` 等（SINDy-PI 不会自动生成乘积，需显式加入）。
- 目标：可拟合多项式、幂函数、指数、对数、Logistic/饱和型等；三角/更高阶需增加二阶导或特定算子。
- 多变量：`c_i, c_i f, c_i f'_{c_i}, f f'_{c_i}, c_i c_j` 等；若需扩散/协同，考虑二阶混合偏导。

## GUI 提示
- 入口：`src/opera/gui/main_window.py`。
- 主要功能：加载数据、配置算子/参数、展示纯谱与响应曲线。
- 若添加算子或输出字段，需同步更新 GUI 显示与标签。

## 示例与测试
- 运行示例：
  - `python examples/demo_discovery_csv.py`
  - `python examples/demo_discovery_eem.py`
  - `python examples/demo_discovery_synthetic.py`
  - `python examples/demo_discovery_visual.py`
- 运行测试：`pytest -q`

## 常见修改提示
- 新增算子：在 `discovery/symbolic.py` 或 `discovery/library/*` 中添加，并确保管线/GUI/测试映射一致。
- 平移估计：填充 `g_shift`（样本×组分），频域相位会自动生效。
- 纯谱校准：`calibrate_pure_spectra_once` 支持 ridge，监测谱崩塌后回退原谱。

## 最小护栏
- 波长轴需等间距（用于直接 FFT/IFFT）。
- `f_response` 非负裁剪；纯谱经厄米对称保证 IFFT 实值。
- 诊断字段保持向后兼容（rank/sigma_gap 等旧名映射已保留）。

---
面向智能体的快捷心智模型：
- 先看 `pipeline.run_discovery` 主流程，再看 `factorization.calibrate_pure_spectra_once` 与 `symbolic.py` 的算子构造。
- 平移=g_shift→频域相位，当前默认零；想恢复估计就把 g_shift 算出来再喂回去。
=======
# OPERA 项目智能体速览

## 项目概况
- 领域：光谱分解与符号发现（SINDy-PI 风格），含 GUI（PySide6）。
- 语言/依赖：Python；核心依赖见 `requirements.txt`（numpy、scipy、sympy、pandas、scikit-learn、matplotlib、PySide6 等）。
- 核心目录：
  - `src/opera/discovery/`: 管线、算子库、稀疏回归与谱分解。
  - `src/opera/gui/`: 主窗口与绘图。
  - `examples/`: CSV/EEM/可视化示例。
  - `tests/`: pytest 冒烟与语义测试。

## 关键概念与变量
- `f_response`：样本×组分的强度响应（替代旧的 `C_real`/`f_est`）。
- `g_shift`：样本×组分的光谱位移（目前默认置零，可外部提供）。
- 纯谱：频域系数 `p_freq_complex`（经厄米对称）与波长域 `p_real`。
- 发现结果：`DiscoveryResult`（见 `types.py`）包含 `f_response`, `g_shift`, `p_real`, 诊断信息等。

## 核心流程（`pipeline.run_discovery`）
1. 预处理、算子矩阵生成（符号/算子见 `symbolic.py`, `library/*`）。
2. 求解 Gamma（线性+稀疏 ALS），再做分解得到算子系数 `xi` 与纯谱频域 `p_freq_complex`。
3. 对频谱施加厄米对称 → IFFT 得到 `p_real`。
4. 线性最小二乘求 `f_response`；`g_shift` 当前置零（弃用 phi/psi 路径）。
5. 可选一次性纯谱校准 `calibrate_pure_spectra_once`（ridge 约束，检测谱崩塌）。
6. 诊断与输出封装为 `DiscoveryResult`。

## 光谱平移处理
- 表示方式：在频域用相位 `exp(i * ω * g_shift)` 表示位移，IFFT 后对应波长平移。
- 现状：积分路径已移除，`g_shift` 默认全零，但接口保留；若外部提供 `g_shift`，所有重建/校准逻辑会自动使用。
- 新增：`DiscoveryConfig.estimate_g_shift=True` 可启用频域相位的非线性 least-squares 拟合，估计每个样本/组分的 `g_shift`（可用 `g_shift_max_abs` 限制幅度）。
- 若需更复杂估计，可在 Gamma 分解后，通过频域线性拟合估计 `g_shift`，或复原旧 phi/psi 逻辑。

## 算子库与符号发现
- 基本算子：`1, c, f, f', c f, c f', f f'` 等（SINDy-PI 不会自动生成乘积，需显式加入）。
- 目标：可拟合多项式、幂函数、指数、对数、Logistic/饱和型等；三角/更高阶需增加二阶导或特定算子。
- 多变量：`c_i, c_i f, c_i f'_{c_i}, f f'_{c_i}, c_i c_j` 等；若需扩散/协同，考虑二阶混合偏导。

## GUI 提示
- 入口：`src/opera/gui/main_window.py`。
- 主要功能：加载数据、配置算子/参数、展示纯谱与响应曲线。
- 若添加算子或输出字段，需同步更新 GUI 显示与标签。

## 示例与测试
- 运行示例：
  - `python examples/demo_discovery_csv.py`
  - `python examples/demo_discovery_eem.py`
  - `python examples/demo_discovery_synthetic.py`
  - `python examples/demo_discovery_visual.py`
- 运行测试：`pytest -q`

## 常见修改提示
- 新增算子：在 `discovery/symbolic.py` 或 `discovery/library/*` 中添加，并确保管线/GUI/测试映射一致。
- 平移估计：填充 `g_shift`（样本×组分），频域相位会自动生效。
- 纯谱校准：`calibrate_pure_spectra_once` 支持 ridge，监测谱崩塌后回退原谱。

## 最小护栏
- 波长轴需等间距（用于直接 FFT/IFFT）。
- `f_response` 非负裁剪；纯谱经厄米对称保证 IFFT 实值。
- 诊断字段保持向后兼容（rank/sigma_gap 等旧名映射已保留）。

---
面向智能体的快捷心智模型：
- 先看 `pipeline.run_discovery` 主流程，再看 `factorization.calibrate_pure_spectra_once` 与 `symbolic.py` 的算子构造。
- 平移=g_shift→频域相位，当前默认零；想恢复估计就把 g_shift 算出来再喂回去。
>>>>>>> 2d9cf06 (Initial commit)
- SINDy-PI 不会自动乘积算子，算子库要显式列出你想要的乘积/幂/导数。