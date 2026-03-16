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
- `f_response`：样本×组分的强度响应。
- 纯谱：波长域 `S_real`（IFFT 谱基），频域 `pure_spectra_complex`。
- 发现结果：`DiscoveryResult`（见 `types.py`）包含 `f_response`, `S_real`, `Xi`, 诊断信息等。
  注：`g_shift`（光谱位移）已彻底移除，历史上始终为零无实际意义。

## 核心流程（`pipeline.run_discovery`）
1. 预处理、算子矩阵生成（算子见 `pipeline_utils.py`、`operator.py`）。
2. 三路管线选择：
   - **默认（pure Euler）**：`construct_pure_library` → SVD谱基投影，`solve_nullspace`
   - **伪逆（`use_inverse_operator=True`）**：`construct_inverse_library` → 伪逆算子，合并弱算子
   - **真弱形式（`use_weak_form=True`）**：`build_weak_form_library` → IBP 无需数值微分
3. `solve_nullspace` 提取 f/g 零空间系数，组装 `Xi`。
4. IFFT 谱基 → `S_real`（纯谱），投影系数 A → `f_response_eval`。
5. 诊断与输出封装为 `DiscoveryResult`。

## 弱形式算子库（Weak-Form Operator Library）

`operator.py` 中有两个相关函数：

- `compute_weak_operators`：近似弱形式（仍需 `d_d_c`），用于 `use_inverse_operator=True` 路径，本质是强形式算子加权。
- `build_weak_form_library`：**真正的弱形式**实现：
  - **不需要 `d_d_c`**，仅使用 `d_hat`（原始频域数据）
  - 通过分部积分（IBP）将导数从含噪数据 D 转移到光滑测试函数 ψ_m：
    ```
    ⟨L_i, ψ_m⟩(ω) = -Σ_n (∂_{c_i}ψ_m(c_n)·c_i(n) + ψ_m(c_n)) · ln D(c_n, ω)
    ```
  - 测试函数的梯度 `∂_{c_i}ψ_m` 由多项式解析计算（精确、无噪声）
  - 库形状 `(M, k_eff)`，与 `solve_nullspace` 直接兼容（M 充当"样本"轴）

### 配置方式

```python
cfg = DiscoveryConfig(
    use_weak_form=True,          # 启用真正弱形式管线
    weak_form_test_degree=2,     # 测试函数多项式阶数（默认 2）
    k_mode="fixed",
    k_value=3,
)
out = run_discovery(spectra, factors, wavelengths, cfg)
```

## 光谱平移处理
- `g_shift`（光谱位移）已从代码库中完全移除。该概念在历史实现中始终为零，且无实际估计路径，属于多余设计。
- 如未来需要重新引入，应在 `pipeline.py` 中添加估计逻辑，并在 `DiscoveryResult` 和 `DiscoveryConfig` 中重新声明相关字段。

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
- 新增算子：在 `discovery/pipeline_utils.py` 或 `discovery/operator.py` 中添加，并确保管线/GUI/测试映射一致。
- 弱形式测试函数阶数：在 `DiscoveryConfig(weak_form_test_degree=N)` 中调整。

## 最小护栏
- 波长轴需等间距（用于直接 FFT/IFFT）。
- 控制变量须构成等间距笛卡尔网格（用于中心差分数值微分）。
- 必须提供至少一个零浓度锚点样本（所有控制因子均为 0）。
- 诊断字段保持向后兼容（rank/sigma_gap 等旧名映射已保留）。

---
面向智能体的快捷心智模型：
- 先看 `pipeline.run_discovery` 主流程，入口为三路管线（pure/inverse/weak_form）。
- 算子库在 `pipeline_utils.py`（pure）和 `operator.py`（inverse/weak_form）中。
- `g_shift` 已移除；如需位移特征须重新设计。
- SINDy-PI 不会自动乘积算子，算子库要显式列出你想要的乘积/幂/导数。

---

## 实现状态清单（统计截止 2026-03-16，已完成清理）

> 图例：✅ 已实现且接入主管线 ｜ 🟡 已实现但未接入/仅独立调用 ｜ ❌ 未实现或存在明显缺口

### 一、预处理层（`preprocess.py`）

| 功能 | 状态 | 备注 |
|------|------|------|
| `validate_inputs` | ✅ | 校验形状/一致性 |
| `compute_fourier_tensor` | ✅ | rfft + 归一化频率轴 |
| `_detect_cartesian_grid` | ✅ | 笛卡尔网格检测 |
| `estimate_control_derivatives_scattered` | ✅ | 仅支持笛卡尔等间距网格 |
| `estimate_control_second_derivatives_scattered` | ✅ | 同上 |
| `build_control_derivative_bundle` | ✅ | 含锚点处理 |
| **散点数据支持** | ❌ | 非等间距笛卡尔网格无法处理 |

### 二、强形式算子库（`pipeline_utils.py`）

| 功能 | 状态 | 备注 |
|------|------|------|
| `construct_pure_library` | ✅ | SVD谱基投影，主管线默认路径 |
| `solve_nullspace` | ✅ | f/g 空间零空间解 |
| `pretty_name` | ✅ | 算子名格式化 |

### 三、算子层（`operator.py`）

| 功能 | 状态 | 备注 |
|------|------|------|
| `construct_inverse_library` | 🟡 | 伪逆算子库，`use_inverse_operator=True` 时启用 |
| `_build_polynomial_test_functions_with_grads` | ✅ | 多项式测试函数及解析梯度 |
| `build_weak_form_library` | ✅ | 真正弱形式（IBP，无需 d_d_c），`use_weak_form=True` 时启用 |
| `compute_weak_operators` | 🟡 | 近似弱形式（仍需 d_d_c），`use_inverse_operator=True` 路径使用 |

### 四、符号识别（`symbolic.py`）

| 功能 | 状态 | 备注 |
|------|------|------|
| `build_latex_blocks_from_xi` | 🟡 | SymPy `dsolve`/`pdsolve` 驱动方程块生成；多变量 PDE 成功率低，大量 `except: pass` 静默回退 |

### 五、发现管线（`pipeline.py`）

| 功能 | 状态 | 备注 |
|------|------|------|
| `run_discovery` 主流程 | ✅ | 三路入口（pure/inverse/weak_form） |
| 锚点检测 | ✅ | |
| 统一输出 `DiscoveryResult` | ✅ | |
| `reconstruction_error` | ❌ | 固定返回 `0.0`，未实际计算 |
| `S_real` 纯谱重建（弱形式/伪逆路径） | ❌ | 弱形式/伪逆路径用 SVD 谱基占位，未真正重建纯谱 |

### 六、GUI（`src/opera/gui/`）

| 功能 | 状态 | 备注 |
|------|------|------|
| 主窗口框架（导入/运行/展示） | ✅ | |
| 后台分析线程 | ✅ | |
| 数据加载（CSV/XLSX/演示数据） | ✅ | |
| 光谱/纯谱/响应曲线可视化 | ✅ | |
| Discovery 模型 LaTeX 渲染（WebEngine） | ✅ | 离线时 MathJax CDN 不可用 |
| `update_library_from_ui` | ❌ | 方法体为空 `pass`，控件变化无响应 |
| `use_inverse_operator` / `use_weak_form` 开关暴露 | ❌ | GUI 无对应控件 |
| 组分评分/能量占比诊断显示 | ❌ | 值均为 `None`，GUI 显示"未输出" |

### 七、测试（`tests/`）

| 测试文件 | 状态 | 覆盖内容 |
|---------|------|---------|
| `test_discovery_smoke.py` | ✅ 6 个测试 | 主管线端到端、锚点检测、三控制变量、固定 K、弱算子 |
| `test_weak_form_library.py` | ✅ 19 个测试 | 弱形式算子库形状、IBP 精确性、管线集成 |
| **symbolic.py 覆盖** | ❌ | `build_latex_blocks_from_xi` 无测试 |

### 八、文档

| 内容 | 状态 | 备注 |
|------|------|------|
| `AGENT_GUIDE.md` | ✅ | 算法说明、弱形式说明、实现状态清单 |
| `README.md` | ❌ | 几乎为空，仅有项目名称 |
| 公共 API docstring | 🟡 | 核心函数均有中文文档字符串，但无 Sphinx/MkDocs 配置 |
| `examples/` 示例脚本 | 🟡 | 4 个脚本存在，但未验证在当前 API 下可运行 |

---

### 已清除的旧代码（2026-03-16）

以下模块因从未接入主管线（`run_discovery`）而已删除：

| 已删除模块 | 原功能说明 |
|-----------|-----------|
| `discovery/joint_diag.py` | 1.md 算法五阶段完整实现（Euler算子→f/g分离→联合对角化→候选库→SINDy-PI零空间），从未被主管线调用 |
| `discovery/decoder.py` | 物理流形解码 `decode_physical_manifolds`，Picard迭代恢复 f(c)/g(c)，从未被主管线调用 |
| `discovery/factorization.py` | `find_joint_nullspace`、`extract_sparse_physical_coefficients`、`select_component_count`、`calibrate_pure_spectra_once`，均未接入主管线 |
| `discovery/library/observable.py` | 12类物理约束算子库，从未被主管线调用 |
| `discovery/library/kron.py` | Kronecker展开 `build_psi`，无上游调用 |
| `discovery/library/lie_basis.py` | Lie基函数 `build_lie_basis`，无上游调用 |
| `discovery/regression/complex_sparse.py` | STRidge稀疏化 `nullspace_sparse_vectors`，无上游调用 |
| `discovery/legacy/` | 遗留备份文件 |
| `tests/test_joint_diag.py` | 已删除模块的测试 |
| `tests/test_library_auto_dims.py` | 已删除模块的测试 |

以下字段因逻辑上始终为零/无意义而已从类型定义和管线中移除：

| 已移除字段 | 位置 | 原因 |
|-----------|------|------|
| `DiscoveryResult.g_shift` | `types.py` | 始终为 `np.zeros(...)`，光谱位移从未实际估计 |
| `DiscoveryConfig.estimate_g_shift` | `types.py` | 开关存在但管线中无对应执行路径 |
| `DiscoveryConfig.g_shift_max_abs` | `types.py` | 同上 |
| `DiscoveryConfig.calibrate_pure_spectra_once` | `types.py` | 字段存在但 `calibrate_pure_spectra_once()` 从未被调用 |
| `DiscoveryConfig.calibration_ridge` | `types.py` | 同上 |
| `DiscoveryConfig.gamma_sparse_iters` | `types.py` | 遗留字段，无对应逻辑 |
| `DiscoveryConfig.gamma_pivot_operator` | `types.py` | 遗留字段，无对应逻辑 |

---

### 优先级最高的待完成项

1. **实现真实的 `reconstruction_error`**（而非固定 0.0）
2. **弱形式/伪逆路径实现真正的纯谱重建**（而非 SVD 谱基占位）
3. **GUI 暴露 `use_weak_form` / `use_inverse_operator` 开关**
4. **填充 `update_library_from_ui` 方法**，使 GUI 参数变化即时响应
5. **补充 `symbolic.py` 的测试覆盖**
6. **完善 `README.md`**，添加安装说明和快速使用示例
