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
5. 可选一次性纯谱校准 `calibrate_pure_spectra_once`（ridge 约束，检测谱崩塌后回退原谱）。
6. 诊断与输出封装为 `DiscoveryResult`。

## 谱-物理联合解析流算法（1.md 核心逻辑）
本项目实现了 `.github/prompts/1.md` 中描述的"谱-物理联合解析流"算法，核心模块在
`src/opera/discovery/joint_diag.py`：

1. **Euler 算子构造**（`build_euler_operators`）：逐点（element-wise）构造算子
   - 一阶：`L_i(c,ω) = D⁻¹ ⊙ (c_i ∂_{c_i}D)`
   - 二阶：`Ξ_{ii}(c,ω) = D⁻¹ ⊙ (c_i² ∂²_{c_i}D) - (L_i² - L_i)`
   - 交叉：`Ξ_{ij}(c,ω) = D⁻¹ ⊙ (c_i c_j ∂²_{c_i c_j}D) - L_i L_j`
2. **f/g 频域分离**（`separate_fg`）：利用 W=[1,-iω] 超定线性系统将算子分解为 f 分量（强度）与 g 分量（相位）
3. **联合对角化**（`joint_diagonalize`）：寻找使所有算子同时（近似）对角化的基 B(ω)
4. **物理候选库构造**（`build_physical_candidate_library`）：为 SINDy-PI 构建多项式/非线性候选函数库

## 弱形式算子库（Weak-Form Operator Library）

### 当前回答：`compute_weak_operators` ≠ 真正弱形式

`operator.py` 中的 `compute_weak_operators` **不是**真正的弱形式实现：
- 仍以 `d_d_c`（对含噪数据的数值微分结果）作为输入
- 只是事后将算子乘以测试函数权重，附加一个近似修正项
- 本质是强形式算子加权，而非分部积分（IBP）

### 真正的弱形式：`build_weak_form_library`

`operator.py` 中的 `build_weak_form_library` 是**真正的弱形式**实现：
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

### IBP 边界项说明
弱形式（IBP）与强形式点乘 `Σ_n ψ_m·L_i` 之间差一个边界项。当测试函数在控制空间
边界处为零（紧支撑）时两者相等（即 WSINDy 的标准设置）。对非紧支撑的多项式测试函数，
边界项非零，但输出仍是物理方程识别的有效弱形式残差，可直接用于稀疏回归。

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
- SINDy-PI 不会自动乘积算子，算子库要显式列出你想要的乘积/幂/导数。
- 联合对角化（1.md 算法）：`joint_diag.py` 中提供了完整的 Euler 算子、f/g 分离和联合对角化实现。

---

## 实现状态清单（统计截止 2026-03-16）

> 图例：✅ 已实现且接入主管线 ｜ 🟡 已实现但未接入/仅独立调用 ｜ ❌ 未实现或存在明显缺口

### 一、预处理层（`preprocess.py`）

| 功能 | 状态 | 备注 |
|------|------|------|
| `validate_inputs` | ✅ | 校验形状/一致性 |
| `compute_fourier_tensor` | ✅ | rfft + 归一化频率轴 |
| `_detect_cartesian_grid` | ✅ | 笛卡尔网格检测 |
| `estimate_control_derivatives_scattered` | 🟡 | 仅支持笛卡尔等间距网格；非网格直接 `raise ValueError`，kNN/RBF 路径已移除 |
| `estimate_control_second_derivatives_scattered` | 🟡 | 同上，仅网格 |
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
| `compute_weak_operators` | 🟡 | 近似弱形式（遗留，仍需 d_d_c），不是真正 IBP |
| `_compute_control_gradient` | 🟡 | 仅笛卡尔网格 |

### 四、联合对角化算法（`joint_diag.py`，1.md 算法）

| 功能 | 状态 | 备注 |
|------|------|------|
| `build_euler_operators` | 🟡 | Euler 算子（一/二阶/交叉），仅独立管线调用 |
| `separate_fg` / `separate_all_operators_fg` | 🟡 | f/g 频域分离 |
| `joint_diagonalize` | 🟡 | 联合对角化（特征值分解） |
| `joint_diag_residual` | 🟡 | 残差用方差近似，非严格 off-diagonal 范数 |
| `build_physical_candidate_library` | 🟡 | 多项式+对数+倒数候选库 |
| `sindy_pi_nullspace` | 🟡 | SINDy-PI 零空间稀疏回归 |
| `run_joint_diag_pipeline` | 🟡 | 完整 1.md 管线入口 |
| **与 `run_discovery` 主管线的连接** | ❌ | `joint_diag.py` 完全独立，主管线从未调用它 |

### 五、因子分解与稀疏化（`factorization.py`）

| 功能 | 状态 | 备注 |
|------|------|------|
| `select_component_count` | 🟡 | fixed/auto/capped-auto，已实现但未被主管线调用 |
| `find_joint_nullspace` | 🟡 | 物理零空间提取（特征值法） |
| `extract_sparse_physical_coefficients` | 🟡 | DEIM 启发稀疏旋转 |
| `calibrate_pure_spectra_once` | 🟡 | 单次复频域纯谱校准，已实现但主管线中**未调用** |

### 六、物理解码（`decoder.py`）

| 功能 | 状态 | 备注 |
|------|------|------|
| `build_1d_first_derivative_matrix` | 🟡 | 非均匀网格中心差分稀疏矩阵 |
| `build_grid_derivative_matrices` | 🟡 | 全局稀疏偏导算子 |
| `_parse_term_operator` | 🟡 | 方程项字符串→算子解析 |
| `decode_physical_manifolds` | 🟡 | Xi 系数→f(c)/g(c) Picard 迭代恢复 |
| **与主管线连接** | ❌ | `run_discovery` 从未调用 `decode_physical_manifolds`；f(c)/g(c) 直接从投影 A 获取 |

### 七、可观测算子库（`library/observable.py`）

| 功能 | 状态 | 备注 |
|------|------|------|
| `build_observable_library` | 🟡 | 12类物理约束算子库（4驱动×3频率变换），接口完整 |
| **与主管线连接** | ❌ | 主管线走 `construct_pure_library`，此模块从未被调用 |

### 八、其他库模块（`library/kron.py`, `library/lie_basis.py`）

| 功能 | 状态 | 备注 |
|------|------|------|
| `build_psi` (Kronecker) | 🟡 | 已实现，无上游调用 |
| `build_lie_basis` | 🟡 | W0/W1/W2 频率基，已实现，无上游调用 |

### 九、稀疏回归（`regression/complex_sparse.py`）

| 功能 | 状态 | 备注 |
|------|------|------|
| `nullspace_sparse_vectors` | 🟡 | STRidge 稀疏化（已实现），主管线走 `solve_nullspace`，此模块未被调用 |

### 十、符号识别（`symbolic.py`）

| 功能 | 状态 | 备注 |
|------|------|------|
| `build_latex_blocks_from_xi` | 🟡 | SymPy `dsolve`/`pdsolve` 驱动方程块生成；多变量 PDE 成功率低，大量 `except: pass` 静默回退；Unicode 编解码问题（`∂²D?`）；仅输出 LaTeX 字符串，无可执行 Python/SymPy 表达式 |

### 十一、发现管线（`pipeline.py`）

| 功能 | 状态 | 备注 |
|------|------|------|
| `run_discovery` 主流程 | ✅ | 三路入口（pure/inverse/weak_form） |
| 锚点检测 | ✅ | |
| 统一输出 `DiscoveryResult` | ✅ | |
| `reconstruction_error` | ❌ | 固定返回 `0.0`，未实际计算 |
| `g_shift` 估计 | ❌ | 固定全零；`estimate_g_shift` 配置字段存在但管线未调用估计逻辑 |
| `S_real` 纯谱重建（弱形式/伪逆路径） | ❌ | 弱形式/伪逆路径用 SVD 谱基占位，未真正重建纯谱 |
| `calibrate_pure_spectra_once` 调用 | ❌ | 定义在 `factorization.py`，主管线未调用 |

### 十二、GUI（`src/opera/gui/`）

| 功能 | 状态 | 备注 |
|------|------|------|
| 主窗口框架（导入/运行/展示） | ✅ | |
| 后台分析线程 | ✅ | |
| 数据加载（CSV/XLSX/演示数据） | ✅ | |
| 光谱/纯谱/响应曲线可视化 | ✅ | |
| Discovery 模型 LaTeX 渲染（WebEngine） | ✅ | 离线时 MathJax CDN 不可用 |
| `update_library_from_ui` | ❌ | 方法体为空 `pass`，控件变化无响应 |
| `use_inverse_operator` / `use_weak_form` 开关暴露 | ❌ | GUI 无对应控件 |
| `estimate_g_shift` 开关暴露 | ❌ | GUI 无对应控件 |
| 组分评分/能量占比诊断显示 | ❌ | 值均为 `None`，GUI 显示"未输出" |

### 十三、测试（`tests/`）

| 测试文件 | 状态 | 覆盖内容 |
|---------|------|---------|
| `test_discovery_smoke.py` | ✅ 18 个测试 | 主管线端到端、锚点检测、三控制变量、固定 K |
| `test_joint_diag.py` | ✅ 14 个测试 | joint_diag.py 全部五个阶段 |
| `test_library_auto_dims.py` | ✅ 0 个测试 | （空文件或未填充） |
| `test_weak_form_library.py` | ✅ 20 个测试 | 弱形式算子库形状、IBP 精确性、管线集成 |
| **decoder.py 覆盖** | ❌ | decode_physical_manifolds 无测试 |
| **observable.py 覆盖** | ❌ | build_observable_library 无测试 |
| **symbolic.py 覆盖** | ❌ | build_latex_blocks_from_xi 无测试 |
| **factorization.py 覆盖** | 🟡 | find_joint_nullspace 有1个测试，其余无 |

### 十四、文档

| 内容 | 状态 | 备注 |
|------|------|------|
| `AGENT_GUIDE.md` | ✅ | 算法说明、弱形式说明、实现状态清单 |
| `README.md` | ❌ | 几乎为空，仅有项目名称 |
| 公共 API docstring | 🟡 | 核心函数均有中文文档字符串，但无 Sphinx/MkDocs 配置 |
| `examples/` 示例脚本 | 🟡 | 4 个脚本存在，但未验证在当前 API 下可运行 |

---

### 优先级最高的待完成项

1. **将 `joint_diag.py` 接入 `run_discovery`**（或通过 `DiscoveryConfig.use_joint_diag=True` 切换），使 1.md 算法真正可用。
2. **实现真实的 `reconstruction_error`**（而非固定 0.0）。
3. **实现 `g_shift` 估计路径**（当 `estimate_g_shift=True` 时）。
4. **将 `calibrate_pure_spectra_once` 接入主管线**（当 `calibrate_pure_spectra_once=True` 时）。
5. **GUI 暴露 `use_weak_form` / `use_inverse_operator` / `estimate_g_shift` 开关**。
6. **填充 `update_library_from_ui` 方法**，使 GUI 参数变化即时响应。
7. **补充 `decoder.py`、`symbolic.py`、`observable.py` 的测试覆盖**。
8. **完善 `README.md`**，添加安装说明和快速使用示例。
