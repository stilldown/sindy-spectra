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

## 物理约束原理：欧拉算子与隐式微分方程发现

**核心理念**：算法本质是用 **微分方程（Euler 型 ODE/PDE）来约束光谱分解**，而非盲源分离后再拟合参数。

### 1. 光谱模型

频域观测数据分解为 K 个组分之和：

```
D̂(c, ω) ≈ Σ_k  A_k(c) · φ_k(ω)
```

其中每个组分的复数幅度满足：

```
A_k(c) = f_k(c) · exp(−i ω_k · g_k(c))
```

- `φ_k(ω)` — 第 k 个纯光谱（频域基向量，由 SVD 或伪逆确定）
- `f_k(c) = exp(Re(ln A_k))` — 第 k 组分的**强度响应**（正实数）
- `g_k(c) = −Im(ln A_k) / ω_k` — 第 k 组分的**相位斜率**（光谱位移）
- `ω_k` — 模式 k 的特征频率（由 `spectral_basis @ diag(ω) @ spectral_basis†` 的对角元给出）

### 2. Euler 算子库（微分方程候选项）

对每个组分 k，构造以下算子（均为 `c` 方向的 Euler 型微分算子）：

| 库条目 | 定义 | 含义 |
|--------|------|------|
| `ln_f` | `Re(ln A_k)` | 对数强度（零阶 Euler 项） |
| `g` | `−Im(ln A_k) / ω_k` | 相位斜率（零阶） |
| `L1_j_f` | `c_j · ∂(ln f_k)/∂c_j` | **一阶 Euler 算子**作用于 ln f |
| `L1_j_g` | `c_j · ∂g_k/∂c_j` | 一阶 Euler 算子作用于 g |
| `Xi2_ij_f` | `c_i c_j ∂²(ln f_k)/∂c_i∂c_j` (i≠j) 或 `c_i² ∂²(ln f_k)/∂c_i² + L1_i_f` (i=j) | **二阶 Euler 算子**（含一阶修正项） |
| `Xi2_ij_g` | 同上，作用于 g | 二阶 Euler 算子作用于 g |
| `ln_f^2`, `g^2` | 平方非线性项 | 非线性 Euler 项 |

> **关键**：以上每个条目均是 `∂(ln f)/∂c`（或更高阶）的 Euler 缩放版本，它们共同构成一族"候选微分方程项"。

### 3. SINDy-PI 零空间 = 隐式 ODE 发现

算法通过 SVD 最小奇异向量找稀疏系数 `ξ`，使得：

```
Σ_j  ξ_j · Ω_j(ln f_k)  ≈  0     （f 方向隐式 ODE）
Σ_j  ξ_j · Ω_j(g_k)     ≈  0     （g 方向隐式 ODE）
```

这等价于**发现约束光谱分解的隐式微分方程**——不同于盲源分离，这里的组分分解被"物理方程"约束，而不是任意自由分离。

### 4. 可发现的物理模型示例

| 物理模型 | f(c) 形式 | 被发现的 ODE | 方程意义 |
|----------|-----------|-------------|---------|
| Beer-Lambert 吸收 | `exp(−ε c)` | `L1_f − ln_f = 0` | `c ∂(ln f)/∂c = ln f` → f 关于 ln c 线性 |
| Beer-Lambert（二阶验证） | `exp(−ε c)` | `Xi2_f − L1_f = 0` | `c² ∂²(ln f)/∂c² + c ∂(ln f)/∂c = c ∂(ln f)/∂c` → 即上式 |
| 高斯型浓度响应 | `exp(−ε c²)` | `L1_f − 2·ln_f = 0` | `c ∂(ln f)/∂c = 2 ln f` → f 关于 c² 指数型 |
| 幂次律 | `c^n` | `Xi2_f = 0` | `c² ∂²(ln f)/∂c² + c ∂(ln f)/∂c = 0` → Euler 方程，解为 c^n |
| Langmuir/Hill（近似） | `K c/(1+K c)` | 含 `ln_f^2` 非线性项 | 超越方程，需高阶库项 |

### 5. 为什么必须有零浓度锚点

- 所有 Euler 算子在 `c_i = 0` 处自然为零：`L_i(c=0) = c_i · ∂(ln f)/∂c_i |_{c=0} = 0`
- 零浓度样本（纯溶剂/空白）提供了**物理参考**（`f(0) = 1` → `ln f(0) = 0`）
- 缺少零锚点会导致算子计算的相对意义丧失，无法确定绝对强度标定

---

## 核心流程（`pipeline.run_discovery`）

1. **输入验证 + 零锚点检测**（`validate_inputs`，`preprocess.py`）
2. **FFT**：光谱矩阵去均值后做 `rfft` → 频域矩阵 `D̂(N, P)` + 频率轴 `ω`
3. **数值微分**：在控制变量网格上计算 `∂D̂/∂c_j`、`∂²D̂/∂c_i∂c_j`（中心差分，要求等间距笛卡尔网格）
4. **Euler 算子库构建**（三路可选）：
   - **默认（SVD pure Euler）**：SVD 取前 K 个谱基 `Φ`，`A = D̂ @ Φ†`，在投影空间计算 α/β/L1/Xi2 → 库形状 `(N, K)`
   - **伪逆（`use_inverse_operator=True`）**：`D̂ @ D̂†`（N×N 帽矩阵），算子在完整样本空间计算
   - **弱形式（`use_weak_form=True`）**：IBP 把微分算子转移到多项式测试函数上，不需要数值微分 `dD̂/dc`
   - **直接 Euler（`use_direct_euler=True`）**：用 `W=[1, −iω]` 伪逆在全频段同时拟合 f/g，K=1（全局方程）
5. **SINDy-PI 零空间**（`solve_nullspace`）：对 f/g 分别做 SVD，取最小奇异向量 = 隐式 ODE 系数 `ξ`
6. **组装输出**：`ξ` 填入 `Xi(1, J, K)`；`irfft(Φ)` → `S_real`（纯组分光谱）；封装为 `DiscoveryResult`

## 弱形式算子库（Weak-Form Operator Library）

`operator.py` 中有两个相关函数：

- `compute_weak_operators`：近似弱形式（仍需 `d_d_c`），用于 `use_inverse_operator=True` 路径，本质是强形式算子加权。
- `build_weak_form_library`：**弱形式 + SVD 组分分离**实现：
  - **不需要 `d_d_c`**，仅使用 `d_hat`（原始频域数据）
  - **关键步骤：先用 SVD 分离组分，再做 IBP**

  > 若直接对 `ln D̂(c,ω)` 做 IBP，由于 D̂ 是多组分叠加，无法分离各组分的方程。
  > 正确做法是先投影到 SVD 谱基将组分分离，再对每个组分的对数投影系数 `ln A_k(c)` 做 IBP。

  完整算法步骤：
  ```
  步骤 1：SVD 谱基    P = Vt[:K, :] ∈ ℂ^{K×P}
  步骤 2：投影        A = D̂ @ P†  ∈ ℂ^{N×K}   （各列 = 单一组分响应）
  步骤 3：对数        ln_A = log(A + ε)  ∈ ℂ^{N×K}
  步骤 4：IBP 内积    ⟨c_i ∂_{c_i} ln A_k, ψ_m⟩
                        = -Σ_n (∂_{c_i}ψ_m(c_n)·c_i(n) + ψ_m(c_n)) · ln A_k(c_n)
  ```
  - 测试函数梯度 `∂_{c_i}ψ_m` 由多项式解析计算（精确、无噪声）
  - 库形状 `(M, K)`，与 `solve_nullspace` 直接兼容（M 充当"样本"轴，K = 谱组分数）
  - 函数返回 5 个值：`(library, names, Psi, spectral_basis, A)`

### 配置方式

```python
cfg = DiscoveryConfig(
    use_weak_form=True,          # 启用弱形式+SVD组分分离管线
    weak_form_test_degree=2,     # 测试函数多项式阶数（默认 2）
    k_mode="fixed",
    k_value=3,
)
out = run_discovery(spectra, factors, wavelengths, cfg)
```

## 光谱平移处理
- `g_shift`（光谱位移）已从代码库中完全移除。该概念在历史实现中始终为零，且无实际估计路径，属于多余设计。
- 如未来需要重新引入，应在 `pipeline.py` 中添加估计逻辑，并在 `DiscoveryResult` 和 `DiscoveryConfig` 中重新声明相关字段。

## 算子库与微分方程约束

代码中的算子库 **不是** 任意特征工程，而是一族 **Euler 型微分方程候选项**：

```
库中每列 = Euler 算子 Ω_j 作用于 ln_f_k（或 g_k）在所有样本点的取值
零空间 ξ = 使 Σ ξ_j Ω_j = 0 的稀疏系数 = 隐式 ODE 的系数
```

**如何扩展算子库**：
- 新增算子类型时，须确保它仍是 `ln_f` 或 `g` 的 Euler-型表达式（否则方程无物理意义）
- 若需高阶 Euler 项（如 `c³ ∂³(ln f)/∂c³`），在 `pipeline_utils.py` / `operator.py` 中添加
- 非线性项（如 `ln_f^2`）用于捕获超越 Euler 线性方程的物理行为（如 Langmuir/Hill）
- 跨组分交叉项（如 `L1_f * L1_g`）目前未实现，若物理上 f 与 g 存在耦合可按需添加
- 多变量控制下，非对角 `Xi2_c{i}c{j}` 算子捕获浓度间的交叉 Euler 偏导（对应混合偏微分方程项）

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
- **算法核心**：用 Euler 微分算子（L1, Xi2）构建 ODE 候选库，SINDy-PI 零空间发现约束光谱分解的隐式方程。
- 先看 `pipeline.run_discovery` 主流程，入口为四路管线（pure/direct_euler/inverse/weak_form）。
- `construct_pure_library`（默认路径）：SVD 分离组分 → 对每个投影组分 A_k 计算 α/β → Euler 算子库。
- `build_direct_euler_library`（`use_direct_euler=True`）：全频段 W=[1,-iω] 拟合 f/g → 单方程（K=1）。
- 算子库条目是 `ln_f` 或 `g` 的 Euler 型微分算子取值，**不是**任意特征。
- `g_shift` 已移除；光谱位移由 `g_k(c)` 通过相位算子体现。

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
| `build_weak_form_library` | ✅ | **弱形式+SVD**：先投影分离组分再 IBP，`use_weak_form=True` 时启用；返回 `(lib, names, Psi, basis, A)` |
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
