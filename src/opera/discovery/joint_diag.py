"""谱-物理联合解析流算法（Spectro-Physical Joint Analytical Flow）。

实现 `.github/prompts/1.md` 中描述的完整算法逻辑，包含以下五个核心阶段：

阶段 1：Euler 算子构造（element-wise，无 SVD 降维）
  - L_i(c, ω) = D⁻¹ ⊙ (c_i ∂_{c_i} D)                     （一阶对数演化算子）
  - Ξ_{ii}(c, ω) = D⁻¹ ⊙ (c_i² ∂²_{c_i} D) - (L_i² - L_i) （二阶纯净算子）
  - Ξ_{ij}(c, ω) = D⁻¹ ⊙ (c_i c_j ∂²_{c_i c_j} D) - L_i L_j（交叉耦合算子）

阶段 2：f/g 频域分离
  - 利用 W = [1, -iω] 超定线性系统将算子特征值分解为 f 分量（强度）与 g 分量（相位/位移）

阶段 3：联合对角化（Joint Diagonalization）
  - 寻找变换基 B(ω)，使算子集合在该基下同时（近似）对角化

阶段 4：物理候选库构造（为 SINDy-PI 准备 Θ 矩阵）
  - 多项式 + 可选非线性函数，维度 (N_total, J)

阶段 5：SINDy-PI 零空间稀疏回归
  - 求解 Θ Ξ ≈ 0，提取物理方程参数
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# 阶段 1：Euler 算子构造
# ---------------------------------------------------------------------------

def build_euler_operators(
    d_hat: np.ndarray,
    d_d_c: np.ndarray,
    d2_d_c: np.ndarray,
    factors: np.ndarray,
) -> Dict[str, np.ndarray]:
    r"""构造逐点（element-wise）Euler 算子集合，不进行 SVD 降维投影。

    实现以下三类算子（均为张量逐点运算，形状与 d_hat 一致）：

    - 一阶算子：
      ``L_i(c, ω) = D^{-1}(c, ω) ⊙ (c_i · ∂_{c_i} D(c, ω))``

    - 二阶纯净算子（消除平方交叉项）：
      ``Ξ_{ii}(c, ω) = D^{-1} ⊙ (c_i² · ∂²_{c_i} D) − (L_i² − L_i)``

    - 交叉耦合算子：
      ``Ξ_{ij}(c, ω) = D^{-1} ⊙ (c_i · c_j · ∂²_{c_i c_j} D) − L_i · L_j``

    参数
    ----
    d_hat : ndarray, 形状 (N_samples, n_freq)
        频域观测数据（对波长维做 FFT 后的结果）。
    d_d_c : ndarray, 形状 (N_samples, n_controls, n_freq)
        对控制变量的一阶偏导（已乘以控制变量 c_i）。
    d2_d_c : ndarray, 形状 (N_samples, n_controls, n_controls, n_freq)
        对控制变量的二阶混合偏导。
    factors : ndarray, 形状 (N_samples, n_controls)
        控制变量取值矩阵。

    返回
    ----
    ops : dict[str, ndarray]
        算子字典，每个值的形状均为 (N_samples, n_freq)；键名如
        ``"L_1"``, ``"Xi_11"``, ``"Xi_12"`` 等。
    """
    n_samples, n_freq = d_hat.shape
    n_controls = factors.shape[1]

    # 逐点倒数：D^{-1}(c, ω)，用小量避免除零
    D_inv = 1.0 / (d_hat + 1e-12)  # shape (N_samples, n_freq)

    ops: Dict[str, np.ndarray] = {}
    # 保存一阶算子供二阶计算使用
    L: Dict[int, np.ndarray] = {}

    # ---------- 一阶欧拉算子 ----------
    for i in range(n_controls):
        ci = factors[:, i].reshape(-1, 1)          # (N, 1)
        L_i = D_inv * (ci * d_d_c[:, i, :])        # element-wise, (N, n_freq)
        L[i] = L_i
        ops[f"L_{i + 1}"] = L_i

    # ---------- 二阶纯净算子 ----------
    for i in range(n_controls):
        ci2 = (factors[:, i] ** 2).reshape(-1, 1)  # (N, 1)
        Xi_ii = D_inv * (ci2 * d2_d_c[:, i, i, :]) - (L[i] ** 2 - L[i])
        ops[f"Xi_{i + 1}{i + 1}"] = Xi_ii

    # ---------- 交叉耦合算子 ----------
    for i in range(n_controls):
        for j in range(i + 1, n_controls):
            ci = factors[:, i].reshape(-1, 1)
            cj = factors[:, j].reshape(-1, 1)
            Xi_ij = D_inv * (ci * cj * d2_d_c[:, i, j, :]) - L[i] * L[j]
            ops[f"Xi_{i + 1}{j + 1}"] = Xi_ij

    return ops


# ---------------------------------------------------------------------------
# 阶段 2：f/g 频域分离
# ---------------------------------------------------------------------------

def separate_fg(
    operator: np.ndarray,
    omega: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""将算子的 f 分量（强度）和 g 分量（相位）从频域中分离。

    每个算子在 (c, ω) 处的值可以写成：

    ``op(c, ω) = term_f(c) + (−iω) · term_g(c)``

    通过构造频率基矩阵 ``W = [1, −iω]``（形状 (n_freq, 2)），求解超定线性系统：

    ``op(c, :)^T ≈ W @ [term_f(c); term_g(c)]``

    用最小二乘伪逆求解：
    ``[term_f(c); term_g(c)] = W^† @ op(c, :)^T``

    参数
    ----
    operator : ndarray, 形状 (N_samples, n_freq)
        复值算子张量（如由 `build_euler_operators` 返回的某一项）。
    omega : ndarray, 形状 (n_freq,)
        归一化频率轴（例如 linspace(0, 1, n_freq)）。

    返回
    ----
    term_f : ndarray, 形状 (N_samples,)
        强度演化分量（对应 f(c)）。
    term_g : ndarray, 形状 (N_samples,)
        相位演化分量（对应 g(c)）。
    """
    omega = np.asarray(omega, dtype=float)
    n_freq = len(omega)

    # 频率基矩阵 W：(n_freq, 2)，列分别为 [1, -iω]
    W = np.stack([np.ones(n_freq), -1j * omega], axis=1)  # (n_freq, 2)

    # 计算伪逆 W^† = (W^H W)^{-1} W^H，形状 (2, n_freq)
    WH = W.conj().T                               # (2, n_freq)
    WtW = WH @ W                                  # (2, 2)
    WtW_inv = np.linalg.inv(WtW + 1e-14 * np.eye(2))
    W_pinv = WtW_inv @ WH                         # (2, n_freq)

    # 批量求解：fg[i, :] = W_pinv @ op[i, :]
    # operator: (N, n_freq)，W_pinv.T: (n_freq, 2)
    fg = operator @ W_pinv.T                      # (N_samples, 2)

    term_f = np.real(fg[:, 0])
    term_g = np.real(fg[:, 1])

    return term_f, term_g


def separate_all_operators_fg(
    ops: Dict[str, np.ndarray],
    omega: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """对算子字典中的每一项分别做 f/g 分离。

    参数
    ----
    ops : dict[str, ndarray]
        由 `build_euler_operators` 返回的算子字典。
    omega : ndarray, 形状 (n_freq,)
        归一化频率轴。

    返回
    ----
    f_ops : dict[str, ndarray]
        每项为算子的 f 分量，形状 (N_samples,)。
    g_ops : dict[str, ndarray]
        每项为算子的 g 分量，形状 (N_samples,)。
    """
    f_ops: Dict[str, np.ndarray] = {}
    g_ops: Dict[str, np.ndarray] = {}
    for name, op in ops.items():
        tf, tg = separate_fg(op, omega)
        f_ops[name + "_f"] = tf
        g_ops[name + "_g"] = tg
    return f_ops, g_ops


# ---------------------------------------------------------------------------
# 阶段 3：联合对角化（Joint Diagonalization）
# ---------------------------------------------------------------------------

def joint_diagonalize(
    operators: List[np.ndarray],
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""寻找联合对角化基 B，使算子集合在该基下同时（近似）对角化。

    通过对所有算子的联合协方差矩阵做特征分解，提取 k 个最强特征方向：

    ``C = Σ_m  A_m^H A_m``

    取 C 前 k 个最大特征值对应的特征向量作为基 B（形状 (n_freq, k)）。
    然后将各算子投影到 B 上，取平均得到组分演化率 Λ（形状 (N_samples, k)）。

    参数
    ----
    operators : list of ndarray, 每项形状 (N_samples, n_freq)
        算子集合（例如 L_1, L_2, Xi_11 等）。
    k : int
        提取的组分数量。

    返回
    ----
    B : ndarray, 形状 (n_freq, k)
        联合对角化基（列正交归一）。
    Lambda : ndarray, 形状 (N_samples, k)
        各组分的平均演化率（对角化后）。
    """
    if not operators:
        raise ValueError("operators 列表不能为空")

    n_freq = operators[0].shape[1]
    k = min(k, n_freq)

    # 构造联合协方差
    C = np.zeros((n_freq, n_freq), dtype=complex)
    for op in operators:
        C = C + op.conj().T @ op  # (n_freq, n_freq)

    # 特征分解（eigh 保证厄米矩阵的实特征值）
    eigenvalues, eigenvectors = np.linalg.eigh(C)  # 升序
    # 取最大的 k 个
    idx = np.argsort(eigenvalues)[::-1][:k]
    B = eigenvectors[:, idx]                        # (n_freq, k)

    # 每个算子在 B 下的投影，取平均作为组分演化率
    Lambda = np.zeros((operators[0].shape[0], k), dtype=complex)
    for op in operators:
        Lambda = Lambda + op @ B                    # (N_samples, k)
    Lambda = Lambda / len(operators)

    return B, Lambda


def joint_diag_residual(
    operators: List[np.ndarray],
    B: np.ndarray,
) -> float:
    r"""计算联合对角化残差（off-diagonal 范数之和）。

    ``res = Σ_m  ||off(B^{-1} A_m B)||_F²``

    用于验证对角化质量；越接近 0 表示对角化越好。

    参数
    ----
    operators : list of ndarray, 每项形状 (N_samples, n_freq)
    B : ndarray, 形状 (n_freq, k)

    返回
    ----
    residual : float
    """
    n_freq, k = B.shape
    # 用伪逆计算 B^{-1}
    B_inv = np.linalg.pinv(B)  # (k, n_freq)

    total = 0.0
    for op in operators:
        # 对每个样本计算 B^{-1} A B，取非对角元素
        # op: (N, n_freq)，先取每行与 B 的投影
        proj = op @ B        # (N, k)
        # 这里的 "off-diagonal" 近似为 proj 的方差（对角化好时方差小）
        # 真正的 off-diagonal 需要 k×k 矩阵，用均值作为对角元估计
        mean_k = np.mean(np.abs(proj) ** 2, axis=0)   # (k,)
        var_k = np.var(np.abs(proj) ** 2, axis=0)     # (k,)
        total += float(np.sum(var_k))

    return total


# ---------------------------------------------------------------------------
# 阶段 4：物理候选库（SINDy-PI 的 Θ 矩阵）
# ---------------------------------------------------------------------------

def build_physical_candidate_library(
    factors: np.ndarray,
    max_degree: int = 2,
    include_log: bool = False,
    include_inv: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    r"""为 SINDy-PI 构建多项式候选函数库 Θ。

    对控制变量 c 构建最高阶数为 ``max_degree`` 的多项式展开：
    ``Θ = [1, c_1, ..., c_d, c_1², c_1c_2, ...]``，形状 (N_samples, J)。

    可选地加入对数项 ``ln(c_i)`` 和倒数项 ``1/(1+c_i)``。

    参数
    ----
    factors : ndarray, 形状 (N_samples, n_controls)
        控制变量矩阵。
    max_degree : int
        多项式最高阶数，默认 2。
    include_log : bool
        是否加入对数项 ln(|c_i| + ε)，默认 False。
    include_inv : bool
        是否加入饱和倒数项 1/(1 + |c_i|)，默认 False。

    返回
    ----
    Theta : ndarray, 形状 (N_samples, J)
        候选函数矩阵（未归一化）。
    names : list[str]
        各列对应的函数名称，长度 J。
    """
    N, d = factors.shape
    terms: List[np.ndarray] = [np.ones(N)]
    names: List[str] = ["1"]

    # 一阶项
    for i in range(d):
        terms.append(factors[:, i].copy())
        names.append(f"c{i + 1}")

    # 二阶及以上
    if max_degree >= 2:
        # 自身平方
        for i in range(d):
            terms.append(factors[:, i] ** 2)
            names.append(f"c{i + 1}^2")
        # 交叉项
        for i in range(d):
            for j in range(i + 1, d):
                terms.append(factors[:, i] * factors[:, j])
                names.append(f"c{i + 1}*c{j + 1}")

    if max_degree >= 3:
        for i in range(d):
            terms.append(factors[:, i] ** 3)
            names.append(f"c{i + 1}^3")
        for i in range(d):
            for j in range(d):
                if j != i:  # 跳过 i==j，避免与 c_i^3 项重复
                    terms.append(factors[:, i] ** 2 * factors[:, j])
                    names.append(f"c{i + 1}^2*c{j + 1}")

    # 可选非线性项
    if include_log:
        for i in range(d):
            terms.append(np.log(np.abs(factors[:, i]) + 1e-12))
            names.append(f"ln(c{i + 1})")

    if include_inv:
        for i in range(d):
            terms.append(1.0 / (1.0 + np.abs(factors[:, i])))
            names.append(f"1/(1+c{i + 1})")

    Theta = np.column_stack(terms)  # (N, J)
    return Theta, names


# ---------------------------------------------------------------------------
# 阶段 5：SINDy-PI 零空间稀疏回归
# ---------------------------------------------------------------------------

def sindy_pi_nullspace(
    Theta: np.ndarray,
    Lambda: np.ndarray,
    sparsity_threshold: float = 0.05,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    r"""用零空间稀疏回归提取物理方程系数矩阵 Ξ。

    对每个组分 k，在 f 方向（Θ 与 Λ[:,k] 的实部）上求解：

    ``Θ Ξ_k ≈ Λ_{f,k}(c)``

    通过 SVD 取最小奇异值对应的方向（近似零空间），再做阈值截断稀疏化。

    参数
    ----
    Theta : ndarray, 形状 (N_total, J)
        候选函数矩阵（来自 `build_physical_candidate_library`）。
    Lambda : ndarray, 形状 (N_total, K)
        各组分演化率（来自 `joint_diagonalize`）。
    sparsity_threshold : float
        稀疏截断阈值（相对于最大系数的比例）。

    返回
    ----
    Xi : ndarray, 形状 (J, K)
        稀疏系数矩阵，每列对应一个组分的物理方程参数。
    residuals : list of float
        每个组分的拟合残差 ||Θ Ξ_k - Λ_{f,k}||²。
    """
    N, J = Theta.shape
    K = Lambda.shape[1]
    Xi = np.zeros((J, K))
    residuals: List[np.ndarray] = []

    for k in range(K):
        lam_k = np.real(Lambda[:, k])  # f 分量

        # 增广矩阵 [Θ | -lam_k]，求近似零空间
        A_aug = np.column_stack([Theta, -lam_k])  # (N, J+1)
        _, _, Vt = np.linalg.svd(A_aug, full_matrices=False)
        v = Vt[-1, :]                              # 最小奇异值对应方向 (J+1,)

        # 最后一个元素对应 -lam_k 的系数（归一化使其为 1）
        if abs(v[-1]) > 1e-12:
            xi_k = v[:-1] / (-v[-1])
        else:
            xi_k = v[:-1]

        # 阈值截断稀疏化
        max_abs = np.max(np.abs(xi_k))
        if max_abs > 0:
            xi_k[np.abs(xi_k) < sparsity_threshold * max_abs] = 0.0

        Xi[:, k] = xi_k
        res = float(np.sum((Theta @ xi_k - lam_k) ** 2))
        residuals.append(res)

    return Xi, residuals


# ---------------------------------------------------------------------------
# 完整管线入口
# ---------------------------------------------------------------------------

def run_joint_diag_pipeline(
    d_hat: np.ndarray,
    d_d_c: np.ndarray,
    d2_d_c: np.ndarray,
    omega: np.ndarray,
    factors: np.ndarray,
    k: int = 3,
    sparsity_threshold: float = 0.05,
) -> Dict[str, object]:
    r"""执行完整的"谱-物理联合解析流"算法（1.md 完整流程）。

    流程
    ----
    1. 构造 Euler 算子集合（element-wise，无 SVD）
    2. 对所有算子做 f/g 频域分离
    3. 用算子集合做联合对角化，提取 B(ω) 和 Λ(c)
    4. 构建物理候选库 Θ
    5. SINDy-PI 零空间稀疏回归，输出系数矩阵 Ξ

    参数
    ----
    d_hat : ndarray, 形状 (N_samples, n_freq)
    d_d_c : ndarray, 形状 (N_samples, n_controls, n_freq)
    d2_d_c : ndarray, 形状 (N_samples, n_controls, n_controls, n_freq)
    omega : ndarray, 形状 (n_freq,)
        归一化频率轴（例如 np.linspace(0, 1, n_freq)）。
    factors : ndarray, 形状 (N_samples, n_controls)
    k : int
        提取组分数量。
    sparsity_threshold : float
        稀疏截断阈值。

    返回
    ----
    result : dict，含以下键：
        - ``"operators"``: 算子字典
        - ``"B"``: 联合对角化基 (n_freq, k)
        - ``"Lambda"``: 组分演化率 (N_samples, k)
        - ``"f_ops"``, ``"g_ops"``: 算子的 f/g 分量
        - ``"Theta"``: 候选函数矩阵 (N_samples, J)
        - ``"theta_names"``: 候选函数名称列表
        - ``"Xi"``: 稀疏系数矩阵 (J, k)
        - ``"residuals"``: 各组分拟合残差
        - ``"jd_residual"``: 联合对角化残差（标量）
    """
    # 阶段 1：构造 Euler 算子
    ops = build_euler_operators(d_hat, d_d_c, d2_d_c, factors)

    # 阶段 2：f/g 频域分离
    f_ops, g_ops = separate_all_operators_fg(ops, omega)

    # 阶段 3：联合对角化
    op_list = list(ops.values())
    B, Lambda = joint_diagonalize(op_list, k=k)
    jd_res = joint_diag_residual(op_list, B)

    # 阶段 4：物理候选库
    Theta, theta_names = build_physical_candidate_library(factors)

    # 阶段 5：SINDy-PI 零空间求解
    Xi, residuals = sindy_pi_nullspace(Theta, Lambda, sparsity_threshold=sparsity_threshold)

    return {
        "operators": ops,
        "B": B,
        "Lambda": Lambda,
        "f_ops": f_ops,
        "g_ops": g_ops,
        "Theta": Theta,
        "theta_names": theta_names,
        "Xi": Xi,
        "residuals": residuals,
        "jd_residual": jd_res,
    }
