"""OPERA Discovery 算子库工具包

模块按功能划分：

* ``observable`` – 构建严格物理约束的观测算子库（12 类结构）。
* ``kron`` – Kronecker 展开辅助函数，用于张量化特征。
* ``lie_basis`` – 频域 Lie 基函数生成。

公共 API 在此处统一导出。
"""

from .observable import build_observable_library
from .kron import build_psi
from .lie_basis import build_lie_basis

__all__ = [
    "build_observable_library",
    "build_psi",
    "build_lie_basis",
]
