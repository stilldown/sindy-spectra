"""OPERA Discovery 新核心入口。"""

from .types import DiscoveryConfig, DiscoveryResult
from .pipeline import run_discovery
from .tensor_utils import (
    flat_to_tensor,
    tensor_to_flat,
    compute_tensor_fourier,
    compute_tensor_control_derivatives,
    build_tensor_euler_library,
    run_tensor_discovery,
)

__all__ = [
    "DiscoveryConfig",
    "DiscoveryResult",
    "run_discovery",
    # 张量路径
    "flat_to_tensor",
    "tensor_to_flat",
    "compute_tensor_fourier",
    "compute_tensor_control_derivatives",
    "build_tensor_euler_library",
    "run_tensor_discovery",
]
