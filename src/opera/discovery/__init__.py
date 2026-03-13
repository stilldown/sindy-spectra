"""OPERA Discovery 新核心入口。"""

from .types import DiscoveryConfig, DiscoveryResult
from .pipeline import run_discovery

__all__ = [
    "DiscoveryConfig",
    "DiscoveryResult",
    "run_discovery",
]
