from __future__ import annotations

from ..discovery import DiscoveryConfig
from ..discovery.types import MatrixConfig


def make_config(
    k_mode: str,
    k_value: int,
    k_max: int,
    rank_energy_threshold: float,
    sparsity_threshold: float,
    zero_anchor_tol: float,
    epsilon: float,
) -> DiscoveryConfig:
    """根据界面参数构造一个 ``DiscoveryConfig`` 对象。

    只是把 GUI 的各个控件值转换为配置字段。
    """
    mat_cfg = {'epsilon': float(epsilon)}
    return DiscoveryConfig(
        k_mode=k_mode,
        k_value=k_value,
        k_max=k_max,
        rank_energy_threshold=rank_energy_threshold,
        sparsity_threshold=sparsity_threshold,
        zero_anchor_tol=zero_anchor_tol,
        matrix=MatrixConfig(**mat_cfg),
    )
