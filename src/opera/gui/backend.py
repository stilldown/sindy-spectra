from __future__ import annotations

from typing import Dict, Any

from .data import load_file, load_demo
from .config import make_config
from .threads import AnalysisThread


from opera.discovery import DiscoveryConfig

# 该文件现在由 ``DiscoveryService`` 封装上一层，
# 真正的实现分散在子模块中以便单元测试和重用。




class DiscoveryService:
    """封装数据加载、配置和分析逻辑的“后端”。

    :ivar cur_data: 最近加载的数据集，格式同 GUI 中的 ``self.cur_data``。
    """

    def __init__(self) -> None:
        self.cur_data: Dict[str, Any] | None = None

    # ---------- 数据处理 ----------

    def load_file(self, filepath: str) -> Dict[str, Any]:
        """委托 ``data.load_file`` 并记录当前数据。"""
        data = load_file(filepath)
        self.cur_data = data
        return data

    def load_demo(self) -> Dict[str, Any]:
        """委托 ``data.load_demo`` 并记录当前数据。"""
        data = load_demo()
        self.cur_data = data
        return data

    # ---------- 配置和分析 ----------
    def make_config(
        self,
        k_mode: str,
        k_value: int,
        k_max: int,
        rank_energy_threshold: float,
        sparsity_threshold: float,
        zero_anchor_tol: float,
        epsilon: float,
    ) -> DiscoveryConfig:
        """生成 ``DiscoveryConfig``，委托给 ``config.make_config``。"""
        return make_config(
            k_mode,
            k_value,
            k_max,
            rank_energy_threshold,
            sparsity_threshold,
            zero_anchor_tol,
            epsilon,
        )

    def create_thread(self, cfg: DiscoveryConfig) -> AnalysisThread:
        """根据当前数据生成分析线程。"""
        if self.cur_data is None:
            raise RuntimeError("没有可用数据，请先加载样本。")
        return AnalysisThread(self.cur_data, cfg)
