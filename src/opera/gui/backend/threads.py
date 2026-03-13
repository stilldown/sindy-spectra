from __future__ import annotations

from typing import Dict

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..discovery import run_discovery, DiscoveryConfig


class AnalysisThread(QThread):
    """后台执行 discovery 的线程，实现与 GUI 解耦。"""
    finished = Signal(object)  # DiscoveryResult
    error = Signal(str)

    def __init__(self, data_dict: Dict[str, np.ndarray], cfg: DiscoveryConfig):
        super().__init__()
        self.data = data_dict
        self.cfg = cfg

    def run(self):
        try:
            result = run_discovery(
                self.data['spectra'],
                self.data['factors'],
                self.data['wavelengths'],
                config=self.cfg,
            )
            self.finished.emit(result)
        except Exception as e:
            import traceback

            traceback.print_exc()
            self.error.emit(str(e))
