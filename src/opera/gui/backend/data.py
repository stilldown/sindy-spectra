from __future__ import annotations

import os
import re
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd


def extract_controls_and_spectra(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """自动识别 DataFrame 中的波长列和控制变量列。

    返回 ``(wavelengths, spectra, factors, factor_names)``。
    """
    cols = list(df.columns)
    spectral_cols: List[str] = []
    for col in cols:
        try:
            float(col)
            spectral_cols.append(col)
        except Exception:
            continue

    if len(spectral_cols) == 0:
        raise ValueError("未识别到光谱列（列名需可转换为浮点数，如波长）")

    non_spectral = [c for c in cols if c not in spectral_cols]
    control_cols = [c for c in non_spectral if c.lower() != 'sample']
    if len(control_cols) == 0:
        raise ValueError("未识别到控制变量列（如 c1,c2,...）")

    def _control_key(name: str):
        m = re.match(r'^[cC](\d+)$', str(name))
        if m:
            return (0, int(m.group(1)))
        return (1, str(name))

    control_cols = sorted(control_cols, key=_control_key)

    wavelengths = np.array([float(w) for w in spectral_cols], dtype=float)

    grouped = (
        df.groupby(control_cols, as_index=False)[spectral_cols]
        .mean()
    )
    factors = grouped[control_cols].to_numpy(dtype=float)
    spectra = grouped[spectral_cols].to_numpy(dtype=float)
    return wavelengths, spectra, factors, control_cols


def load_file(filepath: str) -> Dict[str, Any]:
    """从 CSV/XLSX 文件加载并解析数据。

    返回包含 ``wavelengths``、``spectra``、``factors``、``factor_names`` 的字典。
    """
    if filepath.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    wavelengths, spectra, c_factors, factor_names = extract_controls_and_spectra(df)
    return {
        'wavelengths': wavelengths,
        'spectra': spectra,
        'factors': c_factors,
        'factor_names': factor_names,
    }


def load_demo() -> Dict[str, Any]:
    """加载预置的示例数据集。

    数据文件位于仓库根目录的 ``2_corrected_final.csv``。
    """
    filepath = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', '2_corrected_final.csv'
    )
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到演示数据文件: {filepath}")
    df = pd.read_csv(filepath)
    wavelengths, spectra, c_factors, factor_names = extract_controls_and_spectra(df)
    return {
        'wavelengths': wavelengths,
        'spectra': spectra,
        'factors': c_factors,
        'factor_names': factor_names,
    }
