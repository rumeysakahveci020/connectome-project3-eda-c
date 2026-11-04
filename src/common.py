"""Small helpers shared by both fetch modules."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("fetch")


def recode(series: pd.Series, mapping: dict, column: str) -> pd.Series:
    """Data Conventions: an unmapped source value raises; missing becomes unknown."""
    out = series.map(mapping)
    bad = series[out.isna() & series.notna()]
    if len(bad):
        raise ValueError(f"{column}: unmapped values {bad.value_counts().to_dict()}")
    return out.fillna("unknown")


def first_col(columns, candidates, what, required=True):
    """First candidate present in `columns` (case-insensitive). Prints nothing;
    raises naming every column found, so a wrong guess is easy to fix."""
    lower = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise KeyError(f"no {what} column. Tried {candidates}. Columns found: {sorted(map(str, columns))}")
    return None


def to_int32_na(n: int) -> pd.Series:
    return pd.Series(pd.array([pd.NA] * n, dtype="Int32"))
