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


def region_from_counts(counts: pd.DataFrame) -> tuple[pd.Series, int]:
    """counts: columns neuron_id, region, syn. Region = the one with the most
    synapses; ties go to the alphabetically first name (the tie rule is a week 3
    decision, so the number of ties is returned and logged).
    Returns (Series indexed by neuron_id, number of neurons with a tie)."""
    c = counts.sort_values(["neuron_id", "syn", "region"], ascending=[True, False, True])
    top2 = c.groupby("neuron_id").head(2)
    first = top2.groupby("neuron_id").nth(0).set_index("neuron_id")
    second = top2.groupby("neuron_id").nth(1).set_index("neuron_id")
    ties = int((first["syn"].reindex(second.index) == second["syn"]).sum())
    return first["region"], ties


def to_int32_na(n: int) -> pd.Series:
    return pd.Series(pd.array([pd.NA] * n, dtype="Int32"))
