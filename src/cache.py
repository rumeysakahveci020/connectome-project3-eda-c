"""One helper for every fetch (Data Conventions, "Caching").

Read the file if it exists, otherwise fetch, validate, write.
Only force_refresh=True refetches.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd


def cached(path: Path, fetch_fn: Callable[[], pd.DataFrame],
           validate_fn: Callable[[pd.DataFrame], None] | None = None,
           force_refresh: bool = False) -> pd.DataFrame:
    path = Path(path)
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)
    df = fetch_fn()
    if validate_fn is not None:
        validate_fn(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df
