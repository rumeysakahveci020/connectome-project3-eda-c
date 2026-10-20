"""Read CSV / gzip CSV / feather / parquet tables, loading only the columns that are needed."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _suffix(path) -> str:
    p = str(path).lower()
    for ext in (".feather", ".parquet", ".csv.gz", ".csv", ".tsv"):
        if p.endswith(ext):
            return ext
    return ""


def column_names(path) -> list[str]:
    s = _suffix(path)
    if s == ".feather":
        import pyarrow as pa
        with pa.memory_map(str(path)) as src:
            return list(pa.ipc.open_file(src).schema.names)
    if s == ".parquet":
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(str(path)).schema_arrow.names)
    return list(pd.read_csv(path, nrows=0, sep="\t" if s == ".tsv" else ",").columns)


def read_table(path, columns: list[str] | None = None, str_cols: list[str] | None = None) -> pd.DataFrame:
    """`str_cols`: CSV columns to keep as text (long ids)."""
    path = str(path)
    s = _suffix(path)
    if s == ".feather":
        return pd.read_feather(path, columns=columns)
    if s == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns, dtype={c: str for c in (str_cols or [])},
                       sep="\t" if s == ".tsv" else ",", low_memory=False)


def read_feather_filtered(path, columns: list[str], id_columns: tuple[str, str], keep_ids) -> pd.DataFrame:
    """Stream a large feather file batch by batch and keep only the rows whose two id columns are
    both in `keep_ids`. Needed for the male CNS edge file (151.9 million rows): loading it whole
    runs out of memory on a laptop."""
    import pyarrow as pa
    import pyarrow.compute as pc

    parts = []
    with pa.memory_map(str(path)) as src:
        rdr = pa.ipc.open_file(src)
        ids_cache: dict = {}
        for i in range(rdr.num_record_batches):
            b = rdr.get_batch(i)
            mask = None
            for c in id_columns:
                col = b.column(c)
                if col.type not in ids_cache:
                    ids_cache[col.type] = pa.array(list(keep_ids), type=col.type)
                m = pc.is_in(col, value_set=ids_cache[col.type])
                mask = m if mask is None else pc.and_(mask, m)
            f = b.filter(mask).select(columns)
            if f.num_rows:
                parts.append(f.to_pandas())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
