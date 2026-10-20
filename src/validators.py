"""Validators from the Week 2 Data Conventions ("Validation").

Each check raises ValueError naming the table and the column.
Run after every fetch and in notebooks/00_integration.ipynb.
"""
from __future__ import annotations

import pandas as pd

import config


def coerce(df: pd.DataFrame, schema: dict, name: str = "table") -> pd.DataFrame:
    """Cast to the schema's exact dtypes, in schema order. Values outside a
    closed vocabulary raise (they never turn into NaN)."""
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    extra = [c for c in df.columns if c not in schema]
    if extra:
        raise ValueError(f"{name}: unexpected columns {extra}")
    out = {}
    for col, (dtype, _nullable) in schema.items():
        s = df[col]
        if isinstance(dtype, pd.CategoricalDtype):
            bad = s[s.notna() & ~s.isin(dtype.categories)].unique()
            if len(bad):
                raise ValueError(f"{name}.{col}: values outside the vocabulary: {list(bad)[:10]}")
            out[col] = pd.Series(pd.Categorical(s, dtype=dtype), index=df.index)
        else:
            out[col] = s.astype(dtype)
    return pd.DataFrame(out, index=df.index)


def _check_schema(df: pd.DataFrame, schema: dict, name: str) -> None:
    expected = list(schema)
    if list(df.columns) != expected:
        miss = [c for c in expected if c not in df.columns]
        extra = [c for c in df.columns if c not in expected]
        raise ValueError(f"{name}: columns differ (missing {miss}, unexpected {extra}, "
                         f"or wrong order)")
    for col, (dtype, nullable) in schema.items():
        if df[col].dtype != dtype:
            raise ValueError(f"{name}.{col}: dtype is {df[col].dtype}, config.py says {dtype}")
        if not nullable and df[col].isna().any():
            raise ValueError(f"{name}.{col}: {int(df[col].isna().sum())} missing values in a "
                             f"non-nullable column")


def validate_neurons(neurons: pd.DataFrame) -> None:
    _check_schema(neurons, config.NEURON_SCHEMA, "neurons")
    if neurons.duplicated(["neuron_id", "dataset"]).any():
        raise ValueError("neurons.neuron_id: repeated key (neuron_id, dataset)")


def validate_edges(edges: pd.DataFrame, neurons: pd.DataFrame, roi: bool = False) -> None:
    schema = config.EDGEROI_SCHEMA if roi else config.EDGE_SCHEMA
    name = "edgeroi" if roi else "edges"
    _check_schema(edges, schema, name)
    key = ["pre_id", "post_id", "dataset"] + (["neuropil"] if roi else [])
    if edges.duplicated(key).any():
        raise ValueError(f"{name}.{key}: repeated key")
    ids = pd.Index(neurons["neuron_id"].unique())
    for col in ("pre_id", "post_id"):
        missing = ~edges[col].isin(ids)
        if missing.any():
            raise ValueError(f"{name}.{col}: {int(missing.sum())} ids not in the neurons table "
                             f"(e.g. {edges.loc[missing, col].iloc[0]})")
    if (edges["syn_count"] < edges["min_syn_fetched"]).any():
        raise ValueError(f"{name}.syn_count: values below min_syn_fetched")
    if not roi and "n_post" in neurons and neurons["n_post"].notna().any():
        inn = edges.groupby("post_id")["syn_count"].sum()
        npost = neurons.set_index("neuron_id")["n_post"].dropna()
        both = inn.index.intersection(npost.index)
        bad = inn.loc[both] > npost.loc[both].astype("int64")
        if bad.any():
            raise ValueError(f"{name}.syn_count: {int(bad.sum())} neurons receive more input "
                             f"synapses than n_post")


def assert_interchangeable(a: pd.DataFrame, b: pd.DataFrame) -> None:
    assert list(a.columns) == list(b.columns), "column sets differ"
    assert a.dtypes.equals(b.dtypes), "dtypes differ"
