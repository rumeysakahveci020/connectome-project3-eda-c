"""FlyWire FAFB, materialisation 783 (Data Conventions D6).

Neurons : flywire_annotations v3.2.0
          (Supplemental_file1_neuron_annotations.tsv, tag v3.2.0 on GitHub)
Edges   : the Codex per-neuropil table built from BUHMANN synapses, unthresholded
          (one row per pre, post, neuropil). The Princeton table is only a
          sensitivity table and must never be cached under these names.

Everything is cached at 1 synapse; thresholds are applied in memory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.cache import cached
from src.common import first_col, log, recode, region_from_counts, to_int32_na
from src.tables import column_names, read_table
from src.validators import coerce, validate_edges, validate_neurons

ANNOTATIONS_URL = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
                   f"{config.FAFB_ANNOTATIONS_TAG}/supplemental_files/"
                   "Supplemental_file1_neuron_annotations.tsv")

SIDE_MAP = {"left": "left", "right": "right", "center": "center", "na": "unknown"}
SUPER_CLASS_MAP = {v: v for v in config.SUPER_CLASS_VALUES if v != "unknown"}
NT_MAP = {v: v for v in config.NT_VALUES if v != "unknown"}


def read_annotations(path_or_url: str = ANNOTATIONS_URL) -> pd.DataFrame:
    raw = pd.read_csv(path_or_url, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    print("FAFB annotation columns:", sorted(raw.columns))
    for col in ["side", "super_class", "top_nt"]:
        print(col, raw[col].value_counts(dropna=False).head(15).to_dict())
    return raw


def read_edgeroi(path: str) -> pd.DataFrame:
    """Raw per-neuropil table (Zenodo 10676866, proofread_connections_783.feather, or a Codex CSV)
    -> edgeroi table (schema of config.EDGEROI_SCHEMA).
    Self-connections are dropped and their number logged (FAFB expects 0)."""
    head = column_names(path)
    pre = first_col(head, ["pre_pt_root_id", "pre_root_id", "pre_id"], "presynaptic id")
    post = first_col(head, ["post_pt_root_id", "post_root_id", "post_id"], "postsynaptic id")
    npil = first_col(head, ["neuropil", "roi"], "neuropil")
    w = first_col(head, ["syn_count", "synapse_count", "weight"], "synapse count")
    print("FAFB edge columns:", sorted(head))
    raw = read_table(path, columns=[pre, post, npil, w], str_cols=[pre, post])
    out = pd.DataFrame({
        "pre_id": raw[pre].astype("int64"),
        "post_id": raw[post].astype("int64"),
        "neuropil": raw[npil].astype("string"),
        "syn_count": pd.to_numeric(raw[w]).astype("int64"),
    })
    selfc = int((out["pre_id"] == out["post_id"]).sum())
    log.warning("FAFB self-connections dropped: %d (expected 0)", selfc)
    out = out[out["pre_id"] != out["post_id"]]
    out = out.groupby(["pre_id", "post_id", "neuropil"], as_index=False)["syn_count"].sum()
    out["syn_count"] = out["syn_count"].astype("int32")
    out["dataset"] = "fafb"
    out["release"] = str(config.FAFB_MATERIALIZATION)
    out["min_syn_fetched"] = np.int32(config.FETCH_MIN_SYN)
    out.attrs["self_connections"] = selfc
    return out


def edges_from_edgeroi(edgeroi: pd.DataFrame) -> pd.DataFrame:
    """Sum over neuropils FIRST (D8); thresholds come later, in memory."""
    e = edgeroi.groupby(["pre_id", "post_id"], as_index=False)["syn_count"].sum()
    e["syn_count"] = e["syn_count"].astype("int32")
    e["dataset"] = "fafb"
    e["release"] = str(config.FAFB_MATERIALIZATION)
    e["min_syn_fetched"] = np.int32(config.FETCH_MIN_SYN)
    return e


def region_table(edgeroi: pd.DataFrame) -> tuple[pd.Series, int]:
    """Neuropil with the most synapses (as pre and post), from all synapses."""
    out = edgeroi.groupby(["pre_id", "neuropil"])["syn_count"].sum()
    inn = edgeroi.groupby(["post_id", "neuropil"])["syn_count"].sum()
    out.index.names = inn.index.names = ["neuron_id", "region"]
    tot = out.add(inn, fill_value=0).rename("syn").reset_index()
    tot["region"] = tot["region"].astype(str)
    return region_from_counts(tot)


def build_neurons(ann: pd.DataFrame, edgeroi: pd.DataFrame | None = None,
                  super_class_extra: dict | None = None, proofread_ids: set | None = None) -> pd.DataFrame:
    """`super_class_extra` is for values the group has not yet agreed on. v3.2.0 has a tenth
    level, sensory_ascending (612 neurons), that config.SUPER_CLASS_VALUES lacks. Without an
    explicit mapping the build stops (Data Conventions: unmapped values raise). A mapping given
    here is PROVISIONAL: report it to the group and rebuild once they decide."""
    n = len(ann)
    sc_map = {**SUPER_CLASS_MAP, **(super_class_extra or {})}
    if super_class_extra:
        log.warning("PROVISIONAL super_class mapping %s: not agreed by the group yet", super_class_extra)
    neu = pd.DataFrame({
        "neuron_id": ann["root_id"].astype("int64"),
        "dataset": "fafb",
        "release": str(config.FAFB_MATERIALIZATION),
        "cell_type": ann["cell_type"],
        "cell_class": ann["cell_class"],
        "cell_sub_class": ann["cell_sub_class"],
        "super_class": recode(ann["super_class"], sc_map, "super_class"),
        "side": recode(ann["side"], SIDE_MAP, "side"),
        "region": "unknown",
        "nt_type": recode(ann["top_nt"], NT_MAP, "nt_type"),
        "nt_confidence": pd.to_numeric(ann["top_nt_conf"]).astype("float32"),
        "n_pre": to_int32_na(n), "n_post": to_int32_na(n),
        "flywire_type": pd.Series([pd.NA] * n, dtype="string"),
        "manc_type": pd.Series([pd.NA] * n, dtype="string"),
        "hemibrain_type": ann["hemibrain_type"],
        "dimorphism": ann["dimorphism"],
        "status": ann["status"],
        "hemilineage": ann["ito_lee_hemilineage"],
        "nt_known": ann["known_nt"],
    })
    neu.loc[neu["nt_type"] == "unknown", "nt_confidence"] = np.float32("nan")
    if proofread_ids is not None:
        # The release's neurons are the proofread root ids (139,255). The annotation file has 7 ids
        # that are not in that list; they have no edges. Drop them (reported to the group).
        stale = neu.loc[~neu["neuron_id"].isin(proofread_ids), "neuron_id"]
        log.warning("FAFB annotation ids not in the proofread list, dropped: %d %s", len(stale), sorted(stale.tolist())[:10])
        neu = neu[neu["neuron_id"].isin(proofread_ids)]
    if edgeroi is not None:
        # The published count is 139,255; the annotation file has 139,248. Ids that appear
        # in edges but not in the annotations are added with unknown annotations.
        missing = pd.Index(pd.unique(pd.concat([edgeroi["pre_id"], edgeroi["post_id"]]))
                           ).difference(pd.Index(neu["neuron_id"]))
        if len(missing):
            log.warning("FAFB ids in edges but not in annotations: %d (added as unknown)", len(missing))
            add = pd.DataFrame({"neuron_id": missing.astype("int64"), "dataset": "fafb",
                                "release": str(config.FAFB_MATERIALIZATION),
                                "super_class": "unknown", "side": "unknown",
                                "region": "unknown", "nt_type": "unknown"})
            neu = pd.concat([neu, add], ignore_index=True)
        reg, ties = region_table(edgeroi)
        log.warning("FAFB neurons whose top two neuropils tie: %d (alphabetical rule, week 3 decision)", ties)
        neu["region"] = neu["neuron_id"].map(reg).fillna("unknown").astype(str)
    neu = neu.astype({c: "string" for c in ["release", "cell_type", "cell_class", "cell_sub_class",
                                            "region", "hemibrain_type", "dimorphism", "status",
                                            "hemilineage", "nt_known", "flywire_type", "manc_type"]})
    neu["n_pre"] = pd.array([pd.NA] * len(neu), dtype="Int32")
    neu["n_post"] = pd.array([pd.NA] * len(neu), dtype="Int32")
    neu["flywire_type"] = pd.array([pd.NA] * len(neu), dtype="string")
    neu["manc_type"] = pd.array([pd.NA] * len(neu), dtype="string")
    neu["nt_confidence"] = neu["nt_confidence"].astype("float32")
    neu = neu[list(config.NEURON_SCHEMA)]
    return coerce(neu, config.NEURON_SCHEMA, "fafb neurons")


def fetch_fafb(edges_path: str, annotations_path: str = ANNOTATIONS_URL, force_refresh: bool = False,
               super_class_extra: dict | None = None, proofread_ids_path: str | None = None):
    """Build (or read) the three FAFB cache files and return (neurons, edges, edgeroi)."""
    rel = str(config.FAFB_MATERIALIZATION)
    p_n = config.cache_path("fafb", "neurons", rel, f"ann{config.FAFB_ANNOTATIONS_TAG[1:]}")
    p_e = config.cache_path("fafb", "edges", rel, config.FAFB_SYNAPSE_TABLE, min_syn=1)
    p_r = config.cache_path("fafb", "edgeroi", rel, config.FAFB_SYNAPSE_TABLE, min_syn=1)
    if all(p.exists() for p in (p_n, p_e, p_r)) and not force_refresh:
        return pd.read_parquet(p_n), pd.read_parquet(p_e), pd.read_parquet(p_r)

    proofread = set(np.load(proofread_ids_path).astype("int64").tolist()) if proofread_ids_path else None
    edgeroi = read_edgeroi(edges_path)
    selfc = edgeroi.attrs.get("self_connections", 0)
    neurons = cached(p_n, lambda: build_neurons(read_annotations(annotations_path), edgeroi, super_class_extra, proofread),
                     validate_neurons, force_refresh)
    edgeroi = coerce(edgeroi[list(config.EDGEROI_SCHEMA)], config.EDGEROI_SCHEMA, "fafb edgeroi")
    edges = coerce(edges_from_edgeroi(edgeroi)[list(config.EDGE_SCHEMA)], config.EDGE_SCHEMA, "fafb edges")
    cached(p_e, lambda: edges, lambda d: validate_edges(d, neurons), force_refresh)
    cached(p_r, lambda: edgeroi, lambda d: validate_edges(d, neurons, roi=True), force_refresh)
    log.info("FAFB self-connections dropped: %d", selfc)
    return neurons, edges, edgeroi
