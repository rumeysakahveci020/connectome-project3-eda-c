"""FlyWire FAFB, materialisation 783 (Data Conventions D6).

Neurons : flywire_annotations v3.2.0
          (Supplemental_file1_neuron_annotations.tsv, tag v3.2.0 on GitHub), aligned to the 783 proofread list
          (proofread_root_ids_783.npy); region from FlyWire's per-neuron neuropil counts (Zenodo 10676866).
          Week 4 corrections 1, 2, 3 and 6 are applied by eda_shared v0.2.0, not by code in this file.
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
from eda_shared.corrections import (add_nt_type_corrected, align_to_proofread, fafb_neuropil_counts,
                                    map_super_class, region_from_counts)
from src.common import first_col, log, recode
from src.tables import column_names, read_table
from src.validators import coerce, validate_edges, validate_neurons

ANNOTATIONS_URL = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
                   f"{config.FAFB_ANNOTATIONS_TAG}/supplemental_files/"
                   "Supplemental_file1_neuron_annotations.tsv")

SIDE_MAP = {"left": "left", "right": "right", "center": "center", "na": "unknown"}
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


def read_neuropil_counts(pre_path: str, post_path: str) -> pd.DataFrame:
    """FlyWire per_neuron_neuropil_count_pre_783 / _post_783 (Zenodo 10676866) -> neuron_id, neuropil, count."""
    pre, post = read_table(pre_path), read_table(post_path)
    print("FAFB neuropil count columns (pre):", sorted(map(str, pre.columns)), " (post):", sorted(map(str, post.columns)))
    return fafb_neuropil_counts(pre, post)


def build_neurons(ann: pd.DataFrame, proofread_ids, counts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The corrected FAFB neuron table and the Step 0 report.

    Order (eda_shared README): labels and super_class from the annotation rows, then align to the proofread
    list (correction 3), then region from the counts (6), then nt_type_corrected (1)."""
    n = len(ann)
    sc = map_super_class(ann["super_class"], "fafb")                       # correction 2
    neu = pd.DataFrame({
        "neuron_id": ann["root_id"].astype("int64").to_numpy(),
        "dataset": pd.Categorical(["fafb"] * n, dtype=config.NEURON_SCHEMA["dataset"][0]),
        "release": pd.array([str(config.FAFB_MATERIALIZATION)] * n, dtype="string"),
        "cell_type": ann["cell_type"].astype("string").to_numpy(),
        "cell_class": ann["cell_class"].astype("string").to_numpy(),
        "cell_sub_class": ann["cell_sub_class"].astype("string").to_numpy(),
        "super_class": sc["super_class"].array,
        "side": pd.Categorical(recode(ann["side"], SIDE_MAP, "side").to_numpy(), dtype=config.NEURON_SCHEMA["side"][0]),
        "region": pd.array(["unknown"] * n, dtype="string"),
        "nt_type": pd.Categorical(recode(ann["top_nt"], NT_MAP, "nt_type").to_numpy(), dtype=config.NEURON_SCHEMA["nt_type"][0]),
        "nt_confidence": pd.to_numeric(ann["top_nt_conf"]).astype("float32").to_numpy(),
        "n_pre": pd.array([pd.NA] * n, dtype="Int32"), "n_post": pd.array([pd.NA] * n, dtype="Int32"),
        "flywire_type": pd.array([pd.NA] * n, dtype="string"), "manc_type": pd.array([pd.NA] * n, dtype="string"),
        "hemibrain_type": ann["hemibrain_type"].astype("string").to_numpy(),
        "dimorphism": ann["dimorphism"].astype("string").to_numpy(),
        "status": ann["status"].astype("string").to_numpy(),
        "hemilineage": ann["ito_lee_hemilineage"].astype("string").to_numpy(),
        "nt_known": ann["known_nt"].astype("string").to_numpy(),
        "super_class_source": sc["super_class_source"].array,
    })
    unk = neu["nt_type"].astype(str) == "unknown"
    neu.loc[unk, "nt_confidence"] = np.float32("nan")
    neu, rep = align_to_proofread(neu, proofread_ids)                     # correction 3
    reg = region_from_counts(neu["neuron_id"], counts)                    # correction 6
    neu["region"] = neu["neuron_id"].map(reg.set_index("neuron_id")["region"]).astype("string")
    rep["region_ties"] = int(reg["tied"].sum())
    rep["region_unknown"] = int((reg["region"] == "unknown").sum())
    rep["region_counts"] = neu["region"].value_counts().to_dict()
    log.warning("FAFB: %d annotation rows, %d dropped (not in the release), %d proofread ids added; region ties %d, region unknown %d",
                rep["annotation_rows"], rep["dropped_not_in_release"], rep["added_unknown"], rep["region_ties"], rep["region_unknown"])
    neu = add_nt_type_corrected(neu)                                      # correction 1
    rep["super_class_counts"] = neu["super_class"].astype(str).value_counts().to_dict()
    neu = neu[list(config.NEURON_SCHEMA)]
    return coerce(neu, config.NEURON_SCHEMA, "fafb neurons"), rep


def fetch_fafb(edges_path: str, annotations_path: str = ANNOTATIONS_URL, force_refresh: bool = False,
               proofread_ids_path: str | None = None, count_pre_path: str | None = None,
               count_post_path: str | None = None):
    """Build (or read) the three FAFB cache files and return (neurons, edges, edgeroi)."""
    rel = str(config.FAFB_MATERIALIZATION)
    p_n = config.cache_path("fafb", "neurons", rel, f"ann{config.FAFB_ANNOTATIONS_TAG[1:]}")
    p_e = config.cache_path("fafb", "edges", rel, config.FAFB_SYNAPSE_TABLE, min_syn=1)
    p_r = config.cache_path("fafb", "edgeroi", rel, config.FAFB_SYNAPSE_TABLE, min_syn=1)
    if all(p.exists() for p in (p_n, p_e, p_r)) and not force_refresh:
        return pd.read_parquet(p_n), pd.read_parquet(p_e), pd.read_parquet(p_r)
    if not (proofread_ids_path and count_pre_path and count_post_path):
        raise ValueError("FAFB needs --fafb-proofread-ids, --fafb-count-pre and --fafb-count-post "
                         "(Zenodo 10676866): see eda_shared/README.md, 'Inputs you need'")

    proofread = np.load(proofread_ids_path).astype("int64")
    edgeroi = read_edgeroi(edges_path)
    selfc = edgeroi.attrs.get("self_connections", 0)
    counts = read_neuropil_counts(count_pre_path, count_post_path)
    box = {}

    def make():
        neurons, rep = build_neurons(read_annotations(annotations_path), proofread, counts)
        box["rep"] = rep
        return neurons

    neurons = cached(p_n, make, validate_neurons, force_refresh)
    edgeroi = coerce(edgeroi[list(config.EDGEROI_SCHEMA)], config.EDGEROI_SCHEMA, "fafb edgeroi")
    edges = coerce(edges_from_edgeroi(edgeroi)[list(config.EDGE_SCHEMA)], config.EDGE_SCHEMA, "fafb edges")
    cached(p_e, lambda: edges, lambda d: validate_edges(d, neurons), force_refresh)
    cached(p_r, lambda: edgeroi, lambda d: validate_edges(d, neurons, roi=True), force_refresh)
    log.info("FAFB self-connections dropped: %d", selfc)
    if "rep" in box:
        box["rep"]["self_connections"] = int(selfc)
        write_report("fafb", box["rep"])
    return neurons, edges, edgeroi


def write_report(name: str, rep: dict) -> None:
    """Step 0 numbers that only the build can know (rows dropped, ties...), saved next to the cache."""
    import json
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (config.CACHE_DIR / f"step0_{name}.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
