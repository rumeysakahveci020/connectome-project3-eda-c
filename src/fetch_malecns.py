"""Janelia male CNS v1.0, Traced neurons only, synapse confidence >= 0.5 (D6).

Neurons : neuPrint male-cns:v1.0 (token in NEUPRINT_TOKEN or a gitignored .env), or a local file
          with the same properties (--malecns-neurons).
Edges   : the bulk download for the same release (exact addresses are in Project 4's README),
          one row per (pre, post, roi). Cached at 1 synapse; thresholds in memory.

Column names marked "(confirm)" in the Data Conventions are guesses until the first run prints the
real ones. The functions below print them and raise naming the columns they could not find.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.cache import cached
from src.common import first_col, log, recode, region_from_counts
from src.tables import column_names, read_feather_filtered, read_table
from src.validators import coerce, validate_edges, validate_neurons

# Nerve-cord ROIs, taken from the primary ROI list the male CNS prints. Check this list with Project 4.
NERVE_CORD_ROIS = frozenset({
    "ANm", "IntTct", "LTct", "VNC-unspecified", "CV-unspecified",
    *{f"{b}({s})" for s in "LR" for b in [
        "ADMN", "AbN1", "AbN2", "AbN3", "AbN4", "AbNT", "CvN", "DMetaN", "DProN", "HTct(UTct-T3)",
        "LegNp(T1)", "LegNp(T2)", "LegNp(T3)", "MesoAN", "MesoLN", "MetaLN", "NTct(UTct-T1)",
        "Ov", "PDMN", "PrN", "ProAN", "ProCN", "ProLN", "VProN", "WTct(UTct-T2)",
        "mVAC(T1)", "mVAC(T2)", "mVAC(T3)"]},
})

SIDE_MAP = {"left": "left", "l": "left", "right": "right", "r": "right",
            "center": "center", "centre": "center", "m": "center", "c": "center", "middle": "center",
            "unknown": "unknown", "": "unknown", "na": "unknown"}
NT_MAP = {"acetylcholine": "acetylcholine", "gaba": "gaba", "glutamate": "glutamate",
          "dopamine": "dopamine", "serotonin": "serotonin", "octopamine": "octopamine",
          "histamine": "histamine", "unclear": "unknown", "unknown": "unknown"}
SUPER_CLASS_MAP = {v: v for v in config.SUPER_CLASS_VALUES}   # + normalised spellings, see _norm_sc


def _norm_sc(s: pd.Series) -> pd.Series:
    """Normalise the male CNS superclass spelling onto the FAFB levels where it is unambiguous
    (case, spaces, a trailing _neuron/_neurons). Anything else is left alone and raises."""
    t = s.astype("string").str.strip().str.lower().str.replace(" ", "_", regex=False)
    t = t.str.replace(r"_neurons?$", "", regex=True)
    return t


def fetch_neurons_neuprint():
    """All Traced neurons and their per-ROI synapse counts from neuPrint. Slow: allow time."""
    from src.build_env import load_dotenv_simple
    cache = config.CACHE_DIR.parent / "raw" / "neuprint_malecns_v1.0_traced.pkl"
    if cache.exists():
        log.warning("reading the saved neuPrint answer %s (delete it to download again)", cache)
        return pd.read_pickle(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    from neuprint import Client, NeuronCriteria as NC, fetch_neurons
    load_dotenv_simple()
    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        raise RuntimeError("NEUPRINT_TOKEN is not set (environment variable or gitignored .env)")
    client = Client("https://neuprint.janelia.org", dataset=config.MALECNS_DATASET, token=token)
    neuron_df, roi_df = fetch_neurons(NC(status=config.MALECNS_STATUS), client=client)
    primary = sorted(client.primary_rois)
    pd.to_pickle((neuron_df, roi_df, primary), cache)
    log.warning("saved the neuPrint answer to %s (no second download needed)", cache)
    return neuron_df, roi_df, primary


def roi_connection_counts(raw: pd.DataFrame, primary_rois) -> pd.DataFrame | None:
    """bodyId -> per-ROI connection counts from roiInfo: upstream + downstream.

    Group decision (D15): the Data Conventions count one synapse per sender-receiver contact, and the FAFB
    edge table does the same, so the male CNS field in the same unit is upstream + downstream (not pre + post,
    which counts release sites once). Returns neuron_id, region, syn, or None if roiInfo has no such fields."""
    import json
    if "roiInfo" not in raw.columns:
        return None
    keep = set(primary_rois)
    rows, seen_fields = [], False
    for b, d in zip(raw["bodyId"].tolist(), raw["roiInfo"].tolist()):
        if isinstance(d, str):
            try:
                d = json.loads(d)
            except ValueError:
                continue
        if not isinstance(d, dict):
            continue
        for roi, v in d.items():
            if roi in keep and isinstance(v, dict):
                if "upstream" in v or "downstream" in v:
                    seen_fields = True
                rows.append((b, roi, (v.get("upstream") or 0) + (v.get("downstream") or 0)))
    if not seen_fields:
        return None
    out = pd.DataFrame(rows, columns=["neuron_id", "region", "syn"])
    out["neuron_id"] = out["neuron_id"].astype("int64")
    return out[out["syn"] > 0]


def build_neurons(raw: pd.DataFrame, roi_counts: pd.DataFrame | None, primary_rois,
                  edgeroi: pd.DataFrame | None = None, super_class_extra: dict | None = None) -> pd.DataFrame:
    """raw: neuPrint neuron properties (one row per bodyId). roi_counts: bodyId, roi, pre, post.
    Region comes from the per-ROI synapse counts (roiInfo); if they are not available it is
    computed from the edgeroi table, as for FAFB."""
    print("male CNS neuron columns:", sorted(map(str, raw.columns)))
    c = lambda cands, what, req=True: first_col(raw.columns, cands, what, req)   # noqa: E731
    n = len(raw)
    na_str = lambda: pd.Series([pd.NA] * n, dtype="string", index=raw.index)    # noqa: E731

    def col(cands, what):
        k = c(cands, what, req=False)
        return raw[k].astype("string") if k else na_str()

    soma = c(["somaSide"], "somaSide", False)
    root = c(["rootSide"], "rootSide", False)
    side_src = pd.Series(pd.NA, index=raw.index, dtype="string")
    if soma:
        side_src = raw[soma].astype("string").str.strip().str.lower()
    if root:
        r = raw[root].astype("string").str.strip().str.lower()
        side_src = side_src.where(side_src.notna() & ~side_src.isin(["", "unknown", "na"]), r)
    sc_col = c(["superclass", "superClass"], "superclass")
    sc_map = {**SUPER_CLASS_MAP, **(super_class_extra or {})}
    nt_col = c(["predictedNt"], "predictedNt")
    conf_col = c(["predictedNtConfidence", "predictedNtProb"], "predicted NT confidence", False)

    errors = []

    def safe(fn):
        try:
            return fn()
        except ValueError as exc:                     # collect every mapping problem, report them together
            errors.append(str(exc))
            return pd.Series("unknown", index=raw.index, dtype="string")

    nt = safe(lambda: recode(raw[nt_col].astype("string").str.strip().str.lower(), NT_MAP, "nt_type"))
    conf = pd.to_numeric(raw[conf_col], errors="coerce").astype("float32") if conf_col \
        else pd.Series(np.float32("nan"), index=raw.index, dtype="float32")
    conf = conf.where(nt != "unknown", np.float32("nan"))

    def int32(cands):
        k = c(cands, "count", False)
        return pd.to_numeric(raw[k], errors="coerce").astype("Int32") if k else \
            pd.Series(pd.array([pd.NA] * n, dtype="Int32"), index=raw.index)

    neu = pd.DataFrame({
        "neuron_id": raw[c(["bodyId"], "bodyId")].astype("int64"),
        "dataset": "malecns",
        "release": "v1.0",
        "cell_type": col(["type"], "type"),
        "cell_class": col(["class", "cellClass"], "class"),
        "cell_sub_class": col(["subclass", "subClass", "cellSubclass"], "subclass"),
        "super_class": safe(lambda: recode(_norm_sc(raw[sc_col]), sc_map, "super_class")),
        "side": safe(lambda: recode(side_src, SIDE_MAP, "side")),
        "region": "unknown",
        "nt_type": nt,
        "nt_confidence": conf,
        "n_pre": int32(["pre"]), "n_post": int32(["post"]),
        "flywire_type": col(["flywireType"], "flywireType"),
        "manc_type": col(["mancType"], "mancType"),
        "hemibrain_type": col(["hemibrainType"], "hemibrainType"),
        "dimorphism": col(["dimorphism"], "dimorphism (confirm)"),
        "status": col(["status"], "status"),
        "hemilineage": col(["itoleeHl", "itoleeHemilineage"], "itoleeHl"),
        "nt_known": na_str(),
    })
    neu["neuron_id"] = neu["neuron_id"].astype("int64")
    if errors:
        raise ValueError(" | ".join(errors))

    reg = None
    conn = roi_connection_counts(raw, primary_rois) if primary_rois is not None and len(primary_rois) else None
    if conn is not None and len(conn):
        reg, ties = region_from_counts(conn)
        log.warning("male CNS region: the ROI with the most upstream + downstream (connection counts, group decision D15)")
    elif roi_counts is not None and len(roi_counts):
        log.warning("male CNS region: roiInfo has no upstream/downstream, falling back to pre + post (NOT the group rule)")
        r = roi_counts.rename(columns={"bodyId": "neuron_id", "roi": "region"}).copy()
        r = r[r["region"].isin(set(primary_rois))]
        r["syn"] = pd.to_numeric(r["pre"]).fillna(0) + pd.to_numeric(r["post"]).fillna(0)
        reg, ties = region_from_counts(r[["neuron_id", "region", "syn"]])
    elif edgeroi is not None and not (edgeroi["neuropil"] == "all").all():
        a = edgeroi.groupby(["pre_id", "neuropil"])["syn_count"].sum()
        b = edgeroi.groupby(["post_id", "neuropil"])["syn_count"].sum()
        a.index.names = b.index.names = ["neuron_id", "region"]
        reg, ties = region_from_counts(a.add(b, fill_value=0).rename("syn").reset_index())
    if reg is not None:
        log.warning("male CNS neurons whose top two ROIs tie: %d (alphabetical rule)", ties)
        neu["region"] = neu["neuron_id"].map(reg).fillna("unknown").astype(str)
    neu = neu.astype({c_: "string" for c_ in ["release", "cell_type", "cell_class", "cell_sub_class",
                                              "region", "flywire_type", "manc_type", "hemibrain_type",
                                              "dimorphism", "status", "hemilineage", "nt_known"]})
    neu["nt_known"] = pd.array([pd.NA] * len(neu), dtype="string")
    if (neu["region"] == "unknown").all():
        raise RuntimeError("male CNS: no region information. Use the neuPrint route (it returns the per-ROI "
                           "synapse counts) or an edge file that has a ROI column.")
    return coerce(neu[list(config.NEURON_SCHEMA)], config.NEURON_SCHEMA, "malecns neurons")


def read_edgeroi(path: str, traced_ids: set[int]) -> pd.DataFrame:
    head = column_names(path)
    print("male CNS edge columns:", sorted(map(str, head)))
    pre = first_col(head, ["bodyId_pre", "body_pre", "pre_id", "pre"], "presynaptic id")
    post = first_col(head, ["bodyId_post", "body_post", "post_id", "post"], "postsynaptic id")
    w = first_col(head, ["weight", "syn_count", "synapse_count", "count"], "synapse count")
    roi = first_col(head, ["roi", "neuropil", "primary_post", "primary_roi"], "roi", required=False)
    conf = first_col(head, ["confidence", "conf", "min_confidence"], "confidence", required=False)
    cols = [c for c in (pre, post, w, roi, conf) if c]
    if str(path).lower().endswith(".feather"):
        raw = read_feather_filtered(path, cols, (pre, post), traced_ids)     # streams; never loads the whole file
        log.warning("male CNS edge rows between Traced neurons: %d", len(raw))
    else:
        raw = read_table(path, columns=cols)
    if roi is None:
        log.warning("no ROI column in the edge file: the neuropil column is set to 'all' (region then "
                    "comes from the neuron table, not from the edges)")
        raw["__roi"] = "all"
        roi = "__roi"
    if conf is None:
        log.warning("no confidence column in the edge file: the name says minconf-0.5, so it is assumed "
                    "to be the conf >= %s release (check with Project 4)", config.MALECNS_SYN_CONFIDENCE)
    else:
        raw = raw[raw[conf] >= config.MALECNS_SYN_CONFIDENCE]
    e = pd.DataFrame({"pre_id": raw[pre].astype("int64"), "post_id": raw[post].astype("int64"),
                      "neuropil": raw[roi].astype("string"),
                      "syn_count": pd.to_numeric(raw[w]).astype("int32")})
    del raw
    selfc = int((e["pre_id"] == e["post_id"]).sum())
    log.warning("male CNS self-connections dropped: %d (expected 101 on the whole dataset)", selfc)
    e = e[e["pre_id"] != e["post_id"]]
    if e.duplicated(["pre_id", "post_id", "neuropil"]).any():     # only sums when a pair repeats
        e = e.groupby(["pre_id", "post_id", "neuropil"], as_index=False)["syn_count"].sum()
    e["syn_count"] = e["syn_count"].astype("int32")
    e["dataset"], e["release"] = "malecns", "v1.0"
    e["min_syn_fetched"] = np.int32(config.FETCH_MIN_SYN)
    e.attrs["self_connections"] = selfc
    return e


def edges_from_edgeroi(edgeroi: pd.DataFrame) -> pd.DataFrame:
    if (edgeroi["neuropil"] == "all").all():            # no ROI breakdown in the source: already one row per pair
        e = edgeroi[["pre_id", "post_id", "syn_count"]].copy()
    else:
        e = edgeroi.groupby(["pre_id", "post_id"], as_index=False)["syn_count"].sum()
    e["syn_count"] = e["syn_count"].astype("int32")
    e["dataset"], e["release"] = "malecns", "v1.0"
    e["min_syn_fetched"] = np.int32(config.FETCH_MIN_SYN)
    return e


def load_neurons_from_files(annotations_path: str, nt_path: str | None) -> pd.DataFrame:
    """Bulk-download route: body-annotations (+ body-neurotransmitters, merged on the body id)."""
    raw = read_table(annotations_path)
    idc = first_col(raw.columns, ["bodyId", "body", "body_id"], "body id")
    raw = raw.rename(columns={idc: "bodyId"})
    if nt_path:
        nt = read_table(nt_path)
        nid = first_col(nt.columns, ["bodyId", "body", "body_id"], "body id (neurotransmitter file)")
        print("male CNS neurotransmitter file columns:", sorted(map(str, nt.columns)))
        nt = nt.rename(columns={nid: "bodyId"})
        nt = nt[["bodyId"] + [c for c in nt.columns if c != "bodyId" and c not in raw.columns]]
        raw = raw.merge(nt, on="bodyId", how="left")
    stat = first_col(raw.columns, ["status"], "status", required=False)
    if stat:
        before = len(raw)
        raw = raw[raw[stat].astype(str) == config.MALECNS_STATUS]
        log.warning("male CNS neurons with status %s: %d of %d rows", config.MALECNS_STATUS, len(raw), before)
    return raw.reset_index(drop=True)


def fetch_malecns(edges_path: str, neurons_path: str | None = None, force_refresh: bool = False,
                  super_class_extra: dict | None = None, nt_path: str | None = None):
    src = "traced"
    p_n = config.cache_path("malecns", "neurons", "v1.0", src)
    p_e = config.cache_path("malecns", "edges", "v1.0", src, min_syn=1, conf=config.MALECNS_SYN_CONFIDENCE)
    p_r = config.cache_path("malecns", "edgeroi", "v1.0", src, min_syn=1, conf=config.MALECNS_SYN_CONFIDENCE)
    if all(p.exists() for p in (p_n, p_e, p_r)) and not force_refresh:
        return pd.read_parquet(p_n), pd.read_parquet(p_e), pd.read_parquet(p_r)

    roi_counts = primary = None
    if neurons_path:
        raw = load_neurons_from_files(neurons_path, nt_path)
    else:
        raw, roi_counts, primary = fetch_neurons_neuprint()
    # keep a copy of consensusNt: it has no confidence and is not in the neurons schema, but it
    # is the only transmitter check the male CNS offers (see C1/C2).
    cons = first_col(raw.columns, ["consensusNt"], "consensusNt", required=False)
    if cons:
        extra = pd.DataFrame({"neuron_id": raw["bodyId"].astype("int64"), "consensusNt": raw[cons].astype("string")})
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        extra.to_parquet(config.CACHE_DIR / "malecns_nt_consensus_v1.0.parquet", index=False)
    traced = set(raw["bodyId"].astype("int64"))
    # With neuPrint the region comes from the per-ROI counts, so the neuron table can be built first;
    # a mapping problem then shows up in seconds instead of after reading 25 million edges.
    edgeroi = read_edgeroi(edges_path, traced) if roi_counts is None else None
    neurons = cached(p_n, lambda: build_neurons(raw, roi_counts, primary, edgeroi, super_class_extra),
                     validate_neurons, force_refresh)
    if edgeroi is None:
        edgeroi = read_edgeroi(edges_path, traced)
    edgeroi = coerce(edgeroi[list(config.EDGEROI_SCHEMA)], config.EDGEROI_SCHEMA, "malecns edgeroi")
    edges = coerce(edges_from_edgeroi(edgeroi)[list(config.EDGE_SCHEMA)], config.EDGE_SCHEMA, "malecns edges")
    cached(p_e, lambda: edges, lambda d: validate_edges(d, neurons), force_refresh)
    cached(p_r, lambda: edgeroi, lambda d: validate_edges(d, neurons, roi=True), force_refresh)
    return neurons, edges, edgeroi
