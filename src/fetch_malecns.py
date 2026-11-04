"""Janelia male CNS v1.0, Traced neurons only, synapse confidence >= 0.5 (D6), cropped to the BRAIN PART by synapse
location (Week 4 correction 4).

Neurons : neuPrint male-cns:v1.0 (token in NEUPRINT_TOKEN or a gitignored .env): neuron table, per-ROI table, primary ROIs.
Edges   : syn-partners-male-cns-v1.0-minconf-0.5.feather (bulk download, 6.8 GB): one row per synapse with the
          primary neuropil of the postsynaptic site. It is streamed and counted per (pre, post, neuropil): that is the
          edgeroi table with ONE ROW PER PRIMARY NEUROPIL that eda_shared.malecns_brain_part needs.
          connectome-weights-male-cns-v1.0-minconf-0.5.feather (optional) is used to check that the per-neuropil
          counts add up to the published weights.
Corrections 1, 2, 4 and 6 are applied by eda_shared v0.2.0 (map_super_class, add_nt_type_corrected,
malecns_brain_part, region_from_counts), not by code in this file.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import config
from eda_shared.corrections import add_nt_type_corrected, malecns_brain_part, malecns_neuropil_counts, map_super_class
from src.cache import cached
from src.common import first_col, log, recode
from src.fetch_fafb import write_report
from src.tables import column_names, read_feather_filtered
from src.validators import coerce, validate_edges, validate_neurons

SIDE_MAP = {"left": "left", "l": "left", "right": "right", "r": "right",
            "center": "center", "centre": "center", "m": "center", "c": "center", "middle": "center",
            "unknown": "unknown", "": "unknown", "na": "unknown"}
NT_MAP = {"acetylcholine": "acetylcholine", "gaba": "gaba", "glutamate": "glutamate",
          "dopamine": "dopamine", "serotonin": "serotonin", "octopamine": "octopamine",
          "histamine": "histamine", "unclear": "unknown", "unknown": "unknown"}


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


def build_neurons(raw: pd.DataFrame) -> pd.DataFrame:
    """raw: neuPrint neuron properties (one row per bodyId). Region is a placeholder: malecns_brain_part
    computes it (shared rule, brain neuropils only) and drops the neurons that are not in the brain part."""
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
    nt_col = c(["predictedNt"], "predictedNt")
    conf_col = c(["predictedNtConfidence", "predictedNtProb"], "predicted NT confidence", False)
    sc = map_super_class(raw[sc_col], "malecns")                           # correction 2 (raises on an unmapped value)
    nt = recode(raw[nt_col].astype("string").str.strip().str.lower(), NT_MAP, "nt_type")
    conf = pd.to_numeric(raw[conf_col], errors="coerce").astype("float32") if conf_col \
        else pd.Series(np.float32("nan"), index=raw.index, dtype="float32")
    conf = conf.where(nt != "unknown", np.float32("nan"))

    def int32(cands):
        k = c(cands, "count", False)
        return pd.to_numeric(raw[k], errors="coerce").astype("Int32").to_numpy() if k else pd.array([pd.NA] * n, dtype="Int32")

    neu = pd.DataFrame({
        "neuron_id": raw[c(["bodyId"], "bodyId")].astype("int64").to_numpy(),
        "dataset": pd.Categorical(["malecns"] * n, dtype=config.NEURON_SCHEMA["dataset"][0]),
        "release": pd.array(["v1.0"] * n, dtype="string"),
        "cell_type": col(["type"], "type").to_numpy(),
        "cell_class": col(["class", "cellClass"], "class").to_numpy(),
        "cell_sub_class": col(["subclass", "subClass", "cellSubclass"], "subclass").to_numpy(),
        "super_class": sc["super_class"].array,
        "side": pd.Categorical(recode(side_src, SIDE_MAP, "side").to_numpy(), dtype=config.NEURON_SCHEMA["side"][0]),
        "region": pd.array(["unknown"] * n, dtype="string"),
        "nt_type": pd.Categorical(nt.to_numpy(), dtype=config.NEURON_SCHEMA["nt_type"][0]),
        "nt_confidence": conf.to_numpy(),
        "n_pre": int32(["pre"]), "n_post": int32(["post"]),
        "flywire_type": col(["flywireType"], "flywireType").to_numpy(),
        "manc_type": col(["mancType"], "mancType").to_numpy(),
        "hemibrain_type": col(["hemibrainType"], "hemibrainType").to_numpy(),
        "dimorphism": col(["dimorphism"], "dimorphism (confirm)").to_numpy(),
        "status": col(["status"], "status").to_numpy(),
        "hemilineage": col(["itoleeHl", "itoleeHemilineage"], "itoleeHl").to_numpy(),
        "nt_known": pd.array([pd.NA] * n, dtype="string"),
        "super_class_source": sc["super_class_source"].array,
    })
    neu = add_nt_type_corrected(neu)                                       # correction 1: Kenyon cells are cholinergic
    return neu[list(config.NEURON_SCHEMA)]


def read_syn_partners(path: str, traced_ids, flush_every: int = 100) -> pd.DataFrame:
    """Stream syn-partners (one row per synapse) and count, per (pre, post, primary neuropil), the synapses between
    two Traced neurons. Self-connections are dropped and counted (D9; expected 101 on the whole dataset).
    Never holds more than one record batch of raw rows plus the running counts."""
    import pyarrow as pa
    import pyarrow.compute as pc

    head = column_names(path)
    print("male CNS syn-partners columns:", sorted(map(str, head)))
    pre_c = first_col(head, ["body_pre", "bodyId_pre", "pre_id", "pre"], "presynaptic id")
    post_c = first_col(head, ["body_post", "bodyId_post", "post_id", "post"], "postsynaptic id")
    roi_c = first_col(head, ["primary_post", "primary_roi", "roi", "neuropil"], "primary neuropil")
    keep = np.sort(np.asarray(list(traced_ids), dtype="int64"))
    codes: dict[str, int] = {}
    parts, selfc, rows, kept = [], 0, 0, 0

    def compact():
        nonlocal parts
        if len(parts) > 1:
            parts = [pd.concat(parts, ignore_index=True).groupby(["pre_id", "post_id", "roi"], sort=False, as_index=False)["syn_count"].sum()]

    with pa.memory_map(str(path)) as src:
        rdr = pa.ipc.open_file(src)
        for i in range(rdr.num_record_batches):
            b = rdr.get_batch(i)
            rows += b.num_rows
            pre = b.column(pre_c).to_numpy(zero_copy_only=False).astype("int64")
            post = b.column(post_c).to_numpy(zero_copy_only=False).astype("int64")
            m = np.isin(pre, keep) & np.isin(post, keep)
            if not m.any():
                continue
            roi = b.column(roi_c)
            if not pa.types.is_string(roi.type):
                roi = roi.cast(pa.string())
            roi = pc.dictionary_encode(pc.fill_null(roi, "<unspecified>")).combine_chunks() if isinstance(roi, pa.ChunkedArray) \
                else pc.dictionary_encode(pc.fill_null(roi, "<unspecified>"))
            local = np.array([codes.setdefault(v, len(codes)) for v in roi.dictionary.to_pylist()], dtype="int32")
            rc = local[roi.indices.to_numpy(zero_copy_only=False).astype("int64")]
            is_self = pre == post
            selfc += int((m & is_self).sum())
            m &= ~is_self
            kept += int(m.sum())
            df = pd.DataFrame({"pre_id": pre[m], "post_id": post[m], "roi": rc[m]})
            parts.append(df.groupby(["pre_id", "post_id", "roi"], sort=False).size().rename("syn_count").reset_index())
            if len(parts) >= flush_every:
                compact()
    compact()
    out = parts[0] if parts else pd.DataFrame({"pre_id": [], "post_id": [], "roi": [], "syn_count": []})
    names = np.array(sorted(codes, key=codes.get), dtype=object)
    e = pd.DataFrame({"pre_id": out["pre_id"].astype("int64"), "post_id": out["post_id"].astype("int64"),
                      "neuropil": pd.array(names[out["roi"].to_numpy()] if len(out) else [], dtype="string"),
                      "syn_count": out["syn_count"].astype("int32")})
    log.warning("male CNS syn-partners: %d synapse rows read, %d between Traced neurons (self-connections dropped: %d; expected 101)",
                rows, kept, selfc)
    e["dataset"], e["release"] = "malecns", "v1.0"
    e["min_syn_fetched"] = np.int32(config.FETCH_MIN_SYN)
    e.attrs["self_connections"] = selfc
    return e


def check_against_weights(edgeroi: pd.DataFrame, weights_path: str, traced_ids) -> dict:
    """The per-neuropil counts summed over neuropils must equal connectome-weights for the same pairs. Reported, not raised:
    a difference is a question for the group (which file defines the edge table)."""
    head = column_names(weights_path)
    pre = first_col(head, ["body_pre", "bodyId_pre", "pre_id", "pre"], "presynaptic id")
    post = first_col(head, ["body_post", "bodyId_post", "post_id", "post"], "postsynaptic id")
    w = first_col(head, ["weight", "syn_count", "synapse_count", "count"], "weight")
    ref = read_feather_filtered(weights_path, [pre, post, w], (pre, post), set(map(int, traced_ids)))
    ref = ref[ref[pre] != ref[post]]
    mine = edgeroi.groupby(["pre_id", "post_id"], as_index=False)["syn_count"].sum()
    out = {"weights_pairs": int(len(ref)), "weights_synapses": int(ref[w].sum()),
           "syn_partners_pairs": int(len(mine)), "syn_partners_synapses": int(mine["syn_count"].sum())}
    out["agree"] = out["weights_pairs"] == out["syn_partners_pairs"] and out["weights_synapses"] == out["syn_partners_synapses"]
    (log.info if out["agree"] else log.warning)("male CNS connectome-weights vs syn-partners: %s", out)
    return out


def fetch_malecns(syn_partners_path: str, weights_path: str | None = None, force_refresh: bool = False):
    """Build (or read) the three male CNS cache files (brain part) and return (neurons, edges, edgeroi)."""
    src = "traced"
    p_n = config.cache_path("malecns", "neurons", "v1.0", src)
    p_e = config.cache_path("malecns", "edges", "v1.0", src, min_syn=1, conf=config.MALECNS_SYN_CONFIDENCE)
    p_r = config.cache_path("malecns", "edgeroi", "v1.0", src, min_syn=1, conf=config.MALECNS_SYN_CONFIDENCE)
    if all(p.exists() for p in (p_n, p_e, p_r)) and not force_refresh:
        return pd.read_parquet(p_n), pd.read_parquet(p_e), pd.read_parquet(p_r)

    raw, roi_df, primary = fetch_neurons_neuprint()
    # consensusNt has no confidence and is not in the neurons schema, but it is the only transmitter check the male CNS
    # offers (C1, C2b). Keep a copy next to the cache.
    cons = first_col(raw.columns, ["consensusNt"], "consensusNt", required=False)
    if cons:
        extra = pd.DataFrame({"neuron_id": raw["bodyId"].astype("int64"), "consensusNt": raw[cons].astype("string")})
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        extra.to_parquet(config.CACHE_DIR / "malecns_nt_consensus_v1.0.parquet", index=False)
    neurons_all = build_neurons(raw)                                  # a mapping problem shows up now, not after 6.8 GB
    traced = set(neurons_all["neuron_id"].astype("int64"))
    edgeroi = read_syn_partners(syn_partners_path, traced)
    selfc = edgeroi.attrs.get("self_connections", 0)
    rep = {"neurons_traced": int(len(neurons_all)), "self_connections": int(selfc)}
    if weights_path:
        rep["weights_check"] = check_against_weights(edgeroi, weights_path, traced)
    roi_counts = malecns_neuropil_counts(roi_df[roi_df["roi"].isin(set(primary))])
    nb, eb, erb, brain = malecns_brain_part(neurons_all, edgeroi, roi_counts)      # corrections 4 and 6
    rep.update(brain)
    log.warning("male CNS brain part: %s", brain)
    rep["super_class_counts"] = nb["super_class"].astype(str).value_counts().to_dict()
    rep["region_counts"] = nb["region"].astype(str).value_counts().to_dict()
    neurons = coerce(nb[list(config.NEURON_SCHEMA)], config.NEURON_SCHEMA, "malecns neurons")
    edgeroi_b = coerce(erb[list(config.EDGEROI_SCHEMA)], config.EDGEROI_SCHEMA, "malecns edgeroi")
    edges = coerce(eb[list(config.EDGE_SCHEMA)], config.EDGE_SCHEMA, "malecns edges")
    cached(p_n, lambda: neurons, validate_neurons, force_refresh)
    cached(p_e, lambda: edges, lambda d: validate_edges(d, neurons), force_refresh)
    cached(p_r, lambda: edgeroi_b, lambda d: validate_edges(d, neurons, roi=True), force_refresh)
    write_report("malecns", rep)
    return neurons, edges, edgeroi_b
