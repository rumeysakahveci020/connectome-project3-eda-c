"""Tests for the fetch/cache/validate code (not for eda_shared; those are in eda_shared/tests)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from src import fetch_fafb as F, fetch_malecns as M
from src.cache import cached
from src.validators import assert_interchangeable, coerce, validate_edges, validate_neurons

ANN = Path("data/raw/Supplemental_file1_neuron_annotations.tsv")


def fafb_edge_csv(tmp_path):
    df = pd.DataFrame({
        "pre_root_id": [10, 10, 10, 20, 30, 30], "post_root_id": [20, 20, 30, 30, 10, 30],
        "neuropil": ["MB_CA_R", "MB_PED_R", "MB_CA_R", "ME_R", "SMP_L", "SMP_L"],
        "syn_count": [3, 4, 6, 2, 9, 5]})
    p = tmp_path / "edges.csv"
    df.to_csv(p, index=False)
    return str(p)


def toy_ann():
    return pd.DataFrame({
        "root_id": ["10", "20", "30"], "cell_type": ["KCg-m", None, "MBON01"],
        "cell_class": [None, "x", None], "cell_sub_class": [None] * 3,
        "super_class": ["central", "optic", "central"], "side": ["left", "na", "right"],
        "top_nt": ["acetylcholine", None, "gaba"], "top_nt_conf": ["0.9", "", "0.7"],
        "hemibrain_type": [None] * 3, "dimorphism": [None] * 3, "status": [None] * 3,
        "ito_lee_hemilineage": ["A", None, "B"], "known_nt": ["acetylcholine", None, None]})


def test_fafb_sum_over_neuropils_then_drop_self_and_regions(tmp_path):
    er = F.read_edgeroi(fafb_edge_csv(tmp_path))
    assert er.attrs["self_connections"] == 1                      # 30 -> 30 dropped
    e = F.edges_from_edgeroi(er)
    assert int(e.loc[(e.pre_id == 10) & (e.post_id == 20), "syn_count"].iloc[0]) == 7   # 3 + 4 summed first
    assert (e["syn_count"] >= 5).sum() == 3                       # 10->20 (7), 10->30 (6), 30->10 (9)
    n = F.build_neurons(toy_ann(), er)
    assert n.loc[n.neuron_id == 10, "region"].iloc[0] == "MB_CA_R"
    assert n.loc[n.neuron_id == 20, "side"].iloc[0] == "unknown"
    assert n["nt_confidence"].isna().tolist() == [False, True, False]
    validate_neurons(n)
    validate_edges(coerce(e[list(config.EDGE_SCHEMA)], config.EDGE_SCHEMA), n)


def test_fafb_unmapped_super_class_raises():
    ann = toy_ann()
    ann.loc[0, "super_class"] = "sensory_ascending"
    with pytest.raises(ValueError, match="sensory_ascending"):
        F.build_neurons(ann)
    F.build_neurons(ann, super_class_extra={"sensory_ascending": "ascending"})


@pytest.mark.skipif(not ANN.exists(), reason="real annotation file not downloaded")
def test_fafb_real_annotations_match_the_week2_numbers():
    ann = pd.read_csv(ANN, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    n = F.build_neurons(ann, super_class_extra={"sensory_ascending": "ascending"})
    validate_neurons(n)
    assert len(n) == 139_248
    assert abs(n["cell_class"].isna().mean() - 0.23) < 0.01
    assert abs(n["nt_confidence"].median() - 0.69) < 0.01


def test_validators_catch_problems():
    n = F.build_neurons(toy_ann())
    bad = n.copy()
    bad["neuron_id"] = bad["neuron_id"].astype("float64")
    with pytest.raises(ValueError, match="neuron_id"):
        validate_neurons(bad)
    e = F.edges_from_edgeroi(F.read_edgeroi.__wrapped__ if False else pd.DataFrame(
        {"pre_id": [10], "post_id": [99], "neuropil": ["x"], "syn_count": np.int32([5])}))
    e = coerce(e[list(config.EDGE_SCHEMA)], config.EDGE_SCHEMA)
    with pytest.raises(ValueError, match="post_id"):
        validate_edges(e, n)


def test_cache_reads_second_time_and_refetches_only_the_deleted_file(tmp_path):
    calls = []
    def fetch(name):
        def f():
            calls.append(name)
            return pd.DataFrame({"a": [1]})
        return f
    pa, pb = tmp_path / "a.parquet", tmp_path / "b.parquet"
    cached(pa, fetch("a")); cached(pb, fetch("b"))
    cached(pa, fetch("a")); cached(pb, fetch("b"))
    assert calls == ["a", "b"]
    pb.unlink()
    cached(pa, fetch("a")); cached(pb, fetch("b"))
    assert calls == ["a", "b", "b"]


def malecns_raw():
    return pd.DataFrame({
        "bodyId": [1, 2, 3], "type": ["KCg-m", "MBON01", None], "class": [None] * 3, "subclass": [None] * 3,
        "superclass": ["central_neuron", "Central", "sensory"], "somaSide": ["L", None, "R"],
        "rootSide": ["L", "R", "R"], "predictedNt": ["dopamine", "gaba", "unclear"],
        "predictedNtConfidence": [0.8, 0.9, 0.3], "consensusNt": ["acetylcholine", "gaba", None],
        "pre": [10, 5, 1], "post": [20, 30, 2], "flywireType": [None] * 3, "mancType": [None] * 3,
        "hemibrainType": [None] * 3, "dimorphism": [None] * 3, "status": ["Traced"] * 3, "itoleeHl": [None] * 3})


def test_malecns_neurons_and_edges():
    roi = pd.DataFrame({"bodyId": [1, 1, 2, 3], "roi": ["CA(R)", "PED(R)", "a'L(R)", "LegNp(T1)(L)"],
                        "pre": [5, 1, 2, 1], "post": [6, 1, 3, 1]})
    n = M.build_neurons(malecns_raw(), roi, ["CA(R)", "PED(R)", "a'L(R)", "LegNp(T1)(L)"])
    validate_neurons(n)
    assert n.set_index("neuron_id").loc[1, "region"] == "CA(R)"
    assert n.set_index("neuron_id").loc[2, "side"] == "right"          # somaSide missing -> rootSide
    assert n.set_index("neuron_id").loc[3, "nt_type"] == "unknown"
    assert n["nt_confidence"].isna().tolist() == [False, False, True]
    fa = F.build_neurons(toy_ann())
    assert_interchangeable(fa, n)


def test_malecns_unmapped_side_raises():
    raw = malecns_raw()
    raw.loc[0, "somaSide"] = "X"
    with pytest.raises(ValueError, match="side"):
        M.build_neurons(raw, None, [])
    raw.loc[1, "superclass"] = "cb_weird"
    with pytest.raises(ValueError, match="cb_weird.*side|side.*cb_weird"):   # both problems in one message
        M.build_neurons(raw, None, [])


def test_fafb_feather_edge_file(tmp_path):
    df = pd.DataFrame({"pre_pt_root_id": np.array([10, 10, 20], dtype="uint64") + 720575940000000000,
                       "post_pt_root_id": np.array([20, 30, 30], dtype="uint64") + 720575940000000000,
                       "neuropil": ["MB_CA_R", "MB_CA_R", "SMP_L"], "syn_count": np.array([3, 6, 9], dtype="int32"),
                       "gaba_avg": [0.1, 0.1, 0.1]})
    p = tmp_path / "proofread_connections_783.feather"
    df.to_feather(p)
    er = F.read_edgeroi(str(p))
    assert len(er) == 3 and er["pre_id"].dtype == "int64" and int(er.pre_id.iloc[0]) == 720575940000000010


def test_malecns_bulk_file_route_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    raw = malecns_raw()
    ann = raw.drop(columns=["predictedNt", "predictedNtConfidence", "consensusNt"]).copy()
    ann["status"] = ["Traced", "Traced", "Orphan"]
    ann.to_feather(tmp_path / "ann.feather")
    raw[["bodyId", "predictedNt", "predictedNtConfidence", "consensusNt"]].to_feather(tmp_path / "nt.feather")
    pd.DataFrame({"body_pre": [1, 1, 2, 3], "body_post": [2, 2, 1, 1], "roi": ["CA(R)", "PED(R)", "a'L(R)", "x"],
                  "weight": [3, 4, 6, 9]}).to_feather(tmp_path / "edges.feather")
    n, e, r = M.fetch_malecns(str(tmp_path / "edges.feather"), str(tmp_path / "ann.feather"), True, None,
                              str(tmp_path / "nt.feather"))
    assert sorted(n["neuron_id"]) == [1, 2]                       # the Orphan row is gone
    assert int(e.loc[(e.pre_id == 1) & (e.post_id == 2), "syn_count"].iloc[0]) == 7
    assert (config.CACHE_DIR / "malecns_nt_consensus_v1.0.parquet").exists()
    validate_neurons(n)


def test_malecns_without_any_region_information_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    raw = malecns_raw()
    raw.to_feather(tmp_path / "ann.feather")
    pd.DataFrame({"body_pre": [1], "body_post": [2], "weight": [5]}).to_feather(tmp_path / "edges.feather")
    with pytest.raises(RuntimeError, match="region"):
        M.fetch_malecns(str(tmp_path / "edges.feather"), str(tmp_path / "ann.feather"), True)


def test_fafb_proofread_list_drops_stale_annotation_ids_and_keeps_edge_ids(tmp_path):
    er = F.read_edgeroi(fafb_edge_csv(tmp_path))                    # edges touch 10, 20, 30
    ann = toy_ann()
    stale = ann.iloc[[0]].copy(); stale["root_id"] = "99"           # annotated, not in the release list
    both = pd.concat([ann, stale], ignore_index=True)
    plain = F.build_neurons(both, er)
    assert sorted(plain["neuron_id"]) == [10, 20, 30, 99]
    kept = F.build_neurons(both, er, proofread_ids={10, 20, 30, 31})
    assert sorted(kept["neuron_id"]) == [10, 20, 30]


def test_streaming_feather_filter_matches_a_full_read(tmp_path):
    import pyarrow as pa, pyarrow.feather as pf
    df = pd.DataFrame({"body_pre": np.arange(1, 41, dtype="int64") % 7 + 1, "body_post": np.arange(2, 42, dtype="int64") % 7 + 1,
                       "weight": np.arange(40, dtype="int32") + 1})
    pf.write_feather(pa.Table.from_pandas(df), tmp_path / "e.feather", chunksize=6)      # many record batches
    from src.tables import read_feather_filtered
    keep = {1, 2, 3}
    got = read_feather_filtered(tmp_path / "e.feather", ["body_pre", "body_post", "weight"], ("body_pre", "body_post"), keep)
    want = df[df.body_pre.isin(keep) & df.body_post.isin(keep)].reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want, check_dtype=False)


def test_malecns_neuprint_route_with_a_three_column_edge_file(tmp_path, monkeypatch):
    """The real bulk file has only body_pre, body_post, weight (no ROI, no confidence)."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    raw = malecns_raw()
    roi = pd.DataFrame({"bodyId": [1, 1, 2, 3], "roi": ["CA(R)", "PED(R)", "a'L(R)", "LegNp(T1)(L)"],
                        "pre": [5, 1, 2, 1], "post": [6, 1, 3, 1]})
    monkeypatch.setattr(M, "fetch_neurons_neuprint", lambda: (raw, roi, ["CA(R)", "PED(R)", "a'L(R)", "LegNp(T1)(L)"]))
    pd.DataFrame({"body_pre": [1, 2, 3, 9, 1], "body_post": [2, 1, 1, 1, 1], "weight": [7, 6, 9, 4, 3]}
                 ).to_feather(tmp_path / "edges.feather")
    n, e, r = M.fetch_malecns(str(tmp_path / "edges.feather"), None, True)
    assert sorted(n["neuron_id"]) == [1, 2, 3]
    assert len(e) == 3 and 9 not in set(e.pre_id)                     # outside neuron 9 and the self-connection are gone
    assert set(r["neuropil"].astype(str)) == {"all"}
    assert n.set_index("neuron_id").loc[1, "region"] == "CA(R)"
    validate_neurons(n)


def test_provisional_map_targets_are_valid_vocabulary():
    from src.provisional_maps import PROVISIONAL_SUPER_CLASS
    assert set(PROVISIONAL_SUPER_CLASS.values()) <= set(config.SUPER_CLASS_VALUES)


def test_igraph_and_networkx_triad_census_agree():
    ig = pytest.importorskip("igraph")
    import networkx as nx
    from src import eda_c as C
    rng = np.random.default_rng(3)
    n = 300
    w = rng.pareto(1.2, n) + 1
    w /= w.sum()
    a, b = rng.choice(n, 3000, p=w), rng.choice(n, 3000, p=w)
    e = pd.DataFrame({"pre_id": a, "post_id": b}).query("pre_id != post_id").drop_duplicates()
    g = nx.DiGraph(); g.add_nodes_from(range(n)); g.add_edges_from(zip(e.pre_id, e.post_id))
    assert C._census(e, range(n)) == dict(nx.triadic_census(g))
    assert C.TRIAD_NAMES.index("111D") < C.TRIAD_NAMES.index("111U")


def test_malecns_region_uses_upstream_plus_downstream_when_roiinfo_has_them():
    raw = malecns_raw().copy()
    # neuron 1: pre + post would pick CA(R) (10 vs 4); upstream + downstream picks PED(R) (3 + 20 vs 1 + 1)
    raw["roiInfo"] = [{"CA(R)": {"pre": 8, "post": 2, "upstream": 1, "downstream": 1},
                       "PED(R)": {"pre": 1, "post": 3, "upstream": 3, "downstream": 20}},
                      {"a'L(R)": {"pre": 1, "post": 1, "upstream": 2, "downstream": 2}},
                      {"CA(R)": {"pre": 1, "post": 1, "upstream": 1, "downstream": 1}}][: len(raw)]
    roi = pd.DataFrame({"bodyId": [1, 1], "roi": ["CA(R)", "PED(R)"], "pre": [8, 1], "post": [2, 3]})
    neu = M.build_neurons(raw, roi, ["CA(R)", "PED(R)", "a'L(R)"])
    assert neu.set_index("neuron_id").loc[1, "region"] == "PED(R)"


def test_group_rows_make_the_unspecified_regions_known_and_keep_eda_shared_unchanged():
    from src import eda_utils as U
    n = pd.DataFrame({"neuron_id": [1, 2, 3, 4], "region": ["Optic-unspecified(L)", "Optic-unspecified(R)",
                                                           "CentralBrain-unspecified", "CA(R)"]})
    out = U.flags(n, frozenset())
    assert out["in_brain"].all()
    assert out.set_index("neuron_id")["is_optic"].to_dict() == {1: True, 2: True, 3: False, 4: False}
    assert (out.set_index("neuron_id").loc[[1, 2, 3], "region_group"] == "unknown").all()
    from eda_shared.eda_helpers import load_neuropil_table
    assert "Optic-unspecified" not in load_neuropil_table().index        # the shared table itself is not edited


def _c1_toy():
    import matplotlib
    matplotlib.use("Agg")
    def neurons(rows):
        return pd.DataFrame(rows, columns=["neuron_id", "nt_type", "super_class", "region_group", "is_optic"])
    nt = ["acetylcholine", "gaba", "glutamate"]
    big = [(i, nt[i % 3], "central", "higher_order", False) for i in range(100)]
    small = [(1000 + i, "gaba", "optic", "unknown", False) for i in range(5)]               # 5 neurons: faded
    only_fafb = [(2000 + i, "acetylcholine", "sensory", "sensory_periphery", False) for i in range(80)]   # a row male CNS lacks
    opt = [(3000 + i, nt[i % 3], "optic", "sensory_periphery", True) for i in range(60)]
    edges = pd.DataFrame({"pre_id": [0], "post_id": [1], "syn_count": [5]})
    return {"fafb": (neurons(big + small + only_fafb + opt), edges), "malecns": (neurons(big + opt), edges)}


def test_c1_figure_rows_are_the_same_in_both_panels_small_rows_faded_and_single_group_not_duplicated():
    from src import eda_c as C
    data = _c1_toy()
    fig, _ = C.c1_figure(data, "central", "super_class", False)
    left, right = [[t.get_text() for t in ax.get_yticklabels()] for ax in fig.axes[:2]]
    assert [x.split(" (")[0].rstrip("*") for x in left] == [x.split(" (")[0].rstrip("*") for x in right]   # same rows, same order
    assert any("(none)" in x for x in right) and any(x.startswith("sensory") and "(n=80)" in x for x in left)  # row only FlyWire has
    assert any(x.startswith("optic*") for x in left)                                                         # 5 neurons: marked *
    faded = [p.get_alpha() for p in fig.axes[0].patches if p.get_alpha() is not None and p.get_alpha() < 1]
    assert faded
    fig2, _ = C.c1_figure(data, "optic", "region_group", False)                                              # one group only
    labels = [t.get_text() for t in fig2.axes[0].get_yticklabels()]
    assert len(labels) == 1 and "overall = sensory_periphery" in labels[0]


def test_c3_figure_counts_every_hemilineage_and_explains_itself():
    import matplotlib
    matplotlib.use("Agg")
    from src import eda_c as C
    tab = pd.DataFrame({"neurons": [30] * 12, "dominant": ["acetylcholine"] * 12,
                        "share": [1.0, 1.0, 1.0, 0.99, 0.97, 0.92, 0.9, 0.85, 0.7, 0.55, 0.5, 0.5]},
                       index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "primary", "L"])
    fig = C.c3_figure(tab)
    ax = fig.axes[0]
    assert sum(p.get_height() for p in ax.patches) == len(tab)                 # the old histogram lost the purity-1.0 bar
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "PURE" in texts and "MIXED" in texts and "How to read this plot" in texts and "Least pure" in texts and "primary*" in texts
    assert "58% of 12 hemilineages are pure" in ax.get_title() or "of 12 hemilineages are pure" in ax.get_title()


def test_every_figure_carries_the_fingerprint_of_the_code_that_drew_it(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from src import eda_utils as U
    monkeypatch.setattr(U, "FIG_DIR", tmp_path)
    fp = U.code_fingerprint()
    assert len(fp) == 16
    fig, ax = plt.subplots()
    ax.plot([0, 1])
    path = U.save(fig, "x.png", "caption")
    assert Image.open(path).info.get("Comment") == f"code_sha={fp}"


def test_c1_row_identical_to_overall_says_so_and_c4_region_labels_do_not_collide():
    import matplotlib
    matplotlib.use("Agg")
    from src import eda_c as C
    nt = ["acetylcholine", "gaba", "glutamate"]
    rows = [(i, nt[i % 3], "optic", "sensory_periphery", True) for i in range(60)] + [(1000 + i, "gaba", "optic", "unknown", True) for i in range(70)]
    n = pd.DataFrame(rows, columns=["neuron_id", "nt_type", "super_class", "region_group", "is_optic"])
    n_f = n[n["region_group"] == "sensory_periphery"]                       # FlyWire-like: one group, equal to overall
    edges = pd.DataFrame({"pre_id": [0], "post_id": [1], "syn_count": [5]})
    fig, _ = C.c1_figure({"fafb": (n_f, edges), "malecns": (n, edges)}, "optic", "region_group", False)
    left = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert any("same as overall" in x for x in left)
    net = pd.DataFrame({"neuron_id": range(30), "rule": list(C.RULES)[0:1] * 15 + list(C.RULES)[1:2] * 15, "net": np.linspace(-1, 1, 30),
                        "region_group": ["sensory_periphery", "higher_order", "motor_periphery"] * 10})
    fig4 = C.c4_figure(net, "region_group")
    assert fig4.get_figwidth() >= 9 and all(t.get_rotation() == 25 for t in fig4.axes[0].get_xticklabels())
