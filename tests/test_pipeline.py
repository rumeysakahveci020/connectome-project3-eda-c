"""Tests of the corrected pipeline (eda_shared v0.2.0): the build recovers planted facts from synthetic raw files, the streaming reader,
the validators, the cache, and the figures. Nothing here touches real data or the network."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from src import fetch_fafb as F, fetch_malecns as M
from src.cache import cached
from src.validators import assert_interchangeable, coerce, validate_edges, validate_neurons

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------------------------- the synthetic build
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import make_synthetic_raw as S
    base = tmp_path_factory.mktemp("syn")
    raw = S.main(str(base))
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "CACHE_DIR", base / "cache")
    from src.build_cache import main
    rc = main(["--fafb-edges", str(raw / "proofread_connections_783.feather"), "--fafb-annotations", str(raw / "Supplemental_file1_neuron_annotations.tsv"),
               "--fafb-proofread-ids", str(raw / "proofread_root_ids_783.npy"),
               "--fafb-count-pre", str(raw / "per_neuron_neuropil_count_pre_783.feather"), "--fafb-count-post", str(raw / "per_neuron_neuropil_count_post_783.feather"),
               "--malecns-syn-partners", str(raw / "syn-partners-male-cns-v1.0-minconf-0.5.feather"),
               "--malecns-weights", str(raw / "connectome-weights-male-cns-v1.0-minconf-0.5.feather")])
    assert rc == 0
    import json
    out = {"raw": raw, "S": S}
    for name in ("fafb", "malecns"):
        rel = "783" if name == "fafb" else "v1.0"
        src = f"ann{config.FAFB_ANNOTATIONS_TAG[1:]}" if name == "fafb" else "traced"
        esrc = config.FAFB_SYNAPSE_TABLE if name == "fafb" else "traced"
        kw = {} if name == "fafb" else {"conf": config.MALECNS_SYN_CONFIDENCE}
        out[name] = (pd.read_parquet(config.cache_path(name, "neurons", rel, src)),
                     pd.read_parquet(config.cache_path(name, "edges", rel, esrc, min_syn=1, **kw)),
                     pd.read_parquet(config.cache_path(name, "edgeroi", rel, esrc, min_syn=1, **kw)),
                     json.loads((config.CACHE_DIR / f"step0_{name}.json").read_text()))
    yield out
    mp.undo()


def test_fafb_neuron_set_is_the_proofread_list(built):
    n, e, r, rep = built["fafb"]
    S = built["S"].PLANTED
    assert len(n) == S["fafb_n"]
    assert rep["dropped_not_in_release"] == S["fafb_not_in_release"] and rep["added_unknown"] == S["fafb_not_annotated"]
    assert set(n["neuron_id"]) == set(np.load(built["raw"] / "proofread_root_ids_783.npy"))
    added = n[n["cell_type"].isna()]
    assert len(added) >= S["fafb_not_annotated"] and (added["super_class"].astype(str) == "unknown").all()
    assert (added["region"].astype(str) != "unknown").all()                  # the 14 have synapses, so they get a real region
    assert set(e["pre_id"]).union(e["post_id"]) <= set(n["neuron_id"])         # no edge dropped for lack of a neuron


def test_fafb_region_is_never_unassigned_and_ties_go_to_the_alphabetically_first_name(built):
    n, *_ = built["fafb"]
    assert not n["region"].astype(str).isin(["UNASGD", "None"]).any()
    pre = pd.read_feather(built["raw"] / "per_neuron_neuropil_count_pre_783.feather").rename(columns={"pre_pt_root_id": "neuron_id"})
    post = pd.read_feather(built["raw"] / "per_neuron_neuropil_count_post_783.feather").rename(columns={"post_pt_root_id": "neuron_id"})
    c = pd.concat([pre, post]).query("neuropil != 'UNASGD'").groupby(["neuron_id", "neuropil"], as_index=False)["count"].sum()
    c = c.sort_values(["neuron_id", "count", "neuropil"], ascending=[True, False, True]).drop_duplicates("neuron_id").set_index("neuron_id")["neuropil"]
    got = n.set_index("neuron_id")["region"].astype(str)
    ids = c.index.intersection(got.index)
    base = lambda s: s.str.replace(r"_[LR]$", "", regex=True)                  # canonical names are checked by eda_shared; compare the raw names
    assert (got.loc[ids] == c.loc[ids]).all()


def test_kenyon_cells_are_cholinergic_after_the_correction_in_both_datasets(built):
    for name in ("fafb", "malecns"):
        n = built[name][0]
        kc = n[n["cell_type"].astype("string").str.startswith("KC", na=False)]
        assert len(kc) > 50 and (kc["nt_type_corrected"].astype(str) == "acetylcholine").all()
        assert set(kc["nt_correction_source"].astype(str)) <= {"known_nt", "kc_override", "predicted"}
        assert (n["nt_type"].astype(str) == "dopamine").sum() > (n["nt_type_corrected"].astype(str) == "dopamine").sum()   # nt_type itself is unchanged
    assert (built["malecns"][0].loc[lambda d: d["cell_type"].astype("string").str.startswith("KC", na=False), "nt_correction_source"].astype(str) == "kc_override").all()


def test_super_class_keeps_the_raw_value_and_uses_the_shared_levels(built):
    n = built["fafb"][0]
    assert (n["super_class"].astype(str) == "sensory_ascending").any()
    m = built["malecns"][0]
    assert {"optic", "central", "sensory", "ascending"} <= set(m["super_class"].astype(str))
    assert (m.loc[m["super_class"].astype(str) == "optic", "super_class_source"] == "ol_intrinsic").all()
    assert set(m["super_class"].astype(str)) <= set(config.SUPER_CLASS_VALUES)


def test_malecns_brain_part_drops_nerve_cord_only_neurons_and_keeps_ascending(built):
    n, e, r, rep = built["malecns"]
    assert rep["neurons_traced"] == built["S"].PLANTED["malecns_n"] and 0 < len(n) < rep["neurons_traced"]
    assert rep["neurons_dropped"] == rep["neurons_traced"] - len(n)
    assert rep["ascending_kept"] > 0 and rep["sensory_ascending_kept"] > 0
    assert not n["region"].astype(str).str.startswith("LegNp").any()          # no nerve-cord neuropil is ever a region
    assert set(e["pre_id"]).union(e["post_id"]) <= set(n["neuron_id"])
    assert r["neuropil"].astype(str).str.startswith("LegNp").sum() == 0       # edgeroi_b: brain rows only
    assert int(e["syn_count"].sum()) == rep["synapses_tau1"]
    assert rep["self_connections"] == built["S"].PLANTED["malecns_self"]


def test_malecns_per_neuropil_counts_add_up_to_the_published_weights(built):
    assert built["malecns"][3]["weights_check"]["agree"] is True


def test_both_tables_are_interchangeable_and_valid(built):
    for name in ("fafb", "malecns"):
        n, e, r, _ = built[name]
        validate_neurons(n); validate_edges(e, n); validate_edges(r, n, roi=True)
    assert_interchangeable(built["fafb"][0], built["malecns"][0])
    assert_interchangeable(built["fafb"][1], built["malecns"][1])


def test_step0_tool_prints_the_table(built, capsys, monkeypatch):
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import step0
    monkeypatch.setattr(step0, "ROOT", built["raw"].parent)
    assert step0.main() == 0
    out = capsys.readouterr().out
    assert "FAFB neurons" in out and "MaleCNS brain part: neurons kept" in out and "Transmitter correction" in out and "Kenyon-cell types" in out
    assert (built["raw"].parent / "reports" / "step0_C.md").exists()


def test_mushroom_body_type_mix_flags_a_type_with_two_transmitters(built):
    from src import eda_c as C
    from src import eda_utils as U
    n = U.flags(built["fafb"][0], frozenset())
    mix = C.mb_type_mix(n)
    assert {"cell_type", "neurons", "top_nt", "top_share", "n_nt", "mix"} <= set(mix.columns) and len(mix) >= 2
    kc = mix[mix["cell_type"] == "KCg-m"].iloc[0]
    assert kc["top_nt"] == "acetylcholine" and kc["top_share"] == 1.0 and kc["n_nt"] == 1
    mbon = mix[mix["cell_type"] == "MBON01"]
    assert (mbon["n_nt"] > 1).all()                                            # the planted MBON types mix GABA, glutamate and acetylcholine


def test_config_is_the_week4_version():
    assert config.CONFIG_VERSION == "0.2.0"
    assert {"sensory_ascending", "efferent", "vnc_intrinsic"} <= set(config.SUPER_CLASS_VALUES)
    assert {"nt_type_corrected", "nt_correction_source", "super_class_source"} <= set(config.NEURON_SCHEMA)


# ---------------------------------------------------------------------------------------------- the streaming reader
def test_syn_partners_are_counted_per_neuropil_across_batches(tmp_path):
    import pyarrow as pa, pyarrow.feather as pf
    rows = [(1, 2, "SMP(R)"), (1, 2, "SMP(R)"), (1, 2, "FB"), (1, 2, None), (1, 2, "NotPrimary"),    # pair 1->2: 5 synapses, 4 locations
            (2, 3, "FB"), (3, 3, "FB"),                                                             # 3->3 is a self-connection
            (1, 99, "FB"), (99, 2, "FB")]                                                           # 99 is not Traced
    df = pd.DataFrame(rows, columns=["body_pre", "body_post", "primary_post"])
    pf.write_feather(pa.Table.from_pandas(df), tmp_path / "sp.feather", chunksize=2)                # five record batches
    e = M.read_syn_partners(str(tmp_path / "sp.feather"), {1, 2, 3})
    got = {(r.pre_id, r.post_id, r.neuropil): r.syn_count for r in e.itertuples()}
    assert got == {(1, 2, "SMP(R)"): 2, (1, 2, "FB"): 1, (1, 2, "<unspecified>"): 1, (1, 2, "NotPrimary"): 1, (2, 3, "FB"): 1}
    assert e.attrs["self_connections"] == 1 and e["syn_count"].dtype == "int32"


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


# ---------------------------------------------------------------------------------------------- validators and cache
def test_validators_catch_problems(built):
    n = built["fafb"][0]
    bad = n.copy()
    bad["neuron_id"] = bad["neuron_id"].astype("float64")
    with pytest.raises(ValueError, match="neuron_id"):
        validate_neurons(bad)
    e = built["fafb"][1].iloc[:1].copy()
    e["post_id"] = np.int64(99)
    with pytest.raises(ValueError, match="post_id"):
        validate_edges(e, n)
    with pytest.raises(ValueError, match="vocabulary"):
        coerce(n.assign(nt_type_corrected="unobtainium")[list(config.NEURON_SCHEMA)], config.NEURON_SCHEMA)


def test_an_unmapped_male_cns_superclass_raises_naming_the_value():
    raw = pd.DataFrame({"bodyId": [1], "type": ["x"], "superclass": ["mystery_class"], "predictedNt": ["gaba"], "predictedNtConfidence": [.9]})
    with pytest.raises(ValueError, match="mystery_class"):
        M.build_neurons(raw)


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


def test_fafb_build_without_the_corrected_inputs_says_what_is_missing(tmp_path):
    with pytest.raises(ValueError, match="fafb-count-pre"):
        F.fetch_fafb(str(tmp_path / "none.feather"), force_refresh=True)


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


# ---------------------------------------------------------------------------------------------- figures
def _c1_toy():
    import matplotlib
    matplotlib.use("Agg")
    def neurons(rows):
        d = pd.DataFrame(rows, columns=["neuron_id", "nt_type", "super_class", "region_group", "is_optic"])
        d["nt_type_corrected"] = d["nt_type"]
        return d
    nt = ["acetylcholine", "gaba", "glutamate"]
    big = [(i, nt[i % 3], "central", "higher_order", False) for i in range(100)]
    small = [(1000 + i, "gaba", "optic", "unknown", False) for i in range(5)]               # 5 neurons: faded
    only_fafb = [(2000 + i, "acetylcholine", "sensory", "sensory_periphery", False) for i in range(80)]   # a row male CNS lacks
    opt = [(3000 + i, nt[i % 3], "optic", "sensory_periphery", True) for i in range(60)]
    edges = pd.DataFrame({"pre_id": [0], "post_id": [1], "syn_count": [5]})
    return {"fafb": (neurons(big + small + only_fafb + opt), edges), "malecns": (neurons(big + opt), edges)}


def test_c1_shares_use_the_corrected_label_unless_told_otherwise():
    from src import eda_c as C
    n = pd.DataFrame({"neuron_id": range(10), "nt_type": ["dopamine"] * 10, "nt_type_corrected": ["acetylcholine"] * 8 + ["dopamine"] * 2,
                      "super_class": "central", "region_group": "higher_order", "is_optic": False})
    edges = pd.DataFrame({"pre_id": [0], "post_id": [1], "syn_count": [5]})
    assert C.c1_shares(n, edges, "central", "super_class", False).loc["overall", "acetylcholine"] == pytest.approx(0.8)
    assert C.c1_shares(n, edges, "central", "super_class", False, nt_col="nt_type").loc["overall", "dopamine"] == pytest.approx(1.0)


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
    assert "of 12 hemilineages are pure" in ax.get_title()


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
    n["nt_type_corrected"] = n["nt_type"]
    n_f = n[n["region_group"] == "sensory_periphery"]                       # FlyWire-like: one group, equal to overall
    edges = pd.DataFrame({"pre_id": [0], "post_id": [1], "syn_count": [5]})
    fig, _ = C.c1_figure({"fafb": (n_f, edges), "malecns": (n, edges)}, "optic", "region_group", False)
    left = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert any("same as overall" in x for x in left)
    net = pd.DataFrame({"neuron_id": range(30), "rule": list(C.RULES)[0:1] * 15 + list(C.RULES)[1:2] * 15, "net": np.linspace(-1, 1, 30),
                        "region_group": ["sensory_periphery", "higher_order", "motor_periphery"] * 10})
    fig4 = C.c4_figure(net, "region_group")
    assert fig4.get_figwidth() >= 9 and all(t.get_rotation() == 25 for t in fig4.axes[0].get_xticklabels())
