"""Tests for corrections.py (v0.2.0). Run with:  pytest eda_shared/tests"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from corrections import (MALECNS_SUPER_CLASS, NT_VALUES, SUPER_CLASS_VALUES,  # noqa: E402
                         add_nt_type_corrected, align_to_proofread,
                         fafb_neuropil_counts, malecns_neuropil_counts, region_from_counts,
                         map_super_class,
                         malecns_brain_part, neuropil_location, nt_correction_summary,
                         parse_known_nt, single_known_nt)
from eda_helpers import MALECNS_NERVE_CORD, relative_weight  # noqa: E402

NT = pd.CategoricalDtype(NT_VALUES)


# ---------- correction 1: transmitters ----------

@pytest.mark.parametrize("raw, expected", [
    # strings copied from FlyWire annotations v3.2.0 (known_nt)
    ("acetylcholine", "acetylcholine"),
    ("acetylcholine; sNPF; acetylcholine, sNPF", "acetylcholine"),     # Kenyon cells
    ("histamine", "histamine"),
    ("gaba, nitric oxide", "gaba"),
    ("glutamate,  acetylcholine-negative, gaba-negative", "glutamate"),
    ("gaba-negative", None),                                           # only a negative
    ("allatostatin-a", None),                                          # only a peptide
    ("histamine; acetylcholine, histamine", None),                     # two transmitters
    ("glutamate, tyramine", None),                                     # tyramine counts as one
    ("tyramine", None),                                                # not in NT_VALUES
    ("acetylcholine, nitric oxide, dopamine", None),
    (None, None),
])
def test_single_known_nt(raw, expected):
    assert single_known_nt(raw) == expected


def test_parse_known_nt_ignores_negatives():
    assert parse_known_nt("acetylcholine, gaba-negative, glutamate-negative") == {"acetylcholine"}


def neurons_nt():
    return pd.DataFrame({
        "neuron_id": [1, 2, 3, 4, 5, 6, 7],
        "dataset": ["fafb", "fafb", "fafb", "fafb", "malecns", "malecns", "malecns"],
        "cell_type": ["KCab", "R7", "DA1_lPN", "PS180", "KCg-m", "PKC1", pd.NA],
        "nt_type": pd.Series(["dopamine", "acetylcholine", "acetylcholine", "unknown",
                              "dopamine", "gaba", "glutamate"], dtype=NT),
        "nt_known": pd.Series(["acetylcholine; sNPF; acetylcholine, sNPF", "histamine",
                               "acetylcholine", pd.NA, "acetylcholine", pd.NA, pd.NA],
                              dtype="string"),
    })


def test_nt_type_corrected_rules():
    out = add_nt_type_corrected(neurons_nt())
    assert list(out["nt_type_corrected"].astype(str)) == [
        "acetylcholine",   # fafb KC: known_nt
        "histamine",       # fafb photoreceptor: known_nt
        "acetylcholine",   # fafb, known agrees with predicted: still source known_nt
        "unknown",         # fafb, no known_nt
        "acetylcholine",   # malecns KC: override (malecns nt_known is not used)
        "gaba",            # PKC1 does not start with KC
        "glutamate"]       # missing cell_type
    assert list(out["nt_correction_source"].astype(str)) == [
        "known_nt", "known_nt", "known_nt", "predicted", "kc_override", "predicted", "predicted"]
    assert out["nt_type"].equals(neurons_nt()["nt_type"])          # original untouched
    assert out["nt_type_corrected"].dtype == NT


def test_nt_correction_summary_counts_changes():
    s = nt_correction_summary(add_nt_type_corrected(neurons_nt())).set_index(
        ["dataset", "nt_correction_source"])
    assert s.loc[("fafb", "known_nt"), "rows"] == 3
    assert s.loc[("fafb", "known_nt"), "changed"] == 2
    assert s.loc[("malecns", "kc_override"), "changed"] == 1


def test_nt_rule_refuses_other_datasets():
    n = neurons_nt().assign(dataset="hemibrain")
    with pytest.raises(ValueError, match="no rule"):
        add_nt_type_corrected(n)


# ---------- correction 2: superclass ----------

FAFB_RAW = ["optic", "central", "sensory", "visual_projection", "ascending", "descending",
            "sensory_ascending", "visual_centrifugal", "motor", "endocrine"]
# every value of malecns v1.0 `superclass` (Project 4, reports/evidence.md)
MALECNS_RAW = ["ol_intrinsic", "cb_intrinsic", "vnc_intrinsic", "visual_projection",
               "vnc_sensory", "cb_sensory", "ol_sensory", "ascending_neuron",
               "descending_neuron", "vnc_motor", "visual_centrifugal", "sensory_ascending",
               "cb_motor", "vnc_efferent", "cb_endocrine", "ENS", "vnc_endocrine",
               "sensory_descending", "vnc_sensory_tbc", "efferent_ascending", "cb_sensory_tbc",
               "efferent_descending", "cb_efferent", "visual_projection_tbc",
               "sensory_ascending_tbc", "vnc_tbc"]


def test_fafb_super_class_is_identity_plus_unknown():
    raw = pd.Series(FAFB_RAW + [None], dtype="string")
    out = map_super_class(raw, "fafb")
    assert list(out["super_class"].astype(str)) == FAFB_RAW + ["unknown"]
    assert out["super_class_source"].isna().sum() == 1
    assert out["super_class"].dtype == pd.CategoricalDtype(SUPER_CLASS_VALUES)


def test_every_malecns_value_is_mapped():
    raw = pd.Series(MALECNS_RAW + [None], dtype="string")
    out = map_super_class(raw, "malecns")
    m = dict(zip(MALECNS_RAW, out["super_class"].astype(str)))
    assert m["sensory_ascending"] == m["sensory_ascending_tbc"] == "sensory_ascending"
    assert m["vnc_intrinsic"] == "vnc_intrinsic"
    assert {m[k] for k in ["cb_efferent", "vnc_efferent", "efferent_ascending",
                           "efferent_descending"]} == {"efferent"}
    assert m["visual_projection_tbc"] == "visual_projection"
    assert m["cb_sensory_tbc"] == m["vnc_sensory_tbc"] == "sensory"
    assert m["ENS"] == m["vnc_tbc"] == "unknown"
    assert out["super_class"].iloc[-1] == "unknown"
    assert list(out["super_class_source"].iloc[:-1]) == MALECNS_RAW   # raw kept


def test_unmapped_super_class_raises():
    with pytest.raises(ValueError, match="unmapped"):
        map_super_class(pd.Series(["central", "brand_new_level"]), "fafb")


def test_malecns_map_targets_are_levels():
    assert set(MALECNS_SUPER_CLASS.values()) <= set(SUPER_CLASS_VALUES)


# ---------- correction 3: the FAFB neuron set ----------

def fafb_neurons():
    return pd.DataFrame({
        "neuron_id": pd.Series([10, 11], dtype="int64"),
        "dataset": pd.Series(["fafb", "fafb"], dtype=pd.CategoricalDtype(["fafb", "malecns"])),
        "release": pd.Series(["783", "783"], dtype="string"),
        "cell_type": pd.Series(["KCab", "PS180"], dtype="string"),
        "super_class": pd.Series(["central", "central"],
                                 dtype=pd.CategoricalDtype(SUPER_CLASS_VALUES)),
        "region": pd.Series(["MB_CA_R", "SMP_L"], dtype="string"),
        "nt_type": pd.Series(["dopamine", "acetylcholine"], dtype=NT),
        "nt_confidence": pd.Series([0.9, 0.8], dtype="float32"),
        "n_pre": pd.Series([5, 6], dtype="Int32"),
        "nt_known": pd.Series([pd.NA, pd.NA], dtype="string"),
    })


def test_align_to_proofread_drops_and_adds():
    # annotation rows 10, 11; proofread ids 10, 12 -> 11 dropped, 12 added
    out, rep = align_to_proofread(fafb_neurons(), np.array([12, 10]))
    assert sorted(out["neuron_id"]) == [10, 12]
    assert rep == {"annotation_rows": 2, "dropped_not_in_release": 1, "added_unknown": 1, "total": 2}
    new = out.set_index("neuron_id").loc[12]
    assert new["super_class"] == "unknown" and new["region"] == "unknown"
    assert (out.dtypes == fafb_neurons().dtypes).all()


def test_align_refuses_columns_added_too_early():
    n = add_nt_type_corrected(fafb_neurons())            # nt_correction_source has no "unknown"
    with pytest.raises(ValueError, match="nt_correction_source"):
        align_to_proofread(n, np.array([10, 11, 12]))


def test_fafb_build_order_from_the_readme():
    n = fafb_neurons().drop(columns="super_class")
    sc = map_super_class(pd.Series(["central", "sensory_ascending"]), "fafb")
    n["super_class"] = sc["super_class"].array
    n["super_class_source"] = sc["super_class_source"].array
    n, rep = align_to_proofread(n, np.array([10, 11, 12]))
    pre = pd.DataFrame({"pre_pt_root_id": [10, 11, 12, 12], "neuropil": ["MB_CA_R", "SMP_L", "LO_L", "None"],
                        "count": [5, 7, 3, 9]})
    post = pd.DataFrame({"post_pt_root_id": [10, 12], "neuropil": ["MB_PED_R", "ME_L"], "count": [9, 2]})
    reg = region_from_counts(n["neuron_id"], fafb_neuropil_counts(pre, post))
    n["region"] = n["neuron_id"].map(reg.set_index("neuron_id")["region"]).astype("string")
    n = add_nt_type_corrected(n)
    assert rep["added_unknown"] == 1
    assert list(n["super_class"].astype(str)) == ["central", "sensory_ascending", "unknown"]
    assert n["super_class_source"].isna().tolist() == [False, False, True]
    assert list(n["region"]) == ["MB_PED_R", "SMP_L", "LO_L"]     # 12: "None" (9) is not a candidate
    assert list(n["nt_correction_source"].astype(str)) == ["predicted"] * 3


# ---------- correction 6: the region rule ----------

def test_region_rule_unasgd_ties_and_unknown():
    counts = pd.DataFrame({
        "neuron_id": [1, 1, 1, 2, 2, 3, 4],
        "neuropil": pd.Series(["UNASGD", "SMP_R", "GNG", "SMP_R", "GNG", "None", "AL_L"],
                              dtype=pd.CategoricalDtype(["UNASGD", "SMP_R", "GNG", "None", "AL_L"])),
        "count": [50, 10, 9, 6, 6, 4, 0]})
    r = region_from_counts([1, 2, 3, 4, 5], counts).set_index("neuron_id")
    assert r.loc[1, "region"] == "SMP_R" and not r.loc[1, "tied"]   # UNASGD (50) is not a candidate
    assert r.loc[2, "region"] == "GNG" and r.loc[2, "tied"]          # string order, not category order
    assert (r.loc[[3, 4, 5], "region"] == "unknown").all()           # no location, zero, no rows
    assert r.loc[5, "region_count"] == 0


def test_region_rule_excludes_nerve_cord_for_malecns():
    counts = pd.DataFrame({"neuron_id": [1, 1], "neuropil": ["LegNp(T1)(L)", "GNG"], "count": [90, 2]})
    assert region_from_counts([1], counts, nerve_cord=MALECNS_NERVE_CORD)["region"].iloc[0] == "GNG"
    with pytest.raises(ValueError, match="unmapped"):                # FAFB has no nerve cord names
        region_from_counts([1], counts)


def test_malecns_counts_from_roiinfo():
    roi = pd.DataFrame({"bodyId": [7, 7], "roi": ["ME(R)", "LO(R)"], "pre": [1, 1], "post": [2, 2],
                        "upstream": [5, 1], "downstream": [3, 9]})
    c = malecns_neuropil_counts(roi)
    assert list(c["count"]) == [8, 10]
    assert region_from_counts([7], c, nerve_cord=MALECNS_NERVE_CORD)["region"].iloc[0] == "LO(R)"


# ---------- correction 4: MaleCNS brain part, toy worked by hand ----------
#  neurons: 1 central, 2 ascending (CNS-wide region in the leg neuropil),
#           3 vnc_intrinsic, 4 optic, 5 only synapses outside every ROI,
#           6 a brain synapse with untraced partners only
#  edgeroi (pre, post, neuropil, syn):
#     1->2 SMP(R) 6, 1->2 GNG 2          -> brain edge 1->2 = 8
#     2->1 LegNp(T1)(L) 9, 2->1 GNG 3    -> brain edge 2->1 = 3 (nerve cord dropped)
#     3->2 LegNp(T1)(L) 20               -> nerve cord
#     4->1 ME(R) 4, 4->1 Optic-unspecified(R) 1   -> brain edge 4->1 = 5
#     1->4 <unspecified> 7, 5->1 <unspecified> 2  -> no location, dropped
#     2->4 CentralBrain-unspecified 5    -> brain edge 2->4 = 5
#  roi_counts: 1 ties SMP(R) and GNG at 10 -> GNG (alphabetical); 2 -> GNG;
#              3, 5 have no brain count -> dropped; 6 -> CentralBrain-unspecified

def toy_malecns():
    neurons = pd.DataFrame({
        "neuron_id": pd.Series([1, 2, 3, 4, 5, 6], dtype="int64"),
        "dataset": "malecns",
        "super_class": ["central", "ascending", "vnc_intrinsic", "optic", "central", "central"],
        "region": pd.Series(["SMP(R)", "LegNp(T1)(L)", "ANm", "ME(R)", "UNASGD",
                             "CentralBrain-unspecified"], dtype="string"),
    })
    rows = [(1, 2, "SMP(R)", 6), (1, 2, "GNG", 2), (2, 1, "LegNp(T1)(L)", 9), (2, 1, "GNG", 3),
            (3, 2, "LegNp(T1)(L)", 20), (4, 1, "ME(R)", 4), (4, 1, "Optic-unspecified(R)", 1),
            (1, 4, "<unspecified>", 7), (5, 1, "<unspecified>", 2),
            (2, 4, "CentralBrain-unspecified", 5)]
    er = pd.DataFrame(rows, columns=["pre_id", "post_id", "neuropil", "syn_count"])
    er = er.astype({"pre_id": "int64", "post_id": "int64", "syn_count": "int32",
                    "neuropil": "string"})
    er["dataset"] = "malecns"
    er["release"] = pd.Series(["v1.0"] * len(er), dtype="string")
    er["min_syn_fetched"] = pd.Series([1] * len(er), dtype="int32")
    rc = pd.DataFrame([(1, "SMP(R)", 10), (1, "GNG", 10), (1, "ME(R)", 4),
                       (2, "LegNp(T1)(L)", 50), (2, "GNG", 6), (2, "CentralBrain-unspecified", 5),
                       (3, "LegNp(T1)(L)", 20), (3, "ANm", 3),
                       (4, "ME(R)", 9), (4, "Optic-unspecified(R)", 1),
                       (5, "<unspecified>", 2),
                       (6, "CentralBrain-unspecified", 3)],
                      columns=["neuron_id", "neuropil", "count"])
    return neurons, er, rc


def test_brain_part_by_hand():
    neurons, er, rc = toy_malecns()
    n, e, erb, rep = malecns_brain_part(neurons, er, rc)
    assert list(n["neuron_id"]) == [1, 2, 4, 6]
    assert dict(zip(n["neuron_id"], n["region"])) == {
        1: "GNG", 2: "GNG", 4: "ME(R)", 6: "CentralBrain-unspecified"}
    assert list(map(tuple, e[["pre_id", "post_id", "syn_count"]].to_numpy())) == [
        (1, 2, 8), (2, 1, 3), (2, 4, 5), (4, 1, 5)]
    assert (e["min_syn_fetched"] == 1).all()
    assert e["syn_count"].dtype == np.int32 and e["pre_id"].dtype == np.int64
    assert set(erb["neuropil"]) == {"SMP(R)", "GNG", "ME(R)", "Optic-unspecified(R)",
                                    "CentralBrain-unspecified"}
    assert rep == {
        "neurons_in": 6, "neurons_kept": 4, "neurons_dropped": 2,
        "ascending_kept": 1, "sensory_ascending_kept": 0,
        "region_changed": 2, "region_ties": 1,
        "edgeroi_syn_brain": 21, "edgeroi_syn_brain_unkept": 0,
        "edgeroi_syn_nerve_cord": 29, "edgeroi_syn_no_location": 9,
        "edges_tau1": 4, "edges_ge5": 3, "synapses_tau1": 21}


def test_ascending_neuron_keeps_its_brain_part():
    # the point of correction 4: region in the nerve cord no longer drops the neuron
    neurons, er, rc = toy_malecns()
    n, *_ = malecns_brain_part(neurons, er, rc)
    assert 2 in set(n["neuron_id"])


def test_relative_weight_on_brain_edges_excludes_nerve_cord_input():
    neurons, er, rc = toy_malecns()
    _, e, _, _ = malecns_brain_part(neurons, er, rc)
    w = dict(zip(zip(e["pre_id"], e["post_id"]), relative_weight(e)))
    assert w[(2, 1)] == pytest.approx(3 / 8)       # neuron 1's brain input: 3 + 5
    assert w[(1, 2)] == pytest.approx(1.0)


def test_brain_part_refuses_bad_inputs():
    neurons, er, rc = toy_malecns()
    with pytest.raises(ValueError, match="1 synapse"):
        malecns_brain_part(neurons, er.assign(min_syn_fetched=5), rc)
    with pytest.raises(ValueError, match="'all'"):
        malecns_brain_part(neurons, er.assign(neuropil="all"), rc)
    bad_rc = pd.concat([rc, pd.DataFrame([(1, "CentralBrain", 30)], columns=rc.columns)])
    with pytest.raises(ValueError, match="unmapped"):        # non-primary ROI
        malecns_brain_part(neurons, er, bad_rc)
    with pytest.raises(ValueError, match="malecns"):
        malecns_brain_part(neurons.assign(dataset="fafb"), er, rc)


def test_brain_part_keeps_region_values_with_categorical_dtype():
    neurons, er, rc = toy_malecns()
    neurons["region"] = neurons["region"].astype("category")
    n, *_ = malecns_brain_part(neurons, er, rc)
    assert n["region"].notna().all() and "GNG" in set(n["region"])


def test_neuropil_location():
    names = pd.Series(["ME(R)", "LegNp(T2)(L)", "Optic-unspecified(L)", "VNC-unspecified",
                       "<unspecified>", "NotPrimary", "CA(R)", "CRN"])
    assert list(neuropil_location(names)) == ["brain", "nerve_cord", "brain", "nerve_cord",
                                              "none", "none", "brain", "nerve_cord"]


def test_nerve_cord_list_is_disjoint_from_brain_table():
    from eda_helpers import load_neuropil_table
    assert not set(MALECNS_NERVE_CORD) & set(load_neuropil_table().index)


# ---------- config.py agrees with the vocabularies here ----------

def test_config_matches_when_present():
    cfg_path = HERE.parent / "config.py"
    if not cfg_path.exists():
        pytest.skip("config.py not next to eda_shared/")
    spec = importlib.util.spec_from_file_location("config_under_test", cfg_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    assert getattr(cfg, "CONFIG_VERSION", None) == "0.2.0", "replace config.py with v0.2.0"
    assert cfg.SUPER_CLASS_VALUES == SUPER_CLASS_VALUES
    assert cfg.NT_VALUES == NT_VALUES
    for col in ("nt_type_corrected", "nt_correction_source", "super_class_source"):
        assert col in cfg.NEURON_SCHEMA
