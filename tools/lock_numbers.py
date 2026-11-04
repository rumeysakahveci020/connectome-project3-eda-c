"""Every number the Project 3 lock note cites, recomputed on the CORRECTED tables (Lock Note Assignment: "recompute the EDA
numbers from the corrected tables"). No model is involved: only the structure and labels of the data.

    python tools/lock_numbers.py            # writes reports/lock_numbers.json and reports/lock_numbers.md

Read-only on data/cache. The tie numbers also need data/raw (the FlyWire count files and the saved neuPrint answer); without them that
block is skipped and the note says so.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from src import eda_c as C, eda_utils as U  # noqa: E402

TAUS = (1, 3, 5, 10)
FAST = {"acetylcholine", "gaba", "glutamate"}
MODULATORY = {"dopamine", "serotonin", "octopamine"}


def is_type(ct: pd.Series, prefix: str) -> pd.Series:
    return ct.astype("string").str.upper().str.startswith(prefix, na=False)


def tau_block(n: pd.DataFrame, e: pd.DataFrame, ids: set) -> dict:
    nt = n.set_index("neuron_id")[U.NT_COL].astype(str)
    ct = n.set_index("neuron_id")["cell_type"]
    out = {}
    e1 = C.sub_edges(e, ids, 1)
    kc_mbon_1 = int((is_type(ct.reindex(e1["pre_id"]).reset_index(drop=True), "KC").to_numpy() & is_type(ct.reindex(e1["post_id"]).reset_index(drop=True), "MBON").to_numpy()).sum())
    for tau in TAUS:
        s = C.sub_edges(e, ids, tau)
        out_d = s.groupby("pre_id").size().reindex(list(ids), fill_value=0)
        in_d = s.groupby("post_id").size().reindex(list(ids), fill_value=0)
        pre_nt = nt.reindex(s["pre_id"]).fillna("unknown").to_numpy()
        kcm = int((is_type(ct.reindex(s["pre_id"]).reset_index(drop=True), "KC").to_numpy() & is_type(ct.reindex(s["post_id"]).reset_index(drop=True), "MBON").to_numpy()).sum())
        out[str(tau)] = {
            "neurons": len(ids), "edges": int(len(s)), "edges_per_neuron": float(len(s) / max(len(ids), 1)),
            "density": float(len(s) / (len(ids) * (len(ids) - 1))) if len(ids) > 1 else None,
            "isolated": int(((in_d == 0) & (out_d == 0)).sum()), "no_input": int((in_d == 0).sum()), "no_output": int((out_d == 0).sum()),
            "median_in": float(in_d.median()), "median_out": float(out_d.median()),
            "edges_from_fast_signed": float(np.isin(pre_nt, list(FAST)).mean()) if len(s) else None,
            "edges_from_modulatory": float(np.isin(pre_nt, list(MODULATORY)).mean()) if len(s) else None,
            "edges_from_unknown": float((pre_nt == "unknown").mean()) if len(s) else None,
            "kc_to_mbon_edges": kcm, "kc_to_mbon_kept_vs_tau1": float(kcm / kc_mbon_1) if kc_mbon_1 else None,
            "median_syn_per_edge": float(s["syn_count"].median()) if len(s) else None,
            "synapses": int(s["syn_count"].sum())}
    return out


def tie_block(name: str, n: pd.DataFrame) -> dict | None:
    """Region ties among the neurons of the cache: how many, and how many would change the mushroom-body selection under another tie rule."""
    from eda_shared.corrections import fafb_neuropil_counts, malecns_neuropil_counts, neuropil_location
    from eda_shared.eda_helpers import MALECNS_NERVE_CORD, canonical_neuropil, load_neuropil_table
    raw = ROOT / "data" / "raw"
    table = load_neuropil_table()
    try:
        if name == "fafb":
            pre = pd.read_feather(raw / "per_neuron_neuropil_count_pre_783.feather"); post = pd.read_feather(raw / "per_neuron_neuropil_count_post_783.feather")
            counts, nerve = fafb_neuropil_counts(pre, post), frozenset()
        else:
            _, roi_df, primary = pd.read_pickle(raw / "neuprint_malecns_v1.0_traced.pkl")
            counts, nerve = malecns_neuropil_counts(roi_df[roi_df["roi"].isin(set(primary))]), MALECNS_NERVE_CORD
    except (FileNotFoundError, OSError):
        return None
    counts = counts[counts["neuron_id"].isin(set(n["neuron_id"]))]
    loc = neuropil_location(counts["neuropil"].astype("string"), table, nerve).to_numpy()
    c = counts[(loc == "brain") & (counts["count"].to_numpy() > 0)].groupby(["neuron_id", "neuropil"], as_index=False)["count"].sum()
    c["neuropil"] = c["neuropil"].astype(str)
    c["mx"] = c.groupby("neuron_id")["count"].transform("max")
    top = c[c["count"] == c["mx"]].copy()
    top["is_mb"] = canonical_neuropil(top["neuropil"].astype("string"), table, nerve).isin(C.mb_neuropils()).to_numpy()
    top = top.sort_values(["neuron_id", "neuropil"], kind="mergesort")
    g = top.groupby("neuron_id")
    r = pd.DataFrame({"n_tied": g.size(), "first_is_mb": g["is_mb"].first(), "any_mb": g["is_mb"].any(), "all_mb": g["is_mb"].all()})
    tied = r["n_tied"] > 1
    return {"neurons_with_a_tie": int(tied.sum()), "tied_and_selected_now": int((tied & r["first_is_mb"]).sum()),
            "tied_selected_but_could_leave": int((tied & r["first_is_mb"] & ~r["all_mb"]).sum()),
            "tied_not_selected_but_could_join": int((tied & ~r["first_is_mb"] & r["any_mb"]).sum()),
            "selected_total": int(r["first_is_mb"].sum())}


def main() -> int:
    U._cache_dir()
    out: dict = {"nt_col": U.NT_COL, "taus": list(TAUS)}
    data = {d: U.load(d) for d in C.DS}
    for d, (n, e) in data.items():
        ids = C.subgraph_ids(n, "region")
        ids_ct = C.subgraph_ids(n, "cell_type")
        mb = n[n["neuron_id"].isin(ids)]
        nt = mb[U.NT_COL].astype(str)
        res = {"mb_neurons": len(ids), "mb_neurons_cell_type_selection": len(ids_ct), "overlap_region_and_cell_type": len(ids & ids_ct),
               "mb_is_optic": int(mb["is_optic"].sum()), "mb_super_class": mb["super_class"].astype(str).value_counts().to_dict(),
               "nt_counts": nt.value_counts().to_dict(), "nt_shares": (nt.value_counts(normalize=True)).round(4).to_dict(),
               "source_counts": mb["nt_correction_source"].astype(str).value_counts().to_dict(),
               "unknown_label_share": float((nt == "unknown").mean()),
               "median_confidence_predicted": float(mb.loc[mb["nt_correction_source"].astype(str) == "predicted", "nt_confidence"].median()),
               "kc_neurons": int(is_type(mb["cell_type"], "KC").sum()), "mbon_neurons": int(is_type(mb["cell_type"], "MBON").sum()),
               "tau": tau_block(n, e, ids)}
        # unit signs under the two glutamate rules (units, not edges)
        res["signs"] = {"rule1_glutamate_inhibitory": {"plus": int((nt == "acetylcholine").sum()), "minus": int(nt.isin(["gaba", "glutamate"]).sum()),
                                                       "free": int((~nt.isin(list(FAST))).sum())},
                        "rule2_glutamate_excitatory": {"plus": int(nt.isin(["acetylcholine", "glutamate"]).sum()), "minus": int((nt == "gaba").sum()),
                                                       "free": int((~nt.isin(list(FAST))).sum())}}
        res["type_mix"] = {"types": 0}
        mix = C.mb_type_mix(n)
        res["type_mix"] = {"types": int(len(mix)), "mixed_types": int((mix["top_share"] < 0.9).sum()), "neurons_in_mixed_types": int(mix.loc[mix["top_share"] < 0.9, "neurons"].sum())}
        res["ties"] = tie_block(d, n)
        out[d] = res
    net = C.c4_net_index(*data["fafb"], tau=10)
    top12 = net.drop_duplicates("neuron_id")["neuropil"].value_counts().head(12).index
    sc = C.c4_sign_changes(net[net["neuropil"].isin(top12)], "neuropil")
    r1, r2 = list(C.RULES)
    shift = (sc[r2] - sc[r1]).abs()
    out["c4_corrected"] = {"neuropils": int(len(sc)), "flip": int(sc["changes_sign"].sum()), "flip_names": list(sc.index[sc["changes_sign"]]),
                           "shifted_025": int((shift >= 0.25).sum()), "shift_hi": float(shift.max())}
    f, m = out["fafb"]["tau"], out["malecns"]["tau"]
    out["cross_dataset"] = {"median_syn_per_edge_tau1": {"fafb": f["1"]["median_syn_per_edge"], "malecns": m["1"]["median_syn_per_edge"]},
                            "edges_per_neuron": {t: {"fafb": f[t]["edges_per_neuron"], "malecns": m[t]["edges_per_neuron"]} for t in f}}
    target = f["5"]["edges_per_neuron"]
    # the male CNS threshold whose edges per neuron is closest to FlyWire's at tau = 5 (a density-matched check, found by scanning)
    ids_m = C.subgraph_ids(data["malecns"][0], "region")
    scan = {t: len(C.sub_edges(data["malecns"][1], ids_m, t)) / max(len(ids_m), 1) for t in range(1, 41)}
    out["cross_dataset"]["tau_malecns_matching_fafb_tau5_density"] = int(min(scan, key=lambda t: abs(scan[t] - target)))
    out["cross_dataset"]["fafb_edges_per_neuron_tau5"] = target
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "lock_numbers.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    lines = ["# Numbers for the Project 3 lock note (corrected tables)", ""]
    for d in C.DS:
        r = out[d]
        lines += [f"## {U.LABEL[d]}: {r['mb_neurons']:,} mushroom body neurons (region); cell-type selection {r['mb_neurons_cell_type_selection']:,}; overlap {r['overlap_region_and_cell_type']:,}; is_optic True: {r['mb_is_optic']}",
                  f"- labels (nt_type_corrected): {r['nt_counts']}; source: {r['source_counts']}; unknown {r['unknown_label_share']:.1%}; KC {r['kc_neurons']}, MBON {r['mbon_neurons']}",
                  f"- signs, units: rule 1 {r['signs']['rule1_glutamate_inhibitory']}, rule 2 {r['signs']['rule2_glutamate_excitatory']}",
                  f"- type mix: {r['type_mix']}; ties: {r['ties']}", "", "| tau | edges | per neuron | isolated | no input | no output | median in/out | fast-signed source | KC to MBON kept |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for t, v in r["tau"].items():
            kk = "n/a" if v["kc_to_mbon_kept_vs_tau1"] is None else f"{v['kc_to_mbon_kept_vs_tau1']:.0%} ({v['kc_to_mbon_edges']:,})"
            fs = "n/a" if v["edges_from_fast_signed"] is None else f"{v['edges_from_fast_signed']:.0%}"
            lines.append(f"| {t} | {v['edges']:,} | {v['edges_per_neuron']:.2f} | {v['isolated']:,} | {v['no_input']:,} | {v['no_output']:,} | {v['median_in']:.0f}/{v['median_out']:.0f} | {fs} | {kk} |")
        lines.append("")
    lines += [f"C4 on the corrected labels: {out['c4_corrected']}", f"Cross-dataset: {out['cross_dataset']}"]
    (ROOT / "reports" / "lock_numbers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
