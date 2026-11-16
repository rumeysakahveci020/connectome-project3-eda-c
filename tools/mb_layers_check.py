"""Professor's point C: are the input (Kenyon cell) and output (MBON) layers complete under the region rule?

    python tools/mb_layers_check.py        # writes reports/mb_layers_check.md and reports/mb_layers_dropped.csv

Counts the Kenyon cells and MBONs the region rule keeps and drops, in both datasets, lists where the dropped units went (their new region),
and - if data/raw is there - tests the hypothesis from the Corrections: since the region rule counts OUTPUT synapses too, a neuron that sends
most of its signal outside the mushroom body can get another neuropil as its region. It prints, for kept and dropped units, the share of their
input sites (post) and of their output sites (pre) that lie in mushroom body neuropils.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import eda_c as C, eda_utils as U  # noqa: E402


def is_type(ct: pd.Series, prefix: str) -> pd.Series:
    return ct.astype("string").str.upper().str.startswith(prefix, na=False)


def mb_share(name: str, ids: set) -> pd.DataFrame | None:
    """Per neuron: share of its input sites (post) and output sites (pre) that lie in mushroom body neuropils (brain neuropils only)."""
    from eda_shared.corrections import fafb_neuropil_counts, malecns_neuropil_counts, neuropil_location
    from eda_shared.eda_helpers import MALECNS_NERVE_CORD, canonical_neuropil, load_neuropil_table
    raw = ROOT / "data" / "raw"
    table = load_neuropil_table()
    mb = set(C.mb_neuropils())
    try:
        if name == "fafb":
            pre = pd.read_feather(raw / "per_neuron_neuropil_count_pre_783.feather").rename(columns={"pre_pt_root_id": "neuron_id"})
            post = pd.read_feather(raw / "per_neuron_neuropil_count_post_783.feather").rename(columns={"post_pt_root_id": "neuron_id"})
            nerve = frozenset()
            sides = {"pre": pre[["neuron_id", "neuropil", "count"]], "post": post[["neuron_id", "neuropil", "count"]]}
        else:
            _, roi_df, primary = pd.read_pickle(raw / "neuprint_malecns_v1.0_traced.pkl")
            r = roi_df[roi_df["roi"].isin(set(primary))]
            nerve = MALECNS_NERVE_CORD
            sides = {s: r.rename(columns={"bodyId": "neuron_id", "roi": "neuropil", s: "count"})[["neuron_id", "neuropil", "count"]] for s in ("pre", "post")}
    except (FileNotFoundError, OSError, KeyError):
        return None
    out = {}
    for side, d in sides.items():
        d = d[d["neuron_id"].isin(ids)].copy()
        loc = neuropil_location(d["neuropil"].astype("string"), table, nerve).to_numpy()
        d = d[(loc == "brain") & (d["count"].to_numpy() > 0)]
        d["is_mb"] = canonical_neuropil(d["neuropil"].astype("string"), table, nerve).isin(mb).to_numpy()
        g = d.groupby("neuron_id")
        out[side] = (g.apply(lambda x: x.loc[x["is_mb"], "count"].sum() / x["count"].sum(), include_groups=False)).rename(f"{side}_mb_share")
    return pd.concat(out.values(), axis=1)


def main() -> int:
    U._cache_dir()
    lines = ["# Are the input and output layers complete? (region rule against MB cell types)", ""]
    dropped_all = []
    J = {}
    for d in C.DS:
        n, e = U.load(d)
        n = n.copy()
        R = C.subgraph_ids(n, "region")
        T = C.subgraph_ids(n, "cell_type")
        kc, mbon = set(n.loc[is_type(n["cell_type"], "KC"), "neuron_id"]), set(n.loc[is_type(n["cell_type"], "MBON"), "neuron_id"])
        lines += [f"## {U.LABEL[d]}", "",
                  "| | all neurons of the type | kept by the region rule | in the MB cell-type selection | region rule or cell type (union) |", "| --- | --- | --- | --- | --- |",
                  f"| Kenyon cells (type starts with KC) | {len(kc):,} | {len(kc & R):,} | {len(kc & T):,} | {len(kc & (R | T)):,} |",
                  f"| MBONs (type starts with MBON) | {len(mbon):,} | {len(mbon & R):,} | {len(mbon & T):,} | {len(mbon & (R | T)):,} |",
                  f"| all units | | {len(R):,} | {len(T):,} | {len(R | T):,} |", ""]
        miss = n[n["neuron_id"].isin(T - R)].copy()
        miss["kind"] = np.where(is_type(miss["cell_type"], "KC"), "KC", np.where(is_type(miss["cell_type"], "MBON"), "MBON", "other MB type"))
        miss["dataset"] = d
        lines += [f"MB-type neurons the region rule drops: {len(miss):,}. Where they went (their region under the shared rule):", "",
                  "| kind | region | neurons |", "| --- | --- | --- |"]
        for (k, r), c in miss.groupby(["kind", miss["neuropil"].astype(str)]).size().sort_values(ascending=False).head(18).items():
            lines.append(f"| {k} | {r} | {c:,} |")
        lines.append("")
        types = (n[n["neuron_id"].isin(mbon)].assign(kept=lambda x: x["neuron_id"].isin(R)).groupby("cell_type")["kept"].agg(["size", "sum"]))
        lines += ["MBON types: how many neurons per type the region rule keeps.", "", "| MBON type | neurons | kept |", "| --- | --- | --- |"]
        for t, r in types.sort_values("size", ascending=False).iterrows():
            lines.append(f"| {t} | {int(r['size'])} | {int(r['sum'])} |")
        lines.append("")
        J[d] = {"kc_all": len(kc), "kc_kept": len(kc & R), "mbon_all": len(mbon), "mbon_kept": len(mbon & R), "dropped_kc": len(kc - R), "dropped_mbon": len(mbon - R),
                "dropped_other_mb_types": int(len(miss) - len(kc - R) - len(mbon - R)), "union_units": len(R | T),
                "dropped_regions": {k: v.groupby(v["neuropil"].astype(str)).size().sort_values(ascending=False).head(4).to_dict() for k, v in miss.groupby("kind")}}
        share = mb_share(d, kc | mbon)
        if share is not None:
            sh = share.reindex(sorted(kc | mbon)).join(n.set_index("neuron_id")[["cell_type"]])
            sh["kind"] = np.where(is_type(sh["cell_type"], "KC"), "KC", "MBON")
            sh["kept"] = sh.index.isin(R)
            g = sh.groupby(["kind", "kept"])[["post_mb_share", "pre_mb_share"]].median().round(2).join(sh.groupby(["kind", "kept"]).size().rename("neurons"))
            for (k, kept), r in g.iterrows():
                J[d][f"{k.lower()}_{'kept' if kept else 'dropped'}_inputs_in_mb"] = float(r["post_mb_share"])
                J[d][f"{k.lower()}_{'kept' if kept else 'dropped'}_outputs_in_mb"] = float(r["pre_mb_share"])
            lines += ["Median share of input sites (post) and output sites (pre) inside mushroom body neuropils:", "", "| kind | kept by the region rule | neurons | inputs in MB | outputs in MB |", "| --- | --- | --- | --- | --- |"]
            for (k, kept), r in g.iterrows():
                lines.append(f"| {k} | {kept} | {int(r['neurons']):,} | {r['post_mb_share']} | {r['pre_mb_share']} |")
            lines.append("")
        else:
            lines += ["(no data/raw: the share of synapses inside the mushroom body was not computed)", ""]
        dropped_all.append(miss[["dataset", "neuron_id", "cell_type", "kind", "region", "neuropil"]])
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "mb_layers_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import json
    (ROOT / "reports" / "mb_layers.json").write_text(json.dumps(J, indent=1), encoding="utf-8")
    pd.concat(dropped_all).to_csv(ROOT / "reports" / "mb_layers_dropped.csv", index=False)
    print("\n".join(lines))
    print("wrote reports/mb_layers_check.md, reports/mb_layers_dropped.csv and reports/mb_layers.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
