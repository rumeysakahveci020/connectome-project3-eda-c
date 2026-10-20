"""The three male CNS 'unspecified' region names: how many neurons, under which rule, and what else they touch.

Reads the neuPrint answer saved by the build (data/raw/neuprint_malecns_v1.0_traced.pkl). No network.

    python tools/check_unspecified_regions.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
pkl = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/raw/neuprint_malecns_v1.0_traced.pkl"
NAMES = ["CentralBrain-unspecified", "Optic-unspecified(L)", "Optic-unspecified(R)"]
OPTIC = ("ME", "LO", "LOP", "AME", "LA")

neurons, roi, primary = pd.read_pickle(pkl)
roi = roi.rename(columns={"bodyId": "neuron_id", "roi": "region"}).copy()
roi["pre"] = pd.to_numeric(roi["pre"]).fillna(0)
roi["post"] = pd.to_numeric(roi["post"]).fillna(0)


def top_region(r, weight, tie, roi_set):
    d = r if roi_set == "all" else r[r["region"].isin(set(primary))]
    d = d.assign(syn={"pre+post": d["pre"] + d["post"], "pre": d["pre"], "post": d["post"]}[weight])
    d = d[d["syn"] > 0].sort_values(["neuron_id", "syn", "region"], ascending=[True, False, tie == "first"])
    return d.drop_duplicates("neuron_id").set_index("neuron_id")["region"]


print(f"{'rule':44s} " + " | ".join(f"{n[:26]:>26s}" for n in NAMES) + " | total")
for label, w, t, s in [("pre+post, primary ROIs, ties alphabetical (mine)", "pre+post", "first", "primary"),
                       ("pre+post, primary ROIs, ties reverse alphabetical", "pre+post", "last", "primary"),
                       ("pre only, primary ROIs", "pre", "first", "primary"),
                       ("post only, primary ROIs", "post", "first", "primary"),
                       ("pre+post, all ROIs in roiInfo", "pre+post", "first", "all")]:
    vc = top_region(roi, w, t, s).value_counts()
    counts = [int(vc.get(n, 0)) for n in NAMES]
    print(f"{label:44s} " + " | ".join(f"{c:>26,d}" for c in counts) + f" | {sum(counts):,}")

prim = roi[roi["region"].isin(set(primary))].assign(syn=lambda d: d["pre"] + d["post"])
top = top_region(roi, "pre+post", "first", "primary")
tot = prim.groupby("neuron_id")["syn"].sum()
print("\nFor the neurons whose top ROI is one of the three names (my rule):")
for n in NAMES:
    ids = top.index[top == n]
    sub = prim[prim["neuron_id"].isin(ids)]
    share = (sub[sub["region"] == n].set_index("neuron_id")["syn"] / tot.reindex(ids)).dropna()
    other = sub[sub["region"] != n].sort_values(["neuron_id", "syn"], ascending=[True, False]).drop_duplicates("neuron_id")
    nothing_else = int(len(ids) - other["neuron_id"].nunique())
    optic_second = float(other["region"].str.replace(r"\(.*\)", "", regex=True).isin(OPTIC).mean()) if len(other) else float("nan")
    print(f"\n{n}: {len(ids):,} neurons; median share of their synapses in this ROI {share.median():.0%}; "
          f"{nothing_else} have no other ROI")
    print(f"  second ROI is ME/LO/LOP/AME/LA for {optic_second:.0%} of those that have one; most common second ROIs:")
    print("  " + other["region"].value_counts().head(6).to_dict().__repr__())
