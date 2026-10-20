"""Why did the male CNS mushroom body subgraph change when the region rule changed?

The old cache used pre + post (release sites counted once), the group rule is upstream + downstream (every sender-receiver
contact counted). This puts the two rules side by side on the saved neuPrint answer (no network): which neurons are in the
mushroom body under one rule and not under the other, which cell types they are, and where they go.

    python tools/compare_region_rules.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd  # noqa: E402

from src import eda_c as C, eda_utils as U  # noqa: E402
from src.common import region_from_counts  # noqa: E402
from src.fetch_malecns import NERVE_CORD_ROIS, roi_connection_counts  # noqa: E402

pkl = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/raw/neuprint_malecns_v1.0_traced.pkl"
neurons, roi, primary = pd.read_pickle(pkl)

new_counts = roi_connection_counts(neurons, primary)                                  # upstream + downstream
old_counts = (roi.rename(columns={"bodyId": "neuron_id", "roi": "region"}).query("region in @primary")
              .assign(syn=lambda d: d["pre"] + d["post"])[["neuron_id", "region", "syn"]])
old_counts = old_counts[old_counts["syn"] > 0]
reg_new, ties_new = region_from_counts(new_counts)
reg_old, ties_old = region_from_counts(old_counts)
print(f"neurons with a region: new rule {len(reg_new):,} (ties {ties_new}), old rule {len(reg_old):,} (ties {ties_old})")


def mb_set(reg):
    df = pd.DataFrame({"neuron_id": reg.index.to_numpy(), "region": reg.to_numpy()})
    f = U.flags(df, NERVE_CORD_ROIS)
    return set(f.loc[f["neuropil"].isin(C.mb_neuropils()), "neuron_id"])


old_mb, new_mb = mb_set(reg_old), mb_set(reg_new)
left, entered = old_mb - new_mb, new_mb - old_mb
print(f"mushroom body by region: old rule {len(old_mb):,}, new rule {len(new_mb):,}; left {len(left):,}, entered {len(entered):,}")

info = neurons.set_index("bodyId")
cols = [c for c in ("type", "instance", "superclass") if c in info.columns]
for name, ids in (("LEFT the mushroom body under the new rule", left), ("ENTERED the mushroom body under the new rule", entered)):
    if not ids:
        continue
    sub = info.loc[[i for i in ids if i in info.index], cols].copy()
    sub["old region"] = [reg_old.get(i) for i in sub.index]
    sub["new region"] = [reg_new.get(i) for i in sub.index]
    print(f"\n{name}: {len(sub):,} neurons")
    print("  by type:", sub["type"].astype(str).str.replace(r"[_\-].*$", "", regex=True).value_counts().head(10).to_dict() if "type" in sub else "")
    print("  moves (old -> new):", (sub["old region"].astype(str) + " -> " + sub["new region"].astype(str)).value_counts().head(8).to_dict())
    # how much of the neuron's own signal is in the mushroom body under each rule
    ex = sub.index[:1]
print("\nReading: a neuron that receives most of its contacts in the lobes but sends many more contacts elsewhere (every partner of"
      "\na polyadic site counts under upstream + downstream) moves out under the new rule; outputs weigh more than before.")
