"""FAFB neurons without a region (UNASGD or no synapses at all): the list the group asked to note.

    python tools/check_unassigned.py
The group's note (15 Oct): UNASGD neurons get is_optic False and region_group unknown from eda_shared; in_brain is False for
them, but FAFB is not filtered by in_brain, so they stay in the data. Another student's list had 6 (3 ascending, JO-EV6,
JO-unclear and one R1-6 photoreceptor), another 9 (Codex table). This prints ours, with cell types, so the note can say which.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd  # noqa: E402

from src import eda_utils as U  # noqa: E402

n, e = U.load("fafb")
unk = n[n["region_group"].astype(str) == "unknown"]
print(f"FAFB neurons with region_group unknown: {len(unk):,} of {len(n):,}")
print("their region values:")
print(unk["region"].astype(str).value_counts(dropna=False).head(8).to_string())
connected = set(e["pre_id"]) | set(e["post_id"])
print(f"of the unknown group, {int((~unk['neuron_id'].isin(connected)).sum()):,} have no edge at all")
un = n[n["region"].astype(str).isin(["UNASGD", "None", "none"])]
print(f"\nneurons whose region is UNASGD: {len(un)}")
cols = [c for c in ("neuron_id", "cell_type", "cell_class", "super_class", "side", "region", "is_optic", "in_brain") if c in un.columns]
if len(un):
    print(un[cols].head(30).to_string(index=False))
    print("\nsuper_class of the UNASGD neurons:")
    print(un["super_class"].astype(str).value_counts().to_string())
    photo = un[un["cell_type"].astype(str).str.match(r"^R[1-8]", na=False) | un["cell_class"].astype(str).str.contains("photo", case=False, na=False)] if "cell_class" in un.columns else un.iloc[0:0]
    print(f"photoreceptor-like among them: {len(photo)}")
