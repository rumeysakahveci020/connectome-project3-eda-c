"""Why does the FAFB neuron table have 139,262 rows (Step 0 expects 139,248; published 139,255)?

The build adds every id that appears in the edges but not in the annotation file. Step 0 says
7 ids are expected. This script lists the ones that were added and checks them against the list
of proofread root ids (proofread_root_ids_783.npy, 1.1 MB, from the same Zenodo record).

    python tools/check_fafb_ids.py [proofread_root_ids_783.npy] [annotations.tsv]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

npy = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data/raw/proofread_root_ids_783.npy")
tsv = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "data/raw/Supplemental_file1_neuron_annotations.tsv")
if not npy.exists():
    sys.exit(f"{npy} not found. Download https://zenodo.org/records/10676866/files/proofread_root_ids_783.npy?download=1")

neurons = pd.read_parquet(config.cache_path("fafb", "neurons", "783", f"ann{config.FAFB_ANNOTATIONS_TAG[1:]}"))
edges = pd.read_parquet(config.cache_path("fafb", "edges", "783", config.FAFB_SYNAPSE_TABLE, min_syn=1))
ann = set(pd.read_csv(tsv, sep="\t", dtype=str, usecols=["root_id"])["root_id"].astype("int64"))
proof = set(np.load(npy).astype("int64").tolist())
cache = set(neurons["neuron_id"].astype("int64"))
added = cache - ann
ann_not_proof = ann - proof
added_in_proof = added & proof
added_not_proof = added - proof

print(f"neuron table rows:                        {len(cache):,}")
print(f"annotation file ids:                      {len(ann):,}")
print(f"proofread root id list:                   {len(proof):,}")
print(f"added from edges (not in annotations):    {len(added):,}")
print(f"   of these in the proofread list:        {len(added_in_proof):,}   (the course expects 7)")
print(f"   of these NOT in the proofread list:    {len(added_not_proof):,}")
print(f"annotation ids not in the proofread list: {len(ann_not_proof):,}")
print(f"proofread ids missing from the annotations: {len(proof - ann):,}")
print(f"proofread ids missing from the neuron table: {len(proof - cache):,}")
for name, ids in [("added, in proofread list", added_in_proof), ("added, NOT in proofread list", added_not_proof)]:
    if ids:
        sub = edges[edges["pre_id"].isin(ids) | edges["post_id"].isin(ids)]
        syn = int(sub["syn_count"].sum())
        print(f"\n{name}: {sorted(ids)[:14]}\n   edges touching them: {len(sub):,}, synapses: {syn:,}")
