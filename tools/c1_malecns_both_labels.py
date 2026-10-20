"""Male CNS transmitter composition under predictedNt and under consensusNt, side by side.

    $env:EDA_UNMAPPED_REGION="unknown"      (same setting as for the notebook)
    python tools/c1_malecns_both_labels.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd  # noqa: E402

import config  # noqa: E402
from src import eda_utils as U  # noqa: E402

neurons, _ = U.load("malecns")
cons = pd.read_parquet(config.CACHE_DIR / "malecns_nt_consensus_v1.0.parquet")
m = neurons.merge(cons, on="neuron_id", how="left")
m["consensus"] = m["consensusNt"].astype("string").str.lower().replace({"unclear": "unknown"}).fillna("unknown")
cols = {}
for pop, d in (("central", m[~m["is_optic"]]), ("optic", m[m["is_optic"]])):
    cols[(pop, "predictedNt")] = d["nt_type"].astype(str).value_counts(normalize=True)
    cols[(pop, "consensusNt")] = d["consensus"].value_counts(normalize=True)
    print(f"{pop}: {len(d):,} neurons")
print((pd.DataFrame(cols).reindex(U.NT_ORDER).fillna(0) * 100).round(1).to_string())
