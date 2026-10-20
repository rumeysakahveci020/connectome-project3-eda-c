"""Does the saved neuPrint answer have upstream/downstream in roiInfo, and what are the three unspecified counts under the group rule?

    python tools/check_roiinfo_rule.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.fetch_malecns import roi_connection_counts  # noqa: E402
from src.common import region_from_counts  # noqa: E402

pkl = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/raw/neuprint_malecns_v1.0_traced.pkl"
neurons, roi, primary = pd.read_pickle(pkl)
print("roiInfo column present:", "roiInfo" in neurons.columns)
ex = next((d for d in neurons["roiInfo"] if isinstance(d, dict) and d), None) if "roiInfo" in neurons.columns else None
if ex is None:
    sys.exit("roiInfo is empty or not a dict; send me the type: " + str(type(neurons["roiInfo"].iloc[0])))
k, v = next(iter(ex.items()))
print("example ROI entry:", k, v)
conn = roi_connection_counts(neurons, primary)
if conn is None:
    sys.exit("roiInfo has no upstream/downstream fields in this answer.")
reg, ties = region_from_counts(conn)
vc = reg.value_counts()
names = ["CentralBrain-unspecified", "Optic-unspecified(L)", "Optic-unspecified(R)"]
counts = [int(vc.get(n, 0)) for n in names]
print(f"\nunder upstream + downstream (ties alphabetical: {ties} neurons tie):")
for n, c in zip(names, counts):
    print(f"  {n:26s} {c:>5,d}")
print(f"  total                      {sum(counts):>5,d}   (other students: 761, 750 and 745; the difference comes from the source table and the label spelling)")
