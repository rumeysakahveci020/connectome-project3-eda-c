"""Redraw only the C2 and C3 figures (colour consistency with C1, population in the C3 caption).
Does not touch the notebook, the report or any other figure.

    $env:EDA_UNMAPPED_REGION="unknown"
    python tools/redo_figures_c2_c3.py
"""
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from src import eda_c as C, eda_utils as U  # noqa: E402

CAP = ("FAFB 783 (annotations v3.2.0, Buhmann) and male CNS v1.0 (Traced, conf >= 0.5, brain part). "
       "Provisional flags: eda_shared v0.1.1.")
data = {d: U.load(d) for d in C.DS}
fig = C.c2_figure(data)
print("saved", U.save(fig, U.fig_name("C2", "both", "all"), CAP + " Population: all neurons with nt_confidence."))
hl = C.c3_hemilineages(data["fafb"][0])
fig = C.c3_figure(hl)
print("saved", U.save(fig, U.fig_name("C3", "fafb", "all"),
                      "FAFB 783. Population: neurons with a hemilineage and a fast transmitter (acetylcholine, GABA, glutamate), "
                      "hemilineages with >= 20 such neurons; no threshold. Reference line at 0.9."))
