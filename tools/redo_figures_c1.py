"""Redraw only the eight C1 figures (aligned rows, faded tiny rows, no duplicated optic row, clearer caption).
Does not touch the notebook, the report, the numbers or any other figure. Takes about a minute.

    python tools/redo_figures_c1.py
    python tools/refresh_notebook_images.py      # puts the new C1 images into the saved notebook too
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

CAP = "FAFB 783 (annotations v3.2.0, Buhmann) and male CNS v1.0 (Traced, conf >= 0.5, brain part). Provisional flags: eda_shared v0.1.1."
data = {d: U.load(d) for d in C.DS}
for pop in ("central", "optic"):
    for by in ("super_class", "region_group"):
        for weighted in (False, True):
            fig, _ = C.c1_figure(data, pop, by, weighted)
            tag = by + ("_weighted" if weighted else "")
            print("saved", U.save(fig, U.fig_name("C1", "both", pop, extra=tag), C.c1_caption(CAP, pop, weighted)))
