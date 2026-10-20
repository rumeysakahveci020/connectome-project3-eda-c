"""Loading, populations, colours and figure naming shared by the C notebook.

No network calls: everything is read from data/cache (Data Conventions, "Caching").
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

import config
from eda_shared.eda_helpers import add_provisional_flags, basic_features, canonical_neuropil, load_neuropil_table
from src.fetch_malecns import NERVE_CORD_ROIS

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures" / "eda"
RELEASE = {"fafb": "783", "malecns": "v1.0"}
LABEL = {"fafb": "FAFB", "malecns": "MaleCNS"}
NT_ORDER = ["acetylcholine", "gaba", "glutamate", "dopamine", "serotonin", "octopamine", "histamine", "unknown"]
NT_COLORS = {"acetylcholine": "#1f77b4", "gaba": "#d62728", "glutamate": "#2ca02c", "dopamine": "#ff7f0e",
             "serotonin": "#9467bd", "octopamine": "#8c564b", "histamine": "#e377c2", "unknown": "#9e9e9e"}
SUPER_ORDER = ["sensory", "optic", "visual_projection", "central", "visual_centrifugal",
               "ascending", "descending", "motor", "endocrine", "unknown"]
GROUP_ORDER = ["sensory_periphery", "higher_order", "motor_periphery", "unknown"]
POPULATIONS = ("all", "central", "optic")


def _cache_dir() -> None:
    """EDA_CACHE_DIR only exists for dry runs on synthetic data; real runs use config.CACHE_DIR."""
    override = os.environ.get("EDA_CACHE_DIR")
    if override:
        config.CACHE_DIR = Path(override)


# Group decision (15 Oct): two rows for male CNS ROIs the provisional table does not have. They go in through
# add_provisional_flags(table=...) so that eda_shared stays unchanged; a new eda_shared with the same rows makes
# this addition a no-op.
GROUP_ROWS = pd.DataFrame(
    {"full_name": ["optic lobe, neuropil not specified", "central brain, neuropil not specified"],
     "is_optic": [True, False], "region_group": ["unknown", "unknown"],
     "status": ["group decision", "group decision"],
     "basis": ["Second ROI is ME/LO/LOP/AME/LA for 98-100% of these neurons",
               "Second ROI is central brain (GNG, SAD, SMP...), never optic"]},
    index=pd.Index(["Optic-unspecified", "CentralBrain-unspecified"], name="neuropil"))


def neuropil_table() -> pd.DataFrame:
    t = load_neuropil_table()
    return pd.concat([t, GROUP_ROWS.loc[~GROUP_ROWS.index.isin(t.index)]])


def flags(neurons: pd.DataFrame, nerve) -> pd.DataFrame:
    """add_provisional_flags, with one local escape hatch.

    The shared helper raises on a region name it does not know, and the rule is: report it to the
    group, never add it yourself. If you must look at the data before the group answers, set
    EDA_UNMAPPED_REGION=unknown: the rejected names become 'unknown' (so those neurons count as not in
    the brain) and every one is printed. Do not hand in results made this way without saying so."""
    table = neuropil_table()
    try:
        return add_provisional_flags(neurons, table=table, nerve_cord=nerve)
    except ValueError as exc:
        if os.environ.get("EDA_UNMAPPED_REGION") != "unknown":
            raise ValueError(f"{exc} Report these names to the group. For a private look only, set "
                             "EDA_UNMAPPED_REGION=unknown.") from exc
        counts = neurons["region"].astype("string").value_counts()
        bad = []
        for v in counts.index:
            try:
                canonical_neuropil(pd.Series([v], dtype="string"), table, nerve)
            except ValueError:
                bad.append(v)
        print("WARNING: region names rejected by eda_shared and treated as unknown (REPORT THEM):",
              {v: int(counts[v]) for v in bad})
        fixed = neurons.copy()
        fixed["region"] = fixed["region"].astype("string").where(~fixed["region"].isin(bad), "unknown")
        return add_provisional_flags(fixed, table=table, nerve_cord=nerve)


def load(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(neurons with provisional flags, edges at 1 synapse) for one dataset.

    malecns is restricted to its brain part: in_brain == True, and the edges among those neurons.
    """
    _cache_dir()
    rel = RELEASE[name]
    if name == "fafb":
        n_p = config.cache_path("fafb", "neurons", rel, f"ann{config.FAFB_ANNOTATIONS_TAG[1:]}")
        e_p = config.cache_path("fafb", "edges", rel, config.FAFB_SYNAPSE_TABLE, min_syn=1)
        nerve = frozenset()
    else:
        n_p = config.cache_path("malecns", "neurons", rel, "traced")
        e_p = config.cache_path("malecns", "edges", rel, "traced", min_syn=1, conf=config.MALECNS_SYN_CONFIDENCE)
        nerve = NERVE_CORD_ROIS
    neurons = flags(pd.read_parquet(n_p), nerve)
    edges = pd.read_parquet(e_p)
    if name == "malecns":
        neurons = neurons[neurons["in_brain"]]
        keep = set(neurons["neuron_id"])
        edges = edges[edges["pre_id"].isin(keep) & edges["post_id"].isin(keep)]
    return neurons.reset_index(drop=True), edges.reset_index(drop=True)


def population(neurons: pd.DataFrame, pop: str) -> pd.DataFrame:
    if pop == "all":
        return neurons
    if pop == "central":
        return neurons[~neurons["is_optic"]]
    if pop == "optic":
        return neurons[neurons["is_optic"]]
    raise ValueError(pop)


def fig_name(item: str, datasets: str, pop: str, tau: int | None = None, extra: str | None = None) -> str:
    rel = "783+v1.0" if datasets == "both" else RELEASE[datasets]
    parts = [item, datasets, rel, pop] + ([f"min{tau}"] if tau is not None else []) + ([extra] if extra else [])
    return "_".join(parts) + ".png"


def code_fingerprint() -> str:
    """Short hash of the code that draws the figures. Written into every PNG, so 'was this figure drawn by the current code?' can be
    answered exactly (file dates cannot: copying files from a zip keeps the old dates)."""
    h = hashlib.sha256()
    for f in ("eda_c.py", "eda_utils.py"):
        h.update((Path(__file__).resolve().parent / f).read_bytes().replace(b"\r\n", b"\n"))      # same hash on Windows and Linux line endings
    return h.hexdigest()[:16]


def save(fig, name: str, caption: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.text(0.01, -0.13, caption, fontsize=7, color="#444444", ha="left", va="top", wrap=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", metadata={"Comment": f"code_sha={code_fingerprint()}"})
    return path


def ordered(values, order):
    present = [v for v in order if v in set(values)]
    return present + sorted(set(values) - set(present))
