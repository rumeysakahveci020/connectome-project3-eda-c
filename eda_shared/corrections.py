"""Week 4 Shared Data Corrections, as code (v0.2.0).

One function per correction. Each adds columns or builds a table; none edits
an original column except `region`, which is recomputed by the shared rule.

  1  add_nt_type_corrected      nt_type_corrected, nt_correction_source
  2  map_super_class            super_class (shared levels), super_class_source
  3  align_to_proofread         FAFB neuron set = the 783 proofread ids
  4  malecns_brain_part         neurons, edges and edgeroi of the brain part
  5  (eda_helpers.relative_weight, called on the brain-part edges)
  6  region_from_counts         the region rule, both datasets, all synapses
     (+ fafb_neuropil_counts, malecns_neuropil_counts to build its input)

Order when building the tables:
  fafb     map_super_class -> align_to_proofread -> region_from_counts
           -> add_nt_type_corrected
  malecns  map_super_class -> add_nt_type_corrected -> malecns_brain_part
           (which applies region_from_counts)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

try:                                    # imported as eda_shared.corrections
    from .eda_helpers import MALECNS_NERVE_CORD, canonical_neuropil, load_neuropil_table
except ImportError:                     # imported from inside the folder (tests)
    from eda_helpers import MALECNS_NERVE_CORD, canonical_neuropil, load_neuropil_table

# Vocabularies. They equal config.py v0.2.0; tests/test_corrections.py checks
# this whenever config.py is importable.
NT_VALUES = ["acetylcholine", "glutamate", "gaba", "dopamine",
             "serotonin", "octopamine", "histamine", "unknown"]
SUPER_CLASS_VALUES = ["sensory", "ascending", "descending", "motor", "endocrine",
                      "visual_projection", "visual_centrifugal", "optic", "central",
                      "sensory_ascending", "efferent", "vnc_intrinsic",
                      "unknown"]
NT_CORRECTION_SOURCES = ["known_nt", "kc_override", "predicted"]


# ---------------------------------------------------------------- correction 1

# Small-molecule transmitters that can appear in FlyWire's known_nt. Tyramine
# and glycine are not in NT_VALUES, but a neuron that lists one of them next
# to another transmitter names two transmitters, so it is not corrected.
_TRANSMITTERS = frozenset(NT_VALUES[:-1]) | {"tyramine", "glycine"}


def parse_known_nt(value) -> frozenset[str]:
    """Small-molecule transmitters asserted in one known_nt string.

    known_nt joins evidence with ';' and ',' (for example
    "acetylcholine; sNPF; acetylcholine, sNPF"). Peptides, nitric oxide and
    "<x>-negative" entries are not transmitter assertions and are ignored.
    """
    if value is None or pd.isna(value):
        return frozenset()
    tokens = (t.strip() for t in re.split(r"[;,]", str(value).lower()))
    return frozenset(t for t in tokens if t in _TRANSMITTERS)


def single_known_nt(value) -> str | None:
    """The transmitter if known_nt names exactly one and it is in NT_VALUES."""
    found = parse_known_nt(value)
    if len(found) == 1:
        (nt,) = found
        if nt in NT_VALUES:
            return nt
    return None


def is_kenyon_cell(cell_type: pd.Series) -> pd.Series:
    """Kenyon-cell types: cell_type starts with "KC". Print kenyon_cell_types()
    once per dataset and confirm the list holds only Kenyon cells."""
    return cell_type.astype("string").str.startswith("KC").fillna(False).astype(bool)


def kenyon_cell_types(neurons: pd.DataFrame) -> pd.DataFrame:
    """Neuron counts per Kenyon-cell type, per dataset, for the README check."""
    kc = neurons[is_kenyon_cell(neurons["cell_type"])]
    return (kc.groupby(["dataset", "cell_type"], observed=True).size()
            .rename("neurons").reset_index())


def add_nt_type_corrected(neurons: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with nt_type_corrected and nt_correction_source.

    fafb     nt_known where it names a single transmitter  -> "known_nt";
             otherwise nt_type                              -> "predicted"
    malecns  acetylcholine for Kenyon-cell types            -> "kc_override";
             otherwise nt_type                              -> "predicted"

    nt_type and nt_confidence are not changed. MaleCNS nt_known, where a
    repository filled it, is not used: the agreed MaleCNS rule is the
    Kenyon-cell override only. Use nt_type_corrected for any analysis.
    """
    need = {"dataset", "nt_type", "cell_type"} - set(neurons.columns)
    if need:
        raise ValueError(f"neurons: missing columns {sorted(need)}")
    datasets = set(neurons["dataset"].astype(str).unique())
    if datasets - {"fafb", "malecns"}:
        raise ValueError(f"add_nt_type_corrected: no rule for {sorted(datasets - {'fafb', 'malecns'})}")
    if "fafb" in datasets and "nt_known" not in neurons:
        raise ValueError("neurons: fafb needs the nt_known column")

    out = neurons.copy()
    # positional arrays, so a duplicated index cannot misplace a value
    corrected = out["nt_type"].astype("string").fillna("unknown").to_numpy(dtype=object)
    source = np.full(len(out), "predicted", dtype=object)
    dataset = out["dataset"].astype(str).to_numpy()

    fafb = dataset == "fafb"
    if fafb.any():
        known = np.array([single_known_nt(v) for v in out["nt_known"].to_numpy(dtype=object)],
                         dtype=object)
        hit = fafb & pd.notna(known)
        corrected[hit] = known[hit]
        source[hit] = "known_nt"

    malecns = dataset == "malecns"
    if malecns.any():
        kc = is_kenyon_cell(out["cell_type"]).to_numpy() & malecns
        corrected[kc] = "acetylcholine"
        source[kc] = "kc_override"
    corrected = pd.Series(corrected, index=out.index, dtype="string")
    source = pd.Series(source, index=out.index, dtype="string")

    nt_dtype = (out["nt_type"].dtype if isinstance(out["nt_type"].dtype, pd.CategoricalDtype)
                else pd.CategoricalDtype(NT_VALUES))
    bad = sorted(set(corrected.unique()) - set(nt_dtype.categories))
    if bad:
        raise ValueError(f"nt_type_corrected: values outside the vocabulary {bad}")
    out["nt_type_corrected"] = corrected.astype(nt_dtype)
    out["nt_correction_source"] = source.astype(pd.CategoricalDtype(NT_CORRECTION_SOURCES))
    return out


def nt_correction_summary(neurons: pd.DataFrame) -> pd.DataFrame:
    """Step 0 row "Transmitter correction": rows per dataset and source, and
    how many of them changed value (nt_type_corrected != nt_type)."""
    changed = (neurons["nt_type_corrected"].astype(str)
               != neurons["nt_type"].astype("string").fillna("unknown").astype(str))
    g = neurons.assign(changed=changed).groupby(["dataset", "nt_correction_source"], observed=True)
    return g.agg(rows=("changed", "size"), changed=("changed", "sum")).reset_index()


# ---------------------------------------------------------------- correction 2

FAFB_SUPER_CLASS = {v: v for v in ["sensory", "ascending", "descending", "motor", "endocrine",
                                   "visual_projection", "visual_centrifugal", "optic",
                                   "central", "sensory_ascending", "unknown"]}

# MaleCNS v1.0 `superclass` -> shared level. *_tbc ("to be confirmed") goes to
# its base class. Levels with no FlyWire counterpart: vnc_intrinsic and the
# efferent family get their own levels; ENS (enteric) and vnc_tbc (no base
# class) become unknown. super_class_source keeps the raw value in every case.
MALECNS_SUPER_CLASS = {
    "ol_intrinsic": "optic",
    "cb_intrinsic": "central",
    "vnc_intrinsic": "vnc_intrinsic",
    "visual_projection": "visual_projection", "visual_projection_tbc": "visual_projection",
    "visual_centrifugal": "visual_centrifugal",
    "ol_sensory": "sensory", "cb_sensory": "sensory", "vnc_sensory": "sensory",
    "cb_sensory_tbc": "sensory", "vnc_sensory_tbc": "sensory",
    "sensory_descending": "sensory",
    "sensory_ascending": "sensory_ascending", "sensory_ascending_tbc": "sensory_ascending",
    "ascending_neuron": "ascending",
    "descending_neuron": "descending",
    "cb_motor": "motor", "vnc_motor": "motor",
    "cb_efferent": "efferent", "vnc_efferent": "efferent",
    "efferent_ascending": "efferent", "efferent_descending": "efferent",
    "cb_endocrine": "endocrine", "vnc_endocrine": "endocrine",
    "ENS": "unknown",
    "vnc_tbc": "unknown",
    "unknown": "unknown",
}


def map_super_class(raw: pd.Series, dataset: str) -> pd.DataFrame:
    """Map a source's raw superclass column to the shared levels.

    raw      FAFB `super_class` or MaleCNS `superclass`, exactly as fetched
    returns  DataFrame (same index) with
               super_class         categorical, SUPER_CLASS_VALUES
               super_class_source  the raw value as a string (missing stays missing)
    Missing raw values become unknown. A raw value with no mapping raises.
    A project that needs fewer classes merges levels in its own code; merging
    sensory_ascending into ascending matches Schlegel et al. (2024).
    """
    table = {"fafb": FAFB_SUPER_CLASS, "malecns": MALECNS_SUPER_CLASS}.get(dataset)
    if table is None:
        raise ValueError(f"map_super_class: no mapping for dataset {dataset!r}")
    src = raw.astype("string")
    present = set(src.dropna().unique())
    bad = sorted(present - set(table))
    if bad:
        raise ValueError(f"super_class ({dataset}): unmapped values {bad}. "
                         "Report them to the group; do not map them privately.")
    mapped = src.map(table, na_action="ignore").fillna("unknown")
    return pd.DataFrame({
        "super_class": mapped.astype(pd.CategoricalDtype(SUPER_CLASS_VALUES)),
        "super_class_source": src,
    }, index=raw.index)


# ---------------------------------------------------------------- correction 3

def _unknown_rows(neurons: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    """Rows for `ids` with the columns and dtypes of `neurons`: dataset and
    release copied; "unknown" in categorical columns that have that level and
    in `region` (a placeholder: compute region afterwards); missing values
    elsewhere. A column that can hold neither raises, naming it."""
    n = int(ids.size)
    cols = {}
    for col in neurons.columns:
        dtype = neurons[col].dtype
        if col == "neuron_id":
            cols[col] = pd.Series(ids, dtype=dtype)
        elif col in ("dataset", "release"):
            cols[col] = pd.Series([neurons[col].iloc[0]] * n, dtype=dtype)
        elif isinstance(dtype, pd.CategoricalDtype):
            if "unknown" not in dtype.categories:
                raise ValueError(f"column {col!r} has no 'unknown' level; build the neuron "
                                 "set before adding this column")
            cols[col] = pd.Series(["unknown"] * n, dtype=dtype)
        elif col == "region":
            cols[col] = pd.Series(["unknown"] * n, dtype=dtype)
        elif pd.api.types.is_float_dtype(dtype):
            cols[col] = pd.Series([np.nan] * n, dtype=dtype)
        else:
            try:
                cols[col] = pd.Series([pd.NA] * n, dtype=dtype)
            except (TypeError, ValueError) as err:
                raise ValueError(f"column {col!r} ({dtype}) cannot hold a missing value; "
                                 "build the neuron set before adding derived columns") from err
    return pd.DataFrame(cols)


def align_to_proofread(neurons: pd.DataFrame, proofread_ids) -> tuple[pd.DataFrame, dict]:
    """FAFB neuron set = the 783 proofread ids (Week 4 correction 3).

    neurons        one row per row of the annotation file (v3.2.0)
    proofread_ids  FlyWire's proofread_root_ids_783.npy (139,255 ids)

    Annotation rows whose id is not a 783 proofread id are dropped (v3.2.0
    has 7: newer root ids with no edges in 783). Proofread ids with no
    annotation row are added with unknown labels (v3.2.0 lacks 14; all of
    them have edges). Returns (neurons, report). Call it after
    map_super_class and before region_from_counts, add_nt_type_corrected
    and any derived column.
    """
    if "dataset" in neurons and set(neurons["dataset"].astype(str).unique()) - {"fafb"}:
        raise ValueError("align_to_proofread: fafb rows only")
    pr = np.unique(np.asarray(proofread_ids, dtype=np.int64))
    ids = neurons["neuron_id"].to_numpy(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("neurons: duplicated neuron_id")
    inside = np.isin(ids, pr)
    kept = neurons[inside]
    missing = np.setdiff1d(pr, ids)
    out = pd.concat([kept, _unknown_rows(neurons, missing)], ignore_index=True) if missing.size \
        else kept.reset_index(drop=True)
    report = {"annotation_rows": int(len(neurons)),
              "dropped_not_in_release": int((~inside).sum()),
              "added_unknown": int(missing.size),
              "total": int(len(out))}
    return out, report


# ---------------------------------------------------------- corrections 4 and 6

def neuropil_location(names: pd.Series, table: pd.DataFrame | None = None,
                      nerve_cord=MALECNS_NERVE_CORD) -> pd.Series:
    """Classify neuropil names as "brain", "nerve_cord" or "none".

    brain       every name in the neuropil table (after removing the side
                marker and applying ALIASES), including CentralBrain-unspecified
                and Optic-unspecified
    nerve_cord  every name in `nerve_cord` (pass frozenset() for FAFB)
    none        a synapse outside every neuropil (UNASGD, None, <unspecified>,
                NotPrimary, unknown): its position is not known, so it counts as
                neither brain nor nerve cord
    Any other name raises (for example a non-primary neuPrint ROI such as
    "CentralBrain" or "OL(R)", which would double-count synapses). Missing
    values raise too. Works on the distinct values, so it is fast on large
    tables. Returns a categorical Series with the same index.
    """
    table = load_neuropil_table() if table is None else table
    codes, uniques = pd.factorize(names.astype("string"))       # missing -> -1
    if (codes < 0).any():
        raise ValueError("neuropil: missing values; spell them UNASGD or <unspecified>")
    canon = canonical_neuropil(pd.Series(uniques, dtype="string"), table, nerve_cord)
    levels = ["brain", "nerve_cord", "none"]
    loc_of_unique = np.where(canon == "nerve_cord", 1, np.where(canon == "unknown", 2, 0))
    return pd.Series(pd.Categorical.from_codes(loc_of_unique[codes], levels), index=names.index)


def _pick(df: pd.DataFrame, *names: str) -> str:
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    raise ValueError(f"expected one of the columns {list(names)}, found {list(df.columns)}")


def fafb_neuropil_counts(pre: pd.DataFrame, post: pd.DataFrame) -> pd.DataFrame:
    """FAFB counts for the region rule, from FlyWire's
    per_neuron_neuropil_count_pre_783 and _post_783 files (Zenodo 10676866).
    They count every synapse, proofread partner or not, so inputs plus
    outputs per neuron and neuropil follow the Week 2 rule "from all
    synapses". Returns neuron_id, neuropil, count (FlyWire neuropil names;
    a missing neuropil label becomes UNASGD)."""
    parts = []
    for df, idc in ((pre, ("pre_pt_root_id", "root_id", "neuron_id")),
                    (post, ("post_pt_root_id", "root_id", "neuron_id"))):
        parts.append(pd.DataFrame({
            "neuron_id": df[_pick(df, *idc)].astype("int64").to_numpy(),
            "neuropil": df[_pick(df, "neuropil")].astype("string").fillna("UNASGD").to_numpy(),
            "count": df[_pick(df, "count", "syn_count")].astype("int64").to_numpy()}))
    both = pd.concat(parts, ignore_index=True)
    return both.groupby(["neuron_id", "neuropil"], observed=True, sort=False)["count"].sum().reset_index()


def malecns_neuropil_counts(roi_counts_df: pd.DataFrame) -> pd.DataFrame:
    """MaleCNS counts for the region rule, from neuPrint: the per-ROI table of
    neuprint.fetch_neurons (columns bodyId, roi, upstream, downstream),
    restricted to PRIMARY ROIs (neuprint.fetch_primary_rois()).
    count = upstream + downstream: connections in, plus connections out, with
    all partners. Returns neuron_id, neuropil, count. A non-primary ROI left in
    the table raises later, in region_from_counts."""
    r = roi_counts_df
    return pd.DataFrame({
        "neuron_id": r[_pick(r, "bodyId", "neuron_id")].astype("int64").to_numpy(),
        "neuropil": r[_pick(r, "roi", "neuropil")].astype("string").fillna("<unspecified>").to_numpy(),
        "count": (r[_pick(r, "upstream")].fillna(0).astype("int64")
                  + r[_pick(r, "downstream")].fillna(0).astype("int64")).to_numpy(),
    })


def region_from_counts(neuron_ids, counts: pd.DataFrame, table: pd.DataFrame | None = None,
                       nerve_cord=frozenset()) -> pd.DataFrame:
    """The shared region rule, for both datasets (Week 4 corrections 4 and 6).

    neuron_ids  the neurons to label
    counts      neuron_id, neuropil, count: inputs plus outputs over ALL partners
                (fafb_neuropil_counts / malecns_neuropil_counts)
    nerve_cord  frozenset() for FAFB; MALECNS_NERVE_CORD for MaleCNS

    region = the brain neuropil with the largest count. Not candidates: nerve-
    cord neuropils and synapses outside every neuropil (UNASGD, None,
    <unspecified>, NotPrimary). Ties go to the alphabetically first name
    (string order, whatever the column's dtype). A neuron with no candidate
    gets "unknown". The tie rule is fixed here so the shared table is the same
    everywhere; a project that needs another rule recomputes it from `counts`
    in its own code and says so.

    Returns one row per neuron_id: neuron_id, region (string), region_count
    (0 for unknown), tied (bool).
    """
    table = load_neuropil_table() if table is None else table
    ids = pd.Index(pd.unique(np.asarray(neuron_ids, dtype=np.int64)), name="neuron_id")
    c = counts[counts["neuron_id"].isin(ids)]
    names = c["neuropil"].astype("string").fillna("UNASGD")
    loc = neuropil_location(names, table, nerve_cord).to_numpy()
    keep = (loc == "brain") & (c["count"].to_numpy() > 0)
    c = pd.DataFrame({"neuron_id": c["neuron_id"].to_numpy(np.int64)[keep],
                      "neuropil": names.to_numpy(dtype=object)[keep].astype(str),
                      "count": c["count"].to_numpy()[keep]})
    c = c.groupby(["neuron_id", "neuropil"], sort=False)["count"].sum().reset_index()
    c = c.sort_values(["neuron_id", "count", "neuropil"], ascending=[True, False, True],
                      kind="mergesort")
    top = c.drop_duplicates("neuron_id").set_index("neuron_id")
    second = c[c.duplicated("neuron_id")].drop_duplicates("neuron_id").set_index("neuron_id")["count"]
    out = pd.DataFrame(index=ids)
    out["region"] = top["neuropil"].reindex(ids).astype("string").fillna("unknown")
    out["region_count"] = top["count"].reindex(ids).fillna(0).astype("int64")
    out["tied"] = (second.reindex(ids) == top["count"].reindex(ids)).fillna(False).astype(bool)
    return out.reset_index()


def malecns_brain_part(neurons: pd.DataFrame, edgeroi: pd.DataFrame, roi_counts: pd.DataFrame,
                       nerve_cord=MALECNS_NERVE_CORD, table: pd.DataFrame | None = None):
    """The MaleCNS brain part, cropped by synapse location (Week 4 correction 4).

    neurons     malecns neurons (Traced), with the CNS-wide `region` as fetched
    edgeroi     malecns edgeroi cached at 1 synapse, ONE ROW PER PRIMARY
                NEUROPIL (a table with neuropil "all" cannot be used)
    roi_counts  malecns_neuropil_counts(...): per neuron and primary ROI,
                upstream + downstream over all partners (neuPrint roiInfo)

    Rules
      neurons  kept if they have at least one connection, with any partner,
               in a brain neuropil
      region   region_from_counts over brain neuropils (same rule as FAFB)
      edges    edgeroi rows in brain neuropils between kept neurons, summed
               over neuropils; threshold afterwards (D8 unchanged)

    Returns (neurons_brain, edges_brain, edgeroi_brain, report).
      edges_brain has min_syn_fetched = 1 and is the table to pass to
      basic_features and relative_weight for malecns.
      report is a dict with the Step 0 counts.
    """
    for name, df in (("neurons", neurons), ("edgeroi", edgeroi)):
        if "dataset" in df and set(df["dataset"].astype(str).unique()) - {"malecns"}:
            raise ValueError(f"{name}: malecns rows only")
    if "min_syn_fetched" in edgeroi and (edgeroi["min_syn_fetched"] != 1).any():
        raise ValueError("edgeroi: pass the table cached at 1 synapse")
    if edgeroi["neuropil"].astype("string").isin(["all"]).any():
        raise ValueError("edgeroi: neuropil 'all' found; refetch with one row per neuropil")
    need = {"neuron_id", "neuropil", "count"} - set(roi_counts.columns)
    if need:
        raise ValueError(f"roi_counts: missing columns {sorted(need)}")
    table = load_neuropil_table() if table is None else table

    region = region_from_counts(neurons["neuron_id"], roi_counts, table, nerve_cord)
    region = region[region["region"] != "unknown"]
    keep = pd.Index(region["neuron_id"])

    n_brain = neurons[neurons["neuron_id"].isin(keep)].copy()
    new_region = n_brain["neuron_id"].map(region.set_index("neuron_id")["region"]).astype("string")
    old_region = n_brain["region"].astype("string")
    r_dtype = neurons["region"].dtype
    n_brain["region"] = new_region if isinstance(r_dtype, pd.CategoricalDtype) else new_region.astype(r_dtype)
    n_brain = n_brain.reset_index(drop=True)

    loc = neuropil_location(edgeroi["neuropil"], table, nerve_cord).to_numpy()
    brain = loc == "brain"
    both = edgeroi["pre_id"].isin(keep).to_numpy() & edgeroi["post_id"].isin(keep).to_numpy()
    er_brain = edgeroi[brain & both].reset_index(drop=True)
    if (er_brain["pre_id"] == er_brain["post_id"]).any():
        raise ValueError("edgeroi: self-connections found; drop them when caching (D9)")
    e = er_brain.groupby(["pre_id", "post_id"], observed=True, sort=True)["syn_count"].sum().reset_index()
    for col in ("dataset", "release", "min_syn_fetched"):
        if col in edgeroi:
            e[col] = pd.Series([edgeroi[col].iloc[0]] * len(e), dtype=edgeroi[col].dtype)
    for col in ("pre_id", "post_id", "syn_count"):
        e[col] = e[col].astype(edgeroi[col].dtype)

    syn = edgeroi["syn_count"].to_numpy()
    sc = n_brain["super_class"].astype(str) if "super_class" in n_brain else None
    report = {
        "neurons_in": int(len(neurons)),
        "neurons_kept": int(len(n_brain)),
        "neurons_dropped": int(len(neurons) - len(n_brain)),
        "ascending_kept": int((sc == "ascending").sum()) if sc is not None else None,
        "sensory_ascending_kept": int((sc == "sensory_ascending").sum()) if sc is not None else None,
        "region_changed": int((old_region.fillna("") != new_region.fillna("")).sum()),
        "region_ties": int(region["tied"].sum()),
        "edgeroi_syn_brain": int(syn[brain].sum()),
        "edgeroi_syn_brain_unkept": int(syn[brain & ~both].sum()),
        "edgeroi_syn_nerve_cord": int(syn[loc == "nerve_cord"].sum()),
        "edgeroi_syn_no_location": int(syn[loc == "none"].sum()),
        "edges_tau1": int(len(e)),
        "edges_ge5": int((e["syn_count"] >= 5).sum()),
        "synapses_tau1": int(e["syn_count"].sum()),
    }
    return n_brain, e, er_brain, report
