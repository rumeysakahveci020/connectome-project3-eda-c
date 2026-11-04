from pathlib import Path
import pandas as pd

# v0.2.0 (Week 4 Shared Data Corrections): new super_class levels, three new
# neuron columns. Replace your config.py with this file, unchanged.
CONFIG_VERSION = "0.2.0"

# D6 sources
FAFB_MATERIALIZATION   = 783
FAFB_ANNOTATIONS_TAG   = "v3.2.0"
FAFB_SYNAPSE_TABLE     = "buhmann"      # "princeton" only for sensitivity
MALECNS_DATASET        = "male-cns:v1.0"
MALECNS_STATUS         = "Traced"
MALECNS_SYN_CONFIDENCE = 0.5

# D7-D9 cache
CACHE_DIR             = Path("data/cache")
FETCH_MIN_SYN         = 1
DROP_SELF_CONNECTIONS = True

# D3, D4 vocabularies
DATASET_VALUES = ["fafb", "malecns", "manc", "hemibrain"]
SIDE_VALUES    = ["left", "right", "center", "unknown"]
NT_VALUES      = ["acetylcholine", "glutamate", "gaba", "dopamine",
                  "serotonin", "octopamine", "histamine", "unknown"]
SUPER_CLASS_VALUES = ["sensory", "ascending", "descending", "motor", "endocrine",
                      "visual_projection", "visual_centrifugal", "optic", "central",
                      "sensory_ascending",               # FlyWire from v3.0.0; MaleCNS too
                      "efferent", "vnc_intrinsic",       # MaleCNS only
                      "unknown"]          # mapping: eda_shared.corrections.map_super_class
NT_CORRECTION_SOURCES = ["known_nt", "kc_override", "predicted"]
REGION_VALUES = None                     # after EDA: neuropils of both sources + "unknown"

CAT = lambda values: pd.CategoricalDtype(values)

# D1, D2, D5 neurons: column -> (dtype, nullable)
NEURON_SCHEMA = {
    "neuron_id":      ("int64", False),
    "dataset":        (CAT(DATASET_VALUES), False),
    "release":        ("string", False),
    "cell_type":      ("string", True),
    "cell_class":     ("string", True),
    "cell_sub_class": ("string", True),
    "super_class":    (CAT(SUPER_CLASS_VALUES), False),
    "side":           (CAT(SIDE_VALUES), False),
    "region":         ("string", False),
    "nt_type":        (CAT(NT_VALUES), False),
    "nt_confidence":  ("float32", True),
    "n_pre":          ("Int32", True),   # capital I: nullable integer
    "n_post":         ("Int32", True),
    "flywire_type":   ("string", True),
    "manc_type":      ("string", True),
    "hemibrain_type": ("string", True),
    "dimorphism":     ("string", True),
    "status":         ("string", True),
    "hemilineage":    ("string", True),
    "nt_known":       ("string", True),
    # Week 4, built by eda_shared.corrections
    "nt_type_corrected":    (CAT(NT_VALUES), False),             # use this one in analyses
    "nt_correction_source": (CAT(NT_CORRECTION_SOURCES), False),
    "super_class_source":   ("string", True),                    # raw source value
}

EDGE_SCHEMA = {
    "pre_id":          ("int64", False),
    "post_id":         ("int64", False),
    "syn_count":       ("int32", False),
    "dataset":         (CAT(DATASET_VALUES), False),
    "release":         ("string", False),
    "min_syn_fetched": ("int32", False),
}
EDGEROI_SCHEMA = {**EDGE_SCHEMA, "neuropil": ("string", False)}

def cache_path(dataset, table, release, source, min_syn=None, conf=None):
    name = f"{dataset}_{table}_{release}_{source}"
    if min_syn is not None:
        name += f"_min{min_syn}"
    if conf is not None:
        name += f"_conf{conf}"
    return CACHE_DIR / f"{name}.parquet"
