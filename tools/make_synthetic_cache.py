"""DRY-RUN ONLY. Writes small synthetic caches in the Data Conventions format so that
notebooks/eda_C.ipynb can be tested before the real downloads are in place.

    python tools/make_synthetic_cache.py data/cache_synthetic
    set EDA_CACHE_DIR=data/cache_synthetic   (Windows: $env:EDA_CACHE_DIR="data/cache_synthetic")

Nothing here is real data: never put these numbers in a report or a figure you hand in.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.validators import coerce  # noqa: E402


def make(name: str, seed: int, n: int):
    rng = np.random.default_rng(seed)
    male = name == "malecns"
    kinds = rng.choice(["optic", "kc", "mbon", "dan", "central", "sensory", "other"], n,
                       p=[.30, .20, .02, .03, .30, .10, .05])
    reg_by = {"optic": ["ME_R", "LO_L", "ME_L"], "kc": ["MB_CA_R", "MB_PED_L", "MB_VL_R", "MB_ML_L"],
              "mbon": ["MB_VL_R", "MB_ML_L", "SMP_R"], "dan": ["MB_ML_L", "MB_VL_R", "SMP_L"],
              "central": ["SMP_R", "FB", "LH_L", "AL_R", "PLP_L"], "sensory": ["AL_R", "GNG"], "other": ["GNG", "SAD"]}
    ml = {"MB_CA_R": "CA(R)", "MB_PED_L": "PED(L)", "MB_VL_R": "aL(R)", "MB_ML_L": "bL(L)", "ME_R": "ME(R)",
          "LO_L": "LO(L)", "ME_L": "ME(L)", "SMP_R": "SMP(R)", "SMP_L": "SMP(L)", "LH_L": "LH(L)", "AL_R": "AL(R)",
          "PLP_L": "PLP(L)", "FB": "FB", "GNG": "GNG", "SAD": "SAD"}
    region = np.array([rng.choice(reg_by[k]) for k in kinds], dtype=object)
    if male:
        region = np.array([ml[r] for r in region], dtype=object)
        vnc = rng.random(n) < .04
        region[vnc] = "LegNp(T1)(L)"
    nt = []
    for k in kinds:
        if k == "kc":
            nt.append("dopamine" if (male and rng.random() < .97) else "acetylcholine")
        elif k == "mbon":
            nt.append(rng.choice(["gaba", "glutamate", "acetylcholine"]))
        elif k == "dan":
            nt.append("dopamine")
        elif k == "optic":
            nt.append(rng.choice(["acetylcholine", "glutamate", "gaba", "histamine" if male else "gaba"], p=[.5, .2, .2, .1]))
        else:
            nt.append(rng.choice(["acetylcholine", "gaba", "glutamate", "serotonin", "octopamine", "unknown"],
                                 p=[.55, .15, .15, .05, .03, .07]))
    conf = np.where(np.array(nt) == "unknown", np.nan, np.clip(rng.normal(.93 if male else .69, .08, n), .2, 1.0))
    hl = np.where((kinds == "central") & (rng.random(n) < .6), rng.choice([f"HL{i}" for i in range(40)], n), None)
    known = np.where(rng.random(n) < .3, rng.choice(["acetylcholine", "gaba", "glutamate", "acetylcholine; sNPF",
                                                       "gaba-negative", "glutamate, gaba; glutamate"], n), None)
    neurons = pd.DataFrame({
        "neuron_id": (720575940000000000 if not male else 10000) + np.arange(n) * 7,
        "dataset": name, "release": "v1.0" if male else "783",
        "cell_type": np.where(kinds == "kc", "KCg-m", np.where(kinds == "mbon", "MBON01", np.where(kinds == "dan", "PAM01", "T4a"))),
        "cell_class": None, "cell_sub_class": None,
        "super_class": np.where(kinds == "optic", "optic", np.where(kinds == "sensory", "sensory", "central")),
        "side": rng.choice(["left", "right", "center", "unknown"], n, p=[.48, .47, .02, .03]),
        "region": region, "nt_type": nt, "nt_confidence": conf.astype("float32"),
        "n_pre": pd.array([pd.NA] * n, dtype="Int32"), "n_post": pd.array([pd.NA] * n, dtype="Int32"),
        "flywire_type": None, "manc_type": None, "hemibrain_type": None, "dimorphism": None, "status": None,
        "hemilineage": hl, "nt_known": None if male else known})
    for c in ["cell_type", "cell_class", "cell_sub_class", "region", "release", "flywire_type", "manc_type",
              "hemibrain_type", "dimorphism", "status", "hemilineage", "nt_known"]:
        neurons[c] = neurons[c].astype("string")
    neurons = coerce(neurons[list(config.NEURON_SCHEMA)], config.NEURON_SCHEMA, name)
    # edges: heavy-tailed, denser inside the mushroom body
    w = rng.pareto(1.4, n) + 1
    w[np.isin(kinds, ["kc", "mbon", "dan"])] *= 4
    w /= w.sum()
    m = 14 * n
    pre = rng.choice(n, m, p=w); post = rng.choice(n, m, p=w)
    e = pd.DataFrame({"pre": pre, "post": post}); e = e[e.pre != e.post].drop_duplicates()
    ids = neurons["neuron_id"].to_numpy()
    k = rng.integers(1, 3, len(e))
    e = e.loc[e.index.repeat(k)].reset_index(drop=True)
    e["neuropil"] = [rng.choice([region[a], region[b]]) for a, b in zip(e.pre, e.post)]
    e["syn_count"] = rng.geometric(.22, len(e))
    er = e.groupby(["pre", "post", "neuropil"], as_index=False)["syn_count"].sum()
    er["pre_id"] = ids[er.pre]; er["post_id"] = ids[er.post]
    er["dataset"] = name; er["release"] = neurons["release"].iloc[0]; er["min_syn_fetched"] = 1
    er["syn_count"] = er["syn_count"].astype("int32"); er["min_syn_fetched"] = er["min_syn_fetched"].astype("int32")
    er["neuropil"] = er["neuropil"].astype("string")
    edgeroi = coerce(er[list(config.EDGEROI_SCHEMA)], config.EDGEROI_SCHEMA, name + " edgeroi")
    ed = edgeroi.groupby(["pre_id", "post_id"], as_index=False)["syn_count"].sum()
    ed["dataset"] = name; ed["release"] = neurons["release"].iloc[0]; ed["min_syn_fetched"] = 1
    ed["syn_count"] = ed["syn_count"].astype("int32"); ed["min_syn_fetched"] = ed["min_syn_fetched"].astype("int32")
    edges = coerce(ed[list(config.EDGE_SCHEMA)], config.EDGE_SCHEMA, name + " edges")
    return neurons, edges, edgeroi


def main(out_dir: str) -> None:
    config.CACHE_DIR = Path(out_dir)
    for name, seed, n, rel, src in [("fafb", 1, 5000, "783", "ann3.2.0"), ("malecns", 2, 4000, "v1.0", "traced")]:
        neurons, edges, edgeroi = make(name, seed, n)
        kw = dict(min_syn=1) if name == "fafb" else dict(min_syn=1, conf=config.MALECNS_SYN_CONFIDENCE)
        esrc = config.FAFB_SYNAPSE_TABLE if name == "fafb" else "traced"
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        neurons.to_parquet(config.cache_path(name, "neurons", rel, src), index=False)
        edges.to_parquet(config.cache_path(name, "edges", rel, esrc, **kw), index=False)
        edgeroi.to_parquet(config.cache_path(name, "edgeroi", rel, esrc, **kw), index=False)
        print(name, len(neurons), "neurons", len(edges), "edges")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/cache_synthetic")
