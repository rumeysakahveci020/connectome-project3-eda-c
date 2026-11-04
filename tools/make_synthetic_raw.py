"""DRY-RUN ONLY. Writes small synthetic RAW input files in the formats of the real downloads, so that the whole build
(python -m src.build_cache ...) and the corrections can be tested before the big files are in place.

    python tools/make_synthetic_raw.py data/synthetic          # writes data/synthetic/raw/*

then (the cache goes to a separate folder):
    $env:EDA_CACHE_DIR="data/synthetic/cache"   # config.CACHE_DIR; the neuPrint answer is read from data/synthetic/raw
    python -m src.build_cache --fafb-edges data/synthetic/raw/proofread_connections_783.feather ...   (printed at the end)

The sizes and the planted facts are fixed in PLANTED below; the tests check that the build recovers them.
Nothing here is real data: never put these numbers in a report or a figure you hand in.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PLANTED = {"fafb_n": 3000, "fafb_not_annotated": 14, "fafb_not_in_release": 7, "malecns_n": 2500, "malecns_self": 3}

F_REG = {"optic": ["ME_R", "LO_L", "ME_L"], "kc": ["MB_CA_R", "MB_PED_L", "MB_VL_R", "MB_ML_L"],
         "mbon": ["MB_VL_R", "MB_ML_L", "SMP_R"], "dan": ["MB_ML_L", "MB_VL_R", "SMP_L"],
         "central": ["SMP_R", "FB", "LH_L", "AL_R", "PLP_L"], "sensory": ["AL_R", "GNG"], "sens_asc": ["GNG", "SAD"],
         "other": ["GNG", "SAD"]}
M_NAME = {"MB_CA_R": "CA(R)", "MB_PED_L": "PED(L)", "MB_VL_R": "aL(R)", "MB_ML_L": "bL(L)", "ME_R": "ME(R)", "LO_L": "LO(L)",
          "ME_L": "ME(L)", "SMP_R": "SMP(R)", "SMP_L": "SMP(L)", "LH_L": "LH(R)", "AL_R": "AL(R)", "PLP_L": "PLP(L)", "FB": "FB",
          "GNG": "GNG", "SAD": "SAD"}
M_PRIMARY = sorted(set(M_NAME.values()) | {"CentralBrain-unspecified", "Optic-unspecified(L)", "LegNp(T1)(L)", "LegNp(T2)(R)"})
KINDS = ["optic", "kc", "mbon", "dan", "central", "sensory", "sens_asc", "other"]
P = [.28, .20, .02, .03, .30, .09, .02, .06]


def _feather(df: pd.DataFrame, path: Path, chunk: int = 4000) -> None:
    import pyarrow as pa
    import pyarrow.feather as pf
    pf.write_feather(pa.Table.from_pandas(df, preserve_index=False), str(path), chunksize=chunk)


def make_fafb(raw: Path, rng) -> None:
    n = PLANTED["fafb_n"]
    ids = 720575940000000000 + np.arange(n, dtype="int64") * 7
    kinds = rng.choice(KINDS, n, p=P)
    nt = np.empty(n, dtype=object)
    known = np.full(n, None, dtype=object)
    for i, k in enumerate(kinds):
        if k == "kc":
            nt[i] = "dopamine" if rng.random() < .97 else "acetylcholine"
            known[i] = "acetylcholine"                                 # as in the real file: nt_known covers every Kenyon cell
        elif k == "mbon":
            nt[i] = rng.choice(["gaba", "glutamate", "acetylcholine"])
        elif k == "dan":
            nt[i] = "dopamine"
        elif k == "optic":
            nt[i] = rng.choice(["acetylcholine", "glutamate", "gaba"], p=[.5, .3, .2])
            known[i] = rng.choice([None, "histamine", "glutamate, tyramine", "gaba, nitric oxide", "acetylcholine; sNPF"], p=[.7, .1, .05, .05, .1])
        else:
            nt[i] = rng.choice(["acetylcholine", "gaba", "glutamate", "serotonin", "octopamine"], p=[.55, .2, .15, .05, .05])
            known[i] = rng.choice([None, "glutamate", "gaba-negative"], p=[.85, .1, .05])
    sc = {"optic": "optic", "kc": "central", "mbon": "central", "dan": "central", "central": "central", "sensory": "sensory",
          "sens_asc": "sensory_ascending", "other": "descending"}
    ann = pd.DataFrame({
        "root_id": ids, "cell_type": np.where(kinds == "kc", "KCg-m", np.where(kinds == "mbon", "MBON01", np.where(kinds == "dan", "PAM01", "T4a"))),
        "cell_class": None, "cell_sub_class": None, "super_class": [sc[k] for k in kinds],
        "side": rng.choice(["left", "right", "center", "na"], n, p=[.48, .47, .02, .03]), "top_nt": nt,
        "top_nt_conf": np.clip(rng.normal(.69, .1, n), .2, 1.0).round(3), "hemibrain_type": None, "dimorphism": None, "status": None,
        "ito_lee_hemilineage": np.where((kinds == "central") & (rng.random(n) < .6), rng.choice([f"HL{i}" for i in range(40)], n), None),
        "known_nt": known})
    ann = ann.iloc[:n - PLANTED["fafb_not_annotated"]]                               # 14 proofread ids with no annotation row
    extra = ann.iloc[:PLANTED["fafb_not_in_release"]].copy()
    extra["root_id"] = 720575950000000000 + np.arange(len(extra), dtype="int64")      # 7 annotation rows outside the release
    ann = pd.concat([ann, extra], ignore_index=True)
    ann.astype(str).replace({"None": "", "nan": ""}).to_csv(raw / "Supplemental_file1_neuron_annotations.tsv", sep="\t", index=False)
    np.save(raw / "proofread_root_ids_783.npy", ids)

    region = np.array([rng.choice(F_REG[k]) for k in kinds], dtype=object)
    second = np.array([rng.choice(F_REG[k]) for k in kinds], dtype=object)
    pre_rows, post_rows = [], []
    for i in range(n):
        main = int(rng.integers(60, 500))
        tie = rng.random() < .02
        for rows in (pre_rows, post_rows):
            rows.append((ids[i], region[i], main))
            if second[i] != region[i]:
                rows.append((ids[i], second[i], main if tie else int(rng.integers(1, main))))
            if rng.random() < .1:
                rows.append((ids[i], "UNASGD", int(rng.integers(600, 900))))        # bigger than any region: must never win
    for name, rows, col in (("pre", pre_rows, "pre_pt_root_id"), ("post", post_rows, "post_pt_root_id")):
        d = pd.DataFrame(rows, columns=[col, "neuropil", "count"])
        d = d.groupby([col, "neuropil"], as_index=False)["count"].sum()
        _feather(d, raw / f"per_neuron_neuropil_count_{name}_783.feather")
    m = 12 * n
    pre = rng.integers(0, n, m); post = rng.integers(0, n, m)
    e = pd.DataFrame({"pre": pre, "post": post}); e = e[e.pre != e.post].drop_duplicates()
    e["neuropil"] = [rng.choice([region[a], region[b]]) for a, b in zip(e.pre, e.post)]
    e["syn_count"] = rng.geometric(.22, len(e))
    e = e.groupby(["pre", "post", "neuropil"], as_index=False)["syn_count"].sum()
    out = pd.DataFrame({"pre_pt_root_id": ids[e.pre], "post_pt_root_id": ids[e.post], "neuropil": e.neuropil, "syn_count": e.syn_count})
    _feather(out, raw / "proofread_connections_783.feather")


def make_malecns(raw: Path, rng) -> None:
    n = PLANTED["malecns_n"]
    ids = 10000 + np.arange(n, dtype="int64") * 7
    kinds = rng.choice(KINDS + ["vncnly"], n, p=[.26, .18, .02, .03, .27, .08, .02, .06, .08])
    sc = {"optic": "ol_intrinsic", "kc": "cb_intrinsic", "mbon": "cb_intrinsic", "dan": "cb_intrinsic", "central": "cb_intrinsic",
          "sensory": "cb_sensory", "sens_asc": "sensory_ascending", "other": "ascending_neuron", "vncnly": "vnc_intrinsic"}
    ctype = np.where(kinds == "kc", "KCg-m", np.where(kinds == "mbon", "MBON01", np.where(kinds == "dan", "PAM01", "T4a")))
    pnt = np.array([("dopamine" if (k == "kc" and rng.random() < .99) else rng.choice(["acetylcholine", "gaba", "glutamate", "unclear", "histamine"], p=[.5, .2, .15, .1, .05]))
                    for k in kinds], dtype=object)
    cons = np.where(kinds == "kc", "acetylcholine", pnt)
    neuron_df = pd.DataFrame({"bodyId": ids, "type": ctype, "instance": ctype, "superclass": [sc[k] for k in kinds],
                              "somaSide": rng.choice(["L", "R", "M", "NA"], n, p=[.48, .47, .02, .03]), "status": "Traced",
                              "predictedNt": pnt, "predictedNtConfidence": np.clip(rng.normal(.93, .05, n), .5, 1.0),
                              "consensusNt": cons, "itoleeHl": None, "pre": 100, "post": 100})
    roi_rows, regions = [], {}
    for i, k in enumerate(kinds):
        if k == "vncnly":
            names = ["LegNp(T1)(L)"]
        else:
            names = [M_NAME[r] for r in rng.choice(F_REG[k if k in F_REG else "other"], 2)]
            if k == "other" and rng.random() < .5:
                names.append("LegNp(T2)(R)")                                  # ascending: brain part and nerve cord
            if rng.random() < .03:
                names = ["CentralBrain-unspecified"]
        names = list(dict.fromkeys(names))
        regions[ids[i]] = names
        for j, r in enumerate(names):
            roi_rows.append((ids[i], r, int(rng.integers(50, 400)) // (j + 1), int(rng.integers(50, 400)) // (j + 1)))
    roi_df = pd.DataFrame(roi_rows, columns=["bodyId", "roi", "upstream", "downstream"])
    roi_df["pre"], roi_df["post"] = roi_df["downstream"], roi_df["upstream"]
    roi_df = pd.concat([roi_df, pd.DataFrame({"bodyId": ids[:5], "roi": "CentralBrain", "upstream": 999, "downstream": 999, "pre": 1, "post": 1})])   # not primary: ignored
    pkl = raw / "neuprint_malecns_v1.0_traced.pkl"
    pd.to_pickle((neuron_df, roi_df, M_PRIMARY + ["CentralBrain", "OL(R)"][:0]), pkl)

    # one row per synapse
    m = 9 * n
    pre = rng.integers(0, n, m); post = rng.integers(0, n, m)
    keep = pre != post
    pre, post = pre[keep], post[keep]
    rows = []
    for a, b in zip(pre, post):
        w = int(rng.geometric(.3))
        pool = regions[ids[b]]
        for _ in range(w):
            r = rng.choice(pool)
            u = rng.random()
            rows.append((ids[a], ids[b], None if u < .01 else ("NotPrimary" if u < .02 else r)))
    sp = pd.DataFrame(rows, columns=["body_pre", "body_post", "primary_post"])
    selfs = pd.DataFrame({"body_pre": ids[:PLANTED["malecns_self"]], "body_post": ids[:PLANTED["malecns_self"]], "primary_post": "SMP(R)"})
    junk = pd.DataFrame({"body_pre": [5, 6], "body_post": ids[:2], "primary_post": "SMP(R)"})           # not Traced: must be ignored
    sp = pd.concat([sp, selfs, junk], ignore_index=True)
    sp["conf_pre"] = .9; sp["conf_post"] = .9
    _feather(sp, raw / "syn-partners-male-cns-v1.0-minconf-0.5.feather", chunk=7000)
    wts = sp[sp["body_pre"].isin(ids) & sp["body_post"].isin(ids)].groupby(["body_pre", "body_post"], as_index=False).size().rename(columns={"size": "weight"})
    _feather(wts, raw / "connectome-weights-male-cns-v1.0-minconf-0.5.feather")


def main(out_dir: str) -> Path:
    raw = Path(out_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    make_fafb(raw, np.random.default_rng(11))
    make_malecns(raw, np.random.default_rng(12))
    return raw


if __name__ == "__main__":
    raw = main(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic")
    print("wrote", sorted(p.name for p in raw.iterdir()))
    print("next:\n  EDA_CACHE_DIR=%s python -m src.build_cache --fafb-edges %s/proofread_connections_783.feather --fafb-annotations %s/Supplemental_file1_neuron_annotations.tsv "
          "--fafb-proofread-ids %s/proofread_root_ids_783.npy --fafb-count-pre %s/per_neuron_neuropil_count_pre_783.feather --fafb-count-post %s/per_neuron_neuropil_count_post_783.feather "
          "--malecns-syn-partners %s/syn-partners-male-cns-v1.0-minconf-0.5.feather --malecns-weights %s/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
          % ((raw.parent / "cache",) + (raw,) * 7))
