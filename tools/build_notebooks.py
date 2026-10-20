"""Writes notebooks/eda_C.ipynb and notebooks/00_integration.ipynb (re-run to regenerate)."""
import nbformat as nbf
from pathlib import Path

SETUP = '''import os, sys
from pathlib import Path
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "config.py").exists())
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)      # config.py uses relative paths (data/cache); nbconvert starts in notebooks/'''


def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    n.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return n


EDA = [
("md", """# Workstream C: neurotransmitters and local circuits (Project 3)

Question: how reliable are transmitter labels, and what does a local circuit look like?

Everything below is read from `data/cache`; there are no network calls. EDA looks at structure and labels, never at model performance.
Datasets: FlyWire FAFB (materialisation 783, annotations v3.2.0, Buhmann) and male CNS v1.0 (Traced, confidence >= 0.5, brain part only).
Provisional region flags come from `eda_shared` v0.1.1 and may change after Project 1's review."""),
("code", SETUP),
("code", """import os, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src import eda_utils as U, eda_c as C
from src.report_c import write_report
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
N_SEEDS = int(os.environ.get("EDA_N_SEEDS", 100))      # C6 null graphs; the plan asks for seeds 0-99
WORKERS = int(os.environ.get("EDA_WORKERS", 1))        # raise to use several cores for C6
C6_MAX_MIN = float(os.environ["EDA_C6_MAX_MINUTES"]) if os.environ.get("EDA_C6_MAX_MINUTES") else None   # time budget for C6
R = {"open_points": []}
CAP = "FAFB 783 (annotations v3.2.0, Buhmann) and male CNS v1.0 (Traced, conf >= 0.5, brain part). Provisional flags: eda_shared v0.1.1."
data = {d: U.load(d) for d in C.DS}
R["unspecified"] = {k: int(v) for k, v in data["malecns"][0]["region"].astype(str).value_counts().items() if "unspecified" in k}
for d, (n, e) in data.items():
    print(f"{U.LABEL[d]:8s} neurons {len(n):>9,}   edges {len(e):>11,}   optic {int(n['is_optic'].sum()):>8,}   central {int((~n['is_optic']).sum()):>8,}")"""),

("md", """## C1. Transmitter composition

**Why:** checks D4 and gives the evidence D20 asks for. Raw percentages only; no excitatory/inhibitory summary.
Four figures per population: neurons or output synapses, by super_class or by region_group."""),
("code", """R["C1"] = {}
R["C1_weighted"] = {}
for pop in ("central", "optic"):
    for by in ("super_class", "region_group"):
        for weighted in (False, True):
            fig, tabs = C.c1_figure(data, pop, by, weighted)
            tag = by + ("_weighted" if weighted else "")
            U.save(fig, U.fig_name("C1", "both", pop, extra=tag), C.c1_caption(CAP, pop, weighted))
            if by == "super_class" and not weighted:
                R["C1"][pop] = {d: tabs[d].loc["overall", U.NT_ORDER].astype(float).to_dict() for d in C.DS}
            if by == "super_class" and weighted:
                R["C1_weighted"][pop] = {d: tabs[d].loc["overall", U.NT_ORDER].astype(float).to_dict() for d in C.DS}
            plt.show()
pd.DataFrame({(pop, d): pd.Series(R["C1"][pop][d]) for pop in R["C1"] for d in C.DS}).round(3)"""),

("md", """## C2. How confident are the labels

**Why:** checks D4 (confidence kept, no cutoff at fetch). Labels are predictions; `nt_known` is the only measured check, and it exists only for FAFB.
Never apply one cutoff to both datasets."""),
("code", """fig = C.c2_figure(data)
U.save(fig, U.fig_name("C2", "both", "all"), CAP + " Population: all neurons with nt_confidence (FAFB top_nt_conf, male CNS predictedNtConfidence). No synapse threshold.")
plt.show()
pct = pd.DataFrame({U.LABEL[d]: C.c2_percentiles(data[d][0]) for d in C.DS}).T
display(pct.round(3))
below = {d: float((data[d][0]["nt_confidence"].dropna() < 0.6).mean()) for d in C.DS}
cover, per, cross = C.c2_agreement(data["fafb"][0])
display(cover); display(per.assign(agree=per["agree"].round(3))); display(cross)
tot_n = int(per["neurons"].sum())
R["C2"] = {d: {"p50": float(pct.loc[U.LABEL[d], "p50"]), "below_06": below[d],
               "by_nt": {k: float(v) for k, v in data[d][0].dropna(subset=["nt_confidence"]).groupby("nt_type")["nt_confidence"].median().items()}}
           for d in C.DS}
R["C2"]["agreement"] = float((per["agree"] * per["neurons"]).sum() / tot_n) if tot_n else float("nan")
R["C2"]["agreement_n"] = tot_n
pe = per.drop(index="histamine", errors="ignore")   # the FlyWire classifier has no histamine class, so those neurons can never agree
R["C2"]["agreement_excl_hist"] = float((pe["agree"] * pe["neurons"]).sum() / pe["neurons"].sum())
R["C2"]["agreement_excl_hist_n"] = int(pe["neurons"].sum())
def row_share(known, pred):
    return float(cross.loc[known, pred] / cross.loc[known].sum()) if known in cross.index and pred in cross.columns else float("nan")
R["C2"]["ach_to_da"] = row_share("acetylcholine", "dopamine"); R["C2"]["glu_to_gaba"] = row_share("glutamate", "gaba")"""),
("code", """# C2b (supplementary): male CNS predictedNt against consensusNt. Needs the file written by fetch_malecns.
import config
cons_path = U.config.CACHE_DIR / "malecns_nt_consensus_v1.0.parquet"
tab = C.c2b_consensus(data["malecns"][0], cons_path)
if tab is None:
    print("no consensusNt file in the cache: skipped"); R["C2b"] = None
else:
    display(tab)
    m = data["malecns"][0].merge(pd.read_parquet(cons_path), on="neuron_id", how="left")
    cons = m["consensusNt"].astype("string").str.lower().replace({"unclear": "unknown"})
    agree = float((m["nt_type"].astype(str) == cons).mean())
    kc = m[m["cell_type"].astype("string").str.startswith("KC", na=False)]
    kc_top = kc["nt_type"].astype(str).value_counts()
    R["C2b"] = {"agree": agree, "kc_top": kc_top.index[0] if len(kc_top) else "n/a",
                "kc_share": float(kc_top.iloc[0] / len(kc)) if len(kc) else float("nan")}
    R["open_points"].append("which male CNS transmitter label to use (predictedNt has no sensible Kenyon-cell class)" if R["C2b"]["kc_top"] != "acetylcholine" else "male CNS predictedNt vs consensusNt agree for Kenyon cells")
    cen = m[~m["is_optic"]]
    def share(series, k): return float((series == k).mean())
    cons_c = cen["consensusNt"].astype("string").str.lower().replace({"unclear": "unknown"})
    R["C2b"].update({"central_ach_pred": share(cen["nt_type"].astype(str), "acetylcholine"), "central_da_pred": share(cen["nt_type"].astype(str), "dopamine"),
                     "central_ach_cons": share(cons_c, "acetylcholine"), "central_da_cons": share(cons_c, "dopamine")})
    print(R["C2b"])"""),

("md", """## C3. Do hemilineages share a transmitter

**Why:** checks D1 (Ito-Lee hemilineage kept): how informative the column is where it is filled.
FAFB neurons with a hemilineage and a fast transmitter (acetylcholine, GABA, glutamate); hemilineages with at least 20 such neurons."""),
("code", """hl = C.c3_hemilineages(data["fafb"][0])
fig = C.c3_figure(hl)
U.save(fig, U.fig_name("C3", "fafb", "all"), "FAFB 783. Population: neurons with a hemilineage and a fast transmitter (acetylcholine, GABA, glutamate), hemilineages with >= 20 such neurons; no threshold. A hemilineage is a family of neurons from one progenitor lineage. Reference line at 0.9.")
plt.show()
low = hl[hl["share"] < 0.8]
display(low.round(3))
R["C3"] = {"n": len(hl), "share_ge_09": float((hl["share"] >= 0.9).mean()), "n_lt_08": len(low), "lt_08": list(low.index)}"""),

("md", """## C4. What the glutamate choice changes

**Why:** D20 asks which results change under the opposite glutamate choice; this shows where.
Rule 1: ACh = E; GABA and glutamate = I. Rule 2: ACh and glutamate = E; GABA = I. Modulatory and unknown are left out of both."""),
("code", """net = C.c4_net_index(*data["fafb"], tau=10)
fig = C.c4_figure(net, "region_group")
U.save(fig, U.fig_name("C4", "fafb", "central", tau=10, extra="region_group"), "FAFB 783, Buhmann edges, tau = 10, central postsynaptic neurons. Net index (E - I) / (E + I); modulatory and unknown partners are left out.")
plt.show()
fig = C.c4_figure(net, "neuropil", top=12)
U.save(fig, U.fig_name("C4", "fafb", "central", tau=10, extra="neuropil12"), "FAFB 783, Buhmann edges, tau = 10, central postsynaptic neurons, the 12 neuropils with most neurons. Net index (E - I) / (E + I); modulatory and unknown partners are left out.")
plt.show()
sc_g = C.c4_sign_changes(net, "region_group")
top12 = net.drop_duplicates("neuron_id")["neuropil"].value_counts().head(12).index
sc_n = C.c4_sign_changes(net[net["neuropil"].isin(top12)], "neuropil")
display(sc_g.round(3)); display(sc_n.round(3))
R["C4"] = {"n_groups": len(sc_g), "n_groups_flip": int(sc_g["changes_sign"].sum()), "groups_flip": list(sc_g.index[sc_g["changes_sign"]]),
           "n_neuropils": len(sc_n), "n_neuropils_flip": int(sc_n["changes_sign"].sum()), "neuropils_flip": list(sc_n.index[sc_n["changes_sign"]])}
r1, r2 = list(C.RULES)
shift = (sc_n[r2] - sc_n[r1]).abs()
R["C4"].update({"n_shift": int((shift >= 0.25).sum()), "shift_lo": float(shift[shift >= 0.25].min()), "shift_hi": float(shift.max())})"""),

("md", """## C5. The mushroom body subgraph

**Why:** Project 3's first measurement: is the subgraph large and dense enough to model?
Selection = neurons whose region is a mushroom body neuropil. The list used is printed below. The week 2 subgraph was selected by cell type, so a second selection is shown for continuity; the two are not the same set."""),
("code", """print("mushroom body neuropils used:", C.mb_neuropils())
tab5 = C.c5_table(data)
display(tab5.round(4))
fig = C.c5_ccdf_figure(data, tau=10)
U.save(fig, U.fig_name("C5", "both", "all", tau=10, extra="degree_ccdf"), CAP + " Mushroom body neuropils: " + ", ".join(C.mb_neuropils()) + ". Threshold: tau = 10, degrees among mushroom body neurons. Zero-degree neurons are not drawn; solid = in-degree, dashed = out-degree.")
plt.show()
def pick(d, tau, sel="region"):
    r = tab5[(tab5["dataset"] == U.LABEL[d]) & (tab5["tau"] == tau) & (tab5["selection"] == sel)].iloc[0]
    return r
R["C5"] = {d: {"neurons": int(pick(d, 10).neurons), "edges": int(pick(d, 10).edges), "density": float(pick(d, 10).density),
               "isolated": int(pick(d, 10).isolated), "isolated_5": int(pick(d, 5).isolated),
               "edges_1": int(pick(d, 1).edges), "edges_5": int(pick(d, 5).edges), "edges_5_isolated": int(pick(d, 5).isolated),
               "median_in": float(pick(d, 10).median_in), "median_out": float(pick(d, 10).median_out),
               "ct_isolated_10": int(pick(d, 10, "cell_type").isolated), "ct_isolated_5": int(pick(d, 5, "cell_type").isolated)} for d in C.DS}"""),

("md", """## C6. Non-random structure in the subgraph

**Why:** checks the shared randomiser on real data before every project's null model relies on it, and shows structure beyond degree.
Null: `randomise_edges` with seeds 0 to N_SEEDS - 1 (the plan asks for 0 to 99). z = (observed - null mean) / null standard deviation (sample SD). If the randomiser warns that it cannot mix a dense graph, report the warning with the result."""),
("code", """z = C.c6_motifs(*data["fafb"], tau=10, n_seeds=N_SEEDS, workers=WORKERS, max_minutes=C6_MAX_MIN,
                progress_path=U.ROOT / "reports" / "c6_progress.txt")      # watch this file from a second window
N_USED = z.attrs["n_seeds"]
fig = C.c6_figure(z, tau=10)
U.save(fig, U.fig_name("C6", "fafb", "all", tau=10), f"FAFB 783, Buhmann edges, tau = 10, mushroom body subgraph (the C5 neurons: {', '.join(C.mb_neuropils())}), {N_USED} degree-preserving null graphs (seeds 0-{N_USED-1}). Filled bars: |z| > 2. Symmetric log axis. Triad census by igraph (same counts as networkx triadic_census, tested).")
plt.show()
display(z.round(3))
R["C6"] = {"n_seeds": N_USED, "z_recip": float(z.loc["reciprocal pair", "z"]), "sig": [m for m in z.index if abs(z.loc[m, "z"]) > 2],
           "over": [m for m in z.index if z.loc[m, "z"] > 2], "under": [m for m in z.index if z.loc[m, "z"] < -2],
           "obs_recip": int(z.loc["reciprocal pair", "observed"]), "null_recip": float(z.loc["reciprocal pair", "null_mean"]),
           "z_abs_max": float(z["z"].abs().max())}
import json
(U.ROOT / "reports").mkdir(exist_ok=True)
(U.ROOT / "reports" / "eda_C_numbers.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")"""),

("md", """## One-page summary

Writes a draft of `reports/eda_C.md` from the numbers above. The numbers are exact; edit the sentences in your own words and write the surprise line yourself."""),
("code", """print(write_report(R, U.ROOT / "reports" / "eda_C.md"))"""),
]

INTEGRATION = [
("md", "# 00 Integration: validators and the Step 0 count check\n\nRun after every fetch. Every check raises naming the column."),
("code", SETUP),
("code", """import pandas as pd, config
from src.validators import validate_neurons, validate_edges, assert_interchangeable
rows = {"fafb": ("783", f"ann{config.FAFB_ANNOTATIONS_TAG[1:]}", config.FAFB_SYNAPSE_TABLE, {}),
        "malecns": ("v1.0", "traced", "traced", {"conf": config.MALECNS_SYN_CONFIDENCE})}
T = {}
for d, (rel, nsrc, esrc, kw) in rows.items():
    n = pd.read_parquet(config.cache_path(d, "neurons", rel, nsrc))
    e = pd.read_parquet(config.cache_path(d, "edges", rel, esrc, min_syn=1, **kw))
    r = pd.read_parquet(config.cache_path(d, "edgeroi", rel, esrc, min_syn=1, **kw))
    validate_neurons(n); validate_edges(e, n); validate_edges(r, n, roi=True)
    T[d] = (n, e); print(d, "validators passed:", len(n), "neurons,", len(e), "edges")
assert_interchangeable(T["fafb"][0], T["malecns"][0]); assert_interchangeable(T["fafb"][1], T["malecns"][1])
print("assert_interchangeable passed")"""),
("code", """expected = {"fafb neurons": 139_248, "fafb edges >= 5": 2_700_513, "malecns neurons": 165_122}
got = {"fafb neurons": len(T["fafb"][0]), "fafb edges >= 5": int((T["fafb"][1].syn_count >= 5).sum()),
       "malecns neurons": len(T["malecns"][0]), "malecns edges >= 5 (agree one number with Projects 1 and 4)": int((T["malecns"][1].syn_count >= 5).sum())}
pd.DataFrame({"got": got, "expected": {**expected, "malecns edges >= 5 (agree one number with Projects 1 and 4)": None}})"""),
]

if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "notebooks"
    out.mkdir(exist_ok=True)
    nbf.write(nb(EDA), out / "eda_C.ipynb")
    nbf.write(nb(INTEGRATION), out / "00_integration.ipynb")
    print("written", sorted(p.name for p in out.glob("*.ipynb")))
