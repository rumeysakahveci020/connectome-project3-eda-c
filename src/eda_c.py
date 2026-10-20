"""Workstream C: neurotransmitters and local circuits (Week 3 EDA Plan, items C1-C6).

Every function takes tables built to the Data Conventions and returns the numbers it plots,
so the notebook shows both and the report can quote them. EDA looks at structure and labels,
never at model performance.
"""
from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from eda_shared.eda_helpers import basic_features, load_neuropil_table
from eda_shared.randomiser import randomise_edges
from src import eda_utils as U

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
DS = ("fafb", "malecns")


# ----------------------------------------------------------------------------- C1
_OUT_SYN: dict = {}


def _total_out_syn(neurons: pd.DataFrame, edges: pd.DataFrame) -> pd.Series:
    key = (id(neurons), id(edges))
    if key not in _OUT_SYN:
        _OUT_SYN[key] = basic_features(neurons, edges, tau=10).set_index("neuron_id")["total_out_syn"]
    return _OUT_SYN[key]


def c1_shares(neurons: pd.DataFrame, edges: pd.DataFrame, pop: str, by: str, weighted: bool) -> pd.DataFrame:
    """Share of neurons (or of total_out_syn) per nt_type: overall, then per `by` group."""
    n = U.population(neurons, pop).copy()
    n["w"] = n["neuron_id"].map(_total_out_syn(neurons, edges)).fillna(0).astype(float) if weighted else 1.0
    n["nt_type"] = n["nt_type"].astype(str)
    rows = {}
    def add(label, d):
        tot = d["w"].sum()
        if tot > 0:
            s = d.groupby("nt_type")["w"].sum() / tot
            rows[label] = {**{k: float(s.get(k, 0.0)) for k in U.NT_ORDER}, "n": len(d)}
    add("overall", n)
    order = U.SUPER_ORDER if by == "super_class" else U.GROUP_ORDER
    for g in U.ordered(n[by].astype(str).unique(), order):
        add(g, n[n[by].astype(str) == g])
    return pd.DataFrame(rows).T


SMALL_N = 50          # rows with fewer neurons are drawn faded and are not interpreted

NT_DISPLAY = {"acetylcholine": "Acetylcholine", "gaba": "GABA", "glutamate": "Glutamate", "dopamine": "Dopamine", "serotonin": "Serotonin",
              "octopamine": "Octopamine", "histamine": "Histamine", "unknown": "Unknown / not classified"}


def headline(ax, title: str, how_to_read: str, base: float = 8.0, chars_per_inch: float = 13.0):
    """Title plus a short plain-language 'how to read this plot' line, above the axes. Offsets are in points, so the layout does
    not depend on the figure size. `base` leaves room for a panel title (use ~26 when the axes has one)."""
    import textwrap
    ax.figure.tight_layout()                                     # lay the panels out first; the header text is added afterwards
    width = int(ax.figure.get_figwidth() * chars_per_inch)
    wrapped = "\n".join(textwrap.fill(line, width) for line in how_to_read.split("\n"))
    n_sub = wrapped.count("\n") + 1
    ax.annotate(wrapped, xy=(0, 1), xycoords="axes fraction", xytext=(0, base), textcoords="offset points", ha="left", va="bottom",
                fontsize=8.3, color="#555555", linespacing=1.3, annotation_clip=False)
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, base + 11.5 * n_sub + 7), textcoords="offset points", ha="left",
                va="bottom", fontsize=11, linespacing=1.25, annotation_clip=False)




def c1_caption(cap: str, pop: str, weighted: bool) -> str:
    return (f"{cap} Population: {pop}. Shares of {'total output synapses' if weighted else 'neurons'} per nt_type (FAFB top_nt, male CNS predictedNt). "
            f"Central = is_optic False (also neurons with no assigned region); optic = is_optic True. Rows with fewer than {SMALL_N} neurons (*) are faded "
            f"and not interpreted. Threshold: {'none; all cached edges (1 synapse) summed' if weighted else 'none (neuron counts)'}.")


def c1_figure(data: dict, pop: str, by: str, weighted: bool):
    """data: {'fafb': (neurons, edges), 'malecns': (...)}

    Both panels list the SAME rows in the same order (a row one dataset does not have is labelled '(none)'), so a row can be
    read straight across. Rows with fewer than SMALL_N neurons are faded and marked *. When a population has a single group
    (the optic lobes have only sensory_periphery), the group is not drawn twice: the row is labelled 'overall = <group>'.
    """
    tabs = {d: c1_shares(*data[d], pop, by, weighted) for d in DS}
    order = U.SUPER_ORDER if by == "super_class" else U.GROUP_ORDER
    present = set().union(*[set(t.index) for t in tabs.values()]) - {"overall"}
    groups = [g for g in order if g in present] + sorted(present - set(order))
    suffix = ""
    if len(groups) == 1 and all(groups[0] in t.index and int(t.loc[groups[0], "n"]) == int(t.loc["overall", "n"]) for t in tabs.values()):
        suffix, groups = f" = {groups[0]}", []
    rows = ["overall"] + groups
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(rows) + 2.2), sharex=True)
    for ax, d in zip(axes, DS):
        t = tabs[d]
        left = np.zeros(len(rows))
        containers = []
        for nt in U.NT_ORDER:
            vals = np.array([float(t.loc[r, nt]) * 100 if r in t.index else 0.0 for r in rows])
            containers.append(ax.barh(range(len(rows)), vals, left=left, color=U.NT_COLORS[nt], label=NT_DISPLAY[nt], edgecolor="white", linewidth=0.3))
            left += vals
        labels = []
        for k, r in enumerate(rows):
            if r not in t.index:
                labels.append(f"{r} (none)")
                continue
            n_r = int(t.loc[r, "n"])
            small = n_r < SMALL_N
            same = (r != "overall" and "overall" in t.index and n_r == int(t.loc["overall", "n"])
                    and np.allclose(t.loc[r, U.NT_ORDER].astype(float).to_numpy(), t.loc["overall", U.NT_ORDER].astype(float).to_numpy()))
            labels.append(f"{r}{suffix if r == 'overall' else ''}{'*' if small else ''} (n={n_r:,}{', same as overall' if same else ''})")
            if small:
                for bc in containers:
                    bc.patches[k].set_alpha(0.35)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of output synapses, %" if weighted else "Share of neurons, %")
        ax.set_title(f"{U.LABEL[d]}")
    axes[0].set_ylabel("Neuron class (super_class)" if by == "super_class" else "Brain region group (region_group)")
    top = {d: tabs[d].loc["overall", U.NT_ORDER].astype(float).idxmax() for d in DS if "overall" in tabs[d].index}
    shares = {d: float(tabs[d].loc["overall", top[d]]) for d in top}
    what = "output synapses" if weighted else "neurons"
    if len(set(top.values())) == 1:
        title = (f"{pop.capitalize()} {what}: {next(iter(top.values()))} is the largest transmitter class in both datasets "
                 f"(FAFB {shares['fafb']:.0%}, MaleCNS {shares['malecns']:.0%})")
    else:
        title = (f"{pop.capitalize()} {what}: the largest transmitter class differs "
                 f"({top['fafb']} {shares['fafb']:.0%} in FAFB, {top['malecns']} {shares['malecns']:.0%} in MaleCNS)")
    unit = "output synapses (every neuron counts as many as it has output synapses)" if weighted else "neurons"
    group = "neuron class" if by == "super_class" else "brain region group"
    headline(axes[0], title, f"Each bar is 100% of the {unit} in one {group}; the colours show which transmitter they use. "
                             f"n = number of neurons in the group. (none) = the group does not exist in that dataset; faded rows have fewer than {SMALL_N} neurons.", base=26)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.05), frameon=False)
    return fig, tabs


# ----------------------------------------------------------------------------- C2
def c2_percentiles(neurons: pd.DataFrame) -> pd.DataFrame:
    c = neurons["nt_confidence"].dropna()
    return pd.Series({"n": len(c), "p10": c.quantile(0.10), "p25": c.quantile(0.25), "p50": c.quantile(0.50)})


def c2_figure(data: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, d in zip(axes, DS):
        n = data[d][0]
        n = n[n["nt_confidence"].notna()].copy()
        n["nt_type"] = n["nt_type"].astype(str)
        keep = n["nt_type"].value_counts()
        n = n[n["nt_type"].isin(keep[keep >= 5].index)]
        order = [t for t in U.NT_ORDER if t in set(n["nt_type"])]
        sns.violinplot(data=n, x="nt_type", y="nt_confidence", order=order, cut=0, inner="quartile",
                       hue="nt_type", hue_order=order, palette={t: U.NT_COLORS[t] for t in order},   # same colour per transmitter as in C1
                       legend=False, ax=ax, linewidth=0.8)
        ax.set_ylim(0, 1)
        ax.set_title(U.LABEL[d])
        ax.set_xlabel("Predicted transmitter (nt_type)")
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Classifier confidence (nt_confidence)\n0 = unsure, 1 = sure")
    med = {d: data[d][0]["nt_confidence"].median() for d in DS}
    headline(axes[0], "Transmitter confidence is not comparable across datasets: median "
             f"{med['fafb']:.2f} in FAFB vs {med['malecns']:.2f} in MaleCNS",
             "Each shape (violin) shows how sure the classifier is about the neurons it labelled with that transmitter. The wider the shape, the more neurons "
             "at that confidence; dashed lines mark 25%, 50% (median) and 75% of the neurons.", base=26)
    return fig


CLASSICAL = ["acetylcholine", "gaba", "glutamate", "dopamine", "serotonin", "octopamine", "histamine"]


def parse_known_nt(value) -> frozenset:
    """Classical transmitters asserted in an nt_known string. Co-transmitters (sNPF, tachykinin,
    nitric oxide) are ignored; '<x>-negative' removes x from the evidence instead of adding it."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return frozenset()
    s = re.sub(r"[a-z]+-negative", " ", str(value).lower())
    return frozenset(t for t in CLASSICAL if t in s)


def c2_agreement(fafb_neurons: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """nt_known vs nt_type for FAFB. Only neurons whose nt_known names exactly one classical
    transmitter are compared; the rest are counted, not compared."""
    n = fafb_neurons[fafb_neurons["nt_known"].notna()].copy()
    n["known_set"] = n["nt_known"].map(parse_known_nt)
    n["k"] = n["known_set"].map(lambda s: next(iter(s)) if len(s) == 1 else None)
    cover = pd.Series({"nt_known filled": len(n), "single classical transmitter": int(n["k"].notna().sum()),
                       "several transmitters named": int((n["known_set"].map(len) > 1).sum()),
                       "only a negative or co-transmitter": int((n["known_set"].map(len) == 0).sum())})
    one = n[n["k"].notna()].copy()
    one["agree"] = one["k"] == one["nt_type"].astype(str)
    per = one.groupby("k").agg(neurons=("agree", "size"), agree=("agree", "mean")).sort_values("neurons", ascending=False)
    cross = pd.crosstab(one["k"].rename("nt_known"), one["nt_type"].astype(str).rename("nt_type"))
    return cover.to_frame("count"), per, cross


# ----------------------------------------------------------------------------- C3
FAST = ["acetylcholine", "gaba", "glutamate"]


def c3_hemilineages(fafb_neurons: pd.DataFrame, min_n: int = 20) -> pd.DataFrame:
    n = fafb_neurons[fafb_neurons["hemilineage"].notna() & fafb_neurons["nt_type"].astype(str).isin(FAST)].copy()
    n["nt_type"] = n["nt_type"].astype(str)
    g = n.groupby("hemilineage")
    size = g.size()
    top = g["nt_type"].agg(lambda s: s.value_counts().index[0])
    share = g["nt_type"].agg(lambda s: s.value_counts().iloc[0] / len(s))
    out = pd.DataFrame({"neurons": size, "dominant": top, "share": share})
    return out[out["neurons"] >= min_n].sort_values("share")


def c3_figure(tab: pd.DataFrame):
    """Histogram of hemilineage purity. Written to explain itself: pure and mixed families are coloured and counted, a worked
    example is on the plot, and the three least pure families are named."""
    share = tab["share"].to_numpy(dtype=float)
    bins = np.linspace(0.3, 1.0, 15)                                    # the last edge is exactly 1.0, so purity 1.0 is counted
    counts, edges = np.histogram(share, bins=bins)
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    colours = ["#4d4d4d" if left >= 0.9 - 1e-9 else "#bdbdbd" for left in edges[:-1]]
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=colours, edgecolor="white")
    for left, c in zip(edges[:-1], counts):
        if c:
            ax.text(left + 0.025, c + counts.max() * 0.012, str(int(c)), ha="center", va="bottom", fontsize=8)
    ymax = counts.max() * 1.45
    ax.axvline(0.9, ymax=0.78, color="k", ls="--", lw=1)             # stops below the PURE label so the text is not crossed
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0, ymax)
    n_all, n_pure = len(share), int((share >= 0.9).sum())
    ax.text(0.95, ymax * 0.97, f"PURE: purity of 0.9 or more\n{n_pure} of {n_all} families ({n_pure / n_all:.0%})", ha="center", va="top", fontsize=9, fontweight="bold", color="#4d4d4d")
    ax.text(0.6, ymax * 0.60, f"MIXED: purity below 0.9\n{n_all - n_pure} families", ha="center", va="top", fontsize=9, fontweight="bold", color="#7a7a7a")
    ax.text(0.31, ymax * 0.97, "How to read this plot\n"
            "A hemilineage is a family of neurons born from the same progenitor.\n"
            "Purity = the share of a family's neurons that use its most common\n"
            "fast transmitter (acetylcholine, GABA or glutamate).\n"
            "Example: 10 neurons, 9 cholinergic and 1 GABA, purity 0.9.\n"
            "Each bar counts how many families have that purity.",
            ha="left", va="top", fontsize=8.5, bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc"))
    low = tab.sort_values("share").head(3)
    names = [f"{name}{'*' if name in ('primary', 'putative_primary') else ''} {row['share']:.2f}" for name, row in low.iterrows()]
    note = "Least pure: " + ", ".join(names)
    if any("*" in x for x in names):
        note += "\n* a catalogue label, not a true hemilineage"
    ax.text(0.31, ymax * 0.30, note, ha="left", va="top", fontsize=8, color="#555555")
    ax.set_xlabel("Purity of a hemilineage (share of its neurons using its most common fast transmitter)")
    ax.set_ylabel("Number of hemilineages")
    ax.set_title(f"{n_pure / n_all:.0%} of {n_all} hemilineages are pure: at least 90% of their neurons use one transmitter")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- C4
RULES = {"Rule 1: ACh = E; GABA, Glu = I": ({"acetylcholine"}, {"gaba", "glutamate"}),
         "Rule 2: ACh, Glu = E; GABA = I": ({"acetylcholine", "glutamate"}, {"gaba"})}


def c4_net_index(neurons: pd.DataFrame, edges: pd.DataFrame, tau: int = 10) -> pd.DataFrame:
    """Net index (E - I) / (E + I) per central postsynaptic neuron, under both rules."""
    e = edges[edges["syn_count"] >= tau]
    pre_nt = neurons.set_index("neuron_id")["nt_type"].astype(str)
    e = e.assign(nt=e["pre_id"].map(pre_nt))
    central = neurons[~neurons["is_optic"]]
    e = e[e["post_id"].isin(set(central["neuron_id"]))]
    rows = []
    for rule, (exc, inh) in RULES.items():
        E = e[e["nt"].isin(exc)].groupby("post_id")["syn_count"].sum()
        I = e[e["nt"].isin(inh)].groupby("post_id")["syn_count"].sum()
        t = pd.concat([E.rename("E"), I.rename("I")], axis=1).fillna(0)
        t = t[(t["E"] + t["I"]) > 0]
        t["net"] = (t["E"] - t["I"]) / (t["E"] + t["I"])
        t["rule"] = rule
        rows.append(t.reset_index().rename(columns={"post_id": "neuron_id"}))
    out = pd.concat(rows, ignore_index=True)
    meta = central.set_index("neuron_id")[["region_group", "neuropil"]]
    return out.join(meta, on="neuron_id")


def c4_sign_changes(net: pd.DataFrame, by: str) -> pd.DataFrame:
    med = net.groupby([by, "rule"])["net"].median().unstack("rule")
    r1, r2 = list(RULES)
    med["changes_sign"] = np.sign(med[r1]) != np.sign(med[r2])
    med["n"] = net.groupby(by)["neuron_id"].nunique()
    return med


def c4_figure(net: pd.DataFrame, by: str, top: int | None = None):
    d = net.copy()
    if top:
        keep = d.drop_duplicates("neuron_id")[by].value_counts().head(top).index
        d = d[d[by].isin(keep)]
        order = list(keep)
    else:
        order = U.ordered(d[by].dropna().unique(), U.GROUP_ORDER)
    fig, ax = plt.subplots(figsize=(max(9.0, 0.8 * len(order) + 3), 4.4))
    sns.boxplot(data=d, x=by, y="net", hue="rule", order=order, ax=ax, fliersize=1.5, linewidth=0.8,
                palette=["#4c78a8", "#f58518"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Brain region (neuropil)" if by == "neuropil" else "Brain region group (region_group)")
    ax.set_ylabel("Net input of a neuron, (E - I) / (E + I)\n+1 all excitatory, 0 balanced, -1 all inhibitory")
    ax.tick_params(axis="x", rotation=25)
    plain = {list(RULES)[0]: "Rule 1 (blue): glutamate counted as inhibitory\n(ACh excites; GABA and glutamate inhibit)",
             list(RULES)[1]: "Rule 2 (orange): glutamate counted as excitatory\n(ACh and glutamate excite; GABA inhibits)"}
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [plain.get(x, x) for x in labels], title=None, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    changes = c4_sign_changes(d, by)["changes_sign"].sum()
    noun = "neuropils" if by == "neuropil" else "region groups"
    headline(ax, f"The glutamate choice flips the median sign in {int(changes)} of {len(order)} {noun}",
             "Each box shows the net input of the neurons in one region: +1 means all their input is excitatory (E), -1 all inhibitory (I). "
             "The line inside a box is the median. Blue and orange are the same neurons, counted two ways.")
    return fig


# ----------------------------------------------------------------------------- C5
def mb_neuropils() -> list[str]:
    t = load_neuropil_table()
    return [n for n in t.index if n.startswith("MB_")]


MB_TYPE_PATTERN = r"^(?:KC|MBON|MBIN|MB|PAM|PPL\d|APL$|DPM$)|Kenyon|mushroom|\bDAN\b"


def subgraph_ids(neurons: pd.DataFrame, selection: str = "region") -> set:
    if selection == "region":
        return set(neurons.loc[neurons["neuropil"].isin(mb_neuropils()), "neuron_id"])
    rx = re.compile(MB_TYPE_PATTERN, re.IGNORECASE)
    ct = neurons["cell_type"].astype("string").fillna("")
    return set(neurons.loc[ct.map(lambda v: bool(rx.search(v))), "neuron_id"])


def sub_edges(edges: pd.DataFrame, ids: set, tau: int) -> pd.DataFrame:
    return edges[(edges["syn_count"] >= tau) & edges["pre_id"].isin(ids) & edges["post_id"].isin(ids)]


def c5_table(data: dict, taus=(1, 5, 10), selections=("region", "cell_type")) -> pd.DataFrame:
    rows = []
    for d in DS:
        neurons, edges = data[d]
        for sel in selections:
            ids = subgraph_ids(neurons, sel)
            n = len(ids)
            for tau in taus:
                e = sub_edges(edges, ids, tau)
                out_d = e.groupby("pre_id").size().reindex(list(ids), fill_value=0)
                in_d = e.groupby("post_id").size().reindex(list(ids), fill_value=0)
                connected = (out_d > 0) | (in_d > 0)
                rows.append({"dataset": U.LABEL[d], "selection": sel, "tau": tau, "neurons": n, "edges": len(e),
                             "density": len(e) / (n * (n - 1)) if n > 1 else np.nan,
                             "isolated": int((~connected).sum()),
                             "median_in": float(in_d.median()), "median_out": float(out_d.median()),
                             "median_in_connected": float(in_d[connected].median()) if connected.any() else np.nan,
                             "median_out_connected": float(out_d[connected].median()) if connected.any() else np.nan})
    return pd.DataFrame(rows)


def c5_ccdf_figure(data: dict, tau: int = 10):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colour = {"fafb": "#4c78a8", "malecns": "#e45756"}
    stats: dict = {}
    for d in DS:
        neurons, edges = data[d]
        ids = subgraph_ids(neurons)
        e = sub_edges(edges, ids, tau)
        for kind, col, style in (("in", "post_id", "-"), ("out", "pre_id", "--")):
            deg = e.groupby(col).size().reindex(list(ids), fill_value=0).to_numpy()
            x = np.sort(deg[deg > 0])
            if len(x) == 0:
                continue
            xs, counts = np.unique(x, return_counts=True)
            ccdf = (len(deg) - np.concatenate([[0], np.cumsum(counts)[:-1]]) - (deg == 0).sum()) / len(deg)
            ax.step(xs, ccdf, where="post", color=colour[d], ls=style, label=f"{U.LABEL[d]} {'incoming' if kind == 'in' else 'outgoing'} partners")
            stats.setdefault(d, {})[kind] = (float(np.median(deg)), int(deg.max()))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of partners of a neuron (degree, log scale)")
    ax.set_ylabel("Share of neurons with at least that many partners (log scale)")
    ax.legend(frameon=False, fontsize=8)
    parts = [f"{U.LABEL[d]} {int(stats[d]['in'][0])}/{int(stats[d]['out'][0])}, maximum {stats[d]['in'][1]:,}/{stats[d]['out'][1]:,}" for d in DS if d in stats]
    headline(ax, f"Mushroom body at tau = {tau}: median in/out-degree and maximum degree\n" + "; ".join(parts),
             "Each line answers: what share of neurons have at least this many partners? Solid = incoming connections, dashed = outgoing. "
             "A line that falls fast means most neurons have few partners; a long tail on the right means a few 'hub' neurons with very many partners.")
    return fig


# ----------------------------------------------------------------------------- C6
CONNECTED_TRIADS = ["021D", "021U", "021C", "111D", "111U", "030T", "030C", "201", "120D", "120U", "120C", "210", "300"]
MOTIFS = ["reciprocal pair"] + CONNECTED_TRIADS


TRIAD_NAMES = ["003", "012", "102", "021D", "021U", "021C", "111D", "111U", "030T", "030C", "201",
               "120D", "120U", "120C", "210", "300"]          # the order networkx and igraph both return


def _census(edges: pd.DataFrame, nodes) -> dict:
    """16-type triad census. igraph (C) if installed, else networkx (about 40 times slower; on the real
    mushroom body subgraph one census took 145 s). Both give identical counts (checked in the tests)."""
    try:
        import igraph as ig
    except ImportError:
        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(zip(edges["pre_id"].tolist(), edges["post_id"].tolist()))
        return dict(nx.triadic_census(g))
    idx = pd.Index(list(nodes))
    e = edges.drop_duplicates()
    arr = np.column_stack([idx.get_indexer(e["pre_id"]), idx.get_indexer(e["post_id"])])
    g = ig.Graph(n=len(idx), edges=arr.tolist(), directed=True)
    return dict(zip(TRIAD_NAMES, list(g.triad_census())))


def motif_counts(edges: pd.DataFrame, nodes) -> dict:
    pairs = set(zip(edges["pre_id"].tolist(), edges["post_id"].tolist()))
    recip = sum(1 for (a, b) in pairs if a < b and (b, a) in pairs)
    census = _census(edges, nodes)
    return {"reciprocal pair": recip, **{k: census[k] for k in CONNECTED_TRIADS}}


def _null_one(args):
    kept, nodes, seed = args
    return motif_counts(randomise_edges(kept[["pre_id", "post_id"]], seed=seed), nodes)


def c6_motifs(neurons: pd.DataFrame, edges: pd.DataFrame, tau: int = 10, n_seeds: int = 100,
              workers: int = 1, max_minutes: float | None = None, progress_path=None) -> pd.DataFrame:
    """Observed motif counts and the null from `randomise_edges` seeds 0..n_seeds-1.

    Progress goes to `progress_path` (one line per finished seed) so a long run can be watched from a
    second window. With `max_minutes`, no new seeds start after that time and the table says how many
    were used (tab.attrs['n_seeds']); the plan asks for 100, so report the number you actually ran."""
    import time
    from pathlib import Path
    ids = subgraph_ids(neurons)
    kept = sub_edges(edges, ids, tau)[["pre_id", "post_id"]].reset_index(drop=True)
    t0 = time.time()

    def note(msg):
        if progress_path:
            Path(progress_path).parent.mkdir(parents=True, exist_ok=True)
            with open(progress_path, "a", encoding="utf-8") as fh:
                fh.write(f"[{time.time() - t0:7.0f}s] {msg}\n")

    note(f"subgraph: {len(ids):,} neurons, {len(kept):,} edges at tau={tau}; planned seeds {n_seeds}, workers {workers}")
    obs = motif_counts(kept, ids)
    note("observed motif counts done")
    null, done = [], 0
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        while done < n_seeds:
            if max_minutes and (time.time() - t0) / 60 > max_minutes and done >= 2:
                note(f"time budget of {max_minutes} min reached: stopping at {done} seeds")
                break
            chunk = [(kept, ids, sd) for sd in range(done, min(done + max(workers, 1), n_seeds))]
            null += list(pool.map(_null_one, chunk)) if pool else [_null_one(chunk[0])]
            done += len(chunk)
            note(f"seed {done}/{n_seeds} finished")
    finally:
        if pool:
            pool.shutdown()
    nd = pd.DataFrame(null)
    out = pd.DataFrame({"observed": pd.Series(obs), "null_mean": nd.mean(), "null_sd": nd.std(ddof=1)})
    out["z"] = (out["observed"] - out["null_mean"]) / out["null_sd"].replace(0, np.nan)
    out = out.loc[MOTIFS]
    out.attrs["n_seeds"] = done
    return out


def c6_figure(tab: pd.DataFrame, tau: int = 10):
    fig, ax = plt.subplots(figsize=(9, 4))
    z = tab["z"].fillna(0).to_numpy()
    big = np.abs(tab["z"].fillna(0).to_numpy()) > 2
    for i, (zz, b) in enumerate(zip(z, big)):
        ax.bar(i, zz, color="#4c78a8" if b else "none", edgecolor="#4c78a8", linewidth=1.2)
    ax.axhline(0, color="k", lw=0.8)
    for y in (-2, 2):
        ax.axhline(y, color="grey", lw=0.6, ls=":")
    ax.set_xticks(range(len(tab)))
    ax.set_xticklabels(tab.index, rotation=45, ha="right")
    ax.set_xlabel("Wiring pattern among neurons (motif)")
    ax.set_yscale("symlog", linthresh=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylabel("How different from random (z-score)\ncompressed axis; above 0 = more than random")
    ax.set_ylim(min(float(z.min()) * 1.6, -5.0), max(float(z.max()) * 1.6, 5.0))      # room for the longest bar
    headline(ax, f"{int(big.sum())} of {len(tab)} motifs differ from the degree-preserving null by more than 2 SD",
             "Each bar is one way two or three neurons can be wired together. A bar above 0 means the pattern is more common than in random networks where every "
             "neuron keeps its number of partners; below 0 means rarer; an empty bar means no clear difference.\n"
             "Pattern code = number of two-way / one-way / missing links among the three neurons (201: two two-way links, no one-way link, one missing); "
             "the letters D, U, C, T give the direction (down, up, cyclic, transitive).")
    return fig


# ----------------------------------------------------------------------------- C2b (supplementary)
def c2b_consensus(malecns_neurons: pd.DataFrame, consensus_path) -> pd.DataFrame | None:
    """predictedNt (nt_type) against consensusNt for the male CNS. consensusNt is not in the neurons
    schema; fetch_malecns saves it next to the cache. Returns None if that file is missing."""
    from pathlib import Path
    p = Path(consensus_path)
    if not p.exists():
        return None
    c = pd.read_parquet(p)
    c["consensus"] = c["consensusNt"].astype("string").str.lower().replace({"unclear": "unknown"}).fillna("unknown")
    m = malecns_neurons.merge(c[["neuron_id", "consensus"]], on="neuron_id", how="left")
    return pd.crosstab(m["nt_type"].astype(str).rename("predictedNt (nt_type)"),
                       m["consensus"].fillna("missing").rename("consensusNt"))
