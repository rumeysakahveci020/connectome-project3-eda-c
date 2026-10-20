"""Check every output of the EDA against the cache and the plan. Prints PASS / WARN / FAIL, exits 1 on any FAIL.

    python tools/verify_outputs.py                 # real run
    python tools/verify_outputs.py --no-step0      # skip the real-data counts (synthetic dry run)
    python tools/verify_outputs.py --zip project3_eda_C.zip   # also inspect the hand-in zip

What it does NOT replace: looking at the figures yourself. What it adds: every number in the JSON and the
report is recomputed from the cache, stale figures are detected, the C2 colours are tested, the notebooks are
checked for errors, and the zip is checked for secrets.
"""
import argparse
import json
import re
import sys
import traceback
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from src import eda_c as C, eda_utils as U  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--no-step0", action="store_true")
ap.add_argument("--expect-seeds", type=int, default=100)
ap.add_argument("--zip", default=None)
args = ap.parse_args()

RESULTS = []


def check(name, ok, detail="", warn=False):
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    RESULTS.append((status, name, detail))
    print(f"[{status}] {name}" + (f": {detail}" if detail else ""), flush=True)


def section(title):
    print(f"\n=== {title}", flush=True)


def guarded(fn):
    def run():
        try:
            fn()
        except Exception as exc:                                   # noqa: BLE001
            check(fn.__name__, False, f"crashed: {type(exc).__name__}: {str(exc)[:300]}")
            traceback.print_exc(limit=2)
    return run


def close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


# ------------------------------------------------------------------ A. files and freshness
def expected_figures():
    names = []
    for pop in ("central", "optic"):
        for by in ("super_class", "region_group"):
            for weighted in (False, True):
                names.append(U.fig_name("C1", "both", pop, extra=by + ("_weighted" if weighted else "")))
    names += [U.fig_name("C2", "both", "all"), U.fig_name("C3", "fafb", "all"),
              U.fig_name("C4", "fafb", "central", tau=10, extra="region_group"),
              U.fig_name("C4", "fafb", "central", tau=10, extra="neuropil12"),
              U.fig_name("C5", "both", "all", tau=10, extra="degree_ccdf"), U.fig_name("C6", "fafb", "all", tau=10)]
    return names


@guarded
def files_and_freshness():
    section("A. figures, report, freshness")
    figs = expected_figures()
    missing = [n for n in figs if not (U.FIG_DIR / n).exists()]
    check("all 14 figures exist in figures/eda", not missing, f"missing: {missing}" if missing else "")
    extra = sorted(p.name for p in U.FIG_DIR.glob("*.png") if p.name not in figs) if U.FIG_DIR.exists() else []
    check("no unexpected figure files", not extra, f"extra: {extra}" if extra else "", warn=True)
    caches = list(config.CACHE_DIR.glob("*.parquet"))
    newest_cache = max((p.stat().st_mtime for p in caches), default=0)
    from PIL import Image
    fp = U.code_fingerprint()

    def drawn_by(path):
        try:
            return Image.open(path).info.get("Comment", "") == f"code_sha={fp}"
        except Exception:                                    # noqa: BLE001
            return False
    stale_cache = [n for n in figs if (U.FIG_DIR / n).exists() and (U.FIG_DIR / n).stat().st_mtime < newest_cache]
    stale_code = [n for n in figs if (U.FIG_DIR / n).exists() and not drawn_by(U.FIG_DIR / n)]
    check("no figure is older than the cache", not stale_cache, f"older: {stale_cache}" if stale_cache else "")
    check("every figure was drawn by the current src/eda_c.py and src/eda_utils.py (code fingerprint in the PNG)", not stale_code,
          f"drawn by other code: {stale_code}. Run: python tools/rebuild_all.py --skip-build" if stale_code else "")
    for f in ("reports/eda_C.md", "reports/eda_C_numbers.json", "README.md"):
        check(f"{f} exists", (ROOT / f).exists())


# ------------------------------------------------------------------ B. Step 0
@guarded
def step0():
    section("B. Step 0 counts (Data Conventions)")
    if args.no_step0:
        print("skipped (--no-step0)")
        return
    n_f = pd.read_parquet(config.cache_path("fafb", "neurons", "783", f"ann{config.FAFB_ANNOTATIONS_TAG[1:]}"), columns=["neuron_id"])
    check("FAFB neurons = 139,255 (the proofread list; the plan's 139,248 is the annotation file)", len(n_f) == 139_255, f"{len(n_f):,}")
    sc = pd.read_parquet(config.cache_path("fafb", "edges", "783", config.FAFB_SYNAPSE_TABLE, min_syn=1), columns=["syn_count"])["syn_count"]
    check("FAFB edges >= 5 synapses = 2,700,513", int((sc >= 5).sum()) == 2_700_513, f"{int((sc >= 5).sum()):,}")
    n_m = pd.read_parquet(config.cache_path("malecns", "neurons", "v1.0", "traced"), columns=["neuron_id"])
    check("male CNS neurons = 165,122", len(n_m) == 165_122, f"{len(n_m):,}")
    sm = pd.read_parquet(config.cache_path("malecns", "edges", "v1.0", "traced", min_syn=1, conf=config.MALECNS_SYN_CONFIDENCE), columns=["syn_count"])["syn_count"]
    print(f"[INFO] male CNS edges >= 5 synapses: {int((sm >= 5).sum()):,} (agree one number with Projects 1 and 4)")


# ------------------------------------------------------------------ C. validators
@guarded
def validators():
    section("C. validators and interchangeability")
    from src.validators import assert_interchangeable, validate_edges, validate_neurons
    T = {}
    for d, (rel, nsrc, esrc, kw) in {"fafb": ("783", f"ann{config.FAFB_ANNOTATIONS_TAG[1:]}", config.FAFB_SYNAPSE_TABLE, {}),
                                     "malecns": ("v1.0", "traced", "traced", {"conf": config.MALECNS_SYN_CONFIDENCE})}.items():
        n = pd.read_parquet(config.cache_path(d, "neurons", rel, nsrc))
        e = pd.read_parquet(config.cache_path(d, "edges", rel, esrc, min_syn=1, **kw))
        validate_neurons(n)
        validate_edges(e, n)
        check(f"{d}: validate_neurons and validate_edges pass", True, f"{len(n):,} neurons, {len(e):,} edges")
        T[d] = (n, e)
    assert_interchangeable(T["fafb"][0], T["malecns"][0])
    assert_interchangeable(T["fafb"][1], T["malecns"][1])
    check("assert_interchangeable passes", True)


# ------------------------------------------------------------------ D. numbers recomputed from the cache
DATA = {}


@guarded
def recompute():
    section("D. every number in eda_C_numbers.json recomputed from the cache")
    R = json.load(open(ROOT / "reports" / "eda_C_numbers.json", encoding="utf-8"))
    for d in C.DS:
        DATA[d] = U.load(d)
    check("C6 used the planned number of null graphs", R["C6"]["n_seeds"] == args.expect_seeds, f"{R['C6']['n_seeds']} (plan: {args.expect_seeds})")
    for pop in ("central", "optic"):
        for d in C.DS:
            tab = C.c1_shares(*DATA[d], pop, "super_class", False).loc["overall"]
            bad = [k for k in U.NT_ORDER if not close(tab[k], R["C1"][pop][d][k], 1e-9)]
            check(f"C1 {pop} {d} shares match the cache", not bad, f"differ: {bad}" if bad else "")
    for d in C.DS:
        tab = C.c1_shares(*DATA[d], "optic", "super_class", True).loc["overall"]
        bad = [k for k in U.NT_ORDER if not close(tab[k], R["C1_weighted"]["optic"][d][k], 1e-9)]
        check(f"C1 optic weighted {d} shares match the cache", not bad, f"differ: {bad}" if bad else "")
    for d in C.DS:
        n = DATA[d][0]
        p50 = float(C.c2_percentiles(n)["p50"])
        b06 = float((n["nt_confidence"].dropna() < 0.6).mean())
        check(f"C2 {d}: median and 0.6-cutoff share match", close(p50, R["C2"][d]["p50"]) and close(b06, R["C2"][d]["below_06"]), f"p50 {p50:.4f}, below 0.6 {b06:.4f}")
    check("C3 hemilineage count matches", len(C.c3_hemilineages(DATA["fafb"][0])) == R["C3"]["n"], f"{R['C3']['n']}")
    tab5 = C.c5_table(DATA)
    for d in C.DS:
        r = tab5[(tab5["dataset"] == U.LABEL[d]) & (tab5["tau"] == 10) & (tab5["selection"] == "region")].iloc[0]
        ok = (int(r["neurons"]), int(r["edges"]), int(r["isolated"])) == (R["C5"][d]["neurons"], R["C5"][d]["edges"], R["C5"][d]["isolated"])
        check(f"C5 {d} tau=10 neurons, edges, isolated match", ok, f"{int(r['neurons']):,}, {int(r['edges']):,}, {int(r['isolated'])}")
    ids = C.subgraph_ids(DATA["fafb"][0])
    kept = C.sub_edges(DATA["fafb"][1], ids, 10)[["pre_id", "post_id"]]
    obs = C.motif_counts(kept, ids)["reciprocal pair"]
    check("C6 observed reciprocal pairs match", int(obs) == R["C6"]["obs_recip"], f"{int(obs):,}")
    check("C4: 12 neuropils compared", R["C4"]["n_neuropils"] == 12, f"{R['C4']['n_neuropils']}", warn=True)


@guarded
def independent():
    """The checks in D reuse the project's own functions, so a bug inside one of them would pass. These recompute the same
    results with different code (plain numpy, scipy, python loops), so such a bug shows up."""
    section("D2. independent recomputation (different code path from the project's functions)")
    R = json.load(open(ROOT / "reports" / "eda_C_numbers.json", encoding="utf-8"))
    for d in C.DS:
        DATA.setdefault(d, U.load(d))
    # C1: central acetylcholine share by plain counting
    for d in C.DS:
        n = DATA[d][0]
        c = n[~n["is_optic"].astype(bool)]
        by_hand = float((c["nt_type"].astype(str).to_numpy() == "acetylcholine").sum()) / len(c)
        check(f"C1 central {d}: acetylcholine share by counting", close(by_hand, R["C1"]["central"][d]["acetylcholine"], 1e-9), f"{by_hand:.4f}")
    check("C5 mushroom body neuropils are calyx, peduncle and the two lobes", sorted(C.mb_neuropils()) == ["MB_CA", "MB_ML", "MB_PED", "MB_VL"], str(sorted(C.mb_neuropils())), warn=True)
    # C5: edges and isolated neurons of the mushroom body subgraph with numpy only
    for d in C.DS:
        n, e = DATA[d]
        mb = n.loc[n["neuropil"].isin(C.mb_neuropils()), "neuron_id"].to_numpy()
        pre, post, w = e["pre_id"].to_numpy(), e["post_id"].to_numpy(), e["syn_count"].to_numpy()
        m = (w >= 10) & np.isin(pre, mb) & np.isin(post, mb)
        touched = np.unique(np.concatenate([pre[m], post[m]]))
        check(f"C5 {d}: tau=10 edges and isolated neurons with numpy", (int(m.sum()), int(len(mb) - len(touched))) == (R["C5"][d]["edges"], R["C5"][d]["isolated"]),
              f"{int(m.sum()):,} edges, {int(len(mb) - len(touched))} isolated")
    # C6: reciprocal pairs from a sparse adjacency matrix
    from scipy.sparse import csr_matrix
    n, e = DATA["fafb"]
    mb = n.loc[n["neuropil"].isin(C.mb_neuropils()), "neuron_id"].to_numpy()
    k = (e["syn_count"].to_numpy() >= 10) & np.isin(e["pre_id"].to_numpy(), mb) & np.isin(e["post_id"].to_numpy(), mb)
    idx = {v: i for i, v in enumerate(mb)}
    r = [idx[v] for v in e["pre_id"].to_numpy()[k]]
    c = [idx[v] for v in e["post_id"].to_numpy()[k]]
    A = csr_matrix((np.ones(len(r)), (r, c)), shape=(len(mb), len(mb)))
    A.data[:] = 1
    A.setdiag(0)
    A.eliminate_zeros()
    recip = int(A.multiply(A.T).nnz // 2)
    check("C6 observed reciprocal pairs from a sparse matrix", recip == R["C6"]["obs_recip"], f"{recip:,}")
    # C4: net index of 300 random neurons under both rules by explicit loops
    from src import eda_c as C_
    n, e = DATA["fafb"]
    net = C_.c4_net_index(n, e, 10)
    nt = dict(zip(n["neuron_id"].to_numpy(), n["nt_type"].astype(str).to_numpy()))
    e10 = e[e["syn_count"] >= 10]
    sample = pd.concat([g.sample(min(150, len(g)), random_state=0) for _, g in net.groupby("rule")])
    inc = {k_: g for k_, g in e10[e10["post_id"].isin(set(sample["neuron_id"]))].groupby("post_id")}
    # the plan's two rules, written out here on purpose (not read from the module): Rule 1 ACh = E; GABA, Glu = I. Rule 2 ACh, Glu = E; GABA = I
    names = list(C_.RULES)
    rules = {names[0]: ({"acetylcholine"}, {"gaba", "glutamate"}), names[1]: ({"acetylcholine", "glutamate"}, {"gaba"})}
    bad = 0
    for _, row in sample.iterrows():
        g = inc[row["neuron_id"]]
        exc, inh = rules[row["rule"]]
        E = sum(w for p_, w in zip(g["pre_id"], g["syn_count"]) if nt.get(p_) in exc)
        I = sum(w for p_, w in zip(g["pre_id"], g["syn_count"]) if nt.get(p_) in inh)
        if E + I == 0 or abs((E - I) / (E + I) - row["net"]) > 1e-9:
            bad += 1
    check(f"C4 net index of {len(sample)} random neurons by explicit loops", bad == 0, f"{bad} differ" if bad else "")
    # C3: five hemilineages by a python loop
    tab = C.c3_hemilineages(n)
    fast = n[n["hemilineage"].notna() & n["nt_type"].astype(str).isin(["acetylcholine", "gaba", "glutamate"])]
    bad = 0
    for h in tab.index[: min(5, len(tab))]:
        vals = fast.loc[fast["hemilineage"] == h, "nt_type"].astype(str).tolist()
        top = max(vals.count(t) for t in set(vals)) / len(vals)
        if abs(top - float(tab.loc[h, "share"])) > 1e-12:
            bad += 1
    check("C3 shares of five hemilineages by a python loop", bad == 0, f"{bad} differ" if bad else "")
    figc3 = C.c3_figure(tab)
    heights = sum(p.get_height() for p in figc3.axes[0].patches)
    check("C3 histogram bars add up to the number of hemilineages", int(heights) == len(tab), f"{int(heights)} bars vs {len(tab)} hemilineages")


# ------------------------------------------------------------------ E. colours
def violin_colours_ok(fig):
    """Each violin must be closest to its own transmitter's colour (same colour per transmitter in every figure)."""
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgb
    palette = {k: np.array(to_rgb(v)) for k, v in U.NT_COLORS.items() if k != "unknown"}
    problems = []
    for ax in fig.axes:
        labels = [t.get_text() for t in ax.get_xticklabels()]
        bodies = [c for c in ax.collections if isinstance(c, PolyCollection) and len(c.get_paths())]
        pos = list(ax.get_xticks())
        for c in bodies:
            xs = np.concatenate([p.vertices[:, 0] for p in c.get_paths()])
            centre = float(np.mean(xs))
            i = int(np.argmin([abs(centre - p) for p in pos]))
            label = labels[i] if i < len(labels) else None
            rgb = np.array(c.get_facecolor()[0][:3])
            nearest = min(palette, key=lambda k: float(np.linalg.norm(palette[k] - rgb)))
            if label in palette and nearest != label:
                problems.append(f"{ax.get_title()}:{label}->{nearest}")
    return problems


@guarded
def colours():
    section("E. colours (the plan: same colour for the same transmitter in every figure)")
    fig = C.c2_figure(DATA or {d: U.load(d) for d in C.DS})
    bad = violin_colours_ok(fig)
    check("C2 violin colours equal the C1 transmitter colours", not bad, "; ".join(bad) if bad else "")


@guarded
def c1_rows():
    section("E2. C1 figures: same rows in both panels, tiny rows marked")
    data = DATA or {d: U.load(d) for d in C.DS}
    for pop in ("central", "optic"):
        for by in ("super_class", "region_group"):
            for weighted in (False, True):
                fig, _ = C.c1_figure(data, pop, by, weighted)
                left, right = [[t.get_text() for t in ax.get_yticklabels()] for ax in fig.axes[:2]]
                rows = lambda lab: [x.split(" (")[0].split(" =")[0].rstrip("*") for x in lab]   # noqa: E731
                ok = rows(left) == rows(right)
                check(f"C1 {pop} {by}{' weighted' if weighted else ''}: both panels list the same rows", ok, f"{rows(left)} vs {rows(right)}" if not ok else "")
                import matplotlib.pyplot as plt
                plt.close(fig)


# ------------------------------------------------------------------ F. report consistency
@guarded
def report():
    section("F. the one-page summary agrees with the numbers")
    R = json.load(open(ROOT / "reports" / "eda_C_numbers.json", encoding="utf-8"))
    text = (ROOT / "reports" / "eda_C.md").read_text(encoding="utf-8")
    def pc(x):
        return f"{100 * x:.0f}%"
    c1f, c1m = R["C1"]["central"]["fafb"], R["C1"]["central"]["malecns"]
    c5f, c5m = R["C5"]["fafb"], R["C5"]["malecns"]
    # whole phrases, so a number that appears somewhere else in the text cannot hide a wrong one
    want = {
        "C1 central acetylcholine (FAFB / male CNS)": f"{pc(c1f['acetylcholine'])} in FAFB and {pc(c1m['acetylcholine'])} in the male CNS",
        "C2 medians": f"median confidence is {R['C2']['fafb']['p50']:.2f} in FAFB and {R['C2']['malecns']['p50']:.2f} in the male CNS",
        "C3 hemilineage count": f"of {R['C3']['n']} hemilineages",
        "C5 FAFB neurons, edges, isolated": f"{c5f['neurons']:,} neurons, {c5f['edges']:,} edges and {c5f['isolated']} isolated",
        "C5 male CNS neurons, edges, isolated": f"{c5m['neurons']:,}, {c5m['edges']:,}, {c5m['isolated']}",
        "C6 observed vs expected reciprocal pairs": f"{R['C6']['obs_recip']:,} vs {R['C6']['null_recip']:,.0f}",
        "C6 number of null graphs": f"{R['C6']['n_seeds']} null graphs",
    }
    for name, phrase in want.items():
        check(f"report states {name}", phrase in text, f"expected: '{phrase}'" if phrase not in text else "")
    check("report states the optic population", "Optic:" in text)
    check("report has no placeholder text", "[write" not in text and "TODO" not in text)
    one_page = len(text.split()) <= 640
    check("report fits one page (about 640 words at most)", one_page, f"{len(text.split())} words", warn=True)


# ------------------------------------------------------------------ G. notebooks
@guarded
def notebooks():
    section("G. notebooks are executed and clean")
    for name, need_images in (("eda_C.ipynb", 14), ("00_integration.ipynb", 0)):
        path = ROOT / "notebooks" / name
        nb = json.load(open(path, encoding="utf-8"))
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        empty = [i for i, c in enumerate(code, 1) if not c.get("outputs") and i > 1]
        errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
        images = sum(1 for c in code for o in c.get("outputs", []) if "data" in o and "image/png" in o["data"])
        check(f"{name}: every code cell has saved output", not empty,
              f"empty cells: {empty}. The notebook was not run to the end (interrupted, or still running in VS Code). Close it without saving and run: python tools/rebuild_all.py --skip-build" if empty else "")
        check(f"{name}: no error output", not errors)
        if need_images:
            check(f"{name}: at least {need_images} embedded images", images >= need_images, f"{images}")
        if name == "00_integration.ipynb":
            text = json.dumps(nb)
            check("00_integration prints 'assert_interchangeable passed'", "assert_interchangeable passed" in text)


# ------------------------------------------------------------------ H. zip
@guarded
def the_zip():
    section("H. hand-in zip")
    path = Path(args.zip) if args.zip else ROOT / "project3_eda_C.zip"
    if not path.exists():
        print(f"no zip at {path}, skipped")
        return
    names = zipfile.ZipFile(path).namelist()
    secrets = [n for n in names if re.search(r"(^|/)\.env$", n) or re.match(r"^data/", n) or "/data/raw/" in n]
    check("zip contains no .env and no data folder", not secrets, f"found: {secrets[:5]}" if secrets else "")
    for need in ("README.md", "config.py", "reports/eda_C.md"):
        check(f"zip contains {need}", need in names)
    check("zip contains eda_shared (unchanged, as the plan's layout)", any(n.startswith("eda_shared/") for n in names))
    figs = [n for n in names if n.startswith("figures/eda/") and n.endswith(".png")]
    check("zip contains the 14 figures", len(figs) == 14, f"{len(figs)}")


for step in (files_and_freshness, step0, validators, recompute, independent, colours, c1_rows, report, notebooks, the_zip):
    step()

print("\n=== SUMMARY")
counts = {s: sum(1 for r in RESULTS if r[0] == s) for s in ("PASS", "WARN", "FAIL")}
print(counts)
for s, n, d in RESULTS:
    if s == "FAIL":
        print(f"  FAIL: {n} {d}")
sys.exit(1 if counts["FAIL"] else 0)
