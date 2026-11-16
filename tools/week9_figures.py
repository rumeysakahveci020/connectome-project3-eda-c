"""Figures and tables of the Week 9 naive result, from results/week9/runs.jsonl only (no network, no cache).

    python tools/week9_figures.py                # writes figures/week9/*.png and results/week9/tables/*.csv

fig1_learning_curves   the headline: test accuracy against training copies per pattern (log axis), mean and one standard deviation over seeds, one panel per dataset
fig2_curves_by_P       the same for every P of the grid
fig3_violations        share of signed weights with the wrong sign after the optimiser step and before the clipping, per epoch
fig4_parameters        nominal against effective parameter counts
fig5_ablation          test accuracy when a fraction of the units is ablated (same ablation sets across conditions within a seed)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"masked_sign": "#1f5f99", "dense_params": "#c4500a", "dense_units": "#6b8e23"}
LABELS = {"masked_sign": "masked + sign (biological)", "dense_params": "dense, same parameters (H units)", "dense_units": "dense, same units (N units)"}
DS = {"fafb": "FlyWire", "malecns": "male CNS"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def load(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def band(ax, x, groups, cond, **kw):
    m = groups.mean()
    s = groups.std(ddof=1).fillna(0)
    ax.plot(m.index, m.values, "-o", color=COLORS[cond], label=LABELS[cond], **kw)
    ax.fill_between(m.index, (m - s).values, (m + s).values, color=COLORS[cond], alpha=0.18)


def curves(df, P, ax, ds):
    d = df[(df.dataset == ds) & (df.P == P)]
    for cond in [c for c in COLORS if c in set(d.condition)]:
        band(ax, None, d[d.condition == cond].groupby("n")["test_acc"], cond)
    ax.plot(sorted(d.n.unique()), [d.groupby("n")["majority_test_acc"].mean().loc[n] for n in sorted(d.n.unique())], "--", color="#7a828c", label="trivial: commonest training label")
    ax.set_xscale("log"); ax.set_xlabel("training copies per pattern"); ax.set_ylim(0.4, 1.02)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "results" / "week9" / "runs.jsonl"))
    ap.add_argument("--fig", default=str(ROOT / "figures" / "week9"))
    ap.add_argument("--headline-P", type=int, default=40)
    ap.add_argument("--cell", type=int, nargs=2, default=[40, 10], help="the (P, n) cell for the violation, parameter and ablation figures")
    ap.add_argument("--threshold", type=float, default=0.9, help="test accuracy that defines sample efficiency")
    a = ap.parse_args(argv)
    df = load(Path(a.runs))
    fig_dir, tab_dir = Path(a.fig), Path(a.runs).parent / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    datasets = [d for d in DS if d in set(df.dataset)]

    summ = df.groupby(["dataset", "condition", "P", "n"]).agg(seeds=("seed", "nunique"), test_acc_mean=("test_acc", "mean"), test_acc_sd=("test_acc", lambda x: x.std(ddof=1)),
                                                              val_acc_mean=("val_acc", "mean"), epochs_mean=("epochs_run", "mean"), majority_mean=("majority_test_acc", "mean")).reset_index()
    summ.to_csv(tab_dir / "summary.csv", index=False)
    eff = []
    for (ds, cond, P), g in summ.groupby(["dataset", "condition", "P"]):
        ok = g[g.test_acc_mean >= a.threshold].sort_values("n")
        eff.append({"dataset": ds, "condition": cond, "P": P, f"smallest_n_reaching_{a.threshold}": int(ok.n.iloc[0]) if len(ok) else None})
    pd.DataFrame(eff).to_csv(tab_dir / "sample_efficiency.csv", index=False)

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 3.8), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        curves(df, a.headline_P, ax, ds)
        ax.set_title(f"{DS[ds]}, P = {a.headline_P} patterns")
    axes[0][0].set_ylabel("test accuracy")
    axes[0][-1].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Look at the gap between the lines and at how far each sits above the dashed trivial line", fontsize=10, y=0.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(fig_dir / "fig1_learning_curves.png", dpi=200); plt.close(fig)

    Ps = sorted(df.P.unique())
    fig, axes = plt.subplots(len(datasets), len(Ps), figsize=(3.0 * len(Ps), 2.9 * len(datasets)), squeeze=False, sharey=True)
    for i, ds in enumerate(datasets):
        for j, P in enumerate(Ps):
            curves(df, P, axes[i][j], ds)
            axes[i][j].set_title(f"{DS[ds]}, P = {P}", fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(fig_dir / "fig2_curves_by_P.png", dpi=170); plt.close(fig)

    cell = df[(df.P == a.cell[0]) & (df.n == a.cell[1])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 3.6), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        d = cell[(cell.dataset == ds) & (cell.condition == "masked_sign")]
        if len(d):
            L = min(len(v) for v in d.violation_per_epoch)
            M = np.array([v[:L] for v in d.violation_per_epoch])
            ax.plot(range(1, L + 1), M.mean(0), color=COLORS["masked_sign"]); ax.fill_between(range(1, L + 1), M.mean(0) - M.std(0, ddof=1), M.mean(0) + M.std(0, ddof=1), alpha=0.18, color=COLORS["masked_sign"])
        ax.set_title(f"{DS[ds]} (P = {a.cell[0]}, n = {a.cell[1]})"); ax.set_xlabel("epoch"); ax.set_ylabel("wrong-signed share of signed weights\n(after the step, before clipping)")
    fig.tight_layout(); fig.savefig(fig_dir / "fig3_violations.png", dpi=200); plt.close(fig)

    par = cell.groupby(["dataset", "condition"])[["nominal_params", "mask_weights", "effective_nonzero", "units"]].mean().reset_index()
    par.to_csv(tab_dir / "parameter_counts.csv", index=False)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 3.6), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        p = par[par.dataset == ds].set_index("condition")
        x = np.arange(len(p))
        for k, (col, lab, c) in enumerate([("nominal_params", "nominal (stored)", "#c9ced4"), ("mask_weights", "inside the mask", "#8fb3d6"), ("effective_nonzero", "non-zero after training", "#1f5f99")]):
            ax.bar(x + (k - 1) * 0.26, p[col], 0.26, label=lab, color=c)
        ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([c.replace("_", "\n") for c in p.index], fontsize=9); ax.set_title(DS[ds]); ax.set_ylabel("weights (log)")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "fig4_parameters.png", dpi=200); plt.close(fig)

    rows = []
    for _, r in cell.iterrows():
        for f, acc in r["ablation_test_acc"].items():
            rows.append({"dataset": r["dataset"], "condition": r["condition"], "seed": r["seed"], "fraction": float(f), "test_acc": acc})
    ab = pd.DataFrame(rows)
    ab.groupby(["dataset", "condition", "fraction"])["test_acc"].agg(["mean", "std"]).reset_index().to_csv(tab_dir / "ablation.csv", index=False)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 3.6), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        for cond in [c for c in COLORS if c in set(ab[ab.dataset == ds].condition)]:
            band(ax, None, ab[(ab.dataset == ds) & (ab.condition == cond)].groupby("fraction")["test_acc"], cond)
        ax.set_title(DS[ds]); ax.set_xlabel("fraction of units ablated"); ax.set_ylim(0.4, 1.02)
    axes[0][0].set_ylabel("test accuracy"); axes[0][-1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "fig5_ablation.png", dpi=200); plt.close(fig)
    print("wrote", sorted(p.name for p in fig_dir.iterdir()), "and", sorted(p.name for p in tab_dir.iterdir()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
