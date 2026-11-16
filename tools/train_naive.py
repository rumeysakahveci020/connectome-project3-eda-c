"""Train the networks of the ladder on the mushroom-body subgraph, exactly as the lock note describes.

    python tools/train_naive.py --dataset fafb --condition all --tau 5 --P 40 --n 10 --lr 1e-3 --seed 0 --approved

STOP: the professor asked not to run the naive model before the lock note is approved. On the real caches this script refuses to run
unless you pass --approved, and you should pass it only after the approval. On a synthetic cache (EDA_CACHE_DIR set, see
tools/make_synthetic_raw.py) it runs without the flag: use that to test your setup.

--condition one of dense, random_sparse, degree_preserving, true_mask, or all (the seven rungs: dense without sign, the other three
with and without the sign constraint). --lr-grid picks the learning rate on the validation loss, the same grid for every condition.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src import eda_utils as U  # noqa: E402
from src.model.graphs import CONDITIONS, RUNGS, build_units, make_condition, time_steps  # noqa: E402
from src.model.net import MaskedRNN, train  # noqa: E402
from src.model.task import make_task  # noqa: E402

LR_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


def require_approval(approved: bool, synthetic: bool) -> None:
    if not synthetic and not approved:
        raise SystemExit("STOP: the lock note is not approved yet, so the naive model must not run on the real tables.\n"
                         "Pass --approved only after the approval (or test on a synthetic cache: set EDA_CACHE_DIR).")


def run_rung(g_true, condition: str, sign_on: bool, T: int, task: dict, seed: int, lrs, epochs: int, patience: int, batch: int) -> dict:
    g = make_condition(g_true, condition, seed)
    best = None
    for lr in lrs:
        model = MaskedRNN(g, T, seed, sign_constrained=sign_on)
        res = train(model, task, lr, seed, epochs, patience, batch)
        res["lr"] = lr
        if best is None or res["best_val_loss"] < best["best_val_loss"]:
            best = res
    best.update({"condition": condition, "sign_constrained": sign_on, "units": g.n, "edges": g.n_edges, "T": T})
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["fafb", "malecns"], required=True)
    ap.add_argument("--condition", choices=[*CONDITIONS, "all"], default="all")
    ap.add_argument("--sign", choices=["on", "off", "both"], default="both")
    ap.add_argument("--tau", type=int, default=5)
    ap.add_argument("--glutamate", choices=["inhibitory", "excitatory"], default="inhibitory")
    ap.add_argument("--selection", choices=["union", "region", "cell_type"], default="union")
    ap.add_argument("--P", type=int, default=40)
    ap.add_argument("--n", type=int, default=10, help="training copies per pattern")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=None, help="one learning rate (default: the grid)")
    ap.add_argument("--lr-grid", action="store_true", help="use the lock-note grid and pick the learning rate on the validation loss")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--T", type=int, default=None, help="time steps (default: from the shortest Kenyon cell to MBON paths of both datasets)")
    ap.add_argument("--out", default=str(ROOT / "results" / "naive"))
    ap.add_argument("--approved", action="store_true")
    a = ap.parse_args(argv)
    require_approval(a.approved, synthetic=bool(os.environ.get("EDA_CACHE_DIR")))
    U._cache_dir()
    g_true = build_units(*U.load(a.dataset), a.tau, a.selection, a.glutamate)
    if a.T:
        T = a.T
    else:
        T = max(time_steps(build_units(*U.load(d), a.tau, a.selection, a.glutamate)) for d in ("fafb", "malecns"))
    lrs = LR_GRID if a.lr_grid else ((a.lr,) if a.lr else (1e-3,))
    print(f"{a.dataset}: {g_true.n:,} units, {g_true.n_edges:,} edges at tau = {a.tau}, {len(g_true.kc):,} input and {len(g_true.mbon):,} output units, T = {T}, "
          f"isolated units removed: {g_true.meta['isolated_removed']:,}; learning rates: {list(lrs)}", flush=True)
    task = make_task(a.seed, a.P, a.n)
    rungs = RUNGS if a.condition == "all" else [(a.condition, s) for s in ((False, True) if a.sign == "both" else (a.sign == "on",)) if not (a.condition == "dense" and s)]
    if a.sign != "both" and a.condition == "all":
        rungs = [r for r in rungs if r[1] == (a.sign == "on") or r[0] == "dense"]
    rows = []
    for cond, sign_on in rungs:
        r = run_rung(g_true, cond, sign_on, T, task, a.seed, lrs, a.epochs, a.patience, a.batch)
        rows.append(r)
        print(f"  {cond:<18} sign {'on ' if sign_on else 'off'}  units {r['units']:>6,}  edges {r['edges']:>7,}  params {r['params']:>8,}  lr {r['lr']:.0e}  "
              f"epochs {r['epochs_run']:>3}  train acc {r['train_acc']:.3f}  val acc {r['val_acc']:.3f}  test acc {r['test_acc']:.3f}  ({r['seconds']} s)", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.dataset}_tau{a.tau}_P{a.P}_n{a.n}_seed{a.seed}.json"
    path.write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1, default=str), encoding="utf-8")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
