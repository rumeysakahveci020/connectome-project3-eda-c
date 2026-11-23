"""Week 9, Project 3: the naive result. The masked, sign-constrained network against the dense controls, in PyTorch with Adam and clipping,
10 seeds, on the task, units, threshold and training protocol of the approved lock note (and its addendum).

    python tools/week9_naive.py --tune --approved                       # learning rate per (dataset, condition) on seed 0, validation loss only
    python tools/week9_naive.py --approved                              # all seeds, the P x n grid, both datasets (long; resumable)
    python tools/week9_naive.py --datasets fafb --P 40 --n 10 --seeds 3 --approved      # a small slice

Conditions: masked_sign (the true mask with the sign constraint), masked_free (the same true mask, NO sign constraint: separates the sign effect from the structure effect), dense_params (H = round(sqrt(E)) units, as many weights as the mask has edges),
dense_units (the same N units, N*N weights; heavy, run it on a few cells). The script appends one JSON line per run to results/week9/runs.jsonl and skips
runs already there (--resume is the default). Do not run it on the real tables before the milestone prerequisites hold: --approved is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src import eda_utils as U  # noqa: E402
from src.model.graphs import build_units, make_condition, time_steps  # noqa: E402
from src.model.task import GRID_N, GRID_P, make_task  # noqa: E402

LR_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
CONDITIONS = {"masked_sign": ("true_mask", True, "sparse"), "dense_params": ("dense", False, "sparse"), "dense_units": ("dense_units", False, "dense"),
              "masked_free": ("true_mask", False, "sparse")}
TUNE_CELL = (40, 10)


def require_approval(approved: bool, synthetic: bool) -> None:
    if not synthetic and not approved:
        raise SystemExit("STOP: do not run the naive model until the milestone prerequisites hold (approved lock note, corrected tables, Step 0 matching the group).\n"
                         "Pass --approved only then (a synthetic cache, EDA_CACHE_DIR set, runs without it).")


def save_split(task: dict, seed: int, P: int, n: int, out: Path) -> str:
    """The split as bits (packbits), so week 11 can reuse exactly the same data; returns a hash of the arrays."""
    out.mkdir(parents=True, exist_ok=True)
    arrays = {k: (np.packbits(task[k].astype(np.uint8), axis=1) if k.startswith("X_") else task[k].astype(np.int8)) for k in
              ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "patterns", "labels") if k in task}
    arrays["patterns"] = np.packbits(task["patterns"].astype(np.uint8), axis=1)
    np.savez_compressed(out / f"seed{seed}_P{P}_n{n}.npz", **arrays)
    h = hashlib.sha256()
    for k in sorted(arrays):
        h.update(arrays[k].tobytes())
    return h.hexdigest()[:16]


def one_run(g_cond, T, sign_on, impl, task, seed, lr, epochs, patience, batch) -> dict:
    from src.model.torch_net import TorchMaskedRNN, train_torch
    model = TorchMaskedRNN(g_cond, T, seed, sign_constrained=sign_on, impl=impl)
    res = train_torch(model, task, lr, seed, epochs, patience, batch, readout=g_cond.mbon)
    ytr = task["y_train"]
    majority = 1.0 if (ytr > 0).mean() >= 0.5 else -1.0                           # the trivial reference: always the commonest training label
    res["majority_test_acc"] = float((task["y_test"] == majority).mean())
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["fafb", "malecns"], choices=["fafb", "malecns"])
    ap.add_argument("--conditions", nargs="+", default=["masked_sign", "dense_params"], choices=list(CONDITIONS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--P", type=int, nargs="+", default=list(GRID_P))
    ap.add_argument("--n", type=int, nargs="+", default=list(GRID_N))
    ap.add_argument("--tau", type=int, default=5)
    ap.add_argument("--glutamate", choices=["inhibitory", "excitatory"], default="inhibitory")
    ap.add_argument("--selection", choices=["union", "region", "cell_type"], default="union")
    ap.add_argument("--tune", action="store_true", help="choose the learning rate per (dataset, condition) on seed 0, validation loss, one cell of the grid")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default=str(ROOT / "results" / "week9"))
    ap.add_argument("--approved", action="store_true")
    a = ap.parse_args(argv)
    require_approval(a.approved, synthetic=bool(os.environ.get("EDA_CACHE_DIR")))
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    U._cache_dir()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    graphs = {d: build_units(*U.load(d), a.tau, a.selection, a.glutamate) for d in ("fafb", "malecns")}
    T = max(time_steps(g) for g in graphs.values())
    print({d: (g.n, g.n_edges, len(g.kc), len(g.mbon)) for d, g in graphs.items()}, "T =", T, "(units, edges, input units, output units)", flush=True)
    lr_path = out / "lr_choice.json"
    lr_choice = json.loads(lr_path.read_text()) if lr_path.exists() else {}
    runs_path = out / "runs.jsonl"
    done = {(r["dataset"], r["condition"], r["seed"], r["P"], r["n"]) for r in map(json.loads, runs_path.read_text().splitlines())} if runs_path.exists() else set()

    if a.tune:
        P0, n0 = TUNE_CELL
        task = make_task(0, P0, n0)
        for d in a.datasets:
            for c in a.conditions:
                cond, sign_on, impl = CONDITIONS[c]
                g = make_condition(graphs[d], cond, 0)
                scores = {}
                for lr in LR_GRID:
                    scores[lr] = one_run(g, T, sign_on, impl, task, 0, lr, a.epochs, a.patience, a.batch)["best_val_loss"]
                    print(f"  tune {d} {c} lr {lr:.0e}: best validation loss {scores[lr]:.4f}", flush=True)
                lr_choice[f"{d}/{c}"] = min(scores, key=scores.get)
        lr_path.write_text(json.dumps(lr_choice, indent=1))
        print("learning rates:", lr_choice)
        return 0

    for d in a.datasets:
        for c in a.conditions:
            if f"{d}/{c}" not in lr_choice:
                raise SystemExit(f"no learning rate for {d}/{c}: run with --tune first (validation loss only; the same grid for every condition)")
            cond, sign_on, impl = CONDITIONS[c]
            for seed in range(a.seeds):
                g = make_condition(graphs[d], cond, seed)
                for P in a.P:
                    for n in a.n:
                        if (d, c, seed, P, n) in done:
                            continue
                        task = make_task(seed, P, n)
                        split_hash = save_split(task, seed, P, n, out / "splits") if (P, n) == TUNE_CELL or seed == 0 else None
                        res = one_run(g, T, sign_on, impl, task, seed, lr_choice[f"{d}/{c}"], a.epochs, a.patience, a.batch)
                        row = {"dataset": d, "condition": c, "seed": seed, "P": P, "n": n, "lr": lr_choice[f"{d}/{c}"], "tau": a.tau, "T": T, "split_hash": split_hash, **res}
                        with runs_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(row) + "\n")
                        print(f"{d} {c:<12} seed {seed} P {P:>3} n {n:>3}  test acc {res['test_acc']:.3f}  epochs {res['epochs_run']:>3}  violations {res['violation_mean']:.3f}  ({res['seconds']} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
