# Project 3 – Week 9: naive result and failure argument

Decomposing the Biological Prior: FlyWire FAFB 783 and MaleCNS v1.0 mushroom-body subgraph.
Tag `week9`. Results: `results/week9/`, figures: `figures/week9/`, notebook: `notebooks/week9_naive.ipynb`.

## Locked choices
| Item | Value |
|---|---|
| Graph | Union selection (region rule OR mushroom body cell type), tau = 5 |
| FlyWire | 5,670 units, 47,329 edges (KC 5,177, MBON 96) |
| Male CNS | 4,515 units, 103,580 edges (KC 4,064, MBON 97) |
| Task | Random binary patterns, d = 50, f = 0.5, noise 0.1; validation and test on fresh noisy copies |
| P | 10, 20, 40, 80, 160, 320 |
| n (copies per pattern) | 1, 3, 10 finished (30, 100 planned) |
| Model | Recurrent ReLU, T = 3 steps (longest KC to MBON path + 1) |
| Conditions | `masked_sign` (real mask and signs, clipped to sign after every Adam step); `dense_params` (parameter-matched dense, H = 218 FlyWire / 322 male CNS) |
| Training | Adam, 200 epochs, early stopping patience 20, 3 seeds |
| Learning rate | Chosen once per (dataset, condition) on seed 0 at P = 40, n = 10: masked 0.01 (upper edge of grid), dense 0.003 |
| Metric | Test accuracy; majority baseline 0.51-0.57 |

## Deviations from the plan
- T = 2 in early code never read direct KC to MBON edges; fixed to T = 3.
- n = 30 and 100, seeds 3-4 and 10 seeds not run (laptop time limit; one cell at P = 320, n = 100 takes over 1 hour).
- Sign-free and equal-unit dense controls not run yet.
- Masked lr is at the grid edge, so it may be under-tuned.
- Run graphs have 5,625 (FlyWire) and 4,514 (male CNS) units versus 5,670 and 4,515 in the lock note. Edges are identical. Cause not yet verified (see `check_units.py`).
- One extra P = 40, n = 30 row in `runs.jsonl` is a timing run, excluded from tables.

## Main findings (3-seed means)
- n = 10: no clear winner; differences are 0.00-0.05, close to seed spread (0.001-0.04).
- n = 1: masked higher (0.77 vs 0.66 FlyWire, 0.76 vs 0.67 male CNS), but dense runs collapse to chance: 7/54 FlyWire, 5/54 male CNS, masked 0.
- Clipping: 13.0% (FlyWire) and 15.4% (male CNS) of weights clipped per step; 14-15% end exactly at zero.
- These are preliminary observations, not a final comparison. One contrast cannot separate sign, structure, degree and sparsity.

## Reproduce
```
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute notebooks/week9_naive.ipynb
```

## To do
Seeds 3-4, n = 30/100, sign-free and equal-unit controls, measure rho and learning curves, resolve unit-count mismatch, update report and slides.
