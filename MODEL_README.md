# Training code for Project 3 (the naive model and the ladder)

Follows the lock note line by line. **Do not run it on the real tables before the lock note is approved**: `tools/train_naive.py` refuses
unless you pass `--approved` (a synthetic cache, `EDA_CACHE_DIR`, runs without it, use that to test your setup).

    python tools/train_naive.py --dataset fafb --condition all --tau 5 --P 40 --n 10 --lr 1e-3 --seed 0 --approved
    python tools/train_naive.py --dataset malecns --condition true_mask --sign on --lr-grid --approved       # learning rate picked on the validation loss

| File | What it holds |
| --- | --- |
| `src/model/task.py` | the pattern task (d = 50, f = 0.5, noise 0.1, P and n grids, 2,000 test copies) and the fixed input mapping (7 channels per Kenyon cell) |
| `src/model/graphs.py` | the subgraph (region rule or MB cell type), signs from `nt_type_corrected`, isolated units removed, the four masks (dense, random sparse, degree-preserving via `eda_shared.randomiser`, true) and T from shortest Kenyon-cell-to-MBON paths |
| `src/model/net.py` | the recurrent ReLU network on the mask, backprop by hand (numpy and scipy.sparse), Adam, the projection after every step, early stopping (200 epochs, patience 20) |
| `tools/train_naive.py` | runs one rung or all seven (dense without sign; the other three with and without the sign constraint) |
| `tests/test_model.py` | gradients against finite differences, the sign projection, the masks, the task, a learning check, the approval guard |

## Details the lock note leaves open (now written down; they are also a row "Implementation details" in `tools/make_lock_note.py`)
1. `h_0 = 0`, `h_t = ReLU(W h_{t-1} + W_in x + b)`, t = 1..T, with fixed `W_in` weights `1/sqrt(7)`; the readout reads the MBON units at step T. T = longest shortest Kenyon-cell-to-MBON path + 1 (`time_steps`).
2. Batch size 64. Validation (1,000) and test (2,000) samples are new noisy copies of the same P patterns; nothing is removed from training. P grid 10..320.
3. Initial signs: one sign per unit (own sign, or a coin per unit for free-sign units), the same in every condition. Dense control: last n_out = number of MBONs units are read out, the other H - n_out units take the input.
3. Readout weights start from N(0, 1/n_out), biases at 0; recurrent weights |W_ij| ~ U(0, 2/sqrt(k_i)) with the sign of the presynaptic unit (a fair coin for free or unconstrained units).
4. Random sparse: E distinct pairs drawn uniformly over all units (the input and output units stay the same units).
5. Dense control: H = round(sqrt(E)) units, all H x H weights trainable, no sign vector; the first units are the input units and the last ones the output units, in the same proportions as in the true graph (the lock note fixes H, not the roles).
6. T is the larger of the two datasets' longest shortest Kenyon-cell-to-MBON path at the chosen tau (2 for both at tau = 5).

## How long it takes (one core, measured)
About 20 ms per step of 64 samples at tau = 5 (FlyWire), 49 ms (male CNS), 44 ms (FlyWire, tau = 3). One run with P = 40 and n = 10 and no early stopping
(7 steps per epoch, 200 epochs) takes 0.5 to 1 minute. **The whole grid is large**: all (P, n) pairs, 7 rungs, 5 learning rates, 10 seeds and 2 datasets are
about 700 passes over a grid that costs roughly an hour each, so decide with your supervisor which part of the grid to run (for example the learning rate picked once, on seed 0, and ten seeds at that rate).

## Week 9 (naive result): PyTorch version
`src/model/torch_net.py` is the naive method of the milestone: the same network in PyTorch with `torch.optim.Adam`, `BCEWithLogitsLoss` and clipping (wrong-signed
weights set to zero) after every step. It is **tested against** the hand-written numpy version (`src/model/net.py`: same numbers after Adam steps, to 1e-8) and the
sparse parameterisation is tested against the textbook `W * M` dense-times-mask implementation (same numbers; masked entries stay zero). The numbers in the milestone come
from the PyTorch version; the numpy version is only the cross-check and the starting point for week 11.

    python tools/week9_naive.py --tune --approved        # learning rate per (dataset, condition): seed 0, one cell, validation loss only (5 runs per condition)
    python tools/week9_naive.py --approved               # 10 seeds, the P x n grid, both datasets; appends to results/week9/runs.jsonl and resumes
    python tools/week9_naive.py --conditions dense_units --P 40 --n 10 --approved      # the dense control on the same N units (heavy: N*N weights)
    python tools/week9_figures.py                        # figures/week9/*.png and results/week9/tables/*.csv

Splits are saved as bits in `results/week9/splits/` (seed 0 for every cell, every seed for the tuning cell); they are also deterministic functions of (seed, P, n).
Install PyTorch in an interpreter it supports (CPU is enough), then pin it: `pip freeze > requirements-lock.txt`.
