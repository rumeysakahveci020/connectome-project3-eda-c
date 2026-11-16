"""The synthetic pattern task (Litwin-Kumar et al. 2017) and the fixed input mapping (Caron et al. 2013), as written in the lock note.

Data depend only on (seed, P, n): every condition of one seed sees the same training, validation and test data.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

D = 50                      # input channels
F = 0.5                     # density of a pattern
NOISE = 0.1                 # every bit flips with this probability
K_IN = 7                    # input channels per Kenyon cell
N_TEST = 2000               # fixed test set
N_VAL = 1000                # fixed validation set: new noisy copies of the same P patterns (no pattern is taken out of training)
GRID_P = (10, 20, 40, 80, 160, 320)
GRID_N = (1, 3, 10, 30, 100)


def make_task(seed: int, P: int, n_per_pattern: int, d: int = D, f: float = F, noise: float = NOISE,
              n_test: int = N_TEST, n_val: int = N_VAL) -> dict:
    """P random binary patterns with random +1 / -1 labels; samples are noisy copies. Returns float32 arrays."""
    rng = np.random.default_rng([seed, 1])
    patterns = (rng.random((P, d)) < f).astype(np.int8)
    labels = rng.choice(np.array([-1, 1], dtype=np.int8), size=P)

    def noisy(idx):
        flips = rng.random((len(idx), d)) < noise
        return (patterns[idx] ^ flips.astype(np.int8)).astype(np.float32), labels[idx].astype(np.float32)

    idx = np.repeat(np.arange(P), n_per_pattern)
    rng.shuffle(idx)
    X, y = noisy(idx)                                       # training: all n copies of every pattern
    Xv, yv = noisy(rng.integers(0, P, size=n_val))          # validation and test: new noisy copies of the same P patterns
    Xt, yt = noisy(rng.integers(0, P, size=n_test))
    return {"X_train": X, "y_train": y, "X_val": Xv, "y_val": yv, "X_test": Xt, "y_test": yt,
            "patterns": patterns, "labels": labels}


def input_mapping(n_in: int, seed: int, d: int = D, k: int = K_IN) -> csr_matrix:
    """Fixed input weights: input unit r reads k distinct, randomly chosen channels, each with weight 1/sqrt(k).
    Row r is drawn from its own stream (seed, r), so it does not depend on how many input units there are."""
    rows, cols = [], []
    for r in range(n_in):
        ch = np.random.default_rng([seed, 2, r]).choice(d, size=k, replace=False)
        rows.extend([r] * k)
        cols.extend(ch.tolist())
    return csr_matrix((np.full(len(rows), 1.0 / np.sqrt(k)), (rows, cols)), shape=(n_in, d))
