"""A recurrent ReLU network on a masked graph, trained with Adam and a projection after every step (lock note: Optimiser and budget).

Forward (batch B):   h_0 = 0;   h_t = relu(W h_{t-1} + u + b), t = 1..T, with u = W_in x on the input units (0 elsewhere);   o = w_out . h_T[output units] + c.
Loss: logistic loss (binary cross-entropy) for labels y in {-1, +1}.   Gradients are computed by hand (backprop through time) with numpy and
scipy.sparse, so no deep-learning library is needed. W has one trainable value per edge of the mask: the mask is the sparsity pattern.
Projection after each step: weights leaving a unit with a sign are clipped to that sign; units with a free sign (and every unconstrained
condition) are not clipped.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit

from src.model.graphs import Graph, initial_unit_signs
from src.model.task import D, K_IN, input_mapping


def initial_state(g: Graph, seed: int, sign_constrained: bool) -> dict:
    """The one initialisation rule of the lock note, shared by the numpy and the PyTorch implementation so that both start from the same numbers.
    Edges are stored in CSR order (by target unit, then source unit). |W_ij| ~ U(0, 2 / sqrt(k_i)) with k_i the in-degree of the target; the sign is the
    INITIAL sign of the presynaptic unit (g.init_sign: the unit's own sign, or a fair coin per unit for free-sign units), identical in every condition;
    the constraint only acts through the projection. Readout weights N(0, 1/n_out), biases 0."""
    if sign_constrained and g.sign is None:
        raise ValueError("the dense control has no sign vector: use sign_constrained=False")
    order = np.lexsort((g.pre, g.post))
    pre, post = g.pre[order].astype(np.int64), g.post[order].astype(np.int64)
    indptr = np.concatenate([[0], np.cumsum(np.bincount(post, minlength=g.n))])
    rng = np.random.default_rng([seed, 4])
    indeg = np.maximum(np.bincount(post, minlength=g.n), 1)
    mag = rng.random(len(pre)) * 2.0 / np.sqrt(indeg[post])
    unit0 = g.init_sign if g.init_sign is not None else initial_unit_signs(g.sign if g.sign is not None else np.zeros(g.n, dtype=np.int8), seed)
    s = g.sign[pre] if (sign_constrained and g.sign is not None) else np.zeros(len(pre), dtype=np.int8)
    return {"pre": pre, "post": post, "indptr": indptr, "w": mag * unit0[pre], "sign_edge": s,
            "w_out": rng.standard_normal(len(g.mbon)) / np.sqrt(len(g.mbon))}


class MaskedRNN:
    def __init__(self, g: Graph, T: int, seed: int, sign_constrained: bool, d: int = D, k_in: int = K_IN, dtype=np.float32):
        init = initial_state(g, seed, sign_constrained)
        self.g, self.T, self.dt, self.sc = g, int(T), dtype, bool(sign_constrained)
        self.pre, self.post, self.indptr = init["pre"], init["post"], init["indptr"]
        self.W_in = input_mapping(len(g.kc), seed, d, k_in).astype(dtype)
        self.w = init["w"].astype(dtype)
        self.b = np.zeros(g.n, dtype=dtype)
        self.w_out = init["w_out"].astype(dtype)
        self.c = dtype(0.0)
        self._sign_edge = init["sign_edge"]

    # ------------------------------------------------------------------ forward and backward
    def _W(self) -> csr_matrix:
        """Rows are target units, columns are source units; one stored value per edge of the mask."""
        return csr_matrix((self.w, self.pre, self.indptr), shape=(self.g.n, self.g.n))

    def forward(self, X: np.ndarray):
        B = len(X)
        u = np.zeros((self.g.n, B), dtype=self.dt)
        u[self.g.kc] = self.W_in @ X.T.astype(self.dt)             # input drive, applied at every step
        h = [np.zeros((self.g.n, B), dtype=self.dt)]
        W = self._W()
        z = []
        for _ in range(self.T):
            zt = W @ h[-1] + u + self.b[:, None]
            z.append(zt)
            h.append(np.maximum(zt, 0))
        o = self.w_out @ h[-1][self.g.mbon] + self.c
        return o, (h, z)

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray):
        o, (h, z) = self.forward(X)
        B = len(X)
        m = -y * o
        loss = float(np.mean(np.logaddexp(0.0, m)))
        do = -y * expit(m) / B                                      # dL/do = -y sigmoid(-y o) / B
        gw_out = h[-1][self.g.mbon] @ do
        gc = do.sum()
        dh = np.zeros_like(h[-1])
        dh[self.g.mbon] = np.outer(self.w_out, do)
        gw = np.zeros(len(self.pre), dtype=self.dt)
        gb = np.zeros(self.g.n, dtype=self.dt)
        W = self._W()
        rows = np.repeat(np.arange(self.g.n), np.diff(self.indptr))
        for t in range(self.T, 0, -1):
            dz = dh * (z[t - 1] > 0)
            gb += dz.sum(1)
            gw += np.einsum("eb,eb->e", dz[rows], h[t - 1][self.pre], optimize=True)
            dh = W.T @ dz
        return loss, {"w": gw, "b": gb, "w_out": gw_out.astype(self.dt), "c": np.asarray(gc, dtype=self.dt)}

    def loss_only(self, X: np.ndarray, y: np.ndarray, batch: int = 256):
        tot, correct = 0.0, 0
        for i in range(0, len(X), batch):
            o, _ = self.forward(X[i:i + batch])
            yy = y[i:i + batch]
            tot += float(np.logaddexp(0.0, -yy * o).sum())
            correct += int(((o > 0) == (yy > 0)).sum())
        return tot / len(X), correct / len(X)

    # ------------------------------------------------------------------ projection
    def project(self) -> None:
        """Clip every weight to the sign of its presynaptic unit (lock note); the mask is the sparsity pattern, so it holds by construction."""
        if self.sc:
            s = self._sign_edge
            np.maximum(self.w, 0, out=self.w, where=(s > 0))
            np.minimum(self.w, 0, out=self.w, where=(s < 0))

    def params(self) -> dict:
        return {"w": self.w, "b": self.b, "w_out": self.w_out, "c": self.c}

    def n_params(self) -> int:
        return len(self.w) + len(self.b) + len(self.w_out) + 1


class Adam:
    def __init__(self, model: MaskedRNN, lr: float, betas=(0.9, 0.999), eps: float = 1e-8):
        self.m, self.lr, self.b1, self.b2, self.eps, self.t = model, lr, betas[0], betas[1], eps, 0
        self.s = {k: np.zeros_like(np.asarray(v)) for k, v in model.params().items()}
        self.v = {k: np.zeros_like(np.asarray(v)) for k, v in model.params().items()}

    def step(self, grads: dict) -> None:
        self.t += 1
        for k, p in self.m.params().items():
            g = grads[k]
            self.s[k] = self.b1 * self.s[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh, vh = self.s[k] / (1 - self.b1 ** self.t), self.v[k] / (1 - self.b2 ** self.t)
            upd = (self.lr * mh / (np.sqrt(vh) + self.eps)).astype(self.m.dt)
            if k == "c":
                self.m.c = self.m.dt(self.m.c - upd)
            else:
                p -= upd


def train(model: MaskedRNN, task: dict, lr: float, seed: int, max_epochs: int = 200, patience: int = 20, batch: int = 64, verbose: bool = False) -> dict:
    """Adam + projection; early stopping on the validation loss (the lock note: 200 epochs, patience 20). The test set is only read at the end."""
    opt = Adam(model, lr)
    rng = np.random.default_rng([seed, 5])
    Xtr, ytr, Xva, yva = task["X_train"], task["y_train"], task["X_val"], task["y_val"]
    best, best_state, bad, hist, t0 = np.inf, None, 0, [], time.time()
    for epoch in range(max_epochs):
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            _, grads = model.loss_and_grads(Xtr[idx], ytr[idx])
            opt.step(grads)
            model.project()
        vl, va = model.loss_only(Xva, yva)
        hist.append(vl)
        if vl < best - 1e-9:
            best, bad = vl, 0
            best_state = {k: np.array(v, copy=True) for k, v in model.params().items()}
        else:
            bad += 1
        if verbose and epoch % 10 == 0:
            print(f"epoch {epoch:3d}  val loss {vl:.4f}  val acc {va:.3f}")
        if bad >= patience:
            break
    for k, v in best_state.items():                       # back to the best epoch
        p = model.params()[k]
        if k == "c":
            model.c = model.dt(v)
        else:
            p[...] = v
    trl, tra = model.loss_only(Xtr, ytr)
    val_l, val_a = model.loss_only(Xva, yva)
    tel, tea = model.loss_only(task["X_test"], task["y_test"])
    return {"epochs_run": len(hist), "best_val_loss": float(best), "val_loss": val_l, "val_acc": val_a, "train_loss": trl, "train_acc": tra,
            "test_loss": tel, "test_acc": tea, "seconds": round(time.time() - t0, 1), "params": model.n_params()}
