"""The naive method of the Week 9 milestone: the lock-note network in PyTorch, trained with a standard optimiser (torch.optim.Adam) and
clipping (wrong-signed weights set to zero), exactly the comparison the literature makes.

Two implementations of the same mask, and a test proving they agree:
  impl = "sparse"  one trainable value per edge of the mask (fast; used for the real runs);
  impl = "dense"   a full N x N matrix W multiplied elementwise by a 0/1 mask M in the forward pass (the textbook version): masked entries get a zero
                   gradient and stay at zero under Adam, so the two give the same numbers.
Forward (batch B, h has shape N x B):  h_0 = 0;  h_t = relu((W * M) h_{t-1} + u + b), t = 1..T, u = W_in x on the input units (0 elsewhere);  o = w_out . h_T[output] + c.
Loss: torch.nn.BCEWithLogitsLoss (the logistic loss of the +1/-1 label).  Initialisation: src.model.net.initial_state (one rule for every condition).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from src.model.graphs import Graph
from src.model.net import initial_state
from src.model.task import D, K_IN, input_mapping


class TorchMaskedRNN(nn.Module):
    def __init__(self, g: Graph, T: int, seed: int, sign_constrained: bool, impl: str = "sparse", d: int = D, k_in: int = K_IN, dtype=torch.float32):
        super().__init__()
        init = initial_state(g, seed, sign_constrained)
        self.n_units, self.T, self.sc, self.impl = g.n, int(T), bool(sign_constrained), impl
        self.register_buffer("pre", torch.as_tensor(init["pre"]))
        self.register_buffer("post", torch.as_tensor(init["post"]))
        self.register_buffer("sign_edge", torch.as_tensor(init["sign_edge"].astype(np.int8)))
        self.register_buffer("kc", torch.as_tensor(g.kc.astype(np.int64)))
        self.register_buffer("mbon", torch.as_tensor(g.mbon.astype(np.int64)))
        self.register_buffer("W_in", torch.as_tensor(input_mapping(len(g.kc), seed, d, k_in).toarray(), dtype=dtype))
        w0 = torch.as_tensor(init["w"], dtype=dtype)
        if impl == "sparse":
            self.w = nn.Parameter(w0)
        elif impl == "dense":
            W = torch.zeros(g.n, g.n, dtype=dtype)
            W[self.post, self.pre] = w0
            self.W = nn.Parameter(W)
            self.full = g.n_edges == g.n * g.n                       # a dense control needs no mask
            if not self.full:
                M = torch.zeros(g.n, g.n, dtype=dtype)
                M[self.post, self.pre] = 1.0
                self.register_buffer("M", M)
            su = torch.as_tensor(g.sign.astype(np.int8)) if (sign_constrained and g.sign is not None) else torch.zeros(g.n, dtype=torch.int8)
            self.register_buffer("sign_unit", su)
        else:
            raise ValueError(f"unknown impl {impl!r}")
        self.b = nn.Parameter(torch.zeros(g.n, dtype=dtype))
        self.w_out = nn.Parameter(torch.as_tensor(init["w_out"], dtype=dtype))
        self.c = nn.Parameter(torch.zeros((), dtype=dtype))

    # ------------------------------------------------------------------ forward
    def _recur(self, h: torch.Tensor) -> torch.Tensor:
        if self.impl == "sparse":
            return torch.zeros_like(h).index_add_(0, self.post, self.w[:, None] * h[self.pre])
        W = self.W if self.full else self.W * self.M
        return W @ h

    def forward(self, X: torch.Tensor, unit_mask: torch.Tensor | None = None) -> torch.Tensor:
        u = torch.zeros(self.n_units, X.shape[0], dtype=X.dtype)
        u[self.kc] = self.W_in @ X.T                                 # input drive, applied at every step
        h = torch.zeros(self.n_units, X.shape[0], dtype=X.dtype)
        for _ in range(self.T):
            h = torch.relu(self._recur(h) + u + self.b[:, None])
            if unit_mask is not None:
                h = h * unit_mask[:, None]
        return self.w_out @ h[self.mbon] + self.c

    # ------------------------------------------------------------------ constraints
    @torch.no_grad()
    def violation_fraction(self) -> float:
        """Share of the signed weights (inside the mask) that carry the wrong sign. Measured after the optimiser step and BEFORE the clipping."""
        if not self.sc:
            return 0.0
        if self.impl == "sparse":
            signed = self.sign_edge != 0
            return float(((self.w * self.sign_edge) < 0)[signed].float().mean()) if signed.any() else 0.0
        inside = (self.M != 0) if not self.full else torch.ones_like(self.W, dtype=torch.bool)
        signed = inside & (self.sign_unit[None, :] != 0)
        bad = ((self.W * self.sign_unit[None, :]) < 0) & signed
        return float(bad.sum() / signed.sum().clamp(min=1))

    @torch.no_grad()
    def project(self) -> None:
        """Clipping: every weight leaving a signed unit is set to zero if it has the wrong sign (the projection onto the feasible set)."""
        if not self.sc:
            return
        if self.impl == "sparse":
            s = self.sign_edge
            self.w.copy_(torch.where(s > 0, self.w.clamp(min=0), torch.where(s < 0, self.w.clamp(max=0), self.w)))
        else:
            s = self.sign_unit[None, :]
            self.W.copy_(torch.where(s > 0, self.W.clamp(min=0), torch.where(s < 0, self.W.clamp(max=0), self.W)))

    # ------------------------------------------------------------------ parameter counts
    def edge_weights(self) -> torch.Tensor:
        """The weights inside the mask, as a vector."""
        if self.impl == "sparse":
            return self.w.detach()
        return self.W.detach()[self.post, self.pre]

    def counts(self) -> dict:
        w = self.edge_weights()
        extra = self.n_units + len(self.w_out) + 1
        nominal = self.n_units * self.n_units + extra                  # a masked network stores an N x N matrix: that is its nominal size
        return {"nominal_params": int(nominal), "mask_weights": int(len(w)), "effective_nonzero": int((w != 0).sum()), "units": self.n_units,
                "other_params": int(extra)}


def ablation_mask(n_units: int, readout: np.ndarray, fraction: float, seed: int) -> torch.Tensor:
    """Units that may be ablated: every unit except the readout units. The set depends only on (n_units, readout, fraction, seed), so every
    condition with the same units ablates the same ones within a seed."""
    idx = np.setdiff1d(np.arange(n_units), readout)
    k = int(round(fraction * len(idx)))
    out = np.ones(n_units, dtype=np.float32)
    if k:
        out[np.random.default_rng([seed, 7, int(round(fraction * 1000))]).choice(idx, size=k, replace=False)] = 0.0
    return torch.as_tensor(out)


@torch.no_grad()
def evaluate(model: TorchMaskedRNN, X: torch.Tensor, y01: torch.Tensor, unit_mask=None, batch: int = 256) -> tuple[float, float]:
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    tot, correct = 0.0, 0
    for i in range(0, len(X), batch):
        o = model(X[i:i + batch], unit_mask)
        tot += float(loss_fn(o, y01[i:i + batch]))
        correct += int(((o > 0) == (y01[i:i + batch] > 0.5)).sum())
    return tot / len(X), correct / len(X)


def train_torch(model: TorchMaskedRNN, task: dict, lr: float, seed: int, max_epochs: int = 200, patience: int = 20, batch: int = 64,
                readout: np.ndarray | None = None, ablation_fractions=(0.0, 0.1, 0.2, 0.3, 0.5)) -> dict:
    """Adam + clipping after every step; early stopping on the validation loss; the test set is read once, at the end."""
    to = lambda a: torch.as_tensor(a, dtype=torch.float32)      # noqa: E731
    Xtr, ytr, Xva, yva, Xte, yte = (to(task[k]) for k in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"))
    ytr01, yva01, yte01 = (ytr + 1) / 2, (yva + 1) / 2, (yte + 1) / 2
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng([seed, 5])
    best, best_state, bad, val_hist, viol_hist, t0 = np.inf, None, 0, [], [], time.time()
    for _ in range(max_epochs):
        perm = rng.permutation(len(Xtr))
        v_sum, v_n = 0.0, 0
        for i in range(0, len(Xtr), batch):
            idx = torch.as_tensor(perm[i:i + batch])
            opt.zero_grad()
            loss_fn(model(Xtr[idx]), ytr01[idx]).backward()
            opt.step()
            v_sum += model.violation_fraction()                  # before the clipping
            v_n += 1
            model.project()
        vl, _ = evaluate(model, Xva, yva01)
        val_hist.append(vl)
        viol_hist.append(v_sum / max(v_n, 1))
        if vl < best - 1e-9:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    trl, tra = evaluate(model, Xtr, ytr01)
    val_l, val_a = evaluate(model, Xva, yva01)
    tel, tea = evaluate(model, Xte, yte01)
    res = {"epochs_run": len(val_hist), "best_val_loss": float(best), "val_loss": val_l, "val_acc": val_a, "train_loss": trl, "train_acc": tra,
           "test_loss": tel, "test_acc": tea, "seconds": round(time.time() - t0, 1), "violation_per_epoch": [round(v, 5) for v in viol_hist],
           "violation_mean": float(np.mean(viol_hist)), **model.counts()}
    if readout is not None:
        res["ablation_test_acc"] = {str(f): evaluate(model, Xte, yte01, ablation_mask(model.n_units, readout, f, seed))[1] for f in ablation_fractions}
    return res
