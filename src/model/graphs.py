"""The mushroom-body subgraph as a network graph, and the conditions of the ladder (dense, random sparse, degree-preserving, true mask)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

CONDITIONS = ("dense", "random_sparse", "degree_preserving", "true_mask")
RUNGS = [("dense", False), ("random_sparse", False), ("random_sparse", True), ("degree_preserving", False), ("degree_preserving", True),
         ("true_mask", False), ("true_mask", True)]


@dataclass
class Graph:
    n: int                       # units
    pre: np.ndarray              # edge sources (unit index)
    post: np.ndarray             # edge targets
    kc: np.ndarray               # input units
    mbon: np.ndarray             # output units
    sign: np.ndarray | None      # +1 / -1 / 0 (free) per unit; None where a sign makes no sense (dense)
    name: str = ""
    init_sign: np.ndarray | None = None   # initial sign of every unit (same rule and same vector in every condition)
    meta: dict = field(default_factory=dict)

    @property
    def n_edges(self) -> int:
        return int(len(self.pre))

    def n_params(self, n_readout: bool = True) -> int:
        """Trainable parameters: recurrent weights + one bias per unit + linear readout (weights and bias)."""
        return self.n_edges + self.n + (len(self.mbon) + 1 if n_readout else 0)


def _is_type(ct: pd.Series, prefix: str) -> pd.Series:
    return ct.astype("string").str.upper().str.startswith(prefix, na=False)


def unit_signs(nt: pd.Series, glutamate: str = "inhibitory") -> np.ndarray:
    """Lock note: acetylcholine +, GABA -, glutamate - (default) or + (sensitivity). Everything else has a free sign (0)."""
    g = -1 if glutamate == "inhibitory" else 1
    return nt.astype(str).map({"acetylcholine": 1, "gaba": -1, "glutamate": g}).fillna(0).astype("int8").to_numpy()


def initial_unit_signs(sign: np.ndarray, seed: int) -> np.ndarray:
    """Initial sign per UNIT (not per weight): the unit's own sign where it has one, a fair coin per unit where the sign is free."""
    coin = np.random.default_rng([seed, 6]).choice(np.array([-1, 1], dtype=np.int8), size=len(sign))
    return np.where(sign != 0, sign, coin).astype(np.int8)


def build_units(neurons: pd.DataFrame, edges: pd.DataFrame, tau: int, selection: str = "union", glutamate: str = "inhibitory") -> Graph:
    """The subgraph of the lock note: region rule OR a mushroom-body cell type, edges with at least tau synapses, isolated units removed."""
    from src import eda_c as C, eda_utils as U
    ids = C.subgraph_ids(neurons, "region")
    if selection in ("union", "cell_type"):
        t = C.subgraph_ids(neurons, "cell_type")
        ids = ids | t if selection == "union" else t
    s = C.sub_edges(edges, ids, tau)
    order = np.array(sorted(ids), dtype=np.int64)
    has = np.zeros(len(order), dtype=bool)
    pos = pd.Series(np.arange(len(order)), index=order)
    pre, post = pos.reindex(s["pre_id"]).to_numpy(), pos.reindex(s["post_id"]).to_numpy()
    has[pre] = True
    has[post] = True
    keep = np.where(has)[0]                                 # units with no connection at all are removed
    remap = np.full(len(order), -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    kept_ids = order[keep]
    nn = neurons.set_index("neuron_id").loc[kept_ids]
    ct = nn["cell_type"].reset_index(drop=True)
    return Graph(n=len(keep), pre=remap[pre].astype(np.int32), post=remap[post].astype(np.int32),
                 kc=np.where(_is_type(ct, "KC").to_numpy())[0].astype(np.int32), mbon=np.where(_is_type(ct, "MBON").to_numpy())[0].astype(np.int32),
                 sign=unit_signs(nn[U.NT_COL].reset_index(drop=True), glutamate), name="true_mask",
                 meta={"tau": tau, "selection": selection, "units_before_removal": len(order), "isolated_removed": int(len(order) - len(keep)), "unit_ids": kept_ids})


def shortest_T(g: Graph) -> int:
    """Longest shortest path from a Kenyon cell to a reachable MBON (the number of recurrent time steps of the lock note)."""
    a = csr_matrix((np.ones(g.n_edges), (g.pre, g.post)), shape=(g.n, g.n))
    d = dijkstra(a, directed=True, unweighted=True, indices=g.kc, min_only=True)[g.mbon]
    d = d[np.isfinite(d)]
    if len(d) == 0:
        raise ValueError("no MBON is reachable from a Kenyon cell")
    return int(d.max())


def time_steps(g: Graph) -> int:
    """Number of recurrent steps T of the lock note: the longest shortest KC-to-MBON path plus one. The input is applied at every step and h_0 = 0,
    so the input units are active from step 1, a path of length L reaches its MBON at step L + 1, and every reachable MBON is reached by the readout."""
    return shortest_T(g) + 1


def _random_pairs(n: int, e: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """e distinct directed pairs without self-connections, uniformly at random."""
    keys = np.empty(0, dtype=np.int64)
    while len(keys) < e:
        a = rng.integers(0, n, size=2 * (e - len(keys)) + 16, dtype=np.int64)
        b = rng.integers(0, n, size=len(a), dtype=np.int64)
        k = a[a != b] * n + b[a != b]
        keys = np.unique(np.concatenate([keys, k]))
    keys = rng.permutation(keys)[:e]
    return (keys // n).astype(np.int32), (keys % n).astype(np.int32)


def make_condition(g: Graph, condition: str, seed: int) -> Graph:
    """The four masks of the ladder built from the true graph g. Units, input units and output units stay the same, except in the dense
    control (see below). The sign vector is the same unit-level vector in every sparse condition."""
    init = initial_unit_signs(g.sign, seed) if g.sign is not None else None
    if condition == "true_mask":
        return Graph(g.n, g.pre, g.post, g.kc, g.mbon, g.sign, "true_mask", init, dict(g.meta))
    if condition == "random_sparse":
        pre, post = _random_pairs(g.n, g.n_edges, np.random.default_rng([seed, 3]))
        return Graph(g.n, pre, post, g.kc, g.mbon, g.sign, "random_sparse", init, dict(g.meta))
    if condition == "degree_preserving":
        from eda_shared.randomiser import randomise_edges
        out = randomise_edges(pd.DataFrame({"pre_id": g.pre.astype(np.int64), "post_id": g.post.astype(np.int64)}), seed=int(seed))
        return Graph(g.n, out["pre_id"].to_numpy(np.int32), out["post_id"].to_numpy(np.int32), g.kc, g.mbon, g.sign, "degree_preserving", init, dict(g.meta))
    if condition == "dense":
        # H = round(sqrt(E)) units so that the H*H recurrent weights equal E (parameters match, units do not: H is smaller than N).
        # Readout: n_out = number of MBONs of the true graph (the same readout size as every other condition); the other H - n_out units
        # take the 50-channel input through the fixed 7-channel mapping. Initial unit signs are drawn from the empirical sign distribution of g.
        h = int(round(g.n_edges ** 0.5))
        n_out = min(len(g.mbon), h - 1)
        pre, post = np.meshgrid(np.arange(h, dtype=np.int32), np.arange(h, dtype=np.int32), indexing="ij")
        pool = np.random.default_rng([seed, 7]).choice(g.sign, size=h)
        return Graph(h, pre.ravel(), post.ravel(), np.arange(h - n_out, dtype=np.int32), np.arange(h - n_out, h, dtype=np.int32), None, "dense",
                     initial_unit_signs(pool, seed), {**g.meta, "dense_width": h, "dense_inputs": h - n_out, "dense_outputs": n_out})
    if condition == "dense_units":
        # The dense control on the SAME units as the mask: every pair, self-connections included, so N*N weights; same input and output units; no sign vector.
        pre, post = np.meshgrid(np.arange(g.n, dtype=np.int32), np.arange(g.n, dtype=np.int32), indexing="ij")
        return Graph(g.n, pre.ravel(), post.ravel(), g.kc, g.mbon, None, "dense_units", init, dict(g.meta))
    raise ValueError(f"unknown condition {condition!r}")
