"""Tests of the training code: the hand-written gradients, the sign projection, the conditions of the ladder, the task, and a learning check."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model.graphs import Graph, build_units, make_condition, shortest_T, time_steps, unit_signs
from src.model.net import MaskedRNN, train
from src.model.task import input_mapping, make_task

ROOT = Path(__file__).resolve().parent.parent


def small_graph(seed=0):
    """6 input units, 4 hidden units, 3 output units, with signs; every unit takes part."""
    rng = np.random.default_rng(seed)
    pre, post = [], []
    for j in range(6):
        for i in range(6, 10):
            if rng.random() < 0.6:
                pre.append(j); post.append(i)
    for j in range(6, 10):
        for i in range(10, 13):
            if rng.random() < 0.7:
                pre.append(j); post.append(i)
    for _ in range(6):
        a, b = rng.choice(10, 2, replace=False)
        pre.append(a); post.append(b)
    keys = np.unique(np.array(pre) * 13 + np.array(post))
    sign = np.array([1, 1, 1, 1, 1, 1, 1, -1, 0, 1, 1, 1, 1], dtype=np.int8)
    return Graph(13, (keys // 13).astype(np.int32), (keys % 13).astype(np.int32), np.arange(6, dtype=np.int32), np.arange(10, 13, dtype=np.int32), sign, "true_mask")


def test_gradients_match_finite_differences():
    g = small_graph()
    m = MaskedRNN(g, T=3, seed=3, sign_constrained=False, dtype=np.float64)   # input -> hidden -> output is a 2-step path: T = 3
    rng = np.random.default_rng(1)
    m.b = rng.standard_normal(g.n) * 0.1
    X = (rng.random((5, 50)) < 0.5).astype(np.float64)
    y = np.array([1, -1, 1, 1, -1], dtype=np.float64)
    _, grads = m.loss_and_grads(X, y)
    assert (np.abs(grads["w"]) > 1e-9).sum() >= 5                    # the check is not trivial: many weights carry a gradient
    worst = 0.0
    for name in ("w", "b", "w_out"):
        arr = m.params()[name]
        for k in range(min(len(arr), 25)):
            old = arr[k]
            arr[k] = old + 1e-6; lp = m.loss_and_grads(X, y)[0]
            arr[k] = old - 1e-6; lm = m.loss_and_grads(X, y)[0]
            arr[k] = old
            num = (lp - lm) / 2e-6
            worst = max(worst, abs(grads[name][k] - num) / (abs(num) + 1e-7))
    assert worst < 1e-5


def test_projection_clips_to_the_presynaptic_sign_and_leaves_free_and_unconstrained_units_alone():
    g = small_graph()
    m = MaskedRNN(g, T=2, seed=0, sign_constrained=True)
    pre_sign = g.sign[m.pre]
    assert (np.sign(m.w[pre_sign > 0]) >= 0).all() and (np.sign(m.w[pre_sign < 0]) <= 0).all()        # the initial weights obey the sign vector
    assert np.abs(m.w).max() <= 2.0 / np.sqrt(1) + 1e-6                                                # |W| <= 2 / sqrt(in-degree) <= 2
    m.w[:] = np.random.default_rng(2).standard_normal(len(m.w)).astype(np.float32)                      # a violent step
    m.project()
    assert (m.w[pre_sign > 0] >= 0).all() and (m.w[pre_sign < 0] <= 0).all()
    assert ((m.w[pre_sign == 0] > 0).any() and (m.w[pre_sign == 0] < 0).any())                         # free units keep both signs
    free = MaskedRNN(g, T=2, seed=0, sign_constrained=False)
    free.w[:] = -1.0
    free.project()
    assert (free.w == -1.0).all()                                                                       # no clipping without the constraint


def test_dense_control_has_no_sign_vector_and_refuses_the_constraint():
    g = small_graph()
    d = make_condition(g, "dense", 0)
    assert d.sign is None
    with pytest.raises(ValueError, match="no sign vector"):
        MaskedRNN(d, T=2, seed=0, sign_constrained=True)


def test_conditions_of_the_ladder():
    g = small_graph()
    rs = make_condition(g, "random_sparse", 5)
    assert rs.n_edges == g.n_edges and (rs.pre != rs.post).all() and len(set(zip(rs.pre.tolist(), rs.post.tolist()))) == g.n_edges
    assert set(zip(rs.pre.tolist(), rs.post.tolist())) != set(zip(g.pre.tolist(), g.post.tolist()))
    assert (rs.kc == g.kc).all() and (rs.mbon == g.mbon).all() and (rs.sign == g.sign).all()
    dp = make_condition(g, "degree_preserving", 5)
    assert dp.n_edges == g.n_edges
    assert (np.bincount(dp.pre, minlength=g.n) == np.bincount(g.pre, minlength=g.n)).all()               # every unit keeps its out-degree
    assert (np.bincount(dp.post, minlength=g.n) == np.bincount(g.post, minlength=g.n)).all()             # and its in-degree
    again = make_condition(g, "random_sparse", 5)
    assert (again.pre == rs.pre).all() and (again.post == rs.post).all()                                 # the same seed gives the same mask
    d = make_condition(g, "dense", 0)
    assert d.n == int(round(g.n_edges ** 0.5)) and d.n_edges == d.n ** 2
    assert abs(d.n_edges - g.n_edges) <= d.n                                                             # recurrent weights equal E up to rounding of H
    assert len(d.mbon) == len(g.mbon) and len(d.kc) == d.n - len(d.mbon) and d.kc.max() < d.mbon.min()   # same readout size; the rest takes the input


def test_task_is_the_same_for_every_condition_and_follows_the_lock_note():
    a, b = make_task(7, 40, 10), make_task(7, 40, 10)
    assert all((a[k] == b[k]).all() for k in a)
    assert len(a["X_test"]) == 2000 and set(np.unique(a["y_train"])) <= {-1.0, 1.0}
    assert a["X_train"].shape[1] == 50 and len(a["X_train"]) == 400 and len(a["X_val"]) == 1000      # all n copies train; validation is separate
    assert abs(a["patterns"].mean() - 0.5) < 0.05
    flips = np.mean([(a["X_test"][i] != a["patterns"][np.where((a["patterns"] == a["X_test"][i]).all(1))[0][0]]).mean()
                     if (a["patterns"] == a["X_test"][i]).all(1).any() else 0.1 for i in range(50)])
    assert 0.05 < flips < 0.15                                                                            # noise 0.1
    assert not (make_task(8, 40, 10)["X_train"] == a["X_train"]).all()


def test_each_input_unit_reads_seven_distinct_channels_independent_of_the_number_of_units():
    m5, m9 = input_mapping(5, seed=3), input_mapping(9, seed=3)
    assert (m5.getnnz(axis=1) == 7).all() and np.allclose(m5.data, 1 / np.sqrt(7))
    assert (m9[:5].toarray() == m5.toarray()).all()


def test_lock_note_subgraph_roles_signs_isolated_units_and_T():
    rows = [(i, "KCg-m", "MB_CA", "acetylcholine") for i in range(10)] + [(10 + i, f"MBON0{i}", "SMP", "gaba" if i else "glutamate") for i in range(3)] + \
           [(13, "PAM01", "MB_ML", "dopamine"), (14, "T4a", "ME", "acetylcholine"), (15, "KCg-m", "MB_CA", "acetylcholine")]
    n = pd.DataFrame(rows, columns=["neuron_id", "cell_type", "neuropil", "nt_type_corrected"])
    # KC i -> MBON 10 + (i % 3); MBON 10 -> PAM 13 -> MBON 11 (a two-step path); neuron 15 has no edge; neuron 14 is outside the subgraph
    e = [(i, 10 + (i % 3), 6) for i in range(10)] + [(10, 13, 5), (13, 11, 5), (14, 0, 9), (0, 10, 2)]
    ed = pd.DataFrame(e, columns=["pre_id", "post_id", "syn_count"])
    g = build_units(n, ed, tau=5, selection="union")
    assert g.n == 14 and g.meta["isolated_removed"] == 1                                              # neuron 15 is isolated and removed
    assert len(g.kc) == 10 and len(g.mbon) == 3
    ids = list(g.meta["unit_ids"])
    assert g.sign[ids.index(0)] == 1 and g.sign[ids.index(10)] == -1 and g.sign[ids.index(11)] == -1 and g.sign[ids.index(13)] == 0   # glutamate is inhibitory
    g2 = build_units(n, ed, tau=5, selection="union", glutamate="excitatory")
    assert g2.sign[list(g2.meta["unit_ids"]).index(10)] == 1
    assert shortest_T(g) == 1 and time_steps(g) == 2
    cut = ed.copy()
    cut.loc[(cut.pre_id == 1), "syn_count"] = 1                                                        # tau = 5 drops KC 1 -> MBON 11, which is still reached from 13
    assert unit_signs(pd.Series(["acetylcholine", "gaba", "glutamate", "dopamine", "unknown"])).tolist() == [1, -1, -1, 0, 0]


def test_training_learns_an_easy_task_and_the_test_set_is_only_read_at_the_end():
    rng = np.random.default_rng(0)
    n_kc, n_out = 120, 6
    pre = np.repeat(np.arange(n_kc), n_out).astype(np.int32)
    post = np.tile(np.arange(n_kc, n_kc + n_out), n_kc).astype(np.int32)
    g = Graph(n_kc + n_out, pre, post, np.arange(n_kc, dtype=np.int32), np.arange(n_kc, n_kc + n_out, dtype=np.int32), np.ones(n_kc + n_out, dtype=np.int8), "true_mask")
    task = make_task(0, 6, 30)
    model = MaskedRNN(g, T=2, seed=0, sign_constrained=True)          # input units active at step 1, MBONs reached at step 2
    res = train(model, task, lr=3e-3, seed=0, max_epochs=60, patience=60)
    assert res["train_acc"] > 0.9 and res["test_acc"] > 0.8 and res["test_acc"] <= 1.0
    assert (model.w >= 0).all()                                                                         # all presynaptic units are excitatory
    assert res["val_loss"] <= res["best_val_loss"] + 1e-6


def test_the_script_refuses_the_real_tables_without_approval():
    sys.path.insert(0, str(ROOT / "tools"))
    import train_naive
    with pytest.raises(SystemExit, match="not approved"):
        train_naive.require_approval(approved=False, synthetic=False)
    train_naive.require_approval(approved=True, synthetic=False)
    train_naive.require_approval(approved=False, synthetic=True)


def test_validation_and_test_are_new_copies_of_the_training_patterns_and_no_pattern_is_removed():
    t = make_task(3, 20, 1)                                          # n = 1: one copy per pattern, all of them in training
    assert len(t["X_train"]) == 20 and len(t["X_val"]) == 1000
    pats = t["patterns"]
    for X in (t["X_val"], t["X_test"]):
        d = np.abs(X[:200, None, :] - pats[None]).sum(2)             # Hamming distance to every pattern
        assert (d.min(1) <= 15).all()                                # each copy is a noisy version of one of the P patterns


def test_initial_signs_are_identical_in_every_condition_and_the_coin_is_per_unit():
    g = small_graph()
    inits = [make_condition(g, c, 5).init_sign for c in ("true_mask", "random_sparse", "degree_preserving")]
    assert all((i == inits[0]).all() for i in inits)
    own = g.sign != 0
    assert (inits[0][own] == g.sign[own]).all() and set(np.unique(inits[0])) <= {-1, 1}
    a = MaskedRNN(make_condition(g, "true_mask", 5), 2, 5, True)
    b = MaskedRNN(make_condition(g, "true_mask", 5), 2, 5, False)
    assert (np.sign(a.w) == np.sign(b.w)).all()                       # constrained and free start with the same signs
    ini = inits[0]
    assert (np.sign(a.w) == ini[a.pre]).all()                         # sign of a weight = initial sign of its presynaptic unit


def test_every_reachable_mbon_is_reached_by_the_readout_with_t_equal_to_the_longest_shortest_path_plus_one():
    # chain KC -> A -> MBON: shortest path 2, so T = 3; with T = 2 the MBON would see nothing but its bias
    g = Graph(3, np.array([0, 1], np.int32), np.array([1, 2], np.int32), np.array([0], np.int32), np.array([2], np.int32), np.array([1, 1, 1], np.int8), "true_mask")
    assert shortest_T(g) == 2 and time_steps(g) == 3
    X = np.ones((4, 50), np.float32)
    for T, active in ((2, False), (3, True)):
        m = MaskedRNN(g, T=T, seed=0, sign_constrained=False)
        m.w[:] = 1.0
        m.w_out[:] = 1.0
        o, (h, z) = m.forward(X)
        assert bool((h[-1][2] > 0).all()) == active
