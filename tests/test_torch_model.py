"""Tests of the PyTorch implementation (the naive method of the Week 9 milestone). Skipped where PyTorch is not installed."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.model.graphs import Graph, make_condition  # noqa: E402
from src.model.net import Adam, MaskedRNN  # noqa: E402
from src.model.task import make_task  # noqa: E402
from src.model.torch_net import TorchMaskedRNN, ablation_mask, train_torch  # noqa: E402
from tests.test_model import small_graph  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def batches(seed=0, n=3, b=5):
    rng = np.random.default_rng(seed)
    return [((rng.random((b, 50)) < 0.5).astype(np.float64), rng.choice([-1.0, 1.0], b)) for _ in range(n)]


def step_numpy(g, sc, data, lr, T=2):
    m = MaskedRNN(g, T=T, seed=3, sign_constrained=sc, dtype=np.float64)
    opt = Adam(m, lr)
    for X, y in data:
        _, gr = m.loss_and_grads(X, y)
        opt.step(gr)
        m.project()
    return m


def step_torch(g, sc, data, lr, impl="sparse", T=2):
    m = TorchMaskedRNN(g, T, 3, sc, impl=impl, dtype=torch.float64)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    for X, y in data:
        opt.zero_grad()
        lossf(m(torch.as_tensor(X)), torch.as_tensor((y + 1) / 2)).backward()
        opt.step()
        m.project()
    return m


@pytest.mark.parametrize("sc", [False, True])
def test_pytorch_and_the_hand_written_numpy_network_agree_step_by_step(sc):
    g, data = small_graph(), batches()
    a, b = step_numpy(g, sc, data, lr=0.05), step_torch(g, sc, data, lr=0.05)
    assert np.allclose(a.w, b.edge_weights().numpy(), atol=1e-8)
    assert np.allclose(a.b, b.b.detach().numpy(), atol=1e-8) and np.allclose(a.w_out, b.w_out.detach().numpy(), atol=1e-8)
    assert abs(float(a.c) - float(b.c)) < 1e-8


@pytest.mark.parametrize("sc", [False, True])
def test_the_sparse_and_the_textbook_dense_times_mask_implementations_give_the_same_numbers(sc):
    g, data = small_graph(), batches()
    s, d = step_torch(g, sc, data, 0.05, "sparse"), step_torch(g, sc, data, 0.05, "dense")
    assert np.allclose(s.edge_weights().numpy(), d.edge_weights().numpy(), atol=1e-8)
    outside = d.W.detach() * (1 - d.M)                                  # entries outside the mask: zero gradient, they stay zero under Adam
    assert float(outside.abs().max()) == 0.0


def test_violations_are_measured_before_the_clipping_and_clipping_removes_them():
    g = small_graph()
    m = TorchMaskedRNN(g, 2, 0, True, impl="sparse")
    assert m.violation_fraction() == 0.0                                 # the initial weights obey the sign vector
    with torch.no_grad():
        m.w.copy_(torch.randn_like(m.w))
    before = m.violation_fraction()
    assert before > 0.1
    m.project()
    assert m.violation_fraction() == 0.0
    free = TorchMaskedRNN(g, 2, 0, False)
    assert free.violation_fraction() == 0.0 and free.sc is False


def test_ablation_sets_are_the_same_for_every_condition_with_the_same_units_and_never_touch_the_readout():
    g = small_graph()
    a = ablation_mask(g.n, g.mbon, 0.3, seed=4)
    b = ablation_mask(g.n, g.mbon, 0.3, seed=4)
    assert (a == b).all() and (a[g.mbon] == 1).all() and int((a == 0).sum()) == int(round(0.3 * (g.n - len(g.mbon))))
    assert (ablation_mask(g.n, g.mbon, 0.0, seed=4) == 1).all()
    assert not (ablation_mask(g.n, g.mbon, 0.3, seed=5) == a).all()


def test_parameter_counts_nominal_against_effective_and_the_dense_controls():
    g = small_graph()
    t = TorchMaskedRNN(g, 2, 0, True)
    c = t.counts()
    assert c["mask_weights"] == g.n_edges and c["nominal_params"] == g.n * g.n + g.n + len(g.mbon) + 1 and c["effective_nonzero"] <= g.n_edges
    du = make_condition(g, "dense_units", 0)
    assert du.n == g.n and du.n_edges == g.n ** 2 and du.sign is None and (du.kc == g.kc).all()
    dp = make_condition(g, "dense", 0)
    assert abs(dp.n_edges - g.n_edges) <= dp.n
    m = TorchMaskedRNN(du, 2, 0, False, impl="dense")
    assert m.full and m.counts()["mask_weights"] == g.n ** 2


def test_torch_training_learns_an_easy_task_and_records_violations_and_ablation():
    n_kc, n_out = 120, 6
    pre = np.repeat(np.arange(n_kc), n_out).astype(np.int32)
    post = np.tile(np.arange(n_kc, n_kc + n_out), n_kc).astype(np.int32)
    g = Graph(n_kc + n_out, pre, post, np.arange(n_kc, dtype=np.int32), np.arange(n_kc, n_kc + n_out, dtype=np.int32), np.ones(n_kc + n_out, dtype=np.int8), "true_mask")
    task = make_task(0, 6, 30)
    res = train_torch(TorchMaskedRNN(g, 2, 0, True), task, lr=3e-3, seed=0, max_epochs=60, patience=60, readout=g.mbon)
    assert res["train_acc"] > 0.9 and res["test_acc"] > 0.8
    assert len(res["violation_per_epoch"]) == res["epochs_run"] and 0.0 <= res["violation_mean"] <= 1.0
    assert set(res["ablation_test_acc"]) == {"0.0", "0.1", "0.2", "0.3", "0.5"} and abs(res["ablation_test_acc"]["0.0"] - res["test_acc"]) < 1e-9


def test_the_week9_runner_refuses_the_real_tables_without_approval_and_saves_reusable_splits(tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    import week9_naive
    with pytest.raises(SystemExit, match="prerequisites"):
        week9_naive.require_approval(False, synthetic=False)
    week9_naive.require_approval(False, synthetic=True)
    t = make_task(2, 10, 3)
    h1 = week9_naive.save_split(t, 2, 10, 3, tmp_path)
    h2 = week9_naive.save_split(make_task(2, 10, 3), 2, 10, 3, tmp_path)
    assert h1 == h2 and (tmp_path / "seed2_P10_n3.npz").exists()
    z = np.load(tmp_path / "seed2_P10_n3.npz")
    assert (np.unpackbits(z["X_train"], axis=1)[:, :50] == t["X_train"]).all()
