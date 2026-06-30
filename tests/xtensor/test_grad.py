import pytest


pytest.importorskip("xarray")
pytestmark = pytest.mark.filterwarnings("error")

import numpy as np

import pytensor
import pytensor.tensor as pt
import pytensor.xtensor as px
from pytensor.graph import rewrite_graph
from pytensor.xtensor.type import as_xtensor
from tests.unittest_tools import verify_grad


def grad_through_lowering(cost, wrt):
    """Reference: lower the xtensor graph to tensor ops, then take the gradient."""
    cost = rewrite_graph(cost, include=("lower_xtensor",), clone=True)
    return pt.grad(cost, wrt)


def _x():
    xt = pt.tensor("x", shape=(3, 4))
    return xt, as_xtensor(xt, dims=("a", "b"))


def _y():
    yt = pt.tensor("y", shape=(4, 2))
    return yt, as_xtensor(yt, dims=("b", "c"))


def build_cases():
    xt, x = _x()
    yt, y = _y()
    return [
        ("reduce_sum", (px.math.exp(x).sum("a") * 1.5).sum(), [xt]),
        ("reduce_mean_std", (x.mean("a") + x.std("a")).sum(), [xt]),
        ("cumsum", px.math.exp(x).cumsum("a").sum(), [xt]),
        ("elemwise", (px.math.tanh(x) * px.math.sin(x)).sum(), [xt]),
        ("transpose", (x.transpose("b", "a") ** 2).sum(), [xt]),
        ("concat", px.concat([x, x + 1.0], dim="a").sum(), [xt]),
        ("stack", px.math.exp(x).stack({"z": ("a", "b")}).sum(), [xt]),
        ("rename", (x.rename({"a": "a2"}) ** 2).sum(), [xt]),
        # Swapping names exercises Rename as a positional relabel (not a permutation).
        ("rename_swap", (x.rename({"a": "b", "b": "a"}).sum("a") ** 2).sum(), [xt]),
        ("dot", (px.dot(x, y, dim="b") ** 2).sum(), [xt, yt]),
    ]


@pytest.mark.parametrize(
    "loss, wrt",
    [pytest.param(loss, wrt, id=name) for name, loss, wrt in build_cases()],
)
def test_grad_matches_lowering(loss, wrt):
    # pt.grad must work directly on the un-lowered xtensor graph and agree with the
    # supported "lower first, then grad" path.
    rng = np.random.default_rng(7)
    test_vals = [rng.normal(size=w.type.shape).astype(w.type.dtype) for w in wrt]
    g_direct = pt.grad(loss.values, wrt)
    g_ref = grad_through_lowering(loss.values, wrt)
    fn = pytensor.function(wrt, [*g_direct, *g_ref])
    out = fn(*test_vals)
    n = len(wrt)
    for direct, ref in zip(out[:n], out[n:]):
        np.testing.assert_allclose(direct, ref)


def test_grad_repeated_input():
    # A repeated input must accumulate per-slot cotangents (no factor-of-N error).
    xt = pt.vector("x", shape=(3,))
    x = as_xtensor(xt, dims=("a",))
    x_test = np.array([1.0, 2.0, 3.0])
    for power, loss in [(2, (x * x).sum()), (3, (x * x * x).sum())]:
        g = pytensor.function([xt], pt.grad(loss.values, xt))(x_test)
        np.testing.assert_allclose(g, power * x_test ** (power - 1))


def test_grad_second_order():
    W = pytensor.shared(np.ones((3, 2)), name="W")
    xt = pt.vector("x", shape=(3,))
    x = as_xtensor(xt, dims=("a",))
    y = px.dot(x, as_xtensor(W, dims=("a", "b")), dim="a")
    loss = (y * y).sum()
    g2 = pt.grad(pt.grad(loss.values, W).sum(), W)
    g2_ref = pt.grad(grad_through_lowering(loss.values, W).sum(), W)
    direct, ref = pytensor.function([xt], [g2, g2_ref])(np.arange(3.0))
    np.testing.assert_allclose(direct, ref)


def test_grad_through_indexing():
    # Indexing inputs (slices/integer indices) are non-differentiable, but the array
    # input's gradient is still correct: a scatter of the cotangent into the indexed
    # positions. The engine emits a benign connection_pattern advisory for the index.
    xt = pt.tensor("x", shape=(3, 4))
    x = as_xtensor(xt, dims=("a", "b"))
    loss = (x.isel(a=1) ** 2).sum()
    with pytest.warns(UserWarning, match="connection_pattern"):
        grad = pt.grad(loss.values, xt)
    x_test = np.arange(12.0).reshape(3, 4)
    expected = np.zeros((3, 4))
    expected[1] = 2 * x_test[1]
    np.testing.assert_allclose(pytensor.function([xt], grad)(x_test), expected)


def test_verify_grad():
    rng = np.random.default_rng(seed=420)

    def dot_loss(x, w):
        xx = as_xtensor(x, dims=("a",))
        ww = as_xtensor(w, dims=("a", "b"))
        return (px.dot(xx, ww, dim="a") ** 2).sum().values

    verify_grad(dot_loss, [rng.normal(size=(3,)), rng.normal(size=(3, 2))], rng=rng)
