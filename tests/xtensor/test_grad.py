import pytest


pytest.importorskip("xarray")
pytestmark = pytest.mark.filterwarnings("error")

import numpy as np

import pytensor
import pytensor.tensor as pt
import pytensor.xtensor as px
from pytensor.gradient import pushforward
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
        ("reduce_max", (x.max("a") * 1.5).sum(), [xt]),
        ("reduce_min", (x.min("a") * 1.5).sum(), [xt]),
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
    # The index itself is non-differentiable (an integer xtensor) so it gets an
    # undefined gradient, but the array input's gradient is still correct: a scatter
    # of the cotangent into the indexed positions.
    xt = pt.tensor("x", shape=(3, 4))
    x = as_xtensor(xt, dims=("a", "b"))
    loss = (x.isel(a=1) ** 2).sum()
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


def test_grad_consider_constant():
    # consider_constant on a variable internal to the xtensor region must still stop the
    # gradient there once the region is collapsed for differentiation.
    xt = pt.tensor("x", shape=(3, 4))
    x = as_xtensor(xt, dims=("a", "b"))
    inter = px.math.exp(x)
    loss = ((inter * inter).sum() + (x * x).sum()).values
    g = pytensor.function([xt], pt.grad(loss, xt, consider_constant=[inter]))
    # exp(x) is held constant, so only the (x*x) path contributes: d/dx = 2x.
    x_test = np.random.default_rng(0).normal(size=(3, 4))
    np.testing.assert_allclose(g(x_test), 2 * x_test)


def test_grad_second_order_xtensor_wrt():
    # Differentiating twice w.r.t. an xtensor variable: the collapsed unit is a plain
    # tensor op, so the repeated grad lowers and differentiates it without a residual.
    x = px.xtensor("x", dims=("a",), shape=(3,))
    loss = (x * x * x).sum()  # d2/dx2 = 6x
    g2 = pt.grad(pt.grad(loss.values, x).sum().values, x)
    x_test = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(pytensor.function([x], g2.values)(x_test), 6 * x_test)


def test_grad_chained_regions():
    # Leaving and re-entering the xtensor world collapses into separate units and still
    # matches the lower-then-grad reference (order of collapse must not form a cycle).
    xt = pt.tensor("x", shape=(3, 4))
    x = as_xtensor(xt, dims=("a", "b"))
    reentered = as_xtensor(x.sum("a").values + 1.0, dims=("b",))
    loss = (reentered**2).sum().values
    [direct] = pt.grad(loss, [xt])
    [ref] = grad_through_lowering(loss, [xt])
    d, r = pytensor.function([xt], [direct, ref])(
        np.random.default_rng(0).normal(size=(3, 4))
    )
    np.testing.assert_allclose(d, r)


def test_grad_wrt_exit_variable():
    # wrt a region's tensor exit (`expr.values`): the boundary acts by identity, so the
    # collapse must leave the exit (and everything it depends on) in place for grad to
    # reach it; no xtensor op needs differentiating on the cost -> wrt path.
    xt = pt.tensor("x", shape=(3, 4))
    x = as_xtensor(xt, dims=("a", "b"))
    w = px.math.exp(x).values
    g = pt.grad((w**2).sum(), w)
    x_test = np.random.default_rng(1).normal(size=(3, 4))
    np.testing.assert_allclose(pytensor.function([xt], g)(x_test), 2 * np.exp(x_test))


def test_grad_consider_constant_exit_variable():
    # consider_constant on a region's tensor exit must still stop the gradient there;
    # rewriting the exit out of the graph silently dropped the stop.
    xt = pt.tensor("x", shape=(3,))
    x = as_xtensor(xt, dims=("a",))
    w = px.math.exp(x).values
    cost = (w * w).sum() + (xt**2).sum()
    g = pt.grad(cost, xt, consider_constant=[w])
    x_test = np.array([0.5, 1.0, 1.5])
    np.testing.assert_allclose(pytensor.function([xt], g)(x_test), 2 * x_test)


def test_pushforward():
    # Forward mode double-pullbacks with a seed that references the exit; the exit is
    # then a boundary ancestor and must survive the collapse (collapsing it silently
    # disconnected the seed, returning 0).
    xt = pt.tensor("x", shape=(3,))
    x = as_xtensor(xt, dims=("a",))
    cost = (px.math.exp(x) * 2.0).sum().values
    v = pt.tensor("v", shape=(3,))
    jvp = pushforward(cost, [xt], [v])
    out = pytensor.function([xt, v], jvp)(np.zeros(3), np.ones(3))
    np.testing.assert_allclose(out, 6.0)


def test_grad_unrelated_to_random_region():
    # A random draw elsewhere in the cost graph must not break grad wrt an unrelated
    # variable: RNG-typed region inputs take rng conversions, not tensor conversions.
    theta = pt.scalar("theta")
    rng = px.random.shared_rng(seed=0)
    _, draw = px.random.normal(
        0.0, 1.0, extra_dims={"a": 3}, rng=rng, return_next_rng=True
    )
    cost = draw.sum().values * 0.0 + theta**2
    g = pt.grad(cost, theta)
    np.testing.assert_allclose(g.eval({theta: 3.0}), 6.0)


def test_grad_through_random_region():
    # Reparameterized draw: grad flows through the deterministic use of the draw and
    # matches the lower-then-grad reference, with a free xrng as function input.
    rng = px.random.rng("rng")
    at = pt.tensor("a", shape=(3,))
    a = as_xtensor(at, dims=("d",))
    _, eps = px.random.normal(
        0.0, 1.0, extra_dims={"d": 3}, rng=rng, return_next_rng=True
    )
    loss = ((a * eps) ** 2).sum().values
    g_direct = pt.grad(loss, at)
    g_ref = grad_through_lowering(loss, at)
    a_test = np.arange(1.0, 4.0)
    d, r = pytensor.function([at, rng], [g_direct, g_ref])(
        a_test, np.random.default_rng(3)
    )
    np.testing.assert_allclose(d, r)


def test_grad_through_slice_indexing():
    # Slice components are structural (MakeSlice) and must stay inside the collapsed
    # unit so the indexing lowering can pattern-match them, symbolic bounds included.
    at = pt.tensor("a", shape=(6, 5))
    xa = as_xtensor(at, dims=("i", "j"))
    k = pt.iscalar("k")
    loss = (xa.isel(i=slice(1, 4), j=slice(0, 2)) ** 2).sum().values
    sym_loss = (xa.isel(i=slice(k, 4)) ** 2).sum().values
    outs = [pt.grad(loss, at), *grad_through_lowering(loss, [at])]
    outs += [pt.grad(sym_loss, at), *grad_through_lowering(sym_loss, [at])]
    a_test = np.random.default_rng(2).normal(size=(6, 5))
    g, g_ref, gs, gs_ref = pytensor.function([at, k], outs)(a_test, 1)
    np.testing.assert_allclose(g, g_ref)
    np.testing.assert_allclose(gs, gs_ref)


def test_grad_xtensor_cost_raises():
    # A raw xtensor cost or known_grads key gets a clear error, not a crash deep in
    # the grad internals.
    xt = pt.tensor("x", shape=(3,))
    x = as_xtensor(xt, dims=("a",))
    with pytest.raises(TypeError, match="Convert it to a tensor first"):
        pt.grad((x**2).sum(), xt)
    with pytest.raises(TypeError, match="Convert it to a tensor first"):
        pt.grad(None, xt, known_grads={px.math.exp(x): px.math.exp(x)})


def test_grad_exit_diamond():
    # One collapsed exit feeding several consumers -- the cost directly, a re-entered
    # region, and a node using it twice -- must be rebuilt once and accumulate all paths.
    xt = pt.tensor("x", shape=(3,))
    x = as_xtensor(xt, dims=("a",))
    e = px.math.exp(x).sum("a").values
    reentered = (as_xtensor(e * 2.0, dims=()) ** 2).sum().values
    loss = e + reentered + e * e
    [direct] = pt.grad(loss, [xt])
    [ref] = grad_through_lowering(loss, [xt])
    d, r = pytensor.function([xt], [direct, ref])(np.arange(3.0))
    np.testing.assert_allclose(d, r)


def test_grad_known_grads_exit_key():
    # A tensor exit used as a known_grads key is rewritten consistently with the cost
    # graph, so the supplied cotangent lands on the collapsed unit's output.
    xt = pt.tensor("x", shape=(3,))
    x = as_xtensor(xt, dims=("a",))
    w = px.math.exp(x).sum("a").values
    g = pt.grad(None, xt, known_grads={w: pt.constant(2.0)})
    x_test = np.arange(3.0)
    np.testing.assert_allclose(pytensor.function([xt], g)(x_test), 2.0 * np.exp(x_test))
