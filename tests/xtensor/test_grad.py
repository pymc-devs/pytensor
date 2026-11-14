import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
import pytensor.xtensor as px
from pytensor.gradient import DisconnectedInputError
from pytensor.graph import rewrite_graph
from pytensor.xtensor.gradient import grad, pullback
from pytensor.xtensor.gradient import pushforward as x_pushforward
from pytensor.xtensor.type import as_xtensor
from pytensor.xtensor.vectorization import vectorize_graph


pytestmark = pytest.mark.filterwarnings("error")


def grad_through_lowering(cost, wrt):
    """Reference: lower the xtensor graph to tensor Ops, then use the tensor grad."""
    cost = rewrite_graph(cost, include=("lower_xtensor",), clone=True)
    return pt.grad(cost, wrt)


def build_cases():
    xt = pt.tensor("x", shape=(3, 4))
    x = as_xtensor(xt, dims=("a", "b"))
    yt = pt.tensor("y", shape=(4, 2))
    y = as_xtensor(yt, dims=("b", "c"))
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
        # Swapping names exercises Rename as a positional relabel, not a permutation.
        ("rename_swap", (x.rename({"a": "b", "b": "a"}).sum("a") ** 2).sum(), [xt]),
        ("dot", (px.dot(x, y, dim="b") ** 2).sum(), [xt, yt]),
        # A variable repeated inside a node and reused by an outer node.
        ("repeated_nested", (x * x * x).sum(), [xt]),
    ]


@pytest.mark.parametrize(
    "loss, wrt",
    [pytest.param(loss, wrt, id=name) for name, loss, wrt in build_cases()],
)
def test_grad_matches_lowering(loss, wrt):
    rng = np.random.default_rng(7)
    test_vals = [rng.normal(size=w.type.shape).astype(w.type.dtype) for w in wrt]
    fn = pytensor.function(
        wrt, [*grad(loss.values, wrt), *grad_through_lowering(loss.values, wrt)]
    )
    out = fn(*test_vals)
    for direct, ref in zip(out[: len(wrt)], out[len(wrt) :]):
        np.testing.assert_allclose(direct, ref)


def test_grad_repeated_input():
    # Repeated inputs must accumulate, with no factor-of-N error.
    xt = pt.vector("x", shape=(3,))
    x = as_xtensor(xt, dims=("a",))
    x_test = np.array([1.0, 2.0, 3.0])
    for power, loss in [(2, (x * x).sum()), (3, (x * x * x).sum())]:
        g = pytensor.function([xt], grad(loss.values, xt))(x_test)
        np.testing.assert_allclose(g, power * x_test ** (power - 1))


def test_grad_single_wrt_returns_variable():
    xt = pt.vector("x", shape=(3,))
    loss = ((as_xtensor(xt, dims=("a",))) ** 2).sum().values
    assert isinstance(grad(loss, xt), pytensor.graph.basic.Variable)
    assert isinstance(grad(loss, [xt]), list)


def test_grad_non_scalar_cost_raises():
    xt = pt.vector("x", shape=(3,))
    out = (as_xtensor(xt, dims=("a",)) ** 2).values
    with pytest.raises(TypeError, match="Cost must be a scalar"):
        grad(out, xt)


@pytest.mark.parametrize("second_order", [False, True], ids=["first", "second"])
def test_grad_xtensor_wrt(second_order):
    # Differentiating with respect to an xtensor variable returns an xtensor cotangent
    # carrying its dims, and does so again when repeated.
    x = px.xtensor("x", dims=("a",), shape=(3,))
    loss = (x * x * x).sum()  # d/dx = 3x^2, d2/dx2 = 6x
    g = grad(loss.values, x)
    assert g.type.dims == ("a",)
    if second_order:
        g = grad(g.sum().values, x)
    x_test = np.array([1.0, 2.0, 3.0])
    expected = 6 * x_test if second_order else 3 * x_test**2
    np.testing.assert_allclose(pytensor.function([x], g.values)(x_test), expected)


def test_grad_second_order_matches_lowering():
    W = pytensor.shared(np.ones((3, 2)), name="W")
    xt = pt.vector("x", shape=(3,))
    x = as_xtensor(xt, dims=("a",))
    loss = ((y := px.dot(x, as_xtensor(W, dims=("a", "b")), dim="a")) * y).sum().values
    g2 = grad(grad(loss, W).sum(), W)
    g2_ref = pt.grad(grad_through_lowering(loss, W).sum(), W)
    direct, ref = pytensor.function([xt], [g2, g2_ref])(np.arange(3.0))
    np.testing.assert_allclose(direct, ref)


def test_grad_multiple_wrt():
    xt = pt.vector("x", shape=(3,))
    yt = pt.vector("y", shape=(3,))
    x = as_xtensor(xt, dims=("a",))
    y = as_xtensor(yt, dims=("a",))
    loss = (px.math.exp(x) * y).sum().values
    gx, gy = grad(loss, [xt, yt])
    x_test, y_test = np.array([0.5, 1.0, 1.5]), np.array([2.0, -1.0, 0.5])
    got_x, got_y = pytensor.function([xt, yt], [gx, gy])(x_test, y_test)
    np.testing.assert_allclose(got_x, np.exp(x_test) * y_test)
    np.testing.assert_allclose(got_y, np.exp(x_test))


def test_grad_disconnected_wrt():
    # The tensor rule's own policy applies: raise by default, zeros on request.
    xt = pt.vector("x", shape=(3,))
    zt = pt.vector("z", shape=(3,))
    loss = ((as_xtensor(xt, dims=("a",))) ** 2).sum().values
    with pytest.raises(DisconnectedInputError):
        grad(loss, zt)
    gz = grad(loss, zt, disconnected_inputs="ignore")
    np.testing.assert_allclose(pytensor.function([zt], gz)(np.arange(3.0)), np.zeros(3))


def test_grad_through_indexing():
    xt = pt.tensor("x", shape=(3, 4))
    x = as_xtensor(xt, dims=("a", "b"))
    loss = (x.isel(a=1) ** 2).sum().values
    x_test = np.arange(12.0).reshape(3, 4)
    expected = np.zeros((3, 4))
    expected[1] = 2 * x_test[1]
    np.testing.assert_allclose(
        pytensor.function([xt], grad(loss, xt))(x_test), expected
    )


def test_grad_through_slice_indexing():
    at = pt.tensor("a", shape=(6, 5))
    xa = as_xtensor(at, dims=("i", "j"))
    k = pt.iscalar("k")
    loss = (xa.isel(i=slice(1, 4), j=slice(0, 2)) ** 2).sum().values
    sym_loss = (xa.isel(i=slice(k, 4)) ** 2).sum().values
    outs = [grad(loss, at), *grad_through_lowering(loss, [at])]
    outs += [grad(sym_loss, at), *grad_through_lowering(sym_loss, [at])]
    a_test = np.random.default_rng(2).normal(size=(6, 5))
    g, g_ref, gs, gs_ref = pytensor.function([at, k], outs)(a_test, 1)
    np.testing.assert_allclose(g, g_ref)
    np.testing.assert_allclose(gs, gs_ref)


def test_grad_matches_finite_differences():
    rng = np.random.default_rng(420)
    xt = pt.vector("x", shape=(3,))
    wt = pt.matrix("w", shape=(3, 2))
    x = as_xtensor(xt, dims=("a",))
    w = as_xtensor(wt, dims=("a", "b"))
    loss = (px.dot(x, w, dim="a") ** 2).sum().values
    x_test, w_test = rng.normal(size=(3,)), rng.normal(size=(3, 2))

    cost_fn = pytensor.function([xt, wt], loss)
    got = pytensor.function([xt, wt], grad(loss, xt))(x_test, w_test)
    eps = 1e-6
    expected = np.empty_like(x_test)
    for i in range(x_test.size):
        step = np.zeros_like(x_test)
        step[i] = eps
        expected[i] = (
            cost_fn(x_test + step, w_test) - cost_fn(x_test - step, w_test)
        ) / (2 * eps)
    np.testing.assert_allclose(got, expected, rtol=1e-5)


def test_pullback():
    # A non-scalar output and an arbitrary cotangent, rather than grad's ones seed.
    xt = pt.tensor("x", shape=(3,))
    x = as_xtensor(xt, dims=("a",))
    out = (px.math.exp(x) * x).values
    cot = pt.tensor("cot", shape=(3,))
    [vjp] = pullback(out, [xt], [cot])
    x_test, cot_test = np.array([0.5, 1.0, 1.5]), np.array([2.0, -1.0, 0.5])
    got = pytensor.function([xt, cot], vjp)(x_test, cot_test)
    np.testing.assert_allclose(got, cot_test * np.exp(x_test) * (x_test + 1))


def test_pushforward():
    xt = pt.tensor("x", shape=(3,))
    x = as_xtensor(xt, dims=("a",))
    out = (px.math.exp(x) * x).values
    v = pt.tensor("v", shape=(3,))
    jvp = x_pushforward(out, [xt], [v])
    x_test, v_test = np.array([0.5, 1.0, 1.5]), np.array([1.0, -2.0, 0.5])
    got = pytensor.function([xt, v], jvp)(x_test, v_test)
    np.testing.assert_allclose(got, v_test * np.exp(x_test) * (x_test + 1))


def test_grad_chained_regions():
    # Leaving and re-entering the xtensor world.
    xt = pt.tensor("x", shape=(3, 4))
    x = as_xtensor(xt, dims=("a", "b"))
    reentered = as_xtensor(x.sum("a").values + 1.0, dims=("b",))
    loss = (reentered**2).sum().values
    d, r = pytensor.function(
        [xt], [grad(loss, xt), *grad_through_lowering(loss, [xt])]
    )(np.random.default_rng(0).normal(size=(3, 4)))
    np.testing.assert_allclose(d, r)


def test_grad_diamond():
    # One xtensor result feeding several consumers must accumulate every path.
    xt = pt.tensor("x", shape=(3,))
    x = as_xtensor(xt, dims=("a",))
    e = px.math.exp(x).sum("a").values
    reentered = (as_xtensor(e * 2.0, dims=()) ** 2).sum().values
    loss = e + reentered + e * e
    d, r = pytensor.function(
        [xt], [grad(loss, xt), *grad_through_lowering(loss, [xt])]
    )(np.arange(3.0))
    np.testing.assert_allclose(d, r)


def test_grad_unrelated_to_random_region():
    # A draw elsewhere in the cost must not break grad wrt an unrelated variable.
    theta = pt.scalar("theta")
    rng = px.random.shared_rng(seed=0)
    _, draw = px.random.normal(
        0.0, 1.0, extra_dims={"a": 3}, rng=rng, return_next_rng=True
    )
    cost = draw.sum().values * 0.0 + theta**2
    np.testing.assert_allclose(grad(cost, theta).eval({theta: 3.0}), 6.0)


def test_grad_through_random_region():
    # Reparameterized draw: grad flows through the deterministic use of the draw.
    rng = px.random.rng("rng")
    at = pt.tensor("a", shape=(3,))
    a = as_xtensor(at, dims=("d",))
    _, eps = px.random.normal(
        0.0, 1.0, extra_dims={"d": 3}, rng=rng, return_next_rng=True
    )
    loss = ((a * eps) ** 2).sum().values
    d, r = pytensor.function(
        [at, rng], [grad(loss, at), *grad_through_lowering(loss, [at])]
    )(np.arange(1.0, 4.0), np.random.default_rng(3))
    np.testing.assert_allclose(d, r)


def test_forward_work_is_not_duplicated():
    # Splicing barriers clones the path from wrt to the outputs. Once they are stripped
    # that clone is identical to the original, so it must merge away rather than leave
    # the forward pass computed twice.
    xt = pt.tensor("x", shape=(3,))
    x = as_xtensor(xt, dims=("a",))
    loss = (px.math.exp(x) * x).sum().values
    direct = pytensor.function([xt], [loss, grad(loss, xt)])
    lowered = rewrite_graph(loss, include=("lower_xtensor",), clone=True)
    reference = pytensor.function([xt], [lowered, pt.grad(lowered, xt)])
    assert len(direct.maker.fgraph.apply_nodes) == len(
        reference.maker.fgraph.apply_nodes
    )


def test_pt_grad_points_at_this_module():
    xt = pt.vector("x", shape=(3,))
    loss = ((as_xtensor(xt, dims=("a",))) ** 2).sum().values
    with pytest.raises(NotImplementedError, match=r"pytensor\.xtensor\.gradient\.grad"):
        pt.grad(loss, xt)


def test_vectorize_batched_wrt():
    # Vectorizing a recorded derivative records it again over the vectorized graph.
    # Batch elements are independent, so that is the batched derivative.
    xt = pt.vector("x", shape=(3,))
    x = as_xtensor(xt, dims=("a",))
    g = grad((px.math.exp(x) * x).sum().values, xt)
    xb = pt.matrix("xb", shape=(2, 3))
    gb = vectorize_graph(g, {xt: xb}, new_tensor_dims=("batch",))
    xb_test = np.array([[0.5, 1.0, 1.5], [1.5, 2.0, 2.5]])
    row = pytensor.function([xt], g)
    np.testing.assert_allclose(
        pytensor.function([xb], gb)(xb_test), np.stack([row(r) for r in xb_test])
    )


def test_vectorize_broadcasts_unbatched_cotangent():
    # A seed the batching did not touch applies unchanged to every batch element, but
    # has to be shaped like the batched output for the tensor rule to take it.
    xt = pt.vector("x", shape=(3,))
    x = as_xtensor(xt, dims=("a",))
    cot = pt.vector("c", shape=(3,))
    [vjp] = pullback((px.math.exp(x) * x).values, [xt], [cot])
    xb = pt.matrix("xb", shape=(2, 3))
    vjpb = vectorize_graph(vjp, {xt: xb}, new_tensor_dims=("batch",))
    xb_test = np.array([[0.5, 1.0, 1.5], [1.5, 2.0, 2.5]])
    c_test = np.array([2.0, -1.0, 0.5])
    row = pytensor.function([xt, cot], vjp)
    np.testing.assert_allclose(
        pytensor.function([xb, cot], vjpb)(xb_test, c_test),
        np.stack([row(r, c_test) for r in xb_test]),
    )


def test_grad_wrt_interior_xtensor():
    # An intermediate is kept out of the lowering so it survives as a handle.
    xt = pt.tensor("x", shape=(3,))
    x = as_xtensor(xt, dims=("a",))
    inter = px.math.exp(x)
    loss = (inter * inter).sum().values
    x_test = np.array([0.5, 1.0, 1.5])
    g = grad(loss, inter)
    assert g.type.dims == ("a",)
    np.testing.assert_allclose(
        pytensor.function([xt], g.values)(x_test), 2 * np.exp(x_test)
    )


def test_grad_wrt_interior_tensor():
    xt = pt.tensor("x", shape=(3, 4))
    x = as_xtensor(xt, dims=("a", "b"))
    w = px.math.exp(x).values
    x_test = np.random.default_rng(1).normal(size=(3, 4))
    np.testing.assert_allclose(
        pytensor.function([xt], grad((w**2).sum(), w))(x_test), 2 * np.exp(x_test)
    )


def test_grad_consider_constant_interior():
    # Holding an intermediate constant must actually stop the gradient there; silently
    # ignoring it would hand back a gradient that had not been stopped anywhere.
    xt = pt.tensor("x", shape=(3,))
    x = as_xtensor(xt, dims=("a",))
    inter = px.math.exp(x)
    loss = ((inter * inter).sum() + (x * x).sum()).values
    g = grad(loss, xt, consider_constant=[inter])
    x_test = np.array([0.5, 1.0, 1.5])
    # Only the (x * x) path contributes, so d/dx = 2x.
    np.testing.assert_allclose(pytensor.function([xt], g)(x_test), 2 * x_test)


def test_vectorize_unbatched_wrt_gives_per_example_grads():
    # The gradient is plain tensor Ops, so vectorizing it needs nothing special and
    # gives one derivative per batch element even for a shared parameter.
    W = pytensor.shared(np.ones((3, 2)), name="W")
    xt = pt.vector("x", shape=(3,))
    x = as_xtensor(xt, dims=("a",))
    y = px.dot(x, as_xtensor(W, dims=("a", "b")), dim="a")
    gW = grad((y * y).sum().values, W)
    xb = pt.matrix("xb", shape=(2, 3))
    gWb = vectorize_graph(gW, {xt: xb}, new_tensor_dims=("batch",))
    xb_test = np.array([[0.5, 1.0, 1.5], [1.5, 2.0, 2.5]])
    row = pytensor.function([xt], gW)
    np.testing.assert_allclose(
        pytensor.function([xb], gWb)(xb_test), np.stack([row(r) for r in xb_test])
    )


@pytest.mark.parametrize("labelled", [False, True], ids=["tensor", "xtensor"])
def test_pullback_pushforward_carry_types(labelled):
    # A cotangent carries the type of its output and a tangent that of its wrt. The
    # wrt is the xtensor `x` either way, so both results come back with its dims,
    # whether the graph leaves the xtensor world before the output or not.
    x = px.xtensor("x", dims=("a",), shape=(3,))
    out = px.math.exp(x) * x
    cotangent = px.xtensor("c", dims=("a",), shape=(3,))
    if not labelled:
        out, cotangent = out.values, pt.vector("c", shape=(3,))
    tangent = px.xtensor("t", dims=("a",), shape=(3,))

    x_test, s_test = np.array([0.5, 1.0, 1.5]), np.array([2.0, -1.0, 0.5])
    expected = s_test * np.exp(x_test) * (x_test + 1)

    [vjp] = pullback(out, [x], [cotangent])
    assert vjp.type.dims == ("a",)
    np.testing.assert_allclose(
        pytensor.function([x, cotangent], vjp.values)(x_test, s_test), expected
    )

    jvp = x_pushforward(out, [x], [tangent])
    assert getattr(jvp.type, "dims", None) == (("a",) if labelled else None)
    values = jvp.values if labelled else jvp
    np.testing.assert_allclose(
        pytensor.function([x, tangent], values)(x_test, s_test), expected
    )


@pytest.mark.parametrize(
    "out_shape, seed_shape",
    [((3, 3), (3, 3)), ((2, 3), (3, 2))],
    ids=["square", "rect"],
)
def test_cotangent_aligns_by_dim_name(out_shape, seed_shape):
    # xtensor aligns by dim name, but lowering makes dims positional. A cotangent
    # written in another dim order means the same thing and must be transposed onto the
    # output; otherwise it lands in the gradient transposed, and silently so whenever
    # the shapes happen to allow it -- which the square case covers.
    x = px.xtensor("x", dims=("a", "b"), shape=out_shape)
    cot = px.xtensor("c", dims=("b", "a"), shape=seed_shape)
    [vjp] = pullback(px.math.exp(x), [x], [cot])
    xv = np.arange(np.prod(out_shape), dtype=float).reshape(out_shape)
    cv = np.arange(np.prod(seed_shape), dtype=float).reshape(seed_shape) * 10
    np.testing.assert_allclose(
        pytensor.function([x, cot], vjp.values)(xv, cv), cv.T * np.exp(xv)
    )


def test_tangent_aligns_by_dim_name():
    x = px.xtensor("x", dims=("a", "b"), shape=(3, 3))
    tan = px.xtensor("t", dims=("b", "a"), shape=(3, 3))
    jvp = x_pushforward(px.math.exp(x), [x], [tan])
    xv = np.arange(9.0).reshape(3, 3)
    tv = np.arange(9.0).reshape(3, 3) * 10
    np.testing.assert_allclose(
        pytensor.function([x, tan], jvp.values)(xv, tv), tv.T * np.exp(xv)
    )


def test_unlabelled_cotangent_for_labelled_output_raises():
    x = px.xtensor("x", dims=("a",), shape=(3,))
    with pytest.raises(TypeError, match="both be labelled"):
        pullback(px.math.exp(x), [x], [pt.vector("c", shape=(3,))])
