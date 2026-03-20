import re

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import function, shared
from pytensor.compile import get_mode
from pytensor.configdefaults import config
from pytensor.graph import Apply, Op
from pytensor.scan import until
from pytensor.scan.basic import scan
from pytensor.scan.op import Scan, ScanInfo
from pytensor.tensor import as_tensor, empty, random
from pytensor.tensor.type import dmatrix, dvector, matrix, scalar
from tests.link.jax.test_basic import compare_jax_and_py
from tests.scan.test_basic import ScanCompatibilityTests


jax = pytest.importorskip("jax")


@pytest.mark.parametrize("view", [None, (-1,), slice(-2, None, None)])
def test_scan_sit_sot(view):
    x0 = pt.scalar("x0", dtype="float64")
    xs = scan(
        lambda xtm1: xtm1 + 1,
        outputs_info=[x0],
        n_steps=10,
        return_updates=False,
    )
    if view:
        xs = xs[view]
    test_input_vals = [np.e]
    compare_jax_and_py([x0], [xs], test_input_vals, jax_mode="JAX")


@pytest.mark.parametrize("view", [None, (-1,), slice(-4, -1, None)])
def test_scan_mit_sot(view):
    x0 = pt.vector("x0", dtype="float64", shape=(3,))
    xs = scan(
        lambda xtm3, xtm1: xtm3 + xtm1 + 1,
        outputs_info=[{"initial": x0, "taps": [-3, -1]}],
        n_steps=10,
        return_updates=False,
    )
    if view:
        xs = xs[view]
    test_input_vals = [np.full((3,), np.e)]
    compare_jax_and_py([x0], [xs], test_input_vals, jax_mode="JAX")


@pytest.mark.parametrize("view_x", [None, (-1,), slice(-4, -1, None)])
@pytest.mark.parametrize("view_y", [None, (-1,), slice(-4, -1, None)])
def test_scan_multiple_mit_sot(view_x, view_y):
    x0 = pt.vector("x0", dtype="float64", shape=(3,))
    y0 = pt.vector("y0", dtype="float64", shape=(4,))

    def step(xtm3, xtm1, ytm4, ytm2):
        return xtm3 + ytm4 + 1, xtm1 + ytm2 + 2

    [xs, ys] = scan(
        fn=step,
        outputs_info=[
            {"initial": x0, "taps": [-3, -1]},
            {"initial": y0, "taps": [-4, -2]},
        ],
        n_steps=10,
        return_updates=False,
    )
    if view_x:
        xs = xs[view_x]
    if view_y:
        ys = ys[view_y]

    test_input_vals = [np.full((3,), np.e), np.full((4,), np.pi)]
    compare_jax_and_py([x0, y0], [xs, ys], test_input_vals, jax_mode="JAX")


@pytest.mark.parametrize("view", [None, (-2,), slice(None, None, 2)])
def test_scan_nit_sot(view):
    rng = np.random.default_rng(seed=49)

    xs = pt.vector("x0", dtype="float64", shape=(10,))

    ys = scan(
        lambda x: pt.exp(x), outputs_info=[None], sequences=[xs], return_updates=False
    )
    if view:
        ys = ys[view]
    test_input_vals = [rng.normal(size=10)]
    # We need to remove pushout rewrites, or the whole scan would just be
    # converted to an Elemwise on xs
    jax_fn, _ = compare_jax_and_py(
        [xs], [ys], test_input_vals, jax_mode=get_mode("JAX").excluding("scan_pushout")
    )
    scan_nodes = [
        node for node in jax_fn.maker.fgraph.apply_nodes if isinstance(node.op, Scan)
    ]
    assert len(scan_nodes) == 1


def test_scan_mit_mot():
    def step(xtm1, ytm3, ytm1, rho):
        return (xtm1 + ytm1) * rho, ytm3 * (1 - rho) + ytm1 * rho

    rho = pt.scalar("rho", dtype="float64")
    x0 = pt.vector("xs", shape=(2,))
    y0 = pt.vector("ys", shape=(3,))
    [outs, _] = scan(
        step,
        outputs_info=[x0, {"initial": y0, "taps": [-3, -1]}],
        non_sequences=[rho],
        n_steps=10,
        return_updates=False,
    )
    grads = pt.grad(outs.sum(), wrt=[x0, y0, rho])
    compare_jax_and_py(
        [x0, y0, rho],
        grads,
        [np.arange(2), np.array([0.5, 0.5, 0.5]), np.array(0.95)],
        jax_mode=get_mode("JAX"),
    )


def test_scan_update():
    sh_static = shared(np.array(0.0), name="sh_static")
    sh_update = shared(np.array(1.0), name="sh_update")

    xs, update = scan(
        lambda sh_static, sh_update: (
            sh_static + sh_update,
            {sh_update: sh_update * 2},
        ),
        outputs_info=[None],
        non_sequences=[sh_static, sh_update],
        strict=True,
        n_steps=7,
    )

    jax_fn = function([], xs, updates=update, mode="JAX")
    np.testing.assert_array_equal(jax_fn(), np.array([1, 2, 4, 8, 16, 32, 64]) + 0.0)

    sh_static.set_value(1.0)
    np.testing.assert_array_equal(
        jax_fn(), np.array([128, 256, 512, 1024, 2048, 4096, 8192]) + 1.0
    )

    sh_static.set_value(2.0)
    sh_update.set_value(1.0)
    np.testing.assert_array_equal(jax_fn(), np.array([1, 2, 4, 8, 16, 32, 64]) + 2.0)


def test_scan_rng_update():
    rng = shared(np.random.default_rng(190), name="rng")

    def update_fn(rng):
        new_rng, x = random.normal(rng=rng).owner.outputs
        return x, {rng: new_rng}

    xs, update = scan(
        update_fn,
        outputs_info=[None],
        non_sequences=[rng],
        strict=True,
        n_steps=10,
    )

    # Without updates
    with pytest.warns(
        UserWarning,
        match=re.escape("[rng] will not be used in the compiled JAX graph"),
    ):
        jax_fn = function([], [xs], updates=None, mode="JAX")

    res1, res2 = jax_fn(), jax_fn()
    assert np.unique(res1).size == 10
    assert np.unique(res2).size == 10
    np.testing.assert_array_equal(res1, res2)

    # With updates
    with pytest.warns(
        UserWarning,
        match=re.escape("[rng] will not be used in the compiled JAX graph"),
    ):
        jax_fn = function([], [xs], updates=update, mode="JAX")

    res1, res2 = jax_fn(), jax_fn()
    assert np.unique(res1).size == 10
    assert np.unique(res2).size == 10
    assert np.all(np.not_equal(res1, res2))


@pytest.mark.xfail(raises=NotImplementedError)
def test_scan_while():
    xs = scan(
        lambda x: (x + 1, until(x < 10)),
        outputs_info=[pt.zeros(())],
        n_steps=100,
        return_updates=False,
    )

    compare_jax_and_py([], [xs], [])


def test_scan_mitsot_with_nonseq():
    a_pt = scalar("a")

    def input_step_fn(y_tm1, y_tm3, a):
        y_tm1.name = "y_tm1"
        y_tm3.name = "y_tm3"
        res = (y_tm1 + y_tm3) * a
        res.name = "y_t"
        return res

    y_scan_pt = scan(
        fn=input_step_fn,
        outputs_info=[
            {
                "initial": pt.as_tensor_variable(
                    np.r_[-1.0, 1.3, 0.0].astype(config.floatX)
                ),
                "taps": [-1, -3],
            },
        ],
        non_sequences=[a_pt],
        n_steps=10,
        name="y_scan",
        return_updates=False,
    )
    y_scan_pt.name = "y"
    y_scan_pt.owner.inputs[0].name = "y_all"

    test_input_vals = [np.array(10.0).astype(config.floatX)]
    compare_jax_and_py([a_pt], [y_scan_pt], test_input_vals, jax_mode="JAX")


@pytest.mark.parametrize("x0_func", [dvector, dmatrix])
@pytest.mark.parametrize("A_func", [dmatrix, dmatrix])
def test_nd_scan_sit_sot(x0_func, A_func):
    x0 = x0_func("x0")
    A = A_func("A")

    n_steps = 3
    k = 3

    # Must specify mode = JAX for the inner func to avoid a GEMM Op in the JAX graph
    xs = scan(
        lambda X, A: A @ X,
        non_sequences=[A],
        outputs_info=[x0],
        n_steps=n_steps,
        return_updates=False,
    )

    x0_val = (
        np.arange(k, dtype=config.floatX)
        if x0.ndim == 1
        else np.diag(np.arange(k, dtype=config.floatX))
    )
    A_val = np.eye(k, dtype=config.floatX)

    test_input_vals = [x0_val, A_val]
    compare_jax_and_py([x0, A], [xs], test_input_vals, jax_mode="JAX")


def test_nd_scan_sit_sot_with_seq():
    n_steps = 3
    k = 3

    x = pt.matrix("x0", shape=(n_steps, k))
    A = pt.matrix("A", shape=(k, k))

    # Must specify mode = JAX for the inner func to avoid a GEMM Op in the JAX graph
    xs = scan(
        lambda X, A: A @ X,
        non_sequences=[A],
        sequences=[x],
        n_steps=n_steps,
        return_updates=False,
    )

    x_val = np.arange(n_steps * k, dtype=config.floatX).reshape(n_steps, k)
    A_val = np.eye(k, dtype=config.floatX)

    test_input_vals = [x_val, A_val]
    compare_jax_and_py([x, A], [xs], test_input_vals, jax_mode="JAX")


def test_nd_scan_mit_sot():
    x0 = pt.matrix("x0", shape=(3, 3))
    A = pt.matrix("A", shape=(3, 3))
    B = pt.matrix("B", shape=(3, 3))

    # Must specify mode = JAX for the inner func to avoid a GEMM Op in the JAX graph
    xs = scan(
        lambda xtm3, xtm1, A, B: A @ xtm3 + B @ xtm1,
        outputs_info=[{"initial": x0, "taps": [-3, -1]}],
        non_sequences=[A, B],
        n_steps=10,
        return_updates=False,
    )

    x0_val = np.arange(9, dtype=config.floatX).reshape(3, 3)
    A_val = np.eye(3, dtype=config.floatX)
    B_val = np.eye(3, dtype=config.floatX)

    test_input_vals = [x0_val, A_val, B_val]
    compare_jax_and_py([x0, A, B], [xs], test_input_vals, jax_mode="JAX")


def test_nd_scan_sit_sot_with_carry():
    x0 = pt.vector("x0", shape=(3,))
    A = pt.matrix("A", shape=(3, 3))

    def step(x, A):
        return A @ x, x.sum()

    # Must specify mode = JAX for the inner func to avoid a GEMM Op in the JAX graph
    xs = scan(
        step,
        outputs_info=[x0, None],
        non_sequences=[A],
        n_steps=10,
        mode=get_mode("JAX"),
        return_updates=False,
    )

    x0_val = np.arange(3, dtype=config.floatX)
    A_val = np.eye(3, dtype=config.floatX)

    test_input_vals = [x0_val, A_val]
    compare_jax_and_py([x0, A], xs, test_input_vals, jax_mode="JAX")


def test_default_mode_excludes_incompatible_rewrites():
    # See issue #426
    A = matrix("A")
    B = matrix("B")
    out = scan(
        lambda a, b: a @ b,
        outputs_info=[A],
        non_sequences=[B],
        n_steps=2,
        return_updates=False,
    )
    compare_jax_and_py([A, B], [out], [np.eye(3), np.eye(3)], jax_mode="JAX")


def test_dynamic_sequence_length():
    # Imported here to not trigger import of JAX in non-JAX CI jobs
    from pytensor.link.jax.dispatch.basic import jax_funcify

    class IncWithoutStaticShape(Op):
        def make_node(self, x):
            x = pt.as_tensor_variable(x)
            return Apply(self, [x], [pt.tensor(shape=(None,) * x.type.ndim)])

        def perform(self, node, inputs, outputs):
            outputs[0][0] = inputs[0] + 1

    @jax_funcify.register(IncWithoutStaticShape)
    def _(op, **kwargs):
        return lambda x: x + 1

    inc_without_static_shape = IncWithoutStaticShape()

    x = pt.tensor("x", shape=(None, 3))

    out = scan(
        lambda x: inc_without_static_shape(x),
        outputs_info=[None],
        sequences=[x],
        return_updates=False,
    )
    f = function([x], out, mode=get_mode("JAX").excluding("scan"))
    assert sum(isinstance(node.op, Scan) for node in f.maker.fgraph.apply_nodes) == 1
    np.testing.assert_allclose(f([[1, 2, 3]]), np.array([[2, 3, 4]]))

    # This works if we use JAX scan internally, but not if we use a fori_loop with a buffer allocated by us
    np.testing.assert_allclose(f(np.zeros((0, 3))), np.empty((0, 3)))

    # With known static shape we should always manage, regardless of the internal implementation
    out2 = scan(
        lambda x: pt.specify_shape(inc_without_static_shape(x), x.shape),
        outputs_info=[None],
        sequences=[x],
        return_updates=False,
    )
    f2 = function([x], out2, mode=get_mode("JAX").excluding("scan"))
    np.testing.assert_allclose(f2([[1, 2, 3]]), np.array([[2, 3, 4]]))
    np.testing.assert_allclose(f2(np.zeros((0, 3))), np.empty((0, 3)))


def test_higher_order_derivatives():
    ScanCompatibilityTests.check_higher_order_derivative(mode="JAX")


def test_trace_truncation_regression_bug():
    # Regression bug for a case where the final recurring trace size matched exactly with the number of steps
    n_steps = as_tensor(7, dtype=int)
    x0 = scalar("x0")
    x0_buffer = empty((n_steps,))[0].set(x0)

    # I don't know how to create such a Scan naturally, so we use the internal API
    xtm1 = x0.type()
    scan_op = Scan(
        inputs=[xtm1],
        outputs=[xtm1 + 1],
        info=ScanInfo(
            n_seqs=0,
            mit_mot_in_slices=(),
            mit_mot_out_slices=(),
            mit_sot_in_slices=(),
            sit_sot_in_slices=((-1,),),
            n_nit_sot=0,
            n_untraced_sit_sot=0,
            n_non_seqs=0,
            as_while=False,
        ),
    )

    xs_with_x0 = scan_op(n_steps, x0_buffer)

    compare_jax_and_py(
        [x0],
        [xs_with_x0],
        [np.array(0)],
        jax_mode="JAX",
    )
