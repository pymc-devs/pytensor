import re

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import function, shared
from pytensor.compile import get_mode
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.scan import until
from pytensor.scan.basic import scan
from pytensor.scan.op import Scan
from pytensor.tensor import random
from pytensor.tensor.math import gammaln, log
from pytensor.tensor.type import dmatrix, dvector, lscalar, matrix, scalar, vector
from tests.link.jax.test_basic import compare_jax_and_py


jax = pytest.importorskip("jax")


@pytest.mark.parametrize("view", [None, (-1,), slice(-2, None, None)])
def test_scan_sit_sot(view):
    x0 = pt.scalar("x0", dtype="float64")
    xs, _ = scan(
        lambda xtm1: xtm1 + 1,
        outputs_info=[x0],
        n_steps=10,
    )
    if view:
        xs = xs[view]
    fg = FunctionGraph([x0], [xs])
    test_input_vals = [np.e]
    compare_jax_and_py(fg, test_input_vals, jax_mode="JAX")


@pytest.mark.parametrize("view", [None, (-1,), slice(-4, -1, None)])
def test_scan_mit_sot(view):
    x0 = pt.vector("x0", dtype="float64", shape=(3,))
    xs, _ = scan(
        lambda xtm3, xtm1: xtm3 + xtm1 + 1,
        outputs_info=[{"initial": x0, "taps": [-3, -1]}],
        n_steps=10,
    )
    if view:
        xs = xs[view]
    fg = FunctionGraph([x0], [xs])
    test_input_vals = [np.full((3,), np.e)]
    compare_jax_and_py(fg, test_input_vals, jax_mode="JAX")


@pytest.mark.parametrize("view_x", [None, (-1,), slice(-4, -1, None)])
@pytest.mark.parametrize("view_y", [None, (-1,), slice(-4, -1, None)])
def test_scan_multiple_mit_sot(view_x, view_y):
    x0 = pt.vector("x0", dtype="float64", shape=(3,))
    y0 = pt.vector("y0", dtype="float64", shape=(4,))

    def step(xtm3, xtm1, ytm4, ytm2):
        return xtm3 + ytm4 + 1, xtm1 + ytm2 + 2

    [xs, ys], _ = scan(
        fn=step,
        outputs_info=[
            {"initial": x0, "taps": [-3, -1]},
            {"initial": y0, "taps": [-4, -2]},
        ],
        n_steps=10,
    )
    if view_x:
        xs = xs[view_x]
    if view_y:
        ys = ys[view_y]

    fg = FunctionGraph([x0, y0], [xs, ys])
    test_input_vals = [np.full((3,), np.e), np.full((4,), np.pi)]
    compare_jax_and_py(fg, test_input_vals, jax_mode="JAX")


@pytest.mark.parametrize("view", [None, (-2,), slice(None, None, 2)])
def test_scan_nit_sot(view):
    rng = np.random.default_rng(seed=49)

    xs = pt.vector("x0", dtype="float64", shape=(10,))

    ys, _ = scan(
        lambda x: pt.exp(x),
        outputs_info=[None],
        sequences=[xs],
    )
    if view:
        ys = ys[view]
    fg = FunctionGraph([xs], [ys])
    test_input_vals = [rng.normal(size=10)]
    # We need to remove pushout rewrites, or the whole scan would just be
    # converted to an Elemwise on xs
    jax_fn, _ = compare_jax_and_py(
        fg, test_input_vals, jax_mode=get_mode("JAX").excluding("scan_pushout")
    )
    scan_nodes = [
        node for node in jax_fn.maker.fgraph.apply_nodes if isinstance(node.op, Scan)
    ]
    assert len(scan_nodes) == 1


@pytest.mark.xfail(raises=NotImplementedError)
def test_scan_mit_mot():
    xs = pt.vector("xs", shape=(10,))
    ys, _ = scan(
        lambda xtm2, xtm1: (xtm2 + xtm1),
        outputs_info=[{"initial": xs, "taps": [-2, -1]}],
        n_steps=10,
    )
    grads_wrt_xs = pt.grad(ys.sum(), wrt=xs)
    fg = FunctionGraph([xs], [grads_wrt_xs])
    compare_jax_and_py(fg, [np.arange(10)])


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
    xs, _ = scan(
        lambda x: (x + 1, until(x < 10)),
        outputs_info=[pt.zeros(())],
        n_steps=100,
    )

    fg = FunctionGraph([], [xs])
    compare_jax_and_py(fg, [])


def test_scan_SEIR():
    """Test a scan implementation of a SEIR model.

    SEIR model definition:
    S[t+1] = S[t] - B[t]
    E[t+1] = E[t] +B[t] - C[t]
    I[t+1] = I[t+1] + C[t] - D[t]

    B[t] ~ Binom(S[t], beta)
    C[t] ~ Binom(E[t], gamma)
    D[t] ~ Binom(I[t], delta)
    """

    def binomln(n, k):
        return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

    def binom_log_prob(n, p, value):
        return binomln(n, value) + value * log(p) + (n - value) * log(1 - p)

    # sequences
    at_C = vector("C_t", dtype="int32", shape=(8,))
    at_D = vector("D_t", dtype="int32", shape=(8,))
    # outputs_info (initial conditions)
    st0 = lscalar("s_t0")
    et0 = lscalar("e_t0")
    it0 = lscalar("i_t0")
    logp_c = scalar("logp_c")
    logp_d = scalar("logp_d")
    # non_sequences
    beta = scalar("beta")
    gamma = scalar("gamma")
    delta = scalar("delta")

    # TODO: Use random streams when their JAX conversions are implemented.
    # trng = pytensor.tensor.random.RandomStream(1234)

    def seir_one_step(ct0, dt0, st0, et0, it0, logp_c, logp_d, beta, gamma, delta):
        # bt0 = trng.binomial(n=st0, p=beta)
        bt0 = st0 * beta
        bt0 = bt0.astype(st0.dtype)

        logp_c1 = binom_log_prob(et0, gamma, ct0).astype(logp_c.dtype)
        logp_d1 = binom_log_prob(it0, delta, dt0).astype(logp_d.dtype)

        st1 = st0 - bt0
        et1 = et0 + bt0 - ct0
        it1 = it0 + ct0 - dt0
        return st1, et1, it1, logp_c1, logp_d1

    (st, et, it, logp_c_all, logp_d_all), _ = scan(
        fn=seir_one_step,
        sequences=[at_C, at_D],
        outputs_info=[st0, et0, it0, logp_c, logp_d],
        non_sequences=[beta, gamma, delta],
    )
    st.name = "S_t"
    et.name = "E_t"
    it.name = "I_t"
    logp_c_all.name = "C_t_logp"
    logp_d_all.name = "D_t_logp"

    out_fg = FunctionGraph(
        [at_C, at_D, st0, et0, it0, logp_c, logp_d, beta, gamma, delta],
        [st, et, it, logp_c_all, logp_d_all],
    )

    s0, e0, i0 = 100, 50, 25
    logp_c0 = np.array(0.0, dtype=config.floatX)
    logp_d0 = np.array(0.0, dtype=config.floatX)
    beta_val, gamma_val, delta_val = (
        np.array(val, dtype=config.floatX) for val in [0.277792, 0.135330, 0.108753]
    )
    C = np.array([3, 5, 8, 13, 21, 26, 10, 3], dtype=np.int32)
    D = np.array([1, 2, 3, 7, 9, 11, 5, 1], dtype=np.int32)

    test_input_vals = [
        C,
        D,
        s0,
        e0,
        i0,
        logp_c0,
        logp_d0,
        beta_val,
        gamma_val,
        delta_val,
    ]
    compare_jax_and_py(out_fg, test_input_vals, jax_mode="JAX")


def test_scan_mitsot_with_nonseq():
    a_pt = scalar("a")

    def input_step_fn(y_tm1, y_tm3, a):
        y_tm1.name = "y_tm1"
        y_tm3.name = "y_tm3"
        res = (y_tm1 + y_tm3) * a
        res.name = "y_t"
        return res

    y_scan_pt, _ = scan(
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
    )
    y_scan_pt.name = "y"
    y_scan_pt.owner.inputs[0].name = "y_all"

    out_fg = FunctionGraph([a_pt], [y_scan_pt])

    test_input_vals = [np.array(10.0).astype(config.floatX)]
    compare_jax_and_py(out_fg, test_input_vals, jax_mode="JAX")


@pytest.mark.parametrize("x0_func", [dvector, dmatrix])
@pytest.mark.parametrize("A_func", [dmatrix, dmatrix])
def test_nd_scan_sit_sot(x0_func, A_func):
    x0 = x0_func("x0")
    A = A_func("A")

    n_steps = 3
    k = 3

    # Must specify mode = JAX for the inner func to avoid a GEMM Op in the JAX graph
    xs, _ = scan(
        lambda X, A: A @ X,
        non_sequences=[A],
        outputs_info=[x0],
        n_steps=n_steps,
    )

    x0_val = (
        np.arange(k, dtype=config.floatX)
        if x0.ndim == 1
        else np.diag(np.arange(k, dtype=config.floatX))
    )
    A_val = np.eye(k, dtype=config.floatX)

    fg = FunctionGraph([x0, A], [xs])
    test_input_vals = [x0_val, A_val]
    compare_jax_and_py(fg, test_input_vals, jax_mode="JAX")


def test_nd_scan_sit_sot_with_seq():
    n_steps = 3
    k = 3

    x = pt.matrix("x0", shape=(n_steps, k))
    A = pt.matrix("A", shape=(k, k))

    # Must specify mode = JAX for the inner func to avoid a GEMM Op in the JAX graph
    xs, _ = scan(
        lambda X, A: A @ X,
        non_sequences=[A],
        sequences=[x],
        n_steps=n_steps,
    )

    x_val = np.arange(n_steps * k, dtype=config.floatX).reshape(n_steps, k)
    A_val = np.eye(k, dtype=config.floatX)

    fg = FunctionGraph([x, A], [xs])
    test_input_vals = [x_val, A_val]
    compare_jax_and_py(fg, test_input_vals, jax_mode="JAX")


def test_nd_scan_mit_sot():
    x0 = pt.matrix("x0", shape=(3, 3))
    A = pt.matrix("A", shape=(3, 3))
    B = pt.matrix("B", shape=(3, 3))

    # Must specify mode = JAX for the inner func to avoid a GEMM Op in the JAX graph
    xs, _ = scan(
        lambda xtm3, xtm1, A, B: A @ xtm3 + B @ xtm1,
        outputs_info=[{"initial": x0, "taps": [-3, -1]}],
        non_sequences=[A, B],
        n_steps=10,
    )

    fg = FunctionGraph([x0, A, B], [xs])
    x0_val = np.arange(9, dtype=config.floatX).reshape(3, 3)
    A_val = np.eye(3, dtype=config.floatX)
    B_val = np.eye(3, dtype=config.floatX)

    test_input_vals = [x0_val, A_val, B_val]
    compare_jax_and_py(fg, test_input_vals, jax_mode="JAX")


def test_nd_scan_sit_sot_with_carry():
    x0 = pt.vector("x0", shape=(3,))
    A = pt.matrix("A", shape=(3, 3))

    def step(x, A):
        return A @ x, x.sum()

    # Must specify mode = JAX for the inner func to avoid a GEMM Op in the JAX graph
    xs, _ = scan(
        step,
        outputs_info=[x0, None],
        non_sequences=[A],
        n_steps=10,
        mode=get_mode("JAX"),
    )

    fg = FunctionGraph([x0, A], xs)
    x0_val = np.arange(3, dtype=config.floatX)
    A_val = np.eye(3, dtype=config.floatX)

    test_input_vals = [x0_val, A_val]
    compare_jax_and_py(fg, test_input_vals, jax_mode="JAX")


def test_default_mode_excludes_incompatible_rewrites():
    # See issue #426
    A = matrix("A")
    B = matrix("B")
    out, _ = scan(lambda a, b: a @ b, outputs_info=[A], non_sequences=[B], n_steps=2)
    fg = FunctionGraph([A, B], [out])
    compare_jax_and_py(fg, [np.eye(3), np.eye(3)], jax_mode="JAX")


def test_dynamic_sequence_length():
    x = pt.tensor("x", shape=(None,))
    out, _ = scan(lambda x: x + 1, sequences=[x])

    f = function([x], out, mode=get_mode("JAX").excluding("scan"))
    assert sum(isinstance(node.op, Scan) for node in f.maker.fgraph.apply_nodes) == 1
    np.testing.assert_allclose(f([]), [])
    np.testing.assert_allclose(f([1, 2, 3]), np.array([2, 3, 4]))
