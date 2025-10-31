import re

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import function, ifelse, shared
from pytensor.compile import get_mode
from pytensor.configdefaults import config
from pytensor.graph import Apply, Op
from pytensor.scan import until
from pytensor.scan.basic import scan
from pytensor.scan.op import Scan
from pytensor.tensor import random
from pytensor.tensor.math import gammaln, log
from pytensor.tensor.type import dmatrix, dvector, matrix, scalar, vector
from tests.link.jax.test_basic import compare_jax_and_py


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


def SEIR_model_logp():
    """Setup a Scan implementation of a SEIR model.

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
    C_t = vector("C_t", dtype="int32", shape=(1200,))
    D_t = vector("D_t", dtype="int32", shape=(1200,))
    # outputs_info (initial conditions)
    st0 = scalar("s_t0")
    et0 = scalar("e_t0")
    it0 = scalar("i_t0")
    # non_sequences
    beta = scalar("beta")
    gamma = scalar("gamma")
    delta = scalar("delta")

    def seir_one_step(ct0, dt0, st0, et0, it0, beta, gamma, delta):
        # bt0 = trng.binomial(n=st0, p=beta)
        bt0 = st0 * beta
        bt0 = bt0.astype(st0.dtype)

        logp_c1 = binom_log_prob(et0, gamma, ct0)
        logp_d1 = binom_log_prob(it0, delta, dt0)

        st1 = st0 - bt0
        et1 = et0 + bt0 - ct0
        it1 = it0 + ct0 - dt0
        return st1, et1, it1, logp_c1, logp_d1

    (st, et, it, logp_c_all, logp_d_all) = scan(
        fn=seir_one_step,
        sequences=[C_t, D_t],
        outputs_info=[st0, et0, it0, None, None],
        non_sequences=[beta, gamma, delta],
        return_updates=False,
    )
    st.name = "S_t"
    et.name = "E_t"
    it.name = "I_t"
    logp_c_all.name = "C_t_logp"
    logp_d_all.name = "D_t_logp"

    st0_val, et0_val, it0_val = np.array(100.0), np.array(50.0), np.array(25.0)
    beta_val, gamma_val, delta_val = (
        np.array(0.277792),
        np.array(0.135330),
        np.array(0.108753),
    )
    C_t_val = np.array([3, 5, 8, 13, 21, 26, 10, 3] * 150, dtype=np.int32)
    D_t_val = np.array([1, 2, 3, 7, 9, 11, 5, 1] * 150, dtype=np.int32)
    assert C_t_val.shape == D_t_val.shape == C_t.type.shape == D_t.type.shape

    test_input_vals = [
        C_t_val,
        D_t_val,
        st0_val,
        et0_val,
        it0_val,
        beta_val,
        gamma_val,
        delta_val,
    ]

    loss_graph = logp_c_all.sum() + logp_d_all.sum()

    return dict(
        graph_inputs=[C_t, D_t, st0, et0, it0, beta, gamma, delta],
        differentiable_vars=[st0, et0, it0, beta, gamma, delta],
        test_input_vals=test_input_vals,
        loss_graph=loss_graph,
    )


def cyclical_reduction():
    """Setup a Scan implementation of the cyclical reduction algorithm.

    This solves the matrix equation A @ X @ X + B @ X + C = 0 for X

    Adapted from https://github.com/jessegrabowski/gEconpy/blob/da495b22ac383cb6cb5dec15f305506aebef7302/gEconpy/solvers/cycle_reduction.py#L187
    """

    def stabilize(x, jitter=1e-16):
        return x + jitter * pt.eye(x.shape[0])

    def step(A0, A1, A2, A1_hat, norm, step_num, tol):
        def cycle_step(A0, A1, A2, A1_hat, _norm, step_num):
            tmp = pt.dot(
                pt.vertical_stack(A0, A2),
                pt.linalg.solve(
                    stabilize(A1),
                    pt.horizontal_stack(A0, A2),
                    assume_a="gen",
                    check_finite=False,
                ),
            )

            n = A0.shape[0]
            idx_0 = pt.arange(n)
            idx_1 = idx_0 + n
            A1 = A1 - tmp[idx_0, :][:, idx_1] - tmp[idx_1, :][:, idx_0]
            A0 = -tmp[idx_0, :][:, idx_0]
            A2 = -tmp[idx_1, :][:, idx_1]
            A1_hat = A1_hat - tmp[idx_1, :][:, idx_0]

            A0_L1_norm = pt.linalg.norm(A0, ord=1)

            return A0, A1, A2, A1_hat, A0_L1_norm, step_num + 1

        return ifelse(
            norm < tol,
            (A0, A1, A2, A1_hat, norm, step_num),
            cycle_step(A0, A1, A2, A1_hat, norm, step_num),
        )

    A = pt.matrix("A", shape=(20, 20))
    B = pt.matrix("B", shape=(20, 20))
    C = pt.matrix("C", shape=(20, 20))

    norm = np.array(1e9, dtype="float64")
    step_num = pt.zeros((), dtype="int32")
    max_iter = 100
    tol = 1e-7

    (*_, A1_hat, norm, _n_steps) = scan(
        step,
        outputs_info=[A, B, C, B, norm, step_num],
        non_sequences=[tol],
        n_steps=max_iter,
        return_updates=False,
    )
    A1_hat = A1_hat[-1]

    T = -pt.linalg.solve(stabilize(A1_hat), A, assume_a="gen", check_finite=False)

    rng = np.random.default_rng(sum(map(ord, "cycle_reduction")))
    n = A.type.shape[0]
    A_test = rng.standard_normal(size=(n, n))
    C_test = rng.standard_normal(size=(n, n))
    # B must be invertible, so we make it symmetric positive-definite
    B_rand = rng.standard_normal(size=(n, n))
    B_test = B_rand @ B_rand.T + np.eye(n) * 1e-3

    return dict(
        graph_inputs=[A, B, C],
        differentiable_vars=[A, B, C],
        test_input_vals=[A_test, B_test, C_test],
        loss_graph=pt.sum(T),
    )


@pytest.mark.parametrize("gradient_backend", ["PYTENSOR", "JAX"])
@pytest.mark.parametrize("mode", ("0forward", "1backward", "2both"))
@pytest.mark.parametrize("model", [cyclical_reduction, SEIR_model_logp])
def test_scan_benchmark(model, mode, gradient_backend, benchmark):
    model_dict = model()
    graph_inputs = model_dict["graph_inputs"]
    differentiable_vars = model_dict["differentiable_vars"]
    loss_graph = model_dict["loss_graph"]
    test_input_vals = model_dict["test_input_vals"]

    if gradient_backend == "PYTENSOR":
        backward_loss = pt.grad(
            loss_graph,
            wrt=differentiable_vars,
        )

        match mode:
            # TODO: Restore original test separately
            case "0forward":
                graph_outputs = [loss_graph]
            case "1backward":
                graph_outputs = backward_loss
            case "2both":
                graph_outputs = [loss_graph, *backward_loss]
            case _:
                raise ValueError(f"Unknown mode: {mode}")

        jax_fn, _ = compare_jax_and_py(
            graph_inputs,
            graph_outputs,
            test_input_vals,
            jax_mode="JAX",
        )
        jax_fn.trust_input = True

    else:  # gradient_backend == "JAX"
        import jax

        loss_fn_tuple = function(graph_inputs, loss_graph, mode="JAX").vm.jit_fn

        def loss_fn(*args):
            return loss_fn_tuple(*args)[0]

        match mode:
            case "0forward":
                jax_fn = jax.jit(loss_fn_tuple)
            case "1backward":
                jax_fn = jax.jit(
                    jax.grad(loss_fn, argnums=tuple(range(len(graph_inputs))[2:]))
                )
            case "2both":
                value_and_grad_fn = jax.value_and_grad(
                    loss_fn, argnums=tuple(range(len(graph_inputs))[2:])
                )

                @jax.jit
                def jax_fn(*args):
                    loss, grads = value_and_grad_fn(*args)
                    return loss, *grads

            case _:
                raise ValueError(f"Unknown mode: {mode}")

    def block_until_ready(*inputs, jax_fn=jax_fn):
        return [o.block_until_ready() for o in jax_fn(*inputs)]

    block_until_ready(*test_input_vals)  # Warmup

    benchmark.pedantic(block_until_ready, test_input_vals, rounds=200, iterations=1)
