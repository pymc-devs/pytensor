"""Ops on the LLVM path, against PyTensor's reference implementation."""

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import get_mode
from pytensor.tensor.linalg import cholesky, solve_triangular


pytest.importorskip("pytensor_llvm")

LLVM = get_mode("LLVM")
NO_FALLBACK = LLVM.clone(link_kwargs={"numba_fallback": False})
rng = np.random.default_rng(3)


def compare(inputs, outputs, values, mode=NO_FALLBACK, rtol=1e-10, atol=1e-12, check_dtype=True):
    f = pytensor.function(inputs, outputs, mode=mode, on_unused_input="ignore")
    g = pytensor.function(inputs, outputs, mode="FAST_COMPILE", on_unused_input="ignore")
    res, ref = f(*values), g(*values)
    if not isinstance(res, list):
        res, ref = [res], [ref]
    for a, b in zip(res, ref, strict=True):
        if check_dtype:
            assert np.asarray(a).dtype == np.asarray(b).dtype
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    return f


def tasks(f):
    return f.maker.linker.program.tasks


def test_dump_lists_values_and_tasks():
    x = pt.matrix("x")
    f = compare([x], pt.exp(x).sum(axis=1) + x[:, 0], [rng.random((3, 4))])
    text = f.maker.linker.program.dump()
    assert text.startswith("program: 1 inputs, 1 outputs")
    assert "input 0" in text and "tasks:" in text and "outputs:" in text


@pytest.mark.parametrize(
    "index",
    [
        lambda x, a, b: x[a],
        lambda x, a, b: x[1:, a],
        lambda x, a, b: x[::-1, a:b],
        lambda x, a, b: x[-2:, :, ::2],
        lambda x, a, b: x[a, b, 1],
        lambda x, a, b: x[b:a:-1],
    ],
)
def test_subtensor(index):
    x = pt.tensor3("x")
    a, b = pt.lscalars("a", "b")
    compare([x, a, b], index(x, a, b) * 1.0, [rng.normal(size=(6, 5, 4)), 1, 4])


def test_inc_and_set_subtensor_broadcasting():
    x = pt.matrix("x")
    y = pt.vector("y")
    i = pt.lscalar("i")
    for fn in (pt.inc_subtensor, pt.set_subtensor):
        outs = [fn(x[1:, i], y), fn(x[::2], 2.5), fn(x[i], y[:1] * 3), fn(x[:, ::-1], 1.0)]
        compare([x, y, i], outs, [rng.normal(size=(5, 4)), rng.normal(size=4), 2])


@pytest.mark.parametrize("idx_dtype", ["int64", "int32", "uint8"])
def test_take_patterns(idx_dtype):
    x = pt.tensor3("x")
    v = pt.vector("v", dtype=idx_dtype)
    m = pt.matrix("m", dtype=idx_dtype)
    outs = [x[v], x[:, v], x[:, :, v], x[m], x[:, m], x[v, v], x[:, m, m]]
    vi = [0, 2, 1, 2] if idx_dtype == "uint8" else [0, 2, -1, 2]
    vals = [rng.normal(size=(5, 4, 3)), np.array(vi, dtype=idx_dtype), np.array([[0, 1], [2, 2]], dtype=idx_dtype)]
    compare([x, v, m], outs, vals)


def test_take_out_of_bounds_raises():
    x = pt.vector("x")
    v = pt.lvector("v")
    f = pytensor.function([x, v], x[v] * 2, mode=NO_FALLBACK)
    with pytest.raises(Exception, match="out of bounds"):
        f(np.arange(3.0), np.array([0, 3]))


@pytest.mark.parametrize("set_", [False, True])
def test_inc_and_set_take_with_duplicates(set_):
    x = pt.matrix("x")
    v = pt.lvector("v")
    y = pt.matrix("y")
    fn = pt.set_subtensor if set_ else pt.inc_subtensor
    idx = np.array([0, 2, 2, -1])
    z = pt.matrix("z")
    outs = [fn(x[v], y), fn(x[v], 1.5), fn(x[:, v], z)]
    compare([x, v, y, z], outs, [rng.normal(size=(5, 3)), idx, rng.normal(size=(4, 3)), rng.normal(size=(5, 4))])


def test_shape_alloc_arange_reshape_join_split():
    x = pt.matrix("x")
    n = pt.lscalar("n")
    s = pt.dscalar("s")
    outs = [
        x.shape,
        pt.alloc(s, n, 3),
        pt.alloc(x[0], 4, x.shape[1]),
        pt.arange(n),
        pt.arange(0.5, n, 0.75),
        pt.arange(n, 0, -2),
        x.reshape((-1,)),
        x.T.reshape((2, -1)),
        pt.concatenate([x, x * 2, x[:1]], axis=0),
        pt.concatenate([x, x], axis=1),
        *pt.split(x, [1, 3], n_splits=2, axis=0),
    ]
    compare([x, n, s], outs, [rng.normal(size=(4, 3)), 5, 1.5])


def test_join_and_reshape_errors():
    x = pt.matrix("x")
    y = pt.matrix("y")
    f = pytensor.function([x, y], pt.concatenate([x, y]), mode=NO_FALLBACK)
    with pytest.raises(Exception, match="same shape"):
        f(np.ones((2, 3)), np.ones((2, 4)))
    g = pytensor.function([x], x.reshape((5, -1)), mode=NO_FALLBACK)
    with pytest.raises(Exception, match="reshape"):
        g(np.ones((2, 3)))


def test_assert():
    from pytensor.raise_op import Assert

    x = pt.vector("x")
    out = Assert("x must be positive")(x, pt.all(x > 0)) * 2
    f = compare([x], out, [np.ones(3)])
    with pytest.raises(Exception, match="x must be positive"):
        f(-np.ones(3))


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("op", [pt.cumsum, pt.cumprod])
def test_cumop(op, axis):
    x = pt.matrix("x")
    compare([x], op(x, axis=axis), [rng.random((4, 5)) + 0.5])
    # the reference returns int64 here although the output type is int32: compare values
    xi = pt.matrix("xi", dtype="int32")
    compare([xi], op(xi, axis=axis), [rng.integers(-3, 4, (4, 5)).astype("int32")], check_dtype=False)


@pytest.mark.parametrize("offset", [0, 1, -2, 7])
def test_extract_diag(offset):
    x = pt.tensor3("x")
    compare([x], pt.diagonal(x, offset=offset, axis1=0, axis2=2) * 1.0, [rng.random((4, 3, 5))])
    m = pt.matrix("m")
    compare([m], pt.diag(m) + 0.0, [rng.random((4, 6))])


@pytest.mark.parametrize("dtype", ["float64", "float32", "int64"])
def test_sort(dtype):
    x = pt.matrix("x", dtype=dtype)
    v = (rng.random((5, 6)) * 10).astype(dtype)
    if dtype != "int64":
        v[1, 2] = np.nan
    compare([x], [pt.sort(x, axis=0), pt.sort(x, axis=1), pt.sort(x.T, axis=-1)], [v])


@pytest.mark.parametrize("lower", [True, False])
def test_cholesky_and_solve_triangular(lower):
    a = pt.matrix("a")
    b = pt.matrix("b")
    bv = pt.vector("bv")
    m = rng.random((5, 5))
    spd = m @ m.T + 5 * np.eye(5)
    L = cholesky(a, lower=lower)
    outs = [L, solve_triangular(L, b, lower=lower), solve_triangular(L, bv, lower=lower, unit_diagonal=True)]
    compare([a, b, bv], outs, [spd, rng.random((5, 3)), rng.random(5)], rtol=1e-8)
    f = pytensor.function([a], cholesky(a, lower=lower), mode=NO_FALLBACK)
    assert np.isnan(f(-np.eye(3))).all()


def test_fused_elemwise_patterns():
    x = pt.vector("x")
    y = pt.vector("y")
    idx = pt.lvector("idx")
    m = pt.matrix("m")
    outs = [
        (pt.exp(x) * y[: x.shape[0]]).sum(),
        pt.exp(x[idx]).sum(),
        pt.inc_subtensor(x[idx], pt.exp(y[: idx.shape[0]])),
        (pt.exp(m) * y[: m.shape[1]]).sum(axis=1),
        (pt.log1p(m) * y[: m.shape[1]]).sum(axis=0),
    ]
    f = compare([x, y, idx, m], outs, [rng.random(10), rng.random(30), rng.integers(0, 10, 30), rng.random((5, 7))])
    assert any(s.startswith("FusedElemwise") for s in tasks(f))


def test_scan_sit_sot_mit_sot_nit_sot():
    x0 = pt.dscalar("x0")
    seq = pt.vector("seq")

    def step(s, xm2, xm1):
        nxt = 0.5 * xm1 + 0.25 * xm2 + s
        return nxt, nxt**2

    init = pt.stack([x0, x0 + 1])
    (traj, sq) = pytensor.scan(step, sequences=[seq], outputs_info=[{"initial": init, "taps": [-2, -1]}, None], return_updates=False)
    acc = pytensor.scan(lambda a, b: a + b, sequences=[seq], outputs_info=[x0], return_updates=False)
    compare([x0, seq], [traj, sq, acc], [0.3, rng.random(12)], rtol=1e-9)


def test_scan_gradient_and_while():
    x = pt.vector("x")
    w = pt.dscalar("w")
    ys = pytensor.scan(lambda xi, acc: acc * w + xi, sequences=[x], outputs_info=[pt.zeros(())], return_updates=False)
    cost = ys[-1] ** 2
    compare([x, w], [cost, pytensor.grad(cost, w)], [rng.random(6), 0.7], rtol=1e-8)

    from pytensor.scan.utils import until

    def stop_at(prev):
        nxt = prev * 2
        return nxt, until(nxt > 50)

    out = pytensor.scan(stop_at, outputs_info=[w], n_steps=20, return_updates=False)
    compare([w], out, [1.5])


# ------------------------------------------------------------------ ScalarLoop


def test_scalar_loop_for():
    from pytensor.scalar import ScalarLoop, float64
    from pytensor.tensor.elemwise import Elemwise

    x0, c = float64("x0"), float64("c")
    op = ScalarLoop(init=[x0], update=[x0 * c + 1.0], constant=[c])
    n, x, cc = pt.lscalar("n"), pt.dvector("x"), pt.dscalar("c")
    f = pytensor.function([n, x, cc], Elemwise(op)(n, x, cc), mode=NO_FALLBACK)
    assert f.maker.linker.stats.n_numba == 0
    for steps in (0, 1, 7):
        expected = np.arange(4.0)
        for _ in range(steps):
            expected = expected * 0.5 + 1.0
        np.testing.assert_allclose(f(steps, np.arange(4.0), 0.5), expected)


def test_scalar_loop_while():
    from pytensor.scalar import ScalarLoop, float64, int64
    from pytensor.tensor.elemwise import Elemwise

    y0, i0 = float64("y0"), int64("i0")
    op = ScalarLoop(init=[y0, i0], update=[y0 * 2, i0 + 1], until=(y0 * 2) > 100.0)
    n, y = pt.iscalar("n"), pt.dvector("y")
    outs = Elemwise(op)(n, y, pt.zeros_like(y, dtype="int64"))
    f = pytensor.function([n, y], outs, mode=NO_FALLBACK)
    ref = pytensor.function([n, y], outs, mode="FAST_COMPILE")
    for steps in (0, 3, 50):
        args = (steps, np.array([1.0, 30.0, 200.0]))
        for r, e in zip(f(*args), ref(*args), strict=True):
            np.testing.assert_array_equal(r, e)


def test_betainc_grad_uses_scalar_loop():
    a, b, z = pt.dvectors("a", "b", "z")
    g = pt.grad(pt.betainc(a, b, z).sum(), [a, b])
    f = pytensor.function([a, b, z], g, mode=NO_FALLBACK)
    ref = pytensor.function([a, b, z], g)
    args = (np.array([0.5, 2.0, 5.0]), np.array([1.5, 3.0, 0.7]), np.array([0.2, 0.6, 0.9]))
    for r, e in zip(f(*args), ref(*args), strict=True):
        np.testing.assert_allclose(r, e, rtol=1e-10)


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8"])
def test_integer_pow(dtype):
    x, e = pt.vector(dtype=dtype), pt.vector(dtype=dtype)
    f = pytensor.function([x, e], x**e, mode=NO_FALLBACK)
    xs = np.array([0, 1, 2, 3, 5, 7, 2, 3], dtype=dtype)
    es = np.array([0, 5, 3, 0, 2, 1, 7, 4], dtype=dtype)
    r = f(xs, es)
    assert r.dtype == (xs**es).dtype
    np.testing.assert_array_equal(r, xs**es)
