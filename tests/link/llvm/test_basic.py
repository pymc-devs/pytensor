import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode

from pytensor.link.llvm.linker import LLVMLinker


pytest.importorskip("pytensor_llvm")

LLVM = Mode(linker=LLVMLinker(), optimizer="fast_run")
REF = Mode(linker="py", optimizer="fast_run")


def llvm_mode(**kwargs):
    return Mode(linker=LLVMLinker(**kwargs), optimizer="fast_run")


def compiled(f):
    return f.maker.linker.compiled
rng = np.random.default_rng(42)


def compare(inputs, outputs, values, rtol=1e-10, atol=1e-12, mode=LLVM):
    f = pytensor.function(inputs, outputs, mode=mode)
    g = pytensor.function(inputs, outputs, mode=REF)
    res = f(*values)
    ref = g(*values)
    if not isinstance(res, list):
        res, ref = [res], [ref]
    for r, e in zip(res, ref, strict=True):
        r, e = np.asarray(r), np.asarray(e)
        assert r.dtype == e.dtype, (r.dtype, e.dtype)
        assert r.shape == e.shape, (r.shape, e.shape)
        np.testing.assert_allclose(r, e, rtol=rtol, atol=atol, equal_nan=True)
    return f


UNARY = [
    pt.exp, pt.log, pt.log1p, pt.expm1, pt.sqrt, pt.sin, pt.cos, pt.tan, pt.tanh, pt.sinh,
    pt.cosh, pt.arctan, pt.arcsinh, pt.abs, pt.neg, pt.sqr, pt.sign, pt.floor, pt.ceil,
    pt.round, pt.reciprocal, pt.sigmoid, pt.softplus, pt.erf, pt.erfc, pt.gammaln, pt.psi,
    pt.exp2, pt.log2, pt.log10,
]


@pytest.mark.parametrize("fn", UNARY, ids=lambda f: getattr(f, "__name__", str(f)))
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_unary(fn, dtype):
    x = pt.vector("x", dtype=dtype)
    v = (rng.uniform(0.1, 3.0, 257)).astype(dtype)
    tol = dict(rtol=1e-12, atol=1e-14) if dtype == "float64" else dict(rtol=2e-6, atol=1e-6)
    if fn in (pt.gammaln, pt.psi, pt.erfc, pt.erf, pt.tan):
        tol = dict(rtol=1e-9, atol=1e-12) if dtype == "float64" else dict(rtol=1e-5, atol=1e-5)
    # vectorized libmvec / fastmath may differ by a few ulp
    tol["rtol"] = max(tol["rtol"], 1e-14 * 64) if dtype == "float64" else tol["rtol"]
    compare([x], fn(x), [v], **tol)


def test_softplus_log1mexp_ranges():
    x = pt.vector("x")
    v = np.linspace(-50, 50, 1001)
    compare([x], [pt.softplus(x), pt.log1mexp(-pt.abs(x) - 1e-3)], [v], rtol=1e-12)


@pytest.mark.parametrize("dtype", ["float64", "int64", "int32", "int8", "uint8"])
def test_binary_and_compare(dtype):
    x = pt.vector("x", dtype=dtype)
    y = pt.vector("y", dtype=dtype)
    if dtype.startswith("float"):
        a, b = rng.normal(size=100), rng.normal(size=100)
    else:
        info = np.iinfo(dtype)
        lo, hi = max(info.min, -50), min(info.max, 50)
        a = rng.integers(lo, hi, 100).astype(dtype)
        b = rng.integers(max(lo, 1), hi, 100).astype(dtype)
        if dtype == "uint8":  # keep x - y non-negative for the reference backend
            a = rng.integers(25, 50, 100).astype(dtype)
            b = rng.integers(1, 25, 100).astype(dtype)
    outs = [x + y, x - y, x * y, pt.maximum(x, y), pt.minimum(x, y), x < y, x >= y, pt.eq(x, y),
            pt.switch(x > y, x, y), x // y]
    if dtype.startswith("float"):
        outs += [x / y, x ** 2, pt.arctan2(x, y), x % y]
    compare([x, y], outs, [a, b])


def test_mixed_dtypes_and_casts():
    x = pt.vector("x", dtype="int32")
    y = pt.vector("y", dtype="float32")
    z = pt.vector("z", dtype="float64")
    a = rng.integers(-5, 5, 50).astype("int32")
    b = rng.normal(size=50).astype("float32")
    c = rng.normal(size=50)
    compare([x, y, z], [x + y, y * z, pt.cast(z, "int64") + x, (x > 0) & (y > 0), pt.exp(x)], [a, b, c], rtol=1e-6)


def test_broadcasting_and_strides():
    x = pt.matrix("x")
    r = pt.row("r")
    c = pt.col("c")
    s = pt.scalar("s")
    out = pt.exp(x) * r + c * s - pt.log1p(x ** 2)
    xv = rng.normal(size=(7, 9))
    rv = rng.normal(size=(1, 9))
    cv = rng.normal(size=(7, 1))
    sv = np.array(1.5)
    f = compare([x, r, c, s], out, [xv, rv, cv, sv])
    # non-contiguous inputs take the strided path
    xt = rng.normal(size=(9, 7)).T
    np.testing.assert_allclose(f(xt, rv, cv, sv), np.exp(xt) * rv + cv * 1.5 - np.log1p(xt**2))
    xs = rng.normal(size=(14, 18))[::2, ::2]
    np.testing.assert_allclose(f(xs, rv, cv, sv), np.exp(xs) * rv + cv * 1.5 - np.log1p(xs**2))


def test_transpose_view_of_intermediate():
    x = pt.matrix("x")
    y = pt.matrix("y")
    t = pt.exp(x)
    out = t.T * y + t.T.sum()
    compare([x, y], out, [rng.normal(size=(4, 6)), rng.normal(size=(6, 4))])


@pytest.mark.parametrize("axis", [None, 0, 1, 2, (0, 2), (1, 2), (0, 1)])
@pytest.mark.parametrize("op", ["sum", "prod", "max", "min"])
def test_reductions(axis, op):
    x = pt.tensor3("x")
    v = rng.uniform(0.5, 1.5, size=(5, 6, 7))
    out = getattr(x, op)(axis=axis)
    compare([x], out, [v], rtol=1e-12)
    # transposed input
    compare([x], out, [np.transpose(rng.uniform(0.5, 1.5, size=(7, 6, 5)))], rtol=1e-12)


def test_any_all_and_int_sum():
    x = pt.matrix("x")
    v = rng.normal(size=(5, 8))
    compare([x], [pt.any(x > 1, axis=1), pt.all(x > -2, axis=0), (x > 0).sum(), (x > 0).sum(axis=1)], [v])


def test_dot_variants():
    A = pt.matrix("A")
    B = pt.matrix("B")
    v = pt.vector("v")
    w = pt.vector("w")
    av, bv = rng.normal(size=(30, 20)), rng.normal(size=(20, 10))
    vv, wv = rng.normal(size=20), rng.normal(size=30)
    compare([A, B, v, w], [A @ B, A @ v, w @ A, v @ v, pt.exp(A @ B).sum()], [av, bv, vv, wv], rtol=1e-11)
    # transposed operand
    compare([A, B], A.T @ B, [rng.normal(size=(20, 30)), rng.normal(size=(20, 10))], rtol=1e-11)


def test_sort():
    x = pt.matrix("x")
    v = rng.normal(size=(6, 9))
    v[2, 3] = np.nan
    compare([x], [pt.sort(x, axis=1), pt.sort(x, axis=0), pt.sort(pt.exp(x), axis=-1)], [v])


def test_constants_and_outputs_that_are_inputs():
    x = pt.vector("x")
    c = pt.as_tensor_variable(np.arange(5.0))
    compare([x], [x, x * c + 2.0, c], [rng.normal(size=5)])


def test_inplace_planning_reuses_buffers():
    x = pt.vector("x")
    y = pt.vector("y")
    # a long chain that the fusion rewrite won't merge (reductions in between)
    a = pt.exp(x) + y
    b = a * a.sum()
    c = pt.log1p(b**2) * b.max()
    d = c - c.mean()
    f = compare([x, y], d, [rng.normal(size=100), rng.normal(size=100)])
    info = compiled(f).info()
    n_steps = len(info["schedule"])
    assert info["n_inplace"] >= 1
    # 12 kernels, 11 intermediate buffers, served by far fewer reusable slots
    assert info["n_slots"] <= n_steps // 2


def test_zero_size_and_0d():
    x = pt.matrix("x")
    s = pt.scalar("s")
    compare([x, s], [pt.exp(x) * s, x.sum(axis=0), (s * 2).sum()], [np.zeros((0, 3)), np.array(2.0)])


def test_shape_mismatch_raises():
    x = pt.vector("x")
    y = pt.vector("y")
    f = pytensor.function([x, y], x + y, mode=LLVM)
    with pytest.raises(Exception, match="mismatch"):
        f(np.zeros(3), np.zeros(4))


def test_no_fastmath_bitwise_reduction():
    x = pt.vector("x")
    v = rng.normal(size=1001)
    f = pytensor.function([x], x.sum(), mode=llvm_mode(fastmath=False))
    acc = 0.0
    for e in v:
        acc += e
    assert f(v) == acc


def test_logp_and_grad():
    mu = pt.scalar("mu")
    log_sigma = pt.scalar("log_sigma")
    data = pt.vector("data")
    sigma = pt.exp(log_sigma)
    logp = (-0.5 * ((data - mu) / sigma) ** 2 - pt.log(sigma) - 0.5 * np.log(2 * np.pi)).sum()
    logp += -0.5 * mu**2 + log_sigma
    grads = pytensor.grad(logp, [mu, log_sigma])
    compare([mu, log_sigma, data], [logp, *grads], [np.array(0.3), np.array(-0.2), rng.normal(size=1000)], rtol=1e-11)


@pytest.mark.parametrize("dtype", ["int32", "int64", "uint8"])
def test_integer_mod_and_floordiv_python_semantics(dtype):
    a, b = pt.vector("a", dtype=dtype), pt.vector("b", dtype=dtype)
    signed = not dtype.startswith("u")
    av = np.array([7, -7, 7, -7, 0, 5, 3] if signed else [7, 9, 0, 5, 3, 1, 8], dtype=dtype)
    bv = np.array([3, 3, -3, -3, 4, 0, 1] if signed else [3, 4, 4, 0, 1, 2, 0], dtype=dtype)
    f = pytensor.function([a, b], [a % b, a // b], mode=llvm_mode())
    with np.errstate(divide="ignore"):
        exp = [av % bv, av // bv]
    for r, x in zip(f(av, bv), exp, strict=True):
        np.testing.assert_array_equal(r, x)
