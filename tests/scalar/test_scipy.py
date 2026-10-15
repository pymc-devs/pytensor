"""Regression tests for SciPy-backed scalar C implementations."""

import numpy as np
import pytest
from scipy import special

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode


@pytest.mark.parametrize("name", ["gammainc", "gammaincc"])
def test_gamma_large_transition(name):
    # https://github.com/pymc-devs/pytensor/issues/2291
    a, x = pt.dvectors("a", "x")
    fn = pytensor.function(
        [a, x], getattr(pt, name)(a, x), mode=Mode(linker="c", optimizer=None)
    )
    x_val = np.array([1e4, 1e5, 3e5, 1e6])
    a_val = x_val + 1
    np.testing.assert_allclose(
        fn(a_val, x_val), getattr(special, name)(a_val, x_val), rtol=2e-14
    )


@pytest.mark.parametrize("name", ["gammainc", "gammaincc"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_gamma_c_dtypes(name, dtype):
    a, x = [pt.vector(dtype=dtype) for _ in range(2)]
    out = getattr(pt, name)(a, x)
    fn = pytensor.function([a, x], out, mode=Mode(linker="c", optimizer=None))
    a_val = np.array([1, 3, 1_000_001], dtype=dtype)
    x_val = np.array([2, 4, 1_000_000], dtype=dtype)
    result = fn(a_val, x_val)
    assert result.dtype == out.dtype
    # The C API works in double precision, with the usual scalar output cast.
    expected = getattr(special, name)(a_val.astype("float64"), x_val.astype("float64"))
    np.testing.assert_array_equal(result, expected.astype(out.dtype))
    np.testing.assert_array_equal(fn(a_val, x_val), result)


@pytest.mark.parametrize("name", ["gammainc", "gammaincc"])
def test_gamma_c_domain(name):
    a, x = pt.dvectors("a", "x")
    fn = pytensor.function(
        [a, x], getattr(pt, name)(a, x), mode=Mode(linker="c", optimizer=None)
    )
    points = np.array(
        [
            (0, 0),
            (0, 1),
            (0, np.inf),
            (1, 0),
            (1, np.inf),
            (np.inf, 1),
            (np.inf, np.inf),
            (-1, 1),
            (1, -1),
            (np.nan, 0),
            (np.nan, 1),
            (1, np.nan),
            (1e-12, 1e-12),
            (1, 1e-300),
            (1, 700),
            (1e6, np.nextafter(1e6, 0)),
            (1e6, 1e6),
            (1e6, np.nextafter(1e6, np.inf)),
        ]
    )
    a_val, x_val = points.T
    expected = getattr(special, name)(a_val, x_val)
    np.testing.assert_array_equal(fn(a_val, x_val), expected)


def test_gamma_independent_references():
    a, x = pt.dvectors("a", "x")
    fn = pytensor.function(
        [a, x],
        [pt.gammainc(a, x), pt.gammaincc(a, x)],
        mode=Mode(linker="c", optimizer=None),
    )
    # mpmath 1.3.0, 70 decimal digits, gammainc(a, 0, x, regularized=True).
    a_val = np.array([10_001, 100_001, 300_001, 1_000_001, *[1_000_000.0] * 6])
    x_val = np.array(
        [
            10_000,
            100_000,
            300_000,
            1_000_000,
            np.nextafter(1e6, 0),
            1e6,
            np.nextafter(1e6, np.inf),
            np.nextafter(1e6 + 1, 0),
            1e6 + 1,
            np.nextafter(1e6 + 1, np.inf),
        ]
    )
    p_ref = np.array(
        [
            0.4973404187809923747340704243177944,
            0.4991589569006598761303657269002465,
            0.4995144231265566597122915289766773,
            0.4997340385137163472146182735163907,
            0.5001329807608261482542297627815013,
            0.5001329807608725912443228175032067,
            0.5001329807609190342344158722195054,
            0.5005319227420211203165026498156089,
            0.5005319227420675632369313103511799,
            0.5005319227421140061573599708759375,
        ]
    )
    p, q = fn(a_val, x_val)
    np.testing.assert_allclose(p, p_ref, rtol=5e-15)
    np.testing.assert_allclose(q, 1 - p_ref, rtol=5e-15)
    np.testing.assert_allclose(p + q, 1, rtol=1e-15)
    # Analytic identities provide an independent small-shape/tail check.
    x_val = np.array([1e-10, 0.5, 1, 30, 700])
    p, q = fn(np.ones_like(x_val), x_val)
    np.testing.assert_allclose(p, -np.expm1(-x_val), rtol=5e-15)
    np.testing.assert_allclose(q, np.exp(-x_val), rtol=5e-14)


def test_gamma_fused_composite():
    from pytensor.scalar import Composite
    from pytensor.tensor.elemwise import Elemwise

    a, x = pt.dvectors("a", "x")
    p, q = pt.gammainc(a, x), pt.gammaincc(a, x)
    fn = pytensor.function(
        [a, x],
        [p + 2 * q, p * q + pt.gammainc(a + 1, x)],
        mode=Mode(linker="c", optimizer="fast_run"),
    )
    nodes = fn.maker.fgraph.toposort()
    assert len(nodes) == 1
    assert isinstance(nodes[0].op, Elemwise)
    assert isinstance(nodes[0].op.scalar_op, Composite)
    a_val = np.array([1, 3, 1_000_001.0])
    x_val = np.array([2, 4, 1_000_000.0])
    p_val, q_val = special.gammainc(a_val, x_val), special.gammaincc(a_val, x_val)
    expected = [p_val + 2 * q_val, p_val * q_val + special.gammainc(a_val + 1, x_val)]
    for _ in range(3):
        for result, reference in zip(fn(a_val, x_val), expected, strict=True):
            np.testing.assert_allclose(result, reference, rtol=2e-14)


def test_gamma_scalar_loop():
    from pytensor.scalar import Composite, float64, int64
    from pytensor.scalar.loop import ScalarLoop
    from pytensor.scalar.math import gammainc, gammaincc

    a, x = float64("a"), float64("x")
    steps = int64("steps")
    # Nested Composite -> ScalarLoop -> Composite exercises recursive init forwarding.
    inner = Composite([a, x], [gammainc(a, x) + 2 * gammaincc(a, x)])
    loop = ScalarLoop(init=[x], constant=[a], update=[inner(a, x)])
    outer = Composite([steps, a, x], [loop(steps, x, a) + gammainc(a, x)])
    fn = pytensor.function(
        [steps, a, x], outer(steps, a, x), mode=Mode(linker="c", optimizer=None)
    )
    for n in (0, 1, 3):
        expected = x_val = 1_000_000.0
        a_val = 1_000_001.0
        for _ in range(n):
            expected = special.gammainc(a_val, expected) + 2 * special.gammaincc(
                a_val, expected
            )
        expected += special.gammainc(a_val, x_val)
        np.testing.assert_allclose(fn(n, a_val, x_val), expected, rtol=2e-14)


def test_gamma_scan():
    from pytensor.scalar import Composite
    from pytensor.scan.op import Scan

    a, x = pt.dvectors("a", "x")
    mode = Mode(linker="cvm", optimizer="fast_run")
    output = pytensor.scan(
        lambda ai, xi, previous: (
            previous
            + pt.gammainc(ai, xi + previous)
            + 2 * pt.gammaincc(ai, xi + previous)
        ),
        sequences=[a, x],
        outputs_info=pt.constant(0.0, dtype="float64"),
        return_updates=False,
    )
    fn = pytensor.function([a, x], output, mode=mode)
    scan_ops = [
        node.op for node in fn.maker.fgraph.toposort() if isinstance(node.op, Scan)
    ]
    assert len(scan_ops) == 1
    inner_fn = scan_ops[0].fn
    inner_nodes = inner_fn.maker.fgraph.toposort()
    assert any(
        isinstance(getattr(node.op, "scalar_op", None), Composite)
        for node in inner_nodes
    )
    for node, thunk in zip(inner_fn.vm.nodes, inner_fn.vm.thunks, strict=True):
        if isinstance(getattr(node.op, "scalar_op", None), Composite):
            assert hasattr(thunk, "cthunk")
    a_val = np.array([1, 3, 1_000_001.0])
    x_val = np.array([2, 4, 1_000_000.0])
    expected = []
    previous = 0.0
    for ai, xi in zip(a_val, x_val, strict=True):
        previous += special.gammainc(ai, xi + previous) + 2 * special.gammaincc(
            ai, xi + previous
        )
        expected.append(previous)
    np.testing.assert_allclose(fn(a_val, x_val), expected, rtol=2e-14)


def test_gamma_openmp():
    from pytensor.scalar import Composite, float64
    from pytensor.scalar.math import gammainc, gammaincc
    from pytensor.tensor.elemwise import Elemwise

    sa, sx = float64(), float64()
    composite = Composite([sa, sx], [gammainc(sa, sx) + 2 * gammaincc(sa, sx)])
    a, x = pt.dvectors("a", "x")
    with pytensor.config.change_flags(openmp_elemwise_minsize=0):
        fn = pytensor.function(
            [a, x],
            Elemwise(composite, openmp=True)(a, x),
            mode=Mode(linker="c", optimizer=None),
        )
    if not fn.maker.fgraph.outputs[0].owner.op.openmp:
        pytest.skip("Compiler does not support OpenMP")
    a_val = np.full(4096, 1_000_001.0)
    x_val = np.linspace(999_000, 1_001_000, 4096)
    expected = special.gammainc(a_val, x_val) + 2 * special.gammaincc(a_val, x_val)
    for _ in range(3):
        np.testing.assert_allclose(fn(a_val, x_val), expected, rtol=2e-14)


def test_gamma_cross_process_cache(tmp_path):
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import sys
        import numpy as np
        import pytensor
        import pytensor.tensor as pt
        from pytensor.compile.mode import Mode
        from pytensor.link.c.cmodule import GCC_compiler
        if sys.argv[1] == "reload":
            def no_compile(*args, **kwargs):
                raise AssertionError("Expected the compiled module from the first process")
            GCC_compiler.compile_str = no_compile
        a, x = pt.dvectors("a", "x")
        f = pytensor.function([a, x], pt.gammainc(a, x) + 2 * pt.gammaincc(a, x),
            mode=Mode(linker="c", optimizer="fast_run"))
        np.testing.assert_allclose(f([1000001.], [1000000.]),
            [1.5002659614862837], rtol=2e-14)
    """)
    env = dict(os.environ, PYTENSOR_FLAGS=f"compiledir={tmp_path},mode=FAST_COMPILE")
    for phase in ("compile", "reload"):
        subprocess.run(
            [sys.executable, "-c", script, phase],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    sources = [path.read_text() for path in tmp_path.rglob("mod.cpp")]
    bridge_sources = [
        source for source in sources if "pytensor_scipy_gammainc" in source
    ]
    assert len(bridge_sources) == 1
    assert 'PyImport_ImportModule("scipy.special.cython_special")' in bridge_sources[0]


@pytest.mark.parametrize("failure", ["missing", "not_capsule", "wrong_signature"])
def test_gamma_bad_c_api(tmp_path, failure):
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import sys
        import scipy.special.cython_special as cs
        import pytensor
        import pytensor.tensor as pt
        from pytensor.compile.mode import Mode
        failure = sys.argv[1]
        if failure == "missing":
            del cs.__pyx_capi__["gammainc"]
        elif failure == "not_capsule":
            cs.__pyx_capi__["gammainc"] = object()
        else:
            cs.__pyx_capi__["gammainc"] = cs.__pyx_capi__["gammaln"]
        a, x = pt.dscalars("a", "x")
        try:
            pytensor.function([a, x], pt.gammainc(a, x), mode=Mode(linker="c", optimizer=None))
        except ImportError as error:
            assert "scipy.special.cython_special" in str(error)
            assert "gammainc" in str(error)
        else:
            raise AssertionError("Invalid C API must fail during module initialization")
    """)
    env = dict(os.environ, PYTENSOR_FLAGS=f"compiledir={tmp_path},mode=FAST_COMPILE")
    subprocess.run(
        [sys.executable, "-c", script, failure],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "a_dtype,x_dtype",
    [("float32", "int32"), ("int32", "float64"), ("float64", "float32")],
)
def test_gamma_mixed_dtypes_and_strides(a_dtype, x_dtype):
    a, x = pt.vector(dtype=a_dtype), pt.matrix(dtype=x_dtype)
    out = pt.gammainc(a[:, None], x) + 2 * pt.gammaincc(a[:, None], x)
    fn = pytensor.function([a, x], out, mode=Mode(linker="c", optimizer="fast_run"))
    a_val = np.array([1, 3, 1_000_001], dtype=a_dtype)
    x_val = np.array(
        [[1, 2, 3, 4], [2, 3, 4, 5], [999_000, 1_000_000, 1_001_000, 1_002_000]],
        dtype=x_dtype,
    )[:, ::2]
    assert not x_val.flags.c_contiguous
    expected = special.gammainc(a_val[:, None], x_val) + 2 * special.gammaincc(
        a_val[:, None], x_val
    )
    result = fn(a_val, x_val)
    assert result.dtype == out.dtype
    np.testing.assert_allclose(
        result, expected, rtol=2e-7 if out.dtype == "float32" else 2e-14
    )


def test_scipy_unary_and_duplicate_support():
    from pytensor.scalar import Composite, UnaryScalarOp, float64, upgrade_to_float
    from pytensor.scalar.math import gammainc, gammaincc
    from pytensor.scalar.scipy import SciPyScalarOp
    from pytensor.tensor.elemwise import Elemwise

    class SciPyGammaLn(SciPyScalarOp, UnaryScalarOp):
        nfunc_spec = ("scipy.special.gammaln", 1, 1)

        def c_code_cache_version(self):
            # This class is local to the test and cannot be pickled in the cache.
            return ()

    loggamma = SciPyGammaLn(upgrade_to_float)
    sa, sx = float64(), float64()
    composite = Composite(
        [sa, sx], [gammainc(sa, sx) + gammaincc(sa, sx) + loggamma(sa)]
    )
    a, x = pt.dvectors("a", "x")
    fn = pytensor.function(
        [a, x],
        [pt.gammainc(a, x), Elemwise(composite)(a, x)],
        mode=Mode(linker="c", optimizer=None),
    )
    a_val, x_val = np.array([1.0, 3.0, 1_000_001]), np.array([2.0, 4.0, 1_000_000])
    p, combined = fn(a_val, x_val)
    np.testing.assert_array_equal(p, special.gammainc(a_val, x_val))
    np.testing.assert_allclose(combined, 1 + special.gammaln(a_val), rtol=2e-14)


@pytest.mark.parametrize("name", ["gammainc", "gammaincc"])
def test_gamma_internal_float16(name):
    from pytensor import scalar

    a, x = scalar.float64(), scalar.float64()
    output = scalar.cast(
        getattr(scalar, name)(scalar.cast(a, "float16"), scalar.cast(x, "float16")),
        "float64",
    )
    with pytest.raises(NotImplementedError, match="Float16 C arithmetic"):
        pytensor.function([a, x], output, mode=Mode(linker="c", optimizer=None))
    fn = pytensor.function([a, x], output, mode=Mode(linker="cvm", optimizer=None))
    expected = np.float16(getattr(special, name)(np.float16(1), np.float16(2)))
    np.testing.assert_array_equal(fn(1.0, 2.0), expected)
