"""Regression tests for the SciPy Cython scalar bridge."""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from scipy import special

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode
from pytensor.scalar import Composite, float64, int64, upgrade_to_float
from pytensor.scalar.cython_special import SciPyScalarOp
from pytensor.scalar.loop import ScalarLoop
from pytensor.scalar.math import gammainc, gammaincc
from pytensor.tensor.elemwise import Elemwise


pytestmark = pytest.mark.skipif(
    not pytensor.config.cxx, reason="C++ compiler unavailable"
)


class SciPyTestOp(SciPyScalarOp):
    __props__ = ("nfunc_spec",)

    def __init__(self, name, nin):
        self.nfunc_spec = (f"scipy.special.{name}", nin, 1)
        self.nin = nin
        super().__init__(upgrade_to_float, name=name)


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
    for result, reference in zip(fn(a_val, x_val), expected, strict=True):
        np.testing.assert_allclose(result, reference, rtol=2e-14)


def test_gamma_scalar_loop():
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


def test_gamma_openmp():
    sa, sx = float64(), float64()
    composite = Composite([sa, sx], [gammainc(sa, sx) + 2 * gammaincc(sa, sx)])
    a, x = pt.dvectors("a", "x")
    with pytensor.config.change_flags(openmp_elemwise_minsize=0):
        fn = pytensor.function(
            [a, x],
            Elemwise(composite, openmp=True)(a, x),
            mode=Mode(linker="c", optimizer=None),
        )
    if not fn.maker.fgraph.outputs[0].owner_op.openmp:
        pytest.skip("Compiler does not support OpenMP")
    a_val = np.full(4096, 1_000_001.0)
    x_val = np.linspace(999_000, 1_001_000, 4096)
    expected = special.gammainc(a_val, x_val) + 2 * special.gammaincc(a_val, x_val)
    np.testing.assert_allclose(fn(a_val, x_val), expected, rtol=2e-14)


def test_gamma_cross_process_cache(tmp_path):
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
    bridge_sources = [source for source in sources if "pytensor_scipy_" in source]
    assert len(bridge_sources) == 1
    assert 'PyImport_ImportModule("scipy.special.cython_special")' in bridge_sources[0]


@pytest.mark.parametrize("failure", ["missing", "wrong_signature"])
def test_gamma_bad_c_api(tmp_path, failure):
    script = textwrap.dedent("""
        import sys
        from pathlib import Path
        import scipy.special.cython_special as cs
        import pytensor
        import pytensor.tensor as pt
        from pytensor.compile.mode import Mode
        from pytensor.link.c import cmodule
        failure = sys.argv[1]
        dlimport = cmodule.dlimport
        def corrupt_c_api(path, *args, **kwargs):
            source = Path(path).with_name("mod.cpp")
            if source.exists() and "pytensor_scipy_" in source.read_text():
                if failure == "missing":
                    del cs.__pyx_capi__["gammainc"]
                else:
                    cs.__pyx_capi__["gammainc"] = cs.__pyx_capi__["gammaln"]
            return dlimport(path, *args, **kwargs)
        cmodule.dlimport = corrupt_c_api
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


def test_scipy_unary_and_duplicate_support():
    loggamma = SciPyTestOp("gammaln", 1)
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


@pytest.mark.parametrize(
    "name, values",
    [
        ("betainc", [[0.5, 2.0, 5.0], [1.5, 3.0, 4.0], [0.1, 0.5, 0.9]]),
        ("psi", [[0.5, 1.0, 3.5]]),
        ("iv", [[0.0, 0.5, 2.0], [0.25, 1.0, 4.0]]),
        ("kv", [[0.0, 0.5, 2.0], [0.25, 1.0, 4.0]]),
    ],
)
def test_fused_real_functions(name, values):
    inputs = [pt.dvector() for _ in values]
    op = SciPyTestOp(name, len(inputs))
    fn = pytensor.function(
        inputs, Elemwise(op)(*inputs), mode=Mode(linker="c", optimizer=None)
    )
    np.testing.assert_allclose(fn(*values), getattr(special, name)(*values), rtol=2e-14)


def test_gamma_inverse():
    a, p = pt.dvectors("a", "p")
    fn = pytensor.function(
        [a, p],
        [pt.gammaincinv(a, p), pt.gammainccinv(a, p)],
        mode=Mode(linker="c", optimizer=None),
    )
    p_val = np.array([0.1, 0.5, 0.9])
    lower, upper = fn(np.ones_like(p_val), p_val)
    np.testing.assert_allclose(lower, -np.log1p(-p_val), rtol=2e-14)
    np.testing.assert_allclose(upper, -np.log(p_val), rtol=2e-14)
