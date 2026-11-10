"""PR #2647 workloads: positive gammaln/psi ranges and fused value + gradient."""

import numpy as np
import scipy.special as sc

import pytensor
import pytensor.tensor as pt


def cases(size):
    rng = np.random.default_rng(493)
    for dtype in ("float32", "float64"):
        x = pt.vector("x", dtype=dtype)
        tolerance = (
            dict(rtol=2e-5, atol=2e-6)
            if dtype == "float32"
            else dict(rtol=1e-11, atol=1e-12)
        )
        for name, operation, reference in (
            ("gammaln", pt.gammaln, sc.gammaln),
            ("psi", pt.psi, sc.digamma),
        ):
            function = pytensor.function([x], operation(x), mode="NUMBA")
            for lower, upper in ((0.01, 1), (1, 13), (13, 64), (64, 1024)):
                values = rng.uniform(lower, upper, size).astype(dtype)
                np.testing.assert_allclose(
                    function(values), reference(values), **tolerance
                )
                yield f"{name}/{dtype}/[{lower},{upper})", function, (values,)
        total = pt.gammaln(x).sum()
        function = pytensor.function([x], [total, pt.grad(total, x)], mode="NUMBA")
        values = rng.uniform(0.01, 64, size).astype(dtype)
        value, gradient = function(values)
        np.testing.assert_allclose(
            value, sc.gammaln(values).sum(dtype=value.dtype), **tolerance
        )
        np.testing.assert_allclose(gradient, sc.digamma(values), **tolerance)
        yield f"gammaln_value_gradient/{dtype}/[0.01,64)", function, (values,)
