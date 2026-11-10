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
            expression = operation(x)
            points = {}
            for lower, upper in ((0.01, 1), (1, 13), (13, 64), (64, 1024)):
                values = rng.uniform(lower, upper, size).astype(dtype)
                points[f"[{lower},{upper})"] = (values,)

            def build(x=x, expression=expression):
                return pytensor.function([x], expression, mode="NUMBA")

            def check(result, inputs, reference=reference, tolerance=tolerance):
                np.testing.assert_allclose(result, reference(inputs[0]), **tolerance)

            yield f"{name}/{dtype}", build, points, check
        total = pt.gammaln(x).sum()
        outputs = [total, pt.grad(total, x)]
        values = rng.uniform(0.01, 64, size).astype(dtype)

        def build(x=x, outputs=outputs):
            return pytensor.function([x], outputs, mode="NUMBA")

        def check(result, inputs, tolerance=tolerance):
            value, gradient = result
            np.testing.assert_allclose(
                value, sc.gammaln(inputs[0]).sum(dtype=value.dtype), **tolerance
            )
            np.testing.assert_allclose(gradient, sc.digamma(inputs[0]), **tolerance)

        yield f"gammaln_value_gradient/{dtype}", build, {"[0.01,64)": (values,)}, check
