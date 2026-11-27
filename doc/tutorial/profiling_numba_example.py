import numpy as np

import pytensor
import pytensor.tensor as pt


x = pt.dvector("x")
y = pt.gammaln(pt.exp(x))
f = pytensor.function([x], [y, pt.grad(y.sum(), x)], mode="NUMBA", profile=True)
values = np.random.default_rng(42).normal(size=64_000)
f(values)  # Compile and warm up before sampling.

with f.profile.sample():
    for i in range(2_000):
        f(values)

f.profile.summary(n_apply_to_print=5)
pytensor.dprint(f)
