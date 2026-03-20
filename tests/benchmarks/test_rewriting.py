import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config
from pytensor.graph import FunctionGraph


def _large_fuseable_graph(n):
    from pytensor.tensor.math import add

    sd = pt.dscalar("sd")
    means = pt.dvector("means")

    cst_05 = pt.constant(0.5)
    cst_m05 = pt.constant(-0.5)
    cst_2 = pt.constant(2)
    cst_m2 = pt.constant(-2)
    ones = pt.constant(np.ones(10))

    factors = []
    for i in range(n):
        f = cst_m05 * sd**cst_m2 * (ones - means[i]) ** cst_2 + cst_05 * pt.log(
            cst_05 * (sd**cst_m2) / np.pi
        )
        factors.append(pt.sum(f))

    logp = add(*factors)

    vars = [sd, means]
    dlogp = [pt.grad(logp, v) for v in vars]
    return vars, dlogp


def _deep_small_kernels(n):
    x = pt.matrix("x")
    out = x
    for _ in range(n):
        out = pt.sin(out.T) + pt.cos(out)
    return [x], [out]


@pytest.mark.skipif(not config.cxx, reason="No cxx compiler")
@pytest.mark.parametrize(
    "graph_fn, n, expected_n_repl",
    [
        ("deep_small_kernels", 20, (20, 60)),
        ("large_fuseable_graph", 25, (128, 876)),
    ],
)
def test_fusion_rewrite_benchmark(graph_fn, n, expected_n_repl, benchmark):
    from pytensor.tensor.rewriting.elemwise import FusionOptimizer

    graph_builders = {
        "deep_small_kernels": _deep_small_kernels,
        "large_fuseable_graph": _large_fuseable_graph,
    }
    inps, outs = graph_builders[graph_fn](n)
    fg = FunctionGraph(inps, outs)
    opt = FusionOptimizer()

    def rewrite_func():
        fg_clone = fg.clone()
        _, nb_fused, nb_replacement, *_ = opt.apply(fg_clone)
        return nb_fused, nb_replacement

    assert rewrite_func() == expected_n_repl
    benchmark.pedantic(rewrite_func, rounds=7, iterations=5)
