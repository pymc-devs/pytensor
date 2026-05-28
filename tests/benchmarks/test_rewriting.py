import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
import pytensor.xtensor as px
from pytensor import config
from pytensor.graph import FunctionGraph
from pytensor.graph.rewriting import rewrite_graph
from pytensor.xtensor.shape import stack as xstack


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
        ("large_fuseable_graph", 25, (55, 901)),
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


def _xtensor_attention_graph(n_layers):
    B, T, E, H, HD = 4, 32, 64, 4, 16
    rng = np.random.default_rng(0)

    def attn(x):
        Wqkv = px.as_xtensor(
            pytensor.shared(rng.normal(size=(E, 3, H, HD))),
            dims=("embd", "qkv", "head", "hd"),
        )
        Wproj = px.as_xtensor(
            pytensor.shared(rng.normal(size=(E, E))),
            dims=("embd", "embd_out"),
        )
        qkv = px.dot(x, Wqkv, dim="embd")
        q = qkv.isel(qkv=0).rename(time="time_q")
        k = qkv.isel(qkv=1).rename(time="time_k")
        v = qkv.isel(qkv=2).rename(time="time_k")
        s = px.dot(q, k, dim="hd") / np.sqrt(HD)
        mask = px.as_xtensor(
            pt.tril(pt.ones((T, T), dtype="bool")),
            dims=("time_q", "time_k"),
        )
        a = px.math.softmax(px.where(mask, s, np.float64(-1e9)), dim="time_k")
        o = xstack(px.dot(a, v, dim="time_k"), embd=("head", "hd"))
        return px.dot(o, Wproj, dim="embd").rename(time_q="time", embd_out="embd")

    x_t = pt.tensor("x", shape=(B, T, E))
    x = px.as_xtensor(x_t, dims=("batch", "time", "embd"))
    for _ in range(n_layers):
        x = attn(x)
    return x_t, x.values.sum()


@pytest.mark.parametrize("n_layers", [2, 3, 4])
def test_xtensor_attention_rewrite_benchmark(n_layers, benchmark):
    x_t, loss = _xtensor_attention_graph(n_layers)

    def rewrite_once():
        lowered = rewrite_graph(loss, include=("lower_xtensor",), clone=True)
        grad = pt.grad(lowered, x_t)
        return rewrite_graph(
            [lowered, grad], include=("fast_run",), exclude=("inplace",), clone=True
        )

    benchmark(rewrite_once)
