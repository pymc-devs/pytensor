import pytest

import pytensor.tensor as pt
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import PatternNodeRewriter
from pytensor.graph.rewriting.unify import (
    convert_strs_to_vars,
    match_pattern,
)
from pytensor.tensor import exp, log, mul, sub
from pytensor.tensor.math import erf


def _make_case(case_id):
    x = pt.vector("x")
    if case_id == "shallow_match":
        return x, (log, (exp, "x")), log(exp(x))
    if case_id == "deep_match":
        return x, (log, (exp, (log, (exp, "x")))), log(exp(log(exp(x))))
    if case_id == "repeated_var":
        return x, (mul, "x", "x"), mul(x, x)
    if case_id == "with_constant":
        # The constant must already have the output ndim, otherwise Elemwise wraps it
        # in a DimShuffle and the match only succeeds after constant folding
        return x, (sub, 1.0, (erf, "x")), sub(pt.as_tensor_variable([1.0]), erf(x))
    if case_id == "no_match":
        return x, (log, (exp, "x")), log(x)
    raise ValueError(case_id)


@pytest.mark.parametrize(
    "case_id",
    ["shallow_match", "deep_match", "repeated_var", "with_constant", "no_match"],
)
def test_match_pattern_benchmark(benchmark, case_id):
    _x, pat_tuple, out = _make_case(case_id)
    pat = convert_strs_to_vars(pat_tuple)
    assert (match_pattern(pat, out.owner) is None) == (case_id == "no_match")

    def run():
        return match_pattern(pat, out.owner)

    benchmark(run)


@pytest.mark.parametrize("case_id", ["shallow_match", "deep_match", "no_match"])
def test_pattern_rewriter_transform_benchmark(benchmark, case_id):
    x, pat_tuple, out = _make_case(case_id)
    fg = FunctionGraph([x], [out], clone=False)
    rw = PatternNodeRewriter(pat_tuple, "x", allow_multiple_clients=True)
    node = out.owner

    def run():
        rw.transform(fg, node)

    benchmark(run)
