import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.gradient import pullback
from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op
from research.vjp_sparsity.prototype import jacobian_sparsity_vjp
from research.vjp_sparsity.rewriting import (
    infer_nonzero,
    may_be_nonzero,
    nonzero_lift_db,
    register_nonzero,
)
from tests.sparsity import test_core as pr_core
from tests.sparsity.conftest import detected_and_numeric_patterns
from tests.sparsity.test_elemwise import CASES, CONSERVATIVE_NNZ, matrix, row


@pytest.fixture(scope="module")
def patterns():
    rng = np.random.default_rng(0)
    values = [rng.uniform(0.5, 1.5, (4, 3)), rng.uniform(0.5, 1.5, (1, 3))]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pt, "jacobian_sparsity", jacobian_sparsity_vjp)
        return detected_and_numeric_patterns(CASES, [matrix, row], values)


@pytest.mark.parametrize("title", CASES)
def test_pr_elemwise(patterns, title):
    detected, numeric = patterns[title]
    assert not (numeric & ~detected).any()
    if title in CONSERVATIVE_NNZ:
        assert detected.sum() == CONSERVATIVE_NNZ[title]
    else:
        np.testing.assert_array_equal(detected, numeric)


@pytest.mark.parametrize(
    "test",
    [
        pr_core.test_fallback_is_input_level_reachability,
        pr_core.test_intermediate_and_empty_wrt,
        pr_core.test_wrt_ancestor_repeated_and_folded,
        pr_core.test_sparse_constant_in_graph,
    ],
)
def test_pr_core(test, monkeypatch):
    monkeypatch.setattr(pt, "jacobian_sparsity", jacobian_sparsity_vjp)
    test()


def test_cancellation():
    x = pt.vector("x")
    pattern = jacobian_sparsity_vjp(x - x, x)
    f = pytensor.function([x], pattern, mode="FAST_COMPILE")
    np.testing.assert_array_equal(f(np.ones(3)), np.zeros((3, 3), dtype=bool))


@pytest.mark.parametrize("mode", ["FAST_COMPILE", "CVM", "NUMBA"])
def test_folded_intermediate(mode):
    intermediate = pt.exp(pt.constant(1.0))
    pattern = jacobian_sparsity_vjp(2 * intermediate, intermediate)
    f = pytensor.function([], pattern, mode=mode)
    np.testing.assert_array_equal(f(), [[True]])


def test_repeated_outputs():
    x = pt.vector("x")
    y = 2 * x
    pattern = jacobian_sparsity_vjp([y, y], x)
    f = pytensor.function([x], pattern, mode="FAST_COMPILE")
    expected = np.vstack([np.eye(3), np.eye(3)]).astype(bool)
    np.testing.assert_array_equal(f(np.ones(3)), expected)


def test_existing_zero_gradient_rule():
    from pytensor.gradient import zero_grad

    x = pt.vector("x")
    pattern = jacobian_sparsity_vjp(zero_grad(x), x)
    f = pytensor.function([x], pattern, mode="FAST_COMPILE")
    np.testing.assert_array_equal(f(np.ones(3)), np.zeros((3, 3), dtype=bool))


def test_pr_patterns_are_value_independent():
    patterns = [jacobian_sparsity_vjp(outputs, wrt) for outputs, wrt in CASES.values()]
    f = pytensor.function(
        [matrix, row], patterns, mode="FAST_COMPILE", on_unused_input="ignore"
    )
    positive = f(np.ones((2, 3)), np.ones((1, 3)))
    zero_and_negative = f(
        np.array([[0.0, -2.0, 3.0], [-4.0, 0.0, -6.0]]), -np.ones((1, 3))
    )
    for first, second in zip(positive, zero_and_negative, strict=True):
        np.testing.assert_array_equal(first, second)


def test_infer_from_regular_vjp():
    x = pt.vector("x", shape=(3,))
    cotangent = x.type("cotangent")
    vjp = pullback(pt.sin(x * [0.0, 2.0, 0.0]), x, cotangent)
    mask = infer_nonzero(vjp, cotangent)
    f = pytensor.function([x, cotangent], [mask, vjp], mode="FAST_COMPILE")
    first, first_vjp = f(np.ones(3), [1.0, 0.0, 1.0])
    second, second_vjp = f(np.zeros(3), [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(first, [False, False, False])
    np.testing.assert_array_equal(second, [False, True, False])
    np.testing.assert_array_equal(first_vjp, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(second_vjp, [0.0, 2.0, 0.0])


def test_root_resolution_depends_on_requested_directions():
    left = pt.vector("left")
    right = pt.vector("right")
    vjp = left * right
    function = pytensor.function(
        [left, right],
        [infer_nonzero(vjp, right), infer_nonzero(vjp, left)],
    )
    first, second = function(np.zeros(3), [1.0, 0.0, 1.0])
    np.testing.assert_array_equal(first, [True, False, True])
    np.testing.assert_array_equal(second, [False, False, False])
    first, second = function(np.ones(3), np.zeros(3))
    np.testing.assert_array_equal(first, [False, False, False])
    np.testing.assert_array_equal(second, [True, True, True])


def test_structured_direction():
    index = pt.iscalar("index")
    amplitude = pt.scalar("amplitude")
    seed = pt.zeros(4)[index].set(amplitude)
    function = pytensor.function(
        [index, amplitude],
        [infer_nonzero(seed, []), infer_nonzero(seed, amplitude)],
    )
    structural, selected = function(1, 0.0)
    np.testing.assert_array_equal(structural, [False, True, False, False])
    np.testing.assert_array_equal(selected, [False, False, False, False])
    structural, selected = function(3, 2.0)
    np.testing.assert_array_equal(structural, [False, False, False, True])
    np.testing.assert_array_equal(selected, structural)


class CotangentScale(Op):
    def make_node(self, cotangent):
        return Apply(self, [cotangent], [cotangent.type()])

    def perform(self, node, inputs, outputs):
        outputs[0][0] = inputs[0] * 2

    def infer_shape(self, node, shapes):
        return shapes


def test_register_nonzero_rewrite():
    cotangent = pt.vector("cotangent")
    vjp = CotangentScale()(cotangent)
    pessimistic = infer_nonzero(vjp, cotangent)

    @register_nonzero([CotangentScale])
    def local_nonzero_cotangent_scale(fgraph, node):
        return may_be_nonzero(node.inputs[0])

    try:
        precise = infer_nonzero(vjp, cotangent)
    finally:
        nonzero_lift_db.remove_tags(local_nonzero_cotangent_scale.__name__, "basic")
    f = pytensor.function([cotangent], [pessimistic, precise], mode="FAST_COMPILE")
    before, after = f([0.0, 1.0, 0.0])
    np.testing.assert_array_equal(before, [True, True, True])
    np.testing.assert_array_equal(after, [False, True, False])


def test_runtime_nonzero_predicate_is_not_an_analysis_query():
    condition_input = pt.vector("condition_input")
    x = pt.vector("x")
    y = pt.switch(pt.neq(condition_input, 0), condition_input, x)
    pattern = jacobian_sparsity_vjp(y, x)
    f = pytensor.function(
        [condition_input, x], pattern, mode="FAST_COMPILE", on_unused_input="ignore"
    )
    for values in (np.ones(3), np.zeros(3)):
        np.testing.assert_array_equal(f(values, np.ones(3)), np.eye(3, dtype=bool))


class UnknownShape(Op):
    def make_node(self, value):
        return Apply(self, [value], [value.type()])

    def perform(self, node, inputs, outputs):
        outputs[0][0] = inputs[0].copy()

    def pullback(self, inputs, outputs, output_grads):
        return output_grads


def test_inference_preserves_original_graph():
    x = pt.vector("x")
    intermediate = UnknownShape()(x + 0.0)
    output = intermediate * intermediate
    original = FunctionGraph(outputs=[output], clone=False)
    original_inputs = {node: tuple(node.inputs) for node in original.toposort()}

    pattern = jacobian_sparsity_vjp(output, intermediate)

    for node, inputs in original_inputs.items():
        assert tuple(node.inputs) == inputs
    original.validate()
    f = pytensor.function([x], pattern, mode="FAST_COMPILE")
    np.testing.assert_array_equal(f(np.ones(3)), np.eye(3, dtype=bool))
