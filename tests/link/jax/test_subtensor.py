import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor import subtensor as pt_subtensor
from pytensor.tensor import tensor
from pytensor.tensor.rewriting.jax import (
    boolean_indexing_set_or_inc,
    boolean_indexing_sum,
)
from tests.link.jax.test_basic import compare_jax_and_py


def test_jax_Subtensor_constant():
    shape = (3, 4, 5)
    x_pt = tensor("x", shape=shape, dtype="int")
    x_np = np.arange(np.prod(shape)).reshape(shape)

    # Basic indices
    out_pt = x_pt[1, 2, 0]
    assert isinstance(out_pt.owner.op, pt_subtensor.Subtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_jax_and_py(out_fg, [x_np])

    out_pt = x_pt[1:, 1, :]
    assert isinstance(out_pt.owner.op, pt_subtensor.Subtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_jax_and_py(out_fg, [x_np])

    out_pt = x_pt[:2, 1, :]
    assert isinstance(out_pt.owner.op, pt_subtensor.Subtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_jax_and_py(out_fg, [x_np])

    out_pt = x_pt[1:2, 1, :]
    assert isinstance(out_pt.owner.op, pt_subtensor.Subtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_jax_and_py(out_fg, [x_np])

    # Advanced indexing
    out_pt = pt_subtensor.advanced_subtensor1(x_pt, [1, 2])
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedSubtensor1)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_jax_and_py(out_fg, [x_np])

    out_pt = x_pt[[1, 2], [2, 3]]
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedSubtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_jax_and_py(out_fg, [x_np])

    # Advanced and basic indexing
    out_pt = x_pt[[1, 2], :]
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedSubtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_jax_and_py(out_fg, [x_np])

    out_pt = x_pt[[1, 2], :, [3, 4]]
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedSubtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_jax_and_py(out_fg, [x_np])

    # Flipping
    out_pt = x_pt[::-1]
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_jax_and_py(out_fg, [x_np])

    # Boolean indexing should work if indexes are constant
    out_pt = x_pt[np.random.binomial(1, 0.5, size=(3, 4, 5)).astype(bool)]
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_jax_and_py(out_fg, [x_np])


@pytest.mark.xfail(reason="`a` should be specified as static when JIT-compiling")
def test_jax_Subtensor_dynamic():
    a = pt.iscalar("a")
    x = pt.arange(3)
    out_pt = x[:a]
    assert isinstance(out_pt.owner.op, pt_subtensor.Subtensor)
    out_fg = FunctionGraph([a], [out_pt])
    compare_jax_and_py(out_fg, [1])


def test_jax_Subtensor_dynamic_boolean_mask():
    """JAX does not support resizing arrays with  dynamic boolean masks."""
    from jax.errors import NonConcreteBooleanIndexError

    x_pt = pt.vector("x", dtype="float64")
    out_pt = x_pt[x_pt < 0]
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedSubtensor)

    out_fg = FunctionGraph([x_pt], [out_pt])

    x_pt_test = np.arange(-5, 5)
    with pytest.raises(NonConcreteBooleanIndexError):
        compare_jax_and_py(out_fg, [x_pt_test])


def test_jax_Subtensor_boolean_mask_reexpressible():
    """Summing values with boolean indexing.

    This test ensures that the sum of an `AdvancedSubtensor` `Op`s with boolean
    indexing is replaced with the sum of an equivalent `Switch` `Op`, using the
    `jax_boolean_indexing_sum` rewrite.

    JAX forces users to re-express this logic manually, so this is an
    improvement over its user interface.

    """
    x_pt = pt.matrix("x")
    out_pt = x_pt[x_pt < 0].sum()
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_jax_and_py(out_fg, [np.arange(25).reshape(5, 5).astype(config.floatX)])


def test_boolean_indexing_sum_not_applicable():
    """Test that boolean_indexing_sum does not return an invalid replacement in cases where it doesn't apply."""
    x = pt.matrix("x")
    out = x[x[:, 0] < 0, :].sum(axis=-1)
    fg = FunctionGraph([x], [out])
    assert boolean_indexing_sum.transform(fg, fg.outputs[0].owner) is None

    out = x[x[:, 0] < 0, 0].sum()
    fg = FunctionGraph([x], [out])
    assert boolean_indexing_sum.transform(fg, fg.outputs[0].owner) is None


def test_jax_IncSubtensor():
    rng = np.random.default_rng(213234)

    x_np = rng.uniform(-1, 1, size=(3, 4, 5)).astype(config.floatX)
    x_pt = pt.constant(np.arange(3 * 4 * 5).reshape((3, 4, 5)).astype(config.floatX))

    # "Set" basic indices
    st_pt = pt.as_tensor_variable(np.array(-10.0, dtype=config.floatX))
    out_pt = pt_subtensor.set_subtensor(x_pt[1, 2, 3], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_jax_and_py(out_fg, [])

    st_pt = pt.as_tensor_variable(np.r_[-1.0, 0.0].astype(config.floatX))
    out_pt = pt_subtensor.set_subtensor(x_pt[:2, 0, 0], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_jax_and_py(out_fg, [])

    out_pt = pt_subtensor.set_subtensor(x_pt[0, 1:3, 0], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_jax_and_py(out_fg, [])

    # "Set" advanced indices
    st_pt = pt.as_tensor_variable(
        rng.uniform(-1, 1, size=(2, 4, 5)).astype(config.floatX)
    )
    out_pt = pt_subtensor.set_subtensor(x_pt[np.r_[0, 2]], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_jax_and_py(out_fg, [])

    st_pt = pt.as_tensor_variable(np.r_[-1.0, 0.0].astype(config.floatX))
    out_pt = pt_subtensor.set_subtensor(x_pt[[0, 2], 0, 0], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_jax_and_py(out_fg, [])

    # "Set" boolean indices
    mask_pt = pt.constant(x_np > 0)
    out_pt = pt_subtensor.set_subtensor(x_pt[mask_pt], 0.0)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_jax_and_py(out_fg, [])

    # "Increment" basic indices
    st_pt = pt.as_tensor_variable(np.array(-10.0, dtype=config.floatX))
    out_pt = pt_subtensor.inc_subtensor(x_pt[1, 2, 3], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_jax_and_py(out_fg, [])

    st_pt = pt.as_tensor_variable(np.r_[-1.0, 0.0].astype(config.floatX))
    out_pt = pt_subtensor.inc_subtensor(x_pt[:2, 0, 0], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_jax_and_py(out_fg, [])

    out_pt = pt_subtensor.set_subtensor(x_pt[0, 1:3, 0], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_jax_and_py(out_fg, [])

    # "Increment" advanced indices
    st_pt = pt.as_tensor_variable(
        rng.uniform(-1, 1, size=(2, 4, 5)).astype(config.floatX)
    )
    out_pt = pt_subtensor.inc_subtensor(x_pt[np.r_[0, 2]], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_jax_and_py(out_fg, [])

    st_pt = pt.as_tensor_variable(np.r_[-1.0, 0.0].astype(config.floatX))
    out_pt = pt_subtensor.inc_subtensor(x_pt[[0, 2], 0, 0], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_jax_and_py(out_fg, [])

    # "Increment" boolean indices
    mask_pt = pt.constant(x_np > 0)
    out_pt = pt_subtensor.set_subtensor(x_pt[mask_pt], 1.0)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_jax_and_py(out_fg, [])

    st_pt = pt.as_tensor_variable(x_np[[0, 2], 0, :3])
    out_pt = pt_subtensor.set_subtensor(x_pt[[0, 2], 0, :3], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_jax_and_py(out_fg, [])

    st_pt = pt.as_tensor_variable(x_np[[0, 2], 0, :3])
    out_pt = pt_subtensor.inc_subtensor(x_pt[[0, 2], 0, :3], st_pt)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_jax_and_py(out_fg, [])


def test_jax_IncSubtensor_boolean_indexing_reexpressible():
    """Setting or incrementing values with boolean indexing.

    This test ensures that `AdvancedIncSubtensor` `Op`s with boolean indexing is
    replaced with an equivalent `Switch` `Op`, using the
    `boolean_indexing_set_of_inc` rewrite.

    JAX forces users to re-express this logic manually, so this is an
    improvement over its user interface.

    """
    rng = np.random.default_rng(213234)
    x_np = rng.uniform(-1, 1, size=(4, 5)).astype(config.floatX)

    x_pt = pt.matrix("x")
    mask_pt = pt.as_tensor(x_pt) > 0
    out_pt = pt_subtensor.set_subtensor(x_pt[mask_pt], 0.0)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_jax_and_py(out_fg, [x_np])

    mask_pt = pt.as_tensor(x_pt) > 0
    out_pt = pt_subtensor.inc_subtensor(x_pt[mask_pt], 1.0)
    assert isinstance(out_pt.owner.op, pt_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_jax_and_py(out_fg, [x_np])


def test_boolean_indexing_set_or_inc_not_applicable():
    """Test that `boolean_indexing_set_or_inc` does not return an invalid replacement in cases where it doesn't apply."""
    x = pt.vector("x")
    mask = pt.as_tensor(x) > 0
    out = pt_subtensor.set_subtensor(x[mask], [0, 1, 2])
    fg = FunctionGraph([x], [out])
    assert boolean_indexing_set_or_inc.transform(fg, fg.outputs[0].owner) is None
