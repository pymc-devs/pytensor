import pytest

import pytensor.tensor as pt
from pytensor.assumptions import (
    MATRIX_KEYS,
    AssumptionKey,
    FactState,
    assume,
    register_assumption,
    register_matrix_property_rules,
)
from pytensor.assumptions.core import ASSUMPTION_INFER_REGISTRY
from pytensor.tensor.basic import Alloc
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.reshape import JoinDims, SplitDims
from pytensor.tensor.shape import Reshape, SpecifyShape
from pytensor.tensor.subtensor import IncSubtensor
from tests.assumptions.conftest import make_fgraph


def test_bundle_propagates_through_the_standard_ops():
    TOEPLITZ = AssumptionKey("toeplitz", short_name="toep")
    register_matrix_property_rules(TOEPLITZ)

    x = assume(pt.tensor("x", shape=(4, 3, 3)), toeplitz=True)

    indexed = x[0]
    reshaped = x.reshape((2, 2, 3, 3))
    shape_specified = pt.specify_shape(x, (4, 3, 3))

    _, af = make_fgraph(indexed, reshaped, shape_specified)
    assert af.check(indexed, TOEPLITZ)
    assert af.check(reshaped, TOEPLITZ)
    assert af.check(shape_specified, TOEPLITZ)


def test_bundle_leaves_the_core_axes_alone():
    """The rules protect the trailing two axes -- disturbing them stops propagation."""
    TOEPLITZ = AssumptionKey("toeplitz", short_name="toep")
    register_matrix_property_rules(TOEPLITZ)

    x = assume(pt.matrix("x", shape=(3, 3)), toeplitz=True)
    transposed = x.T

    _, af = make_fgraph(transposed)
    assert af.get(transposed, TOEPLITZ) is FactState.UNKNOWN


@pytest.mark.parametrize(
    "prepend, expected",
    [(True, FactState.FALSE), (False, FactState.TRUE)],
    ids=["prepend-wins", "append-loses"],
)
def test_prepend_decides_which_rule_answers(prepend, expected):
    """Expand-dims is a case the bundle answers, so only order decides the outcome."""
    TOEPLITZ = AssumptionKey("toeplitz", short_name="toep")
    register_matrix_property_rules(TOEPLITZ)

    @register_assumption(TOEPLITZ, DimShuffle, prepend=prepend)
    def _never_survives_expand_dims(key, op, feature, fgraph, node, input_states):
        return [FactState.FALSE] if op.is_expand_dims else [FactState.UNKNOWN]

    x = assume(pt.matrix("x", shape=(3, 3)), toeplitz=True)
    expanded = x[None]

    _, af = make_fgraph(expanded)
    assert af.get(expanded, TOEPLITZ) is expected


@pytest.mark.parametrize("key", MATRIX_KEYS, ids=lambda k: k.name)
@pytest.mark.parametrize(
    "op_type",
    [DimShuffle, Reshape, SpecifyShape, JoinDims, SplitDims, Alloc, IncSubtensor],
    ids=lambda op: op.__name__,
)
def test_core_matrix_keys_carry_the_bundled_rules(key, op_type):
    """The bundle stays in step with what the built-in matrix properties register.

    ``Subtensor`` is excluded: ``SELECTION`` registers its own rule there instead of the
    shared one, so that Op is deliberately not uniform across the built-in keys.
    """
    probe = AssumptionKey("probe")
    register_matrix_property_rules(probe)

    bundled = set(ASSUMPTION_INFER_REGISTRY[(probe, op_type)])
    assert bundled <= set(ASSUMPTION_INFER_REGISTRY[(key, op_type)])
