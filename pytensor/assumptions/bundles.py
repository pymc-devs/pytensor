from pytensor.assumptions.alloc import alloc_propagates_matrix_property
from pytensor.assumptions.core import AssumptionKey, register_assumption
from pytensor.assumptions.dimshuffle import dimshuffle_propagates_matrix_property
from pytensor.assumptions.reshape import (
    join_dims_propagates_matrix_property,
    split_dims_propagates_matrix_property,
)
from pytensor.assumptions.shape import (
    reshape_propagates_matrix_property,
    specify_shape_propagates_matrix_property,
)
from pytensor.assumptions.subtensor import (
    incsubtensor_propagates_matrix_property,
    subtensor_propagates_matrix_property,
)
from pytensor.tensor.basic import Alloc
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.reshape import JoinDims, SplitDims
from pytensor.tensor.shape import Reshape, SpecifyShape
from pytensor.tensor.subtensor import IncSubtensor, Subtensor


def register_matrix_property_rules(key: AssumptionKey) -> None:
    """Register the standard propagation rules for a property of the trailing two axes.

    Every rule here answers one question: does the Op leave the trailing two axes
    undisturbed? The bundle thus suits any property of a matrix that batch dimensions
    carry elementwise, such as triangularity or a fixed sparsity pattern.

    Rules are tried in registration order until one returns a non-UNKNOWN state, so a
    key needing different behavior for one Op registers its own with
    ``register_assumption(..., prepend=True)``.

    Parameters
    ----------
    key : AssumptionKey
        The property to install the rules for.

    Examples
    --------
    .. code-block:: python

        from pytensor.assumptions import AssumptionKey, register_matrix_property_rules

        TOEPLITZ = AssumptionKey("toeplitz", short_name="toep")
        register_matrix_property_rules(TOEPLITZ)
    """
    register_assumption(key, DimShuffle)(dimshuffle_propagates_matrix_property)
    register_assumption(key, Reshape)(reshape_propagates_matrix_property)
    register_assumption(key, SpecifyShape)(specify_shape_propagates_matrix_property)
    register_assumption(key, JoinDims)(join_dims_propagates_matrix_property)
    register_assumption(key, SplitDims)(split_dims_propagates_matrix_property)
    register_assumption(key, Alloc)(alloc_propagates_matrix_property)
    register_assumption(key, Subtensor)(subtensor_propagates_matrix_property)
    register_assumption(key, IncSubtensor)(incsubtensor_propagates_matrix_property)
