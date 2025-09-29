import re
from collections.abc import Sequence
from itertools import product
from typing import cast

import numpy as np
from numpy import nditer
from numpy.lib.array_utils import normalize_axis_tuple

import pytensor
from pytensor.graph import FunctionGraph, Variable
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.utils import hash_from_code


def hash_from_ndarray(data) -> str:
    """
    Return a hash from an ndarray.

    It takes care of the data, shapes, strides and dtype.

    """
    # We need to hash the shapes and strides as hash_from_code only hashes
    # the data buffer. Otherwise, this will cause problem with shapes like:
    # (1, 0) and (2, 0) and problem with inplace transpose.
    # We also need to add the dtype to make the distinction between
    # uint32 and int32 of zeros with the same shape and strides.

    # python hash are not strong, so use sha256 (md5 is not
    # FIPS compatible). To not have too long of hash, I call it again on
    # the concatenation of all parts.
    if not data.flags["C_CONTIGUOUS"]:
        # hash_from_code needs a C-contiguous array.
        data = np.ascontiguousarray(data)
    return hash_from_code(
        hash_from_code(data)
        + hash_from_code(str(data.shape))
        + hash_from_code(str(data.strides))
        + hash_from_code(str(data.dtype))
    )


def shape_of_variables(
    fgraph: FunctionGraph, input_shapes
) -> dict[Variable, tuple[int, ...]]:
    """
    Compute the numeric shape of all intermediate variables given input shapes.

    Parameters
    ----------
    fgraph
        The FunctionGraph in question.
    input_shapes : dict
        A dict mapping input to shape.

    Returns
    -------
    shapes : dict
        A dict mapping variable to shape

    .. warning:: This modifies the fgraph. Not pure.

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> from pytensor.graph.fg import FunctionGraph
    >>> x = pt.matrix("x")
    >>> y = x[512:]
    >>> y.name = "y"
    >>> fgraph = FunctionGraph([x], [y], clone=False)
    >>> d = shape_of_variables(fgraph, {x: (1024, 1024)})
    >>> d[y]
    (array(512), array(1024))
    >>> d[x]
    (array(1024), array(1024))
    """

    if not hasattr(fgraph, "shape_feature"):
        from pytensor.tensor.rewriting.shape import ShapeFeature

        fgraph.attach_feature(ShapeFeature())

    shape_feature = fgraph.shape_feature  # type: ignore[attr-defined]

    input_dims = [
        dimension for inp in fgraph.inputs for dimension in shape_feature.shape_of[inp]
    ]

    output_dims = [
        dimension for shape in shape_feature.shape_of.values() for dimension in shape
    ]

    compute_shapes = pytensor.function(input_dims, output_dims)

    if any(i not in fgraph.inputs for i in input_shapes):
        raise ValueError(
            "input_shapes keys aren't in the fgraph.inputs. FunctionGraph()"
            " interface changed. Now by default, it clones the graph it receives."
            " To have the old behavior, give it this new parameter `clone=False`."
        )

    numeric_input_dims = [dim for inp in fgraph.inputs for dim in input_shapes[inp]]
    numeric_output_dims = compute_shapes(*numeric_input_dims)

    sym_to_num_dict = dict(zip(output_dims, numeric_output_dims, strict=True))

    l = {}
    for var in shape_feature.shape_of:
        l[var] = tuple(sym_to_num_dict[sym] for sym in shape_feature.shape_of[var])
    return l


def import_func_from_string(func_string: str):  # -> Optional[Callable]:
    func = getattr(np, func_string, None)
    if func is not None:
        return func

    # Not inside NumPy or Scipy. So probably another package like scipy.
    module = None
    items = func_string.split(".")
    for idx in range(1, len(items)):
        try:
            module = __import__(".".join(items[:idx]))
        except ImportError:
            break

    if module:
        for sub in items[1:]:
            try:
                module = getattr(module, sub)
            except AttributeError:
                module = None
                break
        return module


def broadcast_static_dim_lengths(
    dim_lengths: Sequence[int | None],
) -> int | None:
    """Apply static broadcast given static dim length of inputs (obtained from var.type.shape).

    Raises
    ------
    ValueError
        When static dim lengths are incompatible
    """

    dim_lengths_set = set(dim_lengths)
    # All dim_lengths are the same
    if len(dim_lengths_set) == 1:
        return next(iter(dim_lengths_set))

    # Only valid indeterminate case
    if dim_lengths_set == {None, 1}:
        return None

    dim_lengths_set.discard(1)
    dim_lengths_set.discard(None)
    if len(dim_lengths_set) > 1:
        raise ValueError
    return next(iter(dim_lengths_set))


# Copied verbatim from numpy.lib.function_base
# https://github.com/numpy/numpy/blob/f2db090eb95b87d48a3318c9a3f9d38b67b0543c/numpy/lib/function_base.py#L1999-L2029
_DIMENSION_NAME = r"\w+"
_CORE_DIMENSION_LIST = f"(?:{_DIMENSION_NAME}(?:,{_DIMENSION_NAME})*)?"
_ARGUMENT = rf"\({_CORE_DIMENSION_LIST}\)"
_ARGUMENT_LIST = f"{_ARGUMENT}(?:,{_ARGUMENT})*"
# Allow no inputs
_SIGNATURE = f"^(?:{_ARGUMENT_LIST})?->{_ARGUMENT_LIST}$"


def _parse_gufunc_signature(
    signature: str,
) -> tuple[
    list[tuple[str, ...]], ...
]:  # mypy doesn't know it's alwayl a length two tuple
    """
    Parse string signatures for a generalized universal function.

    Arguments
    ---------
    signature : string
        Generalized universal function signature, e.g., ``(m,n),(n,p)->(m,p)``
        for ``np.matmul``.

    Returns
    -------
    Tuple of input and output core dimensions parsed from the signature, each
    of the form List[Tuple[str, ...]].
    """
    signature = re.sub(r"\s+", "", signature)

    if not re.match(_SIGNATURE, signature):
        raise ValueError(f"not a valid gufunc signature: {signature}")
    return tuple(
        [
            tuple(re.findall(_DIMENSION_NAME, arg))
            for arg in re.findall(_ARGUMENT, arg_list)
        ]
        if arg_list  # ignore no inputs
        else []
        for arg_list in signature.split("->")
    )


def safe_signature(
    core_inputs_ndim: Sequence[int],
    core_outputs_ndim: Sequence[int],
) -> str:
    def operand_sig(operand_ndim: int, prefix: str) -> str:
        operands = ",".join(f"{prefix}{i}" for i in range(operand_ndim))
        return f"({operands})"

    inputs_sig = ",".join(
        operand_sig(ndim, prefix=f"i{n}") for n, ndim in enumerate(core_inputs_ndim)
    )
    outputs_sig = ",".join(
        operand_sig(ndim, prefix=f"o{n}") for n, ndim in enumerate(core_outputs_ndim)
    )
    return f"{inputs_sig}->{outputs_sig}"


def normalize_reduce_axis(axis, ndim: int) -> tuple[int, ...] | None:
    """Normalize the axis parameter for reduce operations."""
    if axis is None:
        return None

    # scalar inputs are treated as 1D regarding axis in reduce operations
    if axis is not None:
        try:
            axis = normalize_axis_tuple(axis, ndim=max(1, ndim))
        except np.exceptions.AxisError:
            raise np.exceptions.AxisError(axis, ndim=ndim)

    # TODO: If axis tuple is equivalent to None, return None for more canonicalization?
    return cast(tuple, axis)


def faster_broadcast_to(x, shape):
    # Stripped down core logic of `np.broadcast_to`
    return nditer(
        (x,),
        flags=["multi_index", "zerosize_ok"],
        op_flags=["readonly"],
        itershape=shape,
        order="C",
    ).itviews[0]


def faster_ndindex(shape: Sequence[int]):
    """Equivalent to `np.ndindex` but usually 10x faster.

    Unlike `np.ndindex`, this function expects a single sequence of integers

    https://github.com/numpy/numpy/issues/28921
    """
    return product(*(range(s) for s in shape))


def get_static_shape_from_size_variables(
    size_vars: Sequence[Variable],
) -> tuple[int | None, ...]:
    """Get static shape from size variables.

    Parameters
    ----------
    size_vars : Sequence[Variable]
        A sequence of variables representing the size of each dimension.
    Returns
    -------
    tuple[int | None, ...]
        A tuple containing the static lengths of each dimension, or None if
        the length is not statically known.
    """
    from pytensor.tensor.basic import get_scalar_constant_value

    static_lengths: list[None | int] = [None] * len(size_vars)
    for i, length in enumerate(size_vars):
        try:
            static_length = get_scalar_constant_value(length)
        except NotScalarConstantError:
            pass
        else:
            static_lengths[i] = int(static_length)
    return tuple(static_lengths)
