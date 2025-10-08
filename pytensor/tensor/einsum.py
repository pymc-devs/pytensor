import collections
import warnings
from collections.abc import Sequence
from functools import partial, reduce
from itertools import pairwise
from typing import cast

import numpy as np
from numpy._core.einsumfunc import (  # type: ignore[attr-defined]
    _find_contraction,
    _parse_einsum_input,
)
from numpy.lib.array_utils import normalize_axis_index, normalize_axis_tuple

from pytensor.compile.builders import OpFromGraph
from pytensor.tensor import TensorLike
from pytensor.tensor.basic import (
    arange,
    as_tensor,
    expand_dims,
    get_vector_length,
    moveaxis,
    stack,
    transpose,
    where,
)
from pytensor.tensor.extra_ops import broadcast_to
from pytensor.tensor.functional import vectorize
from pytensor.tensor.math import and_, eq, tensordot
from pytensor.tensor.shape import shape_padright
from pytensor.tensor.variable import TensorVariable


PATH = tuple[tuple[int] | tuple[int, int], ...]


class Einsum(OpFromGraph):
    """
    Wrapper Op for Einsum graphs

    Notes
    -----
    The `optimized` prop indicates whether the inner graph was optimized, which can only be done when all shapes are
    statically known. This is now determined at graph creation time only. We could introduce a rewrite that tries to
    optimize the graph if static shapes become known later (e.g., after use of `clone_replace` or shape inference during
    rewrites).

    Also, once the graph is optimized, it could be inlined for potential further optimization that consider the rest of
    the graph.

    This prop is different from the `optimize` kwarg in numpy that determines what kind (if any) of optimization is
    desired. We haven't decided whether we want to provide this functionality.
    """

    def __init__(self, *args, subscripts: str, path: PATH, optimized: bool, **kwargs):
        self.subscripts = subscripts
        self.path = path
        self.optimized = optimized
        super().__init__(*args, **kwargs, strict=True)

    def __str__(self):
        return f"Einsum{{{self.subscripts=}, {self.path=}, {self.optimized=}}}"


def _iota(shape: TensorVariable, axis: int) -> TensorVariable:
    """
    Create an array with values increasing along the specified axis.

    Iota is a multidimensional generalization of the `arange` function. The returned array is filled with whole numbers
    increasing along the specified axis.

    Parameters
    ----------
    shape: TensorVariable
        The shape of the array to be created.
    axis: int
        The axis along which to fill the array with increasing values.

    Returns
    -------
    TensorVariable
        An array with values increasing along the specified axis.

    Examples
    --------
    In the simplest case where ``shape`` is 1d, the output will be equivalent to ``pt.arange``:

    .. testcode::

        import pytensor.tensor as pt
        from pytensor.tensor.einsum import _iota

        shape = pt.as_tensor((5,))
        print(_iota(shape, 0).eval())

    .. testoutput::

         [0 1 2 3 4]

    In higher dimensions, it will look like many concatenated `arange`:

    .. testcode::

        shape = pt.as_tensor((5, 5))
        print(_iota(shape, 1).eval())

    .. testoutput::

        [[0 1 2 3 4]
         [0 1 2 3 4]
         [0 1 2 3 4]
         [0 1 2 3 4]
         [0 1 2 3 4]]

    Setting ``axis=0`` above would result in the transpose of the output.
    """
    len_shape = get_vector_length(shape)
    axis = normalize_axis_index(axis, len_shape)
    values = arange(shape[axis])
    return broadcast_to(shape_padright(values, len_shape - axis - 1), shape)


def _delta(shape: TensorVariable, axes: Sequence[int]) -> TensorVariable:
    """
    Create a Kroncker delta tensor.

    The Kroncker delta function is defined:

    .. math::

        \\delta(i, j) = \begin{cases} 1 & \text{if} \\quad i = j \\ 0 & \text{otherwise} \\end{cases}

    To create a Kronecker tensor, the delta function is applied elementwise to the axes specified. The result is a
    tensor of booleans, with ``True`` where the axis indices coincide, and ``False`` otherwise. See below for examples.

    Parameters
    ----------
    shape: TensorVariable
        The shape of the tensor to be created. Note that `_delta` is not defined for 1d tensors, because there is no
        second axis against which to compare.
    axes: sequence of int
        Axes whose indices should be compared. Note that `_delta` is not defined for a single axis, because there is no
        second axis against which to compare.

    Examples
    --------
    An easy case to understand is when the shape is square and the number of axes is equal to the number of dimensions.
    This will result in a generalized identity tensor, with ``True`` along the main diagonal:

    .. testcode::

        from pytensor.tensor.einsum import _delta
        print(_delta((5, 5), (0, 1)).eval())

    .. testoutput::

        [[ True False False False False]
         [False  True False False False]
         [False False  True False False]
         [False False False  True False]
         [False False False False  True]]

    In the case where the shape is not square, the result will be a tensor with ``True`` along the main diagonal and
    ``False`` elsewhere:

    .. testcode::

        from pytensor.tensor.einsum import _delta
        print(_delta((3, 2), (0, 1)).eval())

    .. testoutput::

        [[ True False]
         [False  True]
         [False False]]

    When there are more than two dimensions in the shape, axes can be only a subset of them, leading to different
    arragements of True and False values. For example for a 3d batch of matrices, choosing axes (0, 2) will lead to
    True values on the column corresponding to the batch index of each matrix:

    .. testcode::

        from pytensor.tensor.einsum import _delta
        print(_delta((3, 3, 3), (0, 2)).eval())

    .. testoutput::

        [[[ True False False]
          [ True False False]
          [ True False False]]

         [[False  True False]
          [False  True False]
          [False  True False]]

         [[False False  True]
          [False False  True]
          [False False  True]]]
    """
    if len(axes) == 1:
        raise ValueError("Need at least two axes to create a delta tensor")
    base_shape = stack([shape[axis] for axis in axes])
    iotas = [_iota(base_shape, i) for i in range(len(axes))]
    eyes = [eq(i1, i2) for i1, i2 in pairwise(iotas)]
    result = reduce(and_, eyes)
    non_axes = [i for i in range(len(tuple(shape))) if i not in axes]
    return broadcast_to(expand_dims(result, non_axes), shape)


def _general_dot(
    vars: tuple[TensorVariable, TensorVariable],
    axes: Sequence[Sequence[int]],  # Should be length 2,
    batch_axes: Sequence[Sequence[int]],  # Should be length 2,
) -> TensorVariable:
    """
    Generalized dot product between two tensors.

    Ultimately ``_general_dot`` is a call to `tensor_dot`, performing a multiply-and-sum ("dot") operation between two
    tensors, along a requested dimension. This function further generalizes this operation by allowing arbitrary
    batch dimensions to be specified for each tensor.


    Parameters
    ----------
    vars: tuple[TensorVariable, TensorVariable]
        The tensors to be ``tensor_dot``ed
    axes: Sequence[Sequence[int]]
        The axes along which to perform the dot product. Should be a sequence of two sequences, one for each tensor.
    batch_axes: Sequence[Sequence[int]]
        The batch axes for each tensor. Should be a sequence of two sequences, one for each tensor.

    Returns
    -------
    TensorVariable
        The result of the ``tensor_dot`` product.

    Examples
    --------
    Perform a batched dot product between two 3d tensors:

    .. testcode::

        import pytensor.tensor as pt
        from pytensor.tensor.einsum import _general_dot
        import numpy as np

        A = pt.tensor(shape=(3, 4, 5))
        B = pt.tensor(shape=(3, 5, 2))

        result = _general_dot((A, B), axes=[[2], [1]], batch_axes=[[0], [0]])

        A_val = np.empty((3, 4, 5))
        B_val = np.empty((3, 5, 2))
        print(tuple(result.shape.eval({A:A_val, B:B_val})))

    .. testoutput::

        (np.int64(3), np.int64(4), np.int64(2))
    """
    # Shortcut for non batched case
    if not batch_axes[0] and not batch_axes[1]:
        return tensordot(*vars, axes=axes)

    # Normalize axes, thankfully numpy helper does not sort axis!
    axes = [
        normalize_axis_tuple(var_axes, var.ndim)
        for var, var_axes in zip(vars, axes, strict=True)
    ]
    batch_axes = [
        normalize_axis_tuple(var_axes, var.ndim)
        for var, var_axes in zip(vars, batch_axes, strict=True)
    ]
    n_batch_axes = [len(var_batch_axes) for var_batch_axes in batch_axes]

    # Move batch axes to the left and recode reduction axes
    new_vars = list(vars)
    new_axes = list(axes)
    for i, (var, var_axes, var_batch_axes, var_n_batch_axes) in enumerate(
        zip(vars, axes, batch_axes, n_batch_axes, strict=True)
    ):
        if var_batch_axes == tuple(range(var_n_batch_axes)):
            # Already on left to right order
            continue

        new_var_batch_axes = tuple(range(var_n_batch_axes))
        new_var = moveaxis(var, var_batch_axes, new_var_batch_axes)

        new_var_axes = []
        for var_axis in var_axes:
            batch_axes_to_the_right = len(
                [batch_axis for batch_axis in var_batch_axes if batch_axis > var_axis]
            )
            new_var_axes.append(var_axis + batch_axes_to_the_right)

        new_vars[i] = new_var
        new_axes[i] = new_var_axes

    lhs, rhs = new_vars
    lhs_axes, rhs_axes = new_axes
    lhs_n_batch_axes, rhs_n_batch_axes = n_batch_axes

    # Create signature of tensordot
    lhs_signature = [f"l{i}" for i in range(lhs.type.ndim)]
    rhs_signature = [f"r{i}" for i in range(rhs.type.ndim)]
    # Aligned axes get the same dimension name
    for i, (lhs_axis, rhs_axis) in enumerate(zip(lhs_axes, rhs_axes, strict=True)):
        lhs_signature[lhs_axis] = rhs_signature[rhs_axis] = f"a{i}"
    # Trim away the batch ndims
    lhs_signature = lhs_signature[lhs_n_batch_axes:]
    rhs_signature = rhs_signature[rhs_n_batch_axes:]
    out_signature = [
        lhs_dim for lhs_dim in lhs_signature if not lhs_dim.startswith("a")
    ] + [rhs_dim for rhs_dim in rhs_signature if not rhs_dim.startswith("a")]
    signature = f"({','.join(lhs_signature)}),({','.join(rhs_signature)})->({','.join(out_signature)})"
    # Adjust axes for core case
    core_lhs_axes = tuple(np.array(lhs_axes) - lhs_n_batch_axes)
    core_rhs_axes = tuple(np.array(rhs_axes) - rhs_n_batch_axes)

    if signature == "(),()->()":
        # Just a multiplication
        out = lhs * rhs
    else:
        out = vectorize(
            partial(tensordot, axes=[core_lhs_axes, core_rhs_axes]), signature=signature
        )(lhs, rhs)

    return cast(TensorVariable, out)


def _contraction_list_from_path(
    subscripts: str, operands: Sequence[TensorVariable], path: PATH
):
    """
    Generate a list of contraction steps based on the provided einsum path.

    Code adapted from einsum_opt: https://github.com/dgasmith/opt_einsum/blob/94c62a05d5ebcedd30f59c90b9926de967ed10b5/opt_einsum/contract.py#L369

    When all shapes are known, the linked einsum_opt implementation is preferred. This implementation is used when
    some or all shapes are not known. As a result, contraction will (always?) be done left-to-right, pushing intermediate
    results to the end of the stack.

    Parameters
    ----------
    subscripts: str
        Einsum signature string describing the computation to be performed.

    operands: Sequence[TensorLike]
        Tensors described by the subscripts.

    path: tuple[tuple[int] | tuple[int, int]]
        A list of tuples, where each tuple describes the indices of the operands to be contracted, sorted in the order
        they should be contracted.

    Returns
    -------
    contraction_list: list
        A list of tuples, where each tuple describes a contraction step. Each tuple contains the following elements:
        - contraction_inds: tuple[int]
            The indices of the operands to be contracted
        - idx_removed: str
            The indices of the contracted indices (those removed from the einsum string at this step)
        - einsum_str: str
            The einsum string for the contraction step
        - remaining: None
            The remaining indices. Included to match the output of opt_einsum.contract_path, but not used.
        - do_blas: None
            Whether to use blas to perform this step. Included to match the output of opt_einsum.contract_path,
            but not used.
    """
    fake_operands = [
        np.zeros([1 if dim == 1 else 0 for dim in x.type.shape]) for x in operands
    ]
    input_subscripts, output_subscript, operands = _parse_einsum_input(
        (subscripts, *fake_operands)
    )

    # Build a few useful list and sets
    input_list = input_subscripts.split(",")
    input_sets = [set(x) for x in input_list]
    output_set = set(output_subscript)

    # Build contraction tuple (positions, gemm, einsum_str, remaining)
    contraction_list = []
    for cnum, contract_inds in enumerate(path):
        # Make sure we remove inds from right to left
        contract_inds = cast(
            tuple[int] | tuple[int, int], tuple(sorted(contract_inds, reverse=True))
        )

        contract_tuple = _find_contraction(contract_inds, input_sets, output_set)
        out_inds, input_sets, idx_removed, _idx_contract = contract_tuple

        tmp_inputs = [input_list.pop(x) for x in contract_inds]

        # Last contraction
        if (cnum - len(path)) == -1:
            idx_result = output_subscript
        else:
            # use tensordot order to minimize transpositions
            all_input_inds = "".join(tmp_inputs)
            idx_result = "".join(sorted(out_inds, key=all_input_inds.find))

        input_list.append(idx_result)
        einsum_str = ",".join(tmp_inputs) + "->" + idx_result

        # We only need the first three inputs to build the forward graph
        contraction = (contract_inds, idx_removed, einsum_str, None, None)
        contraction_list.append(contraction)

    return contraction_list


def _right_to_left_path(n: int) -> tuple[tuple[int, int], ...]:
    # Create a right to left contraction path
    # if n = 5, out = ((4, 3), (3, 2), (2, 1), (1, 0))
    return tuple(pairwise(reversed(range(n))))


def _ensure_not_equal(elements):
    """
    Ensures that any pair in a list of elements are not the same object. If a pair of elements is found to be equal, then one of them is converted to a copy.
    """
    elements = list(elements)
    for i, elem1 in enumerate(elements[:-1]):
        for j, elem2 in enumerate(elements[i + 1 :], start=i + 1):
            if elem1 is elem2:
                elements[j] = elem1.copy()
    return elements


def einsum(subscripts: str, *operands: "TensorLike", optimize=None) -> TensorVariable:
    """
    Multiplication and summation of tensors using the Einstein summation convention.

    Code adapted from JAX: https://github.com/google/jax/blob/534d32a24d7e1efdef206188bb11ae48e9097092/jax/_src/numpy/lax_numpy.py#L5283

    Einsum allows the user to specify a wide range of operations on tensors using the Einstein summation convention. Using
    this notation, many common linear algebraic operations can be succinctly described on higher order tensors.

    Parameters
    ----------
    subscripts: str
        Einsum signature string describing the computation to be performed.

    operands: sequence of TensorVariable
        Tensors to be multiplied and summed.

    Returns
    -------
    TensorVariable
        The result of the einsum operation.

    See Also
    --------
    pytensor.tensor.tensordot: Generalized dot product between two tensors
    pytensor.tensor.dot: Matrix multiplication between two tensors
    numpy.einsum: The numpy implementation of einsum

    Examples
    --------
    Inputs to `pt.einsum` are a string describing the operation to be performed (the "subscripts"), and a sequence of
    tensors to be operated on. The string must follow the following rules:

    1. The string gives inputs and (optionally) outputs. Inputs and outputs are separated by "->".
    2. The input side of the string is a comma-separated list of indices. For each comma-separated index string, there
         must be a corresponding tensor in the input sequence.
    3. For each index string, the number of dimensions in the corresponding tensor must match the number of characters
         in the index string.
    4. Indices are arbitrary strings of characters. If an index appears multiple times in the input side, it must have
        the same shape in each input.
    5. The indices on the output side must be a subset of the indices on the input side -- you cannot introduce new
        indices in the output.
    6. Elipses ("...") can be used to elide multiple indices. This is useful when you have a large number of "batch"
        dimensions that are not implicated in the operation.

    Finally, two rules about these indicies govern how computation is carried out:

    1. Repeated indices on the input side indicate how the tensor should be "aligned" for multiplication.
    2. Indices that appear on the input side but not the output side are summed over.

    The operation of these rules is best understood via examples:

    Example 1: Matrix multiplication

    .. code-block:: python

        import pytensor as pt

        A = pt.matrix("A")
        B = pt.matrix("B")
        C = pt.einsum("ij, jk -> ik", A, B)

    This computation is equivalent to :code:`C = A @ B`. Notice that the ``j`` index is repeated on the input side of the
    signature, and does not appear on the output side. This indicates that the ``j`` dimension of the first tensor should be
    multiplied with the ``j`` dimension of the second tensor, and the resulting tensor's ``j`` dimension should be summed
    away.

    Example 2: Batched matrix multiplication

    .. code-block:: python

        import pytensor as pt

        A = pt.tensor("A", shape=(None, 4, 5))
        B = pt.tensor("B", shape=(None, 5, 6))
        C = pt.einsum("bij, bjk -> bik", A, B)

    This computation is also equivalent to :code:`C = A @ B` because of Pytensor's built-in broadcasting rules, but
    the einsum signature is more explicit about the batch dimensions. The ``b`` and ``j`` indices are repeated on the
    input side. Unlike ``j``, the ``b`` index is also present on the output side, indicating that the batch dimension
    should **not** be summed away. As a result, multiplication will be performed over the ``b, j`` dimensions, and then
    the ``j`` dimension will be summed over. The resulting tensor will have shape ``(None, 4, 6)``.

    Example 3: Batched matrix multiplication with elipses

    .. code-block:: python

        import pytensor as pt

        A = pt.tensor("A", shape=(4, None, None, None, 5))
        B = pt.tensor("B", shape=(5, None, None, None, 6))
        C = pt.einsum("i...j, j...k -> ...ik", A, B)

    This case is the same as above, but inputs ``A`` and ``B`` have multiple batch dimensions. To avoid writing out all
    of the batch dimensions (which we do not care about), we can use ellipses to elide over these dimensions. Notice
    also that we are not required to "sort" the input dimensions in any way. In this example, we are doing a dot
    between the last dimension A and the first dimension of B, which is perfectly valid.

    Example 4: Outer product

    .. code-block:: python

        import pytensor as pt

        x = pt.tensor("x", shape=(3,))
        y = pt.tensor("y", shape=(4,))
        z = pt.einsum("i, j -> ij", x, y)

    This computation is equivalent to :code:`pt.outer(x, y)`. Notice that no indices are repeated on the input side,
    and the output side has two indices. Since there are no indices to align on, the einsum operation will simply
    multiply the two tensors elementwise, broadcasting dimensions ``i`` and ``j``.

    Example 5: Convolution

    .. code-block:: python

            import pytensor as pt
            x = pt.tensor("x", shape=(None, None, None, None, None, None))
            w = pt.tensor("w", shape=(None, None, None, None))
            y = pt.einsum(""bchwkt,fckt->bfhw", x, w)

    Given a batch of images ``x`` with dimensions ``(batch, channel, height, width, kernel_size, num_filters)``
    and a filter ``w``, with dimensions ``(num_filters, channels, kernel_size, num_filters)``,  this einsum operation
    computes the convolution of ``x`` with ``w``. Multiplication is aligned on the batch, num_filters, height, and width
    dimensions. The channel, kernel_size, and num_filters dimensions are summed over. The resulting tensor has shape
    ``(batch, num_filters, height, width)``, reflecting the fact that information from each channel has been mixed
    together.
    """

    if optimize is not None:
        raise NotImplementedError(
            "Optimize kwarg is not implemented in PyTensor. "
            "By default, PyTensor will always optimize the graph if the inputs have static shapes.\n"
            "If you need this functionality open an issue in https://github.com/pymc-devs/pytensor/issues to let us know. "
        )

    tensor_operands = _ensure_not_equal([as_tensor(operand) for operand in operands])
    shapes = [operand.type.shape for operand in tensor_operands]

    path: PATH
    if any(None in shape for shape in shapes):
        # Case 1: At least one of the operands has an unknown shape. In this case, we can't use opt_einsum to optimize
        # the contraction order, so we just use a default path of (1,0) contractions. This will work left-to-right,
        # pushing intermediate results to the end of the stack.
        # We use (1,0) and not (0,1) because that's what opt_einsum tends to prefer, and so the Op signatures will
        # match more often

        # If shapes become known later we will likely want to rebuild the Op (unless we inline it)
        if len(tensor_operands) == 1:
            path = ((0,),)
        else:
            # By default, we try right to left because we assume that most graphs
            # have a lower dimensional rightmost operand
            path = _right_to_left_path(len(tensor_operands))
        contraction_list = _contraction_list_from_path(
            subscripts, tensor_operands, path
        )

        # If there are only 1 or 2 operands, there is no optimization to be done?
        optimized = len(tensor_operands) <= 2
    else:
        # Case 2: All operands have known shapes. In this case, we can use opt_einsum to compute the optimal
        # contraction order.
        _, contraction_list = np.einsum_path(
            subscripts,
            # Numpy einsum_path requires arrays even though only the shapes matter
            # It's not trivial to duck-type our way around because of internal call to `asanyarray`
            *[np.empty(shape) for shape in shapes],
            # einsum_call is not part of public API
            einsum_call=True,  # type: ignore[arg-type]
            optimize="optimal",
        )
        np_path: PATH | tuple[tuple[int, ...]] = tuple(
            contraction[0]  # type: ignore[misc]
            for contraction in contraction_list
        )

        if len(np_path) == 1 and len(np_path[0]) > 2:
            # When there's nothing to optimize, einsum_path reduces all entries simultaneously instead of doing
            # pairwise reductions, which our implementation below demands.
            path = _right_to_left_path(len(tensor_operands))
            contraction_list = _contraction_list_from_path(
                subscripts, tensor_operands, path
            )
        else:
            path = cast(PATH, np_path)

        optimized = True

    def removechars(s, chars):
        return s.translate(str.maketrans(dict.fromkeys(chars)))

    def sum_uniques(
        operand: TensorVariable, names: str, uniques: list[str]
    ) -> tuple[TensorVariable, str]:
        """Reduce unique indices (those that appear only once) in a given contraction step via summing."""
        if uniques:
            axes = [names.index(name) for name in uniques]
            operand = operand.sum(axes)
            names = removechars(names, uniques)
        return operand, names

    def sum_repeats(
        operand: TensorVariable,
        names: str,
        counts: collections.Counter,
        keep_names: str,
    ) -> tuple[TensorVariable, str]:
        """Reduce repeated indices in a given contraction step via summation against an identity matrix."""

        for name, count in counts.items():
            if count > 1:
                axes = [i for i, n in enumerate(names) if n == name]
                eye = _delta(operand.shape, axes)
                operand = where(eye, operand, operand.zeros_like())
                if name not in keep_names:
                    operand = operand.sum(axes)
                    names = names.replace(name, "")
                else:
                    operand = operand.sum(axes[:-1])
                    names = names.replace(name, "", count - 1)
        return operand, names

    def filter_singleton_dims(operand, names, other_operand, other_names):
        op_bcast = operand.type.broadcastable
        other_bcast = other_operand.type.broadcastable
        keep = [
            (not op_bcast[i]) or (j == -1) or other_bcast[j]
            for i, j in enumerate(map(other_names.find, names))
        ]
        keep_axes = [i for i, keep_axis in enumerate(keep) if keep_axis]
        squeeze_axes = [i for i, keep_axis in enumerate(keep) if not keep_axis]
        if squeeze_axes:
            # TODO: We could modify the subscripts to avoid the problem?
            warnings.warn(
                "The same einsum subscript is used for a broadcastable and non-broadcastable dimension. "
                "This can result in a suboptimal contraction path."
            )
        return operand.squeeze(squeeze_axes), "".join(names[i] for i in keep_axes)

    einsum_operands = list(tensor_operands)  # So we can pop
    for operand_indices, contracted_names, einstr, _, _ in contraction_list:
        contracted_names = sorted(contracted_names)
        assert len(contracted_names) == len(set(contracted_names)), (
            "The set was needed!"
        )

        input_str, result_names = einstr.split("->")
        input_names = input_str.split(",")

        # switch on the number of operands to be processed in this loop iteration.
        # every case here sets 'operand' and 'names'.
        if len(operand_indices) == 1:
            operand = einsum_operands.pop(operand_indices[0])
            (names,) = input_names
            counts = collections.Counter(names)

            # sum out unique contracted indices with a single reduce-sum
            uniques = [name for name in contracted_names if counts[name] == 1]
            operand, names = sum_uniques(operand, names, uniques)

            # for every repeated index, do a contraction against an identity matrix
            operand, names = sum_repeats(operand, names, counts, result_names)

        elif len(operand_indices) == 2:
            lhs, rhs = map(einsum_operands.pop, operand_indices)
            lhs_names, rhs_names = input_names

            # handle cases where one side of a contracting or batch dimension is 1
            # but its counterpart is not.
            lhs, lhs_names = filter_singleton_dims(lhs, lhs_names, rhs, rhs_names)
            rhs, rhs_names = filter_singleton_dims(rhs, rhs_names, lhs, lhs_names)

            lhs_counts = collections.Counter(lhs_names)
            rhs_counts = collections.Counter(rhs_names)

            # sum out unique contracted indices in lhs and rhs
            lhs_uniques = [
                name
                for name in contracted_names
                if lhs_counts[name] == 1 and rhs_counts[name] == 0
            ]
            lhs, lhs_names = sum_uniques(lhs, lhs_names, lhs_uniques)

            rhs_uniques = [
                name
                for name in contracted_names
                if rhs_counts[name] == 1 and lhs_counts[name] == 0
            ]
            rhs, rhs_names = sum_uniques(rhs, rhs_names, rhs_uniques)

            # for every repeated index, contract against an identity matrix
            lhs, lhs_names = sum_repeats(
                lhs, lhs_names, lhs_counts, result_names + rhs_names
            )
            rhs, rhs_names = sum_repeats(
                rhs, rhs_names, rhs_counts, result_names + lhs_names
            )

            lhs_or_rhs_names = set(lhs_names) | set(rhs_names)
            contracted_names = [x for x in contracted_names if x in lhs_or_rhs_names]
            lhs_and_rhs_names = set(lhs_names) & set(rhs_names)
            batch_names = [x for x in result_names if x in lhs_and_rhs_names]

            if batch_names:
                lhs_batch, rhs_batch = tuple(
                    zip(
                        *[(lhs_names.find(n), rhs_names.find(n)) for n in batch_names],
                        strict=True,
                    )
                )
            else:
                lhs_batch = rhs_batch = ()

            # contract using dot_general
            batch_names_str = "".join(batch_names)
            if contracted_names:
                lhs_cont, rhs_cont = tuple(
                    zip(
                        *[
                            (lhs_names.index(n), rhs_names.index(n))
                            for n in contracted_names
                        ],
                        strict=True,
                    )
                )
            else:
                lhs_cont = rhs_cont = ()
            deleted_names = batch_names_str + "".join(contracted_names)
            remaining_lhs_names = removechars(lhs_names, deleted_names)
            remaining_rhs_names = removechars(rhs_names, deleted_names)
            # Try both orders of lhs and rhs, in the hope that one of them means we
            # don't need an explicit transpose. opt_einsum likes to contract from
            # right to left, so we expect (rhs,lhs) to have the best chance of not
            # needing a transpose.
            names = batch_names_str + remaining_rhs_names + remaining_lhs_names
            if names == result_names:
                operand = _general_dot(
                    (rhs, lhs), (rhs_cont, lhs_cont), (rhs_batch, lhs_batch)
                )
            else:
                names = batch_names_str + remaining_lhs_names + remaining_rhs_names
                operand = _general_dot(
                    (lhs, rhs),
                    axes=(lhs_cont, rhs_cont),
                    batch_axes=(lhs_batch, rhs_batch),
                )
        else:
            raise ValueError(
                f"Each step of einsum must have 1 or 2 operands, got {len(operand_indices)}, {path=}."
            )

        # the resulting 'operand' with axis labels 'names' should be a permutation of the desired result
        assert len(names) == len(result_names) == len(set(names))
        assert set(names) == set(result_names)
        if names != result_names:
            perm = tuple(names.index(name) for name in result_names)
            operand = transpose(operand, perm)
        einsum_operands.append(operand)  # used in next iteration

    [einsum_result] = einsum_operands

    out = Einsum(
        subscripts=subscripts,
        inputs=list(tensor_operands),
        outputs=[einsum_result],
        path=tuple(path),
        optimized=optimized,
    )(*tensor_operands)
    return cast(TensorVariable, out)
