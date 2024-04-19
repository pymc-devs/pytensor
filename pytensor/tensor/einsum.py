import collections
from collections.abc import Sequence
from functools import partial, reduce
from itertools import pairwise
from typing import cast

import numpy as np
from numpy.core.numeric import (  # type: ignore
    normalize_axis_index,
    normalize_axis_tuple,
)

from pytensor.compile.builders import OpFromGraph
from pytensor.tensor import TensorLike, vectorize
from pytensor.tensor.basic import (
    arange,
    as_tensor,
    get_vector_length,
    moveaxis,
    stack,
    transpose,
    where,
)
from pytensor.tensor.extra_ops import broadcast_to
from pytensor.tensor.math import and_, eq, tensordot
from pytensor.tensor.shape import shape_padright
from pytensor.tensor.variable import TensorVariable


class Einsum(OpFromGraph):
    """
    Wrapper Op for Einsum graphs
    """

    __props__ = ("subscripts", "optimize")

    def __init__(
        self, *args, subscripts: str, optimize: str | None = "optimal", **kwargs
    ):
        self.subscripts = subscripts
        self.optimize = optimize
        super().__init__(*args, **kwargs)


def _iota(shape: TensorVariable, axis: int) -> TensorVariable:
    len_shape = get_vector_length(shape)
    axis = normalize_axis_index(axis, len_shape)
    values = arange(shape[axis])
    return broadcast_to(shape_padright(values, len_shape - axis - 1), shape)


def _delta(shape, axes: Sequence[int]) -> TensorVariable:
    """This utility function exists for creating Kronecker delta arrays."""
    base_shape = stack([shape[axis] for axis in axes])
    iotas = [_iota(base_shape, i) for i in range(len(axes))]
    eyes = [eq(i1, i2) for i1, i2 in pairwise(iotas)]
    result = reduce(and_, eyes)
    return broadcast_to(result, shape)


def _removechars(s, chars):
    return s.translate(str.maketrans(dict.fromkeys(chars)))


def _general_dot(
    vars: tuple[TensorVariable, TensorVariable],
    axes: Sequence[Sequence[int]],  # Should be length 2,
    batch_axes: Sequence[Sequence[int]],  # Should be length 2,
) -> TensorVariable:
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
    for i, (lhs_axis, rhs_axis) in enumerate(zip(lhs_axes, rhs_axes)):
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

    # TODO: tensordot produces very complicated graphs unnecessarily
    #  In some cases we are just doing elemwise addition after some transpositions
    #  We also have some Blockwise(Reshape) that will slow down things!
    out = vectorize(
        partial(tensordot, axes=[core_lhs_axes, core_rhs_axes]), signature=signature
    )(lhs, rhs)

    # # Reorder batch axes according to the original order of lhs
    # original_lhs_batch_axes, _ = batch_axes
    # final_batch_axes = tuple(np.argsort(original_lhs_batch_axes))
    # new_batch_axes = tuple(range(lhs_n_batch_axes))
    # out = moveaxis(out, new_batch_axes, final_batch_axes)

    return cast(TensorVariable, out)


def einsum(subscripts: str, *operands: "TensorLike") -> TensorVariable:
    """
    Multiplication and summation of tensors using the Einstein summation convention.

    Code adapted from JAX: https://github.com/google/jax/blob/534d32a24d7e1efdef206188bb11ae48e9097092/jax/_src/numpy/lax_numpy.py#L5283

    Parameters
    ----------
    subscripts: str

    operands: sequence of TensorVariable
        Tensors to be multiplied and summed.

    Returns
    -------
    TensorVariable
        The result of the einsum operation.
    """
    # TODO: Is this doing something clever about unknown shapes?
    # contract_path = _poly_einsum_handlers.get(ty, _default_poly_einsum_handler)
    # using einsum_call=True here is an internal api for opt_einsum... sorry

    # TODO: Handle None static shapes
    # TODO: Do we need this as dependency?
    from opt_einsum import contract_path

    operands = cast(tuple[TensorVariable], tuple(map(as_tensor, operands)))
    shapes = [operand.type.shape for operand in operands]

    # TODE: Do fast path at creation time, and optimize only in fast_run
    _, contraction_list = contract_path(
        subscripts,
        *shapes,
        einsum_call=True,
        use_blas=True,
        optimize="optimal",
        shapes=True,
    )

    def sum_uniques(
        operand: TensorVariable, names: str, uniques: list[str]
    ) -> tuple[TensorVariable, str]:
        if uniques:
            axes = [names.index(name) for name in uniques]
            operand = operand.sum(axes)
            names = _removechars(names, uniques)
        return operand, names

    def sum_repeats(
        operand: TensorVariable,
        names: str,
        counts: collections.Counter,
        keep_names: str,
    ) -> tuple[TensorVariable, str]:
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

    # def filter_singleton_dims(operand, names, other_shape, other_names):
    #     eq = core.definitely_equal
    #     keep = [
    #         not eq(operand.shape[i], 1) or j == -1 or eq(other_shape[j], 1)
    #         for i, j in enumerate(map(other_names.find, names))
    #     ]
    #     sqez_axes, keep_axes = partition_list(keep, list(range(operand.ndim)))
    #     return lax.squeeze(operand, sqez_axes), "".join(names[i] for i in keep_axes)

    einsum_operands = list(operands)  # So we can pop
    for operand_indices, contracted_names, einstr, _, _ in contraction_list:
        contracted_names = sorted(contracted_names)
        assert len(contracted_names) == len(
            set(contracted_names)
        ), "The set was needed!"

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
            # lhs, lhs_names = filter_singleton_dims(lhs, lhs_names, shape(rhs),
            #                                        rhs_names)
            # rhs, rhs_names = filter_singleton_dims(rhs, rhs_names, shape(lhs),
            #                                        lhs_names)

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
                    zip(*[(lhs_names.find(n), rhs_names.find(n)) for n in batch_names])
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
                        ]
                    )
                )
            else:
                lhs_cont = rhs_cont = ()
            deleted_names = batch_names_str + "".join(contracted_names)
            remaining_lhs_names = _removechars(lhs_names, deleted_names)
            remaining_rhs_names = _removechars(rhs_names, deleted_names)
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
        inputs=list(operands),
        outputs=[einsum_result],
    )(*operands)
    return cast(TensorVariable, out)
