import typing

import numpy as np
from numpy.lib.array_utils import normalize_axis_index

from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor.basic import _validate_axis_argument, arange, as_tensor_variable
from pytensor.tensor.type import TensorType


KIND = typing.Literal["quicksort", "mergesort", "heapsort", "stable"]
KIND_VALUES = typing.get_args(KIND)


def _parse_sort_args(kind: KIND | None, order, stable: bool | None) -> KIND:
    if order is not None:
        raise ValueError("The order argument is not applicable to PyTensor graphs")
    if stable is not None and kind is not None:
        raise ValueError("kind and stable cannot be set at the same time")
    if stable:
        kind = "stable"
    elif kind is None:
        kind = "quicksort"
    if kind not in KIND_VALUES:
        raise ValueError(f"kind must be one of {KIND_VALUES}, got {kind}")
    return kind


def _validate_sort_axis(axis, op_name: str) -> int:
    int_axis: int = _validate_axis_argument(axis, op_name)
    if int_axis < 0:
        raise ValueError(f"{op_name} axis must be non-negative, got {int_axis}.")
    return int_axis


class SortOp(Op):
    """
    This class is a wrapper for numpy sort function.

    """

    __props__ = ("kind", "axis")

    def __init__(self, kind: KIND, axis: int):
        self.kind = kind
        self.axis = _validate_sort_axis(axis, "Sort")

    def make_node(self, input):
        input = as_tensor_variable(input)
        if self.axis >= input.type.ndim:
            raise np.exceptions.AxisError(self.axis, input.type.ndim)
        out_type = input.type()
        return Apply(self, [input], [out_type])

    def perform(self, node, inputs, output_storage):
        [a] = inputs
        z = output_storage[0]
        z[0] = np.sort(a, self.axis, self.kind)

    def infer_shape(self, node, inputs_shapes):
        return [inputs_shapes[0]]

    def pullback(self, inputs, outputs, output_grads):
        [a] = inputs
        indices = self.__get_argsort_indices(a)
        return [output_grads[0][tuple(indices)]]

    def __get_expanded_dim(self, a, i):
        index_shape = [1] * a.ndim
        index_shape[i] = a.shape[i]
        # it's a way to emulate
        # numpy.ogrid[0: a.shape[0], 0: a.shape[1], 0: a.shape[2]]
        index_val = arange(a.shape[i]).reshape(index_shape)
        return index_val

    def __get_argsort_indices(self, a):
        """
        Calculates indices which can be used to reverse sorting operation of
        "a" tensor along "axis".

        Returns
        -------
        1d array if axis is None
        list of length len(a.shape) otherwise

        """

        # The goal is to get gradient wrt input from gradient
        # wrt sort(input, axis)
        idx = argsort(a, self.axis, kind=self.kind)
        # rev_idx is the reverse of previous argsort operation
        rev_idx = argsort(idx, self.axis, kind=self.kind)
        return [
            rev_idx if i == self.axis else self.__get_expanded_dim(a, i)
            for i in range(a.ndim)
        ]

    """
    def pushforward(self, inputs, outputs, eval_points):
        # pushforward can receive DisconnectedType as eval_points.
        # That mean there is no diferientiable path through that input
        # If this imply that you cannot compute some outputs,
        # return disconnected_type() for those.
        if isinstance(eval_points[0].type, DisconnectedType):
            return list(eval_points)
        return self.pullback(inputs, outputs, eval_points)
    """


def sort(
    a, axis=-1, kind: KIND | None = None, order=None, *, stable: bool | None = None
):
    """

    Parameters
    ----------
    a: TensorVariable
        Tensor to be sorted
    axis: int, optional
        Axis along which to sort. If None, the array is flattened before
        sorting. Must be a constant.
    kind: {'quicksort', 'mergesort', 'heapsort' 'stable'}, optional
        Sorting algorithm. Default is 'quicksort' unless stable is defined.
    order: list, optional
        For compatibility with numpy sort signature. Cannot be specified.
    stable: bool, optional
        Same as specifying kind = 'stable'. Cannot be specified at the same time as kind

    Returns
    -------
    array
        A sorted copy of an array.

    """
    kind = _parse_sort_args(kind, order, stable)
    a = as_tensor_variable(a)
    if axis is None:
        a = a.flatten()
        axis = 0
    axis = normalize_axis_index(_validate_axis_argument(axis, "sort"), a.type.ndim)
    return SortOp(kind, axis)(a)


class ArgSortOp(Op):
    """
    This class is a wrapper for numpy argsort function.

    """

    __props__ = ("kind", "axis")

    def __init__(self, kind: KIND, axis: int):
        self.kind = kind
        self.axis = _validate_sort_axis(axis, "ArgSort")

    def make_node(self, input):
        input = as_tensor_variable(input)
        if self.axis >= input.type.ndim:
            raise np.exceptions.AxisError(self.axis, input.type.ndim)
        return Apply(
            self,
            [input],
            [TensorType(dtype="int64", shape=input.type.shape)()],
        )

    def perform(self, node, inputs, output_storage):
        [a] = inputs
        z = output_storage[0]
        z[0] = np.asarray(
            np.argsort(a, self.axis, self.kind),
            dtype=node.outputs[0].dtype,
        )

    def infer_shape(self, node, inputs_shapes):
        return [inputs_shapes[0]]

    def pullback(self, inputs, outputs, output_grads):
        # No grad defined for integers.
        [inp] = inputs
        return [inp.zeros_like()]

    """
    def pushforward(self, inputs, outputs, eval_points):
        # pushforward can receive DisconnectedType as eval_points.
        # That mean there is no diferientiable path through that input
        # If this imply that you cannot compute some outputs,
        # return disconnected_type() for those.
        if isinstance(eval_points[0].type, DisconnectedType):
            return list(eval_points)
        return self.pullback(inputs, outputs, eval_points)
    """


def argsort(
    a, axis=-1, kind: KIND | None = None, order=None, stable: bool | None = None
):
    """
    Returns the indices that would sort an array.

    Perform an indirect sort along the given axis using the algorithm
    specified by the kind keyword.  It returns an array of indices of
    the same shape as a that index data along the given axis in sorted
    order.

    """
    kind = _parse_sort_args(kind, order, stable)
    a = as_tensor_variable(a)
    if axis is None:
        a = a.flatten()
        axis = 0
    axis = normalize_axis_index(_validate_axis_argument(axis, "argsort"), a.type.ndim)
    return ArgSortOp(kind, axis)(a)
