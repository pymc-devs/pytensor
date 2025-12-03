import typing

import numpy as np

from pytensor.gradient import grad_undefined
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor.basic import arange, as_tensor_variable, switch
from pytensor.tensor.math import eq, ge
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


class SortOp(Op):
    """
    This class is a wrapper for numpy sort function.

    """

    __props__ = ("kind",)

    def __init__(self, kind: KIND):
        self.kind = kind

    def make_node(self, input, axis=-1):
        input = as_tensor_variable(input)
        axis = as_tensor_variable(axis, ndim=0, dtype=int)
        if axis.type.numpy_dtype.kind != "i":
            raise ValueError(
                f"Sort axis must have an integer dtype, got {axis.type.dtype}"
            )
        out_type = input.type()
        return Apply(self, [input, axis], [out_type])

    def perform(self, node, inputs, output_storage):
        a, axis = inputs
        z = output_storage[0]
        z[0] = np.sort(a, axis, self.kind)

    def infer_shape(self, fgraph, node, inputs_shapes):
        assert node.inputs[0].ndim == node.outputs[0].ndim
        assert inputs_shapes[1] == ()
        return [inputs_shapes[0]]

    def grad(self, inputs, output_grads):
        a, axis = inputs
        indices = self.__get_argsort_indices(a, axis)
        inp_grad = output_grads[0][tuple(indices)]
        axis_grad = grad_undefined(
            self,
            1,
            axis,
            "The gradient of sort is not defined "
            "with respect to the integer axes itself",
        )
        return [inp_grad, axis_grad]

    def __get_expanded_dim(self, a, axis, i):
        index_shape = [1] * a.ndim
        index_shape[i] = a.shape[i]
        # it's a way to emulate
        # numpy.ogrid[0: a.shape[0], 0: a.shape[1], 0: a.shape[2]]
        index_val = arange(a.shape[i]).reshape(index_shape)
        return index_val

    def __get_argsort_indices(self, a, axis):
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
        idx = argsort(a, axis, kind=self.kind)
        # rev_idx is the reverse of previous argsort operation
        rev_idx = argsort(idx, axis, kind=self.kind)
        indices = []
        axis_data = switch(ge(axis.data, 0), axis.data, a.ndim + axis.data)
        for i in range(a.ndim):
            index_val = switch(
                eq(i, axis_data),
                rev_idx,
                self.__get_expanded_dim(a, axis, i),
            )
            indices.append(index_val)
        return indices

    """
    def R_op(self, inputs, eval_points):
        # R_op can receive None as eval_points.
        # That mean there is no diferientiable path through that input
        # If this imply that you cannot compute some outputs,
        # return None for those.
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
    """


def sort(
    a, axis=-1, kind: KIND | None = None, order=None, *, stable: bool | None = None
):
    """

    Parameters
    ----------
    a: TensorVariable
        Tensor to be sorted
    axis: TensorVariable
        Axis along which to sort. If None, the array is flattened before
        sorting.
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

    if axis is None:
        a = a.flatten()
        axis = 0
    return SortOp(kind)(a, axis)


class ArgSortOp(Op):
    """
    This class is a wrapper for numpy argsort function.

    """

    __props__ = ("kind",)

    def __init__(self, kind: KIND):
        self.kind = kind

    def make_node(self, input, axis=-1):
        input = as_tensor_variable(input)
        axis = as_tensor_variable(axis, ndim=0, dtype=int)
        if axis.type.numpy_dtype.kind != "i":
            raise ValueError(
                f"ArgSort axis must have an integer dtype, got {axis.type.dtype}"
            )
        return Apply(
            self,
            [input, axis],
            [TensorType(dtype="int64", shape=input.type.shape)()],
        )

    def perform(self, node, inputs, output_storage):
        a, axis = inputs
        z = output_storage[0]
        z[0] = np.asarray(
            np.argsort(a, axis, self.kind),
            dtype=node.outputs[0].dtype,
        )

    def infer_shape(self, fgraph, node, inputs_shapes):
        assert node.inputs[0].ndim == node.outputs[0].ndim
        assert inputs_shapes[1] == ()
        return [inputs_shapes[0]]

    def grad(self, inputs, output_grads):
        # No grad defined for integers.
        inp, axis = inputs
        inp_grad = inp.zeros_like()
        axis_grad = grad_undefined(
            self,
            1,
            axis,
            "argsort is not defined for non-integer axes so"
            " argsort(x, axis+eps) is undefined",
        )
        return [inp_grad, axis_grad]

    """
    def R_op(self, inputs, eval_points):
        # R_op can receive None as eval_points.
        # That mean there is no diferientiable path through that input
        # If this imply that you cannot compute some outputs,
        # return None for those.
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
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
    if axis is None:
        a = a.flatten()
        axis = 0
    return ArgSortOp(kind)(a, axis)
