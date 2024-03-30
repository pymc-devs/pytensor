import numpy as np

from pytensor.gradient import grad_undefined
from pytensor.graph.basic import Apply, Constant
from pytensor.graph.op import Op
from pytensor.misc.safe_asarray import _asarray
from pytensor.tensor.basic import arange, as_tensor_variable, switch
from pytensor.tensor.math import eq, ge, mul
from pytensor.tensor.type import TensorType


def _variable_is_none(var):
    return isinstance(var, Constant) and var.data is None


def _check_tensor_is_scalar(var):
    """
    Checks if a tensor variable is scalar, raise ValueError otherwise
    """
    msg = "%(var)s is expected to be 0d tensor, got %(ndim)d"
    if var.ndim != 0:
        raise ValueError(msg % (var, var.ndim))


class SortOp(Op):
    """
    This class is a wrapper for numpy sort function.

    """

    __props__ = ("kind", "order")

    def __init__(self, kind, order=None):
        self.kind = kind
        self.order = order

    def __str__(self):
        return self.__class__.__name__ + f"{{{self.kind}, {self.order}}}"

    def make_node(self, input, axis=-1):
        input = as_tensor_variable(input)
        axis = as_tensor_variable(axis)
        out_type = input.type()
        return Apply(self, [input, axis], [out_type])

    def perform(self, node, inputs, output_storage):
        a = inputs[0]
        axis = inputs[1]
        if axis is not None:
            if axis != int(axis):
                raise ValueError("sort axis must be an integer or None")
            axis = int(axis)
        z = output_storage[0]
        z[0] = np.sort(a, axis, self.kind, self.order)

    def infer_shape(self, fgraph, node, inputs_shapes):
        if _variable_is_none(node.inputs[1]):
            # That means axis = None,
            # So the array is flattened before being sorted
            return [(mul(*inputs_shapes[0]),)]
        # axis should not be None
        # So there should be the same number of dimensions
        # in the input and output
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
        idx = argsort(a, axis, kind=self.kind, order=self.order)
        # rev_idx is the reverse of previous argsort operation
        rev_idx = argsort(idx, axis, kind=self.kind, order=self.order)
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


def sort(a, axis=-1, kind="quicksort", order=None):
    """

    Parameters
    ----------
    a: TensorVariable
        Tensor to be sorted
    axis: TensorVariable
        Axis along which to sort. If None, the array is flattened before
        sorting.
    kind: {'quicksort', 'mergesort', 'heapsort'}, optional
        Sorting algorithm. Default is 'quicksort'.
    order: list, optional
        When `a` is a structured array, this argument specifies which
        fields to compare first, second, and so on. This list does not
        need to include all of the fields.

    Returns
    -------
    array
        A sorted copy of an array.

    """
    if axis is None:
        a = a.flatten()
        axis = 0
    return SortOp(kind, order)(a, axis)


class ArgSortOp(Op):
    """
    This class is a wrapper for numpy argsort function.

    """

    __props__ = ("kind", "order")

    def __init__(self, kind, order=None):
        self.kind = kind
        self.order = order

    def __str__(self):
        return self.__class__.__name__ + f"{{{self.kind}, {self.order}}}"

    def make_node(self, input, axis=-1):
        input = as_tensor_variable(input)
        axis = as_tensor_variable(axis)
        return Apply(
            self,
            [input, axis],
            [TensorType(dtype="int64", shape=input.type.shape)()],
        )

    def perform(self, node, inputs, output_storage):
        a = inputs[0]
        axis = inputs[1]
        if axis is not None:
            if axis != int(axis):
                raise ValueError("sort axis must be an integer or None")
            axis = int(axis)
        z = output_storage[0]
        z[0] = _asarray(
            np.argsort(a, axis, self.kind, self.order), dtype=node.outputs[0].dtype
        )

    def infer_shape(self, fgraph, node, inputs_shapes):
        if _variable_is_none(node.inputs[1]):
            return [(mul(*inputs_shapes[0]),)]
        # axis should not be None, so there should be the same number of
        # dimensions in the input and output
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


def argsort(a, axis=-1, kind="quicksort", order=None):
    """
    Returns the indices that would sort an array.

    Perform an indirect sort along the given axis using the algorithm
    specified by the kind keyword.  It returns an array of indices of
    the same shape as a that index data along the given axis in sorted
    order.

    """
    if axis is None:
        a = a.flatten()
        axis = 0
    return ArgSortOp(kind, order)(a, axis)


def _topk_py_impl(op, x, k, axis, idx_dtype):
    ndim = x.ndim
    assert -ndim <= axis < ndim
    axis %= ndim
    if k == 0:
        raise ValueError("topk: kth cannot be zero")
    elif k > x.shape[axis]:
        raise ValueError(
            f"topk: kth cannot be larger than the size of specified axis {int(axis)}"
        )
    if abs(k) == 1:
        # negative k means min instead of max
        fn_max = [None, np.max, np.min][k]
        fn_argmax = [None, np.argmax, np.argmin][k]
        if not op.return_indices:
            return np.expand_dims(fn_max(x, axis=axis), axis)
        elif op.return_values:
            zi = np.expand_dims(fn_argmax(x, axis=axis), axis)
            idx2 = tuple(
                np.arange(s).reshape((s,) + (1,) * (ndim - i - 1)) if i != axis else zi
                for i, s in enumerate(x.shape)
            )
            zv = x[idx2]
            return zv, zi.astype(idx_dtype)
        else:
            zi = np.expand_dims(fn_argmax(x, axis=axis), axis)
            return zi.astype(idx_dtype)

    if x.shape[axis] == abs(k):
        if not op.return_indices:
            return x.copy()
        else:
            l = axis
            r = ndim - l
            reps = list(x.shape)
            reps[axis] = 1
            zi = np.arange(abs(k), dtype=idx_dtype)
            zi = zi.reshape((1,) * l + (k,) + (1,) * (r - 1))
            zi = np.tile(zi, reps)
            if op.return_values:
                return x.copy(), zi
            else:
                return zi

    idx = [slice(None)] * ndim
    idx[axis] = slice(-k, None) if k > 0 else slice(-k)

    if not op.return_indices:
        zv = np.partition(x, -k, axis=axis)[tuple(idx)]
        return zv
    elif op.return_values:
        zi = np.argpartition(x, -k, axis=axis)[tuple(idx)]
        idx2 = tuple(
            np.arange(s).reshape((s,) + (1,) * (ndim - i - 1)) if i != axis else zi
            for i, s in enumerate(x.shape)
        )
        zv = x[idx2]
        return zv, zi.astype(idx_dtype)
    else:
        zi = np.argpartition(x, -k, axis=axis)[tuple(idx)]
        return zi.astype(idx_dtype)
