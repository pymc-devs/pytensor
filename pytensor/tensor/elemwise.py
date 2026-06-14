from collections.abc import Sequence
from copy import copy
from typing import Literal

import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple

import pytensor.tensor.basic
from pytensor.configdefaults import config
from pytensor.gradient import DisconnectedType, disconnected_type
from pytensor.graph.basic import Apply
from pytensor.graph.null_type import NullType
from pytensor.graph.op import Op
from pytensor.graph.replace import _vectorize_node, _vectorize_not_needed
from pytensor.misc.frozendict import frozendict
from pytensor.printing import Printer, pprint
from pytensor.scalar import get_scalar_type
from pytensor.scalar.basic import upcast
from pytensor.tensor import get_vector_length
from pytensor.tensor.basic import _get_vector_length, as_tensor_variable
from pytensor.tensor.type import (
    TensorType,
    continuous_dtypes,
    discrete_dtypes,
    float_dtypes,
)
from pytensor.tensor.utils import (
    broadcast_static_dim_lengths,
    import_func_from_string,
    normalize_reduce_axis,
)
from pytensor.tensor.variable import TensorVariable


class DimShuffle(Op):
    """
    Allows to reorder the dimensions of a tensor or insert or remove
    broadcastable dimensions.

    In the following examples, 'x' means that we insert a broadcastable
    dimension and a numerical index represents the dimension of the same
    rank in the tensor passed to perform.

    Parameters
    ----------
    input_ndim
        The expected number of dimension of the input
    new_order
        A list representing the relationship between the input's
        dimensions and the output's dimensions. Each element of the
        list can either be an index or 'x'. Indices must be encoded
        as python integers, not pytensor symbolic integers.
        Missing indexes correspond to drop dimensions.

    Notes
    -----
    If `j = new_order[i]` is an index, the output's ith dimension
    will be the input's jth dimension.
    If `new_order[i]` is `x`, the output's ith dimension will
    be 1 and broadcast operations will be allowed to do broadcasting
    over that dimension.

    If `input.type.shape[i] != 1` then `i` must be found in `new_order`.
    Broadcastable dimensions, on the other hand, can be discarded.

    .. code-block:: python

        DimShuffle(input_ndim=3, new_order=["x", 2, "x", 0, 1])

    This `Op` will only work on 3d tensors.
    The first dimension of the output will be broadcastable,
    then we will have the third dimension of the input tensor as
    the second of the resulting tensor, etc. If the tensor has
    shape (20, 30, 40), the resulting tensor will have dimensions
    (1, 40, 1, 20, 30). (AxBxC tensor is mapped to 1xCx1xAxB tensor)

    .. code-block:: python

        DimShuffle(input_ndim=2, new_order=[1])

    This `Op` will only work on 2d tensors with the first dimension broadcastable.
    The second dimension of the input tensor will be the first dimension of the resulting tensor.
    If the tensor has shape (1, 20), the resulting tensor will have shape (20, ).

    Examples
    --------
    .. code-block:: python

        DimShuffle(input_ndim=0, new_order=["x"])  # make a 0d (scalar) into a 1d vector
        DimShuffle(input_ndim=2, new_order=[0, 1])  # identity
        DimShuffle(input_ndim=2, new_order=[1, 0])  # transposition
        # Make a row out of a 1d vector (N to 1xN)
        DimShuffle(input_ndim=1, new_order=["x", 0])
        # Make a colum out of a 1d vector (N to Nx1)
        DimShuffle(input_ndim=1, new_order=[0, "x"])
        DimShuffle(input_ndim=3, new_order=[2, 0, 1])  # AxBxC to CxAxB
        DimShuffle(input_ndim=2, new_order=[0, "x", 1])  # AxB to Ax1xB
        DimShuffle(input_ndim=2, new_order=[1, "x", 0])  # AxB to Bx1xA

    Notes
    -----
    The python implementation of this Op combines numpy.transpose for reordering of the dimensions
    and numpy.reshape for subtracting and adding broadcastable dimensions.
    """

    _f16_ok = True
    check_input = False
    __props__ = ("input_ndim", "new_order")
    view_map = {0: [0]}

    def __init__(self, *, input_ndim: int, new_order: Sequence[int | Literal["x"]]):
        if not isinstance(input_ndim, int):
            raise TypeError(f"input_ndim must be an integer, got {type(int)}")

        self.input_ndim = input_ndim
        self.new_order = new_order = tuple(new_order)
        self._new_order = [(-1 if x == "x" else x) for x in self.new_order]

        for i, j in enumerate(new_order):
            if j != "x":
                if not isinstance(j, int | np.integer):
                    raise TypeError(
                        "DimShuffle indices must be Python ints; got "
                        f"{j} of type {type(j)}."
                    )
                if j >= input_ndim:
                    raise ValueError(
                        f"new_order[{i}] is {j}, but the input only has "
                        f"{input_ndim} axes."
                    )
                if j in new_order[(i + 1) :]:
                    raise ValueError(
                        "The same input dimension may not appear "
                        f"twice in the list of output dimensions: {new_order}"
                    )

        # Tuple of the original dimensions that we keep
        self.shuffle = tuple(x for x in new_order if x != "x")
        # Tuple of input dimensions to drop
        self.drop = drop = tuple(i for i in range(input_ndim) if i not in new_order)
        # tuple of dimensions of the output that are broadcastable and were not in the original input
        self.augment = augment = tuple(
            sorted(i for i, x in enumerate(new_order) if x == "x")
        )
        n_augment = len(self.augment)

        # Used by perform
        self._transposition = self.shuffle + drop

        # Classify the type of dimshuffle for rewrite purposes
        dims_are_shuffled = tuple(sorted(self.shuffle)) != self.shuffle
        self.is_squeeze = drop and not augment and not dims_are_shuffled
        self.is_expand_dims = is_expand_dims = (
            not drop and augment and not dims_are_shuffled
        )
        self.is_left_expand_dims = is_expand_dims and new_order[n_augment:] == tuple(
            range(input_ndim)
        )
        self.is_right_expand_dims = is_expand_dims and new_order[:input_ndim] == tuple(
            range(input_ndim)
        )
        self.is_transpose = not drop and not augment and dims_are_shuffled
        self.is_left_expanded_matrix_transpose = is_left_expanded_matrix_transpose = (
            dims_are_shuffled
            and new_order[n_augment:]
            == (*range(input_ndim - 2), input_ndim - 1, input_ndim - 2)
        )
        self.is_matrix_transpose = not augment and is_left_expanded_matrix_transpose

    def __setstate__(self, state):
        # Old pickles carry ExternalCOp attributes (func_files, ...); drop them,
        # the C implementation now comes from the dispatch registry.
        for key in ("func_files", "func_codes", "func_name", "code_sections"):
            state.pop(key, None)
        self.__dict__.update(state)

    def make_node(self, inp):
        input = as_tensor_variable(inp)
        if input.type.ndim != self.input_ndim:
            raise TypeError(
                "The number of dimensions of the input is incorrect for this op. "
                f"Expected {self.input_ndim}, got {input.type.ndim}."
            )

        input_static_shape = input.type.shape

        # Runtime check for invalid drop
        for d in self.drop:
            if input_static_shape[d] not in (1, None):
                raise TypeError(
                    f"Input dropped dimension {d} must have length 1 but has {input_static_shape[d]}"
                )

        out_static_shape = []
        for dim_idx in self.new_order:
            if dim_idx == "x":
                out_static_shape.append(1)
            else:
                out_static_shape.append(input_static_shape[dim_idx])

        output = TensorType(dtype=input.type.dtype, shape=out_static_shape)()

        return Apply(self, [input], [output])

    def __str__(self):
        if self.is_matrix_transpose:
            return "MatrixTranspose"
        if self.is_expand_dims:
            if len(self.augment) == 1:
                return f"ExpandDims{{axis={self.augment[0]}}}"
            return f"ExpandDims{{axes={self.augment}}}"
        if self.is_squeeze:
            if len(self.drop) == 1:
                return f"Squeeze{{axis={self.drop[0]}}}"
            return f"Squeeze{{axes={self.drop}}}"
        if self.is_transpose:
            return f"Transpose{{axes={self.shuffle}}}"
        return f"DimShuffle{{order=[{','.join(map(str, self.new_order))}]}}"

    def perform(self, node, inp, out):
        (res,) = inp

        # This C-like impl is very slow in Python compared to transpose+reshape
        # new_order = self._new_order
        # old_shape = inp.shape
        # old_strides = inp.strides
        # res = as_strided(
        #     shape = [1 if i == -1 else old_shape[i] for i in new_order],
        #     strides=[0 if i == -1 else old_strides[i] for i in new_order],
        # )

        # Put dropped axis at end
        res = res.transpose(self._transposition)

        # Define new shape without dropped axis and including new ones
        new_shape = list(res.shape[: len(self.shuffle)])
        for augm in self.augment:
            new_shape.insert(augm, 1)
        out[0][0] = res.reshape(new_shape)

    def infer_shape(self, fgraph, node, shapes):
        (ishp,) = shapes
        # transpose
        rval = [ishp[i] for i in self.shuffle]

        # augment
        for augm in self.augment:
            rval.insert(augm, 1)
        return [rval]

    def pushforward(self, inputs, outputs, tangents):
        if any(isinstance(t.type, DisconnectedType) for t in tangents):
            return [disconnected_type()]
        return self(*tangents, return_list=True)

    def pullback(self, inp, outputs, grads):
        (x,) = inp
        (gz,) = grads
        grad_order = ["x"] * x.type.ndim
        for i, v in enumerate(self.new_order):
            if v != "x":
                grad_order[v] = i

        if x.type.dtype in discrete_dtypes:
            return [x.zeros_like(dtype=config.floatX)]
        else:
            return [gz.dimshuffle(grad_order)]


class DimShufflePrinter(Printer):
    def __p(self, new_order, pstate, r):
        if new_order != () and new_order[0] == "x":
            return f"{self.__p(new_order[1:], pstate, r)}"
        #            return "[%s]" % self.__p(new_order[1:], pstate, r)
        if list(new_order) == list(range(r.type.ndim)):
            return pstate.pprinter.process(r)
        if list(new_order) == list(reversed(range(r.type.ndim))):
            return f"{pstate.pprinter.process(r)}.T"
        return f"DimShuffle{{{', '.join(str(o) for o in new_order)}}}({pstate.pprinter.process(r)})"

    def process(self, r, pstate):
        if r.owner is None:
            raise TypeError("Can only print DimShuffle.")
        elif isinstance(r.owner.op, DimShuffle):
            ord = r.owner.op.new_order
            return self.__p(ord, pstate, r.owner.inputs[0])
        else:
            raise TypeError("Can only print DimShuffle.")


pprint.assign(DimShuffle, DimShufflePrinter())


class Elemwise(Op):
    """Generalizes a scalar `Op` to tensors.

    All the inputs must have the same number of dimensions. When the
    `Op` is performed, for each dimension, each input's size for that
    dimension must be the same. As a special case, it can also be one
    but only if the input's `broadcastable` flag is ``True`` for that
    dimension. In that case, the tensor is (virtually) replicated
    along that dimension to match the size of the others.

    The dtypes of the outputs mirror those of the scalar `Op` that is
    being generalized to tensors. In particular, if the calculations
    for an output are done in-place on an input, the output type must
    be the same as the corresponding input type (see the doc of
    `ScalarOp` to get help about controlling the output type)

    Notes
    -----
    -``Elemwise(add)``: represents ``+`` on tensors ``x + y``
    -``Elemwise(add, {0 : 0})``: represents the ``+=`` operation ``x += y``
    -``Elemwise(add, {0 : 1})``: represents ``+=`` on the second argument ``y += x``
    -``Elemwise(mul)(np.random.random((10, 5)), np.random.random((1, 5)))``:
    the second input is completed along the first dimension to match the first input
    -``Elemwise(true_div)(np.random.random(10, 5), np.random.random(10, 1))``: same but along the
    second dimension
    -``Elemwise(int_div)(np.random.random((1, 5)), np.random.random((10, 1)))``:
    the output has size ``(10, 5)``.
    -``Elemwise(log)(np.random.random((3, 4, 5)))``

    """

    __props__ = ("scalar_op", "inplace_pattern")
    # Allow pattern matching on scalar_op positionally
    __match_args__ = ("scalar_op",)

    def __init__(
        self, scalar_op, inplace_pattern=None, name=None, nfunc_spec=None, openmp=None
    ):
        """

        Parameters
        ----------
        scalar_op
            An instance of a subclass of `ScalarOp` which works uniquely
            on scalars.
        inplace_pattern
            A dictionary that maps the index of an output to the
            index of an input so the output is calculated inplace using
            the input's storage. (Just like `Op.destroy_map`, but without the lists.)
        nfunc_spec
            Either ``None`` or a tuple of three elements, ``(nfunc_name, nin,
            nout)`` such that ``getattr(numpy, nfunc_name)`` implements this
            operation, takes ``nin``-many inputs and ``nout``-many outputs.  Note
            that ``nin`` cannot always be inferred from the scalar `Op`'s own
            ``nin`` field, because that value is sometimes zero (meaning a variable
            number of inputs), whereas the NumPy function may not have var-args.

        """
        assert not isinstance(scalar_op, type(self))
        if inplace_pattern is None:
            inplace_pattern = frozendict({})
        self.name = name
        self.scalar_op = scalar_op
        self.inplace_pattern = inplace_pattern
        self.destroy_map = {o: [i] for o, i in self.inplace_pattern.items()}

        if nfunc_spec is None:
            nfunc_spec = getattr(scalar_op, "nfunc_spec", None)
        self.nfunc_spec = nfunc_spec
        self.__setstate__(self.__dict__)
        self.openmp = config.openmp if openmp is None else openmp

    def __getstate__(self):
        d = copy(self.__dict__)
        d.pop("ufunc")
        d.pop("nfunc")
        d.pop("__epydoc_asRoutine", None)
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, "openmp"):
            self.openmp = False
        self.ufunc = None
        self.nfunc = None
        self.inplace_pattern = frozendict(self.inplace_pattern)

    def make_scalar_node(self, *inputs):
        """Create a scalar Apply node matching the dtypes of tensor inputs.

        Used by get_output_info, grad, and backend dispatchers to obtain
        the scalar-level graph corresponding to this Elemwise operation.
        """
        return self.scalar_op.make_node(
            *[get_scalar_type(dtype=i.type.dtype).make_variable() for i in inputs]
        )

    def get_output_info(self, *inputs):
        """Return the outputs dtype and broadcastable pattern and the
        dimshuffled inputs.

        """
        shadow = self.make_scalar_node(*inputs)

        target_length = max(input.type.ndim for input in inputs)

        args = []
        for input in inputs:
            length = input.type.ndim
            difference = target_length - length
            if not difference:
                args.append(input)
            else:
                args.append(input.dimshuffle(["x"] * difference + list(range(length))))
        inputs = args

        # HERE: all the broadcast dims have the same length now

        # cleverness: we iterate over the first, second, third broadcast flag
        # of all inputs in parallel... the all() gives us each output
        # broadcastable bit in turn.

        # it is multiplied by nout because Elemwise supports multiple outputs
        # (nout of them)
        try:
            out_shapes = [
                [
                    broadcast_static_dim_lengths(shape)
                    for shape in zip(*[inp.type.shape for inp in inputs], strict=True)
                ]
            ] * shadow.nout
        except ValueError:
            raise ValueError(
                f"Incompatible Elemwise input shapes {[inp.type.shape for inp in inputs]}"
            )

        # inplace_pattern maps output idx -> input idx
        inplace_pattern = self.inplace_pattern
        if inplace_pattern:
            for overwriter, overwritten in inplace_pattern.items():
                for out_s, in_s in zip(
                    out_shapes[overwriter], inputs[overwritten].type.shape, strict=True
                ):
                    if in_s == 1 and out_s != 1:
                        raise ValueError(
                            "Operation cannot be done inplace on an input "
                            "with broadcasted dimensions."
                        )

        out_dtypes = [o.type.dtype for o in shadow.outputs]
        if any(
            inputs[i].type.dtype != out_dtypes[o] for o, i in inplace_pattern.items()
        ):
            raise TypeError(
                (
                    "Cannot do an inplace operation on incompatible data types.",
                    ([i.type.dtype for i in inputs], out_dtypes, inplace_pattern),
                )
            )
        assert len(out_dtypes) == len(out_shapes)
        return out_dtypes, out_shapes, inputs

    def make_node(self, *inputs):
        """
        If the inputs have different number of dimensions, their shape
        is left-completed to the greatest number of dimensions with 1s
        using DimShuffle.
        """
        inputs = [as_tensor_variable(i) for i in inputs]
        out_dtypes, out_shapes, inputs = self.get_output_info(*inputs)
        outputs = [
            TensorType(dtype=dtype, shape=shape)()
            for dtype, shape in zip(out_dtypes, out_shapes, strict=True)
        ]
        return Apply(self, inputs, outputs)

    def __str__(self):
        if self.name:
            return self.name
        return str(self.scalar_op).capitalize()

    def pushforward(self, inputs, outputs, tangents):
        outs = self(*inputs, return_list=True)
        rval = [disconnected_type() for x in outs]
        # For each output
        for idx, out in enumerate(outs):
            # make such that _bgrads computes only the gradients of the
            # current output on the inputs ( and not all outputs)
            ograds = [x.zeros_like() for x in outs]
            ograds[idx] = pytensor.tensor.basic.ones_like(out)

            bgrads = self._bgrad(inputs, outs, ograds)
            rop_out = None

            for jdx, (inp, eval_point) in enumerate(zip(inputs, tangents, strict=True)):
                # if None, then we can just ignore this branch ..
                # what we do is to assume that for any non-differentiable
                # branch, the gradient is actually 0, which I think is not
                # the right thing to do .. have to talk to Ian and James
                # about it

                if bgrads[jdx] is None or isinstance(
                    bgrads[jdx].type, DisconnectedType
                ):
                    pass
                elif not isinstance(eval_point.type, DisconnectedType):
                    if rop_out is None:
                        rop_out = bgrads[jdx] * eval_point
                    else:
                        rop_out = rop_out + bgrads[jdx] * eval_point

            rval[idx] = disconnected_type() if rop_out is None else rop_out

        return rval

    def connection_pattern(self, node):
        if hasattr(self.scalar_op, "connection_pattern"):
            return self.scalar_op.connection_pattern(node)

        return [[True for output in node.outputs] for ipt in node.inputs]

    def pullback(self, inputs, outs, ograds):
        from pytensor.tensor.math import sum as pt_sum

        # Compute grad with respect to broadcasted input
        rval = self._bgrad(inputs, outs, ograds)

        # sum out the broadcasted dimensions
        for i, ipt in enumerate(inputs):
            if isinstance(rval[i].type, NullType | DisconnectedType):
                continue

            # List of all the dimensions that are broadcastable for input[i] so
            # we can sum over them
            # TODO: only count dimensions that were effectively broadcasted
            to_sum = [
                j
                for j, in_s in enumerate(ipt.type.shape)
                if in_s == 1 and outs[0].type.shape[j] != 1
            ]

            if to_sum:
                sr = pt_sum(rval[i], axis=to_sum, keepdims=True)
                rval[i] = sr

        return rval

    def _bgrad(self, inputs, outputs, ograds):
        # returns grad, with respect to broadcasted versions of inputs

        def as_scalar(t):
            if isinstance(t.type, NullType | DisconnectedType):
                return t
            return get_scalar_type(t.type.dtype)()

        scalar_inputs = list(map(as_scalar, inputs))
        scalar_ograds = list(map(as_scalar, ograds))
        scalar_outputs = self.scalar_op.make_node(
            *[get_scalar_type(dtype=i.type.dtype).make_variable() for i in inputs]
        ).outputs
        scalar_igrads = self.scalar_op.pullback(
            scalar_inputs, scalar_outputs, scalar_ograds
        )
        for igrad in scalar_igrads:
            assert igrad is not None, self.scalar_op

        if not isinstance(scalar_igrads, list | tuple):
            raise TypeError(
                f"{self.scalar_op!s}.grad returned {type(scalar_igrads)!s} instead of list or tuple"
            )

        nd = inputs[0].type.ndim  # this is the same for everyone

        def transform(r):
            # From a graph of ScalarOps, make a graph of Broadcast ops.
            if isinstance(r.type, NullType | DisconnectedType):
                return r
            if r in scalar_inputs:
                return inputs[scalar_inputs.index(r)]
            if r in scalar_outputs:
                return outputs[scalar_outputs.index(r)]
            if r in scalar_ograds:
                return ograds[scalar_ograds.index(r)]
            node = r.owner
            if node is None:
                # the gradient contains a constant, translate it as
                # an equivalent TensorType of size 1 and proper number of
                # dimensions
                res = pytensor.tensor.basic.constant(
                    np.asarray(r.data), dtype=r.type.dtype
                )
                return res.dimshuffle(["x"] * nd)

            new_r = Elemwise(node.op, {})(*[transform(ipt) for ipt in node.inputs])
            if isinstance(new_r, list | tuple):
                # Scalar Op with multiple outputs
                new_r = new_r[r.owner.outputs.index(r)]
            return new_r

        ret = []
        for scalar_igrad, ipt in zip(scalar_igrads, inputs, strict=True):
            if scalar_igrad is None:
                # undefined gradient
                ret.append(None)
                continue
            ret.append(transform(scalar_igrad))

        return ret

    def prepare_node(self, node, storage_map, compute_map, impl):
        # Postpone the ufunc building to the last minutes due to:
        # - NumPy ufunc support only up to 32 operands (inputs and outputs)
        #   But our c code support more.
        # - nfunc is reused for scipy and scipy is optional
        if (len(node.inputs) + len(node.outputs)) > 32 and impl == "py":
            impl = "c"

        if getattr(self, "nfunc_spec", None) and impl != "c":
            self.nfunc = import_func_from_string(self.nfunc_spec[0])

        if (
            (len(node.inputs) + len(node.outputs)) <= 32
            and (self.nfunc is None or self.scalar_op.nin != len(node.inputs))
            and self.ufunc is None
            and impl == "py"
        ):
            ufunc = np.frompyfunc(
                self.scalar_op.impl, len(node.inputs), self.scalar_op.nout
            )
            if self.scalar_op.nin > 0:
                # We can reuse it for many nodes
                self.ufunc = ufunc
            else:
                node.tag.ufunc = ufunc

        # Numpy ufuncs will sometimes perform operations in
        # float16, in particular when the input is int8.
        # This is not something that we want, and we do not
        # do it in the C code, so we specify that the computation
        # should be carried out in the returned dtype.
        # This is done via the "sig" kwarg of the ufunc, its value
        # should be something like "ff->f", where the characters
        # represent the dtype of the inputs and outputs.

        # NumPy 1.10.1 raise an error when giving the signature
        # when the input is complex. So add it only when inputs is int.
        out_dtype = node.outputs[0].dtype
        if (
            out_dtype in float_dtypes
            and isinstance(self.nfunc, np.ufunc)
            and node.inputs[0].dtype in discrete_dtypes
        ):
            char = np.dtype(out_dtype).char
            sig = char * node.nin + "->" + char * node.nout
            node.tag.sig = sig
        node.tag.fake_node = Apply(
            self.scalar_op,
            [
                get_scalar_type(dtype=input.type.dtype).make_variable()
                for input in node.inputs
            ],
            [
                get_scalar_type(dtype=output.type.dtype).make_variable()
                for output in node.outputs
            ],
        )

        self.scalar_op.prepare_node(node.tag.fake_node, None, None, impl)

    def perform(self, node, inputs, output_storage):
        if (len(node.inputs) + len(node.outputs)) > 32:
            # Some versions of NumPy will segfault, other will raise a
            # ValueError, if the number of operands in an ufunc is more than 32.
            # In that case, the C version should be used, or Elemwise fusion
            # should be disabled.
            # FIXME: This no longer calls the C implementation!
            super().perform(node, inputs, output_storage)

        self._check_runtime_broadcast(node, inputs)

        ufunc_args = inputs
        ufunc_kwargs = {}
        # We supported in the past calling manually op.perform.
        # To keep that support we need to sometimes call self.prepare_node
        if self.nfunc is None and self.ufunc is None:
            self.prepare_node(node, None, None, "py")
        if self.nfunc and len(inputs) == self.nfunc_spec[1]:
            ufunc = self.nfunc
            nout = self.nfunc_spec[2]
            if hasattr(node.tag, "sig"):
                ufunc_kwargs["sig"] = node.tag.sig
            # Unfortunately, the else case does not allow us to
            # directly feed the destination arguments to the nfunc
            # since it sometimes requires resizing. Doing this
            # optimization is probably not worth the effort, since we
            # should normally run the C version of the Op.
        else:
            # the second calling form is used because in certain versions of
            # numpy the first (faster) version leads to segfaults
            if self.ufunc:
                ufunc = self.ufunc
            elif not hasattr(node.tag, "ufunc"):
                # It happen that make_thunk isn't called, like in
                # get_underlying_scalar_constant_value
                self.prepare_node(node, None, None, "py")
                # prepare_node will add ufunc to self or the tag
                # depending if we can reuse it or not. So we need to
                # test both again.
                if self.ufunc:
                    ufunc = self.ufunc
                else:
                    ufunc = node.tag.ufunc
            else:
                ufunc = node.tag.ufunc

            nout = ufunc.nout

        with np.errstate(all="ignore"):
            variables = ufunc(*ufunc_args, **ufunc_kwargs)

        if nout == 1:
            variables = [variables]

        # zip strict not specified because we are in a hot loop
        for i, (variable, storage, nout) in enumerate(
            zip(variables, output_storage, node.outputs)
        ):
            storage[0] = variable = np.asarray(variable, dtype=nout.dtype)

            if i in self.inplace_pattern:
                odat = inputs[self.inplace_pattern[i]]
                odat[...] = variable
                storage[0] = odat

            # numpy.real return a view!
            if not variable.flags.owndata:
                storage[0] = variable.copy()

    @staticmethod
    def _check_runtime_broadcast(node, inputs):
        # zip strict not specified because we are in a hot loop
        for dims_and_bcast in zip(
            *[
                zip(input.shape, sinput.type.broadcastable)
                for input, sinput in zip(inputs, node.inputs)
            ],
            strict=False,
        ):
            if any(d != 1 for d, _ in dims_and_bcast) and (1, False) in dims_and_bcast:
                raise ValueError(
                    "Runtime broadcasting not allowed. "
                    "At least one input has a distinct dimension length of 1, but was not marked as broadcastable.\n"
                    "If broadcasting was intended, use `specify_broadcastable` on the relevant input."
                )

    def infer_shape(self, fgraph, node, i_shapes) -> list[tuple[TensorVariable, ...]]:
        from pytensor.tensor.extra_ops import broadcast_shape

        out_shape = broadcast_shape(*i_shapes, arrays_are_shapes=True)
        return [tuple(as_tensor_variable(s) for s in out_shape)] * len(node.outputs)

    def outer(self, x, y):
        from pytensor.tensor.basic import expand_dims

        if self.scalar_op.nin not in (-1, 2):
            raise NotImplementedError("outer is only available for binary operators")

        x_ = expand_dims(x, tuple(range(-y.ndim, 0)))
        y_ = expand_dims(y, tuple(range(x.ndim)))
        return self(x_, y_)


class CAReduce(Op):
    """Reduces a scalar operation along specified axes.

    The scalar op should be both commutative and associative.

    `CAReduce` = Commutative Associative Reduce.

    The output will have the same shape as the input minus the reduced
    dimensions. It will contain the variable of accumulating all values
    over the reduced dimensions using the specified scalar `Op`.

    Notes
    -----
    .. code-block:: python

        CAReduce(add)  # sum (ie, acts like the numpy sum operation)
        CAReduce(mul)  # product
        CAReduce(maximum)  # max
        CAReduce(minimum)  # min
        CAReduce(or_)  # any # not lazy
        CAReduce(and_)  # all # not lazy
        CAReduce(xor)  # a bit at 1 tell that there was an odd number of
        # bit at that position that where 1. 0 it was an
        # even number ...

    In order to (eventually) optimize memory usage patterns,
    `CAReduce` makes zero guarantees on the order in which it
    iterates over the dimensions and the elements of the
    array(s). Therefore, to ensure consistent variables, the scalar
    operation represented by the reduction must be both commutative
    and associative (eg add, multiply, maximum, binary or/and/xor - but not
    subtract, divide or power).

    """

    __props__ = ("scalar_op", "axis", "dtype", "acc_dtype", "upcast_discrete_output")

    # When True, reducing a zero-sized axis is an error (set by reductions with
    # no identity element, e.g. Max/Min).
    error_on_empty_reduce_axis = False

    def __init__(
        self,
        scalar_op,
        axis=None,
        dtype=None,
        acc_dtype=None,
        upcast_discrete_output=False,
    ):
        """

        Parameters
        ----------
        scalar_op
            A binary scalar `Op` with only one output.
            It must be commutative and associative.
        axis
            - the dimension along which we want to reduce
            - list of dimensions that we want to reduce
            - if ``None``, all dimensions are reduced
        dtype
            The dtype of the returned tensor. If ``None``, then we use the default
            dtype which is the same as the input array's dtype except when
            `upcast_discrete_output` is ``True`` and the following holds:

            - the input dtype is a signed integer of precision < 64 bit, in which
            case we use int64
            - the input dtype is an unsigned integer of precision < 64 bit, in
            which case we use uint64

            This default dtype does _not_ depend on the value of `acc_dtype`.
            This behavior is similar in spirit to that of NumPy, except that
            NumPy uses the default machine integer while we always use 64 bit
            integers to avoid platform-dependent behavior.
        acc_dtype
            The dtype of the internal accumulator.
            If ``None`` (default), we use the dtype in the list below,
            or the input dtype if its precision is higher:

            - for int dtypes, we use at least int64;
            - for uint dtypes, we use at least uint64;
            - for float dtypes, we use at least float64;
            - for complex dtypes, we use at least complex128.
        upcast_discrete_output
            See

        """
        if scalar_op.nin not in (-1, 2) or scalar_op.nout != 1:
            raise NotImplementedError(
                "CAReduce only supports binary functions with a single output."
            )

        self.axis = None
        self.scalar_op = scalar_op

        if axis is not None:
            if isinstance(axis, int | np.integer) or (
                isinstance(axis, np.ndarray) and not axis.shape
            ):
                self.axis = (int(axis),)
            else:
                self.axis = tuple(axis)

        self.dtype = dtype if dtype is None else np.dtype(dtype).name
        self.acc_dtype = acc_dtype if acc_dtype is None else np.dtype(acc_dtype).name
        self.upcast_discrete_output = upcast_discrete_output

    @property
    def ufunc(self):
        if hasattr(self, "_ufunc"):
            return self._ufunc

        if hasattr(self.scalar_op, "nfunc_spec") and hasattr(
            np, self.scalar_op.nfunc_spec[0]
        ):
            self._ufunc = getattr(np, self.scalar_op.nfunc_spec[0])
        else:
            self._ufunc = np.frompyfunc(
                self.scalar_op.impl, 2, 1, identity=self.scalar_op.identity
            )

        return self._ufunc

    def _output_dtype(self, idtype):
        if not self.upcast_discrete_output:
            return idtype

        dtype = self.dtype

        if dtype == "OLD":
            return dict(
                int8="int32",
                int16="int32",
                int32="int64",
                uint8="uint32",
                uint16="uint32",
                uint32="uint64",
            ).get(idtype, idtype)
        elif dtype is None:
            # If input has a discrete dtype, upcast it to 64
            return dict(
                bool="int64",
                int8="int64",
                int16="int64",
                int32="int64",
                uint8="uint64",
                uint16="uint64",
                uint32="uint64",
            ).get(idtype, idtype)
        else:
            # The important is that the accumulator dtype does not
            # lose precision. Then, the result can be downcasted.
            return dtype

    def _acc_dtype(self, idtype):
        acc_dtype = self.acc_dtype
        if acc_dtype is None:
            return dict(
                bool="int64",
                int8="int64",
                int16="int64",
                int32="int64",
                uint8="uint64",
                uint16="uint64",
                uint32="uint64",
                float16="float32",
                float32="float64",
                complex64="complex128",
            ).get(idtype, idtype)
        elif acc_dtype in continuous_dtypes and idtype in discrete_dtypes:
            # Specifying a continuous accumulator for discrete input is OK
            return acc_dtype
        else:
            # The conversion has to be considered an upcast.
            upcasted_dtype = upcast(idtype, acc_dtype)
            if acc_dtype != upcasted_dtype:
                raise TypeError(
                    f"Cannot build {self} node with input dtype {idtype} "
                    f"and acc_dtype {acc_dtype}, as precision would be lost. "
                    "To correct this error, you can:\n"
                    "  - not specify acc_dtype, or\n"
                    f"  - use an acc_dtype at least as precise as {upcasted_dtype}.\n"
                    '  - specify "dtype" instead of "acc_dtype", so '
                    "the reduction will be precise, but the result will "
                    'be casted into "dtype" at the end.\n'
                    "If you are expecting the precision loss, you can "
                    f'use tensor.cast(..., dtype="{acc_dtype}"), on your input.'
                )
            return acc_dtype

    def make_node(self, input):
        input = as_tensor_variable(input)
        inp_dtype = input.type.dtype

        # We need to redefine make_node so that, if self.dtype is None,
        # we can infer what dtype should be, and create a node from an Op
        # of the appropriate dtype.
        dtype = self._output_dtype(inp_dtype)
        acc_dtype = self._acc_dtype(inp_dtype)

        assert dtype is not None
        assert acc_dtype is not None

        axis = normalize_reduce_axis(self.axis, ndim=input.type.ndim)

        if axis != self.axis or dtype != self.dtype or acc_dtype != self.acc_dtype:
            op = self.clone(axis=axis, dtype=dtype, acc_dtype=acc_dtype)
        else:
            op = self

        if axis is None:
            out_shape = ()
        else:
            out_shape = tuple(
                s for i, s in enumerate(input.type.shape) if i not in axis
            )

        output = TensorType(dtype=dtype, shape=out_shape)()

        return Apply(op, [input], [output])

    def clone(
        self,
        axis=None,
        dtype=None,
        acc_dtype=None,
        upcast_discrete_output=None,
        **kwargs,
    ):
        if axis is None:
            axis = self.axis
        if dtype is None:
            dtype = self.dtype
        if acc_dtype is None:
            acc_dtype = self.acc_dtype
        if upcast_discrete_output is None:
            upcast_discrete_output = self.upcast_discrete_output

        res = type(self)(
            self.scalar_op,
            axis=axis,
            dtype=dtype,
            acc_dtype=acc_dtype,
            upcast_discrete_output=None,
            **kwargs,
        )

        return res

    def _axis_str(self):
        axis = self.axis
        if axis is None:
            return "axes=None"
        elif len(axis) == 1:
            return f"axis={axis[0]}"
        else:
            return f"axes={list(axis)}"

    def __str__(self):
        if self.acc_dtype != self.dtype:
            return f"{type(self).__name__}{{{self.scalar_op}, {self._axis_str()}, acc={self.acc_dtype}}}"
        else:
            return f"{type(self).__name__}{{{self.scalar_op}, {self._axis_str()}}}"

    def perform(self, node, inp, out):
        (input,) = inp
        (output,) = out
        axis = self.axis

        out_dtype = node.outputs[0].type.dtype

        if self.acc_dtype is not None:
            acc_dtype = self.acc_dtype
        else:
            acc_dtype = out_dtype

        # out_dtype = self.dtype if self.dtype and self.dtype != "OLD" else out_dtype

        input = np.array(input, dtype=acc_dtype)

        out = self.ufunc.reduce(input, axis=axis, dtype=acc_dtype)

        output[0] = np.asarray(out, dtype=out_dtype)

    def infer_shape(self, fgraph, node, shapes):
        (ishape,) = shapes
        axis = self.axis
        if axis is None:
            return ((),)
        return ([ishape[i] for i in range(node.inputs[0].type.ndim) if i not in axis],)


def scalar_elemwise(*symbol, nfunc=None, nin=None, nout=None, symbolname=None):
    """Replace a symbol definition with an `Elemwise`-wrapped version of the corresponding scalar `Op`.

    If it is not ``None``, the `nfunc` argument should be a string such that
    ``getattr(numpy, nfunc)`` implements a vectorized version of the `Elemwise`
    operation.  `nin` is the number of inputs expected by that function, and nout
    is the number of **destination** inputs it takes.  That is, the function
    should take nin + nout inputs. `nout == 0` means that the numpy function does
    not take a NumPy array argument to put its result in.

    """
    import pytensor.scalar as scalar

    def construct(symbol):
        nonlocal symbolname

        symbolname = symbolname or symbol.__name__

        if symbolname.endswith("_inplace"):
            raise ValueError(
                "Creation of automatic inplace elemwise operations deprecated"
            )

        scalar_op = getattr(scalar, symbolname)
        rval = Elemwise(scalar_op, nfunc_spec=(nfunc and (nfunc, nin, nout)))

        if getattr(symbol, "__doc__"):
            rval.__doc__ = symbol.__doc__

        # for the meaning of this see the ./epydoc script
        # it makes epydoc display rval as if it were a function, not an object
        rval.__epydoc_asRoutine = symbol
        rval.__module__ = symbol.__module__

        return rval

    if symbol:
        return construct(symbol[0])
    else:
        return construct


@_get_vector_length.register(Elemwise)
def _get_vector_length_Elemwise(op, var):
    if len(var.owner.inputs) == 1 and len(var.owner.outputs) == 1:
        return get_vector_length(var.owner.inputs[0])

    raise ValueError(f"Length of {var} cannot be determined")


_vectorize_node.register(Elemwise, _vectorize_not_needed)


@_vectorize_node.register(DimShuffle)
def vectorize_dimshuffle(op: DimShuffle, node: Apply, x: TensorVariable) -> Apply:
    batched_ndims = x.type.ndim - node.inputs[0].type.ndim
    if not batched_ndims:
        return node.op.make_node(x)
    # e.g., ds(input_ndim=2, order=(1, "x", 0)) -> ds(input_ndim=4, order=(0, 1, 3, "x", 2))
    # e.g., ds(input_ndim=2, order=(1, "x")) -> ds(input_ndim=4, order=(0, 1, 3, "x"))
    new_order = list(range(batched_ndims)) + [
        "x" if (o == "x") else (o + batched_ndims) for o in op.new_order
    ]
    return x.dimshuffle(new_order).owner


def get_normalized_batch_axes(
    core_axes: None | int | tuple[int, ...],
    core_ndim: int,
    batch_ndim: int,
) -> tuple[int, ...]:
    """Compute batch axes for a batched operation, from the core input ndim and axes.

    e.g., sum(matrix, axis=None) -> sum(tensor4, axis=(2, 3))
    batch_axes(None, 2, 4) -> (2, 3)

    e.g., sum(matrix, axis=0) -> sum(tensor4, axis=(2,))
    batch_axes(0, 2, 4) -> (2,)

    e.g., sum(tensor3, axis=(0, -1)) -> sum(tensor4, axis=(1, 3))
    batch_axes((0, -1), 3, 4) -> (1, 3)
    """
    if core_axes is None:
        core_axes = tuple(range(core_ndim))
    else:
        core_axes = normalize_axis_tuple(core_axes, core_ndim)
    return tuple(core_axis + batch_ndim for core_axis in core_axes)


@_vectorize_node.register(CAReduce)
def vectorize_careduce(op: CAReduce, node: Apply, batch_x: TensorVariable) -> Apply:
    core_ndim = node.inputs[0].type.ndim
    batch_ndim = batch_x.type.ndim - core_ndim

    if not batch_ndim:
        return node.op.make_node(batch_x)

    batch_axes = get_normalized_batch_axes(op.axis, core_ndim, batch_ndim)
    return op.clone(axis=batch_axes).make_node(batch_x)
