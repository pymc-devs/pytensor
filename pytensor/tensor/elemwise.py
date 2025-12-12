from collections.abc import Sequence
from copy import copy
from textwrap import dedent
from typing import Literal

import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple

import pytensor.tensor.basic
from pytensor.configdefaults import config
from pytensor.gradient import DisconnectedType
from pytensor.graph.basic import Apply
from pytensor.graph.null_type import NullType
from pytensor.graph.replace import _vectorize_node, _vectorize_not_needed
from pytensor.graph.utils import MethodNotDefined
from pytensor.link.c.basic import failure_code
from pytensor.link.c.op import COp, ExternalCOp, OpenMPOp
from pytensor.link.c.params_type import ParamsType
from pytensor.misc.frozendict import frozendict
from pytensor.printing import Printer, pprint
from pytensor.scalar import get_scalar_type
from pytensor.scalar.basic import identity as scalar_identity
from pytensor.scalar.basic import int64, upcast
from pytensor.tensor import elemwise_cgen as cgen
from pytensor.tensor import get_vector_length
from pytensor.tensor.basic import _get_vector_length, as_tensor_variable
from pytensor.tensor.type import (
    TensorType,
    continuous_dtypes,
    discrete_dtypes,
    float_dtypes,
    lvector,
)
from pytensor.tensor.utils import (
    broadcast_static_dim_lengths,
    import_func_from_string,
    normalize_reduce_axis,
)
from pytensor.tensor.variable import TensorVariable
from pytensor.utils import uniq


class DimShuffle(ExternalCOp):
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
    c_func_file = "c_code/dimshuffle.c"
    c_func_name = "APPLY_SPECIFIC(cpu_dimshuffle)"
    view_map = {0: [0]}

    @property
    def params_type(self):
        return ParamsType(
            _new_order=lvector,
            input_ndim=int64,
        )

    def __init__(self, *, input_ndim: int, new_order: Sequence[int | Literal["x"]]):
        super().__init__([self.c_func_file], self.c_func_name)

        if not isinstance(input_ndim, int):
            raise TypeError(f"input_ndim must be an integer, got {type(int)}")

        self.input_ndim = input_ndim
        self.new_order = tuple(new_order)
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

        # List of input dimensions to drop
        drop = [i for i in range(input_ndim) if i not in new_order]

        # This is the list of the original dimensions that we keep
        self.shuffle = [x for x in new_order if x != "x"]
        self.transposition = self.shuffle + drop
        # List of dimensions of the output that are broadcastable and were not
        # in the original input
        self.augment = augment = sorted(i for i, x in enumerate(new_order) if x == "x")
        self.drop = drop

        dims_are_shuffled = sorted(self.shuffle) != self.shuffle

        self.is_transpose = dims_are_shuffled and not augment and not drop
        self.is_squeeze = drop and not dims_are_shuffled and not augment
        self.is_expand_dims = augment and not dims_are_shuffled and not drop
        self.is_left_expand_dims = self.is_expand_dims and (
            input_ndim == 0 or new_order[-input_ndim:] == list(range(input_ndim))
        )
        self.is_right_expand_dims = self.is_expand_dims and new_order[
            :input_ndim
        ] == list(range(input_ndim))

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "func_files"):
            # Perhaps we are loading an old `Op` version of DimShuffle.
            # Let's just build the ExternalCOp.
            super().__init__([self.c_func_file], self.c_func_name)

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
        res = res.transpose(self.transposition)

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

    def R_op(self, inputs, eval_points):
        if None in eval_points:
            return [None]
        return self(*eval_points, return_list=True)

    def grad(self, inp, grads):
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


class Elemwise(OpenMPOp):
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
        super().__init__(openmp=openmp)

    def __getstate__(self):
        d = copy(self.__dict__)
        d.pop("ufunc")
        d.pop("nfunc")
        d.pop("__epydoc_asRoutine", None)
        return d

    def __setstate__(self, d):
        super().__setstate__(d)
        self.ufunc = None
        self.nfunc = None
        self.inplace_pattern = frozendict(self.inplace_pattern)

    def get_output_info(self, *inputs):
        """Return the outputs dtype and broadcastable pattern and the
        dimshuffled inputs.

        """
        shadow = self.scalar_op.make_node(
            *[get_scalar_type(dtype=i.type.dtype).make_variable() for i in inputs]
        )

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

    def R_op(self, inputs, eval_points):
        outs = self(*inputs, return_list=True)
        rval = [None for x in outs]
        # For each output
        for idx, out in enumerate(outs):
            # make such that _bgrads computes only the gradients of the
            # current output on the inputs ( and not all outputs)
            ograds = [x.zeros_like() for x in outs]
            ograds[idx] = pytensor.tensor.basic.ones_like(out)

            bgrads = self._bgrad(inputs, outs, ograds)
            rop_out = None

            for jdx, (inp, eval_point) in enumerate(
                zip(inputs, eval_points, strict=True)
            ):
                # if None, then we can just ignore this branch ..
                # what we do is to assume that for any non-differentiable
                # branch, the gradient is actually 0, which I think is not
                # the right thing to do .. have to talk to Ian and James
                # about it

                if bgrads[jdx] is None or isinstance(
                    bgrads[jdx].type, DisconnectedType
                ):
                    pass
                elif eval_point is not None:
                    if rop_out is None:
                        rop_out = bgrads[jdx] * eval_point
                    else:
                        rop_out = rop_out + bgrads[jdx] * eval_point

            rval[idx] = rop_out

        return rval

    def connection_pattern(self, node):
        if hasattr(self.scalar_op, "connection_pattern"):
            return self.scalar_op.connection_pattern(node)

        return [[True for output in node.outputs] for ipt in node.inputs]

    def L_op(self, inputs, outs, ograds):
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

        with config.change_flags(compute_test_value="off"):

            def as_scalar(t):
                if isinstance(t.type, NullType | DisconnectedType):
                    return t
                return get_scalar_type(t.type.dtype)()

            scalar_inputs = list(map(as_scalar, inputs))
            scalar_ograds = list(map(as_scalar, ograds))
            scalar_outputs = self.scalar_op.make_node(
                *[get_scalar_type(dtype=i.type.dtype).make_variable() for i in inputs]
            ).outputs
            scalar_igrads = self.scalar_op.L_op(
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

    def _c_all(self, node, nodename, inames, onames, sub):
        # Some `Op`s directly call `Elemwise._c_all` or `Elemwise.c_code`
        # To not request all of them to call prepare_node(), do it here.
        # There is no harm if it get called multiple times.
        if not hasattr(node.tag, "fake_node"):
            self.prepare_node(node, None, None, "c")
        _inames = inames
        _onames = onames

        inames = uniq(inames)
        inputs = uniq(node.inputs)
        # assert that inames and inputs order stay consistent.
        # This is to protect again futur change of uniq.
        assert len(inames) == len(inputs)
        ii, iii = list(
            zip(*uniq(list(zip(_inames, node.inputs, strict=True))), strict=True)
        )
        assert all(x == y for x, y in zip(ii, inames, strict=True))
        assert all(x == y for x, y in zip(iii, inputs, strict=True))

        defines = ""
        undefs = ""

        # The destroy map is a map of output indices to input indices
        # that overwrite them.  We just convert them to the actual
        # Variables.
        dmap = {
            node.outputs[o]: [node.inputs[i]] for o, i in self.inplace_pattern.items()
        }

        # dtypes of the inputs
        idtypes = [input.type.dtype_specs()[1] for input in inputs]

        # These are the outputs that we will need to allocate
        # (output, name, name of the c type), transposed
        real = list(
            zip(
                *[
                    (r, s, r.type.dtype_specs()[1])
                    for r, s in zip(node.outputs, onames, strict=True)
                    if r not in dmap
                ],
                strict=True,
            )
        )
        if real:
            real_outputs, real_onames, real_odtypes = real
        else:
            real_outputs, real_onames, real_odtypes = [], [], []

        # Outputs that are aliased with an input (inplace)
        # (output, name), transposed (c type name not needed since we don't
        # need to allocate.
        aliased = list(
            zip(
                *[
                    (r, s)
                    for (r, s) in zip(node.outputs, onames, strict=True)
                    if r in dmap
                ],
                strict=True,
            )
        )
        if aliased:
            aliased_outputs, aliased_onames = aliased
        else:
            aliased_outputs, aliased_onames = [], []

        # for each input:
        # same as range(ndim), but with 'x' at all broadcastable positions
        orders = [
            [(s == 1 and "x") or i for i, s in enumerate(input.type.shape)]
            for input in inputs
        ]

        # number of nested loops we will need (all inputs have same
        # dimensionality)
        nnested = len(orders[0])
        sub = dict(sub)
        for i, (input, iname) in enumerate(zip(inputs, inames, strict=True)):
            # the c generators will substitute the input names for
            # references to loop variables lv0, lv1, ...
            sub[f"lv{i}"] = iname

        decl = cgen.make_declare(orders, idtypes, sub)
        checks = cgen.make_checks(orders, idtypes, sub)

        # Check if all inputs (except broadcasted scalar) are fortran.
        # In that case, create a fortran output ndarray.
        z = list(zip(inames, inputs, strict=True))
        alloc_fortran = " && ".join(
            f"PyArray_ISFORTRAN({arr})"
            for arr, var in z
            if not all(s == 1 for s in var.type.shape)
        )
        # If it is a scalar, make it c contig to prevent problem with
        # NumPy C and F contig not always set as both of them.
        if len(alloc_fortran) == 0:
            alloc_fortran = "0"

        alloc = ""
        # We loop over the "real" outputs, i.e., those that are not
        # inplace (must be allocated) and we declare/allocate/check
        # them
        for output, oname, odtype in zip(
            real_outputs, real_onames, real_odtypes, strict=True
        ):
            i += 1  # before this loop, i = number of inputs
            sub[f"lv{i}"] = oname
            sub["olv"] = oname
            alloc += cgen.make_declare(
                [list(range(nnested))], [odtype], dict(sub, lv0=oname)
            )
            alloc += cgen.make_alloc(orders, odtype, sub, fortran=alloc_fortran)
            alloc += cgen.make_checks(
                [list(range(nnested))], [odtype], dict(sub, lv0=oname)
            )
        olv_index = i  # index of the last output

        # We loop over the "aliased" outputs, i.e., those that are
        # inplace (overwrite the contents of one of the inputs) and
        # make the output pointers point to their corresponding input
        # pointers.
        for output, oname in zip(aliased_outputs, aliased_onames, strict=True):
            olv_index = inputs.index(dmap[output][0])
            iname = inames[olv_index]
            # We make the output point to the corresponding input and
            # decrease the reference of whatever the output contained
            # prior to this
            alloc += f"""
            if ({oname}) {{
                Py_XDECREF({oname});
            }}
            {oname} = {iname};
            Py_XINCREF({oname});
            """
            # We alias the scalar variables
            defines += f"#define {oname}_i {iname}_i\n"
            undefs += f"#undef {oname}_i\n"

        # Note: here, olv_index is either the index of the last output
        # which is allocated, OR, if there are any aliased outputs,
        # the index of the last of these aliased outputs.

        # We generate the C code of the inner loop using the scalar op
        if self.openmp:
            # If we are using openmp, we need to get rid of the "goto"
            # statement in sub['fail']. For now we recreate it here.
            fail = failure_code(sub, use_goto=False)
        else:
            fail = sub["fail"]
        task_code = self.scalar_op.c_code(
            node.tag.fake_node,
            nodename + "_scalar_",
            [f"{s}_i" for s in _inames],
            [f"{s}_i" for s in onames],
            dict(sub, fail=fail),
        )
        code = f"""
        {{
            {defines}
            {task_code}
            {undefs}
        }}
        """

        loop_orders = orders + [list(range(nnested))] * len(real_onames)
        dtypes = idtypes + list(real_odtypes)
        if all(
            [o.ndim <= 1 for o in node.outputs]
            or
            # Use simpler code when output ndim == 0 or 1
            # or for broadcated scalar.
            all(s == 1 for s in node.outputs[0].type.shape)
        ):
            if nnested:
                all_code = [("", "")] * (nnested - 1) + [("", code)] + [""]
            else:
                all_code = [code]
            if len(all_code) == 1:
                # No loops
                task_decl = "".join(
                    f"{dtype}& {name}_i = *{name}_iter;\n"
                    for name, dtype in zip(
                        inames + list(real_onames),
                        idtypes + list(real_odtypes),
                        strict=True,
                    )
                )

                preloops = {}
                for i, (loop_order, dtype) in enumerate(
                    zip(loop_orders, dtypes, strict=True)
                ):
                    for j, index in enumerate(loop_order):
                        if index != "x":
                            preloops.setdefault(j, "")
                            preloops[j] += (
                                f"%(lv{i})s_iter = ({dtype}*)(PyArray_DATA(%(lv{i})s));\n"
                            ) % sub
                            break
                    else:  # all broadcastable
                        preloops.setdefault(0, "")
                        preloops[0] += (
                            f"%(lv{i})s_iter = ({dtype}*)(PyArray_DATA(%(lv{i})s));\n"
                        ) % sub

                init_array = preloops.get(0, " ")
                loop = f"""
                {{
                  {defines}
                  {init_array}
                  {task_decl}
                  {task_code}
                  {undefs}
                }}
                """
            else:
                loop = cgen.make_loop(
                    loop_orders=loop_orders,
                    dtypes=dtypes,
                    loop_tasks=all_code,
                    sub=sub,
                    openmp=self.openmp,
                )
        else:
            loop = cgen.make_reordered_loop(
                init_loop_orders=loop_orders,
                olv_index=olv_index,
                dtypes=dtypes,
                inner_task=code,
                sub=sub,
                openmp=self.openmp,
            )

        # If all inputs and outputs are contiguous
        # and the scalar op define optimized code for that case
        # use it! The scalar_op needs to check the type-level shapes itself.
        if (
            all(o.ndim >= 1 for o in node.outputs)
            and
            # Don't use the contig code for broadcasted scalar.
            not all(s == 1 for s in node.outputs[0].type.shape)
        ):
            contig = None
            try:
                contig = self.scalar_op.c_code_contiguous(
                    node, nodename + "_scalar_contig_", _inames, onames, sub
                )
            except MethodNotDefined:
                # Try to make one generic version, this will help the
                # compiler to vectorize the code as their won't be as
                # many ptr and the stride will be hard coded.
                if all(
                    # io.type.shape == node.outputs[1].type.shape
                    # Elemwise does not specify non-broadcastable static/type-levelshape
                    # information for its outputs yet
                    node.outputs[0].type.is_super(io.type)
                    for io in node.inputs + node.outputs
                ) and (
                    len(node.inputs) <= 1
                    # If either one of the inputs has a `None` shape, we cannot
                    # assume they will have the same size
                    or all(
                        len(set(inp_shape)) == 1 and None not in inp_shape
                        for inp_shape in zip(
                            *(inp.type.shape for inp in node.inputs), strict=True
                        )
                    )
                ):
                    z = onames[0]
                    contig = f"""
                    // All output have the same size
                    npy_intp n = PyArray_SIZE({z});
                    """
                    index = ""
                    for x, var in zip(
                        inames + onames, inputs + node.outputs, strict=True
                    ):
                        if not all(s == 1 for s in var.type.shape):
                            contig += f"""
            dtype_{x} * {x}_ptr = (dtype_{x}*) PyArray_DATA({x});
                            """
                            index += f"""
            dtype_{x}& {x}_i = {x}_ptr[i];
                            """
                        else:
                            contig += f"""
            dtype_{x}& {x}_i = ((dtype_{x}*) PyArray_DATA({x}))[0];
                            """
                    if self.openmp:
                        contig += f"""#pragma omp parallel for if(n>={int(config.openmp_elemwise_minsize)})
                        """
                    contig += f"""
                    for(int i=0; i<n; i++){{
                        {index}
                        {task_code};
                    }}
                    """
            if contig is not None:
                z = list(zip(inames + onames, inputs + node.outputs, strict=True))
                all_broadcastable = all(s == 1 for s in var.type.shape)
                cond1 = " && ".join(
                    f"PyArray_ISCONTIGUOUS({arr})"
                    for arr, var in z
                    if not all_broadcastable
                )
                cond2 = " && ".join(
                    f"PyArray_ISFORTRAN({arr})"
                    for arr, var in z
                    if not all_broadcastable
                )
                loop = f"""
            if(({cond1}) || ({cond2})){{
                {contig}
            }}else{{
                {loop}
            }}
            """
        return decl, checks, alloc, loop, ""

    def c_code(self, node, nodename, inames, onames, sub):
        if (
            any(i.dtype == "float16" for i in node.inputs)
            or any(o.dtype == "float16" for o in node.outputs)
            or
            # This is for Composite
            getattr(self.scalar_op, "inner_float16", False)
        ):
            # Disable C code for float16 vars
            raise NotImplementedError()
        code = "\n".join(self._c_all(node, nodename, inames, onames, sub))
        return code

    def c_headers(self, **kwargs):
        return ["<vector>", "<algorithm>"]

    def c_header_dirs(self, **kwargs):
        return self.scalar_op.c_header_dirs(**kwargs)

    def c_support_code(self, **kwargs):
        return self.scalar_op.c_support_code(**kwargs)

    def c_support_code_apply(self, node, nodename):
        support_code = self.scalar_op.c_support_code_apply(node, nodename + "_scalar_")
        return support_code

    def c_code_cache_version_apply(self, node):
        version = [17]  # the version corresponding to the c code in this Op

        # now we insert versions for the ops on which we depend...
        scalar_node = Apply(
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
        version.append(self.scalar_op.c_code_cache_version_apply(scalar_node))
        version.extend(
            get_scalar_type(dtype=i.type.dtype).c_code_cache_version()
            for i in node.inputs + node.outputs
        )
        version.append(("openmp", self.openmp))
        version.append(("openmp_elemwise_minsize", config.openmp_elemwise_minsize))
        if all(version):
            return tuple(version)
        else:
            return ()

    def outer(self, x, y):
        from pytensor.tensor.basic import expand_dims

        if self.scalar_op.nin not in (-1, 2):
            raise NotImplementedError("outer is only available for binary operators")

        x_ = expand_dims(x, tuple(range(-y.ndim, 0)))
        y_ = expand_dims(y, tuple(range(x.ndim)))
        return self(x_, y_)


class CAReduce(COp):
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

    def _c_all(self, node, name, input_names, output_names, sub):
        [inp] = node.inputs
        [out] = node.outputs
        ndim = inp.type.ndim

        [inp_name] = input_names
        [out_name] = output_names

        inp_dtype = inp.type.dtype_specs()[1]
        out_dtype = out.type.dtype_specs()[1]

        acc_dtype = getattr(self, "acc_dtype", None)

        if acc_dtype is not None:
            if acc_dtype == "float16":
                raise MethodNotDefined("no c_code for float16")
            acc_type = TensorType(shape=node.outputs[0].type.shape, dtype=acc_dtype)
            acc_dtype = acc_type.dtype_specs()[1]
        else:
            acc_dtype = out_dtype

        axis = self.axis
        if axis is None:
            axis = list(range(inp.type.ndim))

        if len(axis) == 0:
            # This is just an Elemwise cast operation
            # The acc_dtype is never a downcast compared to the input dtype
            # So we just need a cast to the output dtype.
            var = pytensor.tensor.basic.cast(inp, node.outputs[0].dtype)
            if var is inp:
                var = Elemwise(scalar_identity)(inp)
            assert var.dtype == node.outputs[0].dtype
            return var.owner.op._c_all(var.owner, name, input_names, output_names, sub)

        inp_dims = list(range(ndim))
        non_reduced_dims = [i for i in inp_dims if i not in axis]
        counter = iter(range(ndim))
        acc_dims = ["x" if i in axis else next(counter) for i in range(ndim)]

        sub = sub.copy()
        sub["lv0"] = inp_name
        sub["lv1"] = out_name
        sub["olv"] = out_name

        if acc_dtype != out_dtype:
            # Create an accumulator variable different from the output
            acc_name = "acc"
            setup = acc_type.c_declare(acc_name, sub) + acc_type.c_init(acc_name, sub)
        else:
            # the output is the accumulator variable
            acc_name = out_name
            setup = ""

        # Define strides of input array
        setup += cgen.make_declare(
            [inp_dims], [inp_dtype], sub, compute_stride_jump=False
        ) + cgen.make_checks([inp_dims], [inp_dtype], sub, compute_stride_jump=False)

        # Define strides of output array and allocate it
        out_sub = sub | {"lv0": out_name}
        alloc = (
            cgen.make_declare(
                [acc_dims], [out_dtype], out_sub, compute_stride_jump=False
            )
            + cgen.make_alloc([non_reduced_dims], out_dtype, sub)
            + cgen.make_checks(
                [acc_dims], [out_dtype], out_sub, compute_stride_jump=False
            )
        )

        if acc_dtype != out_dtype:
            # Define strides of accumulation buffer and allocate it
            sub["lv1"] = acc_name
            sub["olv"] = acc_name

            acc_sub = sub | {"lv0": acc_name}
            alloc += (
                cgen.make_declare(
                    [acc_dims], [acc_dtype], acc_sub, compute_stride_jump=False
                )
                + cgen.make_alloc([non_reduced_dims], acc_dtype, sub)
                + cgen.make_checks(
                    [acc_dims], [acc_dtype], acc_sub, compute_stride_jump=False
                )
            )

        identity = self.scalar_op.identity
        if np.isposinf(identity):
            if inp.type.dtype in ("float32", "float64"):
                identity = "__builtin_inf()"
            elif inp.type.dtype.startswith("uint") or inp.type.dtype == "bool":
                identity = "1"
            else:
                identity = "NPY_MAX_" + str(inp.type.dtype).upper()
        elif np.isneginf(identity):
            if inp.type.dtype in ("float32", "float64"):
                identity = "-__builtin_inf()"
            elif inp.type.dtype.startswith("uint") or inp.type.dtype == "bool":
                identity = "0"
            else:
                identity = "NPY_MIN_" + str(inp.type.dtype).upper()
        elif identity is None:
            raise TypeError(f"The {self.scalar_op} does not define an identity.")

        initial_value = f"{acc_name}_i = {identity};"

        inner_task = self.scalar_op.c_code(
            Apply(
                self.scalar_op,
                [
                    get_scalar_type(dtype=iv.type.dtype).make_variable()
                    for iv in (node.inputs * 2)
                ],
                [
                    get_scalar_type(dtype=ov.type.dtype).make_variable()
                    for ov in node.outputs
                ],
            ),
            None,
            [f"{acc_name}_i", f"{inp_name}_i"],
            [f"{acc_name}_i"],
            sub,
        )

        if out.type.ndim == 0:
            # Simple case where everything is reduced, no need for loop ordering
            loop = cgen.make_complete_loop_careduce(
                inp_var=inp_name,
                acc_var=acc_name,
                inp_dtype=inp_dtype,
                acc_dtype=acc_dtype,
                initial_value=initial_value,
                inner_task=inner_task,
                fail_code=sub["fail"],
            )
        else:
            loop = cgen.make_reordered_loop_careduce(
                inp_var=inp_name,
                acc_var=acc_name,
                inp_dtype=inp_dtype,
                acc_dtype=acc_dtype,
                inp_ndim=ndim,
                reduction_axes=axis,
                initial_value=initial_value,
                inner_task=inner_task,
            )

        if acc_dtype != out_dtype:
            cast = dedent(
                f"""
                PyArray_CopyInto({out_name}, {acc_name});
                {acc_type.c_cleanup(acc_name, sub)}
                """
            )
        else:
            cast = ""

        return setup, alloc, loop, cast

    def c_code(self, node, name, inames, onames, sub):
        code = "\n".join(self._c_all(node, name, inames, onames, sub))
        return code

    def c_headers(self, **kwargs):
        # Sometimes, Elemwise's c_code is returned, so we need its headers
        return ["<vector>", "<algorithm>"]

    def c_code_cache_version_apply(self, node):
        # the version corresponding to the c code in this Op
        version = [11]

        # now we insert versions for the ops on which we depend...
        scalar_node = Apply(
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
        version.append(self.scalar_op.c_code_cache_version_apply(scalar_node))
        version.extend(
            get_scalar_type(dtype=i.type.dtype).c_code_cache_version()
            for i in node.inputs + node.outputs
        )
        if all(version):
            return tuple(version)
        else:
            return ()


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
