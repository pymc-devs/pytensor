Adding JAX, Numba and Pytorch support for `Op`\s
================================================

PyTensor is able to convert its graphs into JAX, Numba and Pytorch compiled functions. In order to do
this, each :class:`Op` in an PyTensor graph must have an equivalent JAX/Numba/Pytorch implementation function.

This tutorial will explain how JAX, Numba and Pytorch implementations are created for an :class:`Op`.

Step 1: Identify the PyTensor :class:`Op` you'd like to implement
-----------------------------------------------------------------

Find the source for the PyTensor :class:`Op` you'd like to be supported and
identify the function signature and return values. These can be determined by
looking at the :meth:`Op.make_node` implementation. In general, one needs to be familiar
with PyTensor :class:`Op`\s in order to provide a conversion implementation, so first read
:ref:`creating_an_op` if you are not familiar.

For example, you want to extend support for :class:`CumsumOp`\:

.. code:: python

    class CumsumOp(Op):
        __props__ = ("axis",)

        def __new__(typ, *args, **kwargs):
            obj = object.__new__(CumOp, *args, **kwargs)
            obj.mode = "add"
            return obj


:class:`CumsumOp` turns out to be a variant of :class:`CumOp`\ :class:`Op`
which currently has an :meth:`Op.make_node` as follows:

.. code:: python

    def make_node(self, x):
        x = ptb.as_tensor_variable(x)
        out_type = x.type()

        if self.axis is None:
            out_type = vector(dtype=x.dtype)  # Flatten
        elif self.axis >= x.ndim or self.axis < -x.ndim:
            raise ValueError(f"axis(={self.axis}) out of bounds")

        return Apply(self, [x], [out_type])

The :class:`Apply` instance that's returned specifies the exact types of inputs that
our implementation will receive and the exact types of outputs it's expected to
return--both in terms of their data types and number of dimensions/shapes.
The actual inputs our implementation will receive are necessarily numeric values
or NumPy :class:`ndarray`\s; all that :meth:`Op.make_node` tells us is the
general signature of the underlying computation.

More specifically, the :class:`Apply` implies that there is one input that is
automatically converted to PyTensor variables via :func:`as_tensor_variable`.
There is another parameter, `axis`, that is used to determine the direction
of the operation, hence shape of the output. The check that follows imply that
`axis` must refer to a dimension in the input tensor. The input's elements
could also have any data type (e.g. floats, ints), so our implementation
must be able to handle all the possible data types.

It also tells us that there's only one return value, that it has a data type
determined by :meth:`x.type` i.e., the data type of the original tensor.
This implies that the result is necessarily a matrix.

Some class may have a more complex behavior. For example, the :class:`CumOp`\ :class:`Op`
also has another variant :class:`CumprodOp`\ :class:`Op` with the exact signature
as :class:`CumsumOp`\ :class:`Op`. The difference lies in that the `mode` attribute in
:class:`CumOp` definition:

.. code:: python

    class CumOp(COp):
        # See function cumsum/cumprod for docstring

        __props__ = ("axis", "mode")
        check_input = False
        params_type = ParamsType(
            c_axis=int_t, mode=EnumList(("MODE_ADD", "add"), ("MODE_MUL", "mul"))
        )

        def __init__(self, axis: int | None = None, mode="add"):
            if mode not in ("add", "mul"):
                raise ValueError(f'{type(self).__name__}: Unknown mode "{mode}"')
            self.axis = axis
            self.mode = mode

        c_axis = property(lambda self: np.MAXDIMS if self.axis is None else self.axis)

`__props__` is used to parametrize the general behavior of the :class:`Op`. One need to
pay attention to this to decide whether the implementation should support all variants
or raise an explicit NotImplementedError for cases that are not supported e.g., when
:class:`CumsumOp` of :class:`CumOp("add")` is supported but not :class:`CumprodOp` of
:class:`CumOp("mul")`.

Next, we look at the :meth:`Op.perform` implementation to see exactly
how the inputs and outputs are used to compute the outputs for an :class:`Op`
in Python. This method is effectively what needs to be implemented.

Step 2: Find the relevant method in JAX/Numba/Pytorch (or something close)
--------------------------------------------------------------------------

With a precise idea of what the PyTensor :class:`Op` does we need to figure out how
to implement it in JAX, Numba or Pytorch. In the best case scenario, there is a similarly named
function that performs exactly the same computations as the :class:`Op`. For
example, the :class:`Eye` operator has a JAX equivalent: :func:`jax.numpy.eye`
and a Pytorch equivalent: :func:`torch.eye`.

If we wanted to implement an :class:`Op` like :class:`DimShuffle`, we might need to
recreate the functionality with some custom logic.  In many cases, at least some
custom logic is needed to reformat the inputs and outputs so that they exactly
match the `Op`'s.

Here's an example for :class:`DimShuffle`:


.. tab-set::

        .. tab-item:: JAX

            .. code:: python

                def dimshuffle(x, op):
                    res = jnp.transpose(x, op.transposition)

                    shape = list(res.shape[: len(op.shuffle)])

                    for augm in op.augment:
                        shape.insert(augm, 1)

                    res = jnp.reshape(res, shape)

                    if not op.inplace:
                        res = jnp.copy(res)

                    return res

        .. tab-item:: Numba

            .. code:: python

                def numba_funcify_DimShuffle(op, node, **kwargs):
                    shuffle = tuple(op.shuffle)
                    transposition = tuple(op.transposition)
                    augment = tuple(op.augment)
                    inplace = op.inplace

                    ndim_new_shape = len(shuffle) + len(augment)

                    no_transpose = all(i == j for i, j in enumerate(transposition))
                    if no_transpose:

                        @numba_basic.numba_njit
                        def transpose(x):
                            return x

                    else:

                        @numba_basic.numba_njit
                        def transpose(x):
                            return np.transpose(x, transposition)

                    shape_template = (1,) * ndim_new_shape

                    # When `len(shuffle) == 0`, the `shuffle_shape[j]` expression below
                    # is typed as `getitem(Tuple(), int)`, which has no implementation
                    # (since getting an item from an empty sequence doesn't make sense).
                    # To avoid this compile-time error, we omit the expression altogether.
                    if len(shuffle) > 0:
                        # Use the statically known shape if available
                        if all(length is not None for length in node.outputs[0].type.shape):
                            shape = node.outputs[0].type.shape

                            @numba_basic.numba_njit
                            def find_shape(array_shape):
                                return shape

                        else:

                            @numba_basic.numba_njit
                            def find_shape(array_shape):
                                shape = shape_template
                                j = 0
                                for i in range(ndim_new_shape):
                                    if i not in augment:
                                        length = array_shape[j]
                                        shape = numba_basic.tuple_setitem(shape, i, length)
                                        j = j + 1
                                return shape

                    else:

                        @numba_basic.numba_njit
                        def find_shape(array_shape):
                            return shape_template

                    if ndim_new_shape > 0:

                        @numba_basic.numba_njit
                        def dimshuffle_inner(x, shuffle):
                            x = transpose(x)
                            shuffle_shape = x.shape[: len(shuffle)]
                            new_shape = find_shape(shuffle_shape)

                            # FIXME: Numba's `array.reshape` only accepts C arrays.
                            res_reshape = np.reshape(np.ascontiguousarray(x), new_shape)

                            if not inplace:
                                return res_reshape.copy()
                            else:
                                return res_reshape

                    else:

                        @numba_basic.numba_njit
                        def dimshuffle_inner(x, shuffle):
                            return np.reshape(np.ascontiguousarray(x), ())

                    # Without the following wrapper function we would see this error:
                    # E   No implementation of function Function(<built-in function getitem>) found for signature:
                    # E
                    # E    >>> getitem(UniTuple(int64 x 2), slice<a:b>)
                    # E
                    # E   There are 22 candidate implementations:
                    # E      - Of which 22 did not match due to:
                    # E      Overload of function 'getitem': File: <numerous>: Line N/A.
                    # E        With argument(s): '(UniTuple(int64 x 2), slice<a:b>)':
                    # E       No match.
                    # ...(on this line)...
                    # E           shuffle_shape = res.shape[: len(shuffle)]
                    @numba_basic.numba_njit(inline="always")
                    def dimshuffle(x):
                        return dimshuffle_inner(np.asarray(x), shuffle)

                    return dimshuffle

        .. tab-item:: Pytorch

            .. code:: python

                def dimshuffle(x, op):
                    res = torch.permute(x, op.transposition)

                    shape = list(res.shape[: len(op.shuffle)])

                    for augm in op.augment:
                        shape.insert(augm, 1)

                    res = torch.reshape(res, shape)

                    if not op.inplace:
                        res = res.clone()

                    return res

In this case, :class:`CumOp` is implemented with NumPy's :func:`numpy.cumsum`
and :func:`numpy.cumprod`, which have JAX equivalents: :func:`jax.numpy.cumsum`
and :func:`jax.numpy.cumprod`. The Pytorch equivalents are :func:`torch.cumsum`
and :func:`torch.cumprod`

.. code:: python

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        if self.mode == "add":
            z[0] = np.cumsum(x, axis=self.axis)
        else:
            z[0] = np.cumprod(x, axis=self.axis)

Step 3: Register the function with the respective dispatcher
------------------------------------------------------------

With the PyTensor `Op` replicated, we'll need to register the
function with the backends `Linker`. This is done through the use of
`singledispatch`. If you don't know how `singledispatch` works, see the
`Python documentation <https://docs.python.org/3/library/functools.html#functools.singledispatch>`_.

The relevant dispatch functions created by `singledispatch` are :func:`pytensor.link.numba.dispatch.numba_funcify`,
:func:`pytensor.link.jax.dispatch.jax_funcify` and :func:`pytensor.link.pytorch.dispatch.pytorch_funcify`.

Here's an example for the `CumOp`\ `Op`:

.. tab-set::

    .. tab-item:: JAX

        .. code:: python

            import jax.numpy as jnp

            from pytensor.tensor.extra_ops import CumOp
            from pytensor.link.jax.dispatch import jax_funcify


            @jax_funcify.register(CumOp)
            def jax_funcify_CumOp(op, **kwargs):
                axis = op.axis
                mode = op.mode

                def cumop(x, axis=axis, mode=mode):
                    if mode == "add":
                        return jnp.cumsum(x, axis=axis)
                    else:
                        return jnp.cumprod(x, axis=axis)

                return cumop

        Suppose `jnp.cumprod` does not exist, we will need to register the function as follows:

        .. code:: python

            import jax.numpy as jnp

            from pytensor.tensor.extra_ops import CumOp
            from pytensor.link.jax.dispatch import jax_funcify


                @jax_funcify.register(CumOp)
                def jax_funcify_CumOp(op, **kwargs):
                    axis = op.axis
                    mode = op.mode

                    def cumop(x, axis=axis, mode=mode):
                        if mode == "add":
                            return jnp.cumsum(x, axis=axis)
                        else:
                            raise NotImplementedError("JAX does not support cumprod function at the moment.")

                    return cumop

    .. tab-item:: Numba

        .. code:: python

            import numpy as np

            from pytensor import config
            from pytensor.graph import Apply
            from pytensor.link.numba.dispatch import basic as numba_basic
            from pytensor.tensor import TensorVariable
            from pytensor.tensor.extra_ops import CumOp,

            def numba_funcify_CumOp(op: CumOp, node: Apply, **kwargs):
                axis = op.axis
                mode = op.mode
                ndim = cast(TensorVariable, node.outputs[0]).ndim

                if axis is not None:
                    if axis < 0:
                        axis = ndim + axis
                    if axis < 0 or axis >= ndim:
                        raise ValueError(f"Invalid axis {axis} for array with ndim {ndim}")

                    reaxis_first = (axis, *(i for i in range(ndim) if i != axis))
                    reaxis_first_inv = tuple(np.argsort(reaxis_first))

                if mode == "add":
                    if axis is None or ndim == 1:

                        @numba_basic.numba_njit()
                        def cumop(x):
                            return np.cumsum(x)

                    else:

                        @numba_basic.numba_njit(boundscheck=False)
                        def cumop(x):
                            out_dtype = x.dtype
                            if x.shape[axis] < 2:
                                return x.astype(out_dtype)

                            x_axis_first = x.transpose(reaxis_first)
                            res = np.empty(x_axis_first.shape, dtype=out_dtype)

                            res[0] = x_axis_first[0]
                            for m in range(1, x.shape[axis]):
                                res[m] = res[m - 1] + x_axis_first[m]

                            return res.transpose(reaxis_first_inv)

                else:
                    if axis is None or ndim == 1:

                        @numba_basic.numba_njit()
                        def cumop(x):
                            return np.cumprod(x)

                    else:

                        @numba_basic.numba_njit(boundscheck=False)
                        def cumop(x):
                            out_dtype = x.dtype
                            if x.shape[axis] < 2:
                                return x.astype(out_dtype)

                            x_axis_first = x.transpose(reaxis_first)
                            res = np.empty(x_axis_first.shape, dtype=out_dtype)

                            res[0] = x_axis_first[0]
                            for m in range(1, x.shape[axis]):
                                res[m] = res[m - 1] * x_axis_first[m]

                            return res.transpose(reaxis_first)

                return cumop


    .. tab-item:: Pytorch

        .. code:: python

            import torch

            from pytensor.link.pytorch.dispatch.basic import pytorch_funcify
            from pytensor.tensor.extra_ops import CumOp


            @pytorch_funcify.register(CumOp)
            def pytorch_funcify_Cumop(op, **kwargs):
                axis = op.axis
                mode = op.mode

                def cumop(x,):
                    if axis is None:
                        x = x.reshape(-1)
                        dim = 0
                    else:
                        dim=axis
                    if mode == "add":
                        return torch.cumsum(x, dim=dim)
                    else:
                        return torch.cumprod(x, dim=dim)

                return cumop


        Suppose `torch.cumprod` does not exist, we will need to register the function as follows:

        .. code:: python

            import torch

            from pytensor.tensor.extra_ops import CumOp
            from pytensor.link.pytorch.dispatch import pytorch_funcify


            @pytorch_funcify.register(CumOp)
            def pytorch_funcify_Cumop(op, **kwargs):
                axis = op.axis
                mode = op.mode

                def cumop(x, axis=axis, mode=mode):
                    if mode == "add":
                        return torch.cumsum(x, axis=axis)
                    else:
                        raise NotImplementedError("Pytorch does not support cumprod function at the moment.")

                return cumop

Step 4: Write tests
-------------------
.. tab-set::

    .. tab-item:: JAX

        Test that your registered `Op` is working correctly by adding tests to the
        appropriate test suites in PyTensor (e.g. in ``tests.link.jax``).
        The tests should ensure that your implementation can
        handle the appropriate types of inputs and produce outputs equivalent to `Op.perform`.
        Check the existing tests for the general outline of these kinds of tests. In
        most cases, a helper function can be used to easily verify the correspondence
        between a Numba implementation and its `Op`.

        For example, the :func:`compare_jax_and_py` function streamlines the steps
        involved in making comparisons with `Op.perform`.

        Here's a small example of a test for :class:`CumOp` above:

        .. code:: python

            import numpy as np
            import pytensor.tensor as pt
            from pytensor.configdefaults import config
            from tests.link.jax.test_basic import compare_jax_and_py
            from pytensor.graph import FunctionGraph
            from pytensor.graph.op import get_test_value

            def test_jax_CumOp():
                """Test JAX conversion of the `CumOp` `Op`."""

                # Create a symbolic input for the first input of `CumOp`
                a = pt.matrix("a")

                # Create test value tag for a
                a.tag.test_value = np.arange(9, dtype=config.floatX).reshape((3, 3))

                # Create the output variable
                out = pt.cumsum(a, axis=0)

                # Create a PyTensor `FunctionGraph`
                fgraph = FunctionGraph([a], [out])

                # Pass the graph and inputs to the testing function
                compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

                # For the second mode of CumOp
                out = pt.cumprod(a, axis=1)
                fgraph = FunctionGraph([a], [out])
                compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

        If the variant :class:`CumprodOp` is not implemented, we can add a test for it as follows:

        .. code:: python

            import pytest

            def test_jax_CumOp():
                """Test JAX conversion of the `CumOp` `Op`."""
                a = pt.matrix("a")
                a.tag.test_value = np.arange(9, dtype=config.floatX).reshape((3, 3))

                with pytest.raises(NotImplementedError):
                    out = pt.cumprod(a, axis=1)
                    fgraph = FunctionGraph([a], [out])
                    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


    .. tab-item:: Numba

        Test that your registered `Op` is working correctly by adding tests to the
        appropriate test suites in PyTensor (e.g. in ``tests.link.numba``).
        The tests should ensure that your implementation can
        handle the appropriate types of inputs and produce outputs equivalent to `Op.perform`.
        Check the existing tests for the general outline of these kinds of tests. In
        most cases, a helper function can be used to easily verify the correspondence
        between a Numba implementation and its `Op`.

        For example, the :func:`compare_numba_and_py` function streamlines the steps
        involved in making comparisons with `Op.perform`.

        Here's a small example of a test for :class:`CumOp` above:

        .. code:: python

            from tests.link.numba.test_basic import compare_numba_and_py
            from pytensor.graph import FunctionGraph
            from pytensor.compile.sharedvalue import SharedVariable
            from pytensor.graph.basic import Constant
            from pytensor.tensor import extra_ops

            def test_CumOp(val, axis, mode):
                g = extra_ops.CumOp(axis=axis, mode=mode)(val)
                g_fg = FunctionGraph(outputs=[g])

                compare_numba_and_py(
                    g_fg,
                    [
                        i.tag.test_value
                        for i in g_fg.inputs
                        if not isinstance(i, SharedVariable | Constant)
                    ],
                )



    .. tab-item:: Pytorch

        Test that your registered `Op` is working correctly by adding tests to the
        appropriate test suites in PyTensor (``tests.link.pytorch``). The tests should ensure that your implementation can
        handle the appropriate types of inputs and produce outputs equivalent to `Op.perform`.
        Check the existing tests for the general outline of these kinds of tests. In
        most cases, a helper function can be used to easily verify the correspondence
        between a Pytorch implementation and its `Op`.

        For example, the :func:`compare_pytorch_and_py` function streamlines the steps
        involved in making comparisons with `Op.perform`.

        Here's a small example of a test for :class:`CumOp` above:

        .. code:: python

            import numpy as np
            import pytest
            import pytensor.tensor as pt
            from pytensor.configdefaults import config
            from tests.link.pytorch.test_basic import compare_pytorch_and_py
            from pytensor.graph import FunctionGraph

            @pytest.mark.parametrize(
                "dtype",
                ["float64", "int64"],
            )
            @pytest.mark.parametrize(
                "axis",
                [None, 1, (0,)],
            )
            def test_pytorch_CumOp(axis, dtype):
                """Test PyTorch conversion of the `CumOp` `Op`."""

                # Create a symbolic input for the first input of `CumOp`
                a = pt.matrix("a", dtype=dtype)

                # Create test value
                test_value = np.arange(9, dtype=dtype).reshape((3, 3))

                # Create the output variable
                if isinstance(axis, tuple):
                    with pytest.raises(TypeError, match="axis must be an integer or None."):
                        out = pt.cumsum(a, axis=axis)
                    with pytest.raises(TypeError, match="axis must be an integer or None."):
                        out = pt.cumprod(a, axis=axis)
                else:
                    out = pt.cumsum(a, axis=axis)
                    # Create a PyTensor `FunctionGraph`
                    fgraph = FunctionGraph([a], [out])

                    # Pass the graph and inputs to the testing function
                    compare_pytorch_and_py(fgraph, [test_value])

                    # For the second mode of CumOp
                    out = pt.cumprod(a, axis=axis)
                    fgraph = FunctionGraph([a], [out])
                    compare_pytorch_and_py(fgraph, [test_value])


Note
----
In out previous example of extending JAX, :class:`Eye`\ :class:`Op` was used with the test function as follows:

.. code:: python

    def test_jax_Eye():
        """Test JAX conversion of the `Eye` `Op`."""

        # Create a symbolic input for `Eye`
        x_at = pt.scalar()

        # Create a variable that is the output of an `Eye` `Op`
        eye_var = pt.eye(x_at)

        # Create an PyTensor `FunctionGraph`
        out_fg = FunctionGraph(outputs=[eye_var])

        # Pass the graph and any inputs to the testing function
        compare_jax_and_py(out_fg, [3])

This one nowadays leads to a test failure due to new restrictions in JAX + JIT,
as reported in issue `#654 <https://github.com/pymc-devs/pytensor/issues/654>`_.
All jitted functions now must have constant shape, which means a graph like the
one of :class:`Eye` can never be translated to JAX, since it's fundamentally a
function with dynamic shapes. In other words, only PyTensor graphs with static shapes
can be translated to JAX at the moment.
