Adding JAX and Numba support for `Op`\s
=======================================

PyTensor is able to convert its graphs into JAX and Numba compiled functions. In order to do
this, each :class:`Op` in an PyTensor graph must have an equivalent JAX/Numba implementation function.

This tutorial will explain how JAX and Numba implementations are created for an :class:`Op`.  It will
focus specifically on the JAX case, but the same mechanisms are used for Numba as well.

Step 1: Identify the PyTensor :class:`Op` you'd like to implement in JAX
------------------------------------------------------------------------

Find the source for the PyTensor :class:`Op` you'd like to be supported in JAX, and
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
our JAX implementation will receive and the exact types of outputs it's expected to
return--both in terms of their data types and number of dimensions/shapes.
The actual inputs our implementation will receive are necessarily numeric values
or NumPy :class:`ndarray`\s; all that :meth:`Op.make_node` tells us is the
general signature of the underlying computation.

More specifically, the :class:`Apply` implies that there is one input that is
automatically converted to PyTensor variables via :func:`as_tensor_variable`.
There is another parameter, `axis`, that is used to determine the direction
of the operation, hence shape of the output. The check that follows imply that
`axis` must refer to a dimension in the input tensor. The input's elements
could also have any data type (e.g. floats, ints), so our JAX implementation
must be able to handle all the possible data types.

It also tells us that there's only one return value, that it has a data type
determined by :meth:`x.type()` i.e., the data type of the original tensor.
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
pay attention to this to decide whether the JAX implementation should support all variants
or raise an explicit NotImplementedError for cases that are not supported e.g., when
:class:`CumsumOp` of :class:`CumOp("add")` is supported but not :class:`CumprodOp` of
:class:`CumOp("mul")`.

Next, we look at the :meth:`Op.perform` implementation to see exactly
how the inputs and outputs are used to compute the outputs for an :class:`Op`
in Python. This method is effectively what needs to be implemented in JAX.

Step 2: Find the relevant JAX method (or something close)
---------------------------------------------------------

With a precise idea of what the PyTensor :class:`Op` does we need to figure out how
to implement it in JAX. In the best case scenario, JAX has a similarly named
function that performs exactly the same computations as the :class:`Op`. For
example, the :class:`Eye` operator has a JAX equivalent: :func:`jax.numpy.eye`
(see `the documentation <https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.eye.html?highlight=eye>`_).

If we wanted to implement an :class:`Op` like :class:`IfElse`, we might need to
recreate the functionality with some custom logic.  In many cases, at least some
custom logic is needed to reformat the inputs and outputs so that they exactly
match the `Op`'s.

Here's an example for :class:`IfElse`:

.. code:: python

   def ifelse(cond, *args, n_outs=n_outs):
       res = jax.lax.cond(
           cond, lambda _: args[:n_outs], lambda _: args[n_outs:], operand=None
       )
       return res if n_outs > 1 else res[0]

In this case, :class:`CumOp` is implemented with NumPy's :func:`numpy.cumsum`
and :func:`numpy.cumprod`, which have JAX equivalents: :func:`jax.numpy.cumsum`
and :func:`jax.numpy.cumprod`.

.. code:: python

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        if self.mode == "add":
            z[0] = np.cumsum(x, axis=self.axis)
        else:
            z[0] = np.cumprod(x, axis=self.axis)

Step 3: Register the function with the `jax_funcify` dispatcher
---------------------------------------------------------------

With the PyTensor `Op` replicated in JAX, we'll need to register the
function with the PyTensor JAX `Linker`. This is done through the use of
`singledispatch`. If you don't know how `singledispatch` works, see the
`Python documentation <https://docs.python.org/3/library/functools.html#functools.singledispatch>`_.

The relevant dispatch functions created by `singledispatch` are :func:`pytensor.link.numba.dispatch.numba_funcify` and
:func:`pytensor.link.jax.dispatch.jax_funcify`.

Here's an example for the `CumOp`\ `Op`:

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

Step 4: Write tests
-------------------

Test that your registered `Op` is working correctly by adding tests to the
appropriate test suites in PyTensor (e.g. in ``tests.link.jax`` and one of
the modules in ``tests.link.numba``). The tests should ensure that your implementation can
handle the appropriate types of inputs and produce outputs equivalent to `Op.perform`.
Check the existing tests for the general outline of these kinds of tests. In
most cases, a helper function can be used to easily verify the correspondence
between a JAX/Numba implementation and its `Op`.

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