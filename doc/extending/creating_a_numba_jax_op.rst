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

For example, the :class:`FillDiagonal`\ :class:`Op` current has an :meth:`Op.make_node` as follows:

.. code:: python

    def make_node(self, a, val):
        a = ptb.as_tensor_variable(a)
        val = ptb.as_tensor_variable(val)
        if a.ndim < 2:
            raise TypeError(
                "%s: first parameter must have at least"
                " two dimensions" % self.__class__.__name__
            )
        elif val.ndim != 0:
            raise TypeError(
                f"{self.__class__.__name__}: second parameter must be a scalar"
            )
        val = ptb.cast(val, dtype=upcast(a.dtype, val.dtype))
        if val.dtype != a.dtype:
            raise TypeError(
                "%s: type of second parameter must be the same as"
                " the first's" % self.__class__.__name__
            )
        return Apply(self, [a, val], [a.type()])


The :class:`Apply` instance that's returned specifies the exact types of inputs that
our JAX implementation will receive and the exact types of outputs it's expected to
return--both in terms of their data types and number of dimensions/shapes.
The actual inputs our implementation will receive are necessarily numeric values
or NumPy :class:`ndarray`\s; all that :meth:`Op.make_node` tells us is the
general signature of the underlying computation.

More specifically, the :class:`Apply` implies that the inputs come from two values that are
automatically converted to PyTensor variables via :func:`as_tensor_variable`, and
the ``assert``\s that follow imply that the first one must be a tensor with at least two
dimensions (i.e., matrix) and the second must be a scalar. According to this
logic, the inputs could have any data type (e.g. floats, ints), so our JAX
implementation must be able to handle all the possible data types.

It also tells us that there's only one return value, that it has a data type
determined by :meth:`a.type()` i.e., the data type of the original tensor.
This implies that the result is necessarily a matrix.

Next, we can look at the :meth:`Op.perform` implementation to see exactly
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

In this case, we have to use custom logic to implement the JAX version of
:class:`FillDiagonal` since JAX has no equivalent implementation. We have to use
:meth:`jax.numpy.diag_indices` to find the indices of the diagonal elements and then set
them to the value we want.

Step 3: Register the function with the `jax_funcify` dispatcher
---------------------------------------------------------------

With the PyTensor `Op` replicated in JAX, we'll need to register the
function with the PyTensor JAX `Linker`. This is done through the use of
`singledispatch`. If you don't know how `singledispatch` works, see the
`Python documentation <https://docs.python.org/3/library/functools.html#functools.singledispatch>`_.

The relevant dispatch functions created by `singledispatch` are :func:`pytensor.link.numba.dispatch.numba_funcify` and
:func:`pytensor.link.jax.dispatch.jax_funcify`.

Here's an example for the `FillDiagonal`\ `Op`:

.. code:: python

   import jax.numpy as jnp

   from pytensor.tensor.extra_ops import FillDiagonal
   from pytensor.link.jax.dispatch import jax_funcify


   @jax_funcify.register(FillDiagonal)
    def jax_funcify_FillDiagonal(op, **kwargs):
        def filldiagonal(value, diagonal):
            i, j = jnp.diag_indices(min(value.shape[-2:]))
            return value.at[..., i, j].set(diagonal)

        return filldiagonal


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

Here's a small example of a test for :class:`FillDiagonal`:

.. code:: python
    import numpy as np
    import pytensor.tensor as pt
    import pytensor.tensor.basic as ptb
    from pytensor.configdefaults import config
    from tests.link.jax.test_basic import compare_jax_and_py
    from pytensor.graph import FunctionGraph
    from pytensor.graph.op import get_test_value

    def test_jax_FillDiagonal():
        """Test JAX conversion of the `FillDiagonal` `Op`."""

        # Create a symbolic input for the first input of `FillDiagonal`
        a = pt.matrix("a")

        # Create test value tag for a
        a.tag.test_value = np.arange(9, dtype=config.floatX).reshape((3, 3))

        # Create a scalar value for the second input
        c = ptb.as_tensor(5)

        # Create the output variable
        out = pt.fill_diagonal(a, c)

        # Create a PyTensor `FunctionGraph`
        fgraph = FunctionGraph([a], [out])

        # Pass the graph and inputs to the testing function
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