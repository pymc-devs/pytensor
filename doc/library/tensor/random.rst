
.. _libdoc_tensor_random:

=============================================
:mod:`random` -- Random number functionality
=============================================

.. module:: pytensor.tensor.random
   :synopsis: symbolic random variables


The :mod:`pytensor.tensor.random` module provides random-number drawing functionality
that closely resembles the :mod:`numpy.random` module.


High-level API
==============

PyTensor uses RandomGeneratorVariables (symbolic equivalent to  NumPy random Generators)
to define graphs with pseudo-random draws.

Unlike NumPy, each random-draw definition returns a tuple of (next_rng, draws),
containing the symbolic draws and updated random Generator state.

Users should use next_rng for subsequent random draw operations, and for the final update expression.


For an example of how to use random numbers, see :ref:`Using Random Numbers <using_random_numbers>`.

For a technical explanation of how PyTensor implements random variables see :ref:`prng`.


Creating shared RNG variables
-----------------------------

We recommend users work with shared rngs, and use `updates` machinery to automatically
update the rng contents at the end of each function call.

.. autofunction:: pytensor.tensor.random.variable.shared_rng


.. testcode:: constructors

   init_rng = pt.random.shared_rng(seed=123, name="rng")
   next_rng, x = init_rng.uniform(-1, 1)
   final_rng, y = next_rng.normal(0, 1)

   fn = pytensor.function([], [x, y], updates={init_rng: final_rng})
   x_draw1, y_draw1 = fn()
   x_draw2, y_draw2 = fn()
   assert x_draw1 != y_draw1 != x_draw2 != y_draw2


.. testcode:: set_value

    init_rng.set_value(seed=123)
    x_draw3, y_draw3 = fn()
    assert x_draw3 == x_draw1
    assert y_draw3 == y_draw1


.. warning::
    The same RNG variable should never be used for multiple operations, as it will produce correlated draws. Always use the newly returned rng. See :ref:`rng_reuse_warning` for details.

Not respecting this will trigger a warning.


Using RNG variables
-------------------

If a user prefers to work with non-shared RNG variables, they can do so with

.. autofunction:: pytensor.tensor.random.variable.rng


:class:`RandomGeneratorVariable` objects have distribution methods that mirror
:mod:`pytensor.tensor.random` functions. Each method returns a tuple of
``(next_rng, draw)``:

.. testcode:: constructors

   init_rng = pt.random.rng()
   next_rng, x = init_rng.uniform(-1, 1)
   final_rng, y = next_rng.normal(0, 1)

   fn = pytensor.function([pytensor.In(init_rng, mutable=True)], [final_rng, x, y])

   rng_np = np.random.default_rng(123)

   rng_np, x_draw1, y_draw1 = fn(rng_np)
   x_draw2, y_draw2 = fn(rng_np)
   assert x_draw1 != y_draw1 != x_draw2 != y_draw2

   rng_np = np.random.default_rng(123)
   _, x_draw3, y_draw3 = fn(rng_np)
   assert x_draw3 == x_draw1
   assert y_draw3 == y_draw1


.. note::

    Use `pytensor.In(..., mutable=True)` on initial rng variables to allow direct inplace use by PyTensor. Otherwise a costly deepcopy will be perfomed on the node that first uses it. Alternatively using shared_rng and working with updates grants the same inplace permissions.

.. seealso::

    :ref:`Using Random Numbers <using_random_numbers>` for a tutorial with practical examples.

    :ref:`prng` for a deeper look at how PyTensor implements RNG state threading, inplace optimizations, and backend support.


Distributions
=============

All distributions are available as methods on :class:`RandomGeneratorVariable`
(e.g. ``rng.normal()``) and as module-level functions in :mod:`pytensor.tensor.random`.

.. autoclass:: pytensor.tensor.random.variable.RandomGeneratorVariable
   :members:
   :exclude-members: clone, dprint, eval
