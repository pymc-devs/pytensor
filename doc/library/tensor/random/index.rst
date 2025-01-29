
.. _libdoc_tensor_random_basic:

=============================================
:mod:`random` -- Random number functionality
=============================================

.. module:: pytensor.tensor.random
   :synopsis: symbolic random variables


The :mod:`pytensor.tensor.random` module provides random-number drawing functionality
that closely resembles the :mod:`numpy.random` module.


High-level API
==============

PyTensor assigns NumPy RNG states (i.e. `Generator` objects) to
each `RandomVariable`.  The combination of an RNG state, a specific
`RandomVariable` type (e.g. `NormalRV`), and a set of distribution parameters
uniquely defines the `RandomVariable` instances in a graph.

This means that a "stream" of distinct RNG states is required in order to
produce distinct random variables of the same kind. `RandomStream` provides a
means of generating distinct random variables in a fully reproducible way.

`RandomStream` is also designed to produce simpler graphs and work with more
sophisticated `Op`\s like `Scan`, which makes it a user-friendly random variable
interface in PyTensor.

For an example of how to use random numbers, see :ref:`Using Random Numbers <using_random_numbers>`.
For a technical explanation of how PyTensor implements random variables see :ref:`prng`.


.. class:: RandomStream()

    This is a symbolic stand-in for `numpy.random.Generator`.

    .. method:: updates()

        :returns: a list of all the (state, new_state) update pairs for the
          random variables created by this object

        This can be a convenient shortcut to enumerating all the random
        variables in a large graph in the ``update`` argument to
        `pytensor.function`.

    .. method:: seed(meta_seed)

        `meta_seed` will be used to seed a temporary random number generator,
        that will in turn generate seeds for all random variables
        created by this object (via `gen`).

        :returns: None

    .. method:: gen(op, *args, **kwargs)

        Return the random variable from ``op(*args, **kwargs)``.

        This function also adds the returned variable to an internal list so
        that it can be seeded later by a call to `seed`.

    .. method:: uniform, normal, binomial, multinomial, random_integers, ...

        See :ref: Available distributions `<_libdoc_tensor_random_distributions>`.


   .. testcode:: constructors

      from pytensor.tensor.random.utils import RandomStream

      rng = RandomStream()
      sample = rng.normal(0, 1, size=(2, 2))

      fn = pytensor.function([], sample)
      print(fn(), fn())  # different numbers due to default updates


Low-level objects
=================

.. automodule:: pytensor.tensor.random.op
   :members: RandomVariable, default_rng

.. automodule:: pytensor.tensor.random.type
   :members: RandomType, RandomGeneratorType, random_generator_type

.. automodule:: pytensor.tensor.random.var
    :members: RandomGeneratorSharedVariable
