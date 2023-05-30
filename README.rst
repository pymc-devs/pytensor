.. image:: https://cdn.rawgit.com/pymc-devs/pytensor/main/doc/images/PyTensor_RGB.svg
    :height: 100px
    :alt: PyTensor logo
    :align: center

|Tests Status| |Coverage|

|Project Name| is a fork of `Aesara <https://github.com/aesara-devs/aesara>`__ -- a Python library that allows one to define, optimize, and
efficiently evaluate mathematical expressions involving multi-dimensional arrays.

Features
========

- A hackable, pure-Python codebase
- Extensible graph framework suitable for rapid development of custom operators and symbolic optimizations
- Implements an extensible graph transpilation framework that currently provides
  compilation via C, `JAX <https://github.com/google/jax>`__, and `Numba <https://github.com/numba/numba>`__
- Based on one of the most widely-used Python tensor libraries: `Theano <https://github.com/Theano/Theano>`__

Getting started
===============

.. code-block:: python

    import pytensor
    from pytensor import tensor as pt

    # Declare two symbolic floating-point scalars
    a = pt.dscalar("a")
    b = pt.dscalar("b")

    # Create a simple example expression
    c = a + b

    # Convert the expression into a callable object that takes `(a, b)`
    # values as input and computes the value of `c`.
    f_c = pytensor.function([a, b], c)

    assert f_c(1.5, 2.5) == 4.0

    # Compute the gradient of the example expression with respect to `a`
    dc = pytensor.grad(c, a)

    f_dc = pytensor.function([a, b], dc)

    assert f_dc(1.5, 2.5) == 1.0

    # Compiling functions with `pytensor.function` also optimizes
    # expression graphs by removing unnecessary operations and
    # replacing computations with more efficient ones.

    v = pt.vector("v")
    M = pt.matrix("M")

    d = a/a + (M + a).dot(v)

    pytensor.dprint(d)
    #  Add [id A]
    #  ├─ ExpandDims{axis=0} [id B]
    #  │  └─ True_div [id C]
    #  │     ├─ a [id D]
    #  │     └─ a [id D]
    #  └─ dot [id E]
    #     ├─ Add [id F]
    #     │  ├─ M [id G]
    #     │  └─ ExpandDims{axes=[0, 1]} [id H]
    #     │     └─ a [id D]
    #     └─ v [id I]

    f_d = pytensor.function([a, v, M], d)

    # `a/a` -> `1` and the dot product is replaced with a BLAS function
    # (i.e. CGemv)
    pytensor.dprint(f_d)
    # Add [id A] 5
    #  ├─ [1.] [id B]
    #  └─ CGemv{inplace} [id C] 4
    #     ├─ AllocEmpty{dtype='float64'} [id D] 3
    #     │  └─ Shape_i{0} [id E] 2
    #     │     └─ M [id F]
    #     ├─ 1.0 [id G]
    #     ├─ Add [id H] 1
    #     │  ├─ M [id F]
    #     │  └─ ExpandDims{axes=[0, 1]} [id I] 0
    #     │     └─ a [id J]
    #     ├─ v [id K]
    #     └─ 0.0 [id L]

See `the PyTensor documentation <https://pytensor.readthedocs.io/en/latest/>`__ for in-depth tutorials.


Installation
============

The latest release of |Project Name| can be installed from PyPI using ``pip``:

::

    pip install pytensor


Or via conda-forge:

::

    conda install -c conda-forge pytensor


The current development branch of |Project Name| can be installed from GitHub, also using ``pip``:

::

    pip install git+https://github.com/pymc-devs/pytensor


Contributing
============

We welcome bug reports and fixes and improvements to the documentation.

For more information on contributing, please see the
`contributing guide <https://pytensor.readthedocs.io/en/latest/dev_start_guide.html>`__.

A good place to start contributing is by looking through the issues
`here <https://github.com/pymc-devs/pytensor/issues>`__.


.. |Project Name| replace:: PyTensor
.. |Tests Status| image:: https://github.com/pymc-devs/pytensor/workflows/Tests/badge.svg
  :target: https://github.com/pymc-devs/pytensor/actions?query=workflow%3ATests
.. |Coverage| image:: https://codecov.io/gh/pymc-devs/pytensor/branch/main/graph/badge.svg?token=WVwr8nZYmc
  :target: https://codecov.io/gh/pymc-devs/pytensor
