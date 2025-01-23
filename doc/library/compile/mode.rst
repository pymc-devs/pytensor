
.. _libdoc_compile_mode:

======================================
:mod:`mode` -- controlling compilation
======================================

.. module:: pytensor.compile.mode
   :platform: Unix, Windows
   :synopsis: controlling compilation
.. moduleauthor:: LISA

Guide
=====

The ``mode`` parameter to :func:`pytensor.function` controls how the
inputs-to-outputs graph is transformed into a callable object.

PyTensor defines the following modes by name:

- ``'FAST_COMPILE'``: Apply just a few graph rewrites and only use Python implementations.
- ``'FAST_RUN'``: Apply all rewrites, and use C implementations where possible.
- ``NUMBA``: Apply all relevant related rewrites and compile the whole graph using Numba.
- ``JAX``: Apply all relevant rewrites and compile the whole graph using JAX.
- ``PYTORCH`` Apply all relevant rewrites and compile the whole graph using PyTorch compile.
- ``'DebugMode'``: A mode for debugging. See :ref:`DebugMode <debugmode>` for details.
- ``'NanGuardMode``: :ref:`Nan detector <nanguardmode>`
- ``'DEBUG_MODE'``: Deprecated. Use the string DebugMode.

The default mode is typically ``FAST_RUN``, but it can be controlled via the
configuration variable :attr:`config.mode`, which can be
overridden by passing the keyword argument to :func:`pytensor.function`.

For Numba, JAX, and PyTorch, we exclude rewrites that introduce C-only Ops,
as well as BLAS optimizations, as those are done automatically by the respective backends.

For JAX we also exclude fusion and inplace optimizations, as JAX does not support them
at the user level. They are performed automatically by JAX.

.. TODO::

    For a finer level of control over which rewrites are applied, and whether
    C or Python implementations are used, read.... what exactly?


Reference
=========

.. attribute:: FAST_COMPILE

.. attribute:: FAST_RUN

.. class:: Mode(object)

    Compilation is controlled by two attributes: the :attr:`optimizer` controls how
    an expression graph will be transformed; the :attr:`linker` controls how the
    rewritten expression graph will be evaluated.

    .. attribute:: optimizer

        An :class:`optimizer` instance.

    .. attribute:: linker

        A :class:`linker` instance.

    .. method:: including(*tags)

        Return a new :class:`Mode` instance like this one, but with its
        :attr:`optimizer` modified by including the given tags.

    .. method:: excluding(*tags)

        Return a new :class:`Mode` instance like this one, but with an
        :attr:`optimizer` modified by excluding the given tags.

    .. method:: requiring(*tags)

        Return a new :class:`Mode` instance like this one, but with an
        :attr:`optimizer` modified by requiring the given tags.
