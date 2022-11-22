:orphan:

.. _faq:

==========================
Frequently Asked Questions
==========================


Output slight numerical difference
----------------------------------

Sometimes when you compare the output of PyTensor using different PyTensor flags,
PyTensor versions, CPU and GPU devices, or with other software like NumPy, you
will see small numerical differences.

This is normal. Floating point numbers are approximations of real
numbers. This is why doing a+(b+c) vs (a+b)+c can give small
differences of value.  This is normal. For more details, see: `What
Every Computer Scientist Should Know About Floating-Point Arithmetic
<https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html>`_.


Faster gcc optimization
-----------------------

You can enable faster gcc optimization with the ``cxxflags`` option.
This list of flags was suggested on the mailing list::

    -O3 -ffast-math -ftree-loop-distribution -funroll-loops -ftracer

Use it at your own risk. Some people warned that the ``-ftree-loop-distribution`` optimization resulted in wrong results in the past.

In the past we said that if the ``compiledir`` was not shared by multiple
computers, you could add the ``-march=native`` flag. Now we recommend
to remove this flag as PyTensor does it automatically and safely,
even if the ``compiledir`` is shared by multiple computers with different
CPUs. In fact, PyTensor asks g++ what are the equivalent flags it uses, and re-uses
them directly.


.. _faster-pytensor-function-compilation:

Faster PyTensor Function Compilation
------------------------------------

PyTensor function compilation can be time consuming. It can be sped up by setting
the flag ``mode=FAST_COMPILE`` which instructs PyTensor to skip most
rewrites and disables the generation of any c/cuda code. This is useful
for quickly testing a simple idea.

If C code is necessary, the flag
``optimizer=fast_compile`` can be used instead. It instructs PyTensor to
skip time consuming rewrites but still generate C code.

Similarly using the flag ``optimizer_excluding=inplace`` will speed up
compilation by preventing rewrites that replace operations with a
version that reuses memory where it will not negatively impact the
integrity of the operation. Such rewrites can be time
consuming. However using this flag will result in greater memory usage
because space must be allocated for the results which would be
unnecessary otherwise. In short, using this flag will speed up
compilation but it will also use more memory because
``optimizer_excluding=inplace`` excludes inplace rewrites
resulting in a trade off between speed of compilation and memory
usage.

Alternatively, if the graph is big, using the flag ``cycle_detection=fast``
will speedup the computations by removing some of the inplace
rewrites. This would allow pytensor to skip a time consuming cycle
detection algorithm. If the graph is big enough,we suggest that you use
this flag instead of ``optimizer_excluding=inplace``. It will result in a
computation time that is in between fast compile and fast run.

Faster PyTensor function
------------------------

You can set the PyTensor flag :attr:`allow_gc <config.allow_gc>` to ``False`` to get a speed-up by using
more memory. By default, PyTensor frees intermediate results when we don't need
them anymore. Doing so prevents us from reusing this memory. So disabling the
garbage collection will keep all intermediate results' memory space to allow to
reuse them during the next call to the same PyTensor function, if they are of the
correct shape. The shape could change if the shapes of the inputs change.

.. _unsafe_rewrites:

Unsafe Rewrites
===============


Some PyTensor rewrites make the assumption that the user inputs are
valid. What this means is that if the user provides invalid values (like
incompatible shapes or indexing values that are out of bounds) and
the rewrites are applied, the user error will get lost. Most of the
time, the assumption is that the user inputs are valid. So it is good
to have the rewrite applied, but losing the error is bad.
The newest rewrite in PyTensor with such an assumption will add an
assertion in the graph to keep the user error message. Computing
these assertions could take some time. If you are sure everything is valid
in your graph and want the fastest possible PyTensor, you can enable a
rewrite that will remove the assertions with:
``optimizer_including=local_remove_all_assert``


Faster Small PyTensor function
------------------------------

.. note::

   For PyTensor 0.6 and up.

For PyTensor functions that don't do much work, like a regular logistic
regression, the overhead of checking the input can be significant. You
can disable it by setting ``f.trust_input`` to True.
Make sure the types of arguments you provide match those defined when
the function was compiled.

For example, replace the following

.. testcode:: faster

    import pytensor
    from pytensor import function

    x = pytensor.tensor.type.scalar('x')
    f = function([x], x + 1.)
    f(10.)

with

.. testcode:: faster

    import numpy
    import pytensor
    from pytensor import function

    x = pytensor.tensor.type.scalar('x')
    f = function([x], x + 1.)
    f.trust_input = True
    f(numpy.array([10.], dtype=pytensor.config.floatX))

Also, for small PyTensor functions, you can remove more Python overhead by
making an PyTensor function that does not take any input. You can use shared
variables to achieve this. Then you can call it like this: ``f.vm()`` or
``f.vm(n_calls=N)`` to speed it up. In the last case, only the last
function output (out of N calls) is returned.

You can also use the ``C`` linker that will put all nodes in the same C
compilation unit. This removes some overhead between node in the graph,
but requires that all nodes in the graph have a C implementation:

.. code-block:: python

    x = pytensor.tensor.type.scalar('x')
    f = function([x], (x + 1.) * 2, mode=pytensor.compile.mode.Mode(linker='c'))
    f(10.)

Related Projects
----------------

We try to list in this `wiki page <https://github.com/PyTensor/PyTensor/wiki/Related-projects>`_ other PyTensor related projects.


"What are PyTensor's Limitations?"
----------------------------------

PyTensor offers a good amount of flexibility, but has some limitations too.
You must answer for yourself the following question: How can my algorithm be cleverly written
so as to make the most of what PyTensor can do?

Here is a list of some of the known limitations:

- *While*- or *for*-Loops within an expression graph are supported, but only via
  the :func:`pytensor.scan` op (which puts restrictions on how the loop body can
  interact with the rest of the graph).

- Neither *goto* nor *recursion* is supported or planned within expression graphs.
