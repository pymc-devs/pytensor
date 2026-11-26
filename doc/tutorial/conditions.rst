.. _tutconditions:

==========
Conditions
==========

IfElse vs Switch
================


- Both ops build a condition over symbolic variables.
- ``IfElse`` takes a *boolean* condition and two variables as inputs.
- ``Switch`` takes a *tensor* as condition and two variables as inputs.
  ``switch`` is an elementwise operation and is thus more general than ``ifelse``.
- Whereas ``switch`` evaluates both *output* variables, ``ifelse`` is lazy and only
  evaluates one variable with respect to the condition.

**Example**


.. testcode::

   from pytensor import tensor as pt
   from pytensor.conditional import ifelse
   import pytensor, time, numpy

   a,b = pt.scalars('a', 'b')
   x,y = pt.matrices('x', 'y')

   z_switch = pt.switch(pt.lt(a, b), pt.mean(x), pt.mean(y))
   z_lazy = ifelse(pt.lt(a, b), pt.mean(x), pt.mean(y))

   f_switch = pytensor.function([a, b, x, y], z_switch,
                              mode=pytensor.compile.mode.Mode(linker='vm'))
   f_lazyifelse = pytensor.function([a, b, x, y], z_lazy,
                                  mode=pytensor.compile.mode.Mode(linker='vm'))

   val1 = 0.
   val2 = 1.
   big_mat1 = numpy.ones((10000, 1000))
   big_mat2 = numpy.ones((10000, 1000))

   n_times = 10

   tic = time.perf_counter()
   for i in range(n_times):
       f_switch(val1, val2, big_mat1, big_mat2)
   print('time spent evaluating both values %f sec' % (time.perf_counter() - tic))

   tic = time.perf_counter()
   for i in range(n_times):
       f_lazyifelse(val1, val2, big_mat1, big_mat2)
   print('time spent evaluating one value %f sec' % (time.perf_counter() - tic))

.. testoutput::
   :hide:
   :options: +ELLIPSIS

   time spent evaluating both values ... sec
   time spent evaluating one value ... sec

In this example, the ``IfElse`` op spends less time (about half as much) than ``Switch``
since it computes only one variable out of the two.

.. code-block:: none

   $ python ifelse_switch.py
   time spent evaluating both values 0.6700 sec
   time spent evaluating one value 0.3500 sec

``ifelse`` is lazy in the default mode (Numba) and in the ``CVM``, ``FAST_COMPILE``
and ``JAX`` modes, as well as with ``linker='vm'`` as above. In the ``PYTORCH`` and
``MLX`` modes, ``ifelse`` computes both variables and takes as long as ``switch``.

There is no automatic rewrite replacing a ``switch`` with a
broadcasted scalar to an ``ifelse``, as this is not always faster. See
this `ticket <http://www.assembla.com/spaces/theano/tickets/764>`_.

How the backends evaluate ``ifelse``
====================================

These are implementation details and may change.

- C and Python virtual machines (``CVM``, ``FAST_COMPILE``, ``linker='vm'``): the
  ``IfElse`` thunk is lazy. The virtual machine computes the condition first and then
  only the nodes needed by the branch that is taken.
- Numba and JAX: these backends compile the whole graph into a single function, so
  there is no virtual machine to skip nodes. Instead, the ``introduce_lazy_ifelse``
  rewrite moves the computations used only by the branches into two compiled inner
  graphs and replaces the ``IfElse`` with a ``LazyIfElse``. Numba calls one of them in
  an ``if``/``else``, and JAX passes both to ``jax.lax.cond``. Computations needed by
  both branches, by the condition or by other outputs stay outside and run before
  the condition is checked.
- PyTorch and MLX: ``ifelse`` is not lazy. PyTorch computes both branches and
  returns one of them. MLX has no conditional operation and selects between the
  branches with ``mx.where``.
