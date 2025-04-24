
.. note::

    ***TODO*** Freshen up this old documentation


.. _function_inputs:

============================================
:mod:`io` - defines pytensor.function [TODO]
============================================

.. module:: pytensor.compile.io
   :platform: Unix, Windows
   :synopsis: defines In and Out
.. moduleauthor:: LISA


Inputs
======

The ``inputs`` argument to ``pytensor.function`` is a list, containing the ``Variable`` instances for which values will be specified at the time of the function call.  But inputs can be more than just Variables.
``In`` instances let us attach properties to ``Variables`` to tell function more about how to use them.


.. class:: In(object)

   .. method:: __init__(variable, name=None, value=None, update=None, mutable=False, strict=False, autoname=True, implicit=None)

      ``variable``: a Variable instance. This will be assigned a value
      before running the function, not computed from its owner.

      ``name``: Any type. (If ``autoname_input==True``, defaults to
      ``variable.name``). If ``name`` is a valid Python identifier, this input
      can be set by ``kwarg``, and its value can be accessed by
      ``self.<name>``. The default value is ``None``.

      ``value``: ``Container``. The initial value for this
        input. If update is ``None``, this input acts just like
        an argument with a default value in Python. If update is not ``None``,
        changes to this value will "stick around", whether due to an update
        or a user's explicit action.

      ``update``: Variable instance. This expression Variable will
      replace ``value`` after each function call. The default value is
      ``None``, indicating that no update is to be done.

      ``mutable``: Bool (requires value). If ``True``, permit the
      compiled function to modify the Python object being used as the
      default value. The default value is ``False``.

      ``strict``: Bool (default: ``False`` ). ``True`` means that the value
      you pass for this input must have exactly the right type. Otherwise, it
      may be cast automatically to the proper type.

      ``autoname``: Bool. If set to ``True``, if ``name`` is ``None`` and
      the Variable has a name, it will be taken as the input's
      name. If autoname is set to ``False``, the name is the exact
      value passed as the name parameter (possibly ``None``).

      ``implicit``: Bool or ``None`` (default: ``None``)
            ``True``: This input is implicit in the sense that the user is not allowed
            to provide a value for it. Requires ``value`` to be set.

            ``False``: The user can provide a value for this input. Be careful
            when ``value`` is a container, because providing an input value will
            overwrite the content of this container.

            ``None``: Automatically choose between ``True`` or ``False`` depending on the
            situation. It will be set to ``False`` in all cases except if
            ``value`` is a container (so that there is less risk of accidentally
            overwriting its content without being aware of it).


Update
------

We can define an update to modify the value of a shared variable after each function call.

>>> import pytensor.tensor as pt
>>> from pytensor import function, shared
>>> s = shared(0.0, name='s')
>>> x = pt.scalar('x')
>>> inc = function([x], s, updates={s: s + x})

Each call to ``inc`` returns the current value of ``s`` and then updates it:

>>> inc(5)         # returns 0, then s becomes 0 + 5
array(0.0)
>>> inc(3)         # returns 5, then s becomes 5 + 3
array(5.0)
>>> s.get_value()
array(8.0)

Input Argument Restrictions
---------------------------

The following restrictions apply to the inputs to ``pytensor.function``:

- Every input list element must be a valid ``In`` instance, or must be
  upgradable to a valid ``In`` instance. See the shortcut rules below.

- The same restrictions apply as in Python function definitions:
  default arguments and keyword arguments must come at the end of
  the list. Un-named mandatory arguments must come at the beginning of
  the list.

- Names have to be unique within an input list.  If multiple inputs
  have the same name, then the function will raise an exception. [***Which
  exception?**]

- Two ``In`` instances may not name the same Variable. I.e. you cannot
  give the same parameter multiple times.

If no name is specified explicitly for an In instance, then its name
will be taken from the Variable's name. Note that this feature can cause
harmless-looking input lists to not satisfy the two conditions above.
In such cases, Inputs should be named explicitly to avoid problems
such as duplicate names, and named arguments preceding unnamed ones.
This automatic naming feature can be disabled by instantiating an In
instance explicitly with the ``autoname`` flag set to False.


Shared variables and updates
----------------------------

Shared variables maintain persistent state across function calls. Use the
``updates`` argument to ``pytensor.function`` to specify how shared variables
should be modified after each call. The current value of a shared variable
can be accessed with ``get_value()`` and set with ``set_value()``.

Input Shortcuts
---------------

Every element of the inputs list will be upgraded to an In instance if necessary.

- a Variable instance ``r`` will be upgraded like ``In(r)``

.. _function_outputs:

Outputs
=======

The ``outputs`` argument to function can be one of

- ``None``, or
- a Variable or ``Out`` instance, or
- a list of Variables or ``Out`` instances.

An ``Out`` instance is a structure that lets us attach options to individual output ``Variable`` instances,
similarly to how ``In`` lets us attach options to individual input ``Variable`` instances.

**Out(variable, borrow=False)** returns an ``Out`` instance:

  * ``borrow``

    If ``True``, a reference to function's internal storage
    is OK.  A value returned for this output might be clobbered by running
    the function again, but the function might be faster.

    Default: ``False``




If a single ``Variable`` or ``Out`` instance is given as argument, then the compiled function will return a single value.

If a list of ``Variable`` or ``Out`` instances is given as argument, then the compiled function will return a list of their values.

>>> import numpy
>>> from pytensor.compile.io import Out
>>> x, y, s = pt.matrices('xys')

>>> # print a list of 2 ndarrays
>>> fn1 = pytensor.function([x], [x+x, Out((x+x).T, borrow=True)])
>>> fn1(numpy.asarray([[1,0],[0,1]]))
[array([[ 2.,  0.],
       [ 0.,  2.]]), array([[ 2.,  0.],
       [ 0.,  2.]])]

>>> # print a list of 1 ndarray
>>> fn2 = pytensor.function([x], [x+x])
>>> fn2(numpy.asarray([[1,0],[0,1]]))
[array([[ 2.,  0.],
       [ 0.,  2.]])]

>>> # print an ndarray
>>> fn3 = pytensor.function([x], outputs=x+x)
>>> fn3(numpy.asarray([[1,0],[0,1]]))
array([[ 2.,  0.],
       [ 0.,  2.]])
