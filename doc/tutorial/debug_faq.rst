
.. _debug_faq:

===========================================
Debugging PyTensor: FAQ and Troubleshooting
===========================================

There are many kinds of bugs that might come up in a computer program.
This page is structured as a FAQ.  It provides recipes to tackle common
problems, and introduces some of the tools that we use to find problems in our
own PyTensor code, and even (it happens) in PyTensor's internals, in
:ref:`using_debugmode`.

Isolating the Problem/Testing PyTensor Compiler
-----------------------------------------------

You can run your PyTensor function in a :ref:`DebugMode<using_debugmode>`.
This tests the PyTensor rewrites and helps to find where NaN, inf and other problems come from.

Interpreting Error Messages
---------------------------

Even in its default configuration, PyTensor tries to display useful error
messages. Consider the following faulty code.

.. testcode::

    import numpy as np
    import pytensor
    import pytensor.tensor as pt

    x = pt.vector()
    y = pt.vector()
    z = x + x
    z = z + y
    f = pytensor.function([x, y], z)
    f(np.ones((2,)), np.ones((3,)))

Running the code above we see:

.. testoutput::
   :options: +ELLIPSIS

   Traceback (most recent call last):
     ...
   ValueError: Input dimension mismatch. (input[0].shape[0] = 3, input[1].shape[0] = 2)
   Apply node that caused the error: Elemwise{add,no_inplace}(<TensorType(float64, (?,))>, <TensorType(float64, (?,))>, <TensorType(float64, (?,))>)
   Inputs types: [TensorType(float64, (?,)), TensorType(float64, (?,)), TensorType(float64, (?,))]
   Inputs shapes: [(3,), (2,), (2,)]
   Inputs strides: [(8,), (8,), (8,)]
   Inputs scalar values: ['not scalar', 'not scalar', 'not scalar']

   HINT: Re-running with most PyTensor optimizations disabled could give you a back-traces when this node was created. This can be done with by setting the PyTensor flags 'optimizer=fast_compile'. If that does not work, PyTensor optimizations can be disabled with 'optimizer=None'.
   HINT: Use the PyTensor flag 'exception_verbosity=high' for a debugprint of this apply node.

Arguably the most useful information is approximately half-way through
the error message, where the kind of error is displayed along with its
cause (e.g. ``ValueError: Input dimension mismatch. (input[0].shape[0] = 3, input[1].shape[0] = 2``).
Below it, some other information is given, such as the `Apply` node that
caused the error, as well as the input types, shapes, strides and
scalar values.

The two hints can also be helpful when debugging. Using the PyTensor flag
``optimizer=fast_compile`` or ``optimizer=None`` can often tell you
the faulty line, while ``exception_verbosity=high`` will display a
debug print of the apply node. Using these hints, the end of the error
message becomes :

.. code-block:: none

    Backtrace when the node is created:
      File "test0.py", line 8, in <module>
        z = z + y

    Debugprint of the apply node:
    Elemwise{add,no_inplace} [id A] <TensorType(float64, (?,))> ''
     |Elemwise{add,no_inplace} [id B] <TensorType(float64, (?,))> ''
     | |<TensorType(float64, (?,))> [id C] <TensorType(float64, (?,))>
     | |<TensorType(float64, (?,))> [id C] <TensorType(float64, (?,))>
     |<TensorType(float64, (?,))> [id D] <TensorType(float64, (?,))>

We can here see that the error can be traced back to the line ``z = z + y``.
For this example, using ``optimizer=fast_compile`` worked. If it did not,
you could set ``optimizer=None``.

"How do I print an intermediate value in a function?"
-----------------------------------------------------

PyTensor provides a :class:`Print`\ :class:`Op` to do this.

.. testcode::

    import numpy as np
    import pytensor

    x = pytensor.tensor.dvector('x')

    x_printed = pytensor.printing.Print('this is a very important value')(x)

    f = pytensor.function([x], x * 5)
    f_with_print = pytensor.function([x], x_printed * 5)

    # This runs the graph without any printing
    assert np.array_equal(f([1, 2, 3]), [5, 10, 15])

    # This runs the graph with the message, and value printed
    assert np.array_equal(f_with_print([1, 2, 3]), [5, 10, 15])

.. testoutput::

    this is a very important value __str__ = [ 1.  2.  3.]

Since PyTensor runs your program in a topological order, you won't have precise
control over the order in which multiple :class:`Print`\ `Op`\s are evaluated.  For a more
precise inspection of what's being computed where, when, and how, see the discussion
:ref:`faq_monitormode`.

.. warning::

    Using this :class:`Print`\ `Op` can prevent some PyTensor rewrites from being
    applied.  So, if you use `Print` and the graph now returns NaNs for example,
    try removing the `Print`\s to see if they're the cause or not.


"How do I print a graph (before or after compilation)?"
-------------------------------------------------------

.. TODO: dead links in the next paragraph

PyTensor provides two functions, :func:`pytensor.pp` and
:func:`pytensor.printing.debugprint`, to print a graph to the terminal before or after
compilation.  These two functions print expression graphs in different ways:
:func:`pp` is more compact and somewhat math-like, and :func:`debugprint` is more verbose and true to
the underlying graph objects being printed.
PyTensor also provides :func:`pytensor.printing.pydotprint` that creates a PNG image of the graph.

You can read about them in :ref:`libdoc_printing`.

"The function I compiled is too slow; what's up?"
-------------------------------------------------

First, make sure you're running in ``FAST_RUN`` mode. Even though
``FAST_RUN`` is the default mode, insist by passing ``mode='FAST_RUN'``
to `pytensor.function`  or by setting :attr:`config.mode`
to ``FAST_RUN``.

Second, try the PyTensor :ref:`profiling <tut_profiling>`.  This will tell you which
:class:`Apply` nodes, and which :class:`Op`\s are eating up your CPU cycles.

Tips:

* Use the flags ``floatX=float32`` to require type float32 instead of float64.
  Use the PyTensor constructors `matrix`, `vector`, etc., instead of `dmatrix`, `dvector`, etc.,
  since the latter use the default detected precision and the former use only float64.
* Check in the ``profile`` mode that there is no `Dot`\ `Op` in the post-compilation
  graph while you are multiplying two matrices of the same type. `Dot` should be
  optimized to ``dot22`` when the inputs are matrices and of the same type. This can
  still happen when using ``floatX=float32`` when one of the inputs of the graph is
  of type float64.


.. _faq_monitormode:

"How do I step through a compiled function?"
--------------------------------------------

You can use `MonitorMode` to inspect the inputs and outputs of each
node being executed when the function is called. The code snipped below
shows how to print all inputs and outputs:

.. testcode::

    import pytensor

    def inspect_inputs(fgraph, i, node, fn):
        print(i, node, "input(s) value(s):", [input[0] for input in fn.inputs],
              end='')

    def inspect_outputs(fgraph, i, node, fn):
        print(" output(s) value(s):", [output[0] for output in fn.outputs])

    x = pytensor.tensor.dscalar('x')
    f = pytensor.function([x], [5 * x],
                        mode=pytensor.compile.debug.monitormode.MonitorMode(
                            pre_func=inspect_inputs,
                            post_func=inspect_outputs))
    f(3)

.. testoutput::

    0 Elemwise{mul,no_inplace}(TensorConstant{5.0}, x) input(s) value(s): [array(5.0), array(3.0)] output(s) value(s): [array(15.0)]

When using these ``inspect_inputs`` and ``inspect_outputs`` functions
with `MonitorMode`, you should see (potentially a lot of) printed output.
Every `Apply` node will be printed out, along with its position in the graph,
the arguments to the functions `Op.perform` or `COp.c_code` and the output it
computed.
Admittedly, this may be a huge amount of output to read through if you are using
large tensors, but you can choose to add logic that would, for instance, print
something out only if a certain kind of op were used, at a certain program
position, or only if a particular value showed up in one of the inputs or
outputs.  A typical example is to detect when NaN values are added into
computations, which can be achieved as follows:

.. testcode:: compiled

    import numpy

    import pytensor

    # This is the current suggested detect_nan implementation to
    # show you how it work.  That way, you can modify it for your
    # need.  If you want exactly this method, you can use
    # `pytensor.compile.monitormode.detect_nan` that will always
    # contain the current suggested version.

    def detect_nan(fgraph, i, node, fn):
        for output in fn.outputs:
            if (not isinstance(output[0], np.ndarray) and
                np.isnan(output[0]).any()):
                print('*** NaN detected ***')
                pytensor.printing.debugprint(node)
                print('Inputs : %s' % [input[0] for input in fn.inputs])
                print('Outputs: %s' % [output[0] for output in fn.outputs])
                break

    x = pytensor.tensor.dscalar('x')
    f = pytensor.function(
        [x], [pytensor.tensor.log(x) * x],
        mode=pytensor.compile.debug.monitormode.MonitorMode(
        post_func=detect_nan)
    )
    f(0)  # log(0) * 0 = -inf * 0 = NaN

.. testoutput:: compiled
   :options: +NORMALIZE_WHITESPACE

   *** NaN detected ***
   Elemwise{Composite{(log(i0) * i0)}} [id A] ''
    |x [id B]
   Inputs : [array(0.0)]
   Outputs: [array(nan)]

To help understand what is happening in your graph, you can
disable the `local_elemwise_fusion` and all in-place
rewrites. The first is a speed optimization that merges elemwise
operations together. This makes it harder to know which particular
elemwise causes the problem. The second makes some `Op`\s'
outputs overwrite their inputs. So, if an `Op` creates a bad output, you
will not be able to see the input that was overwritten in the ``post_func``
function. To disable those rewrites, define the `MonitorMode` like this:

.. testcode:: compiled

   mode = pytensor.compile.debug.monitormode.MonitorMode(post_func=detect_nan).excluding(
       'local_elemwise_fusion', 'inplace')
   f = pytensor.function([x], [pytensor.tensor.log(x) * x],
                       mode=mode)

.. note::

    The PyTensor flags ``optimizer_including``, ``optimizer_excluding``
    and ``optimizer_requiring`` aren't used by the `MonitorMode`, they
    are used only by the ``default`` mode. You can't use the ``default``
    mode with `MonitorMode`, as you need to define what you monitor.

To be sure all inputs of the node are available during the call to
``post_func``, you must also disable the garbage collector. Otherwise,
the execution of the node can garbage collect its inputs that aren't
needed anymore by the PyTensor function. This can be done with the PyTensor
flag:

.. code-block:: python

   allow_gc=False


.. TODO: documentation for link.WrapLinkerMany


How to Use ``pdb``
------------------

In the majority of cases, you won't be executing from the interactive shell
but from a set of Python scripts. In such cases, the use of the Python
debugger can come in handy, especially as your models become more complex.
Intermediate results don't necessarily have a clear name and you can get
exceptions which are hard to decipher, due to the "compiled" nature of the
functions.

Consider this example script (``ex.py``):

.. testcode::

   import numpy as np
   import pytensor
   import pytensor.tensor as pt

   a = pt.dmatrix('a')
   b = pt.dmatrix('b')

   f = pytensor.function([a, b], [a * b])

   # Matrices chosen so dimensions are unsuitable for multiplication
   mat1 = np.arange(12).reshape((3, 4))
   mat2 = np.arange(25).reshape((5, 5))

   f(mat1, mat2)

.. testoutput::
   :hide:
   :options: +ELLIPSIS

   Traceback (most recent call last):
     ...
   ValueError: Input dimension mismatch. (input[0].shape[0] = 3, input[1].shape[0] = 5)
   Apply node that caused the error: Elemwise{mul,no_inplace}(a, b)
   Toposort index: 0
   Inputs types: [TensorType(float64, (?, ?)), TensorType(float64, (?, ?))]
   Inputs shapes: [(3, 4), (5, 5)]
   Inputs strides: [(32, 8), (40, 8)]
   Inputs values: ['not shown', 'not shown']
   Outputs clients: [['output']]

   Backtrace when the node is created:
     File "<doctest default[0]>", line 8, in <module>
       f = pytensor.function([a, b], [a * b])

   HINT: Use the PyTensor flag 'exception_verbosity=high' for a debugprint and storage map footprint of this apply node.

This is actually so simple the debugging could be done easily, but it's for
illustrative purposes. As the matrices can't be multiplied element-wise
(unsuitable shapes), we get the following exception:

.. code-block:: none

    File "ex.py", line 14, in <module>
      f(mat1, mat2)
    File "/u/username/PyTensor/pytensor/compile/function/types.py", line 451, in __call__
    File "/u/username/PyTensor/pytensor/graph/link.py", line 271, in streamline_default_f
    File "/u/username/PyTensor/pytensor/graph/link.py", line 267, in streamline_default_f
    File "/u/username/PyTensor/pytensor/graph/cc.py", line 1049, in execute ValueError: ('Input dimension mismatch. (input[0].shape[0] = 3, input[1].shape[0] = 5)', Elemwise{mul,no_inplace}(a, b), Elemwise{mul,no_inplace}(a, b))

The call stack contains some useful information to trace back the source
of the error. There's the script where the compiled function was called --
but if you're using (improperly parameterized) prebuilt modules, the error
might originate from `Op`\s in these modules, not this script. The last line
tells us about the `Op` that caused the exception. In this case it's a ``mul``
involving variables with names ``a`` and ``b``. But suppose we instead had an
intermediate result to which we hadn't given a name.

After learning a few things about the graph structure in PyTensor, we can use
the Python debugger to explore the graph, and then we can get runtime
information about the error. Matrix dimensions, especially, are useful to
pinpoint the source of the error. In the printout, there are also two of the
four dimensions of the matrices involved, but for the sake of example say we'd
need the other dimensions to pinpoint the error. First, we re-launch with the
debugger module and run the program with ``c``:

.. code-block:: text

    python -m pdb ex.py
    > /u/username/experiments/doctmp1/ex.py(1)<module>()
    -> import pytensor
    (Pdb) c

Then we get back the above error printout, but the interpreter breaks in
that state. Useful commands here are

* ``up`` and ``down`` (to move up and down the call stack),
* ``l`` (to print code around the line in the current stack position),
* ``p variable_name`` (to print the string representation of ``variable_name``),
* ``p dir(object_name)``, using the Python :func:`dir` function to print the list of an object's members

Here, for example, I do ``up``, and a simple ``l`` shows me there's a local
variable ``node``. This is the ``node`` from the computation graph, so by
following the ``node.inputs``, ``node.owner`` and ``node.outputs`` links I can
explore around the graph.

That graph is purely symbolic (no data, just symbols to manipulate it
abstractly). To get information about the actual parameters, you explore the
"thunk" objects, which bind the storage for the inputs (and outputs) with the
function itself (a "thunk" is a concept related to closures). Here, to get the
current node's first input's shape, you'd therefore do
``p thunk.inputs[0][0].shape``, which prints out ``(3, 4)``.

.. _faq_dump_fct:

Dumping a Function to help debug
--------------------------------

If you are reading this, there is high chance that you emailed our
mailing list and we asked you to read this section. This section
explain how to dump all the parameter passed to
:func:`pytensor.function`. This is useful to help us reproduce a problem
during compilation and it doesn't request you to make a self contained
example.

For this to work, we need to be able to import the code for all `Op` in
the graph. So if you create your own `Op`, we will need this
code; otherwise, we won't be able to unpickle it.

.. code-block:: python

    # Replace this line:
    pytensor.function(...)
    # with
    pytensor.function_dump(filename, ...)
    # Where `filename` is a string to a file that we will write to.

Then send us ``filename``.


Breakpoint during PyTensor function execution
---------------------------------------------

You can set a breakpoint during the execution of an PyTensor function with
:class:`PdbBreakpoint <pytensor.breakpoint.PdbBreakpoint>`.
:class:`PdbBreakpoint <pytensor.breakpoint.PdbBreakpoint>` automatically
detects available debuggers and uses the first available in the following order:
`pudb`, `ipdb`, or `pdb`.
