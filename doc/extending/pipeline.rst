
.. _pipeline:

====================================
Overview of the compilation pipeline
====================================

Once one has an PyTensor graph, they can use :func:`pytensor.function` to compile a
function that will perform the computations modeled by the graph in Python, C,
Numba, or JAX.

More specifically, :func:`pytensor.function` takes a list of input and output
:ref:`Variables <variable>` that define the precise sub-graphs that
correspond to the desired computations.

Here is an overview of the various steps that are taken during the
compilation performed by :func:`pytensor.function`.


Step 1 - Clone the graph and collect shared variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`pytensor.function` first validates the user-supplied inputs and resolves
profiling settings.  It then calls ``construct_function_ins_and_outs`` which
clones the computation graph, discovers shared variables that appear in the
graph, applies ``givens`` substitutions, and wires up ``updates``.  The result
is a list of :class:`In` (``SymbolicInput``) objects and cloned output
variables, ready for the next stage.


Step 2 - Create a :class:`FunctionGraph` and rewrite it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`pytensor.function` resolves the :class:`Mode` and obtains the
``FunctionMaker`` class via ``mode.function_maker`` (this is overridable—for
example, :class:`DebugMode` substitutes its own maker that performs additional
validation).

``FunctionMaker.__init__`` then:

* Wraps raw inputs/outputs into ``SymbolicInput`` / ``SymbolicOutput``.
* Builds a :class:`FunctionGraph` via ``FunctionMaker.create_fgraph``, which also extracts
  update outputs from the input specs and sets up the
  ``update_mapping``.  If an existing ``fgraph`` is passed (as ``Scan``
  does for its inner loop), ``FunctionMaker.create_fgraph`` augments it with update
  outputs instead of creating a new graph.
* Applies the :term:`rewriter` produced by the :term:`mode` to the
  :class:`FunctionGraph` (via ``prepare_fgraph``).  The rewriter is
  typically obtained through a query on :attr:`optdb`.
* Configures the :term:`linker` with the rewritten graph.

Some relevant :ref:`Features <libdoc_graph_fgraphfeature>` are added to the
:class:`FunctionGraph` during this stage—for instance, features that prevent
rewrites from operating in-place on inputs declared as immutable.


Step 3 - Link the graph to obtain a VM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``FunctionMaker.create`` wires up input storage containers (sharing storage
for shared variables) and calls the :term:`linker`'s ``make_thunk`` method
with the :class:`FunctionGraph` to produce a VM (virtual machine), along with
lists of input and output containers.

Typically, the linker calls the :meth:`FunctionGraph.toposort` method in order to obtain
a linear sequence of operations to perform. How they are linked
together depends on the :class:`Linker` class used. For example, the :class:`CLinker` produces a single
block of C code for the whole computation, whereas the :class:`OpWiseCLinker`
produces one thunk for each individual operation and calls them in
sequence.

The linker is where some options take effect: the ``strict`` flag of
an input makes the associated input container do type checking. The
``borrow`` flag of an output, if ``False``, adds the output to a
``no_recycling`` list, meaning that when the thunk is called the
output containers will be cleared (if they stay there, as would be the
case if ``borrow`` was True, the thunk would be allowed to reuse—or
"recycle"—the storage).

.. note::

    Compiled libraries are stored within a specific compilation directory,
    which by default is set to ``$HOME/.pytensor/compiledir_xxx``, where
    ``xxx`` identifies the platform (under Windows the default location
    is instead ``$LOCALAPPDATA\PyTensor\compiledir_xxx``). It may be manually set
    to a different location either by setting :attr:`config.compiledir` or
    :attr:`config.base_compiledir`, either within your Python script or by
    using one of the configuration mechanisms described in :mod:`config`.

    The compile cache is based upon the C++ code of the graph to be compiled.
    So, if you change compilation configuration variables, such as
    :attr:`config.blas__ldflags`, you will need to manually remove your compile cache,
    using ``pytensor-cache clear``

    PyTensor also implements a lock mechanism that prevents multiple compilations
    within the same compilation directory (to avoid crashes with parallel
    execution of some scripts).

Step 4 - Wrap everything in a Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The VM, input containers, and output containers are wrapped in a
:class:`Function` object that presents a normal Python callable interface.
When called, :meth:`Function.__call__` places user-provided values into the
input containers, runs the VM, copies update outputs back into
shared-variable containers, and returns the results.
