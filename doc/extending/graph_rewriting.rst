
.. _graph_rewriting:

===============
Graph Rewriting
===============

In this document we will explain how graph rewriting works and how graph
rewrites can be constructed in PyTensor.

.. todo::
   The old "optimization" nomenclature is still in use throughout some of these
   documents and the codebase; however, this is being changed to more accurately
   distinguish between general graph rewriting for any purpose and the kind that
   is explicitly intended to "optimize" a graph in some way.


Graph and Node Rewriters
========================

There are two types of basic rewriters: *graph* rewriters and *node* rewriters.

A graph rewriter takes a :class:`FunctionGraph` object (see its
:doc:`documentation </library/graph/fgraph>` for more details) and navigates through it
in a suitable way, replacing some :class:`Variable`\s by others in the process.
A node rewriter, on the other hand, is defined as a function on a
*single* :ref:`apply` node and must return either ``False`` (to mean that
nothing is to be done) or a list of new :class:`Variable`\s that we would like to
substitute for the node's current outputs.

Some graph rewriters navigate the computation graph in a particular fashion
(e.g. in topological order, reverse-topological order, random order, etc.) and
apply one or more node rewriters at each step.  :class:`WalkingGraphRewriter` is
one such example.

Rewriters that are holistic, meaning that they must take into
account dependencies that might be all over the graph, should usually be
graph rewriters. Rewrites that only need a narrow view of sub-graphs are
better defined as node rewrites.

.. rewriter:

Graph Rewriting
---------------

.. class:: GraphRewriter

    .. method:: apply(fgraph)

      This method takes a :class:`FunctionGraph` object which contains the computation graph
      and does modifications in line with what the rewriter is meant
      to do. This is one of the main methods of the rewriter.

    .. method:: add_requirements(fgraph)

      This method takes a :class:`FunctionGraph` object and adds :ref:`features
      <libdoc_graph_fgraphfeature>` to it. These features are "plugins" that are needed
      for the :meth:`GraphRewriter.apply` method to do its job properly.

    .. method:: rewrite(fgraph)

      This is the interface function called by PyTensor.  It calls
      :meth:`GraphRewriter.apply` by default.


Node Rewriting
--------------

A node rewriter is an object which defines the following methods:

.. class:: NodeRewriter

    .. method:: transform(fgraph, node)

      This method takes a :class:`FunctionGraph` and an :class:`Apply` node and
      returns either ``False`` to signify that no changes are to be done or a
      list of :class:`Variable`\s which matches the length of the node's ``outputs``
      list. When the :class:`NodeRewriter` is applied by a :class:`NodeProcessingGraphRewriter`, the outputs
      of the node passed as argument to the :class:`NodeRewriter` will be replaced by
      the list returned.


A Simplification Rule
=====================

For starters, let's define the following simplification:

.. math::

   \frac{xy}{y} = x

We will implement it in three ways: using a graph rewriter, a node rewriter with
a :class:`NodeProcessingGraphRewriter`, and then using the
:class:`PatternNodeRewriter`.

Graph Rewriter Implementation
-----------------------------

Here is the code for a graph rewriter implementing the
simplification described above:

.. testcode::

   import pytensor
   from pytensor.graph.rewriting.basic import GraphRewriter
   from pytensor.graph.features import ReplaceValidate

   class Simplify(GraphRewriter):
       def add_requirements(self, fgraph):
           fgraph.attach_feature(ReplaceValidate())

       def apply(self, fgraph):
           for node in fgraph.toposort():
               if node.op == true_div:
                   x, y = node.inputs
                   z = node.outputs[0]
                   if x.owner and x.owner.op == mul:
                       a, b = x.owner.inputs
                       if y == a:
                           fgraph.replace_validate(z, b)
                       elif y == b:
                           fgraph.replace_validate(z, a)

   simplify = Simplify()


Here's how it works: first, in :meth:`add_requirements`, we add the
:class:`ReplaceValidate` :class:`Feature` located in
:ref:`libdoc_graph_features`. This feature adds the :meth:`replace_validate`
method to ``fgraph``, which is an enhanced version of :meth:`FunctionGraph.replace` that
does additional checks to ensure that we are not messing up the
computation graph.

In a nutshell, :class:`ReplaceValidate` grants access to :meth:`fgraph.replace_validate`,
and :meth:`fgraph.replace_validate` allows us to replace a :class:`Variable` with
another while respecting certain validation constraints. As an
exercise, try to rewrite :class:`Simplify` using :class:`WalkingGraphRewriter`. (Hint: you
want to use the method it publishes instead of the call to toposort)

Then, in :meth:`GraphRewriter.apply` we do the actual job of simplification. We start by
iterating through the graph in topological order. For each node
encountered, we check if it's a ``div`` node. If not, we have nothing
to do here. If so, we put in ``x``, ``y`` and ``z`` the numerator,
denominator and quotient (output) of the division.
The simplification only occurs when the numerator is a multiplication,
so we check for that. If the numerator is a multiplication we put the
two operands in ``a`` and ``b``, so
we can now say that ``z == (a*b)/y``. If ``y==a`` then ``z==b`` and if
``y==b`` then ``z==a``. When either case happens then we can replace
``z`` by either ``a`` or ``b`` using :meth:`FunctionGraph.replace_validate`; otherwise, we do
nothing.

Now, we test the rewriter:

>>> from pytensor.scalar import float64, add, mul, true_div
>>> x = float64('x')
>>> y = float64('y')
>>> z = float64('z')
>>> a = add(z, mul(true_div(mul(y, x), y), true_div(z, x)))
>>> e = pytensor.graph.fg.FunctionGraph([x, y, z], [a])
>>> e
FunctionGraph(add(z, mul(true_div(mul(y, x), y), true_div(z, x))))
>>> simplify.rewrite(e)
>>> e
FunctionGraph(add(z, mul(x, true_div(z, x))))

You can check what happens if you put many
instances of :math:`\frac{xy}{y}` in the graph. Note that it sometimes
won't work for reasons that have nothing to do with the quality of the
rewrite you wrote. For example, consider the following:

>>> x = float64('x')
>>> y = float64('y')
>>> z = float64('z')
>>> a = true_div(mul(add(y, z), x), add(y, z))
>>> e = pytensor.graph.fg.FunctionGraph([x, y, z], [a])
>>> e
FunctionGraph(true_div(mul(add(y, z), x), add(y, z)))
>>> simplify.rewrite(e)
>>> e
FunctionGraph(true_div(mul(add(y, z), x), add(y, z)))

Nothing happened here. The reason is: ``add(y, z) != add(y,
z)``. That is the case for efficiency reasons. To fix this problem we
first need to merge the parts of the graph that represent the same
computation, using the :class:`MergeOptimizer` defined in
:mod:`pytensor.graph.rewriting.basic`.

>>> from pytensor.graph.rewriting.basic import MergeOptimizer
>>> MergeOptimizer().rewrite(e)  # doctest: +ELLIPSIS
(0, ..., None, None, {}, 1, 0)
>>> e
FunctionGraph(true_div(mul(*1 -> add(y, z), x), *1))
>>> simplify.rewrite(e)
>>> e
FunctionGraph(x)

Once the merge is done, both occurrences of ``add(y, z)`` are
collapsed into a single one and is used as an input in two
places. Note that ``add(x, y)`` and ``add(y, x)`` are still considered
to be different because PyTensor has no clue that ``add`` is
commutative. You may write your own graph rewrite to identify
computations that are identical with full knowledge of the rules of
arithmetic that your Ops implement. PyTensor might provide facilities
for this somewhere in the future.

.. note::

   :class:`FunctionGraph` is an PyTensor structure intended for the rewrite
   phase. It is used internally by :func:`pytensor.function` and is rarely
   exposed to the end user.


Node Rewriter Implementation
----------------------------

The local version of the above code would be the following:


.. testcode::

   from pytensor.graph.rewriting.basic import NodeRewriter


   class LocalSimplify(NodeRewriter):
       def transform(self, fgraph, node):
           if node.op == true_div:
               x, y = node.inputs
               if x.owner and x.owner.op == mul:
                   a, b = x.owner.inputs
                   if y == a:
                       return [b]
                   elif y == b:
                       return [a]
           return False

       def tracks(self):
           # This tells certain navigators to only apply this `NodeRewriter`
           # on these kinds of `Op`s
           return [true_div]

   local_simplify = LocalSimplify()


In this case, the transformation is defined in the
:meth:`NodeRewriter.transform` method, which is given an explicit
:class:`Apply` node on which to work.  The entire graph--as a ``fgraph``--is
also provided, in case global information is needed.

If no changes are to be made, ``False`` must be returned; otherwise, a list of replacements for the node's
outputs are returned. This list must have the same length as
:attr:`node.outputs`. If one of :attr:`node.outputs` doesn't have clients
(e.g. available via ``fgraph.clients``), then it is not used elsewhere in the graph and
you can put ``None`` in the returned list to remove it.

In order to apply the node rewriter throughout a graph, we use it in conjunction
with a :class:`NodeProcessingGraphRewriter`.  A :class:`NodeProcessingGraphRewriter` is
a graph rewriter that loops through all nodes in the graph (or a well-defined
subset of them) and applies one or several node rewriters.

>>> x = float64('x')
>>> y = float64('y')
>>> z = float64('z')
>>> a = add(z, mul(true_div(mul(y, x), y), true_div(z, x)))
>>> e = pytensor.graph.fg.FunctionGraph([x, y, z], [a])
>>> e
FunctionGraph(add(z, mul(true_div(mul(y, x), y), true_div(z, x))))
>>> simplify = pytensor.graph.rewriting.basic.WalkingGraphRewriter(local_simplify)
>>> simplify.rewrite(e)
(<pytensor.graph.rewriting.basic.WalkingGraphRewriter object at 0x...>, 1, 5, 3, ..., ..., ...)
>>> e
FunctionGraph(add(z, mul(x, true_div(z, x))))

:class:`SubstitutionNodeRewriter`, :class:`RemovalNodeRewriter`, :class:`PatternNodeRewriter`
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

PyTensor defines some shortcuts to make :class:`NodeRewriter`\s:

.. function:: SubstitutionNodeRewriter(op1, op2)

  Replaces all uses of ``op1`` by ``op2``. In other
  words, the outputs of all :class:`Apply` nodes using ``op1`` by the outputs
  of :class:`Apply` nodes involving ``op2``, where their inputs are the same.

.. function:: RemovalNodeRewriter(op)

  Removes all uses of ``op`` in the following way:
  if ``y = op(x)`` then ``y`` is replaced by ``x``. ``op`` must have as many
  outputs as it has inputs. The first output becomes the first input,
  the second output becomes the second input, and so on.

.. function:: PatternNodeRewriter(pattern1, pattern2)

  Replaces all occurrences of the first pattern by the second pattern.
  See :class:`PatternNodeRewriter`.

.. code::

   from pytensor.scalar import identity
   from pytensor.graph.rewriting.basic import SubstitutionNodeRewriter, RemovalNodeRewriter, PatternNodeRewriter

   # Replacing `add` by `mul` (this is not recommended for primarily
   # mathematical reasons):
   add_to_mul = SubstitutionNodeRewriter(add, mul)

   # Removing `identity`
   remove_identity = RemovalNodeRewriter(identity)

   # The "simplify" operation we've been defining in the past few
   # sections. Note that we need two patterns to account for the
   # permutations of the arguments to `mul`.
   local_simplify_1 = PatternNodeRewriter((true_div, (mul, 'x', 'y'), 'y'), 'x')
   local_simplify_2 = PatternNodeRewriter((true_div, (mul, 'x', 'y'), 'x'), 'y')

.. note::

   :class:`SubstitutionNodeRewriter`, :class:`RemovalNodeRewriter` and :class:`PatternNodeRewriter` produce node rewriters, which
   means that everything we said previously about node rewriters
   apply (e.g. they need to be wrapped in a :class:`NodeProcessingGraphRewriter`, etc.)


When a rewriter can be naturally expressed using :class:`SubstitutionNodeRewriter`, :class:`RemovalNodeRewriter`
or :class:`PatternNodeRewriter`, it is highly recommended to use them.

.. _unification:

Unification, reification and miniKanren
=======================================

For non-trivial pattern matching and replacement, PyTensor optionally exposes
:func:`unify` / :func:`reify` over :mod:`logical-unification` and miniKanren
relations via :class:`KanrenRelationSub`. These rely on a set of optional
packages (:mod:`logical-unification`, :mod:`kanren`, :mod:`etuples`,
:mod:`cons`) and are documented separately in :ref:`unification_kanren`.

.. _testing_rewrites:

Testing Rewrites
================

.. autoclass:: tests.unittest_tools.RewriteTester


.. _optdb:

The Optimization Database (:obj:`optdb`)
========================================

PyTensor exports a symbol called :obj:`optdb` which acts as a sort of ordered
database of rewrites. When a new rewrite is constructed, it must be inserted at
the proper place in the database in order for it to be deployed during function
compilation.

Each rewrite in a database can be assigned a set of tags that serve as a basis
for filtering/querying.

The point of :obj:`optdb` is that one might want to apply many rewrites
to a graph in many unique patterns.

For example, one might want to perform rewrite X, then rewrite Y, then
rewrite Z. Perhaps rewrite Y is an :class:`EquilibriumGraphRewriter` containing
:class:`NodeRewriter`\s A, B and C, which are applied on every node of until
they all fail to change it. If some rewrites fail, we may want an easy way to
turn them off. Similarly, if some rewrites are very CPU-intensive and we don't
want to take the time to apply them, then we should be able to disable them.

The :obj:`optdb` system allows us to tag each rewrite with a unique name,
as well as informative descriptions such as 'stable', 'buggy' or
'cpu_intensive'.

For instance, the rewrite tag ``cxx_only`` is used for rewrites that
insert :class:`Op`\s that have no Python implementation (i.e. they only have C
implementations).  Rewrites with this tag can be skipped when the C backend
is not being used.


Definition of :obj:`optdb`
--------------------------

:obj:`optdb` is an object which is an instance of
:class:`SequenceDB`,
itself a subclass of :class:`RewriteDatabase`.
There exist (for now) two types of :class:`RewriteDatabase`, :class:`SequenceDB` and :class:`EquilibriumDB`.
When given an appropriate :class:`RewriteDatabaseQuery`, :class:`RewriteDatabase` objects build an :class:`Rewriter` matching
the query.

A :class:`SequenceDB` contains :class:`Rewriter` or :class:`RewriteDatabase` objects. Each of them
has a name, an arbitrary number of tags and an integer representing their order
in the sequence. When a :class:`RewriteDatabaseQuery` is applied to a :class:`SequenceDB`, all :class:`Rewriter`\s whose
tags match the query are inserted in proper order in a :class:`SequenceRewriter`, which
is returned. If the :class:`SequenceDB` contains :class:`RewriteDatabase`
instances, the :class:`RewriteDatabaseQuery` will be passed to them as well and the
rewriters they return will be put in their places.

An :class:`EquilibriumDB` contains :class:`NodeRewriter` or :class:`RewriteDatabase` objects. Each of them
has a name and an arbitrary number of tags. When a :class:`RewriteDatabaseQuery` is applied to
an :class:`EquilibriumDB`, all :class:`NodeRewriter`\s that match the query are
inserted into an :class:`EquilibriumGraphRewriter`, which is returned. If the
:class:`EquilibriumDB` contains :class:`RewriteDatabase` instances, the
:class:`RewriteDatabaseQuery` will be passed to them as well and the
:class:`NodeRewriter`\s they return will be put in their places
(note that as of yet no :class:`RewriteDatabase` can produce :class:`NodeRewriter` objects, so this
is a moot point).

PyTensor contains one principal :class:`RewriteDatabase` object, :class:`optdb`, which
contains all of PyTensor's rewriters with proper tags. It is
recommended to insert new :class:`Rewriter`\s in it. As mentioned previously,
:obj:`optdb` is a :class:`SequenceDB`, so, at the top level, PyTensor applies a sequence
of graph rewrites to the graphs it compiles.


:class:`RewriteDatabaseQuery`
-----------------------------

A :class:`RewriteDatabaseQuery` is built by the following call:

.. code-block:: python

   pytensor.graph.rewriting.db.RewriteDatabaseQuery(include, require=None, exclude=None, subquery=None)

.. class:: RewriteDatabaseQuery

    .. attribute:: include

       A set of tags (a tag being a string) such that every
       rewrite obtained through this :class:`RewriteDatabaseQuery` must have **one** of the tags
       listed. This field is required and basically acts as a starting point
       for the search.

    .. attribute:: require

       A set of tags such that every rewrite obtained
       through this :class:`RewriteDatabaseQuery` must have **all** of these tags.

    .. attribute:: exclude

       A set of tags such that every rewrite obtained
       through this :class:`RewriteDatabaseQuery` must have **none** of these tags.

    .. attribute:: subquery

       :obj:`optdb` can contain sub-databases; subquery is a
       dictionary mapping the name of a sub-database to a special :class:`RewriteDatabaseQuery`.
       If no subquery is given for a sub-database, the original :class:`RewriteDatabaseQuery` will be
       used again.

Furthermore, a :class:`RewriteDatabaseQuery` object includes three methods, :meth:`including`,
:meth:`requiring` and :meth:`excluding`, which each produce a new :class:`RewriteDatabaseQuery` object
with the include, require, and exclude sets refined to contain the new entries.


Examples
--------

Here are a few examples of how to use a :class:`RewriteDatabaseQuery` on :obj:`optdb` to produce an
:class:`Rewriter`:

.. testcode::

   from pytensor.graph.rewriting.db import RewriteDatabaseQuery
   from pytensor.compile import optdb

   # This is how the rewrites for the fast_run mode are defined
   fast_run = optdb.query(RewriteDatabaseQuery(include=['fast_run']))

   # This is how the rewrites for the fast_compile mode are defined
   fast_compile = optdb.query(RewriteDatabaseQuery(include=['fast_compile']))

   # This is the same as fast_run but no rewrites will replace
   # any operation by an inplace version. This assumes, of course,
   # that all inplace operations are tagged as 'inplace' (as they
   # should!)
   fast_run_no_inplace = optdb.query(RewriteDatabaseQuery(include=['fast_run'],
                                           exclude=['inplace']))


Registering a :class:`Rewriter`
---------------------------------

Let's say we have a graph rewriter called ``simplify``. We can add
it to :obj:`optdb` as follows:

.. testcode::

   optdb.register('simplify', simplify, 'fast_run', position=0.5)

Once this is done, the ``FAST_RUN`` mode will automatically include the
rewrite, since it was given the ``'fast_run'`` tag. Of course,
already-compiled functions will see no change. The ``position`` parameter
is specific to the type of rewrite database that :obj:`obtdb` is, and
is explained in :ref:`optdb-structure`.



Registering a :class:`NodeRewriter`
-----------------------------------

:class:`NodeRewriter`\s may be registered in two ways:

* Wrap them in a :class:`NodeProcessingGraphRewriter` and insert them like a graph rewriter
  (see previous section).
* Put them in an :class:`EquilibriumDB`.

PyTensor defines two :class:`EquilibriumDB`\s in which one can put node
rewrites:


.. function:: canonicalize

  This contains rewrites that aim to put graphs in a standard "canonical" form:

  * Replace rare or esoterical operations with their equivalents using
    elementary operations.

  * Order operations in a canonical way.
    For example, any sequence of multiplications and divisions can be rewritten to contain at most
    one division (e.g. ``x * x`` can be rewritten to ``x**2``, etc.)

  * Fold constants (e.g. ``Constant(2) * Constant(2)`` becomes ``Constant(4)``).


.. function:: specialize

  This contains rewrites that aim to *specialize* the graph:

  * Replace a combination of operations with a special operation that
    does the same thing (but better).


For each group, all rewrites of the group that are selected by
the :class:`RewriteDatabaseQuery` will be applied on the graph over and over
again until no changes are made.

When using :class:`EquilibriumDB`, be sure to check carefully that your rewrite
leads to a fixed-point (i.e. a graph for which the rewrite cannot be applied
anymore), at which point it returns ``False`` to indicate its job is done. Also
be careful not to undo the work of another rewrites in the group, because the
graph will oscillate between two or more states and nothing will get done.


.. _optdb-structure:

:obj:`optdb` structure
----------------------

:obj:`optdb` contains the following :class:`Rewriters`\s and sub-DBs, with the given
priorities and tags:

+-------+---------------------+------------------------------+
| Order | Name                | Description                  |
+=======+=====================+==============================+
| 0     | merge1              | First merge operation        |
+-------+---------------------+------------------------------+
| 1     | canonicalize        | Simplify the graph           |
+-------+---------------------+------------------------------+
| 2     | specialize          | Add specialized operations   |
+-------+---------------------+------------------------------+
| 49    | merge2              | Second merge operation       |
+-------+---------------------+------------------------------+
| 49.5  | add_destroy_handler | Enable inplace rewrites      |
+-------+---------------------+------------------------------+
| 100   | merge3              | Third merge operation        |
+-------+---------------------+------------------------------+

The merge operations are meant to put together parts of the graph that
represent the same computation. Since rewrites can modify the
graph in such a way that two previously different-looking parts of the
graph become similar, we merge at the beginning, in the middle and at
the very end. Technically, we only really need to do it at the end,
but doing it in previous steps reduces the size of the graph and
therefore increases the efficiency of the process.

See previous section for more information about the canonicalize and
specialize steps.

The ``add_destroy_handler`` step is not really an rewrite. It is
a marker. Basically:

.. warning::

   Any rewrite which inserts inplace operations in the
   computation graph must appear **after** the ``add_destroy_handler``
   "rewriter". In other words, the priority of any such rewrites
   must be **>= 50**. Failure to comply by this restriction can lead
   to the creation of incorrect computation graphs.

The reason the destroy handler is not inserted at the beginning is
that it is costly to run. It is cheaper to run most rewrites
under the assumption there are no inplace operations.


.. _node_processing_rewriter:

:class:`NodeProcessingGraphRewriter`
------------------------------------

.. autoclass:: pytensor.graph.rewriting.basic.NodeProcessingGraphRewriter
    :noindex:


.. _profiling_rewrite:

Profiling PyTensor Function Compilation
=======================================

If one finds that compiling an PyTensor function is taking too much time,
profiling information about each PyTensor rewrite can be obtained. The normal
:ref:`PyTensor profiler <tut_profiling>` provides some
high-level performance information. The indentation shows the included in/subset
relationship between sections. The top of its output look like this:

.. code-block:: none

    Function profiling
    ==================
      Message: PATH_TO_A_FILE:23
      Time in 0 calls to Function.__call__: 0.000000e+00s
      Total compile time: 1.131874e+01s
        Number of Apply nodes: 50
        PyTensor rewriter time: 1.152431e+00s
           PyTensor validate time: 2.790451e-02s
        PyTensor Linker time (includes C, CUDA code generation/compiling): 7.893991e-02s
           Import time 1.153541e-02s
      Time in all call to pytensor.grad() 4.732513e-02s

Explanations:

* ``Total compile time: 1.131874e+01s`` gives the total time spent inside `pytensor.function`.
* ``Number of Apply nodes: 50`` means that after rewriting, there are 50 apply node in the graph.
* ``PyTensor rewrite time: 1.152431e+00s`` means that we spend 1.15s in the rewriting phase of `pytensor.function`.
* ``PyTensor validate time: 2.790451e-02s`` means that we spent 2.8e-2s in the validation phase of rewriting.
* ``PyTensor Linker time (includes C code generation/compiling): 7.893991e-02s`` means that we spent 7.9e-2s in linker phase of `pytensor.function`.
* ``Import time 1.153541e-02s`` is a subset of the linker time where we import the compiled module.
* ``Time in all call to pytensor.grad() 4.732513e-02s`` tells that we spent a total of 4.7e-2s in all calls to `pytensor.grad`. This is outside of the calls to `pytensor.function`.

The *linker* phase includes the generation of the C code, the time spent
by g++ to compile and the time needed by PyTensor to build the object we
return. The C code generation and compilation is cached, so the first
time you compile a function and the following ones could take different
amount of execution time.

Detailed Profiling of PyTensor Rewrites
---------------------------------------

You can get more detailed profiling information about the PyTensor
rewriting phase by setting to ``True`` the PyTensor flags
:attr:`config.profile_optimizer` (this requires ``config.profile`` to be ``True``
as well).

This will output something like this:

.. code-block:: none

    Rewriter Profile
    ----------------
     SequentialGraphRewriter  OPT_FAST_RUN  time 1.152s for 123/50 nodes before/after rewriting
       0.028s for fgraph.validate()
       0.131s for callback
       time      - (name, class, index) - validate time
       0.751816s - ('canonicalize', 'EquilibriumGraphRewriter', 4) - 0.004s
         EquilibriumGraphRewriter      canonicalize
           time 0.751s for 14 passes
           nb nodes (start, end,  max) 108 81 117
           time io_toposort 0.029s
           time in node rewriters 0.687s
           time in graph rewriters 0.010s
            0 - 0.050s 27 (0.000s in global rewrites, 0.002s io_toposort) - 108 nodes - ('local_dimshuffle_lift', 9) ('local_upcast_elemwise_constant_inputs', 5) ('local_shape_to_shape_i', 3) ('local_fill_sink', 3) ('local_fill_to_alloc', 2) ...
            1 - 0.288s 26 (0.002s in global rewrites, 0.002s io_toposort) - 117 nodes - ('local_dimshuffle_lift', 8) ('local_fill_sink', 4) ('constant_folding', 4) ('local_useless_elemwise', 3) ('local_subtensor_make_vector', 3) ...
            2 - 0.044s 13 (0.002s in global rewrites, 0.003s io_toposort) - 96 nodes - ('constant_folding', 4) ('local_dimshuffle_lift', 3) ('local_fill_sink', 3) ('local_useless_elemwise', 1) ('local_fill_to_alloc', 1) ...
            3 - 0.045s 11 (0.000s in global rewrites, 0.002s io_toposort) - 91 nodes - ('constant_folding', 3) ('local_fill_to_alloc', 2) ('local_dimshuffle_lift', 2) ('local_mul_canonizer', 2) ('MergeOptimizer', 1) ...
            4 - 0.035s 8 (0.002s in global rewrites, 0.002s io_toposort) - 93 nodes - ('local_fill_sink', 3) ('local_dimshuffle_lift', 2) ('local_fill_to_alloc', 1) ('MergeOptimizer', 1) ('constant_folding', 1)
            5 - 0.035s 6 (0.000s in global rewrites, 0.002s io_toposort) - 88 nodes - ('local_fill_sink', 2) ('local_dimshuffle_lift', 2) ('local_fill_to_alloc', 1) ('local_mul_canonizer', 1)
            6 - 0.038s 10 (0.001s in global rewrites, 0.002s io_toposort) - 95 nodes - ('local_fill_sink', 3) ('local_dimshuffle_lift', 3) ('constant_folding', 2) ('local_fill_to_alloc', 1) ('MergeOptimizer', 1)
            7 - 0.032s 5 (0.001s in global rewrites, 0.002s io_toposort) - 91 nodes - ('local_fill_sink', 3) ('MergeOptimizer', 1) ('local_dimshuffle_lift', 1)
            8 - 0.034s 5 (0.000s in global rewrites, 0.002s io_toposort) - 92 nodes - ('local_fill_sink', 3) ('MergeOptimizer', 1) ('local_greedy_distributor', 1)
            9 - 0.031s 6 (0.001s in global rewrites, 0.002s io_toposort) - 90 nodes - ('local_fill_sink', 2) ('local_fill_to_alloc', 1) ('MergeOptimizer', 1) ('local_dimshuffle_lift', 1) ('local_greedy_distributor', 1)
           10 - 0.032s 5 (0.000s in global rewrites, 0.002s io_toposort) - 89 nodes - ('local_dimshuffle_lift', 2) ('local_fill_to_alloc', 1) ('MergeOptimizer', 1) ('local_fill_sink', 1)
           11 - 0.030s 5 (0.000s in global rewrites, 0.002s io_toposort) - 88 nodes - ('local_dimshuffle_lift', 2) ('local_fill_to_alloc', 1) ('MergeOptimizer', 1) ('constant_folding', 1)
           12 - 0.026s 1 (0.000s in global rewrites, 0.003s io_toposort) - 81 nodes - ('MergeOptimizer', 1)
           13 - 0.031s 0 (0.000s in global rewrites, 0.003s io_toposort) - 81 nodes -
           times - times applied - nb node created - name:
           0.263s - 15 - 0 - constant_folding
           0.096s - 2 - 14 - local_greedy_distributor
           0.066s - 4 - 19 - local_mul_canonizer
           0.046s - 28 - 57 - local_fill_sink
           0.042s - 35 - 78 - local_dimshuffle_lift
           0.018s - 5 - 15 - local_upcast_elemwise_constant_inputs
           0.010s - 11 - 4 - MergeOptimizer
           0.009s - 4 - 0 - local_useless_elemwise
           0.005s - 11 - 2 - local_fill_to_alloc
           0.004s - 3 - 6 - local_neg_to_mul
           0.002s - 1 - 3 - local_lift_transpose_through_dot
           0.002s - 3 - 4 - local_shape_to_shape_i
           0.002s - 2 - 4 - local_subtensor_lift
           0.001s - 3 - 0 - local_subtensor_make_vector
           0.001s - 1 - 1 - local_sum_all_to_none
           0.131s - in 62 rewrite(s) that where not used (display only those with a runtime > 0)
             0.050s - local_add_canonizer
             0.018s - local_mul_zero
             0.016s - local_one_minus_erf
             0.010s - local_func_inv
             0.006s - local_0_dot_x
             0.005s - local_track_shape_i
             0.004s - local_mul_switch_sink
             0.004s - local_fill_cut
             0.004s - local_one_minus_erf2
             0.003s - local_remove_switch_const_cond
             0.003s - local_cast_cast
             0.002s - local_IncSubtensor_serialize
             0.001s - local_sum_div_dimshuffle
             0.001s - local_div_switch_sink
             0.001s - local_dimshuffle_no_inplace_at_canonicalize
             0.001s - local_cut_useless_reduce
             0.001s - local_reduce_join
             0.000s - local_sum_sum
             0.000s - local_useless_alloc
             0.000s - local_reshape_chain
             0.000s - local_useless_subtensor
             0.000s - local_reshape_lift
             0.000s - local_flatten_lift
             0.000s - local_useless_slice
             0.000s - local_subtensor_of_alloc
             0.000s - local_subtensor_of_dot
             0.000s - local_subtensor_merge
       0.101733s - ('elemwise_fusion', 'SequentialGraphRewriter', 13) - 0.000s
         SequentialGraphRewriter      elemwise_fusion  time 0.102s for 78/50 nodes before/after rewriting
           0.000s for fgraph.validate()
           0.004s for callback
           0.095307s - ('composite_elemwise_fusion', 'FusionOptimizer', 1) - 0.000s
             FusionOptimizer
              nb_iter 3
              nb_replacement 10
              nb_inconsistency_replace 0
              validate_time 0.000249624252319
              callback_time 0.00316381454468
              time_toposort 0.00375390052795
           0.006412s - ('local_add_mul_fusion', 'FusionOptimizer', 0) - 0.000s
             FusionOptimizer
              nb_iter 2
              nb_replacement 3
              nb_inconsistency_replace 0
              validate_time 6.43730163574e-05
              callback_time 0.000783205032349
              time_toposort 0.0035240650177
       0.090089s - ('inplace_elemwise_optimizer', 'FromFunctionGraphRewriter', 30) - 0.019s
       0.048993s - ('BlasOpt', 'SequentialGraphRewriter', 8) - 0.000s
         SequentialGraphRewriter      BlasOpt  time 0.049s for 81/80 nodes before/after rewriting
           0.000s for fgraph.validate()
           0.003s for callback
           0.035997s - ('gemm_optimizer', 'GemmOptimizer', 1) - 0.000s
             GemmOptimizer
              nb_iter 2
              nb_replacement 2
              nb_replacement_didn_t_remove 0
              nb_inconsistency_make 0
              nb_inconsistency_replace 0
              time_canonicalize 0.00720071792603
              time_factor_can 9.05990600586e-06
              time_factor_list 0.00128507614136
              time_toposort 0.00311398506165
              validate_time 4.60147857666e-05
              callback_time 0.00174236297607
           0.004569s - ('local_dot_to_dot22', 'WalkingGraphRewriter', 0) - 0.000s
             WalkingGraphRewriter
               nb_node (start, end, changed) (81, 81, 5)
               init io_toposort 0.00139284133911
               loop time 0.00312399864197
               callback_time 0.00172805786133
           0.002283s - ('local_dot22_to_dot22scalar', 'WalkingGraphRewriter', 2) - 0.000s
             WalkingGraphRewriter
               nb_node (start, end, changed) (80, 80, 0)
               init io_toposort 0.00171804428101
               loop time 0.000502109527588
               callback_time 0.0
           0.002257s - ('local_gemm_to_gemv', 'EquilibriumGraphRewriter', 3) - 0.000s
             EquilibriumGraphRewriter          local_gemm_to_gemv
               time 0.002s for 1 passes
               nb nodes (start, end,  max) 80 80 80
               time io_toposort 0.001s
               time in node rewriters 0.000s
               time in graph rewriters 0.000s
                0 - 0.002s 0 (0.000s in global rewrites, 0.001s io_toposort) - 80 nodes -
           0.002227s - ('use_c_blas', 'WalkingGraphRewriter', 4) - 0.000s
             WalkingGraphRewriter
               nb_node (start, end, changed) (80, 80, 0)
               init io_toposort 0.0014750957489
               loop time 0.00068998336792
               callback_time 0.0
           0.001632s - ('use_scipy_ger', 'WalkingGraphRewriter', 5) - 0.000s
             WalkingGraphRewriter
               nb_node (start, end, changed) (80, 80, 0)
               init io_toposort 0.00138401985168
               loop time 0.000202178955078
               callback_time 0.0
       0.031740s - ('specialize', 'EquilibriumGraphRewriter', 9) - 0.000s
         EquilibriumGraphRewriter      specialize
           time 0.031s for 2 passes
           nb nodes (start, end,  max) 80 78 80
           time io_toposort 0.003s
           time in node rewriters 0.022s
           time in graph rewriters 0.004s
            0 - 0.017s 6 (0.002s in global rewrites, 0.001s io_toposort) - 80 nodes - ('constant_folding', 2) ('local_mul_to_sqr', 1) ('local_elemwise_alloc', 1) ('local_div_to_inv', 1) ('local_mul_specialize', 1)
            1 - 0.014s 0 (0.002s in global rewrites, 0.001s io_toposort) - 78 nodes -
           times - times applied - nb node created - name:
           0.003s - 1 - 1 - local_mul_specialize
           0.002s - 1 - 2 - local_elemwise_alloc
           0.002s - 2 - 0 - constant_folding
           0.001s - 1 - 1 - local_div_to_inv
           0.001s - 1 - 1 - local_mul_to_sqr
           0.016s - in 69 rewrite(s) that where not used (display only those with a runtime > 0)
             0.004s - crossentropy_to_crossentropy_with_softmax_with_bias
             0.002s - local_one_minus_erf
             0.002s - Elemwise{sub,no_inplace}(z, Elemwise{mul,no_inplace}(alpha subject to <function <lambda> at 0x7f475e4da050>, SparseDot(x, y))) -> Usmm{no_inplace}(Elemwise{neg,no_inplace}(alpha), x, y, z)
             0.002s - local_add_specialize
             0.001s - local_func_inv
             0.001s - local_useless_elemwise
             0.001s - local_abs_merge
             0.001s - local_track_shape_i
             0.000s - local_one_minus_erf2
             0.000s - local_sum_mul_by_scalar
             0.000s - local_elemwise_sub_zeros
             0.000s - local_cast_cast
             0.000s - local_alloc_unary
             0.000s - Elemwise{log,no_inplace}(Softmax(x)) -> <function make_out_pattern at 0x7f47619a8410>(x)
             0.000s - local_sum_div_dimshuffle
             0.000s - local_sum_alloc
             0.000s - local_dimshuffle_lift
             0.000s - local_reduce_broadcastable
             0.000s - local_grad_log_erfc_neg
             0.000s - local_advanced_indexing_crossentropy_onehot
             0.000s - local_log_erfc
             0.000s - local_log1p
             0.000s - local_log_add
             0.000s - local_useless_alloc
             0.000s - local_neg_neg
             0.000s - local_neg_div_neg
    ...


To understand this profile here is some explanation of how rewrites work:

* Rewrites are organized in a hierarchy. At the top level, there
  is a :class:`SequentialGraphRewriter`. It contains other rewriters,
  and applies them in the order they were specified. Those sub-rewriters can be
  of other types, but are all **graph** rewriters.

* Each :class:`Rewriter` in the hierarchy will print some stats about
  itself. The information that it prints depends of the type of the
  rewriter.

* The :class:`SequentialGraphRewriter` will print some stats at the start:

    .. code-block:: none

        Rewriter Profile
        ----------------
         SequentialGraphRewriter  OPT_FAST_RUN  time 1.152s for 123/50 nodes before/after rewriting
           0.028s for fgraph.validate()
           0.131s for callback
           time      - (name, class, index) - validate time

  Then it will print, with some additional indentation, each sub-rewriter's profile
  information. These sub-profiles are ordered by the time they took to execute,
  not by their execution order.

  * ``OPT_FAST_RUN`` is the name of the rewriter
  * 1.152s is the total time spent in that rewriter
  * 123/50 means that before this rewriter, there were 123 apply node in the function graph, and after only 50.
  * 0.028s means it spent that time calls to ``fgraph.validate()``
  * 0.131s means it spent that time for callbacks. This is a mechanism that can trigger other execution when there is a change to the FunctionGraph.
  * ``time      - (name, class, index) - validate time`` tells how the information for each sub-rewriter get printed.
  * All other instances of :class:`SequentialGraphRewriter` are described like this. In
    particular, some sub-rewriter from ``OPT_FAST_RUN`` that are also
    :class:`SequentialGraphRewriter`.


* The :class:`SequentialGraphRewriter` will print some stats at the start:

    .. code-block:: none

       0.751816s - ('canonicalize', 'EquilibriumGraphRewriter', 4) - 0.004s
         EquilibriumGraphRewriter      canonicalize
           time 0.751s for 14 passes
           nb nodes (start, end,  max) 108 81 117
           time io_toposort 0.029s
           time in node rewriters 0.687s
           time in graph rewriters 0.010s
            0 - 0.050s 27 (0.000s in global rewrites, 0.002s io_toposort) - 108 nodes - ('local_dimshuffle_lift', 9) ('local_upcast_elemwise_constant_inputs', 5) ('local_shape_to_shape_i', 3) ('local_fill_sink', 3) ('local_fill_to_alloc', 2) ...
            1 - 0.288s 26 (0.002s in global rewrites, 0.002s io_toposort) - 117 nodes - ('local_dimshuffle_lift', 8) ('local_fill_sink', 4) ('constant_folding', 4) ('local_useless_elemwise', 3) ('local_subtensor_make_vector', 3) ...
            2 - 0.044s 13 (0.002s in global rewrites, 0.003s io_toposort) - 96 nodes - ('constant_folding', 4) ('local_dimshuffle_lift', 3) ('local_fill_sink', 3) ('local_useless_elemwise', 1) ('local_fill_to_alloc', 1) ...
            3 - 0.045s 11 (0.000s in global rewrites, 0.002s io_toposort) - 91 nodes - ('constant_folding', 3) ('local_fill_to_alloc', 2) ('local_dimshuffle_lift', 2) ('local_mul_canonizer', 2) ('MergeOptimizer', 1) ...
            4 - 0.035s 8 (0.002s in global rewrites, 0.002s io_toposort) - 93 nodes - ('local_fill_sink', 3) ('local_dimshuffle_lift', 2) ('local_fill_to_alloc', 1) ('MergeOptimizer', 1) ('constant_folding', 1)
            5 - 0.035s 6 (0.000s in global rewrites, 0.002s io_toposort) - 88 nodes - ('local_fill_sink', 2) ('local_dimshuffle_lift', 2) ('local_fill_to_alloc', 1) ('local_mul_canonizer', 1)
            6 - 0.038s 10 (0.001s in global rewrites, 0.002s io_toposort) - 95 nodes - ('local_fill_sink', 3) ('local_dimshuffle_lift', 3) ('constant_folding', 2) ('local_fill_to_alloc', 1) ('MergeOptimizer', 1)
            7 - 0.032s 5 (0.001s in global rewrites, 0.002s io_toposort) - 91 nodes - ('local_fill_sink', 3) ('MergeOptimizer', 1) ('local_dimshuffle_lift', 1)
            8 - 0.034s 5 (0.000s in global rewrites, 0.002s io_toposort) - 92 nodes - ('local_fill_sink', 3) ('MergeOptimizer', 1) ('local_greedy_distributor', 1)
            9 - 0.031s 6 (0.001s in global rewrites, 0.002s io_toposort) - 90 nodes - ('local_fill_sink', 2) ('local_fill_to_alloc', 1) ('MergeOptimizer', 1) ('local_dimshuffle_lift', 1) ('local_greedy_distributor', 1)
           10 - 0.032s 5 (0.000s in global rewrites, 0.002s io_toposort) - 89 nodes - ('local_dimshuffle_lift', 2) ('local_fill_to_alloc', 1) ('MergeOptimizer', 1) ('local_fill_sink', 1)
           11 - 0.030s 5 (0.000s in global rewrites, 0.002s io_toposort) - 88 nodes - ('local_dimshuffle_lift', 2) ('local_fill_to_alloc', 1) ('MergeOptimizer', 1) ('constant_folding', 1)
           12 - 0.026s 1 (0.000s in global rewrites, 0.003s io_toposort) - 81 nodes - ('MergeOptimizer', 1)
           13 - 0.031s 0 (0.000s in global rewrites, 0.003s io_toposort) - 81 nodes -
           times - times applied - nb node created - name:
           0.263s - 15 - 0 - constant_folding
           0.096s - 2 - 14 - local_greedy_distributor
           0.066s - 4 - 19 - local_mul_canonizer
           0.046s - 28 - 57 - local_fill_sink
           0.042s - 35 - 78 - local_dimshuffle_lift
           0.018s - 5 - 15 - local_upcast_elemwise_constant_inputs
           0.010s - 11 - 4 - MergeOptimizer
           0.009s - 4 - 0 - local_useless_elemwise
           0.005s - 11 - 2 - local_fill_to_alloc
           0.004s - 3 - 6 - local_neg_to_mul
           0.002s - 1 - 3 - local_lift_transpose_through_dot
           0.002s - 3 - 4 - local_shape_to_shape_i
           0.002s - 2 - 4 - local_subtensor_lift
           0.001s - 3 - 0 - local_subtensor_make_vector
           0.001s - 1 - 1 - local_sum_all_to_none
           0.131s - in 62 rewrite(s) that where not used (display only those with a runtime > 0)
             0.050s - local_add_canonizer
             0.018s - local_mul_zero
             0.016s - local_one_minus_erf
             0.010s - local_func_inv
             0.006s - local_0_dot_x
             0.005s - local_track_shape_i
             0.004s - local_mul_switch_sink
             0.004s - local_fill_cut
             0.004s - local_one_minus_erf2
             0.003s - local_remove_switch_const_cond
             0.003s - local_cast_cast
             0.002s - local_IncSubtensor_serialize
             0.001s - local_sum_div_dimshuffle
             0.001s - local_div_switch_sink
             0.001s - local_dimshuffle_no_inplace_at_canonicalize
             0.001s - local_cut_useless_reduce
             0.001s - local_reduce_join
             0.000s - local_sum_sum
             0.000s - local_useless_alloc
             0.000s - local_reshape_chain
             0.000s - local_useless_subtensor
             0.000s - local_reshape_lift
             0.000s - local_flatten_lift
             0.000s - local_useless_slice
             0.000s - local_subtensor_of_alloc
             0.000s - local_subtensor_of_dot
             0.000s - local_subtensor_merge

  * ``0.751816s - ('canonicalize', 'EquilibriumGraphRewriter', 4) - 0.004s``
    This line is from :class:`SequentialGraphRewriter`, and indicates information related
    to a sub-rewriter. It means that this sub-rewriter took
    a total of .7s. Its name is ``'canonicalize'``. It is an
    :class:`EquilibriumGraphRewriter`. It was executed at index 4 by the
    :class:`SequentialGraphRewriter`. It spent 0.004s in the *validate* phase.
  * All other lines are from the profiler of the :class:`EquilibriumGraphRewriter`.

  * An :class:`EquilibriumGraphRewriter` does multiple passes on the Apply nodes from
    the graph, trying to apply local and graph rewriters.
    Conceptually, it tries to execute all graph rewriters,
    and to apply all node rewriters on all
    nodes in the graph. If no rewrites got applied during a pass, it
    stops. So it tries to find an equilibrium state where no further rewrites
    can be applied. This is useful when we do not know a fixed order for the
    execution of rewrites.
  * ``time 0.751s for 14 passes`` means that it took .7s and did 14 passes over the graph.

  * ``nb nodes (start, end, max) 108 81 117`` means that at the start,
    the graph had 108 node, at the end, it had 81 and the maximum size
    was 117.

  * Then it prints some global timing information: it spent 0.029s in
    :func:`io_toposort`, all node rewriters took 0.687s together for all
    passes, and graph rewriters took a total of 0.010s.

  * Then we print the timing for each pass, the rewrite that
    got applied, and the number of time they got applied. For example,
    in pass zero, the :func:`local_dimshuffle_lift` rewrite changed the graph
    nine time.

  * Then we print the time spent in each rewriter, the number of times
    they changed the graph and the number of nodes they introduced in
    the graph.

  * Rewrites with that pattern :func:`local_op_lift` indicate that a node
    with that `Op` will be replaced by another node with the same `Op`,
    but will do computation closer to the inputs of the graph: i.e. a "lift" of
    the `Op`.
    For instance, in ``local_op(f(x))``, ``local_op`` is lifted through ``f`` to
    produce ``f(local_op(x))``.

  * Rewrites with that pattern :func:`local_op_sink` is the opposite of
    lifting. For instance, in ``f(local_op(x))``, ``local_op`` is sunk through
    ``f`` to produce ``local_op(f(x))``.

  * Local rewriters can replace any arbitrary node in the graph, not
    only the nodes they receive as input. In this case, the local rewrite returns a
    ``dict``, where the keys are `Variable`\s to be replaced and the
    values are the corresponding replacements.
