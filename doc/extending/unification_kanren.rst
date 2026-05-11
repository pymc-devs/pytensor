.. _unification_kanren:

Unification, reification and miniKanren (optional)
==================================================

.. note::

   The :mod:`logical-unification`, :mod:`kanren`, :mod:`etuples` and :mod:`cons`
   packages are **optional** dependencies. PyTensor's built-in
   :class:`PatternNodeRewriter` ships its own specialized matcher and does not
   require any of them. Install the extra explicitly to use the tools described
   on this page::

       pip install pytensor[kanren]

   or::

       pip install logical-unification kanren etuples cons

   Importing :mod:`pytensor.graph.rewriting.kanren` registers the dispatchers
   that let :func:`unification.unify` / :func:`unification.reify` and miniKanren
   relations walk PyTensor :class:`Apply` nodes, :class:`Op`\s and
   :class:`Type`\s. It also registers :class:`PatternVar` with the
   :class:`unification.Var` ABC, so ``isinstance(x, unification.Var)`` and
   :func:`isvar(x)` keep returning ``True`` for PyTensor pattern variables.
   Make sure that import happens once before calling :func:`unify`,
   :func:`reify` or :func:`kanren.run` on PyTensor graphs::

       import pytensor.graph.rewriting.kanren  # noqa: F401  -- registers dispatchers

.. _unification:

Unification and reification
---------------------------

`Unification and reification
<https://en.wikipedia.org/wiki/Unification_(computer_science)>`_ implement a
more succinct and reusable form of "pattern matching and replacement".
*Use of the unification and reification tools is preferable when
a rewrite's matching and replacement are non-trivial*, so we will briefly explain
them in the following.

PyTensor's unification and reification tools are provided by the
`logical-unification <https://github.com/pythological/unification>`_ package.
The basic tools are :func:`unify`, :func:`reify`, and :class:`var`.  The class :class:`var`
construct *logic variables*, which represent the elements to be unified/matched, :func:`unify`
performs the "matching", and :func:`reify` performs the "replacements".

See :mod:`unification`'s documentation for an introduction to unification and reification.

In order to use :func:`unify` and :func:`reify` with PyTensor graphs, we need an intermediate
structure that will allow us to represent PyTensor graphs that contain :class:`var`\s, because
PyTensor :class:`Op`\s and :class:`Apply` nodes will not accept these foreign objects as inputs.
The `etuples <https://github.com/pythological/etuples>`_ library provides the
:class:`ExpressionTuple` (tuple-like, with caching for evaluation) that fills this role.

Here is an illustration of all the above components used together:

>>> import pytensor.graph.rewriting.kanren  # noqa: F401  -- registers dispatchers
>>> from unification import unify, reify, var
>>> from etuples import etuple
>>> y_lv = var()  # Create a logic variable
>>> y_lv
~_1
>>> s = unify(add(x, y), etuple(add, x, y_lv))
>>> s
{~_1: y}

In this example, :func:`unify` matched the PyTensor graph in the first argument with the "pattern"
given by the :func:`etuple` in the second.  The result is a ``dict`` mapping logic variables to
the objects to which they were successfully unified.  When a :func:`unify` doesn't succeed, it will
return ``False``.

:func:`reify` uses ``dict``\s like the kind produced by :func:`unify` to replace
logic variables within structures:

>>> res = reify(etuple(add, y_lv, y_lv), s)
>>> res
e(<pytensor.scalar.basic.Add at 0x7f54dfa5a350>, y, y)

Since :class:`ExpressionTuple`\s can be evaluated, we can produce a complete PyTensor graph from these
results as follows:

>>> res.evaled_obj
add.0
>>> pytensor.dprint(res.evaled_obj)
add [id A] ''
 |y [id B]
 |y [id B]


Because :class:`ExpressionTuple`\s effectively model `S-expressions
<https://en.wikipedia.org/wiki/S-expression>`_, they can be used with the `cons
<https://github.com/pythological/python-cons>`_ package to unify and reify
graphs structurally.

Let's say we want to match graphs that use the :class:`add`\ :class:`Op` but could have a
varying number of arguments:

>>> from cons import cons
>>> op_lv = var()
>>> args_lv = var()
>>> s = unify(cons(op_lv, args_lv), add(x, y))
>>> s
{~_2: <pytensor.scalar.basic.Add at 0x7f54dfa5a350>, ~_3: e(x, y)}
>>> s = unify(cons(op_lv, args_lv), add(x, y, z))
>>> s
{~_2: <pytensor.scalar.basic.Add at 0x7f54dfa5a350>, ~_3: e(x, y, z)}

From here, we can check ``s[op_lv] == add`` to confirm that we have the correct :class:`Op` and
proceed with our rewrite.

>>> res = reify(cons(mul, args_lv), s)
>>> res
e(<pytensor.scalar.basic.Mul at 0x7f54dfa5ae10>, x, y, z)
>>> pytensor.dprint(res.evaled_obj)
mul [id A] ''
 |x [id B]
 |y [id C]
 |z [id D]


.. _miniKanren_rewrites:

miniKanren
----------

Given that unification and reification are fully implemented for PyTensor objects via the :mod:`unification` package,
the `kanren <https://github.com/pythological/kanren>`_ package can be used with PyTensor graphs, as well.
:mod:`kanren` implements the `miniKanren <http://minikanren.org/>`_ domain-specific language for relational programming.

Refer to the links above for a proper introduction to miniKanren, but suffice it to say that
miniKanren orchestrates the unification and reification operations described above, and
it does so in the context of relational operators (e.g. equations like :math:`x + x = 2 x`).
This means that a relation that--say--represents :math:`x + x = 2 x` can be
utilized in both directions.

Currently, the node rewriter :class:`KanrenRelationSub` provides a means of
turning :mod:`kanren` relations into :class:`NodeRewriter`\s; however,
:mod:`kanren` can always be used directly from within a custom :class:`Rewriter`, so
:class:`KanrenRelationSub` is not necessary.

The following is an example that distributes dot products across additions.

.. code::

    import pytensor
    import pytensor.tensor as pt
    from pytensor.graph.rewriting.kanren import KanrenRelationSub
    from pytensor.graph.rewriting.basic import EquilibriumGraphRewriter
    from pytensor.graph.rewriting.utils import rewrite_graph
    from pytensor.tensor.math import _dot
    from etuples import etuple
    from kanren import conso, eq, fact, heado, tailo
    from kanren.assoccomm import assoc_flatten, associative
    from kanren.core import lall
    from kanren.graph import mapo
    from unification import vars as lvars


    # Make the graph pretty printing results a little more readable
    pytensor.pprint.assign(
        _dot, pytensor.printing.OperatorPrinter("@", -1, "left")
    )

    # Tell `kanren` that `add` is associative
    fact(associative, pt.add)


    def dot_distributeo(in_lv, out_lv):
        """A `kanren` goal constructor relation for the relation ``A.dot(a + b ...) == A.dot(a) + A.dot(b) ...``."""
        A_lv, add_term_lv, add_cdr_lv, dot_cdr_lv, add_flat_lv = lvars(5)

        return lall(
            # Make sure the input is a `_dot`
            eq(in_lv, etuple(_dot, A_lv, add_term_lv)),
            # Make sure the term being `_dot`ed is an `add`
            heado(pt.add, add_term_lv),
            # Flatten the associative pairings of `add` operations
            assoc_flatten(add_term_lv, add_flat_lv),
            # Get the flattened `add` arguments
            tailo(add_cdr_lv, add_flat_lv),
            # Add all the `_dot`ed arguments and set the output
            conso(pt.add, dot_cdr_lv, out_lv),
            # Apply the `_dot` to all the flattened `add` arguments
            mapo(lambda x, y: conso(_dot, etuple(A_lv, x), y), add_cdr_lv, dot_cdr_lv),
        )


    dot_distribute_rewrite = EquilibriumGraphRewriter([KanrenRelationSub(dot_distributeo)], max_use_ratio=10)


Below, we apply `dot_distribute_rewrite` to a few example graphs.  First we create simple test graph:

>>> x_at = pt.vector("x")
>>> y_at = pt.vector("y")
>>> A_at = pt.matrix("A")
>>> test_at = A_pt.dot(x_at + y_at)
>>> print(pytensor.pprint(test_at))
(A @ (x + y))

Next we apply the rewrite to the graph:

>>> res = rewrite_graph(test_at, include=[], custom_rewrite=dot_distribute_rewrite, clone=False)
>>> print(pytensor.pprint(res))
((A @ x) + (A @ y))

We see that the dot product has been distributed, as desired.  Now, let's try a
few more test cases:

>>> z_at = pt.vector("z")
>>> w_at = pt.vector("w")
>>> test_at = A_pt.dot((x_at + y_at) + (z_at + w_at))
>>> print(pytensor.pprint(test_at))
(A @ ((x + y) + (z + w)))
>>> res = rewrite_graph(test_at, include=[], custom_rewrite=dot_distribute_rewrite, clone=False)
>>> print(pytensor.pprint(res))
(((A @ x) + (A @ y)) + ((A @ z) + (A @ w)))

>>> B_at = pt.matrix("B")
>>> w_at = pt.vector("w")
>>> test_at = A_pt.dot(x_at + (y_at + B_pt.dot(z_at + w_at)))
>>> print(pytensor.pprint(test_at))
(A @ (x + (y + ((B @ z) + (B @ w)))))
>>> res = rewrite_graph(test_at, include=[], custom_rewrite=dot_distribute_rewrite, clone=False)
>>> print(pytensor.pprint(res))
((A @ x) + ((A @ y) + ((A @ (B @ z)) + (A @ (B @ w)))))


This example demonstrates how non-trivial matching and replacement logic can
be neatly expressed in miniKanren's DSL, but it doesn't quite demonstrate miniKanren's
relational properties.

To do that, we will create another :class:`Rewriter` that simply reverses the arguments
to the relation :func:`dot_distributeo` and apply it to the distributed result in ``res``:

>>> dot_gather_rewrite = EquilibriumGraphRewriter([KanrenRelationSub(lambda x, y: dot_distributeo(y, x))], max_use_ratio=10)
>>> rev_res = rewrite_graph(res, include=[], custom_rewrite=dot_gather_rewrite, clone=False)
>>> print(pytensor.pprint(rev_res))
(A @ (x + (y + (B @ (z + w)))))

As we can see, the :mod:`kanren` relation works both ways, just like the underlying
mathematical relation does.

miniKanren relations can be used to explore rewrites of graphs in sophisticated
ways.  It also provides a framework that more directly maps to the mathematical
identities that drive graph rewrites.  For some simple examples of relational graph rewriting
in :mod:`kanren` see `here <https://github.com/pythological/kanren/blob/master/doc/graphs.md>`_.  For a
high-level overview of miniKanren's use as a tool for symbolic computation see
`"miniKanren as a Tool for Symbolic Computation in Python" <https://arxiv.org/abs/2005.11644>`_.
