.. _lib_looping:

=========================================
:mod:`looping` -- Structured symbolic loops
=========================================

:func:`pytensor.looping.loop` describes a recurrence with a step function called
as ``f(carry, x)``. Return ``(next_carry, y)`` to update the state and select values
to collect. ``loop`` returns the final carry and the collected values stacked
along a new leading time axis. Return ``None`` for values you do not need.

The step is called once to build a symbolic graph. The compiled function executes
that graph for each iteration. Use symbolic expressions for decisions that depend
on input values.

Carry and gradients
===================

This loop doubles a scalar three times and collects every computed value:

>>> import numpy as np
>>> import pytensor
>>> import pytensor.tensor as pt
>>> from pytensor.looping import loop, shift
>>> initial = pt.scalar("initial")
>>> def double(carry, _):
...     next_carry = carry * 2
...     return next_carry, next_carry
>>> final, values = loop(double, init=initial, length=3)
>>> evaluate = pytensor.function([initial], [final, values, pt.grad(final, initial)])
>>> last, trace, gradient = evaluate(1.0)
>>> assert last == 8 and gradient == 8
>>> np.testing.assert_allclose(trace, [2, 4, 8])

The initial carry is excluded from ``values``. Full gradients are enabled by
default; set ``truncate_gradient`` to the number of steps to backpropagate through
when using truncated backpropagation.

Sequences and captured variables
================================

Provide ``xs`` to iterate over tensors along their leading axes. The shortest
sequence determines the iteration count unless ``length`` is provided. Symbolic
variables referenced by the step are captured automatically:

>>> xs = pt.vector("xs")
>>> scale = pt.scalar("scale")
>>> def accumulate(carry, x):
...     next_carry = carry + x * scale
...     return next_carry, next_carry
>>> final, totals = loop(accumulate, init=initial, xs=xs)
>>> evaluate = pytensor.function([initial, xs, scale], [final, totals])
>>> last, trace = evaluate(10.0, [1.0, 2.0, 3.0], 2.0)
>>> assert last == 22
>>> np.testing.assert_allclose(trace, [12, 16, 22])

An explicit ``length`` may shorten the loop and must not exceed any available
sequence length. Without sequences, ``length`` is required. With zero iterations,
the final carry equals its initial value.

Structured arguments
====================

``init``, ``xs`` and the collected outputs may be nested lists, tuples, namedtuples
and dictionaries with sortable keys. ``None`` and empty containers are preserved.
The next carry must match the structure of ``init``, including namedtuple classes.
Dictionary keys are sorted internally, so insertion order may differ between the
initial carry and the step result. Carry tensors must keep their shape and dtype.

>>> from collections import namedtuple
>>> State = namedtuple("State", "total count")
>>> def count_and_sum(carry, x):
...     state = carry["state"]
...     state = State(state.total + x, state.count + 1)
...     return {"state": state}, [state.total]
>>> final, collected = loop(
...     count_and_sum,
...     init={"state": State(pt.zeros(()), pt.as_tensor(0))},
...     xs=xs,
... )
>>> evaluate = pytensor.function([xs], [final["state"].total, final["state"].count, collected[0]])
>>> last, count, trace = evaluate([1.0, 2.0, 3.0])
>>> assert last == 6 and count == 3
>>> np.testing.assert_allclose(trace, [1, 3, 6])

Time offsets
============

Use :func:`~pytensor.looping.shift` to expose several past carry values. Offsets
must be strictly increasing and negative. The initial tensor's leading axis
supplies the history, from oldest to newest. Unpack slices in offset order and
return ``carry.push(next_value)`` to advance the history:

>>> def fibonacci(carry, _):
...     older, newer = carry
...     next_value = older + newer
...     return carry.push(next_value), next_value
>>> final, values = loop(fibonacci, init=shift([1.0, 1.0], by=(-2, -1)), length=5)
>>> evaluate = pytensor.function([], [final, values])
>>> last, trace = evaluate()
>>> assert last == 13
>>> np.testing.assert_allclose(trace, [2, 3, 5, 8, 13])

An oldest offset of ``-k`` requires at least ``k`` initial entries; extra initial
entries are ignored. The final carry contains the last computed value.

Sequence offsets must be nonnegative. For example, ``shift(xs, by=(0, 1))``
exposes adjacent entries and provides one fewer iteration than ``xs``. Shifted
sequence slices are read-only:

>>> def adjacent_product(_, slices):
...     left, right = slices
...     return None, left * right
>>> _, products = loop(adjacent_product, init=None, xs=shift(xs, by=(0, 1)))
>>> evaluate = pytensor.function([xs], products)
>>> np.testing.assert_allclose(evaluate([1.0, 2.0, 3.0, 4.0]), [2, 6, 12])

Early stopping
==============

Return a third result containing a scalar boolean tensor to stop after the current
iteration. ``length`` or the sequence lengths still bound the maximum iterations.
The stopping iteration is included in the collected outputs:

>>> def until_three(carry, _):
...     next_value = carry + 1
...     return next_value, next_value, next_value >= 3
>>> final, values = loop(until_three, init=initial, length=10)
>>> evaluate = pytensor.function([initial], [final, values])
>>> last, trace = evaluate(0.0)
>>> assert last == 3
>>> np.testing.assert_allclose(trace, [1, 2, 3])

Random states and shared updates
================================

Carry RNG states explicitly to return a reusable final state. RNG states cannot
be collected as stacked outputs; collect tensor samples instead:

>>> from pytensor.tensor.random.type import RandomGeneratorType
>>> rng = RandomGeneratorType()("rng")
>>> def draw(carry, _):
...     next_rng, sample = pt.random.normal(rng=carry, return_next_rng=True)
...     return next_rng, sample
>>> final_rng, samples = loop(draw, init=rng, length=3)
>>> evaluate = pytensor.function([rng], [final_rng, samples])
>>> next_rng, first = evaluate(np.random.default_rng(42))
>>> _, second = evaluate(next_rng)
>>> assert first.shape == second.shape == (3,)
>>> assert not np.array_equal(first, second)

Shared variable updates also belong in the carry. Use the shared variable as
``init`` and compile with ``updates={shared_state: final}``. The step does not
produce an implicit updates dictionary.

Migrating from scan
===================

The user-facing :func:`pytensor.scan` function is deprecated. ``loop`` uses the
same underlying :class:`pytensor.scan.op.Scan` operation and compiler rewrites.
The main API changes are:

* The step always receives ``(carry, x)`` rather than positional sequences,
  recurrent outputs and non-sequences.
* ``init`` supplies carried state, ``xs`` supplies sequences, and ``length``
  replaces ``n_steps``. Use ``xs[::-1]`` to iterate in reverse.
* Non-sequences are captured from the enclosing scope.
* The first step result updates the carry; the second selects what to collect.
  To reproduce a scan output trace, return the updated value in both positions.
* Past and future slices use ``shift``. A scalar boolean third result replaces
  ``until``.
* Shared variables and RNG states are carried explicitly; updates are applied
  when compiling with :func:`pytensor.function`.

API reference
=============

.. automodule:: pytensor.looping
   :members: loop, shift, ShiftedArg, InnerShiftedArg
