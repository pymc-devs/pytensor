.. _prng:

===========================================
Pseudo random number generation in PyTensor
===========================================

PyTensor has native support for `pseudo random number generation (PRNG) <https://en.wikipedia.org/wiki/Pseudorandom_number_generator>`_.

This document describes the details of how PRNGs are implemented in PyTensor, via the RandomVariable Operator.
For a more applied example see :ref:`using_random_numbers`

We also discuss how initial seeding and seeding updates are implemented.

We will use PRNG and RNG interchangeably, keeping in mind we are always talking about PRNGs.

The basics
==========

NumPy
-----

To start off, let's recall how PRNGs works in NumPy

>>> import numpy as np
>>> rng = np.random.default_rng(seed=123)
>>> print(rng.uniform(size=2), rng.uniform(size=2))
[0.68235186 0.05382102] [0.22035987 0.18437181]

In the first line np.random.default_rng(seed) creates a random Generator.

>>> rng # doctest: +ELLIPSIS
Generator(PCG64) at 0x...

Every NumPy Generator holds a BitGenerator, which is able to generate high-quality sequences of pseudo random bits.
NumPy generators' methods convert these sequences of bits into sequences of numbers that follow a specific statistical distribution.
For more details, you can read `NumPy random sampling documentation <https://numpy.org/doc/stable/reference/random>`_.

>>> rng.bit_generator # doctest: +ELLIPSIS
<numpy.random._pcg64.PCG64 ... at 0x...>

>>> rng.bit_generator.state # doctest: +NORMALIZE_WHITESPACE
{'bit_generator': 'PCG64',
 'state': {'state': 143289216567205249174526524509312027761,
  'inc': 17686443629577124697969402389330893883},
 'has_uint32': 0,
 'uinteger': 0}

When we call rng.uniform(size=2), the Generator class requested a new array of pseudo random bits (state) from the BitGenerator,
and used a deterministic mapping function to convert those into a float64 numbers.

It did this twice, because we requested two draws via the size argument.
In the long-run this deterministic mapping function should produce draws that are statistically indistinguishable from a true uniform distribution.

For illustration we implement a very bad mapping function from a bit generator to uniform draws.

.. code:: python

    def bad_uniform_rng(rng, size):
        bit_generator = rng.bit_generator

        uniform_draws = np.empty(size)
        for i in range(size):
            bit_generator.advance(1)
            state = rng.bit_generator.state["state"]["state"]
            last_3_digits = state % 1_000
            uniform_draws[i] = (last_3_digits + 1) / 1_000
        return uniform_draws

    bad_uniform_rng(rng, size=5)
    # array([0.033, 0.972, 0.459, 0.71 , 0.765])

SciPy
-----

SciPy wraps these NumPy routines in a slightly different API.

>>> import scipy.stats as st
>>> rng = np.random.default_rng(123)
>>> print(st.uniform.rvs(size=2, random_state=rng), st.uniform.rvs(size=2, random_state=rng))
[0.68235186 0.05382102] [0.22035987 0.18437181]


PyTensor
--------

PyTensor does not implement its own bit/generators methods.
Just like SciPy, it borrows NumPy routines directly.

The low-level API of PyTensor, using RandomVariables is similar to that of SciPy,
whereas the higher-level API using RandomGeneratorVariable is more like that of NumPy.

Unlike either NumPy or SciPy, PyTensor represents RNG state threading explicitly:
every random draw returns a ``(next_rng, draw)`` tuple,
where ``next_rng`` is the updated generator state that should be used for subsequent draws.

Let's start with the low-level API.

>>> import pytensor
>>> import pytensor.tensor as pt

>>> rng = pt.random.rng("rng")
>>> next_rng, x = pt.random.uniform(rng=rng, size=2, return_next_rng=True)
>>> f = pytensor.function([rng], x)

.. note::

    Due to ongoing API migration we have to pass `return_next_rng=True` to get back the next rng
    This will be the default in a future release and the argument will be removed

We created a function that takes a NumPy RandomGenerator and returns two uniform draws. Let's evaluate it

>>> rng_val = np.random.default_rng(123)
>>> print(f(rng_val), f(rng_val))
[0.68235186 0.05382102] [0.68235186 0.05382102]

The first numbers were exactly the same as the NumPy and SciPy calls, because we are using the very same routines.

Perhaps surprisingly, we got the same results when we called the function the second time!
This is because PyTensor functions do not hold an internal state and do not modify inputs inplace unless requested to.

We made sure that the rng_val was not modified when calling our PyTensor function, by copying it before using it.
This may feel inefficient (and it is), but PyTensor is built on a pure functional approach, which is not allowed to have side-effects by default.

We will later see how we can get around this issue by making the inputs mutable or using shared variables with explicit update rules.

Before that, let's convince ourselves we can actually get different draws, when we modify the bit generator of our input RNG.

>>> _ = rng_val.bit_generator.advance(1)
>>> print(f(rng_val), f(rng_val))
[0.05382102 0.22035987] [0.05382102 0.22035987]

>>> _ = rng_val.bit_generator.advance(1)
>>> print(f(rng_val), f(rng_val))
[0.22035987 0.18437181] [0.22035987 0.18437181]

Updating the bit generator manually is not a good practice.
For starters, it may be unclear how much we have to advance it!

In this case we had to advance it twice to get two completely new draws, because the inner function uses two states.
But other distributions could need more states for a single draw, or they could be clever and reuse the same state for multiple draws.

That is why the RNG variable methods always return a ``(next_rng, draw)`` tuple.
The ``next_rng`` contains the bit generator that was already modified when taking draws, and can be safely used again.

We can compile a function that returns the next_rng explicitly, so that we can use it as the input of the function in subsequent calls.

>>> f = pytensor.function([rng], [next_rng, x])

>>> rng_val = np.random.default_rng(123)
>>> next_rng_val, x_val = f(rng_val)
>>> print(x_val)
[0.68235186 0.05382102]

>>> next_rng_val, x_val = f(next_rng_val)
>>> print(x_val)
[0.22035987 0.18437181]

>>> next_rng_val, x_val = f(next_rng_val)
>>> print(x_val)
[0.1759059  0.81209451]


Shared variables
================

At this point we can make use of PyTensor shared variables.
Shared variables are global variables that don't need (and can't) be passed as explicit inputs to the functions where they are used.

The ``pt.random.shared_rng()`` helper creates a shared RNG variable:

>>> rng = pt.random.shared_rng(np.random.default_rng(123))
>>> next_rng, x = pt.random.uniform(rng=rng, return_next_rng=True)
>>>
>>> f = pytensor.function([], [next_rng, x])
>>>
>>> next_rng_val, x_val = f()
>>> print(x_val)
0.6823518632481435

We can update the value of shared variables across calls.

>>> rng.set_value(next_rng_val)
>>> next_rng_val, x_val = f()
>>> print(x_val)
0.053821018802222675

>>> rng.set_value(next_rng_val)
>>> next_rng_val, x_val = f()
>>> print(x_val)
0.22035987277261138

The real benefit of using shared variables is that we can automate this updating via the aptly named updates kwarg of PyTensor functions.

In this case it makes sense to simply replace the original value by the next_rng_val (there is not really any other operation we can do with PyTensor RNGs)

>>> rng = pt.random.shared_rng(np.random.default_rng(123))
>>> next_rng, x = pt.random.uniform(rng=rng, return_next_rng=True)
>>>
>>> f = pytensor.function([], x, updates={rng: next_rng})
>>>
>>> f(), f(), f()
(array(0.68235186), array(0.05382102), array(0.22035987))


Reseeding
---------

Shared RNG variables can be "reseeded" by setting them to a new RNG with the desired seed

>>> rng = pt.random.shared_rng(np.random.default_rng(123), borrow=True)
>>> next_rng, x = pt.random.normal(rng=rng, return_next_rng=True)
>>>
>>> f = pytensor.function([], x, updates={rng: next_rng})
>>>
>>> print(f(), f())
-0.9891213503478509 -0.3677866514678832
>>> rng.set_value(np.random.default_rng(123), borrow=True)
>>> print(f(), f())
-0.9891213503478509 -0.3677866514678832

.. note::
    You can pass borrow=True to `shared_rng` and `set_value`, to avoid a costly deepcopy of the outer numpy RandomGenerator.


High-level API
==============

PyTensor offers a high level API more similar to that of NumPy,
by allow creating random variables as a method of RandomGeneratorVariables

>>> rng = pt.random.shared_rng(np.random.default_rng(123), name="rng")
>>> next_rng, x = rng.normal()
>>> f = pytensor.function([], x, updates={rng: next_rng})
>>> assert f() != f()

These methods are also available for both shared and non-shared RandomGeneratorVariables.

PyTensor further helps with the creation of shared RNGs with the `seed argument`

>>> rng = pt.random.shared_rng(seed=123)
>>> next_rng, x = rng.normal()
>>> f = pytensor.function([], x, updates={rng: next_rng})
>>> x1 = f()
>>> rng.set_value(seed=123)
>>> x2 = f()
>>> assert x1 == x2


Inplace optimization
====================

As mentioned, RandomVariable Ops default to making a copy of the input RNG before using it, which can be quite slow.

We compile with the Python linker here so the dprint shows the plain RandomVariable form. Some backends specialize it (e.g. Numba wraps it in a ``RandomVariableWithCoreShape``, see the Numba section below).

>>> py_mode = pytensor.Mode(linker="py", optimizer="fast_run")
>>> rng = pt.random.shared_rng(np.random.default_rng(123), name="rng")
>>> next_rng, x = pt.random.uniform(rng=rng, return_next_rng=True)
>>> f = pytensor.function([], x, mode=py_mode)
>>> _ = pytensor.dprint(f, print_destroy_map=True)
uniform_rv{"(),()->()"}.1 [id A] 0
 ├─ rng [id B]
 ├─ NoneConst{None} [id C]
 ├─ 0.0 [id D]
 └─ 1.0 [id E]


>>> %timeit f()  # doctest: +SKIP
81.8 µs ± 15.4 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

>>> rng_np = np.random.default_rng(123)
>>> %timeit rng_np.uniform()  # doctest: +SKIP
2.15 µs ± 63.4 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

Like other PyTensor operators, RandomVariable's can be given permission to modify inputs inplace during their operation.

In this case, there is a `inplace` flag that when `true` tells the RandomVariable Op that it is safe to modify the RNG input inplace.
If the flag is set, the RNG will not be copied before taking random draws.

>>> x.owner.op.inplace
False

For illustration purposes, we will subclass the Uniform Op class and set inplace to True by default.

Users should never do this directly!

>>> class InplaceUniform(type(pt.random.uniform)):
...    inplace = True

>>> inplace_uniform = InplaceUniform()
>>>
>>> rng = pt.random.shared_rng(np.random.default_rng(123), name="rng")
>>> x = inplace_uniform(rng=rng)
>>> x.owner.op.inplace
True

>>> inplace_f = pytensor.function([], x, accept_inplace=True, mode=py_mode)
>>> _ = pytensor.dprint(inplace_f, print_destroy_map=True)
uniform_rv{"(),()->()"}.1 [id A] d={0: [0]} 0
 ├─ rng [id B]
 ├─ NoneConst{None} [id C]
 ├─ 0.0 [id D]
 └─ 1.0 [id E]

The destroy map annotation tells us that the first output of the x variable is allowed to modify the first input.

>>> %timeit inplace_f() # doctest: +SKIP
9.71 µs ± 2.06 µs per loop (mean ± std. dev. of 7 runs, 100000 loops each)

Performance is now much closer to calling NumPy directly, with a small overhead introduced by the PyTensor function.


The `random_make_inplace <https://github.com/pymc-devs/pytensor/blob/3fcf6369d013c597a9c964b2400a3c5e20aa8dce/pytensor/tensor/random/rewriting/basic.py#L42-L52>`_
rewrite automatically replaces RandomVariable Ops by their inplace counterparts, when such operation is deemed safe. This happens when:

#. An input RNG is flagged as `mutable`
#. A RandomVariable Op consumes an intermediate RNG created by another RandomVariable Op

The first case is true when a user uses the `mutable` `kwarg` directly.

>>> from pytensor.compile import In
>>> rng = pt.random.rng("rng")
>>> next_rng, x = rng.uniform()
>>> with pytensor.config.change_flags(optimizer_verbose=True): # doctest: +ELLIPSIS
...     inplace_f = pytensor.function([In(rng, mutable=True)], [x], mode=py_mode)
rewriting: rewrite random_make_inplace replaces ...
>>> _ = inplace_f.dprint(print_destroy_map=True)
uniform_rv{"(),()->()"}.1 [id A] d={0: [0]} 0
 ├─ rng [id B]
 ├─ NoneConst{None} [id C]
 ├─ 0.0 [id D]
 └─ 1.0 [id E]

Or, more commonly, when a shared RNG is used and an update expression is given.
In this case, a RandomVariable is allowed to modify the RNG because the shared variable holding it will be rewritten anyway.

>>> rng = pt.random.shared_rng(np.random.default_rng(123), name="rng")
>>> next_rng, x = rng.uniform()
>>>
>>> inplace_f = pytensor.function([], [x], updates={rng: next_rng}, mode=py_mode)
>>> _ = inplace_f.dprint(print_destroy_map=True)
uniform_rv{"(),()->()"}.1 [id A] d={0: [0]} 0
 ├─ rng [id B]
 ├─ NoneConst{None} [id C]
 ├─ 0.0 [id D]
 └─ 1.0 [id E]
uniform_rv{"(),()->()"}.0 [id A] d={0: [0]} 0
 └─ ···


.. _multiple_random_variables:

Multiple random variables
=========================

Like in NumPy, the suggested way to take multiple random draws in a single graph is to chain (reuse) the same RandomGenerator objects


.. code:: python

    rng = pt.random.shared_rng(seed=None, name="rng")
    next_rng, x = rng.normal()
    next_rng, y = next_rng.normal()


.. _rng_reuse_warning:

.. warning::

    One must be careful NOT to reuse the same RNG variable for multiple draws without threading the updated state. This would produce correlated (identical) draws, since both operations see the same input state.

PyTensor warns when it detects this pattern:

.. code:: python

    rng = pt.random.rng()
    _, x = rng.normal()
    _, y = rng.normal()  # WARNING: rng already used!



Unused RNG consumer optimization (``random_unsafe``)
-----------------------------------------------------

When some RandomVariables in a "chain" are not needed in the final compiled function,
PyTensor can skip unused draws by replacing use of the intermediate RNG output by the corresponding RNG input.
This is done by the ``sidestep_unused_rng_consumer`` rewrite.

Because this optimization alters the RNG state sequence, and therefore the value,
it is tagged as ``random_unsafe`` and can be excluded if exact reproducibility of the RNG stream is needed:

.. code:: python

    rng = pt.random.shared_rng(np.random.default_rng(123), name="rng")

    next_rng, x = rng.normal(name="x")
    final_rng, y = rng.normal(name="y")

    # g will skip the draws of x, meaning they won't match with those of f
    f = pytensor.function([], [x, y])
    g = pytensor.function([], [y])
    assert f() != g()

    # We can exclude random_unsafe rewrites to forbid this behavior
    mode = pytensor.get_default_mode().excluding("random_unsafe")
    h = pytensor.function([], [x], mode=mode)
    assert f() == h()


Multiple RNG variables
----------------------

An alternative approach is to define separate rng variables for each Random Variable in the graph

>>> ss = np.random.SeedSequence(123)
>>> rng_x_np, rng_y_np = [np.random.default_rng(s) for s in ss.spawn(2)]
>>> rng_x = pt.random.shared_rng(rng_x_np, name="rng_x")
>>> rng_y = pt.random.shared_rng(rng_y_np, name="rng_y")
>>>
>>> next_rng_x, x = rng_x.normal(loc=0, scale=10)
>>> next_rng_y, y = rng_y.normal(loc=x, scale=0.1)
>>>
>>> next_rng_x.name = "next_rng_x"
>>> next_rng_y.name = "next_rng_y"
>>>
>>> f = pytensor.function([], [x, y], updates={rng_x: next_rng_x, rng_y: next_rng_y})


.. warning::

    Care must be taken to create good quality independent Generators, following the NumPy best practices https://numpy.org/doc/stable/reference/random/parallel.html#parallel-random-number-generation.



Random variables in inner graphs
================================

Scan
----

Scan works very similar to a function (that is called repeatedly inside an outer scope).

This means that random variables will always return the same output unless the RNG state is threaded across iterations.
If we use an RNG as a non-sequence (i.e. the same value is passed to every iteration), every step sees the same state
and produces the same draw:

>>> rng = pt.random.shared_rng(seed=123, name="rng")

>>> def constant_step(rng):
...     _, x = rng.normal()
...     return x

>>> draws = pytensor.scan(
...     fn=constant_step,
...     outputs_info=[None],
...     non_sequences=[rng],
...     n_steps=5,
...     strict=True,
...     return_updates=False,
... )

>>> f = pytensor.function([], draws)
>>> f(), f()
(array([-0.98912135, -0.98912135, -0.98912135, -0.98912135, -0.98912135]), array([-0.98912135, -0.98912135, -0.98912135, -0.98912135, -0.98912135]))

To get different draws at each step, the RNG should be passed as a recurrent state via ``outputs_info``.
The step function receives the current RNG, uses it for draws, and returns the updated RNG alongside the output.
The final RNG state can then be used to build the update dictionary for the outer function.

>>> rng = pt.random.shared_rng(np.random.default_rng(123))

>>> def random_step(rng):
...     next_rng, x = rng.normal()
...     return x, next_rng

>>> draws, final_rng = pytensor.scan(
...     fn=random_step,
...     outputs_info=[None, rng],
...     n_steps=5,
...     return_updates=False,
... )

>>> f = pytensor.function([], draws, updates={rng: final_rng})
>>> f(), f()
(array([-0.98912135, -0.36778665,  1.28792526,  0.19397442,  0.9202309 ]), array([ 0.57710379, -0.63646365,  0.54195222, -0.31659545, -0.32238912]))

.. note::

    Due to a current limitation, only the last RNG state is returned by Scan. This is unlike TensorVariables in outputs_info for which all intermediate states are returned.


OpFromGraph
-----------

RNG variables can be used directly in OpFromGraph


>>> from pytensor.compile.builders import OpFromGraph

>>> def create_lognormal_ofg():
...     """Create an OpFromGraph that takes rng and returns a lognormal draw and next_rng."""
...     rng = pt.random.rng(name="rng")
...     next_rng, x = rng.normal()
...     return OpFromGraph([rng], [next_rng, pt.exp(x)])

>>> lognormal_ofg = create_lognormal_ofg()

>>> rng = pt.random.shared_rng(seed=123, name="rng")
>>> next_rng, x = lognormal_ofg(rng)
>>> final_rng, y = lognormal_ofg(next_rng)

>>> f = pytensor.function([], [x, y], updates={rng: final_rng})

>>> f(), f(), f()
([array(0.37190332), array(0.69226486)], [array(3.62525729), array(1.21406523)], [array(2.50986985), array(1.78087317)])



Other backends (and their limitations)
======================================

Numba
-----

NumPy random generators can be natively used with the Numba backend.

>>> rng = pt.random.shared_rng(np.random.default_rng(123), name="randomstate_rng")
>>> next_rng, x = rng.normal()
>>> numba_fn = pytensor.function([], x, updates={rng: next_rng}, mode="NUMBA")
>>> _ = numba_fn.dprint(print_type=True)
[normal_rv{"(),()->()"}].1 [id A] <Scalar(float64, shape=())> 0
 ├─ [] [id B] <Vector(int64, shape=(0,))>
 ├─ randomstate_rng [id C] <RandomGeneratorType>
 ├─ NoneConst{None} [id D] <NoneTypeT>
 ├─ 0.0 [id E] <Scalar(float64, shape=())>
 └─ 1.0 [id F] <Scalar(float64, shape=())>
[normal_rv{"(),()->()"}].0 [id A] <RandomGeneratorType> 0
 └─ ···
<BLANKLINE>
Inner graphs:
<BLANKLINE>
[normal_rv{"(),()->()"}] [id A]
 ← normal_rv{"(),()->()"}.0 [id G] <RandomGeneratorType>
    ├─ i1 [id H] <RandomGeneratorType>
    ├─ i2 [id I] <NoneTypeT>
    ├─ i3 [id J] <Scalar(float64, shape=())>
    └─ i4 [id K] <Scalar(float64, shape=())>
 ← normal_rv{"(),()->()"}.1 [id G] <Scalar(float64, shape=())>
    └─ ···

>>> print(numba_fn(), numba_fn())
-0.9891213503478509 -0.3677866514678832

Numba converts regular RandomVariable to RandomVariableWithCoreShape Ops, subtly noted by a bracketed `[normal_rv{...}]` in the dprint.

>>> type(numba_fn.maker.fgraph.outputs[0].owner.op)
<class 'pytensor.tensor.random.op.RandomVariableWithCoreShape'>

JAX
---

JAX uses a different type of PRNG than those of NumPy: a NumPy ``Generator`` holds a PCG64 bit generator state, while JAX advances a threefry key.
Because a single PyTensor graph can be transpiled to multiple backends, the shared RNG cannot store one representation that all backends use natively.

Instead, PyTensor keeps a per-backend *view* of the shared RNG's state (the JAX view expands the bit generator state with a ``jax_state`` entry).
The shared variable is used directly in the graph, and reads and writes are reconciled so that ``get_value`` always observes the latest advancement and ``set_value`` and ``updates`` always take effect, regardless of which backend last touched the RNG.
Reconciliation keeps the bookkeeping consistent, but crossing backends still reseeds the stream (see the warning below).

>>> rng = pt.random.shared_rng(np.random.default_rng(123), name="rng")
>>> next_rng, x = rng.uniform()
>>> jax_fn = pytensor.function([], [x], updates={rng: next_rng}, mode="JAX")
>>> _ = pytensor.dprint(jax_fn, print_type=True) # doctest: +ELLIPSIS
uniform_rv{"(),()->()"}.1 [id A] <Scalar(float64, shape=())> 0
 ├─ rng [id B] <RandomGeneratorType>
 ├─ NoneConst{None} [id C] <NoneTypeT>
 ├─ 0.0 [id D] <Scalar(float32, shape=())>
 └─ 1.0 [id E] <Scalar(float32, shape=())>
uniform_rv{"(),()->()"}.0 [id A] <RandomGeneratorType> 0
 └─ ···

>>> print(jax_fn(), jax_fn())
[Array(0.91459085, dtype=float64)] [Array(0.01298661, dtype=float64)]

``set_value`` propagates to the JAX evaluation, so reseeding the shared RNG resets the stream:

>>> rng.set_value(np.random.default_rng(123))
>>> print(jax_fn(), jax_fn())
[Array(0.91459085, dtype=float64)] [Array(0.01298661, dtype=float64)]

.. warning::

    Because the algorithms differ, once one backend has drawn from the RNG, another backend cannot continue producing that same sequence of numbers.
    It can only reseed its own generator from the current state, which is deterministic but starts a new, unrelated stream.
    This reseed loses the stream in either direction, so crossing backends emits a ``UserWarning``. Within a single backend nothing is converted and the RNG advances at zero cost.

If a function's backend-specific updates are private (it only needs to observe host ``set_value`` and never has to feed its advanced state back to other backends or ``get_value``), pass ``readback=False`` to :func:`pytensor.function`.
This drops all per-call cross-backend bookkeeping from the compiled function.

>>> rng = pt.random.shared_rng(np.random.default_rng(123), name="rng")
>>> next_rng, x = rng.uniform()
>>> jax_fn = pytensor.function([], [x], updates={rng: next_rng}, mode="JAX", readback=False)
>>> print(jax_fn(), jax_fn())
[Array(0.91459085, dtype=float64)] [Array(0.01298661, dtype=float64)]

``set_value`` is still observed, so reseeding with the same seed deterministically reproduces the original draws:

>>> rng.set_value(np.random.default_rng(123))
>>> print(jax_fn(), jax_fn())
[Array(0.91459085, dtype=float64)] [Array(0.01298661, dtype=float64)]

But ``get_value`` no longer observes the function's *own* updates. The JAX view advances privately and is never read back, so the host stays exactly at whatever was last set. Even after a call that advanced the JAX view, ``get_value`` still matches a freshly seeded seed-123 generator:

>>> rng.set_value(np.random.default_rng(123))
>>> _ = jax_fn()   # advances the private JAX view
>>> rng.get_value().uniform() == np.random.default_rng(123).uniform()
True

With the default ``readback=True`` the same check would be ``False``, because the function's update is read back into the shared variable.
