.. _prng:

===========================================
Pseudo random number generation in PyTensor
===========================================

PyTensor has native support for `pseudo random number generation (PRNG) <https://en.wikipedia.org/wiki/Pseudorandom_number_generator>`_.

This document describes the details of how PRNGs are implemented in PyTensor, via the RandomVariable Operator.
For a more applied example see :ref:`using_random_numbers`

We also discuss how initial seeding and seeding updates are implemented, and some harder cases such as using RandomVariables inside Scan, or with other backends like JAX.

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

>>> rng # doctest: +SKIP
Generator(PCG64) at 0x7F6C04535820

Every NumPy Generator holds a BitGenerator, which is able to generate high-quality sequences of pseudo random bits.
NumPy generators' methods convert these sequences of bits into sequences of numbers that follow a specific statistical distribution.
For more details, you can read `NumPy random sampling documentation <https://numpy.org/doc/stable/reference/random>`_.

>>> rng.bit_generator # doctest: +SKIP
<numpy.random._pcg64.PCG64 at 0x7f6c045030f0>

>>> rng.bit_generator.state # doctest: +SKIP
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

>>> bad_uniform_rng(rng, size=5)
array([0.033, 0.972, 0.459, 0.71 , 0.765])

SciPy
-----

SciPy wraps these NumPy routines in a slightly different API.

>>> import scipy.stats as st
>>> rng = np.random.default_rng(seed=123)
>>> print(st.uniform.rvs(size=2, random_state=rng), st.uniform.rvs(size=2, random_state=rng))
[0.68235186 0.05382102] [0.22035987 0.18437181]

PyTensor
--------

PyTensor does not implement its own bit/generators methods.
Just like SciPy, it borrows NumPy routines directly.

The low-level API of PyTensor RNGs is similar to that of SciPy,
whereas the higher-level API of RandomStreams is more like that of NumPy.
We will look at RandomStreams shortly, but we will start with the low-level API.

>>> import pytensor
>>> import pytensor.tensor as pt

>>> rng = pt.random.type.RandomGeneratorType()("rng")
>>> x = pt.random.uniform(size=2, rng=rng)
>>> f = pytensor.function([rng], x)

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

>>> rng_val.bit_generator.advance(1)
>>> print(f(rng_val), f(rng_val))
[0.05382102 0.22035987] [0.05382102 0.22035987]

>>> rng_val.bit_generator.advance(1)
>>> print(f(rng_val), f(rng_val))
[0.22035987 0.18437181] [0.22035987 0.18437181]

Updating the bit generator manually is not a good practice.
For starters, it may be unclear how much we have to advance it!

In this case we had to advance it twice to get two completely new draws, because the inner function uses two states.
But other distributions could need more states for a single draw, or they could be clever and reuse the same state for multiple draws.

Because it is not in generally possible to know how much one should modify the generator's bit generator,
PyTensor RandomVariables actually return the used generator as a hidden output.
This generator can be safely used again because it contains the bit generator that was already modified when taking draws.

>>> next_rng, x = x.owner.outputs
>>> next_rng.type, x.type
(RandomGeneratorType, TensorType(float64, (2,)))

>>> next_rng.name = "next_rng"
>>> x.name = "x"
>>> pytensor.dprint([next_rng, x], print_type=True) # doctest: +SKIP
uniform_rv{"(),()->()"}.0 [id A] <RandomGeneratorType> 'next_rng'
 ├─ rng [id B] <RandomGeneratorType>
 ├─ [2] [id C] <Vector(int64, shape=(1,))>
 ├─ ExpandDims{axis=0} [id D] <Vector(float32, shape=(1,))>
 │  └─ 0.0 [id E] <Scalar(float32, shape=())>
 └─ ExpandDims{axis=0} [id F] <Vector(float32, shape=(1,))>
    └─ 1.0 [id G] <Scalar(float32, shape=())>
uniform_rv{"(),()->()"}.1 [id A] <Vector(float64, shape=(2,))> 'x'
 └─ ···

We can see the single node with [id A], has two outputs, which we named next_rng and x. By default only the second output x is given to the user directly, and the other is "hidden".

We can compile a function that returns the next_rng explicitly, so that we can use it as the input of the function in subsequent calls.

>>> f = pytensor.function([rng], [next_rng, x])

>>> rng_val = np.random.default_rng(123)
>>> next_rng_val, x = f(rng_val)
>>> print(x)
[0.68235186 0.05382102]

>>> next_rng_val, x = f(next_rng_val)
>>> print(x)
[0.22035987 0.18437181]

>>> next_rng_val, x = f(next_rng_val)
>>> print(x)
[0.1759059  0.81209451]

Shared variables
================

At this point we can make use of PyTensor shared variables.
Shared variables are global variables that don't need (and can't) be passed as explicit inputs to the functions where they are used.

>>> rng = pytensor.shared(np.random.default_rng(123))
>>> next_rng, x = pt.random.uniform(rng=rng).owner.outputs
>>>
>>> f = pytensor.function([], [next_rng, x])
>>>
>>> next_rng_val, x = f()
>>> print(x)
0.6823518632481435

We can update the value of shared variables across calls.

>>> rng.set_value(next_rng_val)
>>> next_rng_val, x = f()
>>> print(x)
0.053821018802222675

>>> rng.set_value(next_rng_val)
>>> next_rng_val, x = f()
>>> print(x)
0.22035987277261138

The real benefit of using shared variables is that we can automate this updating via the aptly named updates kwarg of PyTensor functions.

In this case it makes sense to simply replace the original value by the next_rng_val (there is not really any other operation we can do with PyTensor RNGs)

>>> rng = pytensor.shared(np.random.default_rng(123))
>>> next_rng, x = pt.random.uniform(rng=rng).owner.outputs
>>>
>>> f = pytensor.function([], x, updates={rng: next_rng})
>>>
>>> f(), f(), f()
(array(0.68235186), array(0.05382102), array(0.22035987))

Another way of doing that is setting a default_update in the shared RNG variable

>>> rng = pytensor.shared(np.random.default_rng(123))
>>> next_rng, x = pt.random.uniform(rng=rng).owner.outputs
>>>
>>> rng.default_update = next_rng
>>> f = pytensor.function([], x)
>>>
>>> f(), f(), f()
(array(0.68235186), array(0.05382102), array(0.22035987))

This is exactly what RandomStream does behind the scenes

>>> srng = pt.random.RandomStream(seed=123)
>>> x = srng.uniform()
>>> x.owner.inputs[0], x.owner.inputs[0].default_update  # doctest: +SKIP
(RNG(<Generator(PCG64) at 0x7FA45F4A3760>), uniform_rv{"(),()->()"}.0)

>>> f = pytensor.function([], x)
>>> print(f(), f(), f())
0.19365083425294516 0.7541389670292019 0.2762903411491048

From the example here, you can see that RandomStream uses a NumPy-like API in contrast to
the SciPy-like API of `pytensor.tensor.random`. Full documentation can be found at
:doc:`libdoc_tensor_random_basic`.

Shared RNGs are created by default
----------------------------------

If no rng is provided to a RandomVariable Op, a shared RandomGenerator is created automatically.

This can give the appearance that PyTensor functions of random variables don't have any variable inputs,
but this is not true.
They are simply shared variables.

>>> x = pt.random.normal()
>>> x.owner.inputs[0] # doctest: +SKIP
RNG(<Generator(PCG64) at 0x7FA45ED80660>)

Reseeding
---------

Shared RNG variables can be "reseeded" by setting them to the original RNG

>>> rng = pytensor.shared(np.random.default_rng(123))
>>> next_rng, x = pt.random.normal(rng=rng).owner.outputs
>>>
>>> rng.default_update = next_rng
>>> f = pytensor.function([], x)
>>>
>>> print(f(), f())
>>> rng.set_value(np.random.default_rng(123))
>>> print(f(), f())
-0.9891213503478509 -0.3677866514678832
-0.9891213503478509 -0.3677866514678832

RandomStreams provide a helper method to achieve the same

>>> rng = pt.random.RandomStream(seed=123)
>>> x = srng.normal()
>>> f = pytensor.function([], x)
>>>
>>> print(f(), f())
>>> srng.seed(123)
>>> print(f(), f())
-0.5812234917408711 -0.047499225218726786
-0.5812234917408711 -0.047499225218726786

Inplace optimization
====================

As mentioned, RandomVariable Ops default to making a copy of the input RNG before using it, which can be quite slow.

>>> rng = np.random.default_rng(123)
>>> rng_shared = pytensor.shared(rng, name="rng")
>>> x = pt.random.uniform(rng=rng_shared, name="x")
>>> f = pytensor.function([], x)
>>> pytensor.dprint(f, print_destroy_map=True) # doctest: +SKIP
uniform_rv{"(),()->()"}.1 [id A] 'x' 0
 ├─ rng [id B]
 ├─ NoneConst{None} [id C]
 ├─ 0.0 [id D]
 └─ 1.0 [id E]


>>> %timeit f()  # doctest: +SKIP
81.8 µs ± 15.4 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

>>> %timeit rng.uniform()  # doctest: +SKIP
2.15 µs ± 63.4 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

Like other PyTensor operators, RandomVariable's can be given permission to modify inputs inplace during their operation.

In this case, there is a `inplace` flag that when `true` tells the RandomVariable Op that it is safe to modify the RNG input inplace.
If the flag is set, the RNG will not be copied before taking random draws.

>>> x.owner.op.inplace
False

For illustration purposes, we will subclass the Uniform Op class and set inplace to True by default.

Users should never do this directly!

>>> class InplaceUniform(type(pt.random.uniform)):
>>>    inplace = True

>>> inplace_uniform = InplaceUniform()
>>> x = inplace_uniform()
>>> x.owner.op.inplace
True

>>> inplace_f = pytensor.function([], x, accept_inplace=True)
>>> pytensor.dprint(inplace_f, print_destroy_map=True) # doctest: +SKIP
uniform_rv{"(),()->()"}.1 [id A] d={0: [0]} 0
 ├─ RNG(<Generator(PCG64) at 0x7FA45ED81540>) [id B]
 ├─ NoneConst{None} [id C]
 ├─ 0.0 [id D]
 └─ 1.0 [id E]

The destroy map annotation tells us that the first output of the x variable is allowed to modify the first input.

>>> %timeit inplace_f() # doctest: +SKIP
9.71 µs ± 2.06 µs per loop (mean ± std. dev. of 7 runs, 100000 loops each)

Performance is now much closer to calling NumPy directly, with a small overhead introduced by the PyTensor function.

The `random_make_inplace <https://github.com/pymc-devs/pytensor/blob/3fcf6369d013c597a9c964b2400a3c5e20aa8dce/pytensor/tensor/random/rewriting/basic.py#L42-L52>`_
rewrite automatically replaces RandomVariable Ops by their inplace counterparts, when such operation is deemed safe. This happens when:

#. An input RNG is flagged as `mutable` and is used in not used anywhere else.
#. A RNG is created intermediately and not used anywhere else.

The first case is true when a users uses the `mutable` `kwarg` directly.

>>> from pytensor.compile.io import In
>>> rng = pt.random.type.RandomGeneratorType()("rng")
>>> next_rng, x = pt.random.uniform(rng=rng).owner.outputs
>>> with pytensor.config.change_flags(optimizer_verbose=True):
>>>     inplace_f = pytensor.function([In(rng, mutable=True)], [x])
>>> print("")
>>> pytensor.dprint(inplace_f, print_destroy_map=True) # doctest: +SKIP
rewriting: rewrite random_make_inplace replaces uniform_rv{"(),()->()"}.out of uniform_rv{"(),()->()"}(rng, NoneConst{None}, 0.0, 1.0) with uniform_rv{"(),()->()"}.out of uniform_rv{"(),()->()"}(rng, NoneConst{None}, 0.0, 1.0)
uniform_rv{"(),()->()"}.1 [id A] d={0: [0]} 0
 ├─ rng [id B]
 ├─ NoneConst{None} [id C]
 ├─ 0.0 [id D]
 └─ 1.0 [id E]

Or, much more commonly, when a shared RNG is used and a (default or manual) update expression is given.
In this case, a RandomVariable is allowed to modify the RNG because the shared variable holding it will be rewritten anyway.

>>> rng = pytensor.shared(np.random.default_rng(), name="rng")
>>> next_rng, x = pt.random.uniform(rng=rng).owner.outputs
>>>
>>> inplace_f = pytensor.function([], [x], updates={rng: next_rng})
>>> pytensor.dprint(inplace_f, print_destroy_map=True) # doctest: +SKIP
uniform_rv{"(),()->()"}.1 [id A] d={0: [0]} 0
 ├─ rng [id B]
 ├─ NoneConst{None} [id C]
 ├─ 0.0 [id D]
 └─ 1.0 [id E]
uniform_rv{"(),()->()"}.0 [id A] d={0: [0]} 0
 └─ ···

The second case is not very common, because RNGs are not usually chained across multiple RandomVariable Ops.
See more details in the next section.

Multiple random variables
=========================

It's common practice to use separate RNG variables for each RandomVariable in PyTensor.

>>> rng_x = pytensor.shared(np.random.default_rng(123), name="rng_x")
>>> rng_y = pytensor.shared(np.random.default_rng(456), name="rng_y")
>>>
>>> next_rng_x, x = pt.random.normal(loc=0, scale=10, rng=rng_x).owner.outputs
>>> next_rng_y, y = pt.random.normal(loc=x, scale=0.1, rng=rng_y).owner.outputs
>>>
>>> next_rng_x.name = "next_rng_x"
>>> next_rng_y.name = "next_rng_y"
>>> rng_x.default_update = next_rng_x
>>> rng_y.default_update = next_rng_y
>>>
>>> f = pytensor.function([], [x, y])
>>> pytensor.dprint(f, print_type=True) # doctest: +SKIP
normal_rv{"(),()->()"}.1 [id A] <Scalar(float64, shape=())> 0
 ├─ rng_x [id B] <RandomGeneratorType>
 ├─ NoneConst{None} [id C] <NoneTypeT>
 ├─ 0 [id D] <Scalar(int8, shape=())>
 └─ 10 [id E] <Scalar(int8, shape=())>
normal_rv{"(),()->()"}.1 [id F] <Scalar(float64, shape=())> 1
 ├─ rng_y [id G] <RandomGeneratorType>
 ├─ NoneConst{None} [id C] <NoneTypeT>
 ├─ normal_rv{"(),()->()"}.1 [id A] <Scalar(float64, shape=())> 0
 │  └─ ···
 └─ 0.1 [id H] <Scalar(float64, shape=())>
normal_rv{"(),()->()"}.0 [id A] <RandomGeneratorType> 'next_rng_x' 0
 └─ ···
normal_rv{"(),()->()"}.0 [id F] <RandomGeneratorType> 'next_rng_y' 1
 └─ ···

>>> f(), f(), f()
([array(-9.8912135), array(-9.80160951)],
 [array(-3.67786651), array(-3.89026137)],
 [array(12.87925261), array(13.04327299)])

This is what RandomStream does as well

>>> srng = pt.random.RandomStream(seed=123)
>>> x = srng.normal(loc=0, scale=10)
>>> y = srng.normal(loc=x, scale=0.1)
>>>
>>> f = pytensor.function([], [x, y])
>>> pytensor.dprint(f, print_type=True) # doctest: +SKIP
normal_rv{"(),()->()"}.1 [id A] <Scalar(float64, shape=())> 0
 ├─ RNG(<Generator(PCG64) at 0x7FA45ED835A0>) [id B] <RandomGeneratorType>
 ├─ NoneConst{None} [id C] <NoneTypeT>
 ├─ 0 [id D] <Scalar(int8, shape=())>
 └─ 10 [id E] <Scalar(int8, shape=())>
normal_rv{"(),()->()"}.1 [id F] <Scalar(float64, shape=())> 1
 ├─ RNG(<Generator(PCG64) at 0x7FA45ED833E0>) [id G] <RandomGeneratorType>
 ├─ NoneConst{None} [id C] <NoneTypeT>
 ├─ normal_rv{"(),()->()"}.1 [id A] <Scalar(float64, shape=())> 0
 │  └─ ···
 └─ 0.1 [id H] <Scalar(float64, shape=())>
normal_rv{"(),()->()"}.0 [id A] <RandomGeneratorType> 0
 └─ ···
normal_rv{"(),()->()"}.0 [id F] <RandomGeneratorType> 1
 └─ ···

>>> f(), f(), f()
([array(-5.81223492), array(-5.85081162)],
 [array(-0.47499225), array(-0.64636099)],
 [array(-1.11452059), array(-1.09642036)])

We could have used a single rng.

>>> rng_x = pytensor.shared(np.random.default_rng(seed=123), name="rng_x")
>>> next_rng_x, x = pt.random.normal(loc=0, scale=1, rng=rng_x).owner.outputs
>>> next_rng_x.name = "next_rng_x"
>>> next_rng_y, y = pt.random.normal(loc=100, scale=1, rng=next_rng_x).owner.outputs
>>> next_rng_y.name = "next_rng_y"
>>>
>>> f = pytensor.function([], [x, y], updates={rng_x: next_rng_y})
>>> pytensor.dprint(f, print_type=True) # doctest: +SKIP
normal_rv{"(),()->()"}.1 [id A] <Scalar(float64, shape=())> 0
 ├─ rng_x [id B] <RandomGeneratorType>
 ├─ NoneConst{None} [id C] <NoneTypeT>
 ├─ 0 [id D] <Scalar(int8, shape=())>
 └─ 1 [id E] <Scalar(int8, shape=())>
normal_rv{"(),()->()"}.1 [id F] <Scalar(float64, shape=())> 1
 ├─ normal_rv{"(),()->()"}.0 [id A] <RandomGeneratorType> 'next_rng_x' 0
 │  └─ ···
 ├─ NoneConst{None} [id C] <NoneTypeT>
 ├─ 100 [id G] <Scalar(int8, shape=())>
 └─ 1 [id E] <Scalar(int8, shape=())>
normal_rv{"(),()->()"}.0 [id F] <RandomGeneratorType> 'next_rng_y' 1
 └─ ···

>>> f(), f()
([array(-0.98912135), array(99.63221335)],
 [array(1.28792526), array(100.19397442)])

It works, but that graph is slightly unorthodox in PyTensor.

One practical reason why, is that it is more difficult to define the correct update expression for the shared RNG variable.

One techincal reason why, is that it makes rewrites more challenging in cases where RandomVariables could otherwise be manipulated independently.

Creating multiple RNG variables
-------------------------------

RandomStreams generate high quality seeds for multiple variables, following the NumPy best practices https://numpy.org/doc/stable/reference/random/parallel.html#parallel-random-number-generation.

Users who sidestep RandomStreams, either by creating their own RNGs or relying on RandomVariable's default shared RNGs, should follow the same practice!

Random variables in inner graphs
================================

Scan
----

Scan works very similar to a function (that is called repeatedly inside an outer scope).

This means that random variables will always return the same output unless updates are specified.

>>> rng = pytensor.shared(np.random.default_rng(123), name="rng")
>>>
>>> def constant_step(rng):
>>>     return pt.random.normal(rng=rng)
>>>
>>> draws, updates = pytensor.scan(
>>>     fn=constant_step,
>>>     outputs_info=[None],
>>>     non_sequences=[rng],
>>>     n_steps=5,
>>>     strict=True,
>>> )
>>>
>>> f = pytensor.function([], draws, updates=updates)
>>> f(), f()
(array([-0.98912135, -0.98912135, -0.98912135, -0.98912135, -0.98912135]),
 array([-0.98912135, -0.98912135, -0.98912135, -0.98912135, -0.98912135]))

Scan accepts an update dictionary as an output to tell how shared variables should be updated after every iteration.

>>> rng = pytensor.shared(np.random.default_rng(123))
>>>
>>> def random_step(rng):
>>>     next_rng, x = pt.random.normal(rng=rng).owner.outputs
>>>     scan_update = {rng: next_rng}
>>>     return x, scan_update
>>>
>>> draws, updates = pytensor.scan(
>>>     fn=random_step,
>>>     outputs_info=[None],
>>>     non_sequences=[rng],
>>>     n_steps=5,
>>>     strict=True
>>> )
>>>
>>> f = pytensor.function([], draws)
>>> f(), f()
(array([-0.98912135, -0.36778665,  1.28792526,  0.19397442,  0.9202309 ]),
 array([-0.98912135, -0.36778665,  1.28792526,  0.19397442,  0.9202309 ]))

However, we still have to tell the outer function to update the shared RNG across calls, using the last state returned by the Scan

>>> f = pytensor.function([], draws, updates=updates)
>>> f(), f()
(array([-0.98912135, -0.36778665,  1.28792526,  0.19397442,  0.9202309 ]),
 array([ 0.57710379, -0.63646365,  0.54195222, -0.31659545, -0.32238912]))

**Default updates**

Like function, scan also respects shared variables default updates

>>> def random_step():
>>>     rng = pytensor.shared(np.random.default_rng(123), name="rng")
>>>     next_rng, x = pt.random.normal(rng=rng).owner.outputs
>>>     rng.default_update = next_rng
>>>     return x
>>>
>>> draws, updates = pytensor.scan(
>>>     fn=random_step,
>>>     outputs_info=[None],
>>>     non_sequences=[],
>>>     n_steps=5,
>>>     strict=True,
>>> )

>>> f = pytensor.function([], draws)
>>> f(), f()
(array([-0.98912135, -0.36778665,  1.28792526,  0.19397442,  0.9202309 ]),
 array([-0.98912135, -0.36778665,  1.28792526,  0.19397442,  0.9202309 ]))

The outer function still needs to be told the final update rule

>>> f = pytensor.function([], draws, updates=updates)
>>> f(), f()
(array([-0.98912135, -0.36778665,  1.28792526,  0.19397442,  0.9202309 ]),
 array([ 0.57710379, -0.63646365,  0.54195222, -0.31659545, -0.32238912]))

As expected, Scan only looks at default updates for shared variables created inside the user provided function.

>>> rng = pytensor.shared(np.random.default_rng(123), name="rng")
>>> next_rng, x = pt.random.normal(rng=rng).owner.outputs
>>> rng.default_update = next_rng
>>>
>>> def random_step(rng, x):
>>>     return x
>>>
>>> draws, updates = pytensor.scan(
>>>     fn=random_step,
>>>     outputs_info=[None],
>>>     non_sequences=[rng, x],
>>>     n_steps=5,
>>>     strict=True,
>>> )

>>> f = pytensor.function([], draws)
>>> f(), f()
(array([-0.98912135, -0.98912135, -0.98912135, -0.98912135, -0.98912135]),
 array([-0.36778665, -0.36778665, -0.36778665, -0.36778665, -0.36778665]))

**Limitations**

RNGs in Scan are only supported via shared variables in non-sequences at the moment

>>> rng = pt.random.type.RandomGeneratorType()("rng")
>>>
>>> def random_step(rng):
>>>     next_rng, x = pt.random.normal(rng=rng).owner.outputs
>>>     return next_rng, x
>>>
>>> try:
>>>     (next_rngs, draws), updates = pytensor.scan(
>>>         fn=random_step,
>>>         outputs_info=[rng, None],
>>>         n_steps=5,
>>>         strict=True
>>>     )
>>> except TypeError as err:
>>>     print(err)
Tensor type field must be a TensorType; found <class 'pytensor.tensor.random.type.RandomGeneratorType'>.

In the future, RandomGenerator variables may be allowed as explicit recurring states, rendering the internal use of updates optional or unnecessary

OpFromGraph
-----------

In contrast to Scan, non-shared RNG variables can be used directly in OpFromGraph

>>> from pytensor.compile.builders import OpFromGraph
>>>
>>> rng = pt.random.type.RandomGeneratorType()("rng")
>>>
>>> def lognormal(rng):
>>>     next_rng, x = pt.random.normal(rng=rng).owner.outputs
>>>     return [next_rng, pt.exp(x)]
>>>
>>> lognormal_ofg = OpFromGraph([rng], lognormal(rng))

>>> rng_x = pytensor.shared(np.random.default_rng(1), name="rng_x")
>>> rng_y = pytensor.shared(np.random.default_rng(2), name="rng_y")
>>>
>>> next_rng_x, x = lognormal_ofg(rng_x)
>>> next_rng_y, y = lognormal_ofg(rng_y)
>>>
>>> f = pytensor.function([], [x, y], updates={rng_x: next_rng_x, rng_y: next_rng_y})

>>> f(), f(), f()
([array(1.41281503), array(1.20810544)],
 [array(2.27417681), array(0.59288879)],
 [array(1.39157622), array(0.66162024)])

Also in contrast to Scan, there is no special treatment of updates for shared variables used in the inner graphs of OpFromGraph.

Any "updates" must be modeled as explicit outputs and used in the outer graph directly as in the example above.

This is arguably more clean.

Other backends (and their limitations)
======================================

Numba
-----

NumPy random generators can be natively used with the Numba backend.

>>> rng = pytensor.shared(np.random.default_rng(123), name="randomstate_rng")
>>> x = pt.random.normal(rng=rng)
>>> numba_fn = pytensor.function([], x, mode="NUMBA")
>>> pytensor.dprint(numba_fn, print_type=True)
[normal_rv{"(),()->()"}].1 [id A] <Scalar(float64, shape=())> 0
 ├─ [] [id B] <Vector(int64, shape=(0,))>
 ├─ randomstate_rng [id C] <RandomGeneratorType>
 ├─ NoneConst{None} [id D] <NoneTypeT>
 ├─ 0.0 [id E] <Scalar(float32, shape=())>
 └─ 1.0 [id F] <Scalar(float32, shape=())>
Inner graphs:
[normal_rv{"(),()->()"}] [id A]
 ← normal_rv{"(),()->()"}.0 [id G] <RandomGeneratorType>
    ├─ *1-<RandomGeneratorType> [id H] <RandomGeneratorType>
    ├─ *2-<NoneTypeT> [id I] <NoneTypeT>
    ├─ *3-<Scalar(float32, shape=())> [id J] <Scalar(float32, shape=())>
    └─ *4-<Scalar(float32, shape=())> [id K] <Scalar(float32, shape=())>
 ← normal_rv{"(),()->()"}.1 [id G] <Scalar(float64, shape=())>
    └─ ···

>>> print(numba_fn(), numba_fn())
-0.9891213503478509 -0.9891213503478509

JAX
---

JAX uses a different type of PRNG than those of NumPy. This means that the standard shared RNGs cannot be used directly in graphs transpiled to JAX.

Instead a copy of the Shared RNG variable is made, and its bit generator state is expanded with a jax_state entry. This is what's actually used by the JAX random variables.

In general, update rules are still respected, but they won't update/rely on the original shared variable.

>>> import jax
>>> rng = pytensor.shared(np.random.default_rng(123), name="rng")
>>> next_rng, x = pt.random.uniform(rng=rng).owner.outputs
>>> jax_fn = pytensor.function([], [x], updates={rng: next_rng}, mode="JAX")
>>> pytensor.dprint(jax_fn, print_type=True)
uniform_rv{"(),()->()"}.1 [id A] <Scalar(float64, shape=())> 0
 ├─ RNG(<Generator(PCG64) at 0x7FA448D68200>) [id B] <RandomGeneratorType>
 ├─ NoneConst{None} [id C] <NoneTypeT>
 ├─ 0.0 [id D] <Scalar(float32, shape=())>
 └─ 1.0 [id E] <Scalar(float32, shape=())>
uniform_rv{"(),()->()"}.0 [id A] <RandomGeneratorType> 0
 └─ ···

>>> print(jax_fn(), jax_fn())
[Array(0.07577298, dtype=float64)] [Array(0.09217023, dtype=float64)]

>>> # No effect on the jax evaluation
>>> rng.set_value(np.random.default_rng(123))
>>> print(jax_fn(), jax_fn())
[Array(0.13929162, dtype=float64)] [Array(0.45162648, dtype=float64)]

>>> [jax_rng] = jax_fn.input_storage[0].storage
>>> jax_rng
{'bit_generator': Array(1, dtype=int64, weak_type=True),
 'has_uint32': Array(0, dtype=int64, weak_type=True),
 'jax_state': Array([2647707238, 2709433097], dtype=uint32),
 'state': {'inc': Array(-9061352147377205305, dtype=int64),
  'state': Array(-6044258077699604239, dtype=int64)},
 'uinteger': Array(0, dtype=int64, weak_type=True)}

>>> [jax_rng] = jax_fn.input_storage[0].storage
>>> jax_rng["jax_state"] = jax.random.PRNGKey(0)
>>> print(jax_fn(), jax_fn())
[Array(0.57655083, dtype=float64)] [Array(0.50347362, dtype=float64)]

>>> [jax_rng] = jax_fn.input_storage[0].storage
>>> jax_rng["jax_state"] = jax.random.PRNGKey(0)
>>> print(jax_fn(), jax_fn())
[Array(0.57655083, dtype=float64)] [Array(0.50347362, dtype=float64)]

PyTensor could provide shared JAX-like RNGs and allow RandomVariables to accept them,
but that would break the spirit of one graph `->` multiple backends.

Alternatively, PyTensor could try to use a more general type for RNGs that can be used across different backends,
either directly or after some conversion operation (if such operations can be implemented in the different backends).
