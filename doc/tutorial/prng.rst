.. _prng:

===========================================
Pseudo random number generation in PyTensor
===========================================

PyTensor has native support for `pseudo random number generation (PRNG) <https://en.wikipedia.org/wiki/Pseudorandom_number_generator>`_.
This document describes how PRNGs are implemented in PyTensor, via the RandomVariable Operator.
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

Every numpy Generator holds a BitGenerator, which is able to generate high-quality sequences of pseudo random bits.
Numpy generators convert these sequences of bits into sequences of numbers that follow a specific statistical distribution.
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

Scipy wraps these Numpy routines in a slightly different API.