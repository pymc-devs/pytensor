(libdoc_xtensor_random)=
# `xtensor.random` Random number generator operations

## Creating RNG variables

```{eval-rst}
.. autofunction:: pytensor.xtensor.random.variable.rng

.. autofunction:: pytensor.xtensor.random.variable.shared_rng
```

## Distributions

All distributions are available as methods on {class}`~pytensor.xtensor.random.variable.XRandomGeneratorVariable`
(e.g. `rng.normal()`). They accept `core_dims` and `extra_dims` parameters
for named-dimension support.

```{eval-rst}
.. autoclass:: pytensor.xtensor.random.variable.XRandomGeneratorVariable
   :members:
   :exclude-members: clone, dprint, eval, as_xrv, get_parents
```
