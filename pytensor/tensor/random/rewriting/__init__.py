# TODO: This is for backward-compatibility; remove when reasonable.
from pytensor.tensor.random.rewriting.basic import *


# isort: off

# Register Numba and JAX specializations
import pytensor.tensor.random.rewriting.numba
import pytensor.tensor.random.rewriting.jax

# isort: on
