# isort: off
from pytensor.link.jax.dispatch.basic import jax_funcify, jax_typify

# Load dispatch specializations
import pytensor.link.jax.dispatch.blas
import pytensor.link.jax.dispatch.blockwise
import pytensor.link.jax.dispatch.einsum
import pytensor.link.jax.dispatch.elemwise
import pytensor.link.jax.dispatch.extra_ops
import pytensor.link.jax.dispatch.pad
import pytensor.link.jax.dispatch.math
import pytensor.link.jax.dispatch.nlinalg
import pytensor.link.jax.dispatch.random
import pytensor.link.jax.dispatch.scalar
import pytensor.link.jax.dispatch.scan
import pytensor.link.jax.dispatch.shape
import pytensor.link.jax.dispatch.signal
import pytensor.link.jax.dispatch.slinalg
import pytensor.link.jax.dispatch.sort
import pytensor.link.jax.dispatch.sparse
import pytensor.link.jax.dispatch.subtensor
import pytensor.link.jax.dispatch.tensor_basic

# isort: on
