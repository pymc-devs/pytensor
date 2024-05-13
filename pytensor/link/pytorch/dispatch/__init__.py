# isort: off
from pytensor.link.pytorch.dispatch.basic import pytorch_funcify, pytorch_typify

# # Load dispatch specializations
import pytensor.link.pytorch.dispatch.scalar
# import pytensor.link.jax.dispatch.tensor_basic
# import pytensor.link.jax.dispatch.subtensor
# import pytensor.link.jax.dispatch.shape
# import pytensor.link.jax.dispatch.extra_ops
# import pytensor.link.jax.dispatch.nlinalg
# import pytensor.link.jax.dispatch.slinalg
# import pytensor.link.jax.dispatch.random
import pytensor.link.pytorch.dispatch.elemwise
# import pytensor.link.jax.dispatch.scan
# import pytensor.link.jax.dispatch.sparse
# import pytensor.link.jax.dispatch.blockwise

# isort: on
