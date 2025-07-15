# isort: off
from pytensor.link.pytorch.dispatch.basic import pytorch_funcify, pytorch_typify

# # Load dispatch specializations
import pytensor.link.pytorch.dispatch.blas
import pytensor.link.pytorch.dispatch.scalar
import pytensor.link.pytorch.dispatch.elemwise
import pytensor.link.pytorch.dispatch.math
import pytensor.link.pytorch.dispatch.extra_ops
import pytensor.link.pytorch.dispatch.nlinalg
import pytensor.link.pytorch.dispatch.slinalg
import pytensor.link.pytorch.dispatch.shape
import pytensor.link.pytorch.dispatch.sort
import pytensor.link.pytorch.dispatch.subtensor
import pytensor.link.pytorch.dispatch.blockwise
# isort: on
