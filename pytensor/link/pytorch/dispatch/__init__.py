# isort: off
from pytensor.link.pytorch.dispatch.basic import pytorch_funcify, pytorch_typify

# # Load dispatch specializations
import pytensor.link.pytorch.dispatch.scalar
import pytensor.link.pytorch.dispatch.elemwise
import pytensor.link.pytorch.dispatch.extra_ops
# isort: on
