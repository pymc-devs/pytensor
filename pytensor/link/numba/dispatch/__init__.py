# isort: off
from pytensor.link.numba.dispatch.basic import numba_funcify, numba_typify

# Load dispatch specializations
import pytensor.link.numba.dispatch.blockwise
import pytensor.link.numba.dispatch.compile_ops
import pytensor.link.numba.dispatch.elemwise
import pytensor.link.numba.dispatch.extra_ops
import pytensor.link.numba.dispatch.nlinalg
import pytensor.link.numba.dispatch.random
import pytensor.link.numba.dispatch.scan
import pytensor.link.numba.dispatch.scalar
import pytensor.link.numba.dispatch.shape
import pytensor.link.numba.dispatch.signal
import pytensor.link.numba.dispatch.slinalg
import pytensor.link.numba.dispatch.sort
import pytensor.link.numba.dispatch.sparse
import pytensor.link.numba.dispatch.subtensor
import pytensor.link.numba.dispatch.tensor_basic
import pytensor.link.numba.dispatch.typed_list


# isort: on
