import pytensor.tensor.rewriting.basic
import pytensor.tensor.rewriting.blas
import pytensor.tensor.rewriting.blas_c
import pytensor.tensor.rewriting.blas_scipy
import pytensor.tensor.rewriting.blockwise
import pytensor.tensor.rewriting.elemwise
import pytensor.tensor.rewriting.extra_ops

# Register JAX specializations
import pytensor.tensor.rewriting.jax
import pytensor.tensor.rewriting.linalg
import pytensor.tensor.rewriting.math
import pytensor.tensor.rewriting.shape
import pytensor.tensor.rewriting.special
import pytensor.tensor.rewriting.subtensor
import pytensor.tensor.rewriting.uncanonicalize
