import jax
import jax.numpy as jnp

from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor.linalg.products import Expm, KroneckerProduct, Sqrtm


@jax_funcify.register(KroneckerProduct)
def jax_funcify_KroneckerProduct(op, **kwargs):
    def _kron(x, y):
        return jnp.kron(x, y)

    return _kron


@jax_funcify.register(Expm)
def jax_funcify_Expm(op, **kwargs):
    def expm(x):
        return jax.scipy.linalg.expm(x)

    return expm


@jax_funcify.register(Sqrtm)
def jax_funcify_Sqrtm(op, node, **kwargs):
    out_dtype = node.outputs[0].type.dtype

    if out_dtype.startswith("complex"):

        def sqrtm(x):
            return jax.scipy.linalg.sqrtm(x).astype(out_dtype)

        return sqrtm

    def sqrtm(x):
        # LAPACK reports real eigenvalues of a real matrix with an exactly zero
        # imaginary part, so this test matches scipy's real Schur blocks
        eigenvalues = jnp.linalg.eigvals(x)
        has_real_root = ~jnp.any((eigenvalues.imag == 0) & (eigenvalues.real < 0))
        root = jax.scipy.linalg.sqrtm(x).real.astype(out_dtype)
        return jnp.where(has_real_root, root, jnp.nan)

    return sqrtm
