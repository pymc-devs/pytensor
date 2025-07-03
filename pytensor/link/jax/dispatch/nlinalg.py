import jax.numpy as jnp

from pytensor.link.jax.dispatch import jax_funcify
from pytensor.tensor.nlinalg import (
    SVD,
    Det,
    Eig,
    Eigh,
    KroneckerProduct,
    MatrixInverse,
    MatrixPinv,
    SLogDet,
)


@jax_funcify.register(SVD)
def jax_funcify_SVD(op, **kwargs):
    full_matrices = op.full_matrices
    compute_uv = op.compute_uv

    def svd(x, full_matrices=full_matrices, compute_uv=compute_uv):
        return jnp.linalg.svd(x, full_matrices=full_matrices, compute_uv=compute_uv)

    return svd


@jax_funcify.register(Det)
def jax_funcify_Det(op, **kwargs):
    def det(x):
        return jnp.linalg.det(x)

    return det


@jax_funcify.register(SLogDet)
def jax_funcify_SLogDet(op, **kwargs):
    def slogdet(x):
        return jnp.linalg.slogdet(x)

    return slogdet


@jax_funcify.register(Eig)
def jax_funcify_Eig(op, **kwargs):
    def eig(x):
        return jnp.linalg.eig(x)

    return eig


@jax_funcify.register(Eigh)
def jax_funcify_Eigh(op, **kwargs):
    uplo = op.UPLO

    def eigh(x, uplo=uplo):
        return jnp.linalg.eigh(x, UPLO=uplo)

    return eigh


@jax_funcify.register(MatrixInverse)
def jax_funcify_MatrixInverse(op, **kwargs):
    def matrix_inverse(x):
        return jnp.linalg.inv(x)

    return matrix_inverse


@jax_funcify.register(MatrixPinv)
def jax_funcify_Pinv(op, **kwargs):
    def pinv(x):
        return jnp.linalg.pinv(x, hermitian=op.hermitian)

    return pinv


@jax_funcify.register(KroneckerProduct)
def jax_funcify_KroneckerProduct(op, **kwargs):
    def _kron(x, y):
        return jnp.kron(x, y)

    return _kron
