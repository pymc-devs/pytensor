.. _libdoc_linalg:

===================================================================
:mod:`tensor.linalg` --  Linear Algebra Operations
===================================================================

.. module:: tensor.linalg
   :platform: Unix, Windows
   :synopsis: Linear Algebra Operations

The :mod:`pytensor.tensor.linalg` module exposes the user-facing linear
algebra API.

Constructors
============

.. autofunction:: pytensor.tensor.linalg.block_diag

Decomposition
=============

.. autofunction:: pytensor.tensor.linalg.cholesky
.. autofunction:: pytensor.tensor.linalg.lu
.. autofunction:: pytensor.tensor.linalg.lu_factor
.. autofunction:: pytensor.tensor.linalg.pivot_to_permutation
.. autofunction:: pytensor.tensor.linalg.qr
.. autofunction:: pytensor.tensor.linalg.svd
.. autofunction:: pytensor.tensor.linalg.eig
.. autofunction:: pytensor.tensor.linalg.eigh
.. autofunction:: pytensor.tensor.linalg.eigvalsh
.. autofunction:: pytensor.tensor.linalg.schur
.. autofunction:: pytensor.tensor.linalg.qz
.. autofunction:: pytensor.tensor.linalg.ordqz

Inverse
=======

.. autofunction:: pytensor.tensor.linalg.inv
.. autofunction:: pytensor.tensor.linalg.pinv
.. autofunction:: pytensor.tensor.linalg.tensorinv

Products
========

.. autofunction:: pytensor.tensor.linalg.kron
.. autofunction:: pytensor.tensor.linalg.matrix_dot
.. autofunction:: pytensor.tensor.linalg.matrix_power
.. autofunction:: pytensor.tensor.linalg.expm

Solve
=====

.. autofunction:: pytensor.tensor.linalg.solve
.. autofunction:: pytensor.tensor.linalg.solve_triangular
.. autofunction:: pytensor.tensor.linalg.lu_solve
.. autofunction:: pytensor.tensor.linalg.cho_solve
.. autofunction:: pytensor.tensor.linalg.lstsq
.. autofunction:: pytensor.tensor.linalg.tensorsolve
.. autofunction:: pytensor.tensor.linalg.tridiagonal_lu_factor
.. autofunction:: pytensor.tensor.linalg.tridiagonal_lu_solve
.. autofunction:: pytensor.tensor.linalg.solve_continuous_lyapunov
.. autofunction:: pytensor.tensor.linalg.solve_discrete_lyapunov
.. autofunction:: pytensor.tensor.linalg.solve_discrete_are
.. autofunction:: pytensor.tensor.linalg.solve_sylvester

Summary
=======

.. autofunction:: pytensor.tensor.linalg.det
.. autofunction:: pytensor.tensor.linalg.slogdet
.. autofunction:: pytensor.tensor.linalg.norm
.. autofunction:: pytensor.tensor.linalg.trace
