from typing import Any

import numpy as np

from pytensor.link.numba.dispatch import basic as numba_basic


# The solvers reproduce scipy's IEEE arithmetic, and give inf or NaN where a division
# by zero would raise
_SOLVER_JIT_OPTIONS: dict[str, Any] = {"fastmath": False, "error_model": "numpy"}

# Each solver is built per Op as a closure over that Op's compiled objective. numba
# caches a closure over a compiled function, but not a function that receives one as
# an argument, nor any function that calls it.


def _with_input_dtype(objective, dtype):
    """Wrap ``objective`` to take a float ``x`` and evaluate it at ``x`` cast to ``dtype``, as perform does."""
    as_input_dtype = np.dtype(dtype).type

    @numba_basic.numba_njit(**_SOLVER_JIT_OPTIONS)
    def typed_objective(x, *args):
        return objective(np.asarray(as_input_dtype(x)), *args)

    return typed_objective


def _make_value_at(objective):
    @numba_basic.numba_njit(**_SOLVER_JIT_OPTIONS)
    def value_at(x, args):
        return objective(x, *args)[0].item()

    return value_at


def _scalar_types(point_dtype, value_dtype):
    """Return the scalar types a solver casts its literals to, as NumPy casts Python scalars.

    Returns
    -------
    point : type
        Scalar type of the iterates.
    value : type
        Scalar type of the objective's values.
    mixed : type
        Scalar type of expressions that combine iterates and values.
    """
    point = np.dtype(point_dtype).type
    value = np.dtype(value_dtype).type
    return point, value, np.promote_types(point, value).type
