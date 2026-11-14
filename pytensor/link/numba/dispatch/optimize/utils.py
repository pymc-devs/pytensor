from typing import Any

import numpy as np

from pytensor.link.numba.dispatch import basic as numba_basic


# The solvers reproduce scipy's IEEE arithmetic, and give inf or NaN where a division
# by zero would raise
_SOLVER_JIT_OPTIONS: dict[str, Any] = {"fastmath": False, "error_model": "numpy"}


# Each solver is built per Op as a closure over that Op's compiled objective. numba
# caches a closure over a compiled function, but not a function that receives one as
# an argument, nor any function that calls it.
def _scalar_objective(objective, dtype, derivatives):
    # Evaluates at x cast to the input dtype, as perform does, with the extra inputs as one
    # tuple. Returns the value, or the value and its first `derivatives` derivatives, as
    # Python scalars.
    as_input_dtype = np.dtype(dtype).type

    if derivatives == 0:

        def scalar_objective(x, args):
            outputs = objective(np.asarray(as_input_dtype(x)), *args)
            return outputs[0].item()

    elif derivatives == 1:

        def scalar_objective(x, args):
            outputs = objective(np.asarray(as_input_dtype(x)), *args)
            return outputs[0].item(), outputs[1].item()

    else:

        def scalar_objective(x, args):
            outputs = objective(np.asarray(as_input_dtype(x)), *args)
            return outputs[0].item(), outputs[1].item(), outputs[2].item()

    return numba_basic.numba_njit(**_SOLVER_JIT_OPTIONS)(scalar_objective)


def _scalar_types(point_dtype, value_dtype):
    # The scalar types of the iterates, the objective's values, and expressions that mix
    # them, which a solver casts its literals to as NumPy casts Python scalars
    point = np.dtype(point_dtype).type
    value = np.dtype(value_dtype).type
    return point, value, np.promote_types(point, value).type
