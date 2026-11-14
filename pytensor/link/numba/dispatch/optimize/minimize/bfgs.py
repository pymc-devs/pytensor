# Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np

from pytensor.link.numba.dispatch.basic import numba_njit
from pytensor.link.numba.dispatch.optimize.minimize.line_search import (
    CONVERGED,
    _dcsrch,
    _line_search_wolfe2,
)


@numba_njit(fastmath=False)
def _vecnorm(x, order):
    # Loop form of SciPy's vecnorm that skips the np.abs(x) temporary. A NaN
    # entry makes the norm NaN, as in NumPy.
    if order == np.inf:
        total = 0.0
        for i in range(x.size):
            value = abs(x[i])
            if np.isnan(value):
                return value
            total = max(total, value)
        return total
    if order == -np.inf:
        total = np.inf
        for i in range(x.size):
            value = abs(x[i])
            if np.isnan(value):
                return value
            total = min(total, value)
        return total
    total = 0.0
    for i in range(x.size):
        total += abs(x[i]) ** order
    return total ** (1.0 / order)


@numba_njit(fastmath=False, inline="always")
def _line_search_wolfe12(evaluate, x, p, args, f0, previous_f, g0, c1, c2, trial_x):
    d0 = np.dot(g0, p)
    if not np.isfinite(d0) or d0 >= 0:
        return 0.0, f0, g0, False

    alpha = min(1.0, 2.02 * (f0 - previous_f) / d0)
    # SciPy keeps a zero step, which fails both of its searches.
    if alpha <= 0 or not np.isfinite(alpha):
        alpha = 1.0
    # SciPy's line_search_wolfe12 treats DCSRCH warnings as failures.
    step, f, g, status, _ = _dcsrch(
        evaluate,
        x,
        p,
        args,
        f0,
        g0,
        d0,
        alpha,
        c1,
        c2,
        1e-14,
        1e-100,
        1e100,
        100,
        trial_x,
    )
    if status == CONVERGED:
        return step, f, g, True
    return _line_search_wolfe2(evaluate, x, p, args, f0, g0, d0, alpha, c1, c2, trial_x)


@numba_njit(fastmath=False, inline="always")
def _bfgs(evaluate, x, args, gtol, maxiter, norm, xrtol, c1, c2):
    f, g = evaluate(x, *args)
    if maxiter is None:
        maxiter = x.size * 200
    if maxiter <= 0 or not np.isfinite(f) or not np.all(np.isfinite(g)):
        return x, False
    if _vecnorm(g, norm) <= gtol:
        return x, True

    identity = np.eye(x.size)
    p, s, y = np.empty(x.size), np.empty(x.size), np.empty(x.size)
    hy, w, trial_x = np.empty(x.size), np.empty(x.size), np.empty(x.size)
    h = identity
    updated = False
    previous_f = f + np.linalg.norm(g) / 2
    # Like SciPy's k, steps counts accepted steps only. Restarts cannot run away:
    # a second failure before the next accepted update returns.
    steps = 0
    while steps < maxiter:
        np.dot(h, g, p)
        p *= -1.0
        alpha, new_f, new_g, found = _line_search_wolfe12(
            evaluate, x, p, args, f, previous_f, g, c1, c2, trial_x
        )
        if not found:
            if not updated:
                return x, False
            # Like SciPy's L-BFGS-B, discard the curvature history and retry
            # with steepest descent before giving up.
            h = identity
            updated = False
            continue

        steps += 1
        for j in range(x.size):
            s[j] = alpha * p[j]
            y[j] = new_g[j] - g[j]
            x[j] += s[j]
        descent = -np.dot(g, s)
        previous_f, f, g = f, new_f, new_g
        if _vecnorm(g, norm) <= gtol:
            return x, steps < maxiter
        if np.linalg.norm(s) <= xrtol * (xrtol + np.linalg.norm(x)):
            return x, steps < maxiter
        if not np.isfinite(f) or not np.all(np.isfinite(g)):
            return x, False

        curvature = np.dot(y, s)
        if curvature <= np.finfo(np.float64).eps * descent:
            continue
        if not updated:
            # Scale the identity to the observed curvature (Nocedal & Wright
            # eq. 6.20). Otherwise the update must cancel O(1) entries down to
            # 1 / curvature, and rounding can leave h indefinite.
            h = identity * (curvature / np.dot(y, y))
            updated = True
        # Expanded form of SciPy's (I - rho s y') h (I - rho y s') + rho s s',
        # which costs O(n^2) instead of two n x n matrix products. It updates h
        # in place, which is safe because h was rebound to a fresh scaled array
        # when updated became True.
        rho = 1.0 / curvature
        np.dot(h, y, hy)
        coefficient = 0.5 * rho * rho * (curvature + np.dot(y, hy))
        for i in range(x.size):
            w[i] = coefficient * s[i] - rho * hy[i]
        for i in range(x.size):
            for j in range(x.size):
                h[i, j] = (h[i, j] + s[i] * w[j]) + w[i] * s[j]

    return x, False
