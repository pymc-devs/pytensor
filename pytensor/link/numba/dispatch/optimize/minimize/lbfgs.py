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
from pytensor.link.numba.dispatch.optimize.minimize.line_search import FAILED, _dcsrch


# reassoc lets LLVM split and vectorize the dot-product reductions. It only
# reorders sums, so the directions match SciPy's up to rounding.
@numba_njit(fastmath={"reassoc"})
def _direction(g, s_history, y_history, rho_history, gamma, count, head):
    # Two-loop recursion over a ring buffer whose newest pair sits before head.
    m = rho_history.size
    q = -g
    alphas = np.zeros(m)
    for k in range(count):
        i = (head - 1 - k) % m
        total = 0.0
        for j in range(q.size):
            total += s_history[i, j] * q[j]
        alphas[i] = rho_history[i] * total
        for j in range(q.size):
            q[j] -= alphas[i] * y_history[i, j]
    q *= gamma
    for k in range(count - 1, -1, -1):
        i = (head - 1 - k) % m
        total = 0.0
        for j in range(q.size):
            total += y_history[i, j] * q[j]
        beta = rho_history[i] * total
        for j in range(q.size):
            q[j] += (alphas[i] - beta) * s_history[i, j]
    return q


@numba_njit(fastmath=False, inline="always")
def _lnsrlb(evaluate, x, p, args, f0, g0, alpha, maxls, trial_x):
    # SciPy lnsrlb's unbounded DCSRCH settings, which accept warnings as steps.
    d0 = np.dot(g0, p)
    if not np.isfinite(d0) or d0 >= 0:
        return 0.0, f0, g0, False, 0
    alpha, f, g, status, evaluations = _dcsrch(
        evaluate,
        x,
        p,
        args,
        f0,
        g0,
        d0,
        alpha,
        1e-3,
        0.9,
        0.1,
        0.0,
        1e10,
        maxls,
        trial_x,
    )
    return alpha, f, g, status != FAILED, evaluations


@numba_njit(fastmath=False, inline="always")
def _lbfgs(evaluate, x, args, m, ftol, gtol, maxiter, maxfun, maxls):
    if x.size == 0:
        return x, False
    f, g = evaluate(x, *args)
    if not np.isfinite(f) or not np.all(np.isfinite(g)):
        return x, False
    if np.max(np.abs(g)) <= gtol:
        return x, True

    s_history, y_history = np.zeros((m, x.size)), np.zeros((m, x.size))
    rho_history = np.zeros(m)
    s, y, trial_x = np.empty(x.size), np.empty(x.size), np.empty(x.size)
    gamma, count, head, iterations, nfev = 1.0, 0, 0, 0, 1
    while True:
        p = _direction(g, s_history, y_history, rho_history, gamma, count, head)
        alpha = min(1.0 / np.linalg.norm(p), 1e10) if iterations == 0 else 1.0
        alpha, new_f, new_g, found, evaluations = _lnsrlb(
            evaluate, x, p, args, f, g, alpha, maxls, trial_x
        )
        nfev += evaluations
        if not found:
            # SciPy restarts failed searches with an empty L-BFGS history.
            if count == 0:
                return x, False
            count, gamma = 0, 1.0
            continue

        descent = -np.dot(g, p) * alpha
        for j in range(x.size):
            s[j] = alpha * p[j]
            y[j] = new_g[j] - g[j]
            x[j] += s[j]
        old_f, f, g = f, new_f, new_g
        iterations += 1
        # SciPy's driver checks these limits before resuming convergence checks.
        if iterations >= maxiter or nfev > maxfun:
            return x, False
        if np.max(np.abs(g)) <= gtol:
            return x, True
        if old_f - f <= ftol * max(abs(old_f), abs(f), 1.0):
            return x, True

        curvature = np.dot(y, s)
        if curvature <= np.finfo(np.float64).eps * descent:
            continue
        s_history[head] = s
        y_history[head] = y
        rho_history[head] = 1.0 / curvature
        gamma = curvature / np.dot(y, y)
        head = (head + 1) % m
        count = min(count + 1, m)
