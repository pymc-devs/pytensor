"""Unbounded L-BFGS: JAX's two-loop recursion with SciPy stopping/search settings.

Direction: JAX's _lbfgs.py, Algorithm 7.5 of Nocedal/Wright:
https://github.com/jax-ml/jax/blob/jax-v0.10.2/jax/_src/scipy/optimize/_lbfgs.py
Line search and termination: SciPy's lnsrlb/mainlb:
https://github.com/scipy/scipy/blob/v1.18.0/scipy/optimize/src/lbfgsb.c
"""

# Copyright 2020 The JAX Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

from pytensor.link.numba.dispatch._linesearch import _dcstep
from pytensor.link.numba.dispatch.basic import numba_njit


@numba_njit(fastmath=False)
def _direction(g, s_history, y_history, rho_history, gamma, count):
    size = rho_history.size
    q = -g.copy()
    alphas = np.zeros(size)
    for i in range(size - 1, size - count - 1, -1):
        alphas[i] = rho_history[i] * np.dot(s_history[i], q)
        q -= alphas[i] * y_history[i]
    q *= gamma
    for i in range(size - count, size):
        beta = rho_history[i] * np.dot(y_history[i], q)
        q += (alphas[i] - beta) * s_history[i]
    return q


@numba_njit(fastmath=False, inline="always")
def _line_search(evaluate, x, p, args, f0, g0, alpha, maxls):
    # SciPy lnsrlb's unbounded DCSRCH settings, including accepted warnings.
    c1, c2, xtol, stepmax = 1e-3, 0.9, 0.1, 1e10
    d0 = np.dot(g0, p)
    if not np.isfinite(d0) or d0 >= 0:
        return 0.0, f0, g0, False, 0
    bracketed, stage1 = False, True
    stx, fx, dx = 0.0, f0, d0
    sty, fy, dy = 0.0, f0, d0
    stmin, stmax = 0.0, 5 * alpha
    width, previous_width = stepmax, 2 * stepmax
    gtest = c1 * d0
    for evaluation in range(maxls):
        if not np.isfinite(alpha):
            return 0.0, f0, g0, False, evaluation
        f, g = evaluate(x + alpha * p, *args)
        d = np.dot(g, p)
        ftest = f0 + alpha * gtest
        if not np.isfinite(f) or not np.isfinite(d):
            return 0.0, f0, g0, False, evaluation + 1
        if (
            (f <= ftest and abs(d) <= -c2 * d0)
            or (
                bracketed
                and (alpha <= stmin or alpha >= stmax or stmax - stmin <= xtol * stmax)
            )
            or (alpha == stepmax and f <= ftest and d <= gtest)
            or (alpha == 0 and (f > ftest or d >= gtest))
        ):
            return alpha, f, g, True, evaluation + 1

        if stage1 and f <= ftest and d >= 0:
            stage1 = False
        if stage1 and f <= fx and f > ftest:
            stx, fx, dx, sty, fy, dy, alpha, bracketed = _dcstep(
                stx,
                fx - stx * gtest,
                dx - gtest,
                sty,
                fy - sty * gtest,
                dy - gtest,
                alpha,
                f - alpha * gtest,
                d - gtest,
                bracketed,
                stmin,
                stmax,
            )
            fx += stx * gtest
            fy += sty * gtest
            dx += gtest
            dy += gtest
        else:
            stx, fx, dx, sty, fy, dy, alpha, bracketed = _dcstep(
                stx, fx, dx, sty, fy, dy, alpha, f, d, bracketed, stmin, stmax
            )
        if bracketed:
            if abs(sty - stx) >= 0.66 * previous_width:
                alpha = stx + 0.5 * (sty - stx)
            previous_width, width = width, abs(sty - stx)
            stmin, stmax = min(stx, sty), max(stx, sty)
        else:
            stmin = alpha + 1.1 * (alpha - stx)
            stmax = alpha + 4 * (alpha - stx)
        alpha = min(max(alpha, 0.0), stepmax)
        if bracketed and (
            alpha <= stmin or alpha >= stmax or stmax - stmin <= xtol * stmax
        ):
            alpha = stx
    return 0.0, f0, g0, False, maxls


@numba_njit(fastmath=False, inline="always")
def _lbfgs(evaluate, x, args, m, ftol, gtol, maxiter, maxfun, maxls):
    if maxls <= 0:
        raise ValueError("maxls must be positive.")
    if x.size == 0 or m <= 0 or ftol < 0:
        return x, False
    f, g = evaluate(x, *args)
    if not np.isfinite(f) or not np.all(np.isfinite(g)):
        return x, False
    if np.max(np.abs(g)) <= gtol:
        return x, True

    s_history, y_history = np.zeros((m, x.size)), np.zeros((m, x.size))
    rho_history = np.zeros(m)
    gamma, count, iterations, nfev = 1.0, 0, 0, 1
    while True:
        p = _direction(g, s_history, y_history, rho_history, gamma, count)
        alpha = min(1.0 / np.linalg.norm(p), 1e10) if iterations == 0 else 1.0
        alpha, new_f, new_g, found, evaluations = _line_search(
            evaluate, x, p, args, f, g, alpha, maxls
        )
        nfev += evaluations
        if not found:
            # SciPy restarts failed searches with an empty L-BFGS history.
            if count == 0:
                return x, False
            count, gamma = 0, 1.0
            continue

        s, y = alpha * p, new_g - g
        old_f, old_g = f, g
        x, f, g = x + s, new_f, new_g
        iterations += 1
        # SciPy's driver checks these limits before resuming convergence checks.
        if iterations >= maxiter or nfev > maxfun:
            return x, False
        if np.max(np.abs(g)) <= gtol:
            return x, True
        if old_f - f <= ftol * max(abs(old_f), abs(f), 1.0):
            return x, True

        curvature = np.dot(y, s)
        if curvature <= np.finfo(np.float64).eps * (-np.dot(old_g, p) * alpha):
            continue
        s_history[:-1], y_history[:-1] = s_history[1:], y_history[1:]
        rho_history[:-1] = rho_history[1:]
        s_history[-1], y_history[-1] = s, y
        rho_history[-1] = 1.0 / curvature
        gamma = curvature / np.dot(y, y)
        count = min(count + 1, m)
