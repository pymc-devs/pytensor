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


@numba_njit(fastmath=False)
def _precedes(a, b):
    # NumPy's sort order, which puts NaN after every number.
    return a < b or (np.isnan(b) and not np.isnan(a))


@numba_njit(fastmath=False)
def _reinsert_worst(order, fsim):
    # Only the worst vertex changed, so move it to where a stable sort of the
    # values would put it: after every vertex whose value does not exceed it.
    worst = order[-1]
    position = order.size - 1
    while position > 0 and _precedes(fsim[worst], fsim[order[position - 1]]):
        order[position] = order[position - 1]
        position -= 1
    order[position] = worst


@numba_njit(fastmath=False)
def _converged(sim, fsim, order, xatol, fatol):
    # SciPy's max(|sim[1:] - sim[0]|) <= xatol and max(|fsim[0] - fsim[1:]|) <=
    # fatol, without temporaries. A NaN difference fails, as NaN max does.
    best = order[0]
    for k in range(1, order.size):
        row = order[k]
        for i in range(sim.shape[1]):
            if not abs(sim[row, i] - sim[best, i]) <= xatol:
                return False
    for k in range(1, order.size):
        if not abs(fsim[best] - fsim[order[k]]) <= fatol:
            return False
    return True


@numba_njit(fastmath=False)
def _centroid(sim, order, xbar):
    # Sum the n best vertices in sorted order, as SciPy's add.reduce does.
    n = xbar.size
    for i in range(n):
        total = sim[order[0], i]
        for k in range(1, n):
            total += sim[order[k], i]
        xbar[i] = total / n


@numba_njit(fastmath=False, inline="always")
def _nelder_mead(evaluate, x0, args, maxiter, maxfev, xatol, fatol, adaptive):
    # maxiter and maxfev are NaN when unset, and get SciPy's defaults.
    n = x0.size
    if np.isnan(maxiter) and np.isnan(maxfev):
        maxiter, maxfev = 200.0 * n, 200.0 * n
    elif np.isnan(maxiter):
        maxiter = np.inf
    elif np.isnan(maxfev):
        maxfev = np.inf

    if adaptive:
        rho, chi, psi, sigma = 1.0, 1 + 2 / n, 0.75 - 1 / (2 * n), 1 - 1 / n
    else:
        rho, chi, psi, sigma = 1.0, 2.0, 0.5, 0.5
    nonzdelt, zdelt = 0.05, 0.00025

    sim = np.empty((n + 1, n))
    sim[0] = x0
    for k in range(n):
        sim[k + 1] = x0
        if x0[k] != 0:
            sim[k + 1, k] = (1 + nonzdelt) * x0[k]
        else:
            sim[k + 1, k] = zdelt

    # SciPy stops evaluating once maxfev calls are spent, even mid-step; each
    # "if nfev >= maxfev: break" below is where its _MaxFuncCallError fires.
    fsim = np.full(n + 1, np.inf)
    nfev = 0
    for k in range(n + 1):
        if nfev >= maxfev:
            break
        nfev += 1
        fsim[k] = evaluate(sim[k], *args)

    # Vertices stay in their rows; order lists them from best to worst, the
    # order SciPy's sorted copy of the simplex has.
    order = np.argsort(fsim, kind="mergesort")
    xbar, xr, trial = np.empty(n), np.empty(n), np.empty(n)
    iterations = 1
    while nfev < maxfev and iterations < maxiter:
        if _converged(sim, fsim, order, xatol, fatol):
            break

        best, second, worst = order[0], order[-2], order[-1]
        replaced = shrunk = False
        for _ in range(1):
            _centroid(sim, order, xbar)
            for i in range(n):
                xr[i] = (1 + rho) * xbar[i] - rho * sim[worst, i]
            if nfev >= maxfev:
                break
            nfev += 1
            fxr = evaluate(xr, *args)

            if fxr < fsim[best]:
                for i in range(n):
                    trial[i] = (1 + rho * chi) * xbar[i] - rho * chi * sim[worst, i]
                if nfev >= maxfev:
                    break
                nfev += 1
                fxe = evaluate(trial, *args)
                if fxe < fxr:
                    sim[worst], fsim[worst] = trial, fxe
                else:
                    sim[worst], fsim[worst] = xr, fxr
                replaced = True
            elif fxr < fsim[second]:
                sim[worst], fsim[worst] = xr, fxr
                replaced = True
            else:
                if fxr < fsim[worst]:
                    for i in range(n):
                        trial[i] = (1 + psi * rho) * xbar[i] - psi * rho * sim[worst, i]
                    if nfev >= maxfev:
                        break
                    nfev += 1
                    fxc = evaluate(trial, *args)
                    replaced = fxc <= fxr
                    if replaced:
                        sim[worst], fsim[worst] = trial, fxc
                else:
                    for i in range(n):
                        trial[i] = (1 - psi) * xbar[i] + psi * sim[worst, i]
                    if nfev >= maxfev:
                        break
                    nfev += 1
                    fxcc = evaluate(trial, *args)
                    replaced = fxcc < fsim[worst]
                    if replaced:
                        sim[worst], fsim[worst] = trial, fxcc

                if not replaced:
                    shrunk = True
                    exhausted = False
                    for k in range(1, n + 1):
                        row = order[k]
                        for i in range(n):
                            sim[row, i] = sim[best, i] + sigma * (
                                sim[row, i] - sim[best, i]
                            )
                        if nfev >= maxfev:
                            exhausted = True
                            break
                        nfev += 1
                        fsim[row] = evaluate(sim[row], *args)
                    if exhausted:
                        break
            iterations += 1

        if shrunk:
            order = order[np.argsort(fsim[order], kind="mergesort")]
        elif replaced:
            _reinsert_worst(order, fsim)

    success = nfev < maxfev and iterations < maxiter
    return sim[order[0]].copy(), success
