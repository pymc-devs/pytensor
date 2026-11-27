"""The one-pass kernel for gammainc, gammaincc and the k gradient of gammainc."""

import numpy as np

from pytensor.scalar.math import _LANCZOS_DEN, _LANCZOS_G, _LANCZOS_NUM


def make_gammainc_kernel(lgamma, digamma, gammainc, gammaincc, njit):
    """Return ``kernel(k, x) -> (P, Q, dP/dk)`` for float64 scalars.

    One pass of the loops in `_gammainc_grad_loops` also gives the value: P is
    x^k e^-x / Gamma(k + 1) times the series sum and Q is x^k e^-x / Gamma(k)
    times the continued fraction. The prefactor is Cephes' ``igam_fac``, as in
    scipy's gammainc. ``gammainc`` and ``gammaincc`` are scipy's: they give the
    values where the series has P > 0.9, and where a loop does not converge,
    with dP/dk nan there as in `_gammainc_grad_loops`. Other edge cases
    follow `_gammainc_grad_loops` for dP/dk and scipy for the values. ``lgamma``
    and ``digamma`` are only called on positive arguments.
    """
    eps = float(np.finfo(np.float64).eps)
    machep = eps / 2
    maxlog = 7.09782712893383996843e2
    g = _LANCZOS_G
    num, den = _LANCZOS_NUM, _LANCZOS_DEN

    @njit
    def log1pmx(z):
        if abs(z) < 0.5:
            zfac, res = z, 0.0
            for n in range(2, 500):
                zfac *= -z
                term = zfac / n
                res += term
                if abs(term) < machep * abs(res):
                    break
            return res
        return np.log1p(z) - z

    @njit
    def lanczos_sum_expg_scaled(z):
        if abs(z) > 1.0:
            y = 1.0 / z
            n, d = num[12], den[12]
            for i in range(11, -1, -1):
                n = n * y + num[i]
                d = d * y + den[i]
        else:
            n, d = num[0], den[0]
            for i in range(1, 13):
                n = n * z + num[i]
                d = d * z + den[i]
        return n / d

    @njit
    def igam_fac(k, x):
        # x^k e^-x / Gamma(k), with Lanczos near k = x where the logs cancel
        if abs(k - x) > 0.4 * abs(k):
            ax = k * np.log(x) - x - lgamma(k)
            return 0.0 if ax < -maxlog else np.exp(ax)
        fac = k + g - 0.5
        res = np.sqrt(fac / np.e) / lanczos_sum_expg_scaled(k)
        if k < 200.0 and x < 200.0:
            return res * np.exp(k - x) * (x / fac) ** k
        return res * np.exp(k * log1pmx((x - k - g + 0.5) / fac) + x * (0.5 - g) / fac)

    @njit
    def series(k, x):
        s, ds, t, dt, kn = 1.0, 0.0, 1.0, 0.0, k
        for _ in range(100_000):
            kn += 1.0
            r = x / kn
            dt = r * (dt - t / kn)
            t *= r
            s += t
            ds += dt
            if t <= eps * s and abs(dt) <= eps * abs(ds):
                pre = igam_fac(k, x) / k if k > 0.0 else np.exp(-x)
                return pre * s, pre * (s * (np.log(x) - digamma(k + 1.0)) + ds)
        return np.nan, np.nan

    @njit
    def continued_fraction(k, x):
        fac = igam_fac(k, x)
        if fac == 0.0:
            return 0.0, 0.0
        pm2, pm1, qm2, qm1 = 1.0, x + 1.0, x, (x + 2.0 - k) * x
        dpm2, dpm1, dqm2, dqm1 = 0.0, 0.0, 0.0, -x
        f = pm1 / qm1
        df = f * x / qm1
        n = 0.0
        for _ in range(100_000):
            n += 1.0
            z = x + 2.0 * n + 2.0 - k
            yn = (n + 1.0 - k) * n
            p = pm1 * z - pm2 * yn
            q = qm1 * z - qm2 * yn
            dp = dpm1 * z - pm1 - dpm2 * yn + pm2 * n
            dq = dqm1 * z - qm1 - dqm2 * yn + qm2 * n
            if q != 0.0:
                f_new = p / q
                df_new = (dp - f_new * dq) / q
                done = abs(f_new - f) <= eps * abs(f_new) and abs(
                    df_new - df
                ) <= eps * abs(df_new)
                f, df = f_new, df_new
                if done:
                    return fac * f, fac * (f * (np.log(x) - digamma(k)) + df)
            s = eps if abs(p) > 1.0 / eps else 1.0
            pm2, pm1, qm2, qm1 = pm1 * s, p * s, qm1 * s, q * s
            dpm2, dpm1, dqm2, dqm1 = dpm1 * s, dp * s, dqm1 * s, dq * s
        return np.nan, np.nan

    @njit
    def kernel(k, x):
        if not (k >= 0.0 and x >= 0.0):
            return np.nan, np.nan, np.nan
        if x == 0.0:
            p = np.nan if k == 0.0 else 0.0
            return p, 1.0 - p, 0.0
        if np.isinf(x):
            p = np.nan if np.isinf(k) else 1.0
            return p, 1.0 - p, 0.0
        if np.isinf(k):
            return 0.0, 1.0, np.nan
        if x <= 1.0 or x <= k:
            p, dp = series(k, x)
            if k == 0.0:
                return 1.0, 0.0, dp
            if np.isnan(p):
                return gammainc(k, x), gammaincc(k, x), np.nan
            if p > 0.9:
                # 1 - P cancels. The continued fraction always has P > 1/2.
                q = gammaincc(k, x)
                return 1.0 - q, q, dp
            return p, 1.0 - p, dp
        if k == 0.0:
            return 1.0, 0.0, np.nan
        q, dq = continued_fraction(k, x)
        if np.isnan(q):
            return gammainc(k, x), gammaincc(k, x), np.nan
        return 1.0 - q, q, -dq

    return kernel
