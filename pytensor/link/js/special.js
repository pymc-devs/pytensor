// Copyright 2026 Benjamin F. Maier / PyMC Labs. Apache-2.0.
// Ported from pymc-labs/tymc b1c038979eb006b2d9ad1842a5b9ec2640688944:
// src/numerics/special_scalar.ts::{scalar_lgamma, scalar_digamma},
// including its fitted LGAMMA_STIRLING coefficients (D-118).
// Original kernels: Cephes gamma.c::lgam and psi.c::psi, Stephen L. Moshier.
// See special.LICENSE and THIRD_PARTY_NOTICES for attribution and license.
// The non-finite lgamma branch follows scipy.special.gammaln at -Infinity.
const LGAMMA_B = Float64Array.of(-1.378251525691208591e3, -3.88016315134637840924e4, -3.31612992738871184744e5, -1.16237097492762307383e6, -1.72173700820839662146e6, -8.53555664245765465627e5);

const LGAMMA_C = Float64Array.of(-3.51815701436523470549e2, -1.70642106651881159223e4, -2.20528590553854454839e5, -1.13933444367982507207e6, -2.53252307177582951285e6, -2.01889141433532773231e6);

const MAXLGM = 2.556348e305;

const LOG_SQRT_2PI = 0.91893853320467274178;

const LOG_PI = 1.1447298858494002;

const DIGAMMA_A = Float64Array.of(8.33333333333333333333e-2, -2.10927960927960927961e-2, 7.57575757575757575758e-3, -4.16666666666666666667e-3, 3.96825396825396825397e-3, -8.33333333333333333333e-3, 8.33333333333333333333e-2);

const EULER_GAMMA = 0.57721566490153286061;

const LGAMMA_STIRLING = Float64Array.of(0.08333333333333301, -0.0027777777771443445, 0.0007936505752359726, -0.0005952090988257412, 0.0008398280448766838, -0.0018461305249229, 0.004845911896726776, -0.008720909019511999, -0.0023065379165508283, -0.00035104529814113724, -4.124946989069674e-5);

function eval_polynomial(x, coefficients) {
    let result = coefficients[0];
    for (let k = 1; k < coefficients.length; k++) {
        result = result * x + coefficients[k];
    }
    return result;
}

function eval_monic_polynomial(x, coefficients) {
    let result = x + coefficients[0];
    for (let k = 1; k < coefficients.length; k++) {
        result = result * x + coefficients[k];
    }
    return result;
}

function scalar_lgamma(x) {
    if (Number.isNaN(x)) {
        return NaN;
    }
    if (!Number.isFinite(x)) {
        return x;
    }
    if (x < -34) {
        return lgamma_reflection(x);
    }
    // Cephes switches to its asymptotic series at 13 and walks the argument down by a recurrence
    // below that. The fitted correction is good from 4 up, which removes the loop for most of the
    // range a count likelihood visits — nine multiplies at x = 12, none now (D-118).
    if (x < 4) {
        return lgamma_small(x);
    }
    if (x > MAXLGM) {
        return Infinity;
    }
    const inverse = 1 / x;
    const square = inverse * inverse;
    let series = LGAMMA_STIRLING[10];
    for (let k = 9; k >= 0; k--) {
        series = series * square + LGAMMA_STIRLING[k];
    }
    return (x - 0.5) * Math.log(x) - x + LOG_SQRT_2PI + inverse * series;
}

function lgamma_reflection(x) {
    const q = -x;
    const w = scalar_lgamma(q);
    let p = Math.floor(q);
    if (p === q) {
        return Infinity;
    }
    let z = q - p;
    if (z > 0.5) {
        p += 1;
        z = p - q;
    }
    z = q * Math.sin(Math.PI * z);
    if (z === 0) {
        return Infinity;
    }
    return LOG_PI - Math.log(z) - w;
}

function lgamma_small(x) {
    let z = 1;
    let p = 0;
    let u = x;
    while (u >= 3) {
        p -= 1;
        u = x + p;
        z *= u;
    }
    while (u < 2) {
        if (u === 0) {
            return Infinity;
        }
        z /= u;
        p += 1;
        u = x + p;
    }
    if (z < 0) {
        z = -z;
    }
    if (u === 2) {
        return Math.log(z);
    }
    p -= 2;
    const shifted = x + p;
    return Math.log(z) + (shifted * eval_polynomial(shifted, LGAMMA_B)) / eval_monic_polynomial(shifted, LGAMMA_C);
}

function scalar_digamma(x) {
    if (Number.isNaN(x)) {
        return NaN;
    }
    if (x === 0) {
        return Object.is(x, -0) ? Infinity : -Infinity;
    }
    let argument = x;
    let reflection_term = 0;
    if (x < 0) {
        const p = Math.floor(x);
        if (p === x) {
            return NaN;
        }
        let fractional = x - p;
        if (fractional !== 0.5) {
            if (fractional > 0.5) {
                fractional = x - (p + 1);
            }
            reflection_term = Math.PI / Math.tan(Math.PI * fractional);
        }
        argument = 1 - x;
    }
    let result;
    if (argument <= 10 && argument === Math.floor(argument)) {
        result = -EULER_GAMMA;
        for (let i = 1; i < argument; i++) {
            result += 1 / i;
        }
    }
    else {
        let s = argument;
        let w = 0;
        while (s < 10) {
            w += 1 / s;
            s += 1;
        }
        let series = 0;
        if (s < 1e17) {
            const z = 1 / (s * s);
            series = z * eval_polynomial(z, DIGAMMA_A);
        }
        result = Math.log(s) - 0.5 / s - series - w;
    }
    return result - reflection_term;
}
