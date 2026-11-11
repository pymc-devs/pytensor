//! Scalar functions generated code calls (`b.rust_math`).
//!
//! They are registered in the JIT as absolute symbols (`ptl_<name>_f64`, `_f32`) so
//! generated code can call them by name; cached object files stay position independent.
// (the erfc coefficients are written with every digit of their source)
#![allow(clippy::excessive_precision)]

fn digamma(mut x: f64) -> f64 {
    use std::f64::consts::PI;
    if x.is_nan() || x == f64::NEG_INFINITY {
        return f64::NAN;
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
    }
    if x < 0.0 {
        if x == x.floor() {
            return f64::NAN;
        }
        return digamma(1.0 - x) - PI / (PI * x).tan();
    }
    let mut result = 0.0;
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let f = 1.0 / (x * x);
    let t = f
        * (-1.0 / 12.0
            + f * (1.0 / 120.0 + f * (-1.0 / 252.0 + f * (1.0 / 240.0 + f * (-1.0 / 132.0)))));
    result + x.ln() - 0.5 / x + t
}

// Coefficients of erfc for 1.25 <= x < 1/0.35 (RA, SA) and 1/0.35 <= x < 28 (RB, SB),
// from the libm crate (origin: FreeBSD /usr/src/lib/msun/src/s_erf.c):
//
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
const RA0: f64 = -9.86494403484714822705e-03;
const RA1: f64 = -6.93858572707181764372e-01;
const RA2: f64 = -1.05586262253232909814e+01;
const RA3: f64 = -6.23753324503260060396e+01;
const RA4: f64 = -1.62396669462573470355e+02;
const RA5: f64 = -1.84605092906711035994e+02;
const RA6: f64 = -8.12874355063065934246e+01;
const RA7: f64 = -9.81432934416914548592e+00;
const SA1: f64 = 1.96512716674392571292e+01;
const SA2: f64 = 1.37657754143519042600e+02;
const SA3: f64 = 4.34565877475229228821e+02;
const SA4: f64 = 6.45387271733267880336e+02;
const SA5: f64 = 4.29008140027567833386e+02;
const SA6: f64 = 1.08635005541779435134e+02;
const SA7: f64 = 6.57024977031928170135e+00;
const SA8: f64 = -6.04244152148580987438e-02;
const RB0: f64 = -9.86494292470009928597e-03;
const RB1: f64 = -7.99283237680523006574e-01;
const RB2: f64 = -1.77579549177547519889e+01;
const RB3: f64 = -1.60636384855821916062e+02;
const RB4: f64 = -6.37566443368389627722e+02;
const RB5: f64 = -1.02509513161107724954e+03;
const RB6: f64 = -4.83519191608651397019e+02;
const SB1: f64 = 3.03380607434824582924e+01;
const SB2: f64 = 3.25792512996573918826e+02;
const SB3: f64 = 1.53672958608443695994e+03;
const SB4: f64 = 3.19985821950859553908e+03;
const SB5: f64 = 2.55305040643316442583e+03;
const SB6: f64 = 4.74528541206955367215e+02;
const SB7: f64 = -2.24409524465858183362e+01;

/// The scaled complementary error function exp(x²) erfc(x) (scipy.special.erfcx).
///
/// * 1.25 <= x < 12: erfc(x) is exp(-x²) exp(R/S - 0.5625) / x there, with R/S the
///   rational function of 1/x² of `libm`'s erfc, so erfcx(x) is exp(R/S - 0.5625) / x:
///   one exp, nothing that overflows.
/// * x >= 12: the asymptotic series 1/(x√π) Σ (-1)^n (2n-1)!!/(2x²)^n, whose 16th
///   term is below 1e-22 there.
/// * |x| < 1.25: exp(x²) erfc(x), erfc a polynomial there.
/// * x <= -1.25: 2 exp(x²) - erfcx(-x), exp(x²) corrected for the rounding error of x².
pub fn erfcx(x: f64) -> f64 {
    if x >= 1.25 {
        if x >= 12.0 {
            let r = 1.0 / (2.0 * x * x);
            let (mut term, mut sum) = (1.0, 1.0);
            for n in 1..16 {
                term *= -f64::from(2 * n - 1) * r;
                sum += term;
            }
            return sum * (0.5 * std::f64::consts::FRAC_2_SQRT_PI) / x;
        }
        let s = 1.0 / (x * x);
        let (r, big_s) = if x < 1.0 / 0.35 {
            (
                RA0 + s
                    * (RA1 + s * (RA2 + s * (RA3 + s * (RA4 + s * (RA5 + s * (RA6 + s * RA7)))))),
                1.0 + s
                    * (SA1
                        + s * (SA2
                            + s * (SA3 + s * (SA4 + s * (SA5 + s * (SA6 + s * (SA7 + s * SA8))))))),
            )
        } else {
            (
                RB0 + s * (RB1 + s * (RB2 + s * (RB3 + s * (RB4 + s * (RB5 + s * RB6))))),
                1.0 + s
                    * (SB1 + s * (SB2 + s * (SB3 + s * (SB4 + s * (SB5 + s * (SB6 + s * SB7)))))),
            )
        };
        return (r / big_s - 0.5625).exp() / x;
    }
    if x > -1.25 {
        return (x * x).exp() * libm::erfc(x);
    }
    // (NaN went to the last branch: NaN comparisons are false)
    if x.is_nan() {
        return x;
    }
    let x2 = x * x;
    // x² = x2 + low exactly (Dekker's product: x split into halves of 26 bits; an
    // infinite x2 has nothing to correct)
    let low = if x2.is_finite() {
        let c = 134217729.0 * x;
        let hi = c - (c - x);
        let lo = x - hi;
        ((hi * hi - x2) + 2.0 * hi * lo) + lo * lo
    } else {
        0.0
    };
    2.0 * x2.exp() * (1.0 + low) - erfcx(-x)
}

macro_rules! scalar_fns {
    ($($name:ident, $name32:ident => $f:expr;)*) => {
        $(
            #[unsafe(no_mangle)]
            pub extern "C" fn $name(x: f64) -> f64 { ($f)(x) }
            #[unsafe(no_mangle)]
            pub extern "C" fn $name32(x: f32) -> f32 { ($f)(x as f64) as f32 }
        )*
        pub fn scalar_symbols() -> Vec<(&'static str, usize)> {
            vec![$(
                (stringify!($name), $name as extern "C" fn(f64) -> f64 as usize),
                (stringify!($name32), $name32 as extern "C" fn(f32) -> f32 as usize),
            )*]
        }
    };
}

scalar_fns! {
    ptl_erf_f64, ptl_erf_f32 => libm::erf;
    ptl_erfc_f64, ptl_erfc_f32 => libm::erfc;
    ptl_gammaln_f64, ptl_gammaln_f32 => libm::lgamma;
    ptl_gamma_f64, ptl_gamma_f32 => libm::tgamma;
    ptl_psi_f64, ptl_psi_f32 => digamma;
    ptl_erfcx_f64, ptl_erfcx_f32 => erfcx;
}
