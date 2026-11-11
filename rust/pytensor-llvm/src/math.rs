//! Scalar functions generated code calls (`b.rust_math`).
//!
//! They are registered in the JIT as absolute symbols (`ptl_<name>_f64`, `_f32`) so
//! generated code can call them by name; cached object files stay position independent.

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
}
