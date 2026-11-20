//! Random numbers for generated code and kernels: the runtime functions `rng_*` (in the
//! table `ctx->rt`, linked into generated code as `ptl_rt_rng_*`).
//!
//! All draw from the call context's random state (`PtCtx::rng`, Xoshiro256++): one
//! state per session, advanced by every draw, so a seeded state gives
//! reproducible draws. Distributions come from `rand_distr`. Parameters are valid
//! (generated code checks them first); invalid ones give NaN (or -1 for counts).

use crate::runtime::PtCtx;
use rand_core::Rng;
use rand_distr::{Beta, Binomial, Distribution, Gamma, Poisson, StandardNormal, StandardUniform};

// SAFETY contract of all functions here: `ctx` is the live context of the call (the
// runtime functions' contract), used by nobody else meanwhile; pointers point to `n`
// writable elements.

/// SAFETY: `ctx` is live and exclusively ours while the reference is used.
unsafe fn rng<'a>(ctx: *mut PtCtx) -> &'a mut rand_xoshiro::Xoshiro256PlusPlus {
    // SAFETY: the function's contract.
    unsafe { &mut (*ctx).rng }
}

pub unsafe extern "C" fn rng_u64(ctx: *mut PtCtx) -> u64 {
    // SAFETY: see the contract above.
    unsafe { rng(ctx) }.next_u64()
}

/// Uniform on [0, 1).
pub unsafe extern "C" fn rng_uniform(ctx: *mut PtCtx) -> f64 {
    // SAFETY: see the contract above.
    StandardUniform.sample(unsafe { rng(ctx) })
}

pub unsafe extern "C" fn rng_normal(ctx: *mut PtCtx) -> f64 {
    // SAFETY: see the contract above.
    StandardNormal.sample(unsafe { rng(ctx) })
}

pub unsafe extern "C" fn rng_fill_uniform(ctx: *mut PtCtx, out: *mut f64, n: i64) {
    // SAFETY: see the contract above.
    let r = unsafe { rng(ctx) };
    for i in 0..n.max(0) as usize {
        // SAFETY: `out` has n elements (contract).
        unsafe { *out.add(i) = StandardUniform.sample(r) };
    }
}

pub unsafe extern "C" fn rng_fill_normal(ctx: *mut PtCtx, out: *mut f64, n: i64) {
    // SAFETY: see the contract above.
    let r = unsafe { rng(ctx) };
    for i in 0..n.max(0) as usize {
        // SAFETY: `out` has n elements (contract).
        unsafe { *out.add(i) = StandardNormal.sample(r) };
    }
}

/// Gamma with the given shape and scale 1.
pub unsafe extern "C" fn rng_gamma(ctx: *mut PtCtx, shape: f64) -> f64 {
    Gamma::new(shape, 1.0).map_or(f64::NAN, |d| d.sample(unsafe { rng(ctx) }))
}

pub unsafe extern "C" fn rng_beta(ctx: *mut PtCtx, a: f64, b: f64) -> f64 {
    Beta::new(a, b).map_or(f64::NAN, |d| d.sample(unsafe { rng(ctx) }))
}

pub unsafe extern "C" fn rng_poisson(ctx: *mut PtCtx, lam: f64) -> i64 {
    if lam == 0.0 {
        return 0;
    }
    Poisson::new(lam).map_or(-1, |d| {
        let v: f64 = d.sample(unsafe { rng(ctx) });
        v as i64
    })
}

pub unsafe extern "C" fn rng_binomial(ctx: *mut PtCtx, n: i64, p: f64) -> i64 {
    if n < 0 {
        return -1;
    }
    Binomial::new(n as u64, p).map_or(-1, |d| d.sample(unsafe { rng(ctx) }) as i64)
}
