//! Parallel loops for generated code and kernels: the runtime function `parallel_for`
//! (`ptl_rt_parallel_for`), on the runtime's rayon pool.
//!
//! `parallel_for(ctx, n, grain, body, captures)` runs `body(chunk_ctx, captures, start, end)` for
//! the chunks `[k * grain, min(n, (k + 1) * grain))` of `0..n`. The chunks only depend
//! on `n` and `grain`, and each gets its own context with a random state derived from
//! (one draw of the caller's state, chunk index): results, random draws included, are
//! the same whatever the number of threads, and whether the session allows
//! parallelism at all (`PtCtx::parallel`; chunks then run one after the other). A single
//! chunk runs directly on the caller's context.

use crate::runtime::{PtCtx, runtime};
use rand_core::Rng;
use rayon::prelude::*;
use std::cell::RefCell;

/// `i32 body(ctx, captures, start, end)`: 0 on success, else the error is set on `ctx`.
pub type BodyFn = unsafe extern "C" fn(*mut PtCtx, *const u8, i64, i64) -> i32;

thread_local! {
    /// contexts for chunks run on this thread (their scratch memory is reused)
    static CHILDREN: RefCell<Vec<PtCtx>> = const { RefCell::new(Vec::new()) };
}

/// SplitMix64: a seed per chunk from the region's draw.
fn mix(base: u64, k: u64) -> u64 {
    let mut z = base.wrapping_add(k.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Run chunk `k` on a context of this thread.
///
/// SAFETY: `body` has the `BodyFn` contract and `captures` is what it expects.
unsafe fn run_chunk(
    body: BodyFn,
    captures: usize,
    n: i64,
    grain: i64,
    k: i64,
    seed: u64,
    parallel: bool,
) -> Option<(i32, String)> {
    let mut child = CHILDREN.with(|c| c.borrow_mut().pop()).unwrap_or_default();
    child.seed(seed);
    child.parallel = parallel;
    child.error = None;
    let start = k * grain;
    let end = n.min(start + grain);
    let mark = child.scratch_mark();
    // SAFETY: the function's contract; `child` is ours for the call.
    let rc = unsafe { body(&mut child, captures as *const u8, start, end) };
    child.scratch_release(mark);
    let r = if rc != 0 {
        Some((
            rc,
            child
                .error
                .take()
                .map(|(_, m)| m)
                .unwrap_or_else(|| format!("error {rc}")),
        ))
    } else {
        None
    };
    CHILDREN.with(|c| c.borrow_mut().push(child));
    r
}

/// # Safety
/// `ctx` is the caller's live context; `body` has the `BodyFn` contract and `captures` is
/// what it expects (shared by all chunks: read only, or written at chunk-disjoint
/// places).
pub unsafe extern "C" fn rt_parallel_for(
    ctx: *mut PtCtx,
    n: i64,
    grain: i64,
    body: BodyFn,
    captures: *const u8,
) -> i32 {
    let grain = grain.max(1);
    let n = n.max(0);
    if n <= grain {
        // SAFETY: the function's contract.
        return unsafe { body(ctx, captures, 0, n) };
    }
    // SAFETY: the caller's context, ours for the call.
    let parent = unsafe { &mut *ctx };
    let chunks = (n + grain - 1) / grain;
    let base = parent.rng.next_u64();
    let parallel = parent.parallel && runtime().threads() > 1;
    let captures = captures as usize;
    // SAFETY (both): `body` and `captures` as promised by our caller.
    let errors: Vec<Option<(i32, String)>> = if parallel {
        runtime().pool().install(|| {
            (0..chunks)
                .into_par_iter()
                .map(|k| unsafe {
                    run_chunk(body, captures, n, grain, k, mix(base, k as u64), true)
                })
                .collect()
        })
    } else {
        (0..chunks)
            .map(|k| unsafe {
                run_chunk(
                    body,
                    captures,
                    n,
                    grain,
                    k,
                    mix(base, k as u64),
                    parent.parallel,
                )
            })
            .collect()
    };
    // the first failing chunk's error (by index: the same whatever ran first)
    match errors.into_iter().flatten().next() {
        Some((rc, msg)) => {
            parent.error = Some((rc, msg));
            rc
        }
        None => 0,
    }
}
