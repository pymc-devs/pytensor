//! The runtime, and the context of a call.
//!
//! * The **runtime** is process-wide: the table of runtime functions (`PtRuntime`, what
//!   kernels reach through `ctx->rt` and generated code links against as `ptl_rt_<name>`)
//!   and the state behind it (the JIT, the thread pool).
//! * A **context** (`PtCtx`) belongs to a session: errors, scratch memory and the random
//!   state of everything one call runs.
//!
//! ```c
//! typedef struct {
//!     uint32_t version;
//!     void *(*alloc)(PtCtx *, size_t bytes, size_t align, PtBuffer **owner);
//!     void  (*buffer_incref)(PtCtx *, PtBuffer *);
//!     void  (*buffer_decref)(PtCtx *, PtBuffer *);
//!     void *(*scratch)(PtCtx *, size_t bytes);    // 64-byte aligned, valid until the task ends
//!     void  (*set_error)(PtCtx *, int32_t code, const char *msg);
//!     int64_t (*scratch_mark)(PtCtx *);
//!     void  (*scratch_release)(PtCtx *, int64_t mark);
//!     ...                                          // see `runtime_functions!`
//! } PtRuntime;
//! ```
//!
//! Scratch: `rt->scratch(ctx, n)` hands out temporary memory (e.g. faer's work space)
//! that is valid until the task ends. It comes from an arena owned by the context: the
//! arena keeps its largest size, so after the first call a kernel's scratch requests
//! don't allocate. Several requests per kernel are fine; they never move earlier ones.

use crate::abi::{ABI_VERSION, PtBuffer};
use std::alloc::{Layout, alloc, dealloc};
use std::ffi::{CStr, c_char};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// Defines the runtime function table (`PtRuntime`, what kernels reach through
/// `ctx->rt`), its static instance, and the JIT symbol names (`ptl_rt_<name>`) that
/// generated code links against: one list, so the two can't drift. Append only: numba
/// fall-backs read `alloc` at a fixed offset.
macro_rules! runtime_functions {
    ($($(#[$doc:meta])* $name:ident: fn($($arg:ty),*) $(-> $ret:ty)? = $f:path;)*) => {
        #[repr(C)]
        pub struct PtRuntime {
            pub version: u32,
            $($(#[$doc])* pub $name: unsafe extern "C" fn($($arg),*) $(-> $ret)?,)*
        }

        const FUNCTIONS: PtRuntime = PtRuntime { version: ABI_VERSION, $($name: $f,)* };

        impl PtRuntime {
            /// `(ptl_rt_<name>, address)` for every function, for the JIT.
            pub fn symbols(&self) -> Vec<(&'static str, usize)> {
                vec![$((concat!("ptl_rt_", stringify!($name)), self.$name as *const () as usize),)*]
            }
        }
    };
}

runtime_functions! {
    /// A new buffer of `bytes` (aligned to `align`); `*owner` gets its one reference.
    /// NULL with the error set if it can't be allocated.
    alloc: fn(*mut PtCtx, usize, usize, *mut *mut PtBuffer) -> *mut u8 = rt_alloc;
    buffer_incref: fn(*mut PtCtx, *mut PtBuffer) = rt_incref;
    buffer_decref: fn(*mut PtCtx, *mut PtBuffer) = rt_decref;
    /// Scratch memory (64-byte aligned) until the task ends, or until a
    /// `scratch_release` of an earlier mark.
    scratch: fn(*mut PtCtx, usize) -> *mut u8 = rt_scratch;
    set_error: fn(*mut PtCtx, i32, *const c_char) = rt_set_error;
    /// The current scratch position; `scratch_release` frees what was taken since.
    scratch_mark: fn(*mut PtCtx) -> i64 = rt_scratch_mark;
    scratch_release: fn(*mut PtCtx, i64) = rt_scratch_release;
    /// Random numbers from the context's random state (`crate::random`).
    rng_u64: fn(*mut PtCtx) -> u64 = crate::random::rng_u64;
    rng_uniform: fn(*mut PtCtx) -> f64 = crate::random::rng_uniform;
    rng_normal: fn(*mut PtCtx) -> f64 = crate::random::rng_normal;
    rng_fill_uniform: fn(*mut PtCtx, *mut f64, i64) = crate::random::rng_fill_uniform;
    rng_fill_normal: fn(*mut PtCtx, *mut f64, i64) = crate::random::rng_fill_normal;
    rng_gamma: fn(*mut PtCtx, f64) -> f64 = crate::random::rng_gamma;
    rng_beta: fn(*mut PtCtx, f64, f64) -> f64 = crate::random::rng_beta;
    rng_poisson: fn(*mut PtCtx, f64) -> i64 = crate::random::rng_poisson;
    rng_binomial: fn(*mut PtCtx, i64, f64) -> i64 = crate::random::rng_binomial;
    /// `parallel_for(ctx, n, grain, body, captures)`: chunks of `0..n` on the runtime's
    /// thread pool (`crate::parallel`).
    parallel_for: fn(*mut PtCtx, i64, i64, crate::parallel::BodyFn, *const u8) -> i32 = crate::parallel::rt_parallel_for;
}

/// The process-wide runtime: the function table every context points to, and the
/// process-wide state behind it (the JIT; later e.g. device contexts). Starts with the
/// table, so `ctx->rt` is both.
#[repr(C)]
pub struct Runtime {
    pub functions: PtRuntime,
    jit: OnceLock<Mutex<crate::jit::Jit>>,
    /// the thread pool for parallel loops, and its size (0: not chosen yet)
    pool: OnceLock<rayon::ThreadPool>,
    n_threads: AtomicUsize,
}

static RUNTIME: Runtime = Runtime {
    functions: FUNCTIONS,
    jit: OnceLock::new(),
    pool: OnceLock::new(),
    n_threads: AtomicUsize::new(0),
};

pub fn runtime() -> &'static Runtime {
    &RUNTIME
}

impl Runtime {
    /// The thread pool parallel loops run on. Its size is fixed when first used:
    /// `set_threads`, else `PYTENSOR_LLVM_THREADS`, else rayon's default (one per CPU).
    pub fn pool(&self) -> &rayon::ThreadPool {
        self.pool.get_or_init(|| {
            let want = self.n_threads.load(Ordering::Relaxed);
            let env = std::env::var("PYTENSOR_LLVM_THREADS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0);
            let n = if want > 0 { want } else { env };
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .thread_name(|i| format!("pytensor-llvm-{i}"))
                .build()
                .expect("failed to create the thread pool");
            self.n_threads
                .store(pool.current_num_threads(), Ordering::Relaxed);
            pool
        })
    }

    /// The number of threads parallel loops use.
    pub fn threads(&self) -> usize {
        self.pool().current_num_threads()
    }

    /// Choose the pool's size; only before it is first used.
    pub fn set_threads(&self, n: usize) -> Result<(), String> {
        if let Some(p) = self.pool.get() {
            return if p.current_num_threads() == n {
                Ok(())
            } else {
                Err(format!(
                    "the thread pool already has {} threads",
                    p.current_num_threads()
                ))
            };
        }
        self.n_threads.store(n, Ordering::Relaxed);
        Ok(())
    }

    /// The JIT all generated code is linked into.
    pub fn jit(&self) -> &Mutex<crate::jit::Jit> {
        self.jit
            .get_or_init(|| Mutex::new(crate::jit::Jit::new().expect("failed to create LLJIT")))
    }
}

// ------------------------------------------------------------------ the call context

/// The context of a call: per session (one per thread or chain, owned by the
/// caller through its session), shared by everything the call runs, subprograms
/// included. Kernels see only `rt`.
#[repr(C)]
pub struct PtCtx {
    pub rt: *const PtRuntime,
    // runtime-private state below
    pub error: Option<(i32, String)>,
    scratch: Scratch,
    /// positions handed out by `scratch_mark` (tokens are indices)
    marks: Vec<ScratchMark>,
    /// the session's random state: everything the call runs draws from it, in
    /// order (`seed` makes the draws reproducible)
    pub rng: rand_xoshiro::Xoshiro256PlusPlus,
    /// may parallel loops of this session use the thread pool? (Results are
    /// the same either way.)
    pub parallel: bool,
}

/// Bump arena behind `scratch`.
///
/// One persistent block. A request that doesn't fit gets its own overflow block (so
/// earlier pointers stay valid); when everything is released, the persistent block is
/// regrown to the peak demand, so the next use fits without allocating.
struct Scratch {
    block: *mut u8,
    cap: usize,
    off: usize,
    overflow: Vec<(*mut u8, usize)>,
    overflow_bytes: usize,
    peak: usize,
}

#[derive(Clone, Copy)]
struct ScratchMark {
    off: usize,
    n_overflow: usize,
    overflow_bytes: usize,
}

const SCRATCH_ALIGN: usize = 64;

impl Scratch {
    fn new() -> Self {
        Scratch {
            block: std::ptr::null_mut(),
            cap: 0,
            off: 0,
            overflow: Vec::new(),
            overflow_bytes: 0,
            peak: 0,
        }
    }

    fn get(&mut self, bytes: usize) -> *mut u8 {
        let bytes = bytes.max(1).next_multiple_of(SCRATCH_ALIGN);
        let p = if self.off + bytes <= self.cap {
            // SAFETY: off + bytes <= cap, inside the block.
            let p = unsafe { self.block.add(self.off) };
            self.off += bytes;
            p
        } else {
            // SAFETY: bytes >= 64 (non-zero layout).
            let p = unsafe {
                alloc(Layout::from_size_align(bytes, SCRATCH_ALIGN).expect("bad layout"))
            };
            assert!(
                !p.is_null(),
                "out of memory allocating {bytes} bytes of scratch"
            );
            self.overflow.push((p, bytes));
            self.overflow_bytes += bytes;
            p
        };
        self.peak = self.peak.max(self.off + self.overflow_bytes);
        p
    }

    fn mark(&self) -> ScratchMark {
        ScratchMark {
            off: self.off,
            n_overflow: self.overflow.len(),
            overflow_bytes: self.overflow_bytes,
        }
    }

    fn release(&mut self, m: ScratchMark) {
        for (p, n) in self.overflow.drain(m.n_overflow..) {
            // SAFETY: allocated in `get` with this size and alignment, released once (drained).
            unsafe { dealloc(p, Layout::from_size_align_unchecked(n, SCRATCH_ALIGN)) };
        }
        self.overflow_bytes = m.overflow_bytes;
        self.off = m.off;
        if self.off == 0 && self.overflow.is_empty() {
            self.regrow();
        }
    }

    /// Nothing is in use: make the block as big as the peak demand.
    fn regrow(&mut self) {
        if self.peak > self.cap {
            self.free_block();
            let cap = self.peak.next_power_of_two();
            // SAFETY: cap >= peak > 0 (non-zero layout).
            self.block =
                unsafe { alloc(Layout::from_size_align(cap, SCRATCH_ALIGN).expect("bad layout")) };
            assert!(
                !self.block.is_null(),
                "out of memory allocating {cap} bytes of scratch"
            );
            self.cap = cap;
        }
        self.peak = 0;
    }

    fn free_block(&mut self) {
        if !self.block.is_null() {
            // SAFETY: allocated in `regrow` with this size and alignment; reset to null after.
            unsafe {
                dealloc(
                    self.block,
                    Layout::from_size_align_unchecked(self.cap, SCRATCH_ALIGN),
                )
            };
            self.block = std::ptr::null_mut();
            self.cap = 0;
        }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        for (p, n) in self.overflow.drain(..) {
            // SAFETY: allocated in `get` with this size and alignment.
            unsafe { dealloc(p, Layout::from_size_align_unchecked(n, SCRATCH_ALIGN)) };
        }
        self.free_block();
    }
}

// SAFETY: a context belongs to one session (one thread at a time); `rt` points to a
// static and the scratch arena owns its memory.
unsafe impl Send for PtCtx {}

// The runtime functions, called by kernels (generated, numba, Rust). `ctx` is always the
// context the kernel was called with: valid and exclusively the kernel's for the call.
// These are "C" (not "C-unwind") because generated and numba frames can't be unwound;
// a panic here (out of memory) aborts.

unsafe extern "C" fn rt_alloc(
    ctx: *mut PtCtx,
    bytes: usize,
    align: usize,
    owner: *mut *mut PtBuffer,
) -> *mut u8 {
    if bytes > isize::MAX as usize {
        // SAFETY: `owner` is writable and `ctx` live (the contract).
        unsafe {
            *owner = std::ptr::null_mut();
            (*ctx).error = Some((1, format!("could not allocate {} bytes", bytes as isize)));
        }
        return std::ptr::null_mut();
    }
    let b = PtBuffer::new(bytes, align);
    // SAFETY: `owner` points to the kernel's output slot (a PtArray field); `b` is live.
    unsafe {
        *owner = b;
        (*b).data
    }
}

unsafe extern "C" fn rt_incref(_ctx: *mut PtCtx, b: *mut PtBuffer) {
    // SAFETY: kernels only pass owners of arrays they were given or allocated.
    unsafe { PtBuffer::incref(b) }
}

unsafe extern "C" fn rt_decref(_ctx: *mut PtCtx, b: *mut PtBuffer) {
    // (null: nothing held, e.g. a view; generated code releases slots unconditionally)
    if b.is_null() {
        return;
    }
    // SAFETY: as above, and the kernel gives up a reference it holds.
    unsafe { PtBuffer::decref(b) }
}

unsafe extern "C" fn rt_scratch(ctx: *mut PtCtx, bytes: usize) -> *mut u8 {
    if bytes > isize::MAX as usize {
        // SAFETY: `ctx` is live.
        unsafe {
            (*ctx).error = Some((
                1,
                format!("could not allocate {} bytes of scratch", bytes as isize),
            ))
        };
        return std::ptr::null_mut();
    }
    // SAFETY: see above for `ctx`.
    unsafe { (*ctx).scratch.get(bytes) }
}

unsafe extern "C" fn rt_set_error(ctx: *mut PtCtx, code: i32, msg: *const c_char) {
    // SAFETY: `msg` is null or a NUL-terminated string (the ABI contract); `ctx` as above.
    unsafe {
        let m = if msg.is_null() {
            String::new()
        } else {
            CStr::from_ptr(msg).to_string_lossy().into_owned()
        };
        (*ctx).error = Some((code, m));
    }
}

unsafe extern "C" fn rt_scratch_mark(ctx: *mut PtCtx) -> i64 {
    // SAFETY: see above for `ctx`.
    unsafe { (*ctx).scratch_mark() }
}

unsafe extern "C" fn rt_scratch_release(ctx: *mut PtCtx, token: i64) {
    // SAFETY: see above for `ctx`.
    unsafe { (*ctx).scratch_release(token) }
}

impl PtCtx {
    pub fn new() -> Self {
        PtCtx {
            rt: &RUNTIME.functions,
            error: None,
            scratch: Scratch::new(),
            marks: Vec::new(),
            rng: entropy_rng(),
            parallel: true,
        }
    }

    /// Restart the random state from `seed`.
    pub fn seed(&mut self, seed: u64) {
        use rand_core::SeedableRng;
        self.rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
    }

    /// Mark the scratch position; `scratch_release(token)` frees everything handed out
    /// since (and releases later marks). The interpreter marks around every task;
    /// generated code and kernels that call other code (Scan) mark too.
    pub fn scratch_mark(&mut self) -> i64 {
        self.marks.push(self.scratch.mark());
        (self.marks.len() - 1) as i64
    }

    /// Release back to the mark `token`. Unknown tokens are ignored.
    pub fn scratch_release(&mut self, token: i64) {
        if token >= 0 && (token as usize) < self.marks.len() {
            let m = self.marks[token as usize];
            self.marks.truncate(token as usize);
            self.scratch.release(m);
        }
    }

    // Helpers for Rust kernels. SAFETY (all three): `ctx` is the kernel's context
    // (valid, not used by anyone else during the call).

    /// Scratch for Rust kernels (same as `rt->scratch`).
    pub unsafe fn scratch(ctx: *mut PtCtx, bytes: usize) -> *mut u8 {
        // SAFETY: see above for `ctx`.
        unsafe { (*ctx).scratch.get(bytes) }
    }

    pub unsafe fn fail(ctx: *mut PtCtx, code: i32, msg: &str) -> i32 {
        // SAFETY: see above for `ctx`.
        unsafe { (*ctx).error = Some((code, msg.to_string())) };
        code
    }

    /// A new buffer and one reference to it, for a kernel-allocated output.
    pub unsafe fn alloc(ctx: *mut PtCtx, bytes: usize) -> (*mut u8, *mut PtBuffer) {
        let mut owner = std::ptr::null_mut();
        // SAFETY: see above for `ctx`; `rt` is the static runtime table.
        let p = unsafe { ((*(*ctx).rt).alloc)(ctx, bytes, 64, &mut owner) };
        (p, owner)
    }
}

/// A random state seeded from the process's hash randomness (std's per-instance
/// random keys), for contexts nobody seeded.
fn entropy_rng() -> rand_xoshiro::Xoshiro256PlusPlus {
    use rand_core::SeedableRng;
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_usize(&h as *const _ as usize);
    rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(h.finish())
}

impl Default for PtCtx {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_arena_grows_to_peak_and_keeps_pointers() {
        let mut ctx = PtCtx::new();
        let c: *mut PtCtx = &mut ctx;
        unsafe {
            // first task: nothing reserved, everything overflows but stays valid
            let task = ctx.scratch_mark();
            let a = PtCtx::scratch(c, 100);
            let b = PtCtx::scratch(c, 1000);
            a.write_bytes(1, 100);
            b.write_bytes(2, 1000);
            assert_eq!(*a.add(99), 1);
            assert_eq!(a as usize % 64, 0);
            assert_eq!(b as usize % 64, 0);
            ctx.scratch_release(task);
            // second task with the same demand: served from the persistent block
            let task = ctx.scratch_mark();
            let a2 = PtCtx::scratch(c, 100);
            let b2 = PtCtx::scratch(c, 1000);
            assert!(ctx.scratch.overflow.is_empty());
            assert_eq!(b2 as usize - a2 as usize, 128);
            // nested driver: release to a mark frees only what came after it
            let m = ctx.scratch_mark();
            let _ = PtCtx::scratch(c, 1 << 20); // overflow
            assert_eq!(ctx.scratch.overflow.len(), 1);
            ctx.scratch_release(m);
            assert!(ctx.scratch.overflow.is_empty());
            ctx.scratch_release(task);
            assert!(ctx.scratch.cap >= (1 << 20) + 1152);
        }
    }
}
