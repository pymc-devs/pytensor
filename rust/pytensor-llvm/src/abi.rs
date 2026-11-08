//! The kernel ABI, shared by LLVM-generated kernels and precompiled Rust kernels.
//!
//! ```c
//! typedef struct PtBuffer PtBuffer;               // opaque, refcounted
//! typedef struct PtCtx PtCtx;                      // starts with `const PtRuntime *rt`
//!
//! typedef struct {
//!     uint32_t version;
//!     void *(*alloc)(PtCtx *, size_t bytes, size_t align, PtBuffer **owner);
//!     void  (*buffer_incref)(PtCtx *, PtBuffer *);
//!     void  (*buffer_decref)(PtCtx *, PtBuffer *);
//!     void *(*scratch)(PtCtx *, size_t bytes);    // 64-byte aligned, valid until the kernel returns
//!     void  (*set_error)(PtCtx *, int32_t code, const char *msg);
//! } PtRuntime;
//!
//! typedef struct {
//!     char     *data;
//!     int64_t  *shape;     // `ndim` entries; written by the kernel for kernel-allocated
//!     int64_t  *strides;   //   outputs and declared views. Strides are in bytes.
//!     PtBuffer *owner;     // owner of `data`; NULL = borrowed (inputs, planner slots)
//!     int64_t   ndim;
//! } PtArray;               // 40 bytes
//!
//! // args = inputs ++ outputs. params points to a kernel-specific #[repr(C)] struct.
//! int32_t kernel(PtCtx *ctx, PtArray *args, const void *params);  // 0 = ok
//! ```
//!
//! Each output of a kernel has a mode that is fixed in the kernel's spec:
//! * provided: the caller passes memory, shape and strides; the kernel writes elements.
//! * kernel-allocated: the kernel calls `ctx->rt->alloc` and fills data, shape, strides
//!   and owner. Used when the output shape is only known while running.
//! * declared view of input i: the kernel fills data, shape and strides pointing into
//!   input i's memory. The caller sets `owner`; the kernel must not.
//!
//! Scratch: `rt->scratch(ctx, n)` hands out temporary memory (e.g. faer's work
//! space) that is valid until the kernel returns. It comes from an arena owned by the
//! context (one per call state): the arena keeps its largest size, so after the first
//! call a kernel's scratch requests don't allocate. Several requests per kernel are
//! fine; they never move earlier ones.

use std::alloc::{alloc, dealloc, Layout};
use std::ffi::{c_char, c_void, CStr};
use std::sync::atomic::{AtomicUsize, Ordering};

pub const ABI_VERSION: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PtArray {
    pub data: *mut u8,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub owner: *mut PtBuffer,
    pub ndim: i64,
}

pub const PTARRAY_SIZE: usize = std::mem::size_of::<PtArray>();
pub const FIELD_DATA: usize = 0;
pub const FIELD_SHAPE: usize = 1;
pub const FIELD_STRIDES: usize = 2;

/// Shape and strides as slices.
///
/// SAFETY (all four): `shape`/`strides` point to `ndim` readable (writable for the
/// `_mut` versions) i64s, and nothing else writes them while the slice is alive. The
/// runtime gives every variable its own shape/stride storage (`CallState::shapes`,
/// at `dim_off[v]`), so an input's and an output's arrays never alias; at the C API
/// boundary `ndim` is checked against the function's types before these are called.
impl PtArray {
    pub unsafe fn shape(&self) -> &[i64] {
        // SAFETY: see the impl docs above.
        unsafe { slice(self.shape, self.ndim as usize) }
    }
    pub unsafe fn strides(&self) -> &[i64] {
        // SAFETY: see the impl docs above.
        unsafe { slice(self.strides, self.ndim as usize) }
    }
    pub unsafe fn shape_mut(&mut self) -> &mut [i64] {
        // SAFETY: see the impl docs above.
        if self.ndim == 0 { &mut [] } else { unsafe { std::slice::from_raw_parts_mut(self.shape, self.ndim as usize) } }
    }
    pub unsafe fn strides_mut(&mut self) -> &mut [i64] {
        // SAFETY: see the impl docs above.
        if self.ndim == 0 { &mut [] } else { unsafe { std::slice::from_raw_parts_mut(self.strides, self.ndim as usize) } }
    }
}

/// Byte size of an array of `shape`, with numpy's limits: no negative dims, and the
/// size must fit in `isize`. Every allocation sized from a shape goes through this, so
/// a shape whose element count overflows can't get a too-small buffer.
pub fn checked_nbytes(shape: &[i64], itemsize: usize) -> Option<usize> {
    let mut n = itemsize as i64;
    for &s in shape {
        if s < 0 {
            return None;
        }
        n = n.checked_mul(s)?;
    }
    usize::try_from(n).ok()
}

/// Checks for an array handed to us from outside (an input, or caller memory for an
/// output), returning the byte range [lo, hi) it covers.
///
/// Kernels index with shapes and strides without bounds checks, so this is what makes
/// "the array is valid" an explicit precondition we've verified rather than assumed:
/// no negative dims, a non-null pointer for non-empty arrays, data and strides aligned
/// to the itemsize, and an address range that doesn't overflow.
pub fn validate_array(data: *const u8, shape: &[i64], strides: &[i64], itemsize: usize) -> Result<(usize, usize), String> {
    if let Some(d) = shape.iter().position(|&s| s < 0) {
        return Err(format!("negative length {} in dim {d}", shape[d]));
    }
    checked_nbytes(shape, itemsize).ok_or_else(|| format!("shape {shape:?} is too big"))?;
    if shape.iter().any(|&s| s == 0) {
        return Ok((data as usize, data as usize));
    }
    if data.is_null() {
        return Err("null data pointer for a non-empty array".into());
    }
    let isz = itemsize as i64;
    if (data as usize) % itemsize != 0 {
        return Err(format!("data pointer {data:p} is not aligned to {itemsize} bytes"));
    }
    let (mut lo, mut hi) = (0i64, 0i64);
    for (d, (&n, &s)) in shape.iter().zip(strides).enumerate() {
        if n > 1 && s % isz != 0 {
            return Err(format!("stride {s} of dim {d} is not a multiple of the itemsize {itemsize}"));
        }
        let span = (n - 1).checked_mul(s).ok_or("array extent overflows")?;
        if span < 0 { lo = lo.checked_add(span).ok_or("array extent overflows")? } else { hi = hi.checked_add(span).ok_or("array extent overflows")? }
    }
    let base = data as i64;
    let start = base.checked_add(lo).filter(|&x| x >= 0).ok_or("array extent overflows")?;
    let end = base.checked_add(hi).and_then(|x| x.checked_add(isz)).ok_or("array extent overflows")?;
    Ok((start as usize, end as usize))
}

/// Whether two byte ranges overlap (empty ranges never do).
pub fn overlaps(a: (usize, usize), b: (usize, usize)) -> bool {
    a.0 < a.1 && b.0 < b.1 && a.0 < b.1 && b.0 < a.1
}

/// Error message for a shape `checked_nbytes` rejects.
pub fn bad_size(what: &str, shape: &[i64]) -> String {
    if shape.iter().any(|&s| s < 0) {
        format!("{what}: negative dimension in shape {shape:?}")
    } else {
        format!("{what}: array of shape {shape:?} is too big")
    }
}

/// SAFETY: for n > 0, `p` points to n readable i64s that outlive 'a.
pub unsafe fn slice<'a>(p: *const i64, n: usize) -> &'a [i64] {
    // SAFETY: the function's contract.
    if n == 0 { &[] } else { unsafe { std::slice::from_raw_parts(p, n) } }
}

/// Kernels use the "C-unwind" ABI: a panic in a Rust kernel (a bug, or e.g. an
/// out-of-memory) unwinds to `Program::execute`, which turns it into an error, instead
/// of aborting the process. Generated and numba kernels never unwind.
pub type KernelFn = unsafe extern "C-unwind" fn(ctx: *mut PtCtx, args: *mut PtArray, params: *const c_void) -> i32;

// ------------------------------------------------------------------ buffers

/// Refcounted heap buffer. Plain malloc/free: no pooling of dead buffers.
pub struct PtBuffer {
    pub data: *mut u8,
    size: usize,
    align: usize,
    refcount: AtomicUsize,
}

impl PtBuffer {
    pub fn new(bytes: usize, align: usize) -> *mut PtBuffer {
        let align = align.max(64);
        let size = bytes.max(1);
        // SAFETY: size >= 1 (non-zero layout, as `alloc` requires).
        let data = unsafe { alloc(Layout::from_size_align(size, align).expect("bad layout")) };
        assert!(!data.is_null(), "out of memory allocating {bytes} bytes");
        Box::into_raw(Box::new(PtBuffer { data, size, align, refcount: AtomicUsize::new(1) }))
    }

    /// Size in bytes (for the checked mode's bounds checks).
    pub fn size(&self) -> usize {
        self.size
    }

    /// SAFETY: `b` is a live buffer (from `new`) and the caller holds a reference.
    pub unsafe fn incref(b: *mut PtBuffer) {
        // Relaxed is enough to create a reference from an existing one (as in Arc).
        // SAFETY: the caller holds a reference, so `b` is live.
        unsafe { (*b).refcount.fetch_add(1, Ordering::Relaxed) };
    }

    /// SAFETY: `b` is a live buffer and the caller gives up one reference it holds.
    pub unsafe fn decref(b: *mut PtBuffer) {
        // AcqRel: the last owner sees every write made through other references
        // before freeing (as in Arc).
        // SAFETY: the caller holds a reference, so `b` is live.
        if unsafe { (*b).refcount.fetch_sub(1, Ordering::AcqRel) } == 1 {
            // SAFETY: we were the last reference; `b` came from Box::into_raw in `new`
            // and `data` from `alloc` with exactly this size and alignment.
            unsafe {
                let b = Box::from_raw(b);
                dealloc(b.data, Layout::from_size_align_unchecked(b.size, b.align));
            }
        }
    }
}

/// One owned reference to a `PtBuffer`, released when dropped (also while unwinding
/// from a panic). For Rust code that holds buffers across fallible work; the raw
/// pointer form is for the ABI (kernel outputs, the C API, numpy capsules).
pub struct OwnedBuf(std::ptr::NonNull<PtBuffer>);

impl OwnedBuf {
    /// SAFETY: `b` must be a live buffer and the caller gives one reference to us.
    pub unsafe fn from_raw(b: *mut PtBuffer) -> OwnedBuf {
        OwnedBuf(std::ptr::NonNull::new(b).expect("null buffer"))
    }

    pub fn as_ptr(&self) -> *mut PtBuffer {
        self.0.as_ptr()
    }

    /// Hand the reference on (e.g. to a kernel output's `owner`).
    pub fn into_raw(self) -> *mut PtBuffer {
        let p = self.0.as_ptr();
        std::mem::forget(self);
        p
    }
}

impl Drop for OwnedBuf {
    fn drop(&mut self) {
        // SAFETY: we own one reference (from_raw's contract), released exactly once.
        unsafe { PtBuffer::decref(self.0.as_ptr()) }
    }
}

// ------------------------------------------------------------------ context

#[repr(C)]
pub struct PtRuntime {
    pub version: u32,
    pub alloc: unsafe extern "C" fn(*mut PtCtx, usize, usize, *mut *mut PtBuffer) -> *mut u8,
    pub buffer_incref: unsafe extern "C" fn(*mut PtCtx, *mut PtBuffer),
    pub buffer_decref: unsafe extern "C" fn(*mut PtCtx, *mut PtBuffer),
    pub scratch: unsafe extern "C" fn(*mut PtCtx, usize) -> *mut u8,
    pub set_error: unsafe extern "C" fn(*mut PtCtx, i32, *const c_char),
}

#[repr(C)]
pub struct PtCtx {
    pub rt: *const PtRuntime,
    // runtime-private state below; kernels only see `rt`
    pub error: Option<(i32, String)>,
    scratch: Scratch,
}

/// Bump arena behind `rt->scratch`.
///
/// One persistent block, used from offset 0 by each kernel call. A request that
/// doesn't fit gets its own overflow block (so earlier pointers stay valid); when the
/// kernel ends, the persistent block is regrown to the peak demand, so the next call
/// fits without allocating.
struct Scratch {
    block: *mut u8,
    cap: usize,
    off: usize,
    overflow: Vec<(*mut u8, usize)>,
    overflow_bytes: usize,
    peak: usize,
}

/// Position in the scratch arena, for drivers that call kernels repeatedly (Scan).
#[derive(Clone, Copy)]
pub struct ScratchMark {
    off: usize,
    n_overflow: usize,
    overflow_bytes: usize,
}

const SCRATCH_ALIGN: usize = 64;

impl Scratch {
    fn new() -> Self {
        Scratch { block: std::ptr::null_mut(), cap: 0, off: 0, overflow: Vec::new(), overflow_bytes: 0, peak: 0 }
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
            let p = unsafe { alloc(Layout::from_size_align(bytes, SCRATCH_ALIGN).expect("bad layout")) };
            assert!(!p.is_null(), "out of memory allocating {bytes} bytes of scratch");
            self.overflow.push((p, bytes));
            self.overflow_bytes += bytes;
            p
        };
        self.peak = self.peak.max(self.off + self.overflow_bytes);
        p
    }

    fn mark(&self) -> ScratchMark {
        ScratchMark { off: self.off, n_overflow: self.overflow.len(), overflow_bytes: self.overflow_bytes }
    }

    fn release(&mut self, m: ScratchMark) {
        for (p, n) in self.overflow.drain(m.n_overflow..) {
            // SAFETY: allocated in `get` with this size and alignment, released once (drained).
            unsafe { dealloc(p, Layout::from_size_align_unchecked(n, SCRATCH_ALIGN)) };
        }
        self.overflow_bytes = m.overflow_bytes;
        self.off = m.off;
    }

    fn reset(&mut self) {
        self.release(ScratchMark { off: 0, n_overflow: 0, overflow_bytes: 0 });
        if self.peak > self.cap {
            self.free_block();
            let cap = self.peak.next_power_of_two();
            // SAFETY: cap >= peak > 0 (non-zero layout).
            self.block = unsafe { alloc(Layout::from_size_align(cap, SCRATCH_ALIGN).expect("bad layout")) };
            assert!(!self.block.is_null(), "out of memory allocating {cap} bytes of scratch");
            self.cap = cap;
        }
        self.peak = 0;
    }

    fn free_block(&mut self) {
        if !self.block.is_null() {
            // SAFETY: allocated in `reset` with this size and alignment; reset to null after.
            unsafe { dealloc(self.block, Layout::from_size_align_unchecked(self.cap, SCRATCH_ALIGN)) };
            self.block = std::ptr::null_mut();
            self.cap = 0;
        }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        self.release(ScratchMark { off: 0, n_overflow: 0, overflow_bytes: 0 });
        self.free_block();
    }
}

// SAFETY: a context belongs to one CallState (one thread at a time); `rt` points to a
// static and the scratch arena owns its memory.
unsafe impl Send for PtCtx {}

// The runtime table, called by kernels (generated, numba, Rust). `ctx` is always the
// context the kernel was called with: valid and exclusively the kernel's for the call.
// These are "C" (not "C-unwind") because generated and numba frames can't be unwound;
// a panic here (out of memory) aborts.

unsafe extern "C" fn rt_alloc(_ctx: *mut PtCtx, bytes: usize, align: usize, owner: *mut *mut PtBuffer) -> *mut u8 {
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
    // SAFETY: as above, and the kernel gives up a reference it holds.
    unsafe { PtBuffer::decref(b) }
}

unsafe extern "C" fn rt_scratch(ctx: *mut PtCtx, bytes: usize) -> *mut u8 {
    // SAFETY: see above for `ctx`.
    unsafe { (*ctx).scratch.get(bytes) }
}

unsafe extern "C" fn rt_set_error(ctx: *mut PtCtx, code: i32, msg: *const c_char) {
    // SAFETY: `msg` is null or a NUL-terminated string (the ABI contract); `ctx` as above.
    unsafe {
        let m = if msg.is_null() { String::new() } else { CStr::from_ptr(msg).to_string_lossy().into_owned() };
        (*ctx).error = Some((code, m));
    }
}

pub static RUNTIME: PtRuntime = PtRuntime {
    version: ABI_VERSION,
    alloc: rt_alloc,
    buffer_incref: rt_incref,
    buffer_decref: rt_decref,
    scratch: rt_scratch,
    set_error: rt_set_error,
};

impl PtCtx {
    pub fn new() -> Self {
        PtCtx { rt: &RUNTIME, error: None, scratch: Scratch::new() }
    }

    /// Scratch handed out during the last kernel call is dead now; keep the memory.
    pub fn end_kernel(&mut self) {
        self.scratch.reset();
    }

    /// For kernels that call other kernels with the same context (Scan): everything
    /// handed out after `scratch_mark` is released by `scratch_release`.
    pub fn scratch_mark(&self) -> ScratchMark {
        self.scratch.mark()
    }

    pub fn scratch_release(&mut self, m: ScratchMark) {
        self.scratch.release(m)
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

impl Default for PtCtx {
    fn default() -> Self {
        Self::new()
    }
}

// ------------------------------------------------------------------ params

/// Typed kernel parameters, stored with 8-byte alignment so kernels can read them as
/// `#[repr(C)]` structs.
#[derive(Clone, Debug, Default)]
pub struct Params(Vec<u64>);

impl Params {
    pub fn none() -> Self {
        Params(vec![0])
    }
    pub fn of<T: Copy>(v: T) -> Self {
        let n = std::mem::size_of::<T>().div_ceil(8).max(1);
        let mut buf = vec![0u64; n];
        // SAFETY: `buf` has room for T (checked above) and u64 alignment, enough for the
        // Copy types used as params.
        unsafe { std::ptr::write(buf.as_mut_ptr() as *mut T, v) };
        Params(buf)
    }
    pub fn as_ptr(&self) -> *const c_void {
        self.0.as_ptr() as *const c_void
    }
}

pub const MAX_NDIM: usize = 32;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_arena_grows_to_peak_and_keeps_pointers() {
        let mut ctx = PtCtx::new();
        let c: *mut PtCtx = &mut ctx;
        unsafe {
            // first call: nothing reserved, everything overflows but stays valid
            let a = PtCtx::scratch(c, 100);
            let b = PtCtx::scratch(c, 1000);
            a.write_bytes(1, 100);
            b.write_bytes(2, 1000);
            assert_eq!(*a.add(99), 1);
            assert_eq!(a as usize % 64, 0);
            assert_eq!(b as usize % 64, 0);
            ctx.end_kernel();
            // second call with the same demand: served from the persistent block
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
            ctx.end_kernel();
            assert!(ctx.scratch.cap >= (1 << 20) + 1152);
        }
    }
}

#[cfg(test)]
mod unwind_tests {
    use super::*;

    unsafe extern "C-unwind" fn panicking_kernel(_ctx: *mut PtCtx, _args: *mut PtArray, _params: *const c_void) -> i32 {
        panic!("kernel bug")
    }

    #[test]
    fn kernel_panic_unwinds_to_the_caller() {
        // what Program::execute relies on: a panic in a KernelFn called through a
        // function pointer is catchable, not an abort
        let f: KernelFn = panicking_kernel;
        let mut ctx = PtCtx::new();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { f(&mut ctx, std::ptr::null_mut(), std::ptr::null()) }));
        let msg = r.unwrap_err();
        assert_eq!(*msg.downcast_ref::<&str>().unwrap(), "kernel bug");
    }
}
