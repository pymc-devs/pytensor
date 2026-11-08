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

impl PtArray {
    pub unsafe fn shape(&self) -> &[i64] {
        slice(self.shape, self.ndim as usize)
    }
    pub unsafe fn strides(&self) -> &[i64] {
        slice(self.strides, self.ndim as usize)
    }
    pub unsafe fn shape_mut(&mut self) -> &mut [i64] {
        if self.ndim == 0 { &mut [] } else { std::slice::from_raw_parts_mut(self.shape, self.ndim as usize) }
    }
    pub unsafe fn strides_mut(&mut self) -> &mut [i64] {
        if self.ndim == 0 { &mut [] } else { std::slice::from_raw_parts_mut(self.strides, self.ndim as usize) }
    }
}

pub unsafe fn slice<'a>(p: *const i64, n: usize) -> &'a [i64] {
    if n == 0 { &[] } else { std::slice::from_raw_parts(p, n) }
}

pub type KernelFn = unsafe extern "C" fn(ctx: *mut PtCtx, args: *mut PtArray, params: *const c_void) -> i32;

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
        let data = unsafe { alloc(Layout::from_size_align(size, align).expect("bad layout")) };
        assert!(!data.is_null(), "out of memory allocating {bytes} bytes");
        Box::into_raw(Box::new(PtBuffer { data, size, align, refcount: AtomicUsize::new(1) }))
    }

    pub unsafe fn incref(b: *mut PtBuffer) {
        (*b).refcount.fetch_add(1, Ordering::Relaxed);
    }

    pub unsafe fn decref(b: *mut PtBuffer) {
        if (*b).refcount.fetch_sub(1, Ordering::AcqRel) == 1 {
            let b = Box::from_raw(b);
            dealloc(b.data, Layout::from_size_align_unchecked(b.size, b.align));
        }
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
            let p = unsafe { self.block.add(self.off) };
            self.off += bytes;
            p
        } else {
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
            self.block = unsafe { alloc(Layout::from_size_align(cap, SCRATCH_ALIGN).expect("bad layout")) };
            assert!(!self.block.is_null(), "out of memory allocating {cap} bytes of scratch");
            self.cap = cap;
        }
        self.peak = 0;
    }

    fn free_block(&mut self) {
        if !self.block.is_null() {
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

unsafe impl Send for PtCtx {}

unsafe extern "C" fn rt_alloc(_ctx: *mut PtCtx, bytes: usize, align: usize, owner: *mut *mut PtBuffer) -> *mut u8 {
    let b = PtBuffer::new(bytes, align);
    *owner = b;
    (*b).data
}

unsafe extern "C" fn rt_incref(_ctx: *mut PtCtx, b: *mut PtBuffer) {
    PtBuffer::incref(b)
}

unsafe extern "C" fn rt_decref(_ctx: *mut PtCtx, b: *mut PtBuffer) {
    PtBuffer::decref(b)
}

unsafe extern "C" fn rt_scratch(ctx: *mut PtCtx, bytes: usize) -> *mut u8 {
    (*ctx).scratch.get(bytes)
}

unsafe extern "C" fn rt_set_error(ctx: *mut PtCtx, code: i32, msg: *const c_char) {
    let m = if msg.is_null() { String::new() } else { CStr::from_ptr(msg).to_string_lossy().into_owned() };
    (*ctx).error = Some((code, m));
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

    /// Scratch for Rust kernels (same as `rt->scratch`).
    pub unsafe fn scratch(ctx: *mut PtCtx, bytes: usize) -> *mut u8 {
        (*ctx).scratch.get(bytes)
    }

    /// Convenience for Rust kernels.
    pub unsafe fn fail(ctx: *mut PtCtx, code: i32, msg: &str) -> i32 {
        (*ctx).error = Some((code, msg.to_string()));
        code
    }

    pub unsafe fn alloc(ctx: *mut PtCtx, bytes: usize) -> (*mut u8, *mut PtBuffer) {
        let mut owner = std::ptr::null_mut();
        let p = ((*(*ctx).rt).alloc)(ctx, bytes, 64, &mut owner);
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
