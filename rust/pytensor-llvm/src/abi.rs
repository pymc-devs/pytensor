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
//!     void *(*scratch)(PtCtx *, size_t bytes);    // valid until the kernel returns
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
    scratch: Vec<*mut PtBuffer>,
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
    let b = PtBuffer::new(bytes, 64);
    (*ctx).scratch.push(b);
    (*b).data
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
        PtCtx { rt: &RUNTIME, error: None, scratch: Vec::new() }
    }

    /// Free scratch memory handed out during the last kernel call.
    pub fn end_kernel(&mut self) {
        for b in self.scratch.drain(..) {
            unsafe { PtBuffer::decref(b) };
        }
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
