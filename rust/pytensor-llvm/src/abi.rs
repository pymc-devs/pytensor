//! The kernel ABI: the arrays and buffers every kernel exchanges with its caller.
//!
//! ```c
//! typedef struct PtBuffer PtBuffer;               // opaque, refcounted
//! typedef struct PtCtx PtCtx;                      // the context, see `crate::runtime`
//!
//! typedef struct {
//!     char     *data;
//!     int64_t  *shape;     // `ndim` entries; written by the kernel for allocated
//!     int64_t  *strides;   //   outputs and views. Strides are in bytes.
//!     PtBuffer *owner;     // owner of `data`; NULL = borrowed (inputs, arena memory)
//!     int64_t   ndim;
//! } PtArray;               // 40 bytes
//!
//! // args = inputs ++ outputs; 0 = ok
//! int32_t kernel(PtCtx *ctx, PtArray *args, const void *config);
//! ```
//!
//! Each output of a kernel has a mode (`OutputMode`), fixed in the kernel's config:
//! * provided: the caller passes memory, shape and strides; the kernel writes elements.
//! * allocated: the kernel calls `ctx->rt->alloc` and fills data, shape, strides and
//!   owner. Used when the output shape is only known while running.
//! * view of input i: the kernel fills data, shape and strides pointing into input i's
//!   memory. The caller sets `owner`; the kernel must not.
//!
//! Rust kernels are written with `crate::kernel`.

use crate::runtime::PtCtx;
use pyo3::prelude::*;
use std::alloc::{Layout, alloc, dealloc};
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

/// Shape and strides as slices.
///
/// SAFETY (all four): `shape`/`strides` point to `ndim` readable (writable for the
/// `_mut` versions) i64s, and nothing else writes them while the slice is alive. The
/// runtime gives every variable its own shape/stride storage (`CallMemory::shapes`,
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
        if self.ndim == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.shape, self.ndim as usize) }
        }
    }
    pub unsafe fn strides_mut(&mut self) -> &mut [i64] {
        // SAFETY: see the impl docs above.
        if self.ndim == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.strides, self.ndim as usize) }
        }
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
pub fn validate_array(
    data: *const u8,
    shape: &[i64],
    strides: &[i64],
    itemsize: usize,
) -> Result<(usize, usize), String> {
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
        return Err(format!(
            "data pointer {data:p} is not aligned to {itemsize} bytes"
        ));
    }
    let (mut lo, mut hi) = (0i64, 0i64);
    for (d, (&n, &s)) in shape.iter().zip(strides).enumerate() {
        if n > 1 && s % isz != 0 {
            return Err(format!(
                "stride {s} of dim {d} is not a multiple of the itemsize {itemsize}"
            ));
        }
        let span = (n - 1).checked_mul(s).ok_or("array extent overflows")?;
        if span < 0 {
            lo = lo.checked_add(span).ok_or("array extent overflows")?
        } else {
            hi = hi.checked_add(span).ok_or("array extent overflows")?
        }
    }
    let base = data as i64;
    let start = base
        .checked_add(lo)
        .filter(|&x| x >= 0)
        .ok_or("array extent overflows")?;
    let end = base
        .checked_add(hi)
        .and_then(|x| x.checked_add(isz))
        .ok_or("array extent overflows")?;
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
    if n == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(p, n) }
    }
}

/// The kernel ABI: `int32 f(ctx, args, config)`.
///
/// Kernels use the "C-unwind" ABI: a panic in a Rust kernel (a bug, or e.g. an
/// out-of-memory) unwinds to `Program::execute`, which turns it into an error, instead
/// of aborting the process. Generated and numba kernels never unwind.
pub type KernelFn =
    unsafe extern "C-unwind" fn(ctx: *mut PtCtx, args: *mut PtArray, config: *const u8) -> i32;
/// How an output comes to be (DESIGN.md, section 3).
#[pyclass(module = "pytensor_llvm.interp", eq, eq_int, frozen, from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputMode {
    /// the caller provides the memory; the kernel writes the elements
    Provided,
    /// the kernel allocates and hands over one reference
    Allocated,
    /// the kernel returns a view of an input
    View,
}

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
        Box::into_raw(Box::new(PtBuffer {
            data,
            size,
            align,
            refcount: AtomicUsize::new(1),
        }))
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

pub const MAX_NDIM: usize = 32;

#[cfg(test)]
mod unwind_tests {
    use super::*;

    unsafe extern "C-unwind" fn panicking_kernel(
        _ctx: *mut PtCtx,
        _args: *mut PtArray,
        _config: *const u8,
    ) -> i32 {
        panic!("kernel bug")
    }

    #[test]
    fn kernel_panic_unwinds_to_the_caller() {
        // what Program::execute relies on: a panic in a KernelFn called through a
        // function pointer is catchable, not an abort
        let f: KernelFn = panicking_kernel;
        let mut ctx = PtCtx::new();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            f(&mut ctx, std::ptr::null_mut(), std::ptr::null())
        }));
        let msg = r.unwrap_err();
        assert_eq!(*msg.downcast_ref::<&str>().unwrap(), "kernel bug");
    }
}
