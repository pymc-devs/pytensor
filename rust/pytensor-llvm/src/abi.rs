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
//!     int32_t   ndim;
//!     int32_t   dtype;     // a dtype code (`DType::code`, `PT_FLOAT64`, ...)
//! } PtArray;               // 40 bytes
//!
//! // args = inputs ++ outputs; 0 = ok
//! int32_t kernel(PtCtx *ctx, PtArray *args, int32_t n_inputs, int32_t n_outputs,
//!                const void *config);
//! ```
//!
//! The caller fills `ndim`, `dtype` and the `shape` and `strides` pointers of every
//! argument, outputs included; kernels never write these. They are what a kernel can
//! check its arguments against: generated kernels know their types and ignore them,
//! Rust kernels check them (`crate::kernel`).
//!
//! Each output of a kernel has a mode (`OutputMode`), fixed in the kernel's config:
//! * provided: the caller passes memory (`data` is not NULL), shape and strides; the
//!   kernel writes elements.
//! * allocated: `data` is NULL. The kernel calls `ctx->rt->alloc` and fills data, shape,
//!   strides and owner (one reference, the caller's from then on). Used when the output
//!   shape is only known while running.
//! * view of input i: `data` is NULL. The kernel fills data, shape and strides pointing
//!   into input i's memory, and copies input i's `owner` (no new reference: the caller
//!   keeps the input alive as long as the view).
//!
//! `PtArray` is only the layout. Rust code works with the views of `crate::array`
//! (`PtArrayRef`, `PtArrayMut`, `OutSlot`): making one from a `PtArray` is the unsafe
//! step, where the caller vouches for the memory, and everything after is safe.

use crate::runtime::PtCtx;
use pyo3::prelude::*;
use std::alloc::{Layout, alloc, alloc_zeroed, dealloc};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

pub const ABI_VERSION: u32 = 1;

/// An argument of a kernel, as it is passed. See `crate::array` for using one.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PtArray {
    pub data: *mut u8,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub owner: *mut PtBuffer,
    pub ndim: i32,
    pub dtype: i32,
}

const _: () = assert!(size_of::<PtArray>() == 40);

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
    if shape.contains(&0) {
        return Ok((data as usize, data as usize));
    }
    if data.is_null() {
        return Err("null data pointer for a non-empty array".into());
    }
    let isz = itemsize as i64;
    if !(data as usize).is_multiple_of(itemsize) {
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

/// Strides of a C-contiguous array.
pub fn c_strides(shape: &[i64], itemsize: i64) -> Vec<i64> {
    let mut st = vec![0i64; shape.len()];
    let mut acc = itemsize;
    for d in (0..shape.len()).rev() {
        st[d] = acc;
        acc = acc.saturating_mul(shape[d].max(1));
    }
    st
}

/// numpy's C-contiguity: length-1 axes and empty arrays don't count.
pub fn is_c_contiguous(shape: &[i64], strides: &[i64], itemsize: usize) -> bool {
    if shape.contains(&0) {
        return true;
    }
    let mut acc = itemsize as i64;
    for (&n, &s) in shape.iter().zip(strides).rev() {
        if n != 1 && s != acc {
            return false;
        }
        acc *= n;
    }
    true
}

/// The kernel ABI: `int32 f(ctx, args, n_inputs, n_outputs, config)`.
///
/// Kernels use the "C-unwind" ABI: a panic in a Rust kernel (a bug, or e.g. an
/// out-of-memory) unwinds to `Program::execute`, which turns it into an error, instead
/// of aborting the process. Generated and numba kernels never unwind.
pub type KernelFn = unsafe extern "C-unwind" fn(
    ctx: *mut PtCtx,
    args: *mut PtArray,
    n_inputs: i32,
    n_outputs: i32,
    config: *const u8,
) -> i32;

/// How an output comes to be.
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
///
/// Rust code holds references as `OwnedBuf`s; the raw pointer form is for the ABI
/// (`PtArray::owner`, the C API, numpy capsules).
pub struct PtBuffer {
    data: *mut u8,
    size: usize,
    align: usize,
    refcount: AtomicUsize,
}

impl PtBuffer {
    /// A new buffer with one reference. Panics when out of memory.
    fn new(bytes: usize, align: usize, zeroed: bool) -> NonNull<PtBuffer> {
        let align = align.max(64);
        let size = bytes.max(1);
        let layout = Layout::from_size_align(size, align).expect("bad layout");
        // SAFETY: size >= 1 (non-zero layout, as `alloc` requires).
        let data = unsafe {
            if zeroed {
                alloc_zeroed(layout)
            } else {
                alloc(layout)
            }
        };
        assert!(!data.is_null(), "out of memory allocating {bytes} bytes");
        let b = Box::new(PtBuffer {
            data,
            size,
            align,
            refcount: AtomicUsize::new(1),
        });
        NonNull::from(Box::leak(b))
    }

    /// SAFETY: `b` is a live buffer and the caller holds a reference.
    unsafe fn incref(b: *mut PtBuffer) {
        // Relaxed is enough to create a reference from an existing one (as in Arc).
        // SAFETY: the caller holds a reference, so `b` is live.
        unsafe { (*b).refcount.fetch_add(1, Ordering::Relaxed) };
    }

    /// SAFETY: `b` is a live buffer and the caller gives up one reference it holds.
    unsafe fn decref(b: *mut PtBuffer) {
        // AcqRel: the last owner sees every write made through other references
        // before freeing (as in Arc).
        // SAFETY: the caller holds a reference, so `b` is live.
        if unsafe { (*b).refcount.fetch_sub(1, Ordering::AcqRel) } == 1 {
            // SAFETY: we were the last reference; `b` came from a Box in `new` and
            // `data` from `alloc` with exactly this size and alignment.
            unsafe {
                let b = Box::from_raw(b);
                dealloc(b.data, Layout::from_size_align_unchecked(b.size, b.align));
            }
        }
    }
}

/// One reference to a `PtBuffer`, released when dropped (also while unwinding from a
/// panic); cloning takes another reference.
pub struct OwnedBuf(NonNull<PtBuffer>);

// SAFETY: a reference to a buffer with an atomic refcount, as an `Arc<[u8]>` would be.
// The buffer's memory is only reached through raw pointers, whose users (the views of
// `crate::array`) vouch for how it is shared.
unsafe impl Send for OwnedBuf {}
// SAFETY: as above.
unsafe impl Sync for OwnedBuf {}

impl OwnedBuf {
    /// A new buffer of `bytes` (64-byte aligned, not initialized), or None if that is
    /// more than can be allocated. Panics when out of memory.
    pub fn new(bytes: usize) -> Option<OwnedBuf> {
        Self::aligned(bytes, 64)
    }

    /// As `new`, aligned to `align` (at least 64).
    pub fn aligned(bytes: usize, align: usize) -> Option<OwnedBuf> {
        Self::allocate(bytes, align, false)
    }

    /// As `new`, with every byte zero.
    pub fn zeroed(bytes: usize) -> Option<OwnedBuf> {
        Self::allocate(bytes, 64, true)
    }

    fn allocate(bytes: usize, align: usize, zeroed: bool) -> Option<OwnedBuf> {
        if !align.is_power_of_two() || bytes > isize::MAX as usize - align.max(64) {
            return None;
        }
        Some(OwnedBuf(PtBuffer::new(bytes, align, zeroed)))
    }

    /// Whether this is the buffer's only reference.
    pub fn is_unique(&self) -> bool {
        // SAFETY: we hold a reference, so the buffer is live.
        unsafe { self.0.as_ref().refcount.load(Ordering::Acquire) == 1 }
    }

    /// The buffer's memory: `size()` bytes, valid while a reference is held.
    pub fn data(&self) -> *mut u8 {
        // SAFETY: we hold a reference, so the buffer is live.
        unsafe { self.0.as_ref().data }
    }

    /// Size in bytes (at least 1).
    pub fn size(&self) -> usize {
        // SAFETY: we hold a reference, so the buffer is live.
        unsafe { self.0.as_ref().size }
    }

    /// The byte range `[lo, hi)` of the buffer's memory.
    pub fn range(&self) -> (usize, usize) {
        let lo = self.data() as usize;
        (lo, lo + self.size())
    }

    /// Take over a reference from the ABI.
    ///
    /// # Safety
    /// `b` is a live buffer and the caller gives one reference it holds to us.
    pub unsafe fn from_raw(b: *mut PtBuffer) -> OwnedBuf {
        OwnedBuf(NonNull::new(b).expect("null buffer"))
    }

    /// A new reference to a buffer of the ABI.
    ///
    /// # Safety
    /// `b` is a live buffer, kept alive by someone else's reference during the call.
    pub unsafe fn from_borrowed(b: *mut PtBuffer) -> OwnedBuf {
        // SAFETY: the function's contract.
        unsafe { PtBuffer::incref(b) };
        OwnedBuf(NonNull::new(b).expect("null buffer"))
    }

    pub fn as_ptr(&self) -> *mut PtBuffer {
        self.0.as_ptr()
    }

    /// Hand the reference on to the ABI (a kernel output's `owner`, a numpy capsule).
    pub fn into_raw(self) -> *mut PtBuffer {
        let p = self.0.as_ptr();
        std::mem::forget(self);
        p
    }
}

impl Clone for OwnedBuf {
    fn clone(&self) -> Self {
        // SAFETY: we hold a reference, so the buffer is live.
        unsafe { PtBuffer::incref(self.0.as_ptr()) };
        OwnedBuf(self.0)
    }
}

impl Drop for OwnedBuf {
    fn drop(&mut self) {
        // SAFETY: we own one reference (from `new`, `clone` or `from_raw`'s
        // contract), released exactly once.
        unsafe { PtBuffer::decref(self.0.as_ptr()) }
    }
}

pub const MAX_NDIM: usize = 32;

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C-unwind" fn panicking_kernel(
        _ctx: *mut PtCtx,
        _args: *mut PtArray,
        _n_inputs: i32,
        _n_outputs: i32,
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
        // SAFETY: the kernel uses none of its arguments.
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            f(&mut ctx, std::ptr::null_mut(), 0, 0, std::ptr::null())
        }));
        let msg = r.unwrap_err();
        assert_eq!(*msg.downcast_ref::<&str>().unwrap(), "kernel bug");
    }

    #[test]
    fn buffers_are_counted() {
        let a = OwnedBuf::new(100).unwrap();
        let b = a.clone();
        assert_eq!(a.data(), b.data());
        assert!(a.size() >= 100);
        let raw = a.into_raw();
        // SAFETY: the reference `into_raw` gave us.
        drop(unsafe { OwnedBuf::from_raw(raw) });
        // `b` still holds the buffer
        // SAFETY: inside the buffer `b` keeps alive.
        unsafe { b.data().write(1) };
        assert!(OwnedBuf::new(usize::MAX).is_none());
    }
}
